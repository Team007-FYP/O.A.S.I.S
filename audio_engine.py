# =============================================================================
# audio_engine.py
# Part 1 — Multimodal Summarisation Pipeline: Audio Transcription & Diarization.
#
# Pipeline stages handled in this file:
#   A. Whisper large-v3 transcription + word-level forced alignment
#   B. Pyannote-backed speaker diarization
#   C. Speaker-attributed transcript formatting and persistence
#
# VRAM discipline:
#   Each heavy model is explicitly deleted and the CUDA cache is flushed
#   immediately after its stage completes.  This ensures the 40 GB A100 VRAM
#   budget is never exceeded by having two large models resident simultaneously.
# =============================================================================

import os          # Environment variable access, path utilities
import gc          # CPython garbage collector — forces immediate reference cleanup
import torch       # CUDA cache management via torch.cuda.empty_cache()
import whisperx    # WhisperX: transcription, alignment, diarization wrapper

from dotenv import load_dotenv   # Load secrets from a .env file into os.environ

# Load environment variables from .env at module import time.
# After this call, os.getenv("HF_TOKEN") will return the Hugging Face token
# if it is defined in the project's .env file.
load_dotenv()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_audio(audio_path: str, transcript_path: str) -> str:
    """
    Transcribe an audio file, diarize speakers, assign words to speakers,
    and persist the formatted speaker-attributed transcript to disk.

    Parameters
    ----------
    audio_path : str
        Absolute or relative path to the input audio file (e.g. "dataset/A1.wav").
    transcript_path : str
        Destination path for the plain-text transcript (e.g. "outputs/transcripts/T1.txt").

    Returns
    -------
    str
        The path of the saved transcript file (same as *transcript_path*).

    VRAM Budget
    -----------
    Stage A  loads  Whisper large-v3 (~10 GB) + alignment model (~0.5 GB).
             Both are deleted before Stage B begins.
    Stage B  loads  Pyannote diarization pipeline (~2 GB).
             Deleted before Stage C begins.
    Peak concurrent VRAM ≈ 10.5 GB — well within the 40 GB A100 budget and
    leaves headroom for the LLM stages that follow in the pipeline.
    """

    # Retrieve the Hugging Face Hub token required by the Pyannote diarization
    # pipeline (which uses gated model weights on HuggingFace).
    hf_token: str | None = os.getenv("HF_TOKEN")

    # Ensure the output directory exists before we try to write to it.
    os.makedirs(os.path.dirname(transcript_path) or ".", exist_ok=True)

    # =========================================================================
    # STAGE A — TRANSCRIPTION + FORCED ALIGNMENT
    # Load Whisper large-v3 on CUDA with float16 precision, transcribe the
    # audio, then load the language-specific alignment model and align each
    # word to its precise timestamp in the audio stream.
    # =========================================================================

    print("[audio_engine] Stage A — Loading Whisper large-v3 …")

    # Load WhisperX's Whisper wrapper.
    #   model    : "large-v3" — best accuracy; fits comfortably on A100 in float16.
    #   device   : "cuda" — mandatory; CPU inference would be ~50× slower.
    #   compute_type: "float16" — halves VRAM vs float32 with negligible accuracy loss
    #                             on Ampere-class (A100) hardware.
    whisper_model = whisperx.load_model(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )

    print("[audio_engine] Stage A — Transcribing audio …")

    # Transcribe.  Returns a dict with keys "segments" and "language".
    # batch_size=16 maximises GPU utilisation without OOM on A100.
    transcription_result: dict = whisper_model.transcribe(
        audio_path,
        batch_size=16,
    )

    # Detected language code (e.g. "en") — required by the alignment model loader.
    detected_language: str = transcription_result["language"]

    print(f"[audio_engine] Stage A — Detected language: '{detected_language}'. Loading alignment model …")

    # Load the wav2vec2-based alignment model for the detected language.
    # Returns the model object and its metadata dict (used during alignment).
    alignment_model, alignment_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device="cuda",
    )

    print("[audio_engine] Stage A — Aligning transcript …")

    # Align words to precise audio timestamps using the phoneme-level CTC model.
    aligned_result: dict = whisperx.align(
        transcription_result["segments"],
        alignment_model,
        alignment_metadata,
        audio_path,
        device="cuda",
        return_char_alignments=False,   # Word-level only — reduces output size
    )

    # ------------------------------------------------------------------
    # CRITICAL VRAM CLEANUP 1
    # Release Whisper large-v3 and the alignment model before loading
    # the diarization pipeline.  Without this, both models + the
    # Pyannote pipeline would coexist in VRAM simultaneously.
    # ------------------------------------------------------------------
    print("[audio_engine] Stage A complete — Flushing Whisper + alignment models from VRAM …")

    del whisper_model        # Drop reference so CPython can free the object
    del alignment_model      # Drop reference to the wav2vec2 model
    del alignment_metadata   # Drop the metadata dict (small, but good practice)

    gc.collect()                    # Force CPython to process reference-counted frees
    torch.cuda.empty_cache()        # Return all freed CUDA memory blocks to the allocator

    print("[audio_engine] VRAM cleanup 1 complete.")

    # =========================================================================
    # STAGE B — SPEAKER DIARIZATION
    # Load the Pyannote-backed diarization pipeline and segment the audio
    # into speaker-labelled time intervals.
    # =========================================================================

    print("[audio_engine] Stage B — Loading diarization pipeline …")

    # DiarizationPipeline wraps pyannote.audio's speaker-diarization/3.1 model.
    # token=hf_token grants access to the gated pyannote weights on HuggingFace.
    diarization_pipeline = whisperx.diarize.DiarizationPipeline(
        token=hf_token,
        device="cuda",
    )

    print("[audio_engine] Stage B — Running diarization …")

    # Returns a pyannote Annotation object mapping (start, end) -> speaker label.
    diarization_result = diarization_pipeline(audio_path)

    # ------------------------------------------------------------------
    # CRITICAL VRAM CLEANUP 2
    # Release the diarization pipeline before Stage C (which is CPU-only).
    # ------------------------------------------------------------------
    print("[audio_engine] Stage B complete — Flushing diarization pipeline from VRAM …")

    del diarization_pipeline

    gc.collect()
    torch.cuda.empty_cache()

    print("[audio_engine] VRAM cleanup 2 complete.")

    # =========================================================================
    # STAGE C — SPEAKER ASSIGNMENT & TRANSCRIPT FORMATTING
    # Assign each aligned word to the speaker who was active at that timestamp,
    # then collapse the word stream into clean speaker-turn segments.
    # =========================================================================

    print("[audio_engine] Stage C — Assigning words to speakers …")

    # assign_word_speakers merges the diarization intervals with the aligned
    # word timestamps.  Returns a dict with a "segments" list where each segment
    # gains a "speaker" key (e.g. "SPEAKER_00").
    final_result: dict = whisperx.assign_word_speakers(
        diarization_result,
        aligned_result,
    )

    print("[audio_engine] Stage C — Formatting transcript …")

    formatted_lines: list[str] = []

    for segment in final_result.get("segments", []):
        # Each segment dict contains:
        #   "speaker" : str  — e.g. "SPEAKER_00" (may be absent if no diarization match)
        #   "text"    : str  — the transcribed text for this segment
        speaker: str = segment.get("speaker", "SPEAKER_UNKNOWN")
        text: str    = segment.get("text", "").strip()

        if not text:
            # Skip empty segments (can occur at audio boundaries)
            continue

        # Format: "SPEAKER_00: Let's begin the meeting …"
        formatted_lines.append(f"{speaker}: {text}")

    # Join all speaker-turn lines with a single newline separator.
    formatted_transcript: str = "\n".join(formatted_lines)

    # =========================================================================
    # PERSIST TRANSCRIPT TO DISK
    # =========================================================================

    print(f"[audio_engine] Saving transcript → '{transcript_path}' …")

    with open(transcript_path, "w", encoding="utf-8") as fh:
        fh.write(formatted_transcript)

    print(f"[audio_engine] ✓ Transcript saved ({len(formatted_lines)} speaker turns).")

    return transcript_path