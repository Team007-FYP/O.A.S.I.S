# =============================================================================
# main.py
# Master Orchestrator — 40GB A100 Multimodal Summarisation Benchmarking Pipeline.
#
# Input directory layout (under ./dataset/):
#   dataset/Audios/     A1.wav, A2.wav, …      — meeting audio recordings
#   dataset/Slides/     P1.pdf, P2.pdf, …      — corresponding slide decks
#   dataset/Summaries/  S1.txt, S2.txt, …      — ground-truth reference summaries
#
# Execution order per meeting index N:
#   Part 1  audio_engine.py   → A{N}.wav  → T{N}.txt  (transcription + diarization)
#   Part 2  pdf_engine.py     → P{N}.pdf  → P{N}.md   (VLM OCR)
#   Part 3  fusion_engine.py  → T{N}.txt + P{N}.md → C{N}.txt  (context fusion)
#   Part 4  llm_engine.py     → C{N}.txt  → R{N}_*.json  (4 LLM summaries + telemetry)
#   Part 5  evaluator.py      → R{N}_*.json + S{N}.txt → scores injected in-place
#   Part 6  telemetry.py      → outputs/evaluations/ → master_report.md
#
# Resilience:
#   Each meeting index is wrapped in try/except so a single failing audio file
#   does not abort the entire benchmark run.  The traceback is printed and the
#   pipeline continues with the next index.
# =============================================================================

import os           # Directory creation, path joining
import glob         # Wildcard file discovery for audio inputs
import re           # Numeric index extraction from filenames
import traceback    # Full stack trace on per-index failures

# Pipeline module imports
from config import Paths                                # Centralised directory constants
from audio_engine import process_audio                  # Part 1 — transcription + diarization
from pdf_engine import process_pdf                     # Part 2 — VLM OCR (olmOCR-2)
from fusion_engine import fuse_context                  # Part 3 — context assembly
from llm_engine import run_llm_summarization            # Part 4 — 4× LLM summarisation
from evaluator import evaluate_summaries                # Part 5 — ROUGE / BERTScore / Judge
from telemetry import ReportGenerator                   # Part 6 — Markdown master report


# =============================================================================
# DIRECTORY BOOTSTRAP
# Create all output directories up-front so every pipeline stage can write
# without worrying about mkdir at the point of use.
# Also creates the three dataset sub-directories so they exist even on first run.
# Iterating over Paths.__dataclass_fields__ ensures this stays in sync
# automatically whenever a new path is added to the Paths dataclass.
# =============================================================================

def _bootstrap_directories() -> None:
    """Create all configured directories declared in Paths (idempotent)."""
    paths_instance: Paths = Paths()

    for field_name in paths_instance.__dataclass_fields__:
        dir_path: str = getattr(paths_instance, field_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[main] Directory ensured: '{dir_path}'")


# =============================================================================
# UNIVERSAL FILE DISCOVERY
# New layout:
#   dataset/Audios/     → A{N}.wav
#   dataset/Slides/     → P{N}.pdf
#   dataset/Summaries/  → S{N}.txt  (ground-truth; previously GT{N}.txt)
#
# Discovery logic:
#   1. Glob AUDIOS_DIR for A*.wav files.
#   2. Extract numeric index N from each filename.
#   3. Verify P{N}.pdf in SLIDES_DIR and S{N}.txt in SUMMARIES_DIR.
#   4. Warn and skip if any companion file is missing.
# Returns a list of validated (N, wav_path, pdf_path, gt_path) tuples sorted
# numerically by N so processing order is deterministic.
# =============================================================================

def _discover_meeting_files() -> list[tuple[str, str, str, str]]:
    """
    Scan sub-directories for meeting file groups and validate all companions.

    Returns
    -------
    list of (index_str, wav_path, pdf_path, summary_path) tuples, sorted by N.
    """
    paths_instance: Paths = Paths()

    audios_dir:    str = paths_instance.AUDIOS_DIR       # dataset/Audios/
    slides_dir:    str = paths_instance.SLIDES_DIR       # dataset/Slides/
    summaries_dir: str = paths_instance.SUMMARIES_DIR    # dataset/Summaries/

    # Glob for all audio files matching A*.wav inside the Audios sub-folder.
    wav_pattern: str = os.path.join(audios_dir, "A*.wav")
    wav_files: list[str] = sorted(glob.glob(wav_pattern))

    if not wav_files:
        print(
            f"[main] WARNING: No A*.wav files found in '{audios_dir}'. "
            "Nothing to process."
        )
        return []

    print(f"[main] Found {len(wav_files)} audio file(s) in '{audios_dir}'.")

    validated: list[tuple[str, str, str, str]] = []

    for wav_path in wav_files:
        filename: str = os.path.basename(wav_path)   # e.g. "A1.wav"

        # Extract numeric index N from the filename (e.g. "A1.wav" → "1").
        match = re.search(r"A(\d+)\.wav$", filename, re.IGNORECASE)
        if not match:
            print(
                f"[main] WARNING: '{filename}' does not match the expected "
                f"'A{{N}}.wav' pattern. Skipping."
            )
            continue

        N: str = match.group(1)   # Numeric index as string, e.g. "1"

        # Build expected companion file paths in their respective sub-folders.
        pdf_path: str     = os.path.join(slides_dir,    f"P{N}.pdf")
        summary_path: str = os.path.join(summaries_dir, f"S{N}.txt")

        # Validate that both companion files exist before committing this index.
        missing: list[str] = []
        if not os.path.isfile(pdf_path):
            missing.append(f"Slides/P{N}.pdf")
        if not os.path.isfile(summary_path):
            missing.append(f"Summaries/S{N}.txt")

        if missing:
            print(
                f"[main] WARNING: Skipping index {N} — missing companion file(s): "
                + ", ".join(missing)
            )
            continue

        validated.append((N, wav_path, pdf_path, summary_path))
        print(
            f"[main] ✓ Index {N} validated: "
            f"Audios/A{N}.wav | Slides/P{N}.pdf | Summaries/S{N}.txt"
        )

    # Sort by integer value of N — avoids lexicographic misordering (1, 10, 2).
    validated.sort(key=lambda t: int(t[0]))

    return validated


# =============================================================================
# PER-INDEX PIPELINE EXECUTION
# Runs Parts 1–5 for a single meeting index N.
# Called inside the main loop, wrapped in try/except for resilience.
# =============================================================================

def _run_pipeline_for_index(
    N: str,
    wav_path: str,
    pdf_path: str,
    summary_path: str,
) -> None:
    """
    Execute Parts 1–5 of the pipeline for meeting index N.

    Parameters
    ----------
    N : str
        Numeric string index (e.g. "1").
    wav_path : str
        Validated path to the audio file  (dataset/Audios/A{N}.wav).
    pdf_path : str
        Validated path to the slide deck  (dataset/Slides/P{N}.pdf).
    summary_path : str
        Validated path to the ground-truth summary (dataset/Summaries/S{N}.txt).
    """
    paths_instance: Paths = Paths()

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE START — Meeting Index {N}")
    print(f"{'=' * 70}")

    # -------------------------------------------------------------------------
    # PART 1 — AUDIO TRANSCRIPTION + DIARIZATION
    # Input : dataset/Audios/A{N}.wav
    # Output: outputs/transcripts/T{N}.txt
    # -------------------------------------------------------------------------
    print(f"\n[main] ── Part 1: Audio Engine (index={N}) ──")

    transcript_path: str = os.path.join(
        paths_instance.TRANSCRIPTS_DIR, f"T{N}.txt"
    )

    process_audio(wav_path, transcript_path)

    print(f"[main] Part 1 complete → '{transcript_path}'")

    # -------------------------------------------------------------------------
    # PART 2 — PDF OCR (olmOCR-2 VLM)
    # Input : dataset/Slides/P{N}.pdf
    # Output: outputs/ocr/P{N}.md
    # -------------------------------------------------------------------------
    print(f"\n[main] ── Part 2: PDF Engine (index={N}) ──")

    ocr_path: str = os.path.join(
        paths_instance.OCR_DIR, f"P{N}.md"
    )

    process_pdf(pdf_path, ocr_path)

    print(f"[main] Part 2 complete → '{ocr_path}'")

    # -------------------------------------------------------------------------
    # PART 3 — CONTEXT FUSION (CPU-only)
    # Input : T{N}.txt + P{N}.md
    # Output: outputs/contexts/C{N}.txt
    # -------------------------------------------------------------------------
    print(f"\n[main] ── Part 3: Fusion Engine (index={N}) ──")

    context_path: str = os.path.join(
        paths_instance.CONTEXT_DIR, f"C{N}.txt"
    )

    fuse_context(transcript_path, ocr_path, context_path)

    print(f"[main] Part 3 complete → '{context_path}'")

    # -------------------------------------------------------------------------
    # PART 4 — SEQUENTIAL LLM SUMMARISATION
    # Input : C{N}.txt
    # Output: outputs/evaluations/R{N}_glm4.json, _llama.json, _mistral.json, _qwen.json
    # -------------------------------------------------------------------------
    print(f"\n[main] ── Part 4: LLM Engine (index={N}) ──")

    json_paths_dict: dict = run_llm_summarization(
        context_path,
        f"R{N}",                          # base_index e.g. "R1"
        paths_instance.EVALUATIONS_DIR,
    )

    print(
        f"[main] Part 4 complete — "
        f"{len(json_paths_dict)} model result(s) saved."
    )

    # -------------------------------------------------------------------------
    # PART 5 — AUTOMATED EVALUATION
    # Input : R{N}_*.json + dataset/Summaries/S{N}.txt (ground truth)
    # Output: JSON files updated in-place with ROUGE, BERTScore, BARTScore,
    #         NLI hallucination metrics, and LLM-as-a-Judge scores.
    # -------------------------------------------------------------------------
    print(f"\n[main] ── Part 5: Evaluator (index={N}) ──")

    evaluate_summaries(json_paths_dict, summary_path)

    print(f"[main] Part 5 complete — all JSON files updated with scores.")

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — Meeting Index {N}")
    print(f"{'=' * 70}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  MULTIMODAL SUMMARISATION BENCHMARKING PIPELINE")
    print("  40GB A100 | WhisperX + olmOCR-2 + 4× LLM + Llama Judge")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # STEP 0 — Bootstrap all output directories (and dataset sub-dirs)
    # ------------------------------------------------------------------
    print("[main] Bootstrapping directories …")
    _bootstrap_directories()

    # ------------------------------------------------------------------
    # STEP 1 — Discover and validate all meeting file groups
    # ------------------------------------------------------------------
    print("\n[main] Discovering meeting files …")
    meeting_files: list[tuple[str, str, str, str]] = _discover_meeting_files()

    if not meeting_files:
        print("[main] No valid meeting groups found. Exiting.")
        raise SystemExit(1)

    print(
        f"\n[main] {len(meeting_files)} valid meeting group(s) found. "
        f"Starting pipeline …"
    )

    # ------------------------------------------------------------------
    # STEP 2 — Main execution loop (resilient: one failure ≠ full abort)
    # ------------------------------------------------------------------
    successful: list[str] = []
    failed: list[str] = []

    for N, wav_path, pdf_path, summary_path in meeting_files:

        try:
            _run_pipeline_for_index(N, wav_path, pdf_path, summary_path)
            successful.append(N)

        except Exception as exc:
            print(
                f"\n[main] ✗ PIPELINE FAILED for index {N}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            print(f"[main] Continuing to next index …\n")
            failed.append(N)

    # ------------------------------------------------------------------
    # STEP 3 — Post-loop summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXECUTION LOOP COMPLETE")
    print("=" * 70)
    print(f"  ✓ Successful indices : {successful if successful else 'none'}")
    print(f"  ✗ Failed indices     : {failed if failed else 'none'}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # STEP 4 — Part 6: Aggregated Markdown Master Report
    # Scans ALL JSON files in EVALUATIONS_DIR (including prior runs),
    # computes composite scores, and writes master_report.md.
    # Runs even on partial failures so partial results are still reported.
    # ------------------------------------------------------------------
    print("[main] ── Part 6: Generating Master Report ──")

    paths_instance: Paths = Paths()

    report_generator: ReportGenerator = ReportGenerator()
    report_generator.generate_html(
        evals_dir=paths_instance.EVALUATIONS_DIR,
        output_path="master_report.html",
    )

    print("\n[main] ✓ All done. Results: outputs/evaluations/")
    print("[main] ✓ Master report:  master_report.html")