# =============================================================================
# fusion_engine.py
# Part 3 — Multimodal Summarisation Pipeline: CPU-Only Context Fusion.
#
# Role:
#   Merges two text sources from Parts 1 and 2 into a single, structured
#   context document ready for LLM summarisation (Part 4):
#     - T{N}.txt : Speaker-attributed audio transcript (WhisperX + diarization)
#     - P{N}.md  : Structured slide content (LightOnOCR Markdown)
#
# Safety Valve:
#   The fused document is hard-sliced to 120,000 characters before being saved.
#   At ~4 characters per token this corresponds to ~30,000 tokens — safely below
#   the context window of all registered LLMs (32k–128k) while preventing OOM
#   crashes caused by unexpectedly long transcripts or dense slide decks.
#
# Design note:
#   This stage is intentionally CPU-only and stateless — no models are loaded.
#   It consumes negligible resources and adds zero VRAM pressure between the
#   GPU-heavy OCR (Part 2) and LLM summarisation (Part 4) stages.
# =============================================================================

import os                        # Path utilities and directory creation

from config import Prompts       # System-level prompts defined in the central config

# Hard character ceiling — prevents LLM context-window OOM on long documents.
# 120,000 chars ≈ 30,000 tokens (at ~4 chars/token), which fits all registered
# LLMs (GLM-4 9B: 128k, Llama 3.1 8B: 128k, Mistral 7B: 32k, Qwen2.5 7B: 128k).
_MAX_CHARS: int = 120_000


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def fuse_context(
    transcript_path: str,
    ocr_path: str,
    fused_output_path: str,
) -> str:
    """
    Merge the audio transcript and PDF OCR Markdown into a single, structured
    context document, enforcing a 120,000-character safety ceiling.

    Parameters
    ----------
    transcript_path : str
        Path to the WhisperX + diarization transcript
        (e.g. "outputs/transcripts/T1.txt").
    ocr_path : str
        Path to the LightOnOCR Markdown extraction
        (e.g. "outputs/ocr/P1.md").
    fused_output_path : str
        Destination path for the fused context document
        (e.g. "outputs/contexts/C1.txt").

    Returns
    -------
    str
        The path of the saved fused context file (same as *fused_output_path*).
    """

    # =========================================================================
    # READ SOURCE DOCUMENTS
    # =========================================================================

    print(f"[fusion_engine] Reading transcript → '{transcript_path}' …")

    with open(transcript_path, "r", encoding="utf-8") as fh:
        transcript_text: str = fh.read().strip()

    print(f"[fusion_engine] Reading OCR Markdown → '{ocr_path}' …")

    with open(ocr_path, "r", encoding="utf-8") as fh:
        ocr_text: str = fh.read().strip()

    # =========================================================================
    # ASSEMBLE FUSED DOCUMENT
    # Slides first, transcript second — slides provide the structural skeleton
    # (agenda, topics, decisions) while the transcript provides spoken detail.
    # =========================================================================

    fused_text: str = (
        f"=== SYSTEM FUSION PROMPT ===\n"
        f"{Prompts.CONTEXT_FUSION_PROMPT}\n\n"
        f"=== SLIDES & DOCUMENTATION (PDF OCR) ===\n"
        f"{ocr_text}\n\n"
        f"=== SPOKEN AUDIO TRANSCRIPT ===\n"
        f"{transcript_text}"
    )

    # =========================================================================
    # SAFETY VALVE — Hard 120,000-character ceiling
    # Slicing is O(1) on Python strings and always safe even if the document
    # is already shorter than the limit.  No warning is raised here; the caller
    # can compare len(fused_text) before/after the call if truncation logging
    # is needed at the orchestration layer.
    # =========================================================================

    raw_len: int = len(fused_text)
    fused_text = fused_text[:_MAX_CHARS]

    if raw_len > _MAX_CHARS:
        print(
            f"[fusion_engine] ⚠ Safety valve triggered: "
            f"document truncated from {raw_len:,} → {_MAX_CHARS:,} characters."
        )

    # =========================================================================
    # PERSIST TO DISK
    # =========================================================================

    os.makedirs(os.path.dirname(fused_output_path) or ".", exist_ok=True)

    print(f"[fusion_engine] Saving fused context → '{fused_output_path}' …")

    with open(fused_output_path, "w", encoding="utf-8") as fh:
        fh.write(fused_text)

    print(
        f"[fusion_engine] ✓ Fusion complete "
        f"({len(fused_text):,} characters written)."
    )

    return fused_output_path