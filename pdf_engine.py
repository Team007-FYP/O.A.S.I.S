# =============================================================================
# pdf_engine.py
# Part 2 — Multimodal Summarisation Pipeline: PDF → Structured Markdown
#           via Moondream2 (vikhyatk/moondream2, revision 2025-06-21).
#
# API used:
#   model.encode_image(img)  → encoded image tensor
#   model.answer_question(encoded_image, query_text, tokenizer)  → answer str
#
#   model.query() is intentionally NOT used — it is an unstable convenience
#   wrapper that breaks across revisions.  The encode_image + answer_question
#   two-step is the stable, documented Moondream2 API.
#
# VRAM budget:
#   Moondream2 in bfloat16 ≈ 1.8 GB.  Negligible on a 40 GB A100.
#   Per-page encoded tensors are freed after each slide to prevent
#   accumulation across long decks.
# =============================================================================

import os           # Path validation, directory creation
import gc           # CPython garbage collector — forces immediate reclamation
import torch        # CUDA dtype and cache management

from pdf2image import convert_from_path                 # PDF → list[PIL.Image]
from transformers import (
    AutoModelForCausalLM,    # Loads Moondream2 model weights
    AutoTokenizer,           # Loads Moondream2 tokenizer
)


# =============================================================================
# EXTRACTION PROMPT
# Kept at module level so it is defined once and shared across all page
# iterations without re-construction inside the loop.
# Rules are deliberately minimal to avoid instruction dilution.
# =============================================================================

_QUERY_TEXT: str = (
    "1. Transcribe body text and bullet points exactly as they appear, in "
    "natural reading order. Preserve bullet formatting only if the slide "
    "itself uses bullets. "
    "2. Zero hallucination: never invent, infer, or guess any text, number, "
    "or label. If something is genuinely illegible, write [unreadable]. "
    "3. Table detection and skip: if the slide contains a spreadsheet-style "
    "table, grid, or dense numerical/financial data table, do NOT attempt to "
    "transcribe it. Instead respond with exactly: [SKIPPED_TABLE_SLIDE] "
    "and nothing else. "
    "4. No conversational filler, no introductions, no conclusions — begin "
    "extraction immediately."
)

# Sentinel returned by the model when a table/financial slide is detected.
_TABLE_SENTINEL: str = "[SKIPPED_TABLE_SLIDE]"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_pdf(pdf_path: str, md_path: str) -> str:
    """
    Rasterize a PDF slide deck, run each page through Moondream2 for
    structured text extraction, and persist the joined Markdown to disk.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF (e.g. "dataset/Slides/P1.pdf").
    md_path : str
        Destination path for the extracted Markdown (e.g. "outputs/ocr/P1.md").

    Returns
    -------
    str
        The path of the saved output file (same as *md_path*).

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist on disk.
    """

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"[pdf_engine] PDF not found: '{pdf_path}'. "
            "Ensure the file exists in dataset/Slides/ before running the pipeline."
        )

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    # =========================================================================
    # RASTERIZATION
    # 200 DPI balances text legibility with per-page tensor size.
    # Each PIL Image is converted to RGB so Moondream2's image encoder always
    # receives a consistent 3-channel input regardless of the PDF colour space
    # (some PDFs embed RGBA or palette-mode page backgrounds).
    # =========================================================================

    print(f"[pdf_engine] Rasterizing '{pdf_path}' at 200 DPI …")

    raw_pages: list = convert_from_path(pdf_path, dpi=200)
    pages: list     = [p.convert("RGB") for p in raw_pages]

    total_pages: int = len(pages)
    print(f"[pdf_engine] {total_pages} page(s) rasterized and converted to RGB.")

    # =========================================================================
    # LOAD MOONDREAM2
    # device_map={"": "cuda"} places all layers on GPU 0.
    # Moondream2's custom encode_image / answer_question path requires the
    # full model to reside on a single device — partial CPU offloading breaks it.
    # revision="2025-06-21" pins exact weights for reproducible benchmark runs.
    # =========================================================================

    MOONDREAM_ID: str       = "vikhyatk/moondream2"
    MOONDREAM_REVISION: str = "2025-06-21"

    print(f"[pdf_engine] Loading Moondream2 '{MOONDREAM_ID}' (revision={MOONDREAM_REVISION}) …")

    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,          # Required for Moondream2's custom code
        device_map={"": "cuda"},         # Full GPU placement — mandatory for stable API
        torch_dtype=torch.bfloat16,      # ~1.8 GB; optimal on A100 Tensor Cores
    )
    model.eval()   # Disable dropout; required for deterministic extraction

    tokenizer = AutoTokenizer.from_pretrained(
        MOONDREAM_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
    )

    print("[pdf_engine] Moondream2 loaded.")

    # =========================================================================
    # PROCESSING LOOP
    # =========================================================================

    master_text: list[str] = []

    for page_idx, img in enumerate(pages, start=1):

        print(f"[pdf_engine]   Page {page_idx}/{total_pages} …")

        encoded_image = None   # Ensure the name exists for the finally-style cleanup

        try:
            # ------------------------------------------------------------------
            # STABLE API: encode_image → answer_question
            # encode_image runs the vision encoder and caches the image features
            # as a CUDA tensor.  answer_question performs autoregressive decoding
            # conditioned on those features and the query string.
            # ------------------------------------------------------------------
            encoded_image = model.encode_image(img)

            raw_answer: str = model.answer_question(
                encoded_image,
                _QUERY_TEXT,
                tokenizer,
            )

            answer: str = raw_answer.strip()

        except Exception as exc:
            # One bad page must never abort the entire deck.
            print(
                f"[pdf_engine]   WARNING: page {page_idx} extraction failed "
                f"({type(exc).__name__}: {exc}). Using fallback."
            )
            answer = "[Error during extraction]"

        # ------------------------------------------------------------------
        # FORMAT SLIDE ENTRY
        # Table slides are replaced with a human-readable skip notice so the
        # fusion stage does not receive the raw sentinel string.
        # ------------------------------------------------------------------
        if answer == _TABLE_SENTINEL:
            slide_entry: str = f"### Slide {page_idx}\n\n[Skipped: table/financial data detected]"
        else:
            slide_entry = f"### Slide {page_idx}\n\n{answer}"

        master_text.append(slide_entry)

        # ------------------------------------------------------------------
        # PER-PAGE VRAM CLEANUP
        # Freeing the encoded_image tensor after each slide prevents CUDA
        # memory from accumulating linearly with the number of slides.
        # A 50-slide deck would otherwise hold 50 encoded tensors in VRAM
        # simultaneously, which is unnecessary given sequential processing.
        # ------------------------------------------------------------------
        if encoded_image is not None:
            del encoded_image

        torch.cuda.empty_cache()

        print(
            f"[pdf_engine]   Page {page_idx} done "
            f"({'skipped' if answer == _TABLE_SENTINEL else f'{len(answer)} chars'})."
        )

    # =========================================================================
    # STRICT VRAM ANNIHILATION
    # Moondream2 (~1.8 GB) must be fully evicted before Part 4 (LLM engine)
    # loads its ~8 GB summarisation models.
    # =========================================================================

    print("[pdf_engine] Annihilating Moondream2 from VRAM …")

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    print("[pdf_engine] VRAM annihilation complete.")

    # =========================================================================
    # OUTPUT FORMATTING AND PERSISTENCE
    # =========================================================================

    final_output: str = "\n\n---\n\n".join(master_text)

    print(f"[pdf_engine] Saving Markdown → '{md_path}' …")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(final_output)

    print(
        f"[pdf_engine] ✓ Extraction complete — "
        f"{total_pages} page(s) written to '{md_path}'."
    )

    return md_path
