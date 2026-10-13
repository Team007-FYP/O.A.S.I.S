# =============================================================================
# evaluator.py
# Part 5 — Multimodal Summarisation Pipeline: Comprehensive Automated Evaluation.
#
# Four-phase evaluation strategy, each phase loading its own model into VRAM
# and fully annihilating it before the next phase begins:
#
#   Phase A — CPU-only   : ROUGE (rouge1, rouge2, rougeL) + BERTScore
#   Phase B — GPU ~1.6GB : BARTScore (facebook/bart-large-cnn, seq2seq loss)
#   Phase C — GPU ~0.7GB : Hallucination / NLI (cross-encoder/nli-deberta-v3-base)
#   Phase D — GPU ~8GB   : LLM-as-a-Judge (Meta-Llama-3.1-8B-Instruct)
#
# All results are written back in-place to the JSON files produced by Part 4,
# progressively populating the null fields in the JSONSchemaBuilder skeleton.
# =============================================================================

import os          # Path utilities
import json        # Load and persist result JSON files
import re          # Robust integer extraction from judge LLM output
import gc          # CPython garbage collector for immediate object reclamation
import torch       # CUDA dtype, softmax, cache management

from rouge_score import rouge_scorer             # Google ROUGE implementation
from bert_score import score as bert_score_fn    # BERTScore: contextual embedding similarity

from transformers import (
    AutoModelForCausalLM,                  # Part D: LLM-as-a-Judge (decoder-only)
    AutoModelForSeq2SeqLM,                 # Part B: BARTScore (encoder-decoder)
    AutoModelForSequenceClassification,    # Part C: NLI hallucination (classifier)
    AutoTokenizer,                         # Unified tokeniser loader
)

from config import (
    MODEL_CONFIG,      # JUDGE_MODEL identifier
    INFERENCE_CONFIG,  # DTYPE (bfloat16), deterministic decoding settings
)


# =============================================================================
# HELPERS
# =============================================================================

def _load_json(path: str) -> dict:
    """Load and return a JSON file; raises on parse or IO errors."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: dict, path: str) -> None:
    """Serialise *data* to *path* with 2-space indentation and UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _annihilate(*objects) -> None:
    """
    Delete all passed references, run CPython GC, and flush the CUDA allocator.
    This is the canonical three-step VRAM annihilation sequence used across the
    entire pipeline.
    """
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def evaluate_summaries(
    json_file_paths: dict,
    gt_path: str,
) -> None:
    """
    Score all LLM-generated summaries against a ground-truth reference across
    four evaluation phases, writing enriched results back into each JSON file.

    Parameters
    ----------
    json_file_paths : dict
        Mapping of model_key -> path to the JSON result file produced by Part 4.
        Example: {"glm4": ".../R1_glm4.json", "llama": ".../R1_llama.json", …}
    gt_path : str
        Path to the ground-truth reference summary (e.g. "dataset/GT1.txt").

    Returns
    -------
    None
        All results are written back to the JSON files on disk in-place.
    """

    # =========================================================================
    # READ GROUND TRUTH
    # =========================================================================

    print(f"[evaluator] Reading ground truth → '{gt_path}' …")

    with open(gt_path, "r", encoding="utf-8") as fh:
        gt_text: str = fh.read().strip()

    print(f"[evaluator] Ground truth loaded ({len(gt_text):,} characters).")

    # =========================================================================
    # PHASE A — ROUGE & BERTScore (CPU-ONLY)
    # Zero VRAM cost. Run first so the GPU is completely free for Phases B–D.
    # =========================================================================

    print("\n[evaluator] ── Phase A: ROUGE + BERTScore (CPU) ──")

    # RougeScorer is a lightweight Python object; instantiate once and reuse.
    # use_stemmer=True applies Porter stemming before n-gram matching, making
    # scores robust to morphological variation (run / running / ran).
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    for model_key, json_file_path in json_file_paths.items():

        print(f"[evaluator]   [{model_key}] Computing ROUGE + BERTScore …")

        data: dict = _load_json(json_file_path)
        gen_summary: str = data.get("generated_summary", "").strip()

        if not gen_summary:
            print(f"[evaluator]   WARNING: No generated_summary in '{json_file_path}'. Skipping.")
            continue

        # ------------------------------------------------------------------
        # ROUGE
        # rouge_scores is a dict of Score namedtuples: .precision .recall .fmeasure
        # ------------------------------------------------------------------
        rouge_scores = scorer.score(gt_text, gen_summary)

        data["summary_evaluation"]["rouge"] = {
            "rouge1": {
                "precision": round(rouge_scores["rouge1"].precision, 4),
                "recall":    round(rouge_scores["rouge1"].recall,    4),
                "f1":        round(rouge_scores["rouge1"].fmeasure,  4),
            },
            "rouge2": {
                "precision": round(rouge_scores["rouge2"].precision, 4),
                "recall":    round(rouge_scores["rouge2"].recall,    4),
                "f1":        round(rouge_scores["rouge2"].fmeasure,  4),
            },
            "rougeL": {
                "precision": round(rouge_scores["rougeL"].precision, 4),
                "recall":    round(rouge_scores["rougeL"].recall,    4),
                "f1":        round(rouge_scores["rougeL"].fmeasure,  4),
            },
        }

        # ------------------------------------------------------------------
        # BERTScore
        # Computes contextual embedding similarity using roberta-large (default
        # for lang="en").  Returns [1]-element tensors; extract scalars via .item().
        # verbose=False suppresses the tqdm progress bar for clean logging.
        # ------------------------------------------------------------------
        bert_P, bert_R, bert_F1 = bert_score_fn(
            [gen_summary],
            [gt_text],
            lang="en",
            verbose=False,
        )

        data["summary_evaluation"]["bertscore"] = {
            "precision": round(bert_P[0].item(),  4),
            "recall":    round(bert_R[0].item(),  4),
            "f1":        round(bert_F1[0].item(), 4),
            "model":     "roberta-large",
        }

        _save_json(data, json_file_path)

        print(
            f"[evaluator]   ✓ [{model_key}] "
            f"ROUGE-L F1={data['summary_evaluation']['rouge']['rougeL']['f1']} | "
            f"BERTScore F1={data['summary_evaluation']['bertscore']['f1']}"
        )

    print("[evaluator] Phase A complete.\n")

    # =========================================================================
    # PHASE B — BARTScore (facebook/bart-large-cnn)
    # BARTScore computes the average log-probability the BART model assigns to
    # the generated summary conditioned on the reference, via the seq2seq loss.
    # Higher (less negative) scores indicate better quality summaries.
    # Model size: ~1.6 GB in float32.
    # =========================================================================

    print("[evaluator] ── Phase B: BARTScore (GPU) ──")

    bart_model_id: str = "facebook/bart-large-cnn"

    print(f"[evaluator] Loading BARTScore model '{bart_model_id}' …")

    bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_id)

    bart_model = AutoModelForSeq2SeqLM.from_pretrained(
        bart_model_id,
        device_map="cuda",   # BART is float32 by default; ~1.6 GB on A100
    )
    bart_model.eval()

    print("[evaluator] BARTScore model loaded.")

    for model_key, json_file_path in json_file_paths.items():

        print(f"[evaluator]   [{model_key}] Computing BARTScore …")

        data: dict = _load_json(json_file_path)
        gen_summary: str = data.get("generated_summary", "").strip()

        if not gen_summary:
            print(f"[evaluator]   WARNING: No generated_summary. Skipping.")
            continue

        # ------------------------------------------------------------------
        # Tokenize the ground truth as the encoder input and the generated
        # summary as the decoder target (labels).
        # max_length=1024 matches BART-large-CNN's positional embedding limit.
        # truncation=True silently truncates texts beyond the limit.
        # ------------------------------------------------------------------
        inputs = bart_tokenizer(
            gt_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        labels = bart_tokenizer(
            gen_summary,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        # ------------------------------------------------------------------
        # Forward pass: the seq2seq model returns cross-entropy loss when
        # labels are supplied.  Negating gives the log-probability score:
        #   bart_score = -loss  (higher = more likely = better summary)
        # ------------------------------------------------------------------
        with torch.no_grad():
            outputs = bart_model(**inputs, labels=labels["input_ids"])

        bart_score: float = -outputs.loss.item()

        data["summary_evaluation"]["bartscore"]["score"] = round(bart_score, 4)
        data["summary_evaluation"]["bartscore"]["model"] = bart_model_id

        _save_json(data, json_file_path)

        # Free per-iteration CUDA tensors immediately
        del inputs, labels, outputs

        print(f"[evaluator]   ✓ [{model_key}] BARTScore={round(bart_score, 4)}")

    # VRAM ANNIHILATION — Phase B
    print("[evaluator] Annihilating BARTScore model from VRAM …")
    _annihilate(bart_model, bart_tokenizer)
    print("[evaluator] Phase B complete.\n")

    # =========================================================================
    # PHASE C — HALLUCINATION & FACTUAL CONSISTENCY (NLI)
    # Uses a DeBERTa-v3-base cross-encoder fine-tuned for NLI.
    # Premise = ground truth; Hypothesis = generated summary.
    # Label ordering for this model: [contradiction, entailment, neutral]
    # (confirmed from the cross-encoder/nli-deberta-v3-base model card).
    # Model size: ~0.7 GB.
    # =========================================================================

    print("[evaluator] ── Phase C: Hallucination / NLI (GPU) ──")

    nli_model_id: str = "cross-encoder/nli-deberta-v3-base"

    print(f"[evaluator] Loading NLI model '{nli_model_id}' …")

    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)

    nli_model = AutoModelForSequenceClassification.from_pretrained(
        nli_model_id,
        device_map="cuda",
    )
    nli_model.eval()

    print("[evaluator] NLI model loaded.")

    for model_key, json_file_path in json_file_paths.items():

        print(f"[evaluator]   [{model_key}] Computing NLI hallucination score …")

        data: dict = _load_json(json_file_path)
        gen_summary: str = data.get("generated_summary", "").strip()

        if not gen_summary:
            print(f"[evaluator]   WARNING: No generated_summary. Skipping.")
            continue

        # ------------------------------------------------------------------
        # Tokenize the (premise, hypothesis) pair.
        # Premise   = gt_text       (what actually happened)
        # Hypothesis = gen_summary  (what the model claims happened)
        # Contradiction probability ≈ hallucination rate.
        # Entailment probability    ≈ factual consistency score.
        # ------------------------------------------------------------------
        nli_inputs = nli_tokenizer(
            gt_text,
            gen_summary,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            logits = nli_model(**nli_inputs).logits

        # Softmax over the 3 classes: [contradiction, entailment, neutral]
        probs = torch.softmax(logits, dim=1)

        contradiction_prob: float = probs[0][0].item()   # Index 0 → contradiction
        entailment_prob: float    = probs[0][1].item()   # Index 1 → entailment

        data["hallucination"]["hallucination_rate"]        = round(contradiction_prob, 4)
        data["hallucination"]["factual_consistency_score"] = round(entailment_prob,    4)
        data["hallucination"]["entailment_score"]          = round(entailment_prob,    4)
        data["hallucination"]["notes"] = (
            "Calculated natively using cross-encoder/nli-deberta-v3-base."
        )

        _save_json(data, json_file_path)

        del nli_inputs, logits, probs

        print(
            f"[evaluator]   ✓ [{model_key}] "
            f"Entailment={round(entailment_prob, 4)} | "
            f"Contradiction={round(contradiction_prob, 4)}"
        )

    # VRAM ANNIHILATION — Phase C
    print("[evaluator] Annihilating NLI model from VRAM …")
    _annihilate(nli_model, nli_tokenizer)
    print("[evaluator] Phase C complete.\n")

    # =========================================================================
    # PHASE D — LLM-AS-A-JUDGE (Meta-Llama-3.1-8B-Instruct)
    # The judge model is loaded ONCE, scores all summaries sequentially, then
    # fully evicted.  max_new_tokens=10 is intentionally tight — the judge only
    # needs to emit the integer score (optionally preceded by one word).
    # Model size: ~8 GB in bfloat16.
    # =========================================================================

    print("[evaluator] ── Phase D: LLM-as-a-Judge (GPU) ──")

    judge_model_id: str = MODEL_CONFIG.JUDGE_MODEL

    print(f"[evaluator] Loading judge model '{judge_model_id}' …")

    judge_tokenizer = AutoTokenizer.from_pretrained(
        judge_model_id,
        trust_remote_code=True,
    )

    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype=INFERENCE_CONFIG.DTYPE,   # torch.bfloat16
        device_map="cuda",
        trust_remote_code=True,
    )
    judge_model.eval()

    print("[evaluator] Judge model loaded.")

    for model_key, json_file_path in json_file_paths.items():

        print(f"[evaluator]   [{model_key}] Judging summary …")

        data: dict = _load_json(json_file_path)
        gen_summary: str = data.get("generated_summary", "").strip()

        if not gen_summary:
            print(f"[evaluator]   WARNING: No generated_summary. Assigning score 0.")
            data["judge_score"] = 0
            _save_json(data, json_file_path)
            continue

        # ------------------------------------------------------------------
        # JUDGE PROMPT
        # Explicit "Output ONLY the numerical integer score at the end" keeps
        # the generation tightly constrained so regex extraction is reliable.
        # ------------------------------------------------------------------
        judge_instruction: str = (
            "You are an impartial evaluation judge. "
            "Rate the candidate summary against the ground truth summary on a "
            "scale of 1 to 10 based on factual accuracy, conciseness, and coverage. "
            "Output ONLY the numerical integer score at the end.\n\n"
            f"[GROUND TRUTH]: {gt_text}\n\n"
            f"[CANDIDATE SUMMARY]: {gen_summary}"
        )

        judge_messages: list[dict] = [
            {"role": "user", "content": judge_instruction}
        ]

        # ------------------------------------------------------------------
        # PROMPT FORMATTING — chat template with manual fallback
        # ------------------------------------------------------------------
        try:
            judge_inputs = judge_tokenizer.apply_chat_template(
                judge_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Normalise: apply_chat_template may return a raw Tensor
            if isinstance(judge_inputs, torch.Tensor):
                judge_inputs = {"input_ids": judge_inputs}

        except Exception as template_err:
            print(
                f"[evaluator]   Chat template failed ({template_err}); "
                "using manual prompt format."
            )
            fallback: str = f"User: {judge_instruction}\n\nAssistant:"
            judge_inputs = judge_tokenizer(fallback, return_tensors="pt")

        judge_inputs = {k: v.to("cuda") for k, v in judge_inputs.items()}
        prompt_len: int = judge_inputs["input_ids"].shape[1]

        # ------------------------------------------------------------------
        # GENERATE — max_new_tokens=10: enough for a digit (and brief preamble)
        # do_sample=False enforces greedy / deterministic decoding.
        # ------------------------------------------------------------------
        with torch.inference_mode():
            judge_output_ids = judge_model.generate(
                **judge_inputs,
                max_new_tokens=10,   # Score only — intentionally tight
                do_sample=False,
            )

        # Decode only the newly generated tokens (strip prompt echo)
        judge_response: str = judge_tokenizer.decode(
            judge_output_ids[0, prompt_len:],
            skip_special_tokens=True,
        ).strip()

        # ------------------------------------------------------------------
        # EXTRACT INTEGER SCORE
        # \b(?:10|[1-9])\b — matches standalone integers 1–10.
        # matches[-1]: take the LAST occurrence so any "scale of 1 to 10"
        # preamble is ignored in favour of the final verdict digit.
        # ------------------------------------------------------------------
        matches: list[str] = re.findall(r"\b(?:10|[1-9])\b", judge_response)
        judge_score: int = int(matches[-1]) if matches else 0

        data["judge_score"] = judge_score
        _save_json(data, json_file_path)

        del judge_inputs, judge_output_ids

        print(
            f"[evaluator]   ✓ [{model_key}] "
            f"Judge response: '{judge_response[:60]}' → score: {judge_score}"
        )

    # VRAM ANNIHILATION — Phase D
    print("[evaluator] Annihilating judge model from VRAM …")
    _annihilate(judge_model, judge_tokenizer)

    print(
        f"\n[evaluator] ✓ All 4 phases complete. "
        f"{len(json_file_paths)} model result(s) fully evaluated."
    )