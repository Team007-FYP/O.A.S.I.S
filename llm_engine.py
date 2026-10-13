# =============================================================================
# llm_engine.py
# Part 4 — Multimodal Summarisation Pipeline: Sequential LLM Summarisation.
#
# Runs 4 causal LLMs sequentially on a single 40 GB A100 GPU:
#   glm4    — THUDM/glm-4-9b-chat-hf
#   llama   — meta-llama/Meta-Llama-3.1-8B-Instruct
#   mistral — mistralai/Mistral-7B-Instruct-v0.3
#   qwen    — Qwen/Qwen2.5-7B-Instruct
#
# Optimisations vs. the previous version:
#   - Flash Attention 2 (attn_implementation="flash_attention_2") cuts both
#     VRAM usage and inference latency for long fused-context inputs.
#   - Explicit temperature=None / top_p=None / top_k=None suppresses
#     HuggingFace UserWarnings when a model's generation_config.json contains
#     sampling defaults that conflict with do_sample=False.
#   - pad_token_id=tokenizer.eos_token_id prevents generation warnings on
#     models that lack a dedicated padding token.
#   - attention_mask is passed explicitly to model.generate() for correctness
#     with left-padded inputs from apply_chat_template.
#
# VRAM discipline:
#   Each model is fully annihilated (del + gc + empty_cache) before the next
#   model is loaded, giving every model the full 40 GB A100 headroom.
# =============================================================================

import os          # Path utilities, directory creation
import json        # Serialise result dictionaries to disk
import time        # Wall-clock timestamps for latency and throughput
import gc          # CPython garbage collector for immediate object reclamation
import torch       # CUDA cache management via torch.cuda.empty_cache()

from transformers import (
    AutoModelForCausalLM,   # Unified causal LLM loader
    AutoTokenizer,          # Chat-template-aware tokeniser
)

from config import (
    PATHS,            # Output directories for contexts and evaluations
    MODEL_CONFIG,      # LLM_MODELS registry (key → HuggingFace model ID)
    INFERENCE_CONFIG,  # DTYPE, MAX_NEW_TOKENS
    PROMPTS,          # SUMMARIZATION_PROMPT system instruction
)

from telemetry import (
    HardwareSpy,          # Background VRAM peak tracker
    JSONSchemaBuilder,    # Standardised result skeleton factory
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_llm_summarization(
    fused_context_path: str,
    base_index: str,
    output_dir: str,
) -> dict:
    """
    Run all registered LLMs sequentially over the fused multimodal context and
    persist per-model result JSON files enriched with hardware efficiency metrics.

    Parameters
    ----------
    fused_context_path : str
        Path to the fused context document produced by Part 3
        (e.g. "outputs/contexts/C1.txt").
    base_index : str
        Unique identifier prefix for this meeting/run (e.g. "R1").
        Used as the JSON filename prefix and experiment ID in the schema.
    output_dir : str
        Directory where per-model JSON results are written
        (e.g. "outputs/evaluations/").

    Returns
    -------
    dict
        Mapping of model_key -> absolute path to its saved JSON result file.
        Example: {"glm4": ".../R1_glm4.json", "llama": ".../R1_llama.json", …}
    """

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # READ FUSED CONTEXT
    # C{N}.txt is the safety-valved (≤120,000 char) multimodal context document
    # assembled by fusion_engine.py (Part 3).
    # =========================================================================

    print(f"[llm_engine] Reading fused context → '{fused_context_path}' …")

    with open(fused_context_path, "r", encoding="utf-8") as fh:
        fused_context: str = fh.read().strip()

    print(f"[llm_engine] Fused context loaded ({len(fused_context):,} characters).")

    # Accumulator: model_key → path of the saved JSON result file.
    generated_results_paths: dict[str, str] = {}

    model_registry: dict[str, str] = MODEL_CONFIG.LLM_MODELS
    total_models: int = len(model_registry)

    # =========================================================================
    # SEQUENTIAL MODEL LOOP
    # One model resident in VRAM at a time.  Full annihilation between models
    # guarantees the A100's 40 GB is always available to the current model.
    # =========================================================================

    for model_num, (model_key, model_id) in enumerate(model_registry.items(), start=1):

        print(
            f"\n[llm_engine] ── Model {model_num}/{total_models}: "
            f"'{model_key}' ({model_id}) ──"
        )

        # =====================================================================
        # TELEMETRY START
        # HardwareSpy is armed BEFORE model load so the VRAM ramp-up during
        # weight transfer is captured in the peak measurement.
        # =====================================================================

        spy: HardwareSpy = HardwareSpy()
        spy.start()

        t_start: float = time.time()

        # =====================================================================
        # TOKENIZER LOAD
        # =====================================================================

        print(f"[llm_engine]   Loading tokenizer …")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,   # Required for THUDM/glm-4-9b-chat-hf
        )

        # =====================================================================
        # MODEL LOAD — Flash Attention 2
        # attn_implementation="flash_attention_2" replaces the standard scaled
        # dot-product attention kernel with Tri-Dao's memory-efficient FA2 kernel:
        #   - O(N) memory instead of O(N²) for long sequences
        #   - ~2–4× faster prefill on long fused-context inputs
        #   - Requires Ampere+ GPU (A100 ✓) and bfloat16/float16 dtype
        # =====================================================================

        print(
            f"[llm_engine]   Loading model "
            f"(dtype={INFERENCE_CONFIG.DTYPE}, FA2) …"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=INFERENCE_CONFIG.DTYPE,          # torch.bfloat16
            device_map="cuda",                          # All layers on GPU 0
            trust_remote_code=True,
            attn_implementation="flash_attention_2",    # Memory-efficient attention
        )
        model.eval()   # Disable dropout for deterministic inference

        t_loaded: float = time.time()
        latency_sec: float = round(t_loaded - t_start, 4)

        print(f"[llm_engine]   Model loaded in {latency_sec}s.")

        # =====================================================================
        # PROMPT FORMATTING
        # Single user turn: SUMMARIZATION_PROMPT prepended to the fused context
        # with a blank line separator.  No system role to maximise compatibility
        # across all four model families without a chat-template fallback.
        # =====================================================================

        messages: list[dict] = [
            {
                "role": "user",
                "content": PROMPTS.SUMMARIZATION_PROMPT + "\n\n" + fused_context,
            }
        ]

        print(f"[llm_engine]   Formatting prompt …")

        # apply_chat_template with return_dict=True returns a BatchFeature
        # containing input_ids and attention_mask as PyTorch tensors.
        # add_generation_prompt=True appends the assistant turn-start token.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        input_tokens: int = inputs["input_ids"].shape[1]
        print(f"[llm_engine]   Prompt length: {input_tokens} tokens.")

        # =====================================================================
        # OPTIMISED INFERENCE
        # do_sample=False            → greedy decoding (deterministic, T=0.0)
        # temperature/top_p/top_k=None → explicitly disable sampling knobs to
        #                               suppress HF UserWarnings from models
        #                               that set sampling defaults in their
        #                               generation_config.json
        # pad_token_id=eos_token_id  → prevents generation warning on models
        #                               without a dedicated padding token
        # attention_mask             → passed explicitly for correctness with
        #                               any left-padded chat-template output
        # =====================================================================

        print(
            f"[llm_engine]   Generating "
            f"(max_new_tokens={INFERENCE_CONFIG.MAX_NEW_TOKENS}) …"
        )

        with torch.inference_mode():   # Disables autograd — reduces VRAM overhead
            outputs = model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG.MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,                        # Suppress sampling
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,     # Silence pad-token warning
            )

        t_end: float = time.time()

        # Decode only the newly generated tokens — strip the prompt echo.
        summary_text: str = tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        ).strip()

        output_tokens: int = len(outputs[0]) - input_tokens

        inference_time_sec: float = round(t_end - t_loaded, 4)
        throughput_tokens_per_sec: float = round(
            output_tokens / (t_end - t_loaded), 2
        ) if (t_end - t_loaded) > 0 else 0.0

        print(
            f"[llm_engine]   Generated {output_tokens} tokens in "
            f"{inference_time_sec}s ({throughput_tokens_per_sec} tok/s)."
        )

        # =====================================================================
        # TELEMETRY STOP
        # Stopped BEFORE annihilation so the peak reading captures the full
        # VRAM footprint during both model load and inference.
        # =====================================================================

        peak_vram_mb: float = spy.stop()
        print(f"[llm_engine]   Peak VRAM: {peak_vram_mb:.1f} MB.")

        # =====================================================================
        # STRICT VRAM ANNIHILATION
        # del → gc.collect() → empty_cache() releases all three layers:
        #   1. Python object references (del)
        #   2. CPython cyclic garbage (gc.collect)
        #   3. CUDA allocator cache (empty_cache → returns blocks to driver)
        # =====================================================================

        print(f"[llm_engine]   Annihilating model from VRAM …")

        del model
        del tokenizer
        del inputs
        del outputs

        gc.collect()
        torch.cuda.empty_cache()

        print(f"[llm_engine]   VRAM annihilation complete.")

        # =====================================================================
        # BUILD AND PERSIST JSON RESULT
        # =====================================================================

        data: dict = JSONSchemaBuilder.build_skeleton(base_index, model_key)

        # Efficiency telemetry
        data["efficiency"]["inference_time_sec"]        = inference_time_sec
        data["efficiency"]["latency_sec"]               = latency_sec
        data["efficiency"]["throughput_tokens_per_sec"] = throughput_tokens_per_sec
        data["efficiency"]["peak_vram_mb"]              = peak_vram_mb

        # Token counts
        data["input"]["input_tokens"]  = input_tokens
        data["input"]["output_tokens"] = output_tokens

        # Generated summary (consumed by Part 5 evaluator)
        data["generated_summary"] = summary_text

        # Filename: R{N}_{model_key}.json  e.g. R1_glm4.json
        json_filename: str  = f"{base_index}_{model_key}.json"
        json_file_path: str = os.path.join(output_dir, json_filename)

        with open(json_file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        generated_results_paths[model_key] = json_file_path

        print(f"[llm_engine]   ✓ Result saved → '{json_file_path}'")

        # =====================================================================
        # SAVE STANDALONE MARKDOWN SUMMARY
        # =====================================================================
        os.makedirs(PATHS.GEN_SUMMARIES_DIR, exist_ok=True)
        summary_md_filename: str = f"{base_index}_{model_key}.md"
        summary_md_path: str = os.path.join(PATHS.GEN_SUMMARIES_DIR, summary_md_filename)

        summary_file_content: str = (
            f"# Summary: {base_index} | Model: {model_key}\n\n"
            f"{summary_text}\n"
        )

        with open(summary_md_path, "w", encoding="utf-8") as sf:
            sf.write(summary_file_content)

        print(f"[llm_engine]   ✓ Standalone summary saved → '{summary_md_path}'")

    # =========================================================================
    # RETURN RESULTS MAP
    # =========================================================================

    print(
        f"\n[llm_engine] ✓ All {total_models} models complete. "
        f"Keys: {list(generated_results_paths.keys())}"
    )

    return generated_results_paths
