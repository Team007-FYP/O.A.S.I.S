# =============================================================================
# config.py
# Configuration file for the 40GB A100 Multimodal Summarization Pipeline.
# Covers: I/O paths, model registry, inference hyperparameters, system prompts.
# =============================================================================

import os                          # Used for path utilities and env variables
import torch                       # Used for specifying compute dtype
from dataclasses import dataclass  # Used to create structured, typed config classes
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 1. PATHS CONFIGURATION
# Centralises all directory paths used across the pipeline so that changing
# one value here propagates everywhere without touching pipeline code.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)           # frozen=True prevents accidental mutation at runtime
class Paths:
    # Root dataset directory — contains three named sub-folders.
    DATASET_DIR: str = "./dataset"

    # Sub-folder holding audio recordings:
    #   A1.wav, A2.wav, … -> meeting audio files
    AUDIOS_DIR: str = "./dataset/Audios"

    # Sub-folder holding PDF slide decks:
    #   P1.pdf, P2.pdf, … -> corresponding presentation slides
    SLIDES_DIR: str = "./dataset/Slides"

    # Sub-folder holding ground-truth reference summaries:
    #   S1.txt, S2.txt, … -> human-authored reference summaries
    SUMMARIES_DIR: str = "./dataset/Summaries"

    # Directory for Whisper / ASR transcription outputs:
    #   T1.txt  -> plain-text transcript produced from A{N}.wav
    TRANSCRIPTS_DIR: str = "./outputs/transcripts"

    # Directory for OCR model outputs:
    #   P1.md   -> Markdown rendering of slide text extracted from P{N}.pdf
    OCR_DIR: str = "./outputs/ocr"

    # Directory for fused multimodal context documents:
    #   C1.txt  -> merged transcript + slide context ready for summarisation
    CONTEXT_DIR: str = "./outputs/contexts"

    # Directory for per-model evaluation result files:
    #   R1_glm4.json, R1_llama.json, etc. -> ROUGE / BERTScore metrics per model
    EVALUATIONS_DIR: str = "./outputs/evaluations"

    # Directory for standalone generated LLM summaries (.md):
    #   R1_glm4.md, R1_llama.md, R1_mistral.md, R1_qwen.md
    GEN_SUMMARIES_DIR: str = "./outputs/gen_summaries"


# Singleton instance — import this directly throughout the codebase
PATHS: Paths = Paths()


# -----------------------------------------------------------------------------
# 2. MODEL REGISTRY
# Stores HuggingFace Hub identifiers so model references are never hard-coded
# in pipeline scripts.  Add or swap model variants here without touching logic.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    # Dictionary mapping short pipeline keys -> HuggingFace model IDs.
    # Keys are used as filename suffixes (e.g. R1_glm4.json) and CLI flags.
    LLM_MODELS: dict[str, str] = None  # assigned via __post_init__ to allow mutable default

    # Dedicated OCR model: olmOCR-2 is optimised for document-layout-aware
    # text extraction and produces structured Markdown suitable for context fusion.
    OCR_MODEL: str = "allenai/olmOCR-2-7B-1025"

    # Judge / evaluator model used to score generated summaries against ground truth.
    # Llama-3.1-8B-Instruct provides a strong open-weight auto-evaluator baseline.
    JUDGE_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __post_init__(self) -> None:
        # Bypass frozen restriction to set the mutable dict default safely
        object.__setattr__(
            self,
            "LLM_MODELS",
            {
                # GLM-4 9B chat variant fine-tuned for instruction following
                "glm4":    "THUDM/glm-4-9b-chat-hf",

                # Meta Llama 3.1 8B — instruction-tuned; also doubles as JUDGE_MODEL
                "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",

                # Mistral 7B v0.3 — strong multilingual instruction-following model
                "mistral": "mistralai/Mistral-7B-Instruct-v0.3",

                # Qwen 2.5 7B — Alibaba's latest instruction model with long context
                "qwen":    "Qwen/Qwen2.5-7B-Instruct",
            }
        )


# Singleton instance
MODEL_CONFIG: ModelConfig = ModelConfig()


# -----------------------------------------------------------------------------
# 3. INFERENCE HYPERPARAMETERS
# Controls decoding behaviour during generation.  Keeping TEMPERATURE = 0.0
# enforces greedy decoding, which is mandatory for reproducible benchmarking:
# the same prompt will always produce the same output across runs.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceConfig:
    # Greedy decoding — MUST remain 0.0 for deterministic benchmark results.
    # Any value > 0.0 introduces stochastic sampling, breaking reproducibility.
    TEMPERATURE: float = 0.0

    # Maximum number of new tokens the model is allowed to generate per call.
    # 1024 balances summary completeness against A100 VRAM and latency budgets.
    MAX_NEW_TOKENS: int = 1024

    # bfloat16 is the recommended dtype for A100 GPUs:
    #   - Same dynamic range as float32 (8 exponent bits) -> numerically stable
    #   - Half the memory footprint of float32 -> fits larger models / longer contexts
    #   - Natively accelerated by A100 Tensor Cores for maximum throughput
    DTYPE: torch.dtype = torch.bfloat16


# Singleton instance
INFERENCE_CONFIG: InferenceConfig = InferenceConfig()


# -----------------------------------------------------------------------------
# 4. SYSTEM PROMPTS
# Defines the instruction strings injected into the LLM system role.
# Keeping prompts in config (not scattered in scripts) allows prompt versioning
# and A/B testing without touching pipeline logic.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompts:
    # Prompt for Stage 3 — Multimodal Summarisation.
    # The model reads C1.txt (the fused context) and produces a structured summary.
    # "Based strictly on the provided context" discourages hallucination from
    # parametric knowledge not grounded in the actual meeting content.
    SUMMARIZATION_PROMPT: str = (
        "You are an expert meeting analyst. "
        "Based strictly on the provided multimodal context (audio transcript + slide text), "
        "write a comprehensive summary."
    )


# Singleton instance
PROMPTS: Prompts = Prompts()

# -----------------------------------------------------------------------------
# 5. SENTRY OBSERVABILITY & BENCHMARK TELEMETRY
# Configures Sentry Releases, Performance Tracing, and Metric Aggregation.
# Opt-in design: If SENTRY_DSN is empty, the pipeline runs 100% offline.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SentryConfig:
    # Sentry DSN: set here or pass via the SENTRY_DSN environment variable
    DSN: str = os.getenv("SENTRY_DSN", "")
    
    # Sentry Release: groups this entire 30-meeting benchmark cycle
    RELEASE: str = "oasis@ami-30-meetings"
    
    # Environment tag
    ENVIRONMENT: str = "research-benchmarking"
    
    # Trace 100% of meeting pipeline transactions for performance profiling
    TRACES_SAMPLE_RATE: float = 1.0
    
    @property
    def is_enabled(self) -> bool:
        """Returns True only if a valid DSN is provided."""
        return bool(self.DSN and self.DSN.strip())


# Singleton instance
SENTRY_CONFIG: SentryConfig = SentryConfig()

# -----------------------------------------------------------------------------
# 6. RUNTIME SANITY CHECKS
# Executed once on import to surface misconfiguration early, before any model
# is loaded or GPU memory is allocated.
# -----------------------------------------------------------------------------

def _validate_config() -> None:
    """Raise immediately if any configuration invariant is violated."""

    # Enforce deterministic benchmarking constraint
    assert INFERENCE_CONFIG.TEMPERATURE == 0.0, (
        f"TEMPERATURE must be 0.0 for deterministic benchmarking, "
        f"got {INFERENCE_CONFIG.TEMPERATURE}"
    )

    # Ensure every registered LLM key maps to a non-empty model ID string
    for key, model_id in MODEL_CONFIG.LLM_MODELS.items():
        assert isinstance(model_id, str) and model_id.strip(), (
            f"LLM_MODELS['{key}'] must be a non-empty string, got: {model_id!r}"
        )

    # Warn if running on CPU — the pipeline is designed exclusively for A100 GPU
    if not torch.cuda.is_available():
        import warnings
        warnings.warn(
            "CUDA is not available. This pipeline is optimised for a 40GB A100 GPU. "
            "CPU execution will be extremely slow and may run out of memory.",
            RuntimeWarning,
            stacklevel=2,
        )


# Run validation at import time
_validate_config()
