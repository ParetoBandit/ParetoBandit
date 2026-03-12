"""
Configuration Constants for BanditGPT

Centralized configuration for all data paths, model definitions, tuned
hyperparameters, and immutable artifacts used across the project.

Canonical data layout
---------------------
All experimental data lives under ``data_collection/``.  The canonical files
are the *only* authoritative copies; everything else is in ``archive/``.

Models
~~~~~~
K=2 benchmark portfolio : Llama-3.1-8B (weak) + Gemini-2.5-Pro (strong).
K=3 onboarding target   : K=2 + Mistral-Large-2512 (mid, added via
                           ``register_model()``).

Judges
~~~~~~
All rewards scored by a fixed unbiased PoLL panel:
    DeepSeek-R1, GPT-4.1-mini, Claude-3.5-Haiku.
Continuous v3 rubric (logic x constraint x utility).
"""

from pathlib import Path
from typing import Any, Dict

# ==============================================================================
# Model Configuration
# ==============================================================================

DEFAULT_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"

K2_ARM_ORDER = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-pro",
]
K3_ARM_ORDER = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]

STRONG_MODEL_EQUIVALENTS = ["openai/gpt-4.1", "openai/gpt-4.1"]

# ==============================================================================
# Artifact Paths
# ==============================================================================

_PACKAGE_DIR = Path(__file__).parent.parent
_PACKAGE_DATA_DIR = _PACKAGE_DIR / "data"
_PACKAGE_ARTIFACTS_DIR = _PACKAGE_DATA_DIR / "artifacts"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_25.joblib"
FULL_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_32.joblib"
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"

# ==============================================================================
# Canonical Data Paths
# ==============================================================================

DATA_COLLECTION_DIR = PROJECT_ROOT / "data_collection"
OFFLINE_DATASET_DIR = DATA_COLLECTION_DIR / "rewards"
PROMPTS_DIR = DATA_COLLECTION_DIR / "prompts"
CACHE_DIR = DATA_COLLECTION_DIR / "cache"

# ── Model configs ─────────────────────────────────────────────────────
K3_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k3.json"
K4_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k4.json"

# ── Source prompt data ────────────────────────────────────────────────
LMSYS_BATTLES_PATH = PROMPTS_DIR / "lmarena_battles_en.jsonl"

# ==============================================================================
# Canonical Reward Data  (K=3 benchmark: 11,983 prompts)
# ==============================================================================
#
# K=3 portfolio: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Pro.
# Each row is a single prompt with per-arm rewards and costs for all 3 models.
# Every record judged by the canonical PoLL panel (R1 + GPT-4.1-mini + Haiku).
# Continuous v3 rubric (logic x constraint x utility).
#
#   CONSTANT              PROMPTS   FILE
#   ──────────────────    ───────   ──────────────────────
#   TRAIN_DATA_PATH         8,374   train.jsonl
#   VAL_DATA_PATH           1,785   val.jsonl
#   HOLDOUT_DATA_PATH       1,824   test.jsonl
#   ──────────────────    ───────
#   TOTAL                  11,983
#
# Split: stratified by difficulty, seed=42, ratio 70/15/15.
# Non-overlapping by prompt — no prompt appears in more than one split.
# K=2 experiments use the same files, ignoring the Mistral arm.

TRAIN_DATA_PATH = OFFLINE_DATASET_DIR / "train.jsonl"
VAL_DATA_PATH = OFFLINE_DATASET_DIR / "val.jsonl"
HOLDOUT_DATA_PATH = OFFLINE_DATASET_DIR / "test.jsonl"

# ==============================================================================
# Warmup Priors
# ==============================================================================
#
# Built from the canonical train split using
# ``scripts/generate_multimodel_warmup_priors.py``.
# PCA-25 embeddings (28.5% variance), all-MiniLM-L6-v2 encoder.
# Upgraded from PCA-20 after full hparam sweep confirmed d=25 maximises
# test Pareto AUC (+0.841%) with hybrid policy.

WARMUP_PRIORS_DIR = DATA_COLLECTION_DIR / "warmup_priors"
K2_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_k2_25comp.joblib"
K2_WARMUP_PRIORS_TEXTFEAT_PATH = WARMUP_PRIORS_DIR / "priors_k2_textfeat.joblib"

# ==============================================================================
# Pre-computed Embeddings
# ==============================================================================

EMBEDDINGS_CACHE_DIR = DATA_COLLECTION_DIR / "embeddings"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings_pca25.npz"
RAW_EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "raw_embeddings.npz"

# ==============================================================================
# Calibrated Router & Model Registry
# ==============================================================================

BANDIT_DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"
CANONICAL_CALIBRATED_ROUTER_PATH = (
    BANDIT_DATA_DIR / "artifacts" / "canonical_router_calibrated.joblib"
)
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"

# ==============================================================================
# Best K=2 Hyperparameters (from benchmark sweep on val, 160 configs x 3 seeds)
# ==============================================================================
#
# Selected by Pareto AUC on val.  Test seeds independent (offset 1000+).
# See experiments/benchmark/results/hparam_tuning_k2.json for full sweep.

BEST_K2_HPARAMS: Dict[str, Any] = {
    "alpha": 1.00,
    "prior_n_effective": 10.0,
    "policy": "hybrid",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Best hybrid K=2 config (PCA-25, ``corralling=False``).

Val AUC = 0.8686 (+0.818%), Test AUC = 0.8699 (+0.841%).
CostSave@95% = 40.7%, CostSave@99% = 14.3% on test.
Used for Exp 1 (headline CostSave), Exp 2 (learning curve), Exp 5 (features).
"""

BEST_K2_CORRALLING_HPARAMS: Dict[str, Any] = {
    "alpha": 0.05,
    "prior_n_effective": 50.0,
    "policy": "hybrid",
    "use_corralling": True,
    "forgetting_factor": 1.0,
}
"""Best K=2 config with ``corralling=True`` (PCA-only).

Val AUC = 0.8665 (+0.580%).
Used for Exp 3 (onboarding), Exp 4 (Corralling ablation), Exp 6 (warmup).
Corralling pays a small overhead under good priors but provides safety
against prior misspecification (see Exp 4).
"""

# Placeholder — to be filled after tabula_rasa sweep completes.
BEST_K2_TABULA_RASA_HPARAMS: Dict[str, Any] = {
    "alpha": 0.30,
    "prior_n_effective": 1.0,
    "policy": "tabula_rasa",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Best K=2 config for pure tabula rasa (``priors='none'``).

Val AUC = 0.8614 (from hparam_tuning_k2.json).
Notably uses alpha=0.3 — lower than the warmup config's alpha=1.0,
because without priors the reward estimates are noisier and a smaller
exploration bonus avoids over-exploration early on.
Used as a baseline in Exp 6 (warmup ablation) to show the value of priors.
"""

# ==============================================================================
# Legacy Aliases (for backward compatibility with older experiment scripts)
# ==============================================================================

WARMUP_PRIORS_PATH = K2_WARMUP_PRIORS_PATH
REWARDS_PATH = OFFLINE_DATASET_DIR / "archive" / "k4_canonical" / "rewards.jsonl"
CANONICAL_PROMPTS_PATH = PROMPTS_DIR / "archive" / "prompts.jsonl"

TRAIN_DATA_PATH_ALL_MODELS = (
    OFFLINE_DATASET_DIR / "archive" / "legacy_k10"
    / "train_rewards_complete_all_models.jsonl.gz"
)
VAL_DATA_PATH_ALL_MODELS = (
    OFFLINE_DATASET_DIR / "archive" / "legacy_k10"
    / "val_rewards_complete_all_models.jsonl.gz"
)
HOLDOUT_DATA_PATH_ALL_MODELS = (
    OFFLINE_DATASET_DIR / "archive" / "legacy_k10"
    / "holdout_rewards_complete_all_models.jsonl.gz"
)
DEV_DATA_PATH_ALL_MODELS = (
    OFFLINE_DATASET_DIR / "archive" / "legacy_k10"
    / "dev_rewards_complete_all_models.jsonl.gz"
)
