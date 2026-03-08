"""
Configuration Constants for BanditGPT

Centralized configuration for all data paths, model definitions, and immutable
artifacts used across the project.

Canonical data layout
---------------------
All experimental data lives under ``data_collection/``.  The canonical files
are the *only* authoritative copies; everything else is in ``archive/``.

Models
~~~~~~
K=3 bandit portfolio : Llama-3.1-8B (budget), Mistral-Large-2512 (mid),
                       Gemini-2.5-Pro (premium).
K=4 full portfolio   : K=3 + Gemini-2.5-Flash (positive-transfer candidate).

Judges
~~~~~~
All rewards scored by a fixed unbiased PoLL panel:
    DeepSeek-R1, Qwen-2.5-72B-Instruct, Claude-3.5-Haiku.
Continuous v3 rubric (logic × constraint × utility).
"""

from pathlib import Path

# ==============================================================================
# Model Configuration
# ==============================================================================

# IMPORTANT: The shipped PCA artifact is trained for this encoder.
# Changing it requires regenerating PCA and warmup priors.
DEFAULT_SENTENCE_TRANSFORMER = "BAAI/bge-m3"

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

DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_15.joblib"
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

# ── Canonical prompt set (4,162 diverse prompts from LMSYS arena) ─────
CANONICAL_PROMPTS_PATH = PROMPTS_DIR / "prompts.jsonl"
LMSYS_BATTLES_PATH = PROMPTS_DIR / "lmarena_battles_en.jsonl"

# ==============================================================================
# Canonical Reward Data  (4,009 prompts × 4 models = 16,036 records)
# ==============================================================================
#
# K=4 portfolio: Llama-3.1-8B, Mistral-Large-2512, Gemini-2.5-Flash,
#                Gemini-2.5-Pro.
# Every record judged by the canonical PoLL panel (R1 + Qwen-72B + Haiku).
# Perfectly balanced: every prompt has exactly 4 model records.
#
#   CONSTANT              PROMPTS   RECORDS   FILE
#   ──────────────────    ───────   ───────   ──────────────────────
#   REWARDS_PATH            4,009    16,036   rewards.jsonl
#   TRAIN_DATA_PATH         1,002     4,008   train.jsonl.gz
#   VAL_DATA_PATH           1,503     6,012   val.jsonl.gz
#   HOLDOUT_DATA_PATH       1,504     6,016   holdout.jsonl.gz
#   ──────────────────    ───────   ───────
#   TOTAL                   4,009    16,036
#
# Split seed: np.random.RandomState(42), ratio 25/37.5/37.5.
# Non-overlapping by prompt — no prompt appears in more than one split.

REWARDS_PATH    = OFFLINE_DATASET_DIR / "rewards.jsonl"
TRAIN_DATA_PATH = OFFLINE_DATASET_DIR / "train.jsonl.gz"
VAL_DATA_PATH   = OFFLINE_DATASET_DIR / "val.jsonl.gz"
HOLDOUT_DATA_PATH = OFFLINE_DATASET_DIR / "holdout.jsonl.gz"

# ==============================================================================
# Warmup Priors  (K=3 bandit models, 32-component whitened PCA)
# ==============================================================================
#
# Built from the canonical train split (1,002 prompts) using
# ``scripts/generate_multimodel_warmup_priors.py --no-split``.
# Plasticity = 0.1, PCA-whitened, BAAI/bge-m3 encoder.

WARMUP_PRIORS_DIR = DATA_COLLECTION_DIR / "warmup_priors"
WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_k3.joblib"

# ==============================================================================
# Pre-computed Embeddings
# ==============================================================================

EMBEDDINGS_CACHE_DIR = DATA_COLLECTION_DIR / "embeddings"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings_pca15.npz"
RAW_EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "raw_embeddings.npz"

# ==============================================================================
# Calibrated Router & Model Registry
# ==============================================================================

BANDIT_DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"
CANONICAL_CALIBRATED_ROUTER_PATH = BANDIT_DATA_DIR / "artifacts" / "canonical_router_calibrated.joblib"
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"

# ==============================================================================
# Legacy Aliases (for backward compatibility with older experiment scripts)
# ==============================================================================

TRAIN_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "archive" / "legacy_k10" / "train_rewards_complete_all_models.jsonl.gz"
VAL_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "archive" / "legacy_k10" / "val_rewards_complete_all_models.jsonl.gz"
HOLDOUT_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "archive" / "legacy_k10" / "holdout_rewards_complete_all_models.jsonl.gz"
DEV_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "archive" / "legacy_k10" / "dev_rewards_complete_all_models.jsonl.gz"
K4_REWARDS_PATH = REWARDS_PATH
K4_TRAIN_DATA_PATH = TRAIN_DATA_PATH
K4_VAL_DATA_PATH = VAL_DATA_PATH
K4_HOLDOUT_DATA_PATH = HOLDOUT_DATA_PATH
