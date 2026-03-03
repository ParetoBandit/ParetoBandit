"""
Configuration Constants for BanditGPT

This package contains all configuration constants, parameters, and immutable data
files (model registries) used across the project. Centralized constant management
for better maintainability.

Data files:
    models.json       — Production model registry (default 2-model portfolio)
    models_all.json   — Extended model registry (43+ models for experiments)
"""

from pathlib import Path

# ==============================================================================
# Model Configuration
# ==============================================================================

# SentenceTransformer model used for semantic embeddings throughout the project.
#
# Rationale:
# - `all-MiniLM-L6-v2` is an excellent speed baseline, but it is no longer
#   state-of-the-art on embedding quality.
# - `BAAI/bge-m3` is a strong modern embedding model (multilingual + strong
#   retrieval / semantic matching), which improves routing signal quality.
#
# IMPORTANT: The shipped PCA artifact (`pca_32.joblib`) is trained specifically
# for this encoder. If you change this default, you must regenerate PCA (and any
# warmup priors) using `bandit_gpt.calibration.train_pca()` and
# `bandit_gpt.calibration.generate_warmup_priors()`.
DEFAULT_SENTENCE_TRANSFORMER = "BAAI/bge-m3"

# Model tier mapping for capability-equivalent substitutions
# Used when a model is no longer available but has a capability-tier equivalent
# e.g., gpt-4-turbo → gpt-4o (both are strong models in the same capability tier)
STRONG_MODEL_EQUIVALENTS = ["openai/gpt-4-turbo", "openai/gpt-4-turbo"]

# ==============================================================================
# Artifact Paths
# ==============================================================================

# Package-internal paths (resolve correctly both in dev and after pip install)
_PACKAGE_DIR = Path(__file__).parent.parent
_PACKAGE_DATA_DIR = _PACKAGE_DIR / "data"
_PACKAGE_ARTIFACTS_DIR = _PACKAGE_DATA_DIR / "artifacts"

# Source-tree paths (used only by experiment scripts outside the package)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# PCA model (32 components) trained on RouteLLM battle data using
# DEFAULT_SENTENCE_TRANSFORMER.
#
# Trained on 80K RouteLLM battles (independent dataset from dev/holdout — no contamination).
# Shipped inside the wheel so first-time users skip JIT retraining.
DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_32.joblib"

# Generic PCA: trained on C4 web text (no routing connection).
# Provides unbiased baseline for routing signal analysis.
# Generate with: python3 scripts/train_pca_generic.py --n-components 32
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"

# Path to warmup priors trained on RouteLLM data (K=2: Mixtral + GPT-4-Turbo)
DEFAULT_WARMUP_PRIORS_PATH = ARTIFACTS_DIR / "priors_warmup.joblib"

# Path to warmup priors trained on 43-model evaluation data (K>2 experiments)
MULTIMODEL_WARMUP_PRIORS_PATH = ARTIFACTS_DIR / "priors_warmup_43model.joblib"

# Optional: K=2 warmup priors extracted from the multi-model artifact.
# Useful when you want a single warmup source across portfolios or to avoid
# dependence on external corpora for K=2.
K2_WARMUP_FROM_MULTIMODEL_PATH = ARTIFACTS_DIR / "priors_warmup_k2_from_43model.joblib"

# Three-way split definition for K>2 experiments (prior-train / online-learn / holdout)
THREE_WAY_SPLITS_PATH = ARTIFACTS_DIR / "splits_three_way.json"

# Canonical offline dataset paths
OFFLINE_DATASET_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data" / "offline_dataset"

# 2-model datasets (Mixtral + GPT-4-Turbo only - the models the router chooses between)
CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_2models.jsonl.gz"
CANONICAL_HOLDOUT_DATA_PATH = OFFLINE_DATASET_DIR / "holdout_rewards_2models.jsonl.gz"

# 3-model datasets (includes GPT-4o for reference/analysis - NOT for routing)
DEV_DATA_PATH_3MODELS = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"
HOLDOUT_DATA_PATH_3MODELS = OFFLINE_DATASET_DIR / "holdout_rewards_complete.jsonl.gz"

# All models datasets (includes all available models from LMSys Arena)
DEV_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "dev_rewards_complete_all_models.jsonl.gz"
HOLDOUT_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "holdout_rewards_complete_all_models.jsonl.gz"

# RouteLLM battles rewards dataset (corrected winner labels)
ROUTELLM_BATTLES_REWARDS_PATH = OFFLINE_DATASET_DIR / "routellm_battles_rewards.jsonl"

# Calibrated router path
BANDIT_DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"
CANONICAL_CALIBRATED_ROUTER_PATH = BANDIT_DATA_DIR / "artifacts" / "canonical_router_calibrated.joblib"

# Model registry path
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"
