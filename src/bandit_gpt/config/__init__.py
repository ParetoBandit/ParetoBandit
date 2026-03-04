"""
Configuration Constants for BanditGPT

This package contains all configuration constants, parameters, and immutable data
files (model registries) used across the project. Centralized constant management
for better maintainability.

Data files:
    models.json       — Consolidated model registry (85+ models with pricing)
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
STRONG_MODEL_EQUIVALENTS = ["openai/gpt-4.1", "openai/gpt-4.1"]

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

# PCA model trained on offline battle data using DEFAULT_SENTENCE_TRANSFORMER.
#
# Trained on 80K offline battles (independent dataset from dev/holdout — no contamination).
# Shipped inside the wheel so first-time users skip JIT retraining.
#
# 6 components capture ~13.8% of variance — sufficient for routing signal
# while maintaining a healthy samples-per-feature ratio for both K=2 and K=10.
# The full 32-component artifact is retained for reference / ablations.
DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_6.joblib"
FULL_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_32.joblib"

# Generic PCA: trained on C4 web text (no routing connection).
# Provides unbiased baseline for routing signal analysis.
# Generate with: python3 scripts/train_pca_generic.py --n-components 32
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"

# Path to warmup priors trained on offline battle data (K=2: Mixtral + GPT-4-Turbo)
DEFAULT_WARMUP_PRIORS_PATH = ARTIFACTS_DIR / "priors_warmup.joblib"

# Path to warmup priors trained on 43-model evaluation data (K>2 experiments)
# 6-component variants are the default; 32-comp retained for ablation reference.
MULTIMODEL_WARMUP_PRIORS_PATH = ARTIFACTS_DIR / "priors_warmup_43model_6comp.joblib"
MULTIMODEL_WARMUP_PRIORS_PATH_32 = ARTIFACTS_DIR / "priors_warmup_43model.joblib"

# K=2 warmup priors extracted from the multi-model artifact.
# Single warmup source across portfolios — avoids dependence on external corpora.
K2_WARMUP_FROM_MULTIMODEL_PATH = ARTIFACTS_DIR / "priors_warmup_k2_from_43model_6comp.joblib"
K2_WARMUP_FROM_MULTIMODEL_PATH_32 = ARTIFACTS_DIR / "priors_warmup_k2_from_43model.joblib"

# Three-way split definition for K>2 experiments (prior-train / online-learn / holdout)
THREE_WAY_SPLITS_PATH = ARTIFACTS_DIR / "splits_three_way.json"

# Canonical data paths — all reward data lives in data_collection/
DATA_COLLECTION_DIR = PROJECT_ROOT / "data_collection"
OFFLINE_DATASET_DIR = DATA_COLLECTION_DIR / "rewards"
PROMPTS_DIR = DATA_COLLECTION_DIR / "prompts"
CACHE_DIR = DATA_COLLECTION_DIR / "cache"
LMSYS_BATTLES_PATH = PROMPTS_DIR / "lmarena_battles_en.jsonl"
K10_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k10.json"

# All models datasets (44 models: original 43 + gemini-2.5-flash)
DEV_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "dev_rewards_complete_all_models.jsonl.gz"
HOLDOUT_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "holdout_rewards_complete_all_models.jsonl.gz"

# Legacy aliases — subset files removed; experiments filter by their own model list
CANONICAL_DEV_DATA_PATH = DEV_DATA_PATH_ALL_MODELS
CANONICAL_HOLDOUT_DATA_PATH = HOLDOUT_DATA_PATH_ALL_MODELS
DEV_DATA_PATH_3MODELS = DEV_DATA_PATH_ALL_MODELS
HOLDOUT_DATA_PATH_3MODELS = HOLDOUT_DATA_PATH_ALL_MODELS

# Offline battles rewards dataset (corrected winner labels)
ROUTELLM_BATTLES_REWARDS_PATH = OFFLINE_DATASET_DIR / "routellm_battles_rewards.jsonl"

# Calibrated router path
BANDIT_DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"
CANONICAL_CALIBRATED_ROUTER_PATH = BANDIT_DATA_DIR / "artifacts" / "canonical_router_calibrated.joblib"

# Model registry path
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"
