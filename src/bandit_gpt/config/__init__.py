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

# PCA model trained on LMSYS Chatbot Arena prompts using DEFAULT_SENTENCE_TRANSFORMER.
#
# Trained on ~46K LMSYS arena prompts (strictly disjoint from dev/holdout
# experimental data).  Shipped inside the wheel so first-time users skip
# JIT retraining.
#
# 15 components capture ~21% of variance — balances routing signal
# richness with a healthy samples-per-feature ratio for both K=2 and K=10.
# The full 32-component artifact is retained for reference / ablations.
DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_15.joblib"
FULL_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_32.joblib"

# Generic PCA: trained on C4 web text (no routing connection).
# Provides unbiased baseline for routing signal analysis.
# Generate with: python3 scripts/train_pca_generic.py --n-components 32
GENERIC_PCA_PATH = ARTIFACTS_DIR / "pca_32_generic.joblib"

# ==============================================================================
# Canonical Data Layout  (~4,071 prompts total, 44 models each)
# ==============================================================================
#
# Three non-overlapping reward files — one per split, one row per (model, prompt):
#
#   CONSTANT                      PROMPTS   FILE
#   ──────────────────────────    ───────   ──────────────────────────────────────────────────
#   TRAIN_DATA_PATH_ALL_MODELS    1,028     train_rewards_complete_all_models.jsonl.gz
#   VAL_DATA_PATH_ALL_MODELS      1,543     val_rewards_complete_all_models.jsonl.gz
#   HOLDOUT_DATA_PATH_ALL_MODELS  1,500     holdout_rewards_complete_all_models.jsonl.gz
#   ──────────────────────────    ───────
#   TOTAL                         4,071
#
# Each file is self-contained — no secondary split definition is required.
# DEV_DATA_PATH_ALL_MODELS is retained as a convenience alias for the union
# of train + val (2,854 prompts) when split membership does not matter.

# Canonical data paths — all reward data lives in data_collection/
DATA_COLLECTION_DIR = PROJECT_ROOT / "data_collection"

OFFLINE_DATASET_DIR = DATA_COLLECTION_DIR / "rewards"
PROMPTS_DIR = DATA_COLLECTION_DIR / "prompts"
CACHE_DIR = DATA_COLLECTION_DIR / "cache"
LMSYS_BATTLES_PATH = PROMPTS_DIR / "lmarena_battles_en.jsonl"
K2_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k2.json"
K3_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k3.json"
K4_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k4.json"
K5_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k5.json"
K10_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k10.json"

# Primary per-split reward files (legacy 44-model portfolio)
TRAIN_DATA_PATH_ALL_MODELS    = OFFLINE_DATASET_DIR / "train_rewards_complete_all_models.jsonl.gz"
VAL_DATA_PATH_ALL_MODELS      = OFFLINE_DATASET_DIR / "val_rewards_complete_all_models.jsonl.gz"
HOLDOUT_DATA_PATH_ALL_MODELS  = OFFLINE_DATASET_DIR / "holdout_rewards_complete_all_models.jsonl.gz"

# Convenience alias: train + val combined (2,854 prompts).
# Use when split membership is irrelevant (e.g. embedding pre-computation).
DEV_DATA_PATH_ALL_MODELS = OFFLINE_DATASET_DIR / "dev_rewards_complete_all_models.jsonl.gz"

# ==============================================================================
# K=4 Data Layout  (4,133 prompts, 4 models: Llama-3.1-8B, Gemini-2.5-Flash,
#                    GPT-4.1, GPT-4.1-Mini)
# ==============================================================================
#
# CoT-rubric judged rewards (v2) with same-provider judge exclusion.
# Perfectly balanced: every prompt has exactly 4 model records.
#
#   CONSTANT                   PROMPTS   FILE
#   ────────────────────────   ───────   ──────────────────────────────
#   K4_TRAIN_DATA_PATH         1,033     k4_train_rewards.jsonl.gz
#   K4_CAL_DATA_PATH           1,549     k4_cal_rewards.jsonl.gz
#   K4_HOLDOUT_DATA_PATH       1,551     k4_holdout_rewards.jsonl.gz
#   ────────────────────────   ───────
#   TOTAL                      4,133
#
# Split seed: np.random.RandomState(42), ratio 25/37.5/37.5.

K4_TRAIN_DATA_PATH   = OFFLINE_DATASET_DIR / "k4_train_rewards.jsonl.gz"
K4_CAL_DATA_PATH     = OFFLINE_DATASET_DIR / "k4_cal_rewards.jsonl.gz"
K4_HOLDOUT_DATA_PATH = OFFLINE_DATASET_DIR / "k4_holdout_rewards.jsonl.gz"
K4_DEV_DATA_PATH     = OFFLINE_DATASET_DIR / "k4_dev_rewards.jsonl.gz"

# Portfolio-specific warmup priors.
#
# 15-component priors (legacy, built from the 44-model dev pool):
WARMUP_PRIORS_DIR = DATA_COLLECTION_DIR / "warmup_priors"
K2_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k2_15comp.joblib"
K3_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k3_15comp.joblib"
K5_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k5_15comp.joblib"
K10_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k10_15comp.joblib"
MULTIMODEL_WARMUP_PRIORS_PATH = K10_WARMUP_PRIORS_PATH
#
# 32-component priors (built from K=4 canonical train split for ablations).
# K=2 reuses the K=3 file — the router auto-selects only the 2 models it needs.
K2_WARMUP_PRIORS_32_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k3_32comp.joblib"
K3_WARMUP_PRIORS_32_PATH = WARMUP_PRIORS_DIR / "priors_warmup_k3_32comp.joblib"

# Pre-computed embeddings cache (generated by scripts/precompute_embeddings.py).
EMBEDDINGS_CACHE_DIR = DATA_COLLECTION_DIR / "embeddings"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "embeddings_pca15.npz"
RAW_EMBEDDINGS_CACHE_PATH = EMBEDDINGS_CACHE_DIR / "raw_embeddings.npz"



# Calibrated router path
BANDIT_DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"
CANONICAL_CALIBRATED_ROUTER_PATH = BANDIT_DATA_DIR / "artifacts" / "canonical_router_calibrated.joblib"

# Model registry path
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"
