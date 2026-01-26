"""
Configuration Constants for BanditGPT

This module contains all configuration constants and parameters used across the project.
Centralized constant management for better maintainability.
"""

from pathlib import Path

# ==============================================================================
# Model Configuration
# ==============================================================================

# Sentence Transformer model used for semantic embeddings throughout the project
DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"

# Model tier mapping for capability-equivalent substitutions
# Used when a model is no longer available but has a capability-tier equivalent
# e.g., gpt-4-turbo → gpt-4o (both are strong models in the same capability tier)
STRONG_MODEL_EQUIVALENTS = ["openai/gpt-4-turbo", "openai/gpt-4o"]

# ==============================================================================
# Artifact Paths
# ==============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "src" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# Path to the PCA model (32 components) trained on RouteLLM data
# 32 components capture 35.14% variance vs 29.01% for 23 components (+6.14% improvement)
DEFAULT_PCA_PATH = ARTIFACTS_DIR / "pca_32.joblib"

# Path to warmup priors trained on RouteLLM data
DEFAULT_WARMUP_PRIORS_PATH = ARTIFACTS_DIR / "priors_warmup.joblib"

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
DEFAULT_MODEL_REGISTRY_PATH = PROJECT_ROOT / "src" / "bandit_gpt" / "config" / "models.json"

