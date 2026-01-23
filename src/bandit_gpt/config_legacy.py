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

# Path to the PCA model (23 components) trained on RouteLLM data
DEFAULT_PCA_PATH = ARTIFACTS_DIR / "pca_23.joblib"

# Path to warmup priors trained on RouteLLM data
DEFAULT_WARMUP_PRIORS_PATH = DATA_DIR / "routellm" / "artifacts" / "priors_warmup_routellm_pca24.joblib"

