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
K=3 portfolio: Llama-3.1-8B (budget) + Mistral-Large-2512 (mid) +
               Gemini-2.5-Pro (premium).

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

K3_ARM_ORDER = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]

# ==============================================================================
# Artifact Paths
# ==============================================================================

_PACKAGE_DIR = Path(__file__).parent.parent
_PACKAGE_ARTIFACTS_DIR = _PACKAGE_DIR / "data" / "artifacts"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_25.joblib"
DEFAULT_WARMUP_PRIORS_PATH = _PACKAGE_ARTIFACTS_DIR / "priors_k3_25comp.joblib"
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"

# ==============================================================================
# Canonical Data Paths
# ==============================================================================

DATA_COLLECTION_DIR = PROJECT_ROOT / "data_collection"
OFFLINE_DATASET_DIR = DATA_COLLECTION_DIR / "rewards"

# ── Model configs ─────────────────────────────────────────────────────
K4_MODELS_PATH = DATA_COLLECTION_DIR / "config" / "models_k4.json"

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

TRAIN_DATA_PATH = OFFLINE_DATASET_DIR / "train.jsonl"
VAL_DATA_PATH = OFFLINE_DATASET_DIR / "val.jsonl"
HOLDOUT_DATA_PATH = OFFLINE_DATASET_DIR / "test.jsonl"

# ── Judge robustness appendix ─────────────────────────────────────────
CALIBRATION_DIR = OFFLINE_DATASET_DIR / "calibration"
PARETO_REWARDS_PATH = (
    DATA_COLLECTION_DIR / "pareto_dataset" / "pareto_rewards.jsonl"
)

# ==============================================================================
# Warmup Priors
# ==============================================================================
#
# Built from the canonical train split using
# ``scripts/generate_multimodel_warmup_priors.py``.
# PCA-25 embeddings (28.5% variance), all-MiniLM-L6-v2 encoder.

WARMUP_PRIORS_DIR = DATA_COLLECTION_DIR / "warmup_priors"
K3_WARMUP_PRIORS_PATH = WARMUP_PRIORS_DIR / "priors_k3_25comp.joblib"

# ==============================================================================
# Best K=3 Hyperparameters (from appendix alpha sweep, 3-split protocol)
# ==============================================================================
#
# Protocol: train on train.jsonl → select on val.jsonl → report on test.jsonl.
# Per-seed Pareto AUC (10 seeds) with fixed-model endpoints and cost range
# anchored to arm cost extremes.  Avoids phantom-frontier artifacts.
# Source of truth: experiments_v2/appendix/hparam_sweep/results/best_hparams.json
# PCA fixed at d=25 (~28.5% cumulative variance) to retain a broad semantic
# representation.  A PCA ablation (Appendix I) confirms the Pareto AUC surface
# is flat across d in [6, 25], validating this design choice.

BEST_K3_HPARAMS: Dict[str, Any] = {
    "alpha": 0.01,
    "pca_components": 25,
    "prior_n_effective": 5000.0,
    "policy": "disjoint",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Best K=3 BanditGPT config (warmup priors, PCA-25, disjoint LinUCB).

Val Pareto AUC = 0.9295, test Pareto AUC = 0.9289 (Δ=+0.37% vs fixed).
Monotonically decreasing in alpha: strong warmup priors (n_eff=5000) encode
reliable initial beliefs, so near-zero exploration suffices.  The top three
alphas (0.01, 0.05, 0.10) are within 0.07% AUC, indicating robustness.
"""

BEST_K3_TABULA_RASA_HPARAMS: Dict[str, Any] = {
    "alpha": 0.25,
    "pca_components": 25,
    "prior_n_effective": 1.0,
    "policy": "disjoint",
    "use_corralling": False,
    "forgetting_factor": 1.0,
}
"""Best K=3 Tabula Rasa config (cold start, PCA-25, no priors).

Val Pareto AUC = 0.9277, test Pareto AUC = 0.9268 (Δ=+0.14% vs fixed).
Requires 25x more exploration (alpha=0.25 vs 0.01) than BanditGPT to
compensate for lack of warmup priors.  After the train-phase (8K prompts),
moderate exploration outperforms both conservative and aggressive strategies.
"""
