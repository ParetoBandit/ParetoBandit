"""
Configuration Constants for ParetoBandit

Centralized configuration for all data paths, model definitions, tuned
hyperparameters, and immutable artifacts used across the project.

Canonical data layout
---------------------
All experimental data lives under ``data_collection/``.  The canonical files
are the *only* authoritative copies; everything else is in ``archive/``.

Models
~~~~~~
K=3 portfolio: Llama-3.1-8B (budget) + Mistral-Large-2512 (mid-cost) +
               Gemini-2.5-Pro (high-cost).

Judges
~~~~~~
All experiment rewards scored by DeepSeek-R1 (sole evaluator).
Continuous v3 rubric: Reasoning Quality 40%, Instruction Following 30%,
Communication Quality 30%.  Cross-judge robustness validated with
GPT-4.1-mini and Claude-3.7-Sonnet on a 2,000-prompt subset
(see Appendix: Judge Robustness).
"""

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ==============================================================================
# Model Configuration
# ==============================================================================

DEFAULT_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"

# ==============================================================================
# Reproducibility — Unified Seed Count
# ==============================================================================
#
# All experiments use the same number of random seeds so that confidence
# intervals are directly comparable across the paper.  Individual experiments
# keep their own SEED_OFFSET to ensure non-overlapping RNG streams.

N_SEEDS: int = 20

K3_ARM_ORDER: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]

K3_ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

K3_DEFAULT_SWAP_ARMS: Tuple[str, str] = (
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
)
"""Default Llama <-> Mistral reward swap for non-stationary experiments."""

K3_ALL_SWAP_PAIRS: List[Tuple[str, str]] = list(combinations(K3_ARM_ORDER, 2))
"""All C(K,2)=3 pairwise reward/cost swap pairs for the K=3 portfolio."""

# ==============================================================================
# Artifact Paths
# ==============================================================================

_PACKAGE_DIR = Path(__file__).parent.parent
_PACKAGE_ARTIFACTS_DIR = _PACKAGE_DIR / "data" / "artifacts"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_PCA_PATH = _PACKAGE_ARTIFACTS_DIR / "pca_25.joblib"
DEFAULT_WARMUP_PRIORS_PATH = _PACKAGE_ARTIFACTS_DIR / "priors_k3_25comp.joblib"
DEFAULT_MODEL_REGISTRY_PATH = Path(__file__).parent / "models.json"

K3_MODELS_CONFIG_PATH = (
    PROJECT_ROOT / "data_collection" / "config" / "models_k3.json"
)

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
# Every record judged by DeepSeek-R1 (sole evaluator).
# Continuous v3 rubric: Reasoning Quality 40%, Instruction Following 30%,
# Communication Quality 30%.
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
# Budget Pacer Defaults
# ==============================================================================

DEFAULT_PACER_LR: float = 0.05
DEFAULT_PACER_LAMBDA_MAX: float = 5.0
DEFAULT_PACER_EMA_ALPHA: float = 0.05

# ==============================================================================
# Non-stationary Experiment Defaults
# ==============================================================================

DEFAULT_NONSTAT_COST_PENALTY: float = 0.20
"""Cost penalty weight used across all non-stationary experiments (02, 03,
appendix forgetting sweep, appendix gamma trajectory).
"""

# ==============================================================================
# Gemini Cost-Drop Scenario (Experiment 03 / Appendix gamma trajectories)
# ==============================================================================

GEMINI_COST_DROP = {
    "model_id": "google/gemini-2.5-pro",
    "new_input_cost_per_m": 0.10,
    "new_output_cost_per_m": 0.10,
}
"""Simulated Gemini-Pro price drop used in cost-drift experiments."""

# ==============================================================================
# Catastrophic Failure Scenario (Experiment 03)
# ==============================================================================

K3_FAILURE_ARM: str = "mistralai/mistral-large-2512"
"""Model that suffers catastrophic failure in Experiment 03."""

K3_FAILURE_REWARD: float = 0.05
"""Fixed low reward during failure (API returns garbage, not exact zero)."""

# ==============================================================================
# K=3 Budget Targets (Experiment 03 / Appendix model onboarding)
# ==============================================================================

K3_BUDGET_TARGETS: List[float] = [3.0e-4, 6.62e-4, 1.87e-3]
K3_BUDGET_LABELS: List[str] = ["tight", "moderate", "loose"]

# ==============================================================================
# Best K=3 Hyperparameters (Exp 05: T_adapt-constrained Pareto knee-point)
# ==============================================================================
#
# Protocol: priors from train.jsonl (offline only) → val.jsonl split into
# disjoint burn-in (first 1/3) + eval (remaining 2/3) for selection →
# report on test.jsonl (burn-in on full val, eval on test).
# Per-seed budget-paced Pareto AUC (20 seeds) with fixed-model endpoints
# and cost range anchored to arm cost extremes.
#
# Selection: T_adapt-constrained 2D grid search over (alpha, gamma).
# n_eff is derived from gamma via:  n_eff = (gamma^{-T_adapt} - 1)/(1-gamma)
# with T_adapt = 500 — a deployment parameter encoding the practitioner's
# target reaction time (e.g. ~5 min at 100 QPS).  Sensitivity analysis
# over T_adapt in {250, 500, 1000} confirms stable knee-point selection.
# The Pareto frontier of (budget-paced AUC, catastrophic-failure Phase-2
# reward) is built; the knee point (maximum perpendicular distance from
# the endpoint line) is selected.
#
# Source of truth: experiments/05_hparam_optimization/results/best_hparams.json
# PCA fixed at d=25 (~28.5% cumulative variance) to retain a broad semantic
# representation.  A PCA ablation (Appendix I) confirms the Pareto AUC surface
# is flat across d in [6, 25], validating this design choice.

BEST_K3_HPARAMS: Dict[str, Any] = {
    "alpha": 0.01,
    "pca_components": 25,
    "prior_n_effective": 1163.9,
    "forgetting_factor": 0.997,
}
"""Best K=3 ParetoBandit config (warmup priors, PCA-25, disjoint LinUCB).

Selected via T_adapt-constrained Pareto knee-point method:
- gamma and n_eff are coupled through the adaptation horizon
  T_adapt = 500 (deployment parameter: practitioner's target reaction
  time; stable across T_adapt in {250, 500, 1000}):
  n_eff = (gamma^{-T_adapt} - 1) / (1 - gamma).
- 2D grid search over (alpha, gamma) with n_eff derived.
- The Pareto frontier of (budget-paced AUC, Phase-2 failure reward)
  is built across all configs; the knee point (maximum perpendicular
  distance from the endpoint line) is selected.
- Cross-arm validation confirms the selected config generalises to
  failure of all K arms (not just the tuning arm, Mistral).
- Source: experiments/05_hparam_optimization/results/best_hparams.json
Val BP AUC = 0.9273, val P2 reward = 0.7167.
Test BP AUC = 0.9218 (delta = -0.38%).
"""

BEST_K3_TABULA_RASA_HPARAMS: Dict[str, Any] = {
    "alpha": 0.01,
    "pca_components": 25,
    "prior_n_effective": 1.0,
    "forgetting_factor": 0.995,
}
"""Best K=3 Tabula Rasa config (cold start, PCA-25, no priors).

Selected via T_adapt-constrained Pareto knee-point method (same protocol
as ParetoBandit but with n_eff=1.0, no warmup priors).
- Val BP AUC = 0.9257, val P2 reward = 0.7137.
- Test BP AUC = 0.9216 (delta = -0.41%).
Unlike the prior epsilon-constraint protocol (which locked Tabula Rasa to
gamma=1.0), the knee-point method selects gamma=0.995 — the cold-start
variant *can* afford forgetting when the selection criterion is a balanced
trade-off rather than a hard AUC floor.  However, stationary AUC and
failure resilience both remain lower than ParetoBandit, confirming that
warmup priors provide a structural advantage for non-stationary adaptation.
"""
