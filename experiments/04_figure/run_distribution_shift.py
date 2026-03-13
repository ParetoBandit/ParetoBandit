#!/usr/bin/env python3
"""
Figure 4: Adaptive Drift Detection Under Distribution Shift (K=2).

Demonstrates that the router can automatically detect prior miscalibration
under a distribution shift and self-adapt by resetting to tabula rasa —
no human intervention required.

Experimental setup
------------------
The router is deployed on a two-phase data stream:

  **Phase 1** (in-distribution): Pareto benchmark prompts where the
  warmup priors are well-calibrated.

  **Phase 2** (shifted): K4 real-world prompts with a **controlled
  synthetic perturbation** applied.  The perturbation couples a mean
  shift in the top PCA embedding components with a Llama reward boost,
  simulating a domain transition where the cheaper model becomes newly
  competitive (e.g., a product change introduces simpler, formulaic
  prompts).  This design follows the calibration protocol in
  ``calibrate_drift_threshold.py``, which established that pure
  covariate shift without reward landscape change does not degrade
  regret.

Four conditions are compared at a fixed cost penalty (λ=0.2):

  - **Warmup-only (ff=1.0)**: Priors loaded, no adaptation.  Baseline
    that becomes increasingly miscalibrated under the shift.
  - **Oracle ff=0.999**: Same priors but with gradual forgetting from
    step 0.  Represents a human operator who pre-configures decay.
  - **Adaptive (Reset)**: Starts with full prior trust.  The covariate
    shift detector monitors prompt embeddings and automatically resets
    to tabula rasa when it detects a significant shift.
  - **Tabula Rasa**: No priors, learns from scratch.  Reference floor.

Outputs (``results/``)
    distribution_shift_results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    BEST_K2_HPARAMS,
    BEST_K2_TABULA_RASA_HPARAMS,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.storage import EphemeralContextStore
from bandit_gpt.router import BanditRouter

from utils.embeddings import load_embedding_cache, embed_dataset_cached
from utils.simulation import compute_normalized_costs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

K4_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "rewards"
    / "archive" / "k4_canonical" / "rewards.jsonl"
)
K4_MODELS_CONFIG_PATH = (
    PROJECT_ROOT / "data_collection" / "config" / "models_k4.json"
)

ARM_ORDER: List[str] = K2_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

N_SEEDS: int = 20
SEED_OFFSET: int = 42
ONLINE_FRACTION: float = 0.70

LC_CHECKPOINT_INTERVAL: int = 50
HEADLINE_LAMBDA: float = 0.2

# Production-realistic prior strength.  With n_warmup ≈ 80k in the
# priors file, n_eff=5000 gives scale ≈ 6.25%, meaning priors persist
# for hundreds of online steps — matching a real deployment where the
# practitioner invested significant cost in offline data collection.
PRIOR_N_EFFECTIVE: float = 5000.0

# Alpha re-tuned for n_eff=5000 (tune_alpha_high_neff.py, Pareto AUC
# on val).  At strong prior trust the optimal exploration is lower:
# α=0.5 at n_eff=5000 achieves AUC=0.8686, only −0.03% vs Figure 1
# (α=1.0, n_eff=50, AUC=0.8688).
WARMUP_ALPHA_AT_HIGH_NEFF: float = 0.5

# Phase 1 (in-distribution): Pareto prompts the router sees before
# the traffic distribution shifts.  Must exceed DRIFT_BURN_IN so the
# detector establishes its baseline on "normal" traffic.
PHASE1_N_PARETO: int = 500

# ---- Controlled synthetic shift parameters ----
# The natural Pareto→K4 shift is benign (priors transfer well), so we
# inject a controlled perturbation to simulate a domain transition where
# the cheaper model becomes newly competitive.  This couples an embedding
# mean shift with a reward landscape change, the same mechanism validated
# by calibrate_drift_threshold.py.
#
# SHIFT_MAGNITUDE: σ units applied to top N_SHIFT_COMPONENTS PCA axes.
#   - 2.0σ was the calibration sweet spot (Reset saves 13.5 regret, Δ
#     well above noise).
# REWARD_BOOST_PER_SIGMA: Llama reward boost per σ of embedding shift.
#   - At 2.0σ: Llama reward += 0.08, Llama best-arm fraction rises from
#     19% to ~35%.
SHIFT_MAGNITUDE: float = 2.0
N_SHIFT_COMPONENTS: int = 8
REWARD_BOOST_PER_SIGMA: float = 0.04

# Drift detection parameters for the Adaptive condition.
# Threshold is in units of baseline standard deviations of the
# chi-squared embedding score (0 = disabled, 2.0 = 2σ, conservative).
# Calibrated via regret-grounded sweep (calibrate_drift_threshold.py):
# crossover at 1.5σ shift → sigma_excess ≈ 3.7 → threshold = 2.0σ.
DRIFT_THRESHOLD: float = 2.0
DRIFT_BURN_IN: int = 50
DRIFT_EMA_ALPHA: float = 0.05
DRIFT_CONFIRMATION_WINDOW: int = 20

# ============================================================================
# Condition definitions
# ============================================================================

CONDITIONS: List[Dict[str, Any]] = [
    {
        "label": "Warmup-only",
        "warmup": True,
        "forgetting_factor": 1.0,
        "drift_threshold": 0.0,
    },
    {
        "label": "Oracle ff=0.999",
        "warmup": True,
        "forgetting_factor": 0.999,
        "drift_threshold": 0.0,
    },
    {
        "label": "Adaptive (Reset)",
        "warmup": True,
        "forgetting_factor": 1.0,
        "drift_threshold": DRIFT_THRESHOLD,
    },
    {
        "label": "Tabula Rasa",
        "warmup": False,
        "forgetting_factor": 1.0,
        "drift_threshold": 0.0,
    },
]


# ============================================================================
# K4 data loading — aggregate per-model records into per-prompt K=2 format
# ============================================================================


def _load_k4_model_pricing() -> Dict[str, Dict[str, float]]:
    """Load per-million-token pricing from the K4 model config."""
    with open(K4_MODELS_CONFIG_PATH) as f:
        entries = json.load(f)["models"]
    pricing: Dict[str, Dict[str, float]] = {}
    for entry in entries:
        pricing[entry["model_id"]] = {
            "input_cost_per_m": entry["input_cost_per_m"],
            "output_cost_per_m": entry["output_cost_per_m"],
        }
    return pricing


def _estimate_request_cost(
    prompt: str,
    response: str,
    input_cost_per_m: float,
    output_cost_per_m: float,
) -> float:
    """Estimate per-request dollar cost from text lengths.

    Uses the standard approximation of 1 token ≈ 4 bytes of UTF-8.
    """
    input_tokens = max(len(prompt.encode("utf-8")) / 4, 1)
    output_tokens = max(len(response.encode("utf-8")) / 4, 1)
    return (
        input_tokens * input_cost_per_m
        + output_tokens * output_cost_per_m
    ) / 1_000_000


def load_k4_as_per_prompt(
    arm_order: List[str],
) -> List[Dict[str, Any]]:
    """Load K4 canonical rewards and aggregate to per-prompt records.

    Extracts DeepSeek-R1 rewards for consistency with the Pareto benchmark
    dataset (which was judged by R1 only).  Costs are estimated from
    response token counts and model pricing.

    Args:
        arm_order: Model identifiers to include as arms.

    Returns:
        List of dicts with ``prompt``, ``arms``, and ``source`` keys,
        matching the canonical ``train.jsonl`` schema.
    """
    pricing = _load_k4_model_pricing()
    arm_set = set(arm_order)

    by_prompt: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with open(K4_REWARDS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            model_id = rec["model_id"]
            if model_id not in arm_set:
                continue
            prompt = rec["prompt"].strip()
            if prompt not in by_prompt:
                by_prompt[prompt] = {}

            r1_reward: Optional[float] = None
            for jd in rec.get("judge_details", []):
                if jd["judge"] == "deepseek/deepseek-r1":
                    r1_reward = jd["reward"]
                    break
            if r1_reward is None:
                continue

            model_pricing = pricing.get(model_id, {})
            cost = _estimate_request_cost(
                prompt, rec.get("response", ""),
                model_pricing.get("input_cost_per_m", 0.1),
                model_pricing.get("output_cost_per_m", 0.1),
            )
            by_prompt[prompt][model_id] = {"reward": r1_reward, "cost": cost}

    result: List[Dict[str, Any]] = []
    for prompt, arms in by_prompt.items():
        if all(a in arms for a in arm_order):
            result.append({
                "prompt": prompt,
                "arms": {a: arms[a] for a in arm_order},
                "source": "k4_canonical",
            })
    return result


def load_pareto_split(path: Path, arm_order: List[str]) -> List[Dict[str, Any]]:
    """Load a Pareto JSONL split, keeping only the requested arms.

    Args:
        path: Path to a canonical JSONL file (train/val/test).
        arm_order: Model identifiers to retain.

    Returns:
        List of per-prompt dicts with ``prompt``, ``arms``, ``source``.
    """
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            arms = {
                a: {"reward": rec["arms"][a]["reward"],
                    "cost": rec["arms"][a]["cost"]}
                for a in arm_order if a in rec["arms"]
            }
            if len(arms) == len(arm_order):
                records.append({
                    "prompt": rec["prompt"],
                    "arms": arms,
                    "source": rec.get("source", "pareto"),
                })
    return records


# ============================================================================
# Embedding helpers
# ============================================================================


def embed_records(
    records: List[Dict[str, Any]],
    embedding_cache: Dict[str, np.ndarray],
    encoder: Any,
    pca: Any,
) -> List[np.ndarray]:
    """Embed a list of per-prompt records using the cache + live fallback."""
    return embed_dataset_cached(records, embedding_cache, encoder, pca)


# ============================================================================
# Distribution shift quantification
# ============================================================================


def quantify_distribution_shift(
    pareto_emb: np.ndarray,
    k4_emb: np.ndarray,
    pareto_records: List[Dict[str, Any]],
    k4_records: List[Dict[str, Any]],
    arm_order: List[str],
    *,
    n_psi_bins: int = 10,
) -> Dict[str, Any]:
    """Compute distribution shift statistics between two prompt sets.

    Reports Population Stability Index (PSI) on the first principal
    component and Kolmogorov-Smirnov statistics, plus best-arm
    distribution comparison.

    Args:
        pareto_emb: Embedding matrix for pareto data (n_pareto, d).
        k4_emb: Embedding matrix for K4 data (n_k4, d).
        pareto_records: Pareto per-prompt records.
        k4_records: K4 per-prompt records.
        arm_order: Model identifiers.
        n_psi_bins: Number of bins for PSI computation.

    Returns:
        Dict with PSI, KS statistics, and best-arm distribution.
    """
    pc0_pareto = pareto_emb[:, 0]
    pc0_k4 = k4_emb[:, 0]

    ks_stat, ks_p = sp_stats.ks_2samp(pc0_pareto, pc0_k4)

    combined = np.concatenate([pc0_pareto, pc0_k4])
    edges = np.percentile(combined, np.linspace(0, 100, n_psi_bins + 1))
    edges[-1] += 1e-6
    h_pareto, _ = np.histogram(pc0_pareto, bins=edges, density=True)
    h_k4, _ = np.histogram(pc0_k4, bins=edges, density=True)
    h_pareto = np.clip(h_pareto, 1e-8, None)
    h_k4 = np.clip(h_k4, 1e-8, None)
    p = h_pareto / h_pareto.sum()
    q = h_k4 / h_k4.sum()
    psi = float(np.sum((p - q) * np.log(p / q)))

    def _best_arm_dist(records: List[Dict]) -> Dict[str, float]:
        counts: Dict[str, int] = {a: 0 for a in arm_order}
        for rec in records:
            best = max(arm_order, key=lambda a: rec["arms"][a]["reward"])
            counts[best] += 1
        total = len(records)
        return {a: counts[a] / total for a in arm_order}

    pareto_best = _best_arm_dist(pareto_records)
    k4_best = _best_arm_dist(k4_records)

    reward_comparison: Dict[str, Dict[str, float]] = {}
    for arm in arm_order:
        p_mean = float(np.mean([r["arms"][arm]["reward"] for r in pareto_records]))
        k_mean = float(np.mean([r["arms"][arm]["reward"] for r in k4_records]))
        reward_comparison[ARM_SHORT[arm]] = {
            "pareto_mean": p_mean,
            "k4_mean": k_mean,
            "delta": k_mean - p_mean,
        }

    severity = (
        "Negligible" if psi < 0.10 else
        "Moderate" if psi < 0.25 else
        "Substantial"
    )

    return {
        "psi": psi,
        "psi_severity": severity,
        "ks_stat": ks_stat,
        "ks_p_value": ks_p,
        "n_pareto": len(pareto_records),
        "n_k4": len(k4_records),
        "best_arm_pareto": pareto_best,
        "best_arm_k4": k4_best,
        "reward_comparison": reward_comparison,
    }


# ============================================================================
# Prior miscalibration
# ============================================================================


def measure_prior_miscalibration(
    warmup_path: str,
    k4_records: List[Dict[str, Any]],
    k4_emb: List[np.ndarray],
    arm_order: List[str],
    feature_dim: int,
    *,
    prior_n_effective: float,
) -> Dict[str, Any]:
    """Measure how miscalibrated warmup priors are on the K4 distribution.

    Builds a router with warmup priors (no online data) and evaluates its
    step-0 predictions on K4 data.  Compares the prior-implied arm
    selection to the actual best arm.

    Args:
        warmup_path: Path to K2 warmup priors joblib file.
        k4_records: K4 per-prompt records.
        k4_emb: K4 embeddings.
        arm_order: Model identifiers.
        feature_dim: Context vector dimensionality.
        prior_n_effective: Effective sample size for prior scaling.

    Returns:
        Dict with miscalibration metrics.
    """
    from utils.simulation import build_model_registry

    registry = build_model_registry(arm_order)
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup",
        warmup_path=warmup_path,
        prior_n_effective=prior_n_effective,
        alpha=BEST_K2_HPARAMS["alpha"],
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=1.0,
    )

    prior_choices: Dict[str, int] = {a: 0 for a in arm_order}
    correct = 0
    total_reward = 0.0
    oracle_reward = 0.0

    for i, rec in enumerate(k4_records):
        emb = k4_emb[i]
        model, _log = router.route(emb)
        prior_choices[model] += 1
        total_reward += rec["arms"][model]["reward"]
        best_arm = max(arm_order, key=lambda a: rec["arms"][a]["reward"])
        oracle_reward += rec["arms"][best_arm]["reward"]
        if model == best_arm:
            correct += 1

    n = len(k4_records)
    return {
        "prior_arm_fractions": {
            ARM_SHORT[a]: prior_choices[a] / n for a in arm_order
        },
        "prior_accuracy": correct / n,
        "prior_mean_reward": total_reward / n,
        "oracle_mean_reward": oracle_reward / n,
        "reward_gap": (oracle_reward - total_reward) / n,
    }


# ============================================================================
# Learning curve with drift-detection tracking
# ============================================================================


def run_learning_curve(
    label: str,
    phase1_records: List[Dict[str, Any]],
    phase1_emb: List[np.ndarray],
    phase2_records: List[Dict[str, Any]],
    phase2_emb: List[np.ndarray],
    eval_records: List[Dict[str, Any]],
    eval_emb: List[np.ndarray],
    arm_order: List[str],
    feature_dim: int,
    normalized_costs: Dict[str, float],
    *,
    alpha: float,
    prior_n_effective: float,
    warmup_path: Optional[str],
    cost_penalty: float = 0.0,
    forgetting_factor: float = 1.0,
    drift_threshold: float = 0.0,
    drift_burn_in_steps: int = DRIFT_BURN_IN,
    drift_ema_alpha: float = DRIFT_EMA_ALPHA,
    drift_confirmation_window: int = DRIFT_CONFIRMATION_WINDOW,
    policy: str = "hybrid",
    n_seeds: int = N_SEEDS,
    seed_offset: int = SEED_OFFSET,
    checkpoint_interval: int = LC_CHECKPOINT_INTERVAL,
) -> List[Dict[str, Any]]:
    """Run two-phase learning curves with periodic frozen holdout evaluation.

    The router first processes **Phase 1** prompts (in-distribution,
    e.g. Pareto benchmark) where priors are well-calibrated, then
    **Phase 2** prompts (cross-distribution, e.g. K4 real-world traffic)
    where priors are miscalibrated.  Within each phase, prompts are
    shuffled independently per seed, but phase order is preserved.

    The drift detector's burn-in occurs during Phase 1, establishing a
    baseline embedding distribution (per-component mean/std).  When
    Phase 2 begins, the chi-squared covariate shift score increases
    and the detector fires if the threshold is exceeded.

    Cost-adjusted regret per step::

        R_t = max_a [r_a(x_t) - λ·nc_a] - [r_chosen(x_t) - λ·nc_chosen]

    Holdout evaluation (frozen snapshots) is always on Phase 2 data
    (the deployment distribution we care about).

    Args:
        label: Human-readable condition label.
        phase1_records: In-distribution prompts (Pareto).
        phase1_emb: Phase 1 embeddings.
        phase2_records: Cross-distribution prompts (K4 online split).
        phase2_emb: Phase 2 embeddings.
        eval_records: Holdout evaluation prompts (K4 holdout).
        eval_emb: Holdout embeddings.
        arm_order: Model identifiers.
        feature_dim: Context vector dimensionality.
        normalized_costs: ``{model_id: normalized_cost}`` in [0, 1].
        alpha: UCB exploration parameter.
        prior_n_effective: Prior scaling factor.
        warmup_path: Path to warmup priors (None for tabula rasa).
        cost_penalty: Lambda for cost-adjusted regret and routing.
        forgetting_factor: Initial forgetting factor.
        drift_threshold: Sigma-based threshold for the DriftDetector.
                       When exceeded, the router resets to tabula rasa.
                       (0.0 = disabled, 2.0 = 2σ conservative).
        drift_burn_in_steps: Burn-in period for the drift detector.
        drift_ema_alpha: EMA smoothing factor for drift detector.
        drift_confirmation_window: Consecutive above-threshold readings
                       required before drift is confirmed.
        policy: Bandit policy type.
        n_seeds: Number of seeds.
        seed_offset: Base seed offset.
        checkpoint_interval: Steps between frozen evaluations.

    Returns:
        List of checkpoint dicts aggregated across seeds.  Each contains
        step, reward, cost, arm_fractions, cumulative_regret,
        drift_state, forgetting_factor, and the phase boundary index.
    """
    from utils.simulation import build_model_registry

    n_p1 = len(phase1_records)
    n_p2 = len(phase2_records)
    n_train = n_p1 + n_p2
    n_eval = len(eval_records)

    checkpoints = sorted(set(
        [0] + list(range(checkpoint_interval, n_train, checkpoint_interval))
        + [n_train]
    ))

    per_seed_curves: List[Dict[int, Dict[str, Any]]] = []

    for s in range(n_seeds):
        seed = seed_offset + s
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        registry = build_model_registry(arm_order)
        fs = FeatureService.for_precomputed(feature_dim)
        store = EphemeralContextStore()

        is_tabula_rasa = warmup_path is None
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=fs,
            context_store=store,
            priors="none" if is_tabula_rasa else "warmup",
            warmup_path=None if is_tabula_rasa else warmup_path,
            prior_n_effective=prior_n_effective,
            alpha=alpha,
            use_corralling=False,
            cost_penalty=cost_penalty,
            forgetting_factor=forgetting_factor,
            drift_threshold=drift_threshold,
            drift_burn_in_steps=drift_burn_in_steps,
            drift_ema_alpha=drift_ema_alpha,
            drift_confirmation_window=drift_confirmation_window,
            policy="disjoint" if is_tabula_rasa else policy,
        )

        # Build two-phase stream: shuffle within each phase, preserve order.
        p1_order = rng.permutation(n_p1)
        p2_order = rng.permutation(n_p2)

        all_records: List[Dict[str, Any]] = (
            [phase1_records[i] for i in p1_order]
            + [phase2_records[i] for i in p2_order]
        )
        all_emb: List[np.ndarray] = (
            [phase1_emb[i] for i in p1_order]
            + [phase2_emb[i] for i in p2_order]
        )

        curve: Dict[int, Dict[str, Any]] = {}
        checkpoint_set = set(checkpoints)
        cumulative_regret: float = 0.0

        def _frozen_eval() -> Dict[str, Any]:
            """Evaluate on K4 holdout via select_arm (no state mutation).

            Uses ``bandit.select_arm()`` directly, which is a pure read
            of A_inv/b.  This avoids ``route()`` which mutates:
              - drift_detector (burn-in, EMA updates)
              - bandit.t (time counter via mark_selected)
            """
            cp: Optional[Dict[str, float]] = None
            if cost_penalty > 0:
                cp = {
                    m: cost_penalty * normalized_costs[m]
                    for m in arm_order
                }
            rewards: List[float] = []
            costs: List[float] = []
            arm_counts: Dict[str, int] = {a: 0 for a in arm_order}
            for j in range(n_eval):
                model, _score = router.bandit.select_arm(
                    eval_emb[j], cost_penalties=cp,
                )
                rewards.append(eval_records[j]["arms"][model]["reward"])
                costs.append(eval_records[j]["arms"][model]["cost"])
                arm_counts[model] += 1

            snapshot: Dict[str, Any] = {
                "reward": float(np.mean(rewards)),
                "cost": float(np.mean(costs)),
                "arm_counts": arm_counts,
                "cumulative_regret": cumulative_regret,
                "forgetting_factor": router.bandit.gamma,
                "n_resets": getattr(router, "_n_resets", 0),
            }

            if router.drift_detector is not None:
                snapshot["drift_state"] = router.drift_detector.get_state()
            else:
                snapshot["drift_state"] = None

            return snapshot

        if 0 in checkpoint_set:
            curve[0] = _frozen_eval()

        for t in range(n_train):
            emb = all_emb[t]
            rec = all_records[t]
            model, log = router.route(emb)
            reward = rec["arms"][model]["reward"]
            router.process_feedback(log.request_id, reward=reward)

            oracle_utility = max(
                rec["arms"][a]["reward"] - cost_penalty * normalized_costs[a]
                for a in arm_order
            )
            chosen_utility = reward - cost_penalty * normalized_costs[model]
            cumulative_regret += oracle_utility - chosen_utility

            step = t + 1
            if step in checkpoint_set:
                curve[step] = _frozen_eval()

        per_seed_curves.append(curve)

    # ------------------------------------------------------------------
    # Aggregate across seeds
    # ------------------------------------------------------------------
    result: List[Dict[str, Any]] = []
    for step in checkpoints:
        seed_data = [c[step] for c in per_seed_curves if step in c]
        if not seed_data:
            continue

        rewards = [d["reward"] for d in seed_data]
        costs = [d["cost"] for d in seed_data]
        regrets = [d["cumulative_regret"] for d in seed_data]
        ffs = [d["forgetting_factor"] for d in seed_data]
        resets = [d.get("n_resets", 0) for d in seed_data]

        arm_frac: Dict[str, float] = {}
        arm_frac_std: Dict[str, float] = {}
        for arm in arm_order:
            fracs = [d["arm_counts"][arm] / n_eval for d in seed_data]
            arm_frac[ARM_SHORT[arm]] = float(np.mean(fracs))
            arm_frac_std[ARM_SHORT[arm]] = float(np.std(fracs))

        drift_states = [d["drift_state"] for d in seed_data if d["drift_state"] is not None]
        agg_drift: Optional[Dict[str, Any]] = None
        if drift_states:
            agg_drift = {
                "mean_drift_score": float(np.mean([ds["drift_score"] for ds in drift_states])),
                "std_drift_score": float(np.std([ds["drift_score"] for ds in drift_states])),
                "mean_ema_chi2": float(np.mean([ds["ema_chi2"] for ds in drift_states])),
                "mean_baseline": float(np.mean([ds["baseline"] for ds in drift_states])),
                "mean_baseline_std": float(np.mean([ds["baseline_std"] for ds in drift_states])),
                "frac_drifting": float(np.mean([1.0 if ds["is_drifting"] else 0.0 for ds in drift_states])),
                "frac_confirmed": float(np.mean([1.0 if ds["confirmed"] else 0.0 for ds in drift_states])),
                "mean_threshold": float(np.mean([ds["threshold"] for ds in drift_states])),
                "mean_consecutive_above": float(np.mean([ds["consecutive_above"] for ds in drift_states])),
            }

        entry: Dict[str, Any] = {
            "step": step,
            "phase": "in-dist" if step <= n_p1 else "shifted",
            "phase_boundary": n_p1,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "mean_cumulative_regret": float(np.mean(regrets)),
            "std_cumulative_regret": float(np.std(regrets)),
            "arm_fractions": arm_frac,
            "arm_fractions_std": arm_frac_std,
            "mean_forgetting_factor": float(np.mean(ffs)),
            "std_forgetting_factor": float(np.std(ffs)),
            "mean_n_resets": float(np.mean(resets)),
            "drift_state": agg_drift,
            "n_seeds": len(seed_data),
            "label": label,
        }
        result.append(entry)

    return result


# ============================================================================
# Split helpers
# ============================================================================


def split_online_holdout(
    records: List[Dict[str, Any]],
    embeddings: List[np.ndarray],
    online_fraction: float = ONLINE_FRACTION,
    seed: int = 42,
) -> Tuple[
    List[Dict[str, Any]], List[np.ndarray],
    List[Dict[str, Any]], List[np.ndarray],
]:
    """Split data into online learning pool and holdout evaluation set.

    Args:
        records: Per-prompt records.
        embeddings: Aligned embedding list.
        online_fraction: Fraction for online learning.
        seed: Random seed for reproducibility.

    Returns:
        ``(online_records, online_emb, holdout_records, holdout_emb)``
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    idx = rng.permutation(n)
    n_online = int(n * online_fraction)

    online_idx = idx[:n_online]
    holdout_idx = idx[n_online:]

    return (
        [records[i] for i in online_idx],
        [embeddings[i] for i in online_idx],
        [records[i] for i in holdout_idx],
        [embeddings[i] for i in holdout_idx],
    )


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:
    """Run the two-phase adaptive drift detection experiment (K=2).

    Phase 1: Pareto benchmark prompts (in-distribution for the priors).
    Phase 2: K4 real-world prompts (cross-distribution shift).

    The drift detector's burn-in occurs during Phase 1, establishing
    a baseline embedding distribution.  When Phase 2 begins, the
    chi-squared covariate shift score increases, triggering adaptation.
    Prior strength is set to a production-realistic
    ``PRIOR_N_EFFECTIVE`` so priors persist for hundreds of steps.
    """
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("FIGURE 4: ADAPTIVE DRIFT DETECTION (TWO-PHASE)")
    logger.info("  K=2 (Llama-3.1-8B + Gemini-2.5-Pro)")
    logger.info(f"  {N_SEEDS} seeds, λ={HEADLINE_LAMBDA}, "
                f"n_eff={PRIOR_N_EFFECTIVE}")
    logger.info(f"  Phase 1: {PHASE1_N_PARETO} Pareto prompts (in-dist)")
    logger.info(f"  Phase 2: K4 prompts + synthetic {SHIFT_MAGNITUDE:.1f}σ "
                f"shift (embedding + reward)")
    logger.info(f"  Drift: threshold={DRIFT_THRESHOLD}σ (tabula rasa reset), "
                f"burn_in={DRIFT_BURN_IN}, "
                f"confirm_window={DRIFT_CONFIRMATION_WINDOW}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    logger.info("\nLoading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    feature_dim = pca.n_components_ + 1

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    # ------------------------------------------------------------------
    # Hyperparameters from config
    # ------------------------------------------------------------------
    hparams_warmup = {
        **BEST_K2_HPARAMS,
        "alpha": WARMUP_ALPHA_AT_HIGH_NEFF,
    }
    hparams_tr = BEST_K2_TABULA_RASA_HPARAMS

    logger.info(f"  Warmup hparams: alpha={hparams_warmup['alpha']} "
                f"(re-tuned for n_eff={PRIOR_N_EFFECTIVE})")
    logger.info(f"  Production n_eff: {PRIOR_N_EFFECTIVE}")
    logger.info(f"  Tabula rasa: alpha={hparams_tr['alpha']}")

    if not K2_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=2 warmup priors not found: {K2_WARMUP_PRIORS_PATH}\n"
            "Generate with: python scripts/generate_multimodel_warmup_priors.py"
        )
    warmup_path = str(K2_WARMUP_PRIORS_PATH)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("\nLoading Pareto benchmark data ...")
    pareto_all = load_pareto_split(TRAIN_DATA_PATH, ARM_ORDER)
    logger.info(f"  Pareto train: {len(pareto_all)}")

    logger.info("\nLoading K4 canonical data (cross-distribution) ...")
    k4_all = load_k4_as_per_prompt(ARM_ORDER)
    logger.info(f"  K4 prompts with complete K=2 arms: {len(k4_all)}")

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    logger.info("\nEmbedding prompts ...")
    pareto_all_emb = embed_records(
        pareto_all, embedding_cache, encoder, pca,
    )
    k4_all_emb = embed_records(k4_all, embedding_cache, encoder, pca)
    logger.info(f"  Pareto: {len(pareto_all_emb)} vectors")
    logger.info(f"  K4: {len(k4_all_emb)} vectors")

    # ------------------------------------------------------------------
    # Build Phase 1 (in-dist Pareto) and Phase 2 (shifted K4)
    # ------------------------------------------------------------------
    # Phase 1: sample PHASE1_N_PARETO prompts from Pareto train.
    # These are the prompts the router sees before the shift.
    rng_split = np.random.default_rng(0)
    p1_idx = rng_split.choice(len(pareto_all), size=PHASE1_N_PARETO, replace=False)
    phase1_records = [pareto_all[i] for i in p1_idx]
    phase1_emb = [pareto_all_emb[i] for i in p1_idx]
    logger.info(f"\n  Phase 1 (in-dist): {len(phase1_records)} Pareto prompts")

    # Phase 2: K4 split into online + holdout.
    logger.info(f"  Splitting K4 into online ({ONLINE_FRACTION:.0%}) "
                f"and holdout ({1 - ONLINE_FRACTION:.0%}) ...")
    k4_online_raw, k4_online_emb_raw, k4_holdout_raw, k4_holdout_emb_raw = (
        split_online_holdout(k4_all, k4_all_emb)
    )

    # ------------------------------------------------------------------
    # Apply controlled synthetic shift (embedding + reward perturbation)
    # ------------------------------------------------------------------
    llama_arm = ARM_ORDER[0]
    reward_boost = SHIFT_MAGNITUDE * REWARD_BOOST_PER_SIGMA

    # Embedding shift: add SHIFT_MAGNITUDE * ref_std to top PCA components
    pareto_matrix = np.array(phase1_emb)
    ref_std_per_comp = pareto_matrix.std(axis=0)
    shift_vec = np.zeros(feature_dim)
    for j in range(min(N_SHIFT_COMPONENTS, feature_dim - 1)):
        shift_vec[j] = SHIFT_MAGNITUDE * ref_std_per_comp[j]

    def _apply_shift(
        records: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """Shift embeddings and boost Llama reward for a batch of records."""
        shifted_emb = [emb + shift_vec for emb in embeddings]
        shifted_rec = []
        for rec in records:
            new_rec = {
                "prompt": rec["prompt"],
                "arms": {},
                "source": rec.get("source", "k4_canonical"),
            }
            for arm in ARM_ORDER:
                r = rec["arms"][arm]["reward"]
                c = rec["arms"][arm]["cost"]
                if arm == llama_arm:
                    r = min(1.0, r + reward_boost)
                new_rec["arms"][arm] = {"reward": r, "cost": c}
            shifted_rec.append(new_rec)
        return shifted_rec, shifted_emb

    k4_online, k4_online_emb = _apply_shift(k4_online_raw, k4_online_emb_raw)
    k4_holdout, k4_holdout_emb = _apply_shift(k4_holdout_raw, k4_holdout_emb_raw)

    orig_llama_r = float(np.mean(
        [r["arms"][llama_arm]["reward"] for r in k4_online_raw]
    ))
    new_llama_r = float(np.mean(
        [r["arms"][llama_arm]["reward"] for r in k4_online]
    ))
    logger.info(f"\n  Synthetic shift applied: {SHIFT_MAGNITUDE:.1f}σ on top "
                f"{N_SHIFT_COMPONENTS} PCA components")
    logger.info(f"  Llama reward boost: +{reward_boost:.3f} "
                f"({orig_llama_r:.4f} → {new_llama_r:.4f})")

    logger.info(f"  Phase 2 (shifted): {len(k4_online)} K4 online prompts")
    logger.info(f"  Holdout eval: {len(k4_holdout)} K4 holdout prompts")
    logger.info(f"  Total online stream: {len(phase1_records) + len(k4_online)} "
                f"(shift at step {len(phase1_records)})")

    # ------------------------------------------------------------------
    # Quantify distribution shift (paper text) — on shifted data
    # ------------------------------------------------------------------
    logger.info("\nQuantifying distribution shift (after synthetic perturbation) ...")
    pareto_emb_matrix = np.array(pareto_all_emb)
    k4_shifted_all, k4_shifted_all_emb = _apply_shift(k4_all, k4_all_emb)
    k4_emb_matrix = np.array(k4_shifted_all_emb)
    shift_stats = quantify_distribution_shift(
        pareto_emb_matrix, k4_emb_matrix,
        pareto_all, k4_shifted_all, ARM_ORDER,
    )
    logger.info(f"  PSI: {shift_stats['psi']:.3f} ({shift_stats['psi_severity']})")
    logger.info(f"  KS stat: {shift_stats['ks_stat']:.3f}, "
                f"p={shift_stats['ks_p_value']:.2e}")
    logger.info("  Best-arm distribution:")
    for arm in ARM_ORDER:
        p_frac = shift_stats["best_arm_pareto"][arm]
        k_frac = shift_stats["best_arm_k4"][arm]
        logger.info(f"    {ARM_SHORT[arm]}: "
                     f"pareto={p_frac:.1%} → k4={k_frac:.1%} "
                     f"(Δ={k_frac - p_frac:+.1%})")

    # ------------------------------------------------------------------
    # Prior miscalibration on K4 at production n_eff
    # ------------------------------------------------------------------
    logger.info(f"\nMeasuring prior miscalibration on K4 (n_eff={PRIOR_N_EFFECTIVE}) ...")
    miscal = measure_prior_miscalibration(
        warmup_path, k4_holdout, k4_holdout_emb, ARM_ORDER, feature_dim,
        prior_n_effective=PRIOR_N_EFFECTIVE,
    )
    logger.info(f"  Prior accuracy on K4: {miscal['prior_accuracy']:.1%}")
    logger.info(f"  Prior mean reward: {miscal['prior_mean_reward']:.4f}")
    logger.info(f"  Oracle mean reward: {miscal['oracle_mean_reward']:.4f}")
    logger.info(f"  Reward gap: {miscal['reward_gap']:.4f}")

    # ==================================================================
    # Normalized costs
    # ==================================================================
    from utils.simulation import build_model_registry as _build_registry
    registry = _build_registry(ARM_ORDER)
    norm_costs = compute_normalized_costs(registry, ARM_ORDER)
    logger.info("\nNormalized costs (log-scale [0,1]):")
    for arm in ARM_ORDER:
        logger.info(f"  {ARM_SHORT[arm]}: {norm_costs[arm]:.4f}")

    # ==================================================================
    # Cost-adjusted oracle arm fractions (shifted K4)
    # ==================================================================
    logger.info("\n" + "-" * 70)
    logger.info("DIAGNOSTIC: Cost-adjusted oracle arm fractions (shifted K4)")
    logger.info("-" * 70)
    oracle_arm_counts: Dict[str, int] = {a: 0 for a in ARM_ORDER}
    for rec in k4_shifted_all:
        utilities = {
            a: rec["arms"][a]["reward"] - HEADLINE_LAMBDA * norm_costs[a]
            for a in ARM_ORDER
        }
        best_arm = max(utilities, key=utilities.get)  # type: ignore[arg-type]
        oracle_arm_counts[best_arm] += 1
    n_total = len(k4_shifted_all)
    oracle_llama_frac = oracle_arm_counts[ARM_ORDER[0]] / n_total
    logger.info(f"  λ={HEADLINE_LAMBDA:.2f}: "
                 + "  ".join(
                     f"{ARM_SHORT[a]}={oracle_arm_counts[a]/n_total:.1%}"
                     for a in ARM_ORDER
                 ))

    # ==================================================================
    # RUN 4 CONDITIONS (two-phase stream)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"RUNNING 4 CONDITIONS (λ={HEADLINE_LAMBDA}, "
                f"n_eff={PRIOR_N_EFFECTIVE})")
    logger.info("=" * 70)

    conditions_results: Dict[str, List[Dict[str, Any]]] = {}

    for cond in CONDITIONS:
        cond_label = cond["label"]
        is_warmup = cond["warmup"]
        ff = cond["forgetting_factor"]
        dt = cond["drift_threshold"]

        hp = hparams_warmup if is_warmup else hparams_tr
        wp = warmup_path if is_warmup else None
        pol = hp.get("policy", "hybrid")
        n_eff = PRIOR_N_EFFECTIVE if is_warmup else hparams_tr["prior_n_effective"]
        if not is_warmup:
            pol = "disjoint"

        logger.info(f"\n--- {cond_label} (ff={ff}, drift_threshold={dt}, "
                    f"n_eff={n_eff}) ---")
        logger.info(f"  alpha={hp['alpha']}, policy={pol}, {N_SEEDS} seeds ...")

        lc = run_learning_curve(
            cond_label,
            phase1_records, phase1_emb,
            k4_online, k4_online_emb,
            k4_holdout, k4_holdout_emb,
            ARM_ORDER, feature_dim, norm_costs,
            alpha=hp["alpha"],
            prior_n_effective=n_eff,
            warmup_path=wp,
            cost_penalty=HEADLINE_LAMBDA,
            forgetting_factor=ff,
            drift_threshold=dt,
            drift_burn_in_steps=DRIFT_BURN_IN,
            drift_ema_alpha=DRIFT_EMA_ALPHA,
            drift_confirmation_window=DRIFT_CONFIRMATION_WINDOW,
            policy=pol,
        )

        conditions_results[cond_label] = lc

        if lc:
            af = lc[-1].get("arm_fractions", {})
            reg = lc[-1]["mean_cumulative_regret"]
            ff_final = lc[-1].get("mean_forgetting_factor", ff)
            n_rst = lc[-1].get("mean_n_resets", 0)
            ds = lc[-1].get("drift_state")
            logger.info(f"  Final regret={reg:.2f}, arm_frac={af}, "
                        f"ff={ff_final:.4f}, resets={n_rst:.1f}")
            if ds:
                logger.info(f"  Drift: score={ds['mean_drift_score']:.4f}, "
                            f"baseline={ds['mean_baseline']:.4f}, "
                            f"ema_chi2={ds['mean_ema_chi2']:.4f}, "
                            f"frac_drifting={ds['frac_drifting']:.0%}")

    # ==================================================================
    # Static baselines
    # ==================================================================
    oracle_reward = float(np.mean([
        max(r["arms"][a]["reward"] for a in ARM_ORDER) for r in k4_holdout
    ]))
    static_baselines: Dict[str, Dict[str, float]] = {}
    for arm in ARM_ORDER:
        static_baselines[ARM_SHORT[arm]] = {
            "reward": float(np.mean([
                r["arms"][arm]["reward"] for r in k4_holdout
            ])),
            "cost": float(np.mean([
                r["arms"][arm]["cost"] for r in k4_holdout
            ])),
            "normalized_cost": norm_costs[arm],
        }

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"SUMMARY (λ={HEADLINE_LAMBDA}, n_eff={PRIOR_N_EFFECTIVE}, "
                f"shift={SHIFT_MAGNITUDE:.1f}σ)")
    logger.info("=" * 70)

    logger.info(f"\n  Oracle reward: {oracle_reward:.4f}")
    logger.info(f"  Oracle Llama%: {oracle_llama_frac:.1%}")
    for bl_name, bl in static_baselines.items():
        logger.info(f"  {bl_name}-only: R={bl['reward']:.4f}, "
                     f"norm_cost={bl['normalized_cost']:.4f}")

    logger.info(f"\n  Conditions:")
    for cond_label, lc in conditions_results.items():
        if lc:
            logger.info(f"    {cond_label}: "
                        f"regret={lc[-1]['mean_cumulative_regret']:.2f}, "
                        f"Llama%={lc[-1].get('arm_fractions', {}).get('Llama-8B', 0):.1%}, "
                        f"ff={lc[-1].get('mean_forgetting_factor', 0):.4f}")

    # ==================================================================
    # Serialise
    # ==================================================================
    def _strip_arrays(obj: Any) -> Any:
        """Recursively convert numpy arrays to lists for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _strip_arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip_arrays(v) for v in obj]
        return obj

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Figure 4: Adaptive Drift Detection — Two-Phase (K=2)",
            "n_seeds": N_SEEDS,
            "checkpoint_interval": LC_CHECKPOINT_INTERVAL,
            "arm_order": ARM_ORDER,
            "arm_short": ARM_SHORT,
            "online_fraction": ONLINE_FRACTION,
            "headline_lambda": HEADLINE_LAMBDA,
            "prior_n_effective": PRIOR_N_EFFECTIVE,
            "phase1_n_pareto": PHASE1_N_PARETO,
            "shift_magnitude": SHIFT_MAGNITUDE,
            "n_shift_components": N_SHIFT_COMPONENTS,
            "reward_boost_per_sigma": REWARD_BOOST_PER_SIGMA,
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_adaptation": "tabula_rasa_reset",
            "drift_burn_in": DRIFT_BURN_IN,
            "drift_ema_alpha": DRIFT_EMA_ALPHA,
            "drift_confirmation_window": DRIFT_CONFIRMATION_WINDOW,
            "normalized_costs": {ARM_SHORT[a]: norm_costs[a] for a in ARM_ORDER},
            "oracle_llama_frac": oracle_llama_frac,
            "hparams_warmup_only": hparams_warmup,
            "warmup_alpha_retuned_at_neff": WARMUP_ALPHA_AT_HIGH_NEFF,
            "hparams_tabula_rasa": hparams_tr,
            "warmup_priors_path": str(K2_WARMUP_PRIORS_PATH),
            "pareto_source": str(TRAIN_DATA_PATH),
            "k4_source": str(K4_REWARDS_PATH),
        },
        "distribution_shift": shift_stats,
        "prior_miscalibration": miscal,
        "data_split": {
            "pareto_total": len(pareto_all),
            "phase1_pareto": len(phase1_records),
            "k4_total": len(k4_all),
            "k4_online": len(k4_online),
            "k4_holdout": len(k4_holdout),
        },
        "cross_dist": {
            "phase1_n": len(phase1_records),
            "phase2_n": len(k4_online),
            "n_holdout": len(k4_holdout),
            "oracle_reward": oracle_reward,
            "static_baselines": static_baselines,
            "conditions": conditions_results,
        },
    }

    out_path = output_dir / "distribution_shift_results.json"
    with open(out_path, "w") as f:
        json.dump(_strip_arrays(results), f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults → {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_experiment()
