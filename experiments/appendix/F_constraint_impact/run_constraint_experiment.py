


#!/usr/bin/env python3
"""
Hard Constraint Impact Experiment
=================================

Validates banditGPT's Layer 1 hard per-request constraint filtering—the
mechanism claimed as contribution (3) in the abstract.

Protocol:
  1. Load the K=3 portfolio with cost + latency metadata
  2. For each constraint regime × N_TRIALS seeds:
       a. Instantiate router with warmup priors
       b. Train on online-learn set (533 prompts) WITH constraints active
       c. Freeze; evaluate on holdout (750 prompts) WITH constraints active
       d. Track: quality, realized cost, latency, eligible K', violations
  3. Produce JSON results consumed by the figure generator

Conditions:
  A. Cost ceiling sweep (λ=0): 7 thresholds progressively admitting more models
  B. Latency ceiling sweep (λ=0): 5 thresholds
  C. Production deployment scenarios: 5 combined (cost + latency) profiles
  D. Constrained Pareto frontiers: 3 constraint regimes × λ sweep

Output:
  results/constraint_impact_results.json
"""

import sys
import json
import gzip
import time
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    K3_WARMUP_PRIORS_PATH,
    K3_MODELS_PATH,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from bandit_gpt.router import BanditRouter
from utils.router_factory import create_experiment_router
from utils.rewards import extract_reward
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PORTFOLIO (K=3) — extended with latency metadata
# ============================================================================

from utils.model_pricing import load_model_catalog as _load_catalog

K3_MODELS, _K3_CAT_BASE = _load_catalog(K3_MODELS_PATH)

_LATENCY_LOOKUP = {
    "meta-llama/llama-3.1-8b-instruct": 0.35,
    "google/gemini-2.5-flash": 0.30,
    "openai/gpt-4.1": 0.65,
}

MODEL_CATALOG: Dict[str, Dict] = {}
for _mid in K3_MODELS:
    _base = _K3_CAT_BASE[_mid]
    MODEL_CATALOG[_mid] = {
        **_base,
        "time_to_first_token_seconds": _LATENCY_LOOKUP.get(_mid, 0.50),
    }

# Standard request size for cost estimation (matches 04_figure convention)
INPUT_TOKENS = 100
OUTPUT_TOKENS = 400

# ============================================================================
# CONSTRAINT CONFIGURATIONS
# ============================================================================

# Cost ceiling sweep: progressively admits more models
# Thresholds chosen at natural breakpoints in the cost distribution
COST_CEILINGS = [0.0001, 0.0003, 0.0005, 0.002, 0.005, 0.01, None]

# Latency ceiling sweep (seconds)
LATENCY_CEILINGS = [0.40, 0.50, 0.70, 1.00, None]

# Production deployment scenarios: (name, max_cost, max_latency, description)
PRODUCTION_SCENARIOS = [
    ("Unconstrained",       None,    None, "No constraints (baseline)"),
    ("Budget-Micro",        0.0003,  None, "Free-tier startup, cost-only"),
    ("Budget-Mid",          0.002,   None, "Mid-tier SaaS budget"),
    ("Latency-Strict",      None,    0.50, "Real-time chatbot, latency-only"),
    ("Enterprise-SLA",      0.005,   0.70, "Balanced enterprise deployment"),
    ("Premium-SLA",         0.01,    1.00, "Quality-focused with safety rails"),
]

# Constrained Pareto: sweep λ under different constraint regimes
PARETO_CONSTRAINT_REGIMES = [
    ("Tight ($c \\leq$ \\$0.5m)", 0.0005, None),
    ("Moderate ($c \\leq$ \\$5m)", 0.005, None),
    ("Unconstrained", None, None),
]
PARETO_LAMBDA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

N_TRIALS = 50
SEED_OFFSET = 42
TARGET_NEFF = 10.0
ALPHA_START = 2.0
CORRALLING_LR = 0.1
CORRALLING_GAMMA = 0.05

# ============================================================================
# DATA LOADING (shared with 04_figure)
# ============================================================================

def _entry_reward(entry: Dict) -> float:
    """Delegate to canonical reward extraction (``experiments/utils/rewards.py``)."""
    return extract_reward(entry)


def load_rewards(data_path: Path, prompts: List[str], models: List[str]) -> List[Dict]:
    prompt_set = set(prompts)
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            p, m = entry["prompt"], entry["model_id"]
            if p in prompt_set and m in model_set:
                rewards[p][m] = _entry_reward(entry)
    return [
        {"prompt": p, "rewards": rewards[p]}
        for p in prompts
        if p in rewards and len(rewards[p]) == len(models)
    ]


def load_all_holdout_rewards(data_path: Path, models: List[str]) -> List[Dict]:
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok") and entry["model_id"] in model_set:
                rewards[entry["prompt"]][entry["model_id"]] = _entry_reward(entry)
    K = len(models)
    return [
        {"prompt": p, "rewards": r}
        for p, r in rewards.items()
        if len(r) == K
    ]


# ============================================================================
# HELPERS
# ============================================================================

def build_model_registry(models: List[str]) -> Dict[str, Dict]:
    """Registry including cost AND latency for constraint filtering."""
    return {
        m: {
            "input_cost_per_m": MODEL_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": MODEL_CATALOG[m]["output_cost_per_m"],
            "time_to_first_token_seconds": MODEL_CATALOG[m]["time_to_first_token_seconds"],
        }
        for m in models
    }


def eligible_models_via_router(
    models: List[str],
    max_cost: Optional[float],
    max_latency: Optional[float],
) -> List[str]:
    """Determine eligible models by calling the library's own filter.

    Instantiates a throwaway BanditRouter and delegates to
    ``_filter_by_constraints`` so the eligibility logic lives in
    exactly one place—the production code path.
    """
    from bandit_gpt.feature_service import FeatureService
    from bandit_gpt.storage import EphemeralContextStore

    registry = build_model_registry(models)
    fs = FeatureService.for_precomputed(33)
    store = EphemeralContextStore()
    tmp_router = BanditRouter(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
    )
    return tmp_router._filter_by_constraints(
        candidates=list(registry.keys()),
        prompt=np.zeros(33),
        max_cost=max_cost,
        max_latency=max_latency,
        quality_floor=None,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )


# ============================================================================
# CORE EVALUATION
# ============================================================================

def run_constrained_trial(
    models: List[str],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: str,
    max_cost: Optional[float],
    max_latency: Optional[float],
    cost_penalty: float,
    seed: int,
    train_unconstrained: bool = False,
) -> Dict:
    """Single trial: train → evaluate with constraints.

    Parameters
    ----------
    train_unconstrained:
        If True, online learning runs WITHOUT constraints (full portfolio),
        but holdout evaluation still applies constraints.  Used by the
        ablation that separates "restricted exploration" from "restricted
        evaluation."
    """
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)

    all_raw = [p["rewards"][m] for p in train_data for m in models]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = max(r_max - r_min, 1e-6)

    np.random.seed(seed)
    router = create_experiment_router(
        model_registry=build_model_registry(models),
        feature_dim=dim,
        prior_n_effective=TARGET_NEFF,
        alpha=ALPHA_START,
        warmup_path=warmup_path,
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=cost_penalty,
    )

    # During training: respect constraints unless ablation flag is set
    train_mc = None if train_unconstrained else max_cost
    train_ml = None if train_unconstrained else max_latency

    for p, x in zip(train_data, train_emb):
        m, log = router.route(
            x, max_cost=train_mc, max_latency=train_ml,
            input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
            total_steps=burn_in,
        )
        norm_r = (p["rewards"][m] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

    # Holdout evaluation always applies constraints.
    # Violation checking uses RoutingLog fields (cost_usd, latency_s)
    # computed by the library's _estimate_cost / _estimate_latency.
    rewards, realized_costs, latencies = [], [], []
    violations_cost, violations_latency = 0, 0
    model_counts = {m: 0 for m in models}

    for p, x in zip(eval_data, eval_emb):
        m, log = router.route(
            x, max_cost=max_cost, max_latency=max_latency,
            input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
            total_steps=burn_in,
        )
        rewards.append(p["rewards"][m])
        realized_costs.append(log.cost_usd)
        latencies.append(log.latency_s)
        model_counts[m] += 1

        if max_cost is not None and log.cost_usd > max_cost:
            violations_cost += 1
        if max_latency is not None and log.latency_s > max_latency:
            violations_latency += 1

    n = len(eval_data)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_cost": float(np.mean(realized_costs)),
        "mean_latency": float(np.mean(latencies)),
        "violations_cost": violations_cost,
        "violations_latency": violations_latency,
        "violations_total": violations_cost + violations_latency,
        "n_eval": n,
        "model_counts": model_counts,
        "per_trial_reward": float(np.mean(rewards)),
    }


def run_sweep(
    models, train_data, eval_data, train_emb, eval_emb,
    warmup_path, max_cost, max_latency, cost_penalty,
    n_trials, label, train_unconstrained=False,
):
    """Run n_trials of a single constraint configuration."""
    trial_results = []
    for trial in range(n_trials):
        r = run_constrained_trial(
            models, train_data, eval_data, train_emb, eval_emb,
            warmup_path, max_cost, max_latency, cost_penalty,
            seed=SEED_OFFSET + trial,
            train_unconstrained=train_unconstrained,
        )
        trial_results.append(r)

    rewards = [t["mean_reward"] for t in trial_results]
    costs_r = [t["mean_cost"] for t in trial_results]
    latencies = [t["mean_latency"] for t in trial_results]
    total_violations = sum(t["violations_total"] for t in trial_results)
    total_evals = sum(t["n_eval"] for t in trial_results)

    elig = eligible_models_via_router(models, max_cost, max_latency)

    return {
        "label": label,
        "max_cost": max_cost,
        "max_latency": max_latency,
        "cost_penalty": cost_penalty,
        "eligible_K": len(elig),
        "eligible_models": [MODEL_CATALOG[m]["display"] for m in elig],
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if n_trials > 1 else 0.0,
        "ci95_reward": float(1.96 * np.std(rewards, ddof=1) / np.sqrt(n_trials)) if n_trials > 1 else 0.0,
        "mean_cost": float(np.mean(costs_r)),
        "std_cost": float(np.std(costs_r, ddof=1)) if n_trials > 1 else 0.0,
        "mean_latency": float(np.mean(latencies)),
        "violation_rate": total_violations / total_evals if total_evals > 0 else 0.0,
        "total_violations": total_violations,
        "total_evals": total_evals,
        "n_trials": n_trials,
        "per_trial_rewards": rewards,
    }


# ============================================================================
# BASELINES (all respect constraints for fair comparison)
# ============================================================================

def constrained_baselines(eval_data, models, costs,
                          max_cost=None, max_latency=None,
                          n_random_trials=20, seed_offset=SEED_OFFSET):
    """Oracle, best-static, and random baselines restricted to eligible models.

    Returns a dict with keys: oracle, best_static, random — each containing
    reward and cost statistics.  All baselines respect the same hard
    constraints so comparisons with the bandit are fair.
    """
    elig = eligible_models_via_router(models, max_cost, max_latency)
    if not elig:
        elig = models

    # Oracle: per-prompt best eligible model
    oracle_r = oracle_c = 0.0
    for p in eval_data:
        best_m = max(elig, key=lambda m: p["rewards"][m])
        oracle_r += p["rewards"][best_m]
        oracle_c += costs[best_m]
    n = len(eval_data)
    oracle_r /= n
    oracle_c /= n

    # Best static: single eligible model with highest mean reward
    static_results = {}
    for m in elig:
        sr = np.mean([p["rewards"][m] for p in eval_data])
        static_results[m] = {"reward": float(sr), "cost": costs[m]}
    best_static_m = max(static_results, key=lambda m: static_results[m]["reward"])

    # Random: uniform over eligible models
    trial_r, trial_c = [], []
    for t in range(n_random_trials):
        rng = np.random.RandomState(seed_offset + t)
        r_sum = c_sum = 0.0
        for p in eval_data:
            m = elig[rng.randint(len(elig))]
            r_sum += p["rewards"][m]
            c_sum += costs[m]
        trial_r.append(r_sum / n)
        trial_c.append(c_sum / n)

    return {
        "oracle": {"reward": oracle_r, "cost": oracle_c},
        "best_static": {
            "model": best_static_m,
            "display": MODEL_CATALOG[best_static_m]["display"],
            "reward": static_results[best_static_m]["reward"],
            "cost": static_results[best_static_m]["cost"],
        },
        "random": {
            "reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)),
            "cost": float(np.mean(trial_c)),
        },
    }


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

def paired_wilcoxon(rewards_a, rewards_b):
    """Wilcoxon signed-rank test on paired trial-level mean rewards."""
    from scipy.stats import wilcoxon
    diff = np.array(rewards_a) - np.array(rewards_b)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}
    stat, p = wilcoxon(diff)
    d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    return {"statistic": float(stat), "p_value": float(p), "effect_size": d}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Hard Constraint Impact Experiment (K=3)")
    logger.info("=" * 70)

    # --- Load data ----------------------------------------------------------
    logger.info("\n1. Loading splits and encoder ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits = json.load(f)
    online_prompts = splits["online_learn_pool"]

    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    from utils.embeddings import load_embedding_cache, embed_dataset_cached
    _emb_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    models = K3_MODELS
    costs = {m: MODEL_CATALOG[m]["cost"] for m in models}

    logger.info(f"\n  Portfolio ({len(models)} models):")
    for m in sorted(models, key=lambda m: costs[m]):
        c = MODEL_CATALOG[m]
        logger.info(f"    {c['display']:<22} cost=${costs[m]:.6f}  TTFT={c['time_to_first_token_seconds']:.2f}s")

    logger.info(f"\n2. Loading rewards ...")
    train_data = load_rewards(DEV_DATA_PATH_ALL_MODELS, online_prompts, models)
    eval_data = load_all_holdout_rewards(HOLDOUT_DATA_PATH_ALL_MODELS, models)
    logger.info(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    logger.info("\n3. Embedding prompts ...")
    train_emb = embed_dataset_cached(train_data, _emb_cache, encoder, pca)
    eval_emb = embed_dataset_cached(eval_data, _emb_cache, encoder, pca)
    logger.info(f"  Dimension: {train_emb[0].shape[0]}")

    warmup_path = str(K3_WARMUP_PRIORS_PATH)
    results = {}

    # --- A: Cost ceiling sweep ---------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("A. Cost Ceiling Sweep (λ=0)")
    logger.info("=" * 70)

    cost_sweep = []
    for mc in COST_CEILINGS:
        elig = eligible_models_via_router(models, mc, None)
        label = f"$c ≤ ${mc}" if mc else "Unconstrained"
        logger.info(f"\n  {label} → K'={len(elig)}: {[MODEL_CATALOG[m]['display'] for m in elig]}")

        r = run_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            warmup_path, max_cost=mc, max_latency=None,
            cost_penalty=0.0, n_trials=N_TRIALS, label=label,
        )
        baselines = constrained_baselines(eval_data, models, costs, mc, None)
        r["baselines"] = baselines
        cost_sweep.append(r)
        logger.info(f"    banditGPT R={r['mean_reward']:.4f}±{r['std_reward']:.4f}  "
                     f"best-static R={baselines['best_static']['reward']:.4f} "
                     f"({baselines['best_static']['display']})  "
                     f"violations={r['total_violations']}")

    results["cost_sweep"] = cost_sweep

    # --- B: Latency ceiling sweep ------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("B. Latency Ceiling Sweep (λ=0)")
    logger.info("=" * 70)

    latency_sweep = []
    for ml in LATENCY_CEILINGS:
        elig = eligible_models_via_router(models, None, ml)
        label = f"TTFT ≤ {ml}s" if ml else "Unconstrained"
        logger.info(f"\n  {label} → K'={len(elig)}: {[MODEL_CATALOG[m]['display'] for m in elig]}")

        r = run_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            warmup_path, max_cost=None, max_latency=ml,
            cost_penalty=0.0, n_trials=N_TRIALS, label=label,
        )
        baselines = constrained_baselines(eval_data, models, costs, None, ml)
        r["baselines"] = baselines
        latency_sweep.append(r)
        logger.info(f"    banditGPT R={r['mean_reward']:.4f}±{r['std_reward']:.4f}  "
                     f"best-static R={baselines['best_static']['reward']:.4f}  "
                     f"violations={r['total_violations']}")

    results["latency_sweep"] = latency_sweep

    # --- C: Production scenarios (combined constraints only) ----------------
    # Single-dimension scenarios (cost-only, latency-only) are already
    # covered by the sweeps above.  Only run combined-constraint scenarios
    # and the unconstrained baseline here.
    logger.info(f"\n{'='*70}")
    logger.info("C. Production Deployment Scenarios (combined constraints)")
    logger.info("=" * 70)

    scenarios = []
    for name, mc, ml, desc in PRODUCTION_SCENARIOS:
        elig = eligible_models_via_router(models, mc, ml)
        logger.info(f"\n  {name}: {desc}")
        logger.info(f"    cost≤{mc}, latency≤{ml} → K'={len(elig)}")

        r = run_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            warmup_path, max_cost=mc, max_latency=ml,
            cost_penalty=0.0, n_trials=N_TRIALS, label=name,
        )
        baselines = constrained_baselines(eval_data, models, costs, mc, ml)
        r["baselines"] = baselines
        r["description"] = desc
        scenarios.append(r)
        logger.info(f"    banditGPT R={r['mean_reward']:.4f}±{r['std_reward']:.4f}  "
                     f"best-static R={baselines['best_static']['reward']:.4f}  "
                     f"violations={r['total_violations']}")

    results["production_scenarios"] = scenarios

    # --- D: Constrained Pareto frontiers -----------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("D. Constrained Pareto Frontiers (λ sweep)")
    logger.info("=" * 70)

    pareto = {}
    for regime_name, mc, ml in PARETO_CONSTRAINT_REGIMES:
        elig = eligible_models_via_router(models, mc, ml)
        logger.info(f"\n  Regime: {regime_name} → K'={len(elig)}")
        regime_points = []

        for lam in PARETO_LAMBDA_VALUES:
            r = run_sweep(
                models, train_data, eval_data, train_emb, eval_emb,
                warmup_path, max_cost=mc, max_latency=ml,
                cost_penalty=lam, n_trials=N_TRIALS, label=f"λ={lam}",
            )
            regime_points.append(r)
            logger.info(f"    λ={lam:<5} R={r['mean_reward']:.4f}±{r['std_reward']:.4f}  "
                         f"C=${r['mean_cost']:.6f}")

        pareto[regime_name] = {
            "max_cost": mc,
            "max_latency": ml,
            "eligible_K": len(elig),
            "points": regime_points,
        }

    results["constrained_pareto"] = pareto

    # --- E: Ablation — train unconstrained, eval constrained ---------------
    # Separates the quality effect of "restricted model set at eval time"
    # from "restricted exploration during training."
    logger.info(f"\n{'='*70}")
    logger.info("E. Ablation: Train Unconstrained → Eval Constrained")
    logger.info("=" * 70)

    ablation_cost = 0.0005
    ablation_results = {}
    elig = eligible_models_via_router(models, ablation_cost, None)
    logger.info(f"  Constraint: cost ≤ ${ablation_cost}  → K'={len(elig)}")

    # (a) Always-on: constraints during both train + eval (standard)
    always_on = run_sweep(
        models, train_data, eval_data, train_emb, eval_emb,
        warmup_path, max_cost=ablation_cost, max_latency=None,
        cost_penalty=0.0, n_trials=N_TRIALS, label="always-on",
    )
    # (b) Eval-only: unconstrained training, constrained evaluation
    eval_only = run_sweep(
        models, train_data, eval_data, train_emb, eval_emb,
        warmup_path, max_cost=ablation_cost, max_latency=None,
        cost_penalty=0.0, n_trials=N_TRIALS, label="eval-only",
        train_unconstrained=True,
    )

    stat_test = paired_wilcoxon(
        always_on["per_trial_rewards"], eval_only["per_trial_rewards"]
    )
    ablation_results = {
        "constraint": {"max_cost": ablation_cost, "eligible_K": len(elig)},
        "always_on": always_on,
        "eval_only": eval_only,
        "paired_test": stat_test,
    }
    logger.info(f"  Always-on:  R={always_on['mean_reward']:.4f}±{always_on['std_reward']:.4f}")
    logger.info(f"  Eval-only:  R={eval_only['mean_reward']:.4f}±{eval_only['std_reward']:.4f}")
    logger.info(f"  Wilcoxon p={stat_test['p_value']:.4f}  d={stat_test['effect_size']:.3f}")

    results["ablation_train_unconstrained"] = ablation_results

    # --- Metadata -----------------------------------------------------------
    results["metadata"] = {
        "K": len(models),
        "n_trials": N_TRIALS,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "models": [
            {
                "id": m,
                "display": MODEL_CATALOG[m]["display"],
                "cost_per_request": costs[m],
                "ttft_seconds": MODEL_CATALOG[m]["time_to_first_token_seconds"],
                "tier": MODEL_CATALOG[m]["tier"],
            }
            for m in sorted(models, key=lambda m: costs[m])
        ],
    }

    # --- Save ---------------------------------------------------------------
    out_path = Path(__file__).parent / "results" / "constraint_impact_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")
    logger.info("Done.")


if __name__ == "__main__":
    main()
