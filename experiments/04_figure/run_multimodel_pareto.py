#!/usr/bin/env python3
"""
Multi-Model Pareto Frontier: K=5 and K=10
==========================================

Primary experiment demonstrating banditGPT with multi-model portfolios (K > 2).

Protocol (mirrors 03_figure for direct comparison):
  1. Load three-way split (prior-train already consumed by warmup priors)
  2. Pre-compute embeddings for online-learn and holdout prompts
  3. For each (K, λ) configuration × N_TRIALS seeds:
       a. Instantiate router with 43-model warmup priors
       b. Train on online-learn set (533 prompts)
       c. Freeze; evaluate on holdout set (750 prompts)
  4. Compute baselines: oracle, best-static, random, ε-greedy, tabula rasa
  5. Produce JSON results consumed by the figure generator

Portfolios:
  K=5:  Llama-3.1-8B, Mixtral-8x7B, Gemini-2.5-Flash, Claude-Sonnet-4, GPT-4.1
  K=10: Above + Llama-4-Maverick, Gemma-3-27B, Claude-Haiku-4.5, GPT-4-Turbo, DeepSeek-V3

Output:
  results/multimodel_pareto_results.json
"""

import sys
import json
import gzip
import copy
import time
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from utils.router_factory import create_experiment_router
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PORTFOLIOS
# ============================================================================

# Per-million-token pricing from OpenRouter (Feb 2026).
# The router uses input_cost_per_m and output_cost_per_m internally to compute
# normalized cost penalties; the "cost" field is the per-request cost
# (100 input + 400 output tokens) used for the Pareto frontier x-axis.
def _req_cost(inp, out):
    return (100 * inp + 400 * out) / 1_000_000

MODEL_CATALOG = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        "input_cost_per_m": 0.05, "output_cost_per_m": 0.05,
        "cost": _req_cost(0.05, 0.05),
        "tier": "cheap", "provider": "meta",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        "input_cost_per_m": 0.54, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.54, 0.60),
        "tier": "cheap", "provider": "mistral",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        "input_cost_per_m": 0.10, "output_cost_per_m": 0.10,
        "cost": _req_cost(0.10, 0.10),
        "tier": "cheap", "provider": "google",
    },
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        "input_cost_per_m": 0.80, "output_cost_per_m": 4.00,
        "cost": _req_cost(0.80, 4.00),
        "tier": "mid", "provider": "anthropic",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        "input_cost_per_m": 0.27, "output_cost_per_m": 1.10,
        "cost": _req_cost(0.27, 1.10),
        "tier": "mid", "provider": "deepseek",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        "input_cost_per_m": 0.15, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.15, 0.60),
        "tier": "mid", "provider": "google",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        "input_cost_per_m": 0.20, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.20, 0.60),
        "tier": "mid", "provider": "meta",
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        "input_cost_per_m": 3.00, "output_cost_per_m": 15.00,
        "cost": _req_cost(3.00, 15.00),
        "tier": "expensive", "provider": "anthropic",
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        "input_cost_per_m": 10.00, "output_cost_per_m": 30.00,
        "cost": _req_cost(10.00, 30.00),
        "tier": "expensive", "provider": "openai",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        "input_cost_per_m": 2.00, "output_cost_per_m": 8.00,
        "cost": _req_cost(2.00, 8.00),
        "tier": "expensive", "provider": "openai",
    },
}

PORTFOLIO_K5 = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "google/gemini-2.5-flash-preview-09-2025",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]

PORTFOLIO_K10 = PORTFOLIO_K5 + [
    "meta-llama/llama-4-maverick",
    "google/gemma-3-27b-it",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4-turbo",
    "deepseek/deepseek-chat-v3-0324",
]

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

N_TRIALS = 20
SEED_OFFSET = 42
TARGET_NEFF = 10.0
ALPHA_START = 2.0
CORRALLING_LR = 0.1
CORRALLING_GAMMA = 0.05

LAMBDA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

LEARNING_CURVE_CHECKPOINTS = [0, 10, 25, 50, 100, 150, 200, 300, 400, 533]


# ============================================================================
# DATA LOADING
# ============================================================================

def _entry_reward(entry: Dict) -> float:
    """Extract reward from a data entry using mean judge agreement.

    Uses the average of individual judge votes rather than the binarised
    majority vote (``raw_score``).  This preserves evaluative signal from
    the multi-judge panel — e.g. a 2-out-of-3 pass (0.667) is
    distinguished from a unanimous pass (1.0).
    """
    judges = entry.get("judge_details")
    if judges:
        return float(np.mean([j["vote"] for j in judges]))
    return float(entry["raw_score"])


def load_rewards(data_path: Path, prompts: List[str], models: List[str],
                 ) -> List[Dict]:
    """Load rewards for specific prompts and models from gzipped JSONL."""
    prompt_set = set(prompts)
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            p = entry["prompt"]
            m = entry["model_id"]
            if p in prompt_set and m in model_set:
                rewards[p][m] = _entry_reward(entry)

    data = []
    for p in prompts:
        if p in rewards and len(rewards[p]) == len(models):
            data.append({"prompt": p, "rewards": rewards[p]})
    return data


# ============================================================================
# BASELINES
# ============================================================================

def oracle_route(eval_data: List[Dict], models: List[str],
                 costs: Dict[str, float]) -> Tuple[float, float]:
    r_total = c_total = 0.0
    for p in eval_data:
        best_m = max(models, key=lambda m: p["rewards"][m])
        r_total += p["rewards"][best_m]
        c_total += costs[best_m]
    n = len(eval_data)
    return r_total / n, c_total / n


def static_route(eval_data: List[Dict], model: str,
                 costs: Dict[str, float]) -> Tuple[float, float]:
    r_total = sum(p["rewards"][model] for p in eval_data)
    c_total = costs[model] * len(eval_data)
    n = len(eval_data)
    return r_total / n, c_total / n


def random_route(eval_data: List[Dict], models: List[str],
                 costs: Dict[str, float],
                 n_trials: int = 20, seed_offset: int = SEED_OFFSET,
                 ) -> Dict[str, float]:
    """Random routing averaged over *n_trials* seeds."""
    trial_r, trial_c = [], []
    for trial in range(n_trials):
        rng = np.random.RandomState(seed_offset + trial)
        r_total = c_total = 0.0
        for p in eval_data:
            m = models[rng.randint(len(models))]
            r_total += p["rewards"][m]
            c_total += costs[m]
        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)
    return {
        "reward": float(np.mean(trial_r)),
        "std_reward": float(np.std(trial_r, ddof=1)),
        "cost": float(np.mean(trial_c)),
        "std_cost": float(np.std(trial_c, ddof=1)),
        "n_trials": n_trials,
    }


def epsilon_greedy_route(train_data, eval_data, models, costs,
                         epsilon=0.1,
                         n_trials: int = 20, seed_offset: int = SEED_OFFSET,
                         ) -> Dict[str, float]:
    """ε-greedy averaged over *n_trials* seeds."""
    model_rewards = {m: [] for m in models}
    for p in train_data:
        for m in models:
            model_rewards[m].append(p["rewards"][m])
    model_means = {m: np.mean(v) for m, v in model_rewards.items()}
    best_model = max(model_means, key=model_means.get)

    trial_r, trial_c = [], []
    for trial in range(n_trials):
        rng = np.random.RandomState(seed_offset + trial)
        r_total = c_total = 0.0
        for p in eval_data:
            if rng.random() < epsilon:
                m = models[rng.randint(len(models))]
            else:
                m = best_model
            r_total += p["rewards"][m]
            c_total += costs[m]
        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)
    return {
        "reward": float(np.mean(trial_r)),
        "std_reward": float(np.std(trial_r, ddof=1)),
        "cost": float(np.mean(trial_c)),
        "std_cost": float(np.std(trial_c, ddof=1)),
        "n_trials": n_trials,
    }


# ============================================================================
# BANDIT ROUTING
# ============================================================================

def build_model_registry(models: List[str]) -> Dict[str, Dict]:
    """Build model registry in the format BanditRouter.__init__ expects."""
    return {
        m: {
            "input_cost_per_m": MODEL_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": MODEL_CATALOG[m]["output_cost_per_m"],
        }
        for m in models
    }


def evaluate_frozen(router, eval_data, eval_embeddings, costs, total_steps):
    """Frozen holdout evaluation — no learning."""
    rng_state = np.random.get_state()
    r_total = c_total = 0.0
    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=total_steps)
        r_total += p["rewards"][model]
        c_total += costs[model]
    np.random.set_state(rng_state)
    n = len(eval_data)
    return r_total / n, c_total / n


def run_pareto_sweep(
    models, train_data, eval_data, train_emb, eval_emb,
    warmup_path, costs, lambda_values, n_trials,
    use_corralling=True, label="banditGPT",
):
    """Sweep λ with N_TRIALS seeds → list of (λ, mean_r, std_r, mean_c, std_c)."""
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)

    all_raw = [s for p in train_data for m in models for s in [p["rewards"][m]]]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)
            router = create_experiment_router(
                model_registry=build_model_registry(models),
                feature_dim=dim,
                prior_n_effective=TARGET_NEFF,
                alpha=ALPHA_START,
                warmup_path=str(warmup_path),
                use_corralling=use_corralling,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
            )
            # Train
            for p, x in zip(train_data, train_emb):
                m, log = router.route(x, total_steps=burn_in)
                norm_r = (p["rewards"][m] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)
            # Eval
            r, c = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            trial_r.append(r)
            trial_c.append(c)

        results.append({
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "n_trials": n_trials,
            "label": label,
        })
        logger.info(f"    λ={lam:<5} | R={np.mean(trial_r):.4f}±{np.std(trial_r):.4f} "
                     f"| C=${np.mean(trial_c):.6f}")
    return results


def run_learning_curve(
    models, train_data, eval_data, train_emb, eval_emb,
    warmup_path, costs, n_trials, checkpoints,
    use_corralling=True, label="banditGPT",
):
    """Learning curve: quality vs online learning steps."""
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)

    all_raw = [s for p in train_data for m in models for s in [p["rewards"][m]]]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    checkpoint_set = set(checkpoints)
    by_step = {s: {"rewards": [], "costs": []} for s in checkpoints}

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=build_model_registry(models),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=ALPHA_START,
            warmup_path=str(warmup_path),
            use_corralling=use_corralling,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=0.0,
        )
        if 0 in checkpoint_set:
            r, c = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            by_step[0]["rewards"].append(r)
            by_step[0]["costs"].append(c)

        for step_idx, (p, x) in enumerate(zip(train_data, train_emb)):
            m, log = router.route(x, total_steps=burn_in)
            norm_r = (p["rewards"][m] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)
            current = step_idx + 1
            if current in checkpoint_set:
                r, c = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
                by_step[current]["rewards"].append(r)
                by_step[current]["costs"].append(c)

        if (trial + 1) % 5 == 0:
            logger.info(f"    Trial {trial+1}/{n_trials}")

    curve = []
    for s in sorted(checkpoints):
        rr = by_step[s]["rewards"]
        if rr:
            curve.append({
                "step": s,
                "mean_reward": float(np.mean(rr)),
                "std_reward": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
                "n_trials": len(rr),
                "label": label,
            })
    return curve


def run_traffic_allocation(
    models, train_data, eval_data, train_emb, eval_emb,
    warmup_path, costs, n_trials=20,
):
    """Track per-model selection fractions on holdout after training."""
    dim = train_emb[0].shape[0]
    burn_in = len(train_data)

    all_raw = [s for p in train_data for m in models for s in [p["rewards"][m]]]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    all_counts = {m: [] for m in models}

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=build_model_registry(models),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=ALPHA_START,
            warmup_path=str(warmup_path),
            use_corralling=True,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=0.0,
        )
        for p, x in zip(train_data, train_emb):
            m, log = router.route(x, total_steps=burn_in)
            norm_r = (p["rewards"][m] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)

        counts = {m: 0 for m in models}
        rng_state = np.random.get_state()
        for p, x in zip(eval_data, eval_emb):
            m, _log = router.route(x, total_steps=burn_in)
            counts[m] += 1
        np.random.set_state(rng_state)

        for m in models:
            all_counts[m].append(counts[m] / len(eval_data))

    return {
        m: {
            "mean_frac": float(np.mean(all_counts[m])),
            "std_frac": float(np.std(all_counts[m])),
            "display": MODEL_CATALOG[m]["display"],
        }
        for m in models
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("Multi-Model Pareto Frontier: K=5 and K=10")
    logger.info("=" * 70)

    # --- Load splits -------------------------------------------------------
    logger.info("\n1. Loading three-way split ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits = json.load(f)
    online_prompts = splits["online_learn_pool"]
    logger.info(f"  Online learning: {len(online_prompts)} prompts")

    # --- Load encoder / PCA ------------------------------------------------
    logger.info("\n2. Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components")

    results_all = {}

    for portfolio_name, models in [("K5", PORTFOLIO_K5), ("K10", PORTFOLIO_K10)]:
        K = len(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"PORTFOLIO: {portfolio_name} ({K} models)")
        logger.info("=" * 70)
        for m in models:
            logger.info(f"  {MODEL_CATALOG[m]['display']:<25} "
                        f"cost=${MODEL_CATALOG[m]['cost']:.6f}")

        costs = {m: MODEL_CATALOG[m]["cost"] for m in models}

        # --- Load rewards --------------------------------------------------
        logger.info(f"\n  Loading rewards for {K} models ...")
        train_data = load_rewards(DEV_DATA_PATH_ALL_MODELS, online_prompts, models)
        holdout_prompts_raw = load_rewards(
            HOLDOUT_DATA_PATH_ALL_MODELS,
            # load ALL holdout prompts (we don't have the list, load everything)
            [],  # empty → need different approach
            models,
        )
        # Reload holdout properly: load all prompts from holdout file
        holdout_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
        model_set = set(models)
        with gzip.open(HOLDOUT_DATA_PATH_ALL_MODELS, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok") and entry["model_id"] in model_set:
                    holdout_rewards[entry["prompt"]][entry["model_id"]] = _entry_reward(entry)
        eval_data = [
            {"prompt": p, "rewards": r}
            for p, r in holdout_rewards.items()
            if len(r) == K
        ]

        logger.info(f"  Train: {len(train_data)} prompts | Eval: {len(eval_data)} prompts")

        # --- Embeddings ----------------------------------------------------
        logger.info("  Embedding prompts ...")
        train_emb = [
            embed_prompt(p["prompt"], encoder, pca) for p in train_data
        ]
        eval_emb = [
            embed_prompt(p["prompt"], encoder, pca) for p in eval_data
        ]
        logger.info(f"  Embedding dim: {train_emb[0].shape[0]}")

        # --- Baselines -----------------------------------------------------
        logger.info("\n  Computing baselines ...")
        oracle_r, oracle_c = oracle_route(eval_data, models, costs)
        logger.info(f"    Oracle:  R={oracle_r:.4f}  C=${oracle_c:.6f}")

        static_results = {}
        for m in models:
            sr, sc = static_route(eval_data, m, costs)
            static_results[m] = {"reward": sr, "cost": sc}
            logger.info(f"    Static {MODEL_CATALOG[m]['display']:<20}: "
                        f"R={sr:.4f}  C=${sc:.6f}")

        rand = random_route(eval_data, models, costs, n_trials=N_TRIALS)
        logger.info(f"    Random ({N_TRIALS} seeds):  R={rand['reward']:.4f}±{rand['std_reward']:.4f}  C=${rand['cost']:.6f}")

        eg = epsilon_greedy_route(train_data, eval_data, models, costs, n_trials=N_TRIALS)
        logger.info(f"    ε-Greedy ({N_TRIALS} seeds):  R={eg['reward']:.4f}±{eg['std_reward']:.4f}  C=${eg['cost']:.6f}")

        # --- banditGPT (Corralling + priors) --------------------------------
        logger.info(f"\n  banditGPT Pareto sweep ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pareto_full = run_pareto_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            MULTIMODEL_WARMUP_PRIORS_PATH, costs, LAMBDA_VALUES, N_TRIALS,
            use_corralling=True, label="banditGPT",
        )

        # --- Tabula rasa ablation (no priors, no Corralling) ----------------
        logger.info(f"\n  Tabula rasa ablation ({len(LAMBDA_VALUES)} λ × {N_TRIALS} trials) ...")
        pareto_tabula = run_pareto_sweep(
            models, train_data, eval_data, train_emb, eval_emb,
            MULTIMODEL_WARMUP_PRIORS_PATH, costs, LAMBDA_VALUES, N_TRIALS,
            use_corralling=False, label="tabula_rasa",
        )

        # --- Learning curve -------------------------------------------------
        logger.info(f"\n  Learning curve ({N_TRIALS} trials) ...")
        curve = run_learning_curve(
            models, train_data, eval_data, train_emb, eval_emb,
            MULTIMODEL_WARMUP_PRIORS_PATH, costs, N_TRIALS,
            LEARNING_CURVE_CHECKPOINTS,
        )

        # --- Traffic allocation ---------------------------------------------
        logger.info(f"\n  Traffic allocation ({N_TRIALS} trials) ...")
        traffic = run_traffic_allocation(
            models, train_data, eval_data, train_emb, eval_emb,
            MULTIMODEL_WARMUP_PRIORS_PATH, costs,
        )

        # --- Assemble -------------------------------------------------------
        best_static_m = max(static_results, key=lambda m: static_results[m]["reward"])
        weak_r = min(static_results[m]["reward"] for m in models)
        peak_bandit = max(pareto_full, key=lambda x: x["mean_reward"])
        gap_closure = (
            (peak_bandit["mean_reward"] - weak_r) / (oracle_r - weak_r) * 100
            if oracle_r > weak_r else 0.0
        )

        results_all[portfolio_name] = {
            "K": K,
            "models": [{"id": m, **MODEL_CATALOG[m]} for m in models],
            "oracle": {"reward": oracle_r, "cost": oracle_c},
            "random": rand,
            "epsilon_greedy": eg,
            "static": static_results,
            "best_static": {
                "model": best_static_m,
                "reward": static_results[best_static_m]["reward"],
                "cost": static_results[best_static_m]["cost"],
            },
            "pareto_banditgpt": pareto_full,
            "pareto_tabula_rasa": pareto_tabula,
            "learning_curve": curve,
            "traffic_allocation": traffic,
            "gap_closure_pct": gap_closure,
            "peak_bandit_reward": peak_bandit["mean_reward"],
            "n_train": len(train_data),
            "n_eval": len(eval_data),
            "n_trials": N_TRIALS,
        }

        logger.info(f"\n  SUMMARY ({portfolio_name}):")
        logger.info(f"    Oracle:           {oracle_r:.4f}")
        logger.info(f"    banditGPT peak:   {peak_bandit['mean_reward']:.4f} "
                     f"± {peak_bandit['std_reward']:.4f}")
        logger.info(f"    Best static:      {static_results[best_static_m]['reward']:.4f} "
                     f"({MODEL_CATALOG[best_static_m]['display']})")
        logger.info(f"    ε-Greedy:         {eg['reward']:.4f} ± {eg['std_reward']:.4f}")
        logger.info(f"    Random:           {rand['reward']:.4f} ± {rand['std_reward']:.4f}")
        logger.info(f"    Gap closure:      {gap_closure:.1f}%")

    # --- Save results ------------------------------------------------------
    out_path = Path(__file__).parent / "results" / "multimodel_pareto_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
