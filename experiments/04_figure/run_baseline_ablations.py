#!/usr/bin/env python3
"""
Architectural Ablation Study: Isolating Each Component of banditGPT
====================================================================

Tests progressively richer routing strategies to quantify the marginal
contribution of each architectural component:

    1. Random               – No learning, no context, no priors
    2. Non-contextual EMA   – Learns per-model means, no context features
    3. ε-greedy (no priors) – Contextual features, random exploration, no priors
    4. ε-greedy (w/ priors) – Contextual features, random exploration, warmup priors
    5. LinUCB (no priors)   – Contextual UCB exploration, no priors ("Tabula Rasa Only")
    6. LinUCB (w/ priors)   – Contextual UCB exploration, warmup priors ("UCB Only")
    7. banditGPT-Hybrid     – Corralling over warmup + tabula rasa experts

Protocol (matches generate_pareto_frontier.py exactly):
    - Training: Dev set (N=1,121), bandit feedback (observe reward of chosen arm only)
    - Evaluation: Holdout set (N=750), pure exploitation (no updates)
    - Seeds: 20 independent trials per (method, λ) pair
    - Lambda sweep: Same 10 values as banditGPT-Hybrid
    - Hyperparameters: α decay 2.0→0.1, neff=10, same normalization bounds
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging
import time
from typing import Dict, List, Tuple
from scipy import stats as sp_stats
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split,
    normalize_prior_strength,
    banditgpt_hybrid_routing,
)
from bandit_gpt.router import (
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
    infer_model_family,
)
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
import joblib

# Factory for creating production BanditRouter instances in experiments
sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding cache — encode each prompt ONCE, reuse across all trials/methods
# ---------------------------------------------------------------------------

def precompute_embeddings(
    data: List[Dict], encoder: SentenceTransformer, pca
) -> List[np.ndarray]:
    """Encode all prompts once. Returns list aligned with data indices."""
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]

# ---------------------------------------------------------------------------
# Shared constants (match generate_pareto_frontier.py)
# ---------------------------------------------------------------------------
N_TRIALS = 20
SEED_OFFSET = 42  # seeds 42..61
ALPHA_START = 2.0
ALPHA_END = 0.1
TARGET_SAMPLE_SIZE = 10.0
COST_PENALTIES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


# ============================================================================
# BASELINE 1: Random Routing
# ============================================================================

def random_routing(
    eval_data: List[Dict], model_costs: Dict
) -> Tuple[float, float]:
    """
    Uniform random model selection. No learning, no context.
    Lower bound — any learning-based method should beat this.
    """
    models = list(eval_data[0]["rewards"].keys())
    total_reward = 0.0
    total_cost = 0.0

    for p in eval_data:
        selected = np.random.choice(models)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 2: Non-Contextual EMA Tracker
# ============================================================================

def ema_routing(
    train_data: List[Dict],
    eval_data: List[Dict],
    model_costs: Dict,
    alpha: float = 0.15,
    epsilon: float = 0.1,
    lambda_penalty: float = 0.0,
) -> Tuple[float, float]:
    """
    Non-contextual exponential moving average tracker with ε-greedy exploration.

    Tracks per-model running averages of (reward - λ·normalized_cost) and
    exploits the best model (1-ε) of the time.  Uses NO context features —
    isolates the value of contextual routing.

    Args:
        alpha: EMA smoothing parameter (higher = more weight on recent data)
        epsilon: Exploration rate during training
        lambda_penalty: Cost penalty weight
    """
    models = list(train_data[0]["rewards"].keys())

    # Normalization bounds (from train data only — zero leakage)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    # Initialize EMA estimates to 0.5 (neutral prior)
    ema = {m: 0.5 for m in models}

    # Training phase
    for p in train_data:
        # ε-greedy selection on cost-adjusted EMA values
        if np.random.random() < epsilon:
            selected = np.random.choice(models)
        else:
            scores = {
                m: ema[m] - lambda_penalty * model_costs[m].get("normalized_cost", 0.0)
                for m in models
            }
            selected = max(scores, key=scores.get)

        # Observe reward and update EMA for selected model only (bandit feedback)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        ema[selected] = alpha * norm_r + (1 - alpha) * ema[selected]

    # Evaluation phase — pure exploitation (ε=0)
    total_reward = 0.0
    total_cost = 0.0

    for p in eval_data:
        scores = {
            m: ema[m] - lambda_penalty * model_costs[m].get("normalized_cost", 0.0)
            for m in models
        }
        selected = max(scores, key=scores.get)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 3 & 4: ε-Greedy (contextual, with or without priors)
# ============================================================================

def epsilon_greedy_routing(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    epsilon: float = 0.1,
    lambda_penalty: float = 0.0,
    warmup_priors: Dict = None,
) -> Tuple[float, float]:
    """
    Contextual ε-greedy: exploit the best model via ridge regression (1-ε)
    of the time, explore uniformly at random ε of the time.

    When warmup_priors is provided, uses CostAwareLinUCBRouter (with priors).
    Otherwise uses CostAwareTabulaRasaRouter (learns from scratch).

    Hyperparameters intentionally match banditGPT-Hybrid experts:
        - alpha_start=2.0 → alpha_end=0.1 (same decay schedule)
        - neff=10 for priors (same normalization)
    """
    models = list(train_data[0]["rewards"].keys())
    family_map = {m: infer_model_family(m) for m in models}
    dim = len(train_embeddings[0])

    if warmup_priors is not None:
        scaled_priors = normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE)
        router = CostAwareLinUCBRouter(
            models=models,
            warmup_priors=scaled_priors,
            model_costs=model_costs,
            alpha_start=ALPHA_START,
            alpha_end=ALPHA_END,
            cost_penalty=lambda_penalty,
            family_map=family_map,
        )
    else:
        router = CostAwareTabulaRasaRouter(
            models=models,
            context_dim=dim,
            model_costs=model_costs,
            alpha_start=ALPHA_START,
            alpha_end=ALPHA_END,
            cost_penalty=lambda_penalty,
            family_map=family_map,
        )

    # Normalization bounds (train data only)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Training phase — ε-greedy selection
    for i, p in enumerate(train_data):
        x = train_embeddings[i]

        if np.random.random() < epsilon:
            selected = np.random.choice(models)
        else:
            selected = router.select_model(x, total_steps=burn_in_steps)

        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)

    # Evaluation phase — pure exploitation (ε=0)
    total_reward = 0.0
    total_cost = 0.0

    for i, p in enumerate(eval_data):
        x = eval_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 5 & 6: LinUCB Only (with or without priors, no Corralling)
# ============================================================================

def linucb_only_routing(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    lambda_penalty: float = 0.0,
    warmup_priors: Dict = None,
) -> Tuple[float, float]:
    """
    Single LinUCB expert (no Corralling meta-learner).

    When warmup_priors is provided, initializes from 80k battle priors
    ("UCB Only" — tests value of Corralling beyond single-expert LinUCB).
    Otherwise, learns from scratch ("Tabula Rasa Only").

    Hyperparameters match banditGPT-Hybrid experts exactly:
        - alpha_start=2.0, alpha_end=0.1 (same decay schedule)
        - neff=10 for priors (same normalization)
        - Same zero-leakage normalization bounds
    """
    models = list(train_data[0]["rewards"].keys())
    family_map = {m: infer_model_family(m) for m in models}
    dim = len(train_embeddings[0])

    if warmup_priors is not None:
        scaled_priors = normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE)
        router = CostAwareLinUCBRouter(
            models=models,
            warmup_priors=scaled_priors,
            model_costs=model_costs,
            alpha_start=ALPHA_START,
            alpha_end=ALPHA_END,
            cost_penalty=lambda_penalty,
            family_map=family_map,
        )
    else:
        router = CostAwareTabulaRasaRouter(
            models=models,
            context_dim=dim,
            model_costs=model_costs,
            alpha_start=ALPHA_START,
            alpha_end=ALPHA_END,
            cost_penalty=lambda_penalty,
            family_map=family_map,
        )

    # Normalization bounds (train data only)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Training phase — UCB selection (pure LinUCB, no ε-exploration overlay)
    for i, p in enumerate(train_data):
        x = train_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)

    # Evaluation phase — pure exploitation
    total_reward = 0.0
    total_cost = 0.0

    for i, p in enumerate(eval_data):
        x = eval_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 7: banditGPT-Hybrid (Corralling) — PRODUCTION ROUTER
# ============================================================================

def banditgpt_hybrid_routing_cached(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    warmup_path: str,
    lambda_penalty: float = 0.0,
) -> Tuple[float, float]:
    """
    banditGPT-Hybrid using the **production BanditRouter**.

    Exercises the full ``BanditRouter.create()`` → ``route()`` →
    ``process_feedback()`` code path with pre-computed embeddings.
    """
    dim = len(train_embeddings[0])
    router = create_experiment_router(
        model_registry=None,
        feature_dim=dim,
        prior_n_effective=TARGET_SAMPLE_SIZE,
        alpha=ALPHA_START,
        warmup_path=warmup_path,
        cost_penalty=lambda_penalty,
    )

    # Normalization bounds (train data only — zero leakage)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Phase 1: Burn-in on dev set
    for i, p in enumerate(train_data):
        model, log = router.route(train_embeddings[i], total_steps=burn_in_steps)
        norm_r = (p["rewards"][model] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

    # Phase 2: Evaluation on holdout (no updates)
    total_reward = 0.0
    total_cost = 0.0
    for i, p in enumerate(eval_data):
        model, _log = router.route(eval_embeddings[i], total_steps=burn_in_steps)
        total_reward += p["rewards"][model]
        total_cost += model_costs[model]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# RUNNER
# ============================================================================

def run_method(
    method_name: str,
    method_func,
    test_lambdas: List[float],
    n_trials: int,
) -> List[Dict]:
    """Run a single method across all lambda values and trials."""
    results = []

    for lambda_val in test_lambdas:
        trial_rewards = []
        trial_costs = []

        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)
            r, c = method_func(lambda_val)
            trial_rewards.append(r)
            trial_costs.append(c)

        avg_r = np.mean(trial_rewards)
        avg_c = np.mean(trial_costs)
        std_r = np.std(trial_rewards, ddof=1) if n_trials > 1 else 0.0
        std_c = np.std(trial_costs, ddof=1) if n_trials > 1 else 0.0
        t_crit = sp_stats.t.ppf(0.975, n_trials - 1) if n_trials > 1 else 1.96
        ci95_r = t_crit * std_r / np.sqrt(n_trials)
        ci95_c = t_crit * std_c / np.sqrt(n_trials)

        results.append({
            "lambda": lambda_val,
            "reward": avg_r,
            "cost": avg_c,
            "reward_std": std_r,
            "cost_std": std_c,
            "reward_ci95": ci95_r,
            "cost_ci95": ci95_c,
            "n_trials": n_trials,
        })

        logger.info(
            f"  λ={lambda_val:<5.2f}  Reward={avg_r:.4f}±{ci95_r:.4f}  "
            f"Cost=${avg_c:.6f}±${ci95_c:.6f}  ({n_trials} trials)"
        )

    return results


def main():
    logger.info("=" * 70)
    logger.info("ARCHITECTURAL ABLATION STUDY")
    logger.info("=" * 70)
    logger.info(
        "\nIsolating each banditGPT component:\n"
        "  1. Random               – lower bound\n"
        "  2. EMA Tracker          – non-contextual learning\n"
        "  3. ε-greedy (no priors) – contextual, random exploration, no priors\n"
        "  4. ε-greedy (w/ priors) – contextual, random exploration, warmup priors\n"
        "  5. LinUCB (no priors)   – contextual, UCB exploration, no priors\n"
        "  6. LinUCB (w/ priors)   – contextual, UCB exploration, warmup priors\n"
        "  7. banditGPT-Hybrid     – Corralling over warmup + tabula rasa\n"
        "\nProtocol: identical to generate_pareto_frontier.py\n"
        f"  Seeds: {N_TRIALS} trials (seeds {SEED_OFFSET}..{SEED_OFFSET + N_TRIALS - 1})\n"
        f"  Lambda sweep: {len(COST_PENALTIES)} values\n"
        f"  Alpha decay: {ALPHA_START}→{ALPHA_END}\n"
        f"  Prior neff: {TARGET_SAMPLE_SIZE}"
    )

    # -----------------------------------------------------------------------
    # Load data, encoder, PCA, priors (same as generate_pareto_frontier.py)
    # -----------------------------------------------------------------------
    logger.info("\n--- Loading data and models ---")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    # Resolve warmup priors path (prefer sanitized version)
    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)
    warmup_priors = joblib.load(warmup_path)

    # Normalize costs (same as generate_pareto_frontier.py)
    models = list(eval_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost

    normalized_costs = {}
    for model_id in models:
        raw_cost = model_costs[model_id]["cost"]
        normalized = (raw_cost - min_cost) / cost_range if cost_range > 0 else 0.0
        normalized_costs[model_id] = {
            "cost": raw_cost,
            "normalized_cost": normalized,
        }

    # -----------------------------------------------------------------------
    # Pre-compute embeddings ONCE (eliminates ~95% of runtime)
    # -----------------------------------------------------------------------
    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(
        f"  Encoded {len(train_emb)} train + {len(eval_emb)} eval prompts "
        f"in {time.time() - t0:.1f}s  (dim={len(train_emb[0])})"
    )

    # -----------------------------------------------------------------------
    # Run ablations
    # -----------------------------------------------------------------------
    all_results = {}
    t_start = time.time()

    # 1. Random (lambda-independent)
    logger.info("\n[1/7] Random Routing")
    all_results["Random"] = run_method(
        "Random",
        lambda _lam: random_routing(eval_data, normalized_costs),
        [0.0],  # Lambda doesn't affect random; run once
        N_TRIALS,
    )

    # 2. EMA Tracker (non-contextual)
    logger.info("\n[2/7] EMA Tracker (non-contextual, ε=0.1, α_ema=0.15)")
    all_results["EMA Tracker"] = run_method(
        "EMA Tracker",
        lambda lam: ema_routing(train_data, eval_data, normalized_costs, lambda_penalty=lam),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 3. ε-greedy without priors
    logger.info("\n[3/7] ε-greedy (ε=0.1, no priors)")
    all_results["ε-greedy (no priors)"] = run_method(
        "ε-greedy (no priors)",
        lambda lam: epsilon_greedy_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            epsilon=0.1, lambda_penalty=lam, warmup_priors=None,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 4. ε-greedy with warmup priors
    logger.info("\n[4/7] ε-greedy (ε=0.1, with warmup priors)")
    all_results["ε-greedy (w/ priors)"] = run_method(
        "ε-greedy (w/ priors)",
        lambda lam: epsilon_greedy_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            epsilon=0.1, lambda_penalty=lam, warmup_priors=warmup_priors,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 5. LinUCB without priors (tabula rasa only)
    logger.info("\n[5/7] LinUCB Only (no priors — tabula rasa)")
    all_results["LinUCB (no priors)"] = run_method(
        "LinUCB (no priors)",
        lambda lam: linucb_only_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, warmup_priors=None,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 6. LinUCB with warmup priors (UCB only)
    logger.info("\n[6/7] LinUCB Only (with warmup priors)")
    all_results["LinUCB (w/ priors)"] = run_method(
        "LinUCB (w/ priors)",
        lambda lam: linucb_only_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, warmup_priors=warmup_priors,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 7. banditGPT-Hybrid (Corralling) — PRODUCTION BanditRouter
    logger.info("\n[7/7] banditGPT-Hybrid (Production BanditRouter)")
    all_results["banditGPT-Hybrid"] = run_method(
        "banditGPT-Hybrid",
        lambda lam: banditgpt_hybrid_routing_cached(
            train_data, eval_data, train_emb, eval_emb,
            model_costs=normalized_costs, warmup_path=warmup_path,
            lambda_penalty=lam,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    elapsed = time.time() - t_start
    logger.info(f"\n--- All ablations complete in {elapsed:.0f}s ---")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_ablations.json"

    ablation_data = {
        "metadata": {
            "description": (
                "Architectural ablation: isolates contribution of "
                "context features, UCB exploration, warmup priors, and Corralling"
            ),
            "n_eval_prompts": len(eval_data),
            "n_train_prompts": len(train_data),
            "n_trials": N_TRIALS,
            "seeds": list(range(SEED_OFFSET, SEED_OFFSET + N_TRIALS)),
            "cost_penalties": COST_PENALTIES,
            "alpha_schedule": f"{ALPHA_START}→{ALPHA_END}",
            "prior_neff": TARGET_SAMPLE_SIZE,
        },
        "methods": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(ablation_data, f, indent=2)

    logger.info(f"\n--- Saved: {output_file} ---")

    # -----------------------------------------------------------------------
    # Summary table at λ=0.0 (quality-focused)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 78)
    logger.info("SUMMARY AT λ=0.0 (QUALITY-FOCUSED)")
    logger.info("=" * 78)
    logger.info(
        f"{'Method':<28} | {'Reward':<20} | {'Cost':<20} | Component Tested"
    )
    logger.info("-" * 78)

    component_labels = {
        "Random":               "lower bound",
        "EMA Tracker":          "+ learning (no context)",
        "ε-greedy (no priors)": "+ context features",
        "ε-greedy (w/ priors)": "+ warmup priors",
        "LinUCB (no priors)":   "+ UCB exploration (vs ε)",
        "LinUCB (w/ priors)":   "+ UCB + priors",
        "banditGPT-Hybrid":     "+ Corralling meta-learner",
    }

    for method, label in component_labels.items():
        if method not in all_results:
            continue
        # Find λ=0.0 entry (first entry, or only entry for Random)
        entry = all_results[method][0]
        r, ci = entry["reward"], entry["reward_ci95"]
        c = entry["cost"]
        logger.info(
            f"{method:<28} | {r:.4f} ± {ci:.4f}       | ${c:.6f}         | {label}"
        )

    logger.info("=" * 78)


if __name__ == "__main__":
    main()
