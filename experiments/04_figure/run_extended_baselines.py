#!/usr/bin/env python3
"""
Extended Baseline Comparison
==========================================================

Complements run_baseline_ablations.py with three additional algorithmic
families for a comprehensive evaluation:

    8.  LinTS (no priors)   — Thompson Sampling, tabula rasa
    9.  LinTS (w/ priors)   — Thompson Sampling, warmup priors
    10. Learned Proj (16D)  — Trainable projection + LinUCB (tests PILOT hypothesis)
    11. Learned Proj (32D)  — Full-dim projection + LinUCB (no dimensionality reduction)
    12. Cost Threshold       — Difficulty-threshold heuristic (no learning)

Protocol (identical to run_baseline_ablations.py):
    - Training: Dev set (N=1,121), bandit feedback (observe reward of chosen arm only)
    - Evaluation: Holdout set (N=750), pure exploitation (no updates)
    - Seeds: 20 independent trials per (method, λ) pair
    - Lambda sweep: Same 10 values as banditGPT-Hybrid
    - Hyperparameters: v²=0.25 (LinTS), proj_lr=0.01 (Learned Proj)

Rationale:
    - LinTS (Thompson Sampling) is the dominant production bandit algorithm.
      Its absence from the comparison would be a notable gap.
    - Learned Projection tests the PILOT hypothesis (jointly-learned embeddings)
      within our framework, avoiding "unfaithful reimplementation" criticism.
    - Cost Threshold provides a "no-learning" reference that quantifies the
      marginal value of online adaptation.
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
)
from run_baseline_ablations import (
    precompute_embeddings,
    run_method,
    N_TRIALS,
    SEED_OFFSET,
    ALPHA_START,
    ALPHA_END,
    TARGET_SAMPLE_SIZE,
    COST_PENALTIES,
)
from bandit_gpt.baselines import (
    CostAwareLinTSRouter,
    CostAwareLearnedProjRouter,
    CostThresholdRouter,
)
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# BASELINE 8 & 9: Linear Thompson Sampling (with or without priors)
# ============================================================================

def lints_routing(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    lambda_penalty: float = 0.0,
    warmup_priors: Dict = None,
    noise_variance: float = 0.25,
) -> Tuple[float, float]:
    """
    Linear Thompson Sampling: posterior sampling instead of UCB.

    Replaces the deterministic α√(xᵀA⁻¹x) exploration bonus with stochastic
    θ̃ ~ N(θ̂, v²·A⁻¹). No alpha scheduling needed.

    When warmup_priors is provided, initializes from 80k battle priors.
    Otherwise, learns from scratch (tabula rasa).
    """
    models = list(train_data[0]["rewards"].keys())
    dim = len(train_embeddings[0])

    priors = None
    if warmup_priors is not None:
        priors = normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE)

    router = CostAwareLinTSRouter(
        models=models,
        context_dim=dim,
        model_costs=model_costs,
        cost_penalty=lambda_penalty,
        noise_variance=noise_variance,
        warmup_priors=priors,
    )

    # Normalization bounds (train data only — zero leakage)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Training phase — Thompson Sampling selection
    for i, p in enumerate(train_data):
        x = train_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)

    # Evaluation phase — still uses TS (exploitation via narrow posterior)
    total_reward = 0.0
    total_cost = 0.0

    for i, p in enumerate(eval_data):
        x = eval_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 10 & 11: Learned Projection + LinUCB
# ============================================================================

def learned_proj_routing(
    train_data: List[Dict],
    eval_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    lambda_penalty: float = 0.0,
    proj_dim: int = 16,
    proj_lr: float = 0.01,
) -> Tuple[float, float]:
    """
    Learned Projection + LinUCB: trainable feature representation.

    Tests PILOT's core hypothesis: "Does jointly learning the feature
    representation improve routing quality?" — within our framework.

    The projection matrix W (proj_dim × raw_dim) is updated via online
    gradient descent on the squared prediction error after each reward.
    """
    models = list(train_data[0]["rewards"].keys())
    raw_dim = len(train_embeddings[0])

    router = CostAwareLearnedProjRouter(
        models=models,
        raw_dim=raw_dim,
        model_costs=model_costs,
        proj_dim=proj_dim,
        cost_penalty=lambda_penalty,
        proj_lr=proj_lr,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,
    )

    # Normalization bounds (train data only)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Training phase — UCB selection with projection learning
    for i, p in enumerate(train_data):
        x = train_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)

    # Evaluation phase — pure exploitation (fixed projection)
    total_reward = 0.0
    total_cost = 0.0

    for i, p in enumerate(eval_data):
        x = eval_embeddings[i]
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# BASELINE 12: Cost Threshold Heuristic (no learning)
# ============================================================================

def cost_threshold_routing(
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    model_costs: Dict,
    threshold: float = 1.0,
) -> Tuple[float, float]:
    """
    Difficulty-threshold routing: no bandit learning.

    If embedding norm > threshold → expensive model, else → cheap model.
    Sweeping threshold generates a Pareto curve.
    """
    models = list(eval_data[0]["rewards"].keys())

    router = CostThresholdRouter(
        models=models,
        model_costs=model_costs,
        threshold=threshold,
        difficulty_feature="norm",
    )

    total_reward = 0.0
    total_cost = 0.0

    for i, p in enumerate(eval_data):
        x = eval_embeddings[i]
        selected = router.select_model(x)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# ============================================================================
# RUNNER
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("EXTENDED BASELINE COMPARISON")
    logger.info("=" * 70)
    logger.info(
        "\nNew algorithmic families:\n"
        "  8.  LinTS (no priors)   — Thompson Sampling, tabula rasa\n"
        "  9.  LinTS (w/ priors)   — Thompson Sampling, warmup priors\n"
        "  10. Learned Proj (16D)  — Trainable projection + LinUCB\n"
        "  11. Learned Proj (32D)  — Full-dim trainable projection + LinUCB\n"
        "  12. Cost Threshold      — Difficulty-threshold heuristic (no learning)\n"
        f"\nProtocol: identical to run_baseline_ablations.py\n"
        f"  Seeds: {N_TRIALS} trials (seeds {SEED_OFFSET}..{SEED_OFFSET + N_TRIALS - 1})\n"
        f"  Lambda sweep: {len(COST_PENALTIES)} values\n"
    )

    # -----------------------------------------------------------------------
    # Load data, encoder, PCA, priors (same as run_baseline_ablations.py)
    # -----------------------------------------------------------------------
    logger.info("--- Loading data and models ---")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    # Use sanitized priors
    sanitized_path = (
        Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    )
    warmup_priors = joblib.load(
        sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH
    )

    # Normalize costs (same as other experiments)
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
    # Pre-compute embeddings ONCE
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
    # Run extended baselines
    # -----------------------------------------------------------------------
    all_results = {}
    t_start = time.time()

    # 8. LinTS without priors (tabula rasa Thompson Sampling)
    logger.info("\n[1/5] LinTS (v²=0.25, no priors — tabula rasa Thompson Sampling)")
    all_results["LinTS (no priors)"] = run_method(
        "LinTS (no priors)",
        lambda lam: lints_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, warmup_priors=None, noise_variance=0.25,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 9. LinTS with warmup priors
    logger.info("\n[2/5] LinTS (v²=0.25, with warmup priors)")
    all_results["LinTS (w/ priors)"] = run_method(
        "LinTS (w/ priors)",
        lambda lam: lints_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, warmup_priors=warmup_priors, noise_variance=0.25,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 10. Learned Projection (16D) — compressed representation
    logger.info("\n[3/5] Learned Projection (proj_dim=16, lr=0.01)")
    all_results["Learned Proj (16D)"] = run_method(
        "Learned Proj (16D)",
        lambda lam: learned_proj_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, proj_dim=16, proj_lr=0.01,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 11. Learned Projection (32D) — full-dim, no compression
    logger.info("\n[4/5] Learned Projection (proj_dim=32, lr=0.01)")
    all_results["Learned Proj (32D)"] = run_method(
        "Learned Proj (32D)",
        lambda lam: learned_proj_routing(
            train_data, eval_data, train_emb, eval_emb, normalized_costs,
            lambda_penalty=lam, proj_dim=32, proj_lr=0.01,
        ),
        COST_PENALTIES,
        N_TRIALS,
    )

    # 12. Cost Threshold (sweep thresholds to generate Pareto curve)
    # Instead of lambda sweep, we sweep the difficulty threshold directly
    logger.info("\n[5/5] Cost Threshold (difficulty-threshold heuristic)")

    # Compute norm statistics to set meaningful thresholds
    norms = [float(np.linalg.norm(x[:-1])) for x in eval_emb]  # Exclude bias
    norm_min, norm_max = min(norms), max(norms)
    norm_mean = np.mean(norms)
    logger.info(
        f"  Embedding norms: min={norm_min:.3f}, max={norm_max:.3f}, mean={norm_mean:.3f}"
    )

    # Sweep thresholds from "route everything to expensive" to "route everything to cheap"
    thresholds = np.linspace(norm_min - 0.1, norm_max + 0.1, 15)
    threshold_results = []

    for i, thresh in enumerate(thresholds):
        trial_rewards = []
        trial_costs = []

        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            r, c = cost_threshold_routing(
                eval_data, eval_emb, normalized_costs, threshold=thresh
            )
            trial_rewards.append(r)
            trial_costs.append(c)

        avg_r = np.mean(trial_rewards)
        avg_c = np.mean(trial_costs)
        std_r = np.std(trial_rewards, ddof=1) if N_TRIALS > 1 else 0.0
        std_c = np.std(trial_costs, ddof=1) if N_TRIALS > 1 else 0.0
        t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1) if N_TRIALS > 1 else 1.96
        ci95_r = t_crit * std_r / np.sqrt(N_TRIALS)
        ci95_c = t_crit * std_c / np.sqrt(N_TRIALS)

        threshold_results.append({
            "threshold": float(thresh),
            "reward": avg_r,
            "cost": avg_c,
            "reward_std": std_r,
            "cost_std": std_c,
            "reward_ci95": ci95_r,
            "cost_ci95": ci95_c,
            "n_trials": N_TRIALS,
        })

        logger.info(
            f"  t={thresh:<6.3f}  Reward={avg_r:.4f}±{ci95_r:.4f}  "
            f"Cost=${avg_c:.6f}±${ci95_c:.6f}"
        )

    all_results["Cost Threshold"] = threshold_results

    elapsed = time.time() - t_start
    logger.info(f"\n--- All extended baselines complete in {elapsed:.0f}s ---")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "extended_baselines.json"

    extended_data = {
        "metadata": {
            "description": (
                "Extended baseline comparison for comprehensive evaluation. "
                "Adds Thompson Sampling, learned projection, and cost threshold "
                "to the existing ablation suite."
            ),
            "n_eval_prompts": len(eval_data),
            "n_train_prompts": len(train_data),
            "n_trials": N_TRIALS,
            "seeds": list(range(SEED_OFFSET, SEED_OFFSET + N_TRIALS)),
            "cost_penalties": COST_PENALTIES,
            "new_baselines": [
                "LinTS (no priors): v²=0.25, Thompson Sampling family",
                "LinTS (w/ priors): v²=0.25, Thompson Sampling with warmup",
                "Learned Proj (16D): proj_dim=16, lr=0.01, tests PILOT hypothesis",
                "Learned Proj (32D): proj_dim=32, lr=0.01, full-dim projection",
                "Cost Threshold: difficulty-threshold heuristic, no learning",
            ],
        },
        "methods": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(extended_data, f, indent=2)

    logger.info(f"\n--- Saved: {output_file} ---")

    # -----------------------------------------------------------------------
    # Summary table at λ=0.0 (quality-focused)
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 85)
    logger.info("SUMMARY AT λ=0.0 (QUALITY-FOCUSED)")
    logger.info("=" * 85)
    logger.info(
        f"{'Method':<28} | {'Reward':<20} | {'Cost':<20} | Algorithmic Family"
    )
    logger.info("-" * 85)

    family_labels = {
        "LinTS (no priors)":    "Thompson Sampling (tabula rasa)",
        "LinTS (w/ priors)":    "Thompson Sampling (warmup priors)",
        "Learned Proj (16D)":   "Representation learning (compressed)",
        "Learned Proj (32D)":   "Representation learning (full-dim)",
        "Cost Threshold":       "Heuristic (no learning)",
    }

    for method, label in family_labels.items():
        if method not in all_results:
            continue
        entry = all_results[method][0]
        r = entry["reward"]
        ci = entry.get("reward_ci95", entry.get("reward_std", 0.0))
        c = entry["cost"]
        logger.info(
            f"{method:<28} | {r:.4f} ± {ci:.4f}       | ${c:.6f}         | {label}"
        )

    logger.info("=" * 85)

    # -----------------------------------------------------------------------
    # Cross-reference with existing ablation results
    # -----------------------------------------------------------------------
    existing_path = output_dir / "baseline_ablations.json"
    if existing_path.exists():
        logger.info("\n📊 CROSS-REFERENCE WITH EXISTING ABLATIONS (λ=0.0):")
        logger.info("-" * 85)

        with open(existing_path) as f:
            existing = json.load(f)

        # Show existing results for comparison
        existing_methods = existing.get("methods", {})
        for method in ["Random", "EMA Tracker", "ε-greedy (no priors)",
                       "LinUCB (no priors)", "LinUCB (w/ priors)", "banditGPT-Hybrid"]:
            if method in existing_methods:
                entry = existing_methods[method][0]
                r = entry["reward"]
                ci = entry.get("reward_ci95", entry.get("reward_std", 0.0))
                c = entry["cost"]
                logger.info(
                    f"  {method:<28} | {r:.4f} ± {ci:.4f}       | ${c:.6f}"
                )

        logger.info("-" * 85)
        logger.info("  (New baselines from this run shown above)")
    else:
        logger.info(
            "\n⚠️  No existing baseline_ablations.json found. "
            "Run run_baseline_ablations.py first for full comparison."
        )


if __name__ == "__main__":
    main()
