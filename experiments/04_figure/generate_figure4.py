#!/usr/bin/env python3
"""
Figure 4: Honest Pareto Comparison with Learning Curve

Two-panel figure addressing fairness concerns in the RouteLLM comparison:

Panel A — Pareto Frontier (Honest)
  - banditGPT with filled λ sweep (15 values) for dense frontier coverage
  - RouteLLM-MF (28 thresholds, pre-trained on 100k OOD pairs)
  - Regime annotations showing where each method excels

Panel B — Learning Curve (New)
  - banditGPT holdout quality vs. number of online learning steps
  - RouteLLM peak quality as horizontal reference
  - Shows the crossover point: where online adaptation surpasses pre-training
  - Key finding: ~N in-distribution examples can surpass 100k OOD pre-trained pairs

Data asymmetry (both sides have advantages):
  - RouteLLM:   100k OOD supervised pairs → strong out-of-box, zero adaptation
  - banditGPT:  80k OOD priors + 1,121 in-distribution online learning → higher ceiling

All banditGPT conditions use the production BanditRouter via
create_experiment_router() with default alpha schedules (warmup: constant
α=2.0, tabula rasa: decaying α=1.0→0.01) and corralling η=0.1.

Usage:
    python generate_figure4.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
import joblib

sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING CACHE — avoids recomputing for every trial
# =============================================================================

def precompute_embeddings(data: List[Dict], encoder, pca) -> List[np.ndarray]:
    """Pre-compute PCA-reduced embeddings for all prompts."""
    logger.info(f"  Embedding {len(data)} prompts...")
    embeddings = []
    for i, p in enumerate(data):
        x = embed_prompt(p["prompt"], encoder, pca)
        embeddings.append(x)
        if (i + 1) % 500 == 0:
            logger.info(f"    {i+1}/{len(data)} done")
    logger.info(f"  ✓ {len(embeddings)} embeddings cached (dim={embeddings[0].shape[0]})")
    return embeddings


# =============================================================================
# FROZEN EVALUATION — no state changes to the router
# =============================================================================

def evaluate_frozen(router, eval_data: List[Dict], eval_embeddings: List[np.ndarray],
                    model_costs: Dict, burn_in_steps: int) -> Tuple[float, float]:
    """
    Evaluate router on holdout WITHOUT modifying learned state.

    Calls route() but never process_feedback(), so bandit parameters
    (A, b matrices) and Corralling weights remain unchanged.
    Numpy random state is saved/restored so training trajectory is unaffected.
    """
    rng_state = np.random.get_state()

    total_reward = 0.0
    total_cost = 0.0

    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][model]
        total_cost += model_costs[model]["cost"]

    np.random.set_state(rng_state)

    return total_reward / len(eval_data), total_cost / len(eval_data)


# =============================================================================
# LEARNING CURVE EXPERIMENT
# =============================================================================

def run_learning_curve(train_data, eval_data, train_embeddings, eval_embeddings,
                       warmup_path, model_costs, lambda_penalty=0.0,
                       n_trials=20, checkpoint_steps=None):
    """
    Train banditGPT with periodic frozen holdout evaluation to produce
    a learning curve (quality vs. number of online learning steps).

    Uses the production BanditRouter via create_experiment_router() with
    default alpha schedules (warmup: constant α=2.0, tabula rasa: decaying).

    Returns:
        List of dicts: [{step, mean_reward, std_reward, mean_cost, std_cost, n_trials}]
    """
    if checkpoint_steps is None:
        checkpoint_steps = [0, 25, 50, 100, 200, 300, 400, 500, 700, 900, 1121]

    checkpoint_set = set(checkpoint_steps)
    results_by_step = {step: {"rewards": [], "costs": []} for step in checkpoint_steps}

    dim = train_embeddings[0].shape[0]
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    t_start = time.time()

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        router = create_experiment_router(
            model_registry=None,
            feature_dim=dim,
            prior_n_effective=10.0,
            alpha=2.0,
            warmup_path=warmup_path,
            cost_penalty=lambda_penalty,
        )

        if 0 in checkpoint_set:
            r, c = evaluate_frozen(router, eval_data, eval_embeddings,
                                   model_costs, burn_in_steps)
            results_by_step[0]["rewards"].append(r)
            results_by_step[0]["costs"].append(c)

        for step_idx, (p, x) in enumerate(zip(train_data, train_embeddings)):
            model, log = router.route(x, total_steps=burn_in_steps)
            norm_r = (p["rewards"][model] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)

            current_step = step_idx + 1
            if current_step in checkpoint_set:
                r, c = evaluate_frozen(router, eval_data, eval_embeddings,
                                       model_costs, burn_in_steps)
                results_by_step[current_step]["rewards"].append(r)
                results_by_step[current_step]["costs"].append(c)

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)
        logger.info(f"  Trial {trial+1:2d}/{n_trials} | "
                     f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    # Aggregate statistics
    curve_data = []
    for step in sorted(checkpoint_steps):
        rewards = results_by_step[step]["rewards"]
        costs = results_by_step[step]["costs"]
        if rewards:
            curve_data.append({
                "step": step,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                "mean_cost": float(np.mean(costs)),
                "std_cost": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                "n_trials": len(rewards)
            })

    return curve_data


# =============================================================================
# ADDITIONAL λ SWEEP — fill the frontier gap
# =============================================================================

def run_lambda_sweep(train_data, eval_data, train_embeddings, eval_embeddings,
                     warmup_path, model_costs, lambda_values, n_trials=20):
    """
    Run banditGPT for additional lambda values using pre-computed embeddings.
    Same protocol as the main experiment (train on dev, freeze, eval on holdout).
    Uses the production BanditRouter via create_experiment_router().
    """
    dim = train_embeddings[0].shape[0]
    burn_in_steps = len(train_data)

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    results = []

    for i, lambda_val in enumerate(lambda_values, 1):
        trial_rewards = []
        trial_costs = []

        for trial in range(n_trials):
            np.random.seed(42 + trial)
            router = create_experiment_router(
                model_registry=None,
                feature_dim=dim,
                prior_n_effective=10.0,
                alpha=2.0,
                warmup_path=warmup_path,
                cost_penalty=lambda_val,
            )

            for p, x in zip(train_data, train_embeddings):
                model, log = router.route(x, total_steps=burn_in_steps)
                norm_r = (p["rewards"][model] - r_min) / r_range
                router.process_feedback(log.request_id, norm_r)

            r, c = evaluate_frozen(router, eval_data, eval_embeddings,
                                   model_costs, burn_in_steps)
            trial_rewards.append(r)
            trial_costs.append(c)

        avg_r = np.mean(trial_rewards)
        avg_c = np.mean(trial_costs)
        std_r = np.std(trial_rewards, ddof=1) if len(trial_rewards) > 1 else 0.0
        std_c = np.std(trial_costs, ddof=1) if len(trial_costs) > 1 else 0.0

        results.append({
            "lambda": lambda_val,
            "mean_cost": float(avg_c),
            "mean_reward": float(avg_r),
            "std_cost": float(std_c),
            "std_reward": float(std_r),
            "n_trials": n_trials
        })

        logger.info(f"  [{i}/{len(lambda_values)}] λ={lambda_val:.2f}: "
                     f"R={avg_r:.4f}±{std_r:.4f}, C=${avg_c:.6f}")

    return results


# =============================================================================
# PARETO HULL COMPUTATION
# =============================================================================

def compute_pareto_hull(points, stats=None):
    """
    Compute the Pareto-optimal (monotone upper envelope) from a set of
    (cost, reward) points sorted by cost.

    Returns:
        hull_costs, hull_rewards, hull_stats (if stats provided),
        dom_costs, dom_rewards
    """
    sorted_indices = sorted(range(len(points)), key=lambda i: points[i][0])
    sorted_pts = [points[i] for i in sorted_indices]
    sorted_stats = [stats[i] for i in sorted_indices] if stats else None

    hull_c, hull_r, hull_s = [], [], []
    dom_c, dom_r = [], []
    max_r = -float('inf')

    for idx, (c, r) in enumerate(sorted_pts):
        if r > max_r:
            hull_c.append(c)
            hull_r.append(r)
            if sorted_stats:
                hull_s.append(sorted_stats[idx])
            max_r = r
        else:
            dom_c.append(c)
            dom_r.append(r)

    return hull_c, hull_r, hull_s, dom_c, dom_r


# =============================================================================
# TWO-PANEL PLOT
# =============================================================================

def plot_two_panel(bandit_points, routellm_points, oracle_point, static_points,
                   learning_curve, routellm_peak_reward, n_eval,
                   output_dir, bandit_stats=None):
    """
    Two-panel publication figure.

    Panel A: Honest Pareto frontier with regime annotations
    Panel B: Learning curve with RouteLLM crossover
    """
    from scipy import stats as sp_stats

    # Wong 2011 colorblind-safe palette
    BLUE   = '#0072B2'
    RED    = '#D55E00'
    GREEN  = '#009E73'
    GRAY   = '#999999'
    ORANGE = '#E69F00'
    LIGHT_BLUE = '#56B4E9'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    constrained_layout=True)

    # =====================================================================
    # PANEL A: Honest Pareto Frontier
    # =====================================================================

    # Compute Pareto hulls
    rl_hull_c, rl_hull_r, _, rl_dom_c, rl_dom_r = compute_pareto_hull(routellm_points)
    bg_hull_c, bg_hull_r, bg_hull_s, bg_dom_c, bg_dom_r = compute_pareto_hull(
        bandit_points, bandit_stats)

    # RouteLLM frontier
    ax1.plot(rl_hull_c, rl_hull_r, color=RED, linewidth=2.5, alpha=0.85,
             marker='o', markersize=5, zorder=4,
             label='RouteLLM-MF frontier')
    ax1.scatter([c for c, _ in routellm_points], [r for _, r in routellm_points],
                color=RED, alpha=0.12, s=18, zorder=1)
    if rl_dom_c:
        ax1.scatter(rl_dom_c, rl_dom_r, color=RED, marker='x', s=60,
                    linewidths=1.5, alpha=0.4, zorder=3)

    # banditGPT frontier
    ax1.plot(bg_hull_c, bg_hull_r, color=BLUE, linewidth=2.5, alpha=0.9,
             marker='D', markersize=5, zorder=5,
             label='banditGPT frontier')
    ax1.scatter([c for c, _ in bandit_points], [r for _, r in bandit_points],
                color=BLUE, alpha=0.15, s=18, zorder=1)
    if bg_dom_c:
        ax1.scatter(bg_dom_c, bg_dom_r, color=BLUE, marker='x', s=60,
                    linewidths=1.5, alpha=0.4, zorder=3)

    # Error bars on banditGPT hull (95% CI with t₁₉)
    if bg_hull_s and any(s.get("reward_std", 0) > 0 for s in bg_hull_s):
        t_crit = sp_stats.t.ppf(0.975, df=19)
        ci_r = [t_crit * s.get("reward_std", 0) / np.sqrt(20) for s in bg_hull_s]
        ax1.errorbar(bg_hull_c, bg_hull_r, yerr=ci_r,
                     fmt='none', ecolor=BLUE, alpha=0.3, capsize=3, capthick=1.5,
                     zorder=6)

    # Oracle omitted from plot — reported in text (Section 5.1) as theoretical
    # upper bound (0.953) to avoid visual clutter. Static baselines also omitted
    # (implicit in frontier endpoints: leftmost ≈ Mixtral, rightmost ≈ GPT-4).

    # Regime shading — find cost where banditGPT hull crosses above RouteLLM hull
    mixtral_r = list(static_points.values())[0][1]

    def interpolate_hull(hull_c, hull_r, query_costs):
        """Interpolate a Pareto hull (step-function: reward doesn't increase
        between hull points, so use the reward of the last hull point at or
        below the query cost)."""
        result = []
        for qc in query_costs:
            if qc <= hull_c[0]:
                result.append(hull_r[0])
            elif qc >= hull_c[-1]:
                result.append(hull_r[-1])
            else:
                # Linear interpolation between adjacent hull points
                for i in range(len(hull_c) - 1):
                    if hull_c[i] <= qc <= hull_c[i + 1]:
                        frac = (qc - hull_c[i]) / (hull_c[i + 1] - hull_c[i])
                        result.append(hull_r[i] + frac * (hull_r[i + 1] - hull_r[i]))
                        break
        return np.array(result)

    # Sample both hulls at fine cost grid to find crossover
    cost_grid = np.linspace(0.0003, 0.011, 500)
    bg_interp = interpolate_hull(bg_hull_c, bg_hull_r, cost_grid)
    rl_interp = interpolate_hull(rl_hull_c, rl_hull_r, cost_grid)

    crossover_cost = None
    for i in range(len(cost_grid)):
        if bg_interp[i] > rl_interp[i]:
            crossover_cost = cost_grid[i]
            break

    if crossover_cost is not None:
        # Shade from the minimum non-trivial cost to the crossover (RouteLLM advantage)
        shade_left = max(0.0003, rl_hull_c[0])
        ax1.axvspan(shade_left, crossover_cost, alpha=0.04, color=RED, zorder=0)
        # Shade from crossover to the right edge (banditGPT advantage)
        ax1.axvspan(crossover_cost, 0.011, alpha=0.04, color=BLUE, zorder=0)
        # Labels
        mid_rl = (shade_left + crossover_cost) / 2
        mid_bg = (crossover_cost + 0.010) / 2
        ax1.text(mid_rl, 0.935, 'RouteLLM\nadvantage', ha='center',
                 fontsize=7.5, color=RED, alpha=0.6, style='italic')
        ax1.text(mid_bg, 0.935, 'banditGPT\nadvantage', ha='center',
                 fontsize=7.5, color=BLUE, alpha=0.6, style='italic')

    ax1.set_xlabel('Average Cost per Request ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Reward (Quality)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Pareto Frontier', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.92)
    ax1.grid(True, alpha=0.15, linestyle='--')
    ax1.set_xlim(left=-0.0003)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.4f}'))
    ax1.tick_params(labelsize=9)

    # =====================================================================
    # PANEL B: Learning Curve
    # =====================================================================

    steps   = [d["step"]        for d in learning_curve]
    rewards = [d["mean_reward"] for d in learning_curve]
    stds    = [d["std_reward"]  for d in learning_curve]
    n_t     = learning_curve[0]["n_trials"]
    t_crit  = sp_stats.t.ppf(0.975, df=n_t - 1)

    ci_upper = [r + t_crit * s / np.sqrt(n_t) for r, s in zip(rewards, stds)]
    ci_lower = [r - t_crit * s / np.sqrt(n_t) for r, s in zip(rewards, stds)]

    # banditGPT learning curve
    ax2.plot(steps, rewards, color=BLUE, linewidth=2.5, marker='D', markersize=4,
             label=f'banditGPT (online, n={n_t} trials)', zorder=5)
    ax2.fill_between(steps, ci_lower, ci_upper, color=BLUE, alpha=0.12, zorder=2)

    # RouteLLM reference (horizontal)
    ax2.axhline(y=routellm_peak_reward, color=RED, linestyle='--', linewidth=2,
                alpha=0.8, zorder=3,
                label=f'RouteLLM peak ({routellm_peak_reward:.3f}, 100k OOD)')

    # Mixtral baseline
    ax2.axhline(y=mixtral_r, color=GRAY, linestyle=':', linewidth=1.5,
                alpha=0.6, zorder=3,
                label=f'Mixtral static ({mixtral_r:.3f})')

    # Oracle omitted from plot — reported in text as theoretical upper bound (0.953)

    # Find and annotate crossover point
    crossover_step = None
    crossover_reward = None
    for i in range(len(steps)):
        if rewards[i] >= routellm_peak_reward:
            crossover_step = steps[i]
            crossover_reward = rewards[i]
            break

    if crossover_step is not None:
        ax2.axvline(x=crossover_step, color=ORANGE, linestyle=':', linewidth=1.5,
                    alpha=0.6)
        # Place annotation to the left of the crossover point
        y_annot = max(rewards) - 0.005  # near the top of the curve
        ax2.annotate(
            f'Crossover\n@ step {crossover_step}',
            xy=(crossover_step, crossover_reward),
            xytext=(max(crossover_step - 200, 150), y_annot),
            fontsize=9, color=ORANGE, fontweight='bold',
            ha='center',
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5),
            zorder=10
        )

    # Data access annotation
    ax2.text(0.03, 0.04, (
        'RouteLLM: pre-trained on 100k OOD supervised pairs\n'
        'banditGPT: online learning on in-distribution prompts'
    ), transform=ax2.transAxes, fontsize=7.5, color='#555555', style='italic',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#cccccc', alpha=0.92))

    ax2.set_xlabel('Online Learning Steps (dev prompts seen)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'Holdout Quality (frozen eval, N={n_eval})', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Online Adaptation Value', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='center right', fontsize=8, framealpha=0.92)
    ax2.grid(True, alpha=0.15, linestyle='--')
    ax2.set_xlim(-30, max(steps) + 50)
    ax2.tick_params(labelsize=9)

    # Save
    path_300 = output_dir / 'figure4.png'
    plt.savefig(path_300, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"\n✅ Saved: {path_300}")

    path_600 = output_dir / 'figure4_hires.png'
    plt.savefig(path_600, dpi=600, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved: {path_600}")

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("FIGURE 5 (REVISED): HONEST PARETO + LEARNING CURVE")
    logger.info("=" * 70)
    logger.info("\nTwo-panel figure:")
    logger.info("  (a) Pareto frontier (α₀=1.0) with regime annotations")
    logger.info("  (b) Learning curve: online adaptation value vs pre-training")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 1. LOAD DATA, MODELS, PRIORS
    # =================================================================
    logger.info("\n[1/6] Loading data and models...")
    model_costs_raw = load_model_costs()
    train_data, eval_data, data_stats = load_dataset_with_split()

    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)
    logger.info(f"  ✓ Warmup priors: {warmup_path}")

    models = list(eval_data[0]["rewards"].keys())
    logger.info(f"  Models: {models}")
    logger.info(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # =================================================================
    # 2. PRE-COMPUTE EMBEDDINGS
    # =================================================================
    logger.info("\n[2/6] Pre-computing embeddings (one-time cost)...")
    train_embeddings = precompute_embeddings(train_data, encoder, pca)
    eval_embeddings  = precompute_embeddings(eval_data,  encoder, pca)

    # =================================================================
    # 3. LOAD EXISTING RESULTS
    # =================================================================
    logger.info("\n[3/6] Loading existing results (RouteLLM, Oracle, static baselines)...")
    existing_path = output_dir / "pareto_results.json"

    if not existing_path.exists():
        raise FileNotFoundError(
            f"Existing results not found: {existing_path}\n"
            f"Run generate_pareto_frontier.py first."
        )

    with open(existing_path) as f:
        existing = json.load(f)

    routellm_points = [(p["cost"], p["reward"])
                       for p in existing["strategies"].get("RouteLLM-MF", [])]
    existing_bandit = existing["strategies"].get("banditGPT-Hybrid", [])
    existing_bandit_points = [(p["cost"], p["reward"]) for p in existing_bandit]
    existing_bandit_stats  = [{"cost_std":   p.get("cost_std", 0.0),
                               "reward_std": p.get("reward_std", 0.0)}
                              for p in existing_bandit]

    oracle_raw = existing["strategies"]["Oracle"][0]
    oracle_point = (oracle_raw["cost"], oracle_raw["reward"])

    static_points = {}
    for key in existing["strategies"]:
        if key.startswith("Static-"):
            p = existing["strategies"][key][0]
            short = key.replace("Static-", "")
            static_points[short] = (p["cost"], p["reward"])

    rl_peak_reward = max(r for _, r in routellm_points)

    logger.info(f"  RouteLLM points:  {len(routellm_points)}")
    logger.info(f"  banditGPT points: {len(existing_bandit_points)}")
    logger.info(f"  RouteLLM peak:    {rl_peak_reward:.4f}")
    logger.info(f"  Oracle:           {oracle_point[1]:.4f}")

    # =================================================================
    # 4. PARETO FRONTIER — full λ sweep, production BanditRouter
    # =================================================================
    logger.info("\n[4/6] Running full λ sweep (production defaults, 20 trials per λ)...")

    all_lambdas = sorted(set(
        [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        + [0.25, 0.3, 0.35, 0.4, 0.45]
    ))
    logger.info(f"  λ values: {all_lambdas}")

    lambda_results = run_lambda_sweep(
        train_data, eval_data, train_embeddings, eval_embeddings,
        warmup_path, model_costs_raw,
        lambda_values=all_lambdas, n_trials=20,
    )

    all_bandit_points = [(res["mean_cost"], res["mean_reward"]) for res in lambda_results]
    all_bandit_stats  = [{"cost_std": res["std_cost"], "reward_std": res["std_reward"]}
                         for res in lambda_results]

    logger.info(f"  Total banditGPT points: {len(all_bandit_points)}")

    # =================================================================
    # 5. LEARNING CURVE — production defaults, λ=0.0
    # =================================================================
    logger.info("\n[5/6] Running learning curve (production defaults, λ=0.0, 20 trials, 11 checkpoints)...")

    learning_curve = run_learning_curve(
        train_data, eval_data, train_embeddings, eval_embeddings,
        warmup_path, model_costs_raw,
        lambda_penalty=0.0, n_trials=20,
        checkpoint_steps=[0, 25, 50, 100, 200, 300, 400, 500, 700, 900, 1121],
    )

    from scipy import stats as sp_stats
    logger.info(f"\n  {'Step':>6} | {'Reward':>12} | {'95% CI':>20}")
    logger.info("  " + "-" * 46)
    for d in learning_curve:
        t_c = sp_stats.t.ppf(0.975, df=d["n_trials"] - 1)
        ci = t_c * d["std_reward"] / np.sqrt(d["n_trials"])
        flag = " ← surpasses RouteLLM" if d["mean_reward"] >= rl_peak_reward else ""
        logger.info(f"  {d['step']:>6} | {d['mean_reward']:.4f} ± {ci:.4f} | "
                     f"[{d['mean_reward']-ci:.4f}, {d['mean_reward']+ci:.4f}]{flag}")

    # =================================================================
    # 6. GENERATE TWO-PANEL PLOT
    # =================================================================
    logger.info("\n[6/6] Generating two-panel figure...")

    plot_two_panel(
        bandit_points=all_bandit_points,
        routellm_points=routellm_points,
        oracle_point=oracle_point,
        static_points=static_points,
        learning_curve=learning_curve,
        routellm_peak_reward=rl_peak_reward,
        n_eval=data_stats["eval_prompts"],
        output_dir=output_dir,
        bandit_stats=all_bandit_stats
    )

    # =================================================================
    # SAVE RESULTS
    # =================================================================
    results_out = {
        "metadata": {
            "description": "Figure 4: Honest Pareto + Learning Curve (production BanditRouter)",
            "n_eval": data_stats["eval_prompts"],
            "n_train": data_stats["train_prompts"],
            "n_trials": 20,
            "router": "BanditRouter via create_experiment_router(alpha=2.0)",
            "alpha_schedule": "warmup: constant 2.0, tabula_rasa: 1.0→0.01",
            "corralling_lr": 0.1,
            "lambda_values": all_lambdas,
            "data_asymmetry": {
                "RouteLLM":  "100k OOD supervised pairs (Augment-100k), zero in-distribution",
                "banditGPT": "80k OOD battle priors + 1,121 in-distribution online learning"
            }
        },
        "pareto": {
            "banditGPT": [
                {"cost": float(c), "reward": float(r),
                 "cost_std": float(s.get("cost_std", 0)),
                 "reward_std": float(s.get("reward_std", 0)),
                 "lambda": float(lam)}
                for (c, r), s, lam in zip(all_bandit_points, all_bandit_stats, all_lambdas)
            ],
            "RouteLLM-MF": [{"cost": float(c), "reward": float(r)}
                            for c, r in routellm_points],
            "Oracle": {"cost": float(oracle_point[0]),
                       "reward": float(oracle_point[1])},
            "Static": {n: {"cost": float(c), "reward": float(r)}
                       for n, (c, r) in static_points.items()}
        },
        "learning_curve": learning_curve
    }

    out_path = output_dir / "figure4_results.json"
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    logger.info(f"\n✅ Results saved: {out_path}")

    # =================================================================
    # SUMMARY
    # =================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    crossover = None
    for d in learning_curve:
        if d["mean_reward"] >= rl_peak_reward:
            crossover = d["step"]
            break

    cold_start_r = learning_curve[0]["mean_reward"]
    final_r = learning_curve[-1]["mean_reward"]

    logger.info(f"\n  Production BanditRouter (α=2.0, η=0.1)")
    logger.info(f"  RouteLLM peak quality:     {rl_peak_reward:.4f}  (100k OOD pre-trained pairs)")
    logger.info(f"  banditGPT cold-start:      {cold_start_r:.4f}  (priors only, 0 in-distribution)")
    logger.info(f"  banditGPT final (1,121):   {final_r:.4f}  (after online learning)")

    if crossover is not None:
        logger.info(f"\n  ⭐ Crossover at step {crossover}:")
        logger.info(f"     {crossover} in-distribution prompts surpass 100k OOD pre-trained pairs")
    else:
        logger.info(f"\n  banditGPT did not surpass RouteLLM's peak within 1,121 steps")

    logger.info(f"\n  Outputs:")
    logger.info(f"    {output_dir / 'figure4.png'}")
    logger.info(f"    {output_dir / 'figure4_hires.png'}")
    logger.info(f"    {out_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
