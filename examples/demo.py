#!/usr/bin/env python3
"""ParetoBandit Interactive Demo — synthetic data, no downloads required.

Generates synthetic prompt and reward data that mirrors the characteristics
of the real K=3 benchmark (Llama-8B / Mistral-Large / Gemini-Pro), then
runs three scenarios that showcase the library's core capabilities:

    **Scenario 1 — Budget-Paced Routing**
        Sweeps budget targets and shows how ParetoBandit smoothly
        interpolates between cheap/low-quality and expensive/high-quality
        models while respecting an operator-set dollar budget.

    **Scenario 2 — Quality Degradation & Recovery**
        Simulates a silent quality regression on the mid-tier model,
        demonstrating that geometric forgetting detects the drop,
        redistributes traffic, and recovers when quality is restored.

    **Scenario 3 — Cost Drift & Recovery**
        Simulates a dramatic Gemini-Pro price drop, showing how the
        BudgetPacer exploits cheap premium routing during the drop and
        restores budget-compliant routing when prices are corrected.

    **Scenario 4 — Configuration Comparison**
        Varies ``alpha``, ``forgetting_factor``, and ``cost_penalty``
        to illustrate how each knob shapes the quality-cost trade-off.

All plots are saved to ``<output_dir>/`` (default ``examples/results/``).
No sentence-transformer download or external dataset is needed — the script
generates everything from scratch using NumPy.

Usage::

    # Run with defaults (~30 s on a laptop)
    python examples/demo.py

    # Override configuration via CLI
    python examples/demo.py --n-prompts 5000 --n-seeds 10 --seed 123

    # Run a single scenario
    python examples/demo.py --scenario 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from pareto_bandit.types import RouterConfig  # noqa: F401 — available for user customization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in (
    "pareto_bandit.router",
    "pareto_bandit.feature_service",
    "pareto_bandit.policy",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ═══════════════════════════════════════════════════════════════════════════

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_TEAL = "#56B4E9"
CB_GRAY = "#999999"

# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Model Definitions
# ═══════════════════════════════════════════════════════════════════════════

ARM_ORDER: List[str] = [
    "budget-llm/llama-8b",
    "midtier-llm/mistral-large",
    "premium-llm/gemini-pro",
]

ARM_SHORT: Dict[str, str] = {
    "budget-llm/llama-8b": "Llama-8B",
    "midtier-llm/mistral-large": "Mistral-Large",
    "premium-llm/gemini-pro": "Gemini-Pro",
}

ARM_COLORS: Dict[str, str] = {
    "budget-llm/llama-8b": CB_TEAL,
    "midtier-llm/mistral-large": CB_ORANGE,
    "premium-llm/gemini-pro": CB_BLUE,
}

SYNTHETIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "budget-llm/llama-8b": {
        "input_cost_per_m": 0.05,
        "output_cost_per_m": 0.10,
    },
    "midtier-llm/mistral-large": {
        "input_cost_per_m": 2.00,
        "output_cost_per_m": 6.00,
    },
    "premium-llm/gemini-pro": {
        "input_cost_per_m": 1.25,
        "output_cost_per_m": 10.00,
    },
}

# Mean per-request cost in USD, calibrated to real K=3 benchmark data:
#   Llama  ~$2.9e-5/req  (cheapest — ~400 tokens)
#   Mistral ~$5.0e-4/req  (mid-tier — variable token count)
#   Gemini  ~$1.5e-2/req  (premium — reasoning traces produce long outputs)
# This yields a ~500x spread between cheapest and most expensive, matching
# the empirical distribution in data_collection/rewards/train.jsonl.
_MEAN_COST_PER_REQ: Dict[str, float] = {
    "budget-llm/llama-8b": 2.9e-05,
    "midtier-llm/mistral-large": 5.0e-04,
    "premium-llm/gemini-pro": 1.5e-02,
}


# ═══════════════════════════════════════════════════════════════════════════
# Demo Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DemoConfig:
    """Top-level configuration for the demo.

    Modify these values to explore different operating regimes.
    All parameters can also be overridden via CLI flags.
    """

    n_prompts: int = 3000
    """Total synthetic prompts (split 2/3 train, 1/3 test)."""

    n_features: int = 25
    """Embedding dimensionality (matches PCA-25 in the paper)."""

    seed: int = 42
    """Master RNG seed for full reproducibility."""

    n_seeds: int = 5
    """Independent seeds per condition (more = tighter CIs, slower)."""

    alpha: float = 0.01
    """LinUCB exploration coefficient (default from paper)."""

    forgetting_factor: float = 0.997
    """Geometric discount on sufficient statistics (1.0 = stationary)."""

    cost_penalty: float = 0.3
    """Static cost penalty weight in the UCB score."""

    n_budget_targets: int = 7
    """Number of log-spaced budget targets for Scenario 1."""

    output_dir: str = str(PROJECT_ROOT / "examples" / "results")
    """Directory for saved plots."""

    scenario: Optional[int] = None
    """Run only this scenario (1, 2, 3, or 4).  None = run all."""


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SyntheticSplit:
    """One split of the synthetic dataset, ready for bandit simulation.

    Attributes:
        embeddings: Feature matrix of shape ``(n, n_features + 1)``.
            The last column is a bias term (always 1.0), matching the
            convention used by ``FeatureService.for_precomputed()``.
        rewards: Per-arm reward arrays ``{model_id: ndarray(n,)}``.
        costs: Per-arm cost arrays ``{model_id: ndarray(n,)}``.
    """

    embeddings: np.ndarray
    rewards: Dict[str, np.ndarray]
    costs: Dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return self.embeddings.shape[0]


def generate_synthetic_dataset(
    n_prompts: int = 3000,
    n_features: int = 25,
    seed: int = 42,
) -> Tuple[SyntheticSplit, SyntheticSplit]:
    """Build synthetic prompt features, rewards, and costs.

    The generative model uses a mixture of 5 Gaussians (one per prompt
    archetype) with per-arm linear reward functions, producing a
    contextual structure the bandit can learn to exploit.

    **Reward design:**

    - *Premium* (Gemini-like): highest average quality, dominant on hard
      prompts (reasoning, math clusters).
    - *Mid-tier* (Mistral-like): competitive across most archetypes,
      best on balanced/general prompts.
    - *Budget* (Llama-like): wins on easy/chat prompts where all models
      score well but cost matters.

    This creates a realistic quality-cost trade-off where routing adds
    value compared to always picking one model.

    Args:
        n_prompts: Total prompts to generate (split 2:1 train/test).
        n_features: Feature dimensionality (excluding bias).
        seed: RNG seed.

    Returns:
        ``(train_split, test_split)`` tuple.
    """
    rng = np.random.default_rng(seed)
    dim = n_features  # feature dims before bias

    # --- Cluster structure (5 archetypes) ---
    n_clusters = 5
    cluster_assignments = rng.integers(0, n_clusters, size=n_prompts)
    cluster_centers = rng.standard_normal((n_clusters, dim)) * 0.8

    X = np.zeros((n_prompts, dim))
    for k in range(n_clusters):
        mask = cluster_assignments == k
        n_k = mask.sum()
        X[mask] = cluster_centers[k] + rng.standard_normal((n_k, dim)) * 0.4

    # Normalize features to unit variance per dimension
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Append bias column (convention: last column = 1.0)
    X_bias = np.hstack([X, np.ones((n_prompts, 1))])

    # --- Per-arm linear reward models ---
    # Weight vectors are small so that context modulates quality
    # within a realistic band (~0.05-0.10 swing) around each arm's
    # baseline, matching the empirical reward range [0.7, 1.0] seen
    # in the real K=3 benchmark.
    #
    # Crucially, *Mistral* has the highest baseline — matching the real
    # data where Mistral-Large is the most frequently selected arm and
    # the "best arm" on the majority of prompts.  Gemini is expensive
    # and only dominant on a subset of hard reasoning/math prompts.
    arm_weights: Dict[str, np.ndarray] = {}
    arm_biases: Dict[str, float] = {}

    # Budget model: lower baseline, but competitive on "easy" clusters
    w_budget = rng.standard_normal(dim) * 0.02
    w_budget[:3] = 0.06
    arm_weights["budget-llm/llama-8b"] = w_budget
    arm_biases["budget-llm/llama-8b"] = 0.82

    # Mid-tier model: highest baseline, broadly the best arm
    w_mid = rng.standard_normal(dim) * 0.02
    w_mid[5:10] = 0.05
    arm_weights["midtier-llm/mistral-large"] = w_mid
    arm_biases["midtier-llm/mistral-large"] = 0.93

    # Premium model: strong on hard clusters only (reasoning/math)
    w_premium = rng.standard_normal(dim) * 0.02
    w_premium[10:15] = 0.07
    arm_weights["premium-llm/gemini-pro"] = w_premium
    arm_biases["premium-llm/gemini-pro"] = 0.88

    # Generate rewards with modest noise
    rewards: Dict[str, np.ndarray] = {}
    noise_scale = 0.04
    for arm_id in ARM_ORDER:
        linear = X @ arm_weights[arm_id] + arm_biases[arm_id]
        noisy = linear + rng.normal(0, noise_scale, size=n_prompts)
        rewards[arm_id] = np.clip(noisy, 0.0, 1.0)

    # Generate costs (log-normal around each arm's mean)
    costs: Dict[str, np.ndarray] = {}
    for arm_id in ARM_ORDER:
        mu = _MEAN_COST_PER_REQ[arm_id]
        costs[arm_id] = np.clip(
            rng.lognormal(
                mean=np.log(mu), sigma=0.3, size=n_prompts,
            ),
            mu * 0.3, mu * 3.0,
        )

    # --- Train / test split (2:1) ---
    n_train = int(n_prompts * 2 / 3)
    idx = rng.permutation(n_prompts)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def _make_split(indices: np.ndarray) -> SyntheticSplit:
        return SyntheticSplit(
            embeddings=X_bias[indices],
            rewards={a: rewards[a][indices] for a in ARM_ORDER},
            costs={a: costs[a][indices] for a in ARM_ORDER},
        )

    train = _make_split(train_idx)
    test = _make_split(test_idx)

    logger.info(
        "Synthetic data: %d train, %d test, %d features (+bias)",
        train.n, test.n, dim,
    )
    for arm_id in ARM_ORDER:
        logger.info(
            "  %-28s  reward=%.3f±%.3f  cost=$%.6f",
            arm_id,
            float(np.mean(rewards[arm_id])),
            float(np.std(rewards[arm_id])),
            float(np.mean(costs[arm_id])),
        )
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _create_router(
    feature_dim: int,
    *,
    alpha: float = 0.01,
    forgetting_factor: float = 0.997,
    cost_penalty: float = 0.3,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: Optional[int] = None,
) -> BanditRouter:
    """Build a router on the synthetic registry with cold-start priors."""
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=dict(SYNTHETIC_REGISTRY),
        feature_service=fs,
        context_store=store,
        priors="none",
        alpha=alpha,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
        bandit_seed=seed,
    )


@dataclass
class TrialMetrics:
    """Aggregate metrics from a single bandit trial."""

    mean_reward: float
    mean_cost: float
    model_fractions: Dict[str, float]
    per_step_models: List[str] = field(default_factory=list)
    per_step_rewards: List[float] = field(default_factory=list)
    per_step_costs: List[float] = field(default_factory=list)


def run_trial(
    train: SyntheticSplit,
    test: SyntheticSplit,
    *,
    alpha: float = 0.01,
    forgetting_factor: float = 0.997,
    cost_penalty: float = 0.3,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int = 0,
    record_steps: bool = False,
) -> TrialMetrics:
    """Run one online-learning then evaluation trial.

    The router learns on the *train* split (shuffled), then is evaluated
    on the *test* split (shuffled).  During evaluation the router
    continues to learn (standard bandit regret protocol).  Running
    reward and cost sums are accumulated regardless of ``record_steps``
    for lightweight aggregate computation.

    Args:
        train: Online-learning data.
        test: Held-out evaluation data.
        alpha: Exploration coefficient.
        forgetting_factor: Geometric discount (1.0 = stationary).
        cost_penalty: Static cost-penalty weight.
        budget_pacer: Optional budget pacer instance.
        seed: Random seed.
        record_steps: If True, record per-step trace for plotting.

    Returns:
        Aggregate and optional per-step metrics.
    """
    feature_dim = train.embeddings.shape[1]
    rng = np.random.default_rng(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        feature_dim,
        alpha=alpha,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
        seed=seed,
    )

    # --- Online learning (train split) ---
    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # --- Evaluation (test split) ---
    test_order = rng.permutation(test.n)
    step_models: List[str] = []
    step_rewards: List[float] = []
    step_costs: List[float] = []
    model_counts: Dict[str, int] = {m: 0 for m in ARM_ORDER}
    reward_sum = 0.0
    cost_sum = 0.0

    for i in test_order:
        model, log = router.route(test.embeddings[i])
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])

        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        model_counts[model] += 1
        reward_sum += reward
        cost_sum += cost

        if record_steps:
            step_models.append(model)
            step_rewards.append(reward)
            step_costs.append(cost)

    n_test = len(test_order)
    return TrialMetrics(
        mean_reward=reward_sum / n_test,
        mean_cost=cost_sum / n_test,
        model_fractions={m: cnt / n_test for m, cnt in model_counts.items()},
        per_step_models=step_models,
        per_step_rewards=step_rewards,
        per_step_costs=step_costs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — Budget-Paced Routing
# ═══════════════════════════════════════════════════════════════════════════


def _compute_budget_targets(
    train: SyntheticSplit,
    n_targets: int = 5,
) -> List[float]:
    """Log-spaced budget targets spanning arm cost extremes."""
    per_arm_means = [float(np.mean(train.costs[m])) for m in ARM_ORDER]
    lo, hi = min(per_arm_means), max(per_arm_means)
    return list(np.geomspace(lo, hi, num=n_targets))


def run_scenario_1(
    cfg: DemoConfig,
    train: SyntheticSplit,
    test: SyntheticSplit,
) -> Path:
    """Budget-paced routing sweep with Pareto frontier plot.

    Returns:
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 1: Budget-Paced LLM Routing")
    print("=" * 65)

    targets = _compute_budget_targets(train, cfg.n_budget_targets)
    target_strs = [f"${t:.2e}" if t < 1e-4 else f"${t:.5f}" for t in targets]
    print(f"  Budget targets ($/req): {target_strs}")

    # --- Fixed-model baselines ---
    baselines: List[Dict[str, Any]] = []
    for arm in ARM_ORDER:
        r = float(np.mean(test.rewards[arm]))
        c = float(np.mean(test.costs[arm]))
        baselines.append({
            "model_id": arm,
            "mean_reward": r,
            "mean_cost": c,
        })
        print(f"  Baseline {ARM_SHORT[arm]:<16s}  reward={r:.4f}  cost=${c:.6f}")

    # --- Budget sweep ---
    sweep_results: List[Dict[str, Any]] = []
    for target in targets:
        seed_rewards: List[float] = []
        seed_costs: List[float] = []
        seed_fracs: List[Dict[str, float]] = []

        for s in range(cfg.n_seeds):
            pacer = BudgetPacer(
                target_avg_spend_usd=target,
                mode=PacingMode.ADAPTIVE,
            )
            # cost_penalty=0 lets the BudgetPacer alone manage the
            # cost-quality trade-off (matching the paper's protocol).
            trial = run_trial(
                train, test,
                alpha=cfg.alpha,
                forgetting_factor=cfg.forgetting_factor,
                cost_penalty=0.0,
                budget_pacer=pacer,
                seed=cfg.seed + s,
            )
            seed_rewards.append(trial.mean_reward)
            seed_costs.append(trial.mean_cost)
            seed_fracs.append(trial.model_fractions)

        avg_fracs = {
            m: float(np.mean([f[m] for f in seed_fracs])) for m in ARM_ORDER
        }
        row = {
            "target_spend": target,
            "mean_reward": float(np.mean(seed_rewards)),
            "se_reward": float(np.std(seed_rewards, ddof=1) / np.sqrt(cfg.n_seeds))
            if cfg.n_seeds > 1 else 0.0,
            "mean_cost": float(np.mean(seed_costs)),
            "se_cost": float(np.std(seed_costs, ddof=1) / np.sqrt(cfg.n_seeds))
            if cfg.n_seeds > 1 else 0.0,
            "model_fractions": avg_fracs,
        }
        sweep_results.append(row)
        util = row["mean_cost"] / target if target > 0 else 0
        print(
            f"  target=${target:.2e}  reward={row['mean_reward']:.4f}"
            f"±{row['se_reward']:.4f}  cost=${row['mean_cost']:.2e}"
            f"  util={util:.2f}×"
        )

    # --- Plot (3-panel: Quality vs Cost, Budget Compliance, Model Mix) ---
    from matplotlib.lines import Line2D

    def _dollar_fmt(x: float, _pos: Any = None) -> str:
        if x >= 0.01:
            return f"${x:.3f}"
        if x >= 0.001:
            return f"${x:.4f}"
        return f"${x:.5f}"

    targets_arr = np.array([r["target_spend"] for r in sweep_results])
    costs_arr = np.array([r["mean_cost"] for r in sweep_results])
    rewards_arr = np.array([r["mean_reward"] for r in sweep_results])
    se_r = np.array([r["se_reward"] for r in sweep_results])
    se_c = np.array([r["se_cost"] for r in sweep_results])

    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.25, 1.0, 1.0],
        wspace=0.38, left=0.05, right=0.97, top=0.86, bottom=0.15,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel A: Quality vs Cost ("What you get for your money") ──
    ax_a.plot(
        costs_arr, rewards_arr,
        color=CB_BLUE, linewidth=2.5, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=CB_BLUE, markeredgewidth=2.0,
        zorder=6,
    )
    ax_a.errorbar(
        costs_arr, rewards_arr, yerr=se_r,
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    # Color-code points by budget utilization
    for r in sweep_results:
        util = r["mean_cost"] / r["target_spend"] if r["target_spend"] > 0 else 0
        if 0.90 <= util <= 1.10:
            mc = CB_GREEN
        elif util < 0.90:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_a.plot(
            r["mean_cost"], r["mean_reward"],
            "o", markersize=7, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=7,
        )

    for b in baselines:
        mid = b["model_id"]
        ax_a.plot(
            b["mean_cost"], b["mean_reward"],
            marker="*", markersize=14,
            markerfacecolor=ARM_COLORS[mid],
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=10, linestyle="none",
        )
        x_off = 8 if "llama" in mid else (-8 if "gemini" in mid else 8)
        ha = "left" if x_off > 0 else "right"
        ax_a.annotate(
            ARM_SHORT[mid],
            xy=(b["mean_cost"], b["mean_reward"]),
            xytext=(x_off, 4), textcoords="offset points",
            fontsize=9.5, color=ARM_COLORS[mid], fontweight="bold",
            fontstyle="italic", ha=ha,
        )

    y_lo = min(b["mean_reward"] for b in baselines) - 0.02
    y_hi = max(r["mean_reward"] for r in sweep_results) + 0.02
    ax_a.set_ylim(y_lo, y_hi)
    ax_a.set_xlabel("Cost per Request (USD)", fontsize=12)
    ax_a.set_ylabel("Mean Quality (Reward)", fontsize=12)
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax_a.grid(True, alpha=0.15, linewidth=0.5)
    ax_a.set_title("(a)  Quality vs. Budget", fontsize=13, fontweight="bold", pad=8)

    legend_a = [
        Line2D([0], [0], color=CB_BLUE, linewidth=2.5, marker="o",
               markerfacecolor="white", markeredgecolor=CB_BLUE,
               markersize=7, label="ParetoBandit"),
        Line2D([0], [0], color="none", marker="*", markerfacecolor=CB_GRAY,
               markeredgecolor="black", markersize=12,
               label="Fixed single-model"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_GREEN,
               markeredgecolor=CB_BLUE, markersize=7,
               label="On-budget (±10%)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_ORANGE,
               markeredgecolor=CB_BLUE, markersize=7,
               label="Under-budget"),
    ]
    ax_a.legend(handles=legend_a, fontsize=8, loc="lower right", framealpha=0.9)

    # ── Panel B: Budget Compliance ("The budget actually works") ──
    diag_range = np.geomspace(
        targets_arr.min() * 0.5, targets_arr.max() * 2.0, 100,
    )
    ax_b.plot(
        diag_range, diag_range,
        color=CB_GRAY, linestyle="--", linewidth=1.0, alpha=0.6,
        label="Perfect compliance", zorder=3,
    )
    ax_b.fill_between(
        diag_range, diag_range * 0.90, diag_range * 1.10,
        color=CB_GREEN, alpha=0.25, label="±10% band", zorder=2,
    )

    for r in sweep_results:
        util = r["mean_cost"] / r["target_spend"] if r["target_spend"] > 0 else 0
        if 0.90 <= util <= 1.10:
            mc = CB_GREEN
        elif util < 0.90:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_b.plot(
            r["target_spend"], r["mean_cost"],
            "o", markersize=9, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=6,
        )
        ax_b.annotate(
            f"{util:.2f}×",
            xy=(r["target_spend"], r["mean_cost"]),
            xytext=(7, 0), textcoords="offset points",
            fontsize=9, color="0.3", ha="left", va="center",
        )

    ax_b.errorbar(
        targets_arr, costs_arr, yerr=se_c,
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    ax_b.set_xlabel("Budget Target ($/request)", fontsize=12)
    ax_b.set_ylabel("Realized Cost ($/request)", fontsize=12)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    shared_lim = (targets_arr.min() * 0.5, targets_arr.max() * 2.5)
    ax_b.set_xlim(shared_lim)
    ax_b.set_ylim(shared_lim)
    ax_b.grid(True, alpha=0.15, linewidth=0.5)
    ax_b.set_title("(b)  Budget Compliance", fontsize=13, fontweight="bold", pad=8)
    ax_b.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    # ── Panel C: Model Allocation ("How it allocates") ──
    x_pos = np.arange(len(sweep_results))
    bottom = np.zeros(len(sweep_results))
    bar_width = 0.65

    for arm in ARM_ORDER:
        fracs = np.array([r["model_fractions"][arm] for r in sweep_results])
        ax_c.bar(
            x_pos, fracs, bar_width, bottom=bottom,
            label=ARM_SHORT[arm], color=ARM_COLORS[arm],
            edgecolor="white", linewidth=0.5,
        )
        bottom += fracs

    budget_labels = [_dollar_fmt(t) for t in targets]
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(budget_labels, rotation=40, ha="right", fontsize=8.5)
    ax_c.set_xlabel("Budget Target ($/request)", fontsize=12)
    ax_c.set_ylabel("Selection Fraction", fontsize=12)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax_c.set_title("(c)  Model Allocation", fontsize=13, fontweight="bold", pad=8)
    ax_c.legend(fontsize=9, loc="center left", framealpha=0.9,
                bbox_to_anchor=(0.0, 0.5))

    fig.suptitle(
        "Budget-Paced LLM Routing (Synthetic K=3)",
        fontsize=15, fontweight="bold",
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario1_budget_pacing.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2 — Quality Degradation & Recovery
# ═══════════════════════════════════════════════════════════════════════════


def _run_phased_trial(
    train: SyntheticSplit,
    test: SyntheticSplit,
    *,
    degraded_arm: str = "midtier-llm/mistral-large",
    degraded_reward: float = 0.50,
    alpha: float = 0.01,
    forgetting_factor: float = 0.997,
    cost_penalty: float = 0.3,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int = 0,
) -> Tuple[List[str], List[float], List[float], int]:
    """Three-phase trial: normal -> degradation -> recovery.

    Uses the first half of the *train* split for online learning, then
    concatenates the second half with the *test* split to form a longer
    evaluation sequence (3 equal phases).  This gives enough steps for
    adaptation dynamics to manifest clearly.

    During Phase 2, the degraded arm's rewards are replaced with a low
    constant.  Phase 3 restores normal rewards.

    Args:
        train: Online-learning data (first half used for learning,
            second half contributes to evaluation phases).
        test: Held-out evaluation data.
        degraded_arm: Model whose quality regresses in Phase 2.
        degraded_reward: Replacement reward during degradation.
        alpha: Exploration coefficient.
        forgetting_factor: Geometric discount.
        cost_penalty: Static cost-penalty weight.
        budget_pacer: Optional budget pacer for cost-constrained routing.
        seed: Random seed.

    Returns:
        ``(step_models, step_rewards, step_costs, phase_size)`` —
        per-step lists covering all three phases, plus the phase length.
    """
    feature_dim = train.embeddings.shape[1]
    rng = np.random.default_rng(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        feature_dim,
        alpha=alpha,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
        seed=seed,
    )

    # Use first half of train for learning, second half joins eval pool
    train_order = rng.permutation(train.n)
    learn_n = train.n // 2
    learn_idx = train_order[:learn_n]
    extra_eval_idx = train_order[learn_n:]

    for i in learn_idx:
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Build combined evaluation pool
    eval_emb = np.vstack([train.embeddings[extra_eval_idx], test.embeddings])
    eval_rewards = {
        a: np.concatenate([train.rewards[a][extra_eval_idx], test.rewards[a]])
        for a in ARM_ORDER
    }
    eval_costs = {
        a: np.concatenate([train.costs[a][extra_eval_idx], test.costs[a]])
        for a in ARM_ORDER
    }
    n_eval = eval_emb.shape[0]
    eval_order = rng.permutation(n_eval)

    # --- Three-phase evaluation ---
    phase_size = n_eval // 3
    phases = [
        eval_order[:phase_size],
        eval_order[phase_size:2 * phase_size],
        eval_order[2 * phase_size:3 * phase_size],
    ]

    step_models: List[str] = []
    step_rewards: List[float] = []
    step_costs: List[float] = []

    for phase_idx, phase_indices in enumerate(phases):
        for i in phase_indices:
            model, log = router.route(eval_emb[i])

            if phase_idx == 1 and model == degraded_arm:
                reward = degraded_reward
            else:
                reward = float(eval_rewards[model][i])
            cost = float(eval_costs[model][i])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            step_models.append(model)
            step_rewards.append(reward)
            step_costs.append(cost)

    return step_models, step_rewards, step_costs, phase_size


@dataclass
class _AveragedCurves:
    """Seed-averaged rolling curves for one condition in a phased trial."""

    frac_gemini: np.ndarray
    mean_reward: np.ndarray
    mean_cost: np.ndarray
    phase_size: int


def _phase_geometry(train: SyntheticSplit, test: SyntheticSplit) -> Tuple[int, int, int]:
    """Compute (phase_size, total_steps, window) for the phased trial layout.

    These depend only on data dimensions, not on seed or hyperparameters,
    so they can be computed once before running any trials.
    """
    learn_n = train.n // 2
    n_eval = (train.n - learn_n) + test.n
    phase_size = n_eval // 3
    total_steps = 3 * phase_size
    window = max(20, total_steps // 30)
    return phase_size, total_steps, window


_BUDGET_LABELS = ["tight", "moderate", "loose"]
_BUDGET_COLORS: Dict[str, str] = {
    "tight": CB_RED,
    "moderate": CB_BLUE,
    "loose": CB_PURPLE,
}
_BUDGET_NICE: Dict[str, str] = {}  # populated at runtime with dollar amounts

GEMINI_ARM = "premium-llm/gemini-pro"
_NAIVE_COLOR = CB_GRAY
_UNCONSTRAINED_COLOR = CB_GREEN
_PHASE2_SHADE = CB_RED
_PHASE_LABELS_S2 = ["Normal", "Mistral Failure", "Recovered"]


def _compute_degradation_budgets(train: SyntheticSplit) -> Dict[str, float]:
    """Three budget targets spanning the cost range for degradation trials.

    Returns tight / moderate / loose targets analogous to Experiment 03.
    """
    mean_costs = [float(np.mean(train.costs[m])) for m in ARM_ORDER]
    lo, hi = min(mean_costs), max(mean_costs)
    tight, moderate, loose = np.geomspace(lo * 10, hi * 0.15, num=3)
    return {"tight": tight, "moderate": moderate, "loose": loose}


def _rolling_mean(arr: List[float], w: int) -> np.ndarray:
    a = np.array(arr, dtype=float)
    return np.convolve(a, np.ones(w) / w, mode="valid")


def _rolling_fraction(models: List[str], arm: str, w: int) -> np.ndarray:
    indicator = np.array([1.0 if m == arm else 0.0 for m in models])
    return np.convolve(indicator, np.ones(w) / w, mode="valid")


def _add_phase_shading_s2(
    ax: plt.Axes,
    phase_boundaries: List[int],
) -> None:
    """Shade Phase 2 and label all three phases (Exp 03-style)."""
    from matplotlib.transforms import blended_transform_factory

    ax.axvspan(
        phase_boundaries[0], phase_boundaries[1],
        alpha=0.07, color=_PHASE2_SHADE, zorder=0,
    )
    for b in phase_boundaries[:2]:
        ax.axvline(
            b, color="black", linestyle="--",
            linewidth=1.2, alpha=0.5, zorder=1,
        )
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    midpoints = [
        phase_boundaries[0] / 2,
        (phase_boundaries[0] + phase_boundaries[1]) / 2,
        (phase_boundaries[1] + phase_boundaries[2]) / 2,
    ]
    for mid, label in zip(midpoints, _PHASE_LABELS_S2):
        ax.text(
            mid, 0.97, label,
            transform=trans, ha="center", va="top",
            fontsize=10, fontweight="bold", color="#333333",
        )


def _run_multi_seed_phased(
    trial_fn: Any,
    trial_kwargs: Dict[str, Any],
    n_seeds: int,
    base_seed: int,
    window: int,
    target_arm: str,
) -> _AveragedCurves:
    """Run a phased trial function over multiple seeds and average rolling curves.

    Args:
        trial_fn: Either ``_run_phased_trial`` or ``_run_cost_drift_trial``.
        trial_kwargs: Keyword arguments forwarded to *trial_fn* (excluding ``seed``).
        n_seeds: Number of independent seeds.
        base_seed: Starting seed (incremented by 1 per trial).
        window: Rolling-window width for smoothing.
        target_arm: Arm whose selection fraction is tracked (e.g. Gemini-Pro).

    Returns:
        Seed-averaged rolling curves.
    """
    all_frac: List[np.ndarray] = []
    all_rwd: List[np.ndarray] = []
    all_cost: List[np.ndarray] = []
    phase_size = 0

    for s in range(n_seeds):
        models, rewards, costs, ps = trial_fn(seed=base_seed + s, **trial_kwargs)
        phase_size = ps
        all_frac.append(_rolling_fraction(models, target_arm, window))
        all_rwd.append(_rolling_mean(rewards, window))
        all_cost.append(_rolling_mean(costs, window))

    return _AveragedCurves(
        frac_gemini=np.mean(all_frac, axis=0),
        mean_reward=np.mean(all_rwd, axis=0),
        mean_cost=np.mean(all_cost, axis=0),
        phase_size=phase_size,
    )


def _plot_phased_3panel(
    conditions: Dict[str, _AveragedCurves],
    budget_targets: Dict[str, float],
    phase_boundaries: List[int],
    window: int,
    x_axis: np.ndarray,
    *,
    phase_labels: List[str],
    shade_color: str,
    suptitle: str,
    out_path: Path,
) -> Path:
    """Draw the Exp 02 / Exp 03-style 3x1 stacked figure.

    Conditions whose keys start with ``"ParetoBandit"`` are drawn as solid
    lines colour-coded by budget label.  ``"Naive Bandit"`` is gray dashed
    and ``"Unconstrained"`` is green dash-dot.

    Args:
        conditions: ``{label: _AveragedCurves}`` mapping.
        budget_targets: ``{budget_label: target_spend}`` for cost panel.
        phase_boundaries: ``[p1_end, p2_end, p3_end]`` step indices.
        window: Rolling-window width (used in panel title).
        x_axis: Shared x-axis array.
        phase_labels: Three-element list of phase names.
        shade_color: Fill colour for the Phase 2 band.
        suptitle: Figure super-title.
        out_path: Where to save the PNG.

    Returns:
        *out_path* after saving.
    """
    from matplotlib.transforms import blended_transform_factory

    def _add_shading(ax: plt.Axes) -> None:
        ax.axvspan(phase_boundaries[0], phase_boundaries[1],
                   alpha=0.07, color=shade_color, zorder=0)
        for b in phase_boundaries[:2]:
            ax.axvline(b, color="black", linestyle="--",
                       linewidth=1.2, alpha=0.5, zorder=1)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        mids = [
            phase_boundaries[0] / 2,
            (phase_boundaries[0] + phase_boundaries[1]) / 2,
            (phase_boundaries[1] + phase_boundaries[2]) / 2,
        ]
        for mid, lab in zip(mids, phase_labels):
            ax.text(mid, 0.97, lab, transform=trans, ha="center",
                    va="top", fontsize=10, fontweight="bold", color="#333333")

    fig, (ax_gem, ax_rwd, ax_cost) = plt.subplots(
        3, 1, figsize=(10, 12), sharex=True,
    )

    def _plot(ax: plt.Axes, series: np.ndarray, label: str, color: str,
              ls: str = "-", lw: float = 2.2, zo: int = 4) -> None:
        ax.plot(x_axis, series, color=color, linewidth=lw,
                linestyle=ls, label=label, zorder=zo)

    # Categorise conditions by role
    for cond_label, curves in conditions.items():
        if cond_label.startswith("ParetoBandit"):
            bl = cond_label.split("(")[1].rstrip(")")
            color = _BUDGET_COLORS[bl]
            nice = _BUDGET_NICE[bl]
            _plot(ax_gem, curves.frac_gemini, nice, color)
            _plot(ax_rwd, curves.mean_reward, nice, color)
            _plot(ax_cost, curves.mean_cost, nice, color)
        elif cond_label.startswith("Naive"):
            _plot(ax_gem, curves.frac_gemini, cond_label,
                  _NAIVE_COLOR, ls="--", lw=1.8, zo=3)
            _plot(ax_rwd, curves.mean_reward, cond_label,
                  _NAIVE_COLOR, ls="--", lw=1.8, zo=3)
            _plot(ax_cost, curves.mean_cost, cond_label,
                  _NAIVE_COLOR, ls="--", lw=1.8, zo=3)
        elif cond_label == "Unconstrained":
            uc_label = r"Unconstrained ($\lambda_s{=}0$)"
            _plot(ax_gem, curves.frac_gemini, uc_label,
                  _UNCONSTRAINED_COLOR, ls="-.", lw=2.0, zo=3)
            _plot(ax_rwd, curves.mean_reward, uc_label,
                  _UNCONSTRAINED_COLOR, ls="-.", lw=2.0, zo=3)
            _plot(ax_cost, curves.mean_cost, uc_label,
                  _UNCONSTRAINED_COLOR, ls="-.", lw=2.0, zo=3)

    # Panel (a) formatting
    _add_shading(ax_gem)
    ax_gem.set_ylabel("Fraction", fontsize=12)
    ax_gem.set_ylim(-0.02, 1.02)
    ax_gem.grid(True, alpha=0.2, linewidth=0.5)
    ax_gem.set_title("(a)  Gemini-Pro Selection Fraction",
                     fontsize=13, fontweight="bold", pad=8)
    ax_gem.tick_params(labelbottom=False)

    # Panel (b) formatting
    _add_shading(ax_rwd)
    ax_rwd.set_ylabel("Mean Reward", fontsize=12)
    ax_rwd.grid(True, alpha=0.2, linewidth=0.5)
    ax_rwd.set_title(f"(b)  Windowed Mean Reward (window={window})",
                     fontsize=13, fontweight="bold", pad=8)
    ax_rwd.tick_params(labelbottom=False)

    # Panel (c) formatting + budget-target lines
    target_label_data: List[Tuple[float, str, str]] = []
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        color = _BUDGET_COLORS[bl]
        ax_cost.axhline(bt, color=color, linestyle=":", linewidth=1.4,
                        alpha=0.6, zorder=1)
        target_label_data.append((bt, bl, color))

    y_lo, y_hi = ax_cost.get_ylim()
    min_sep = 0.045 * (y_hi - y_lo)
    sorted_tl = sorted(target_label_data, key=lambda x: x[0])
    adj_y = [e[0] for e in sorted_tl]
    for i in range(1, len(adj_y)):
        if adj_y[i] - adj_y[i - 1] < min_sep:
            mid = (adj_y[i] + adj_y[i - 1]) / 2
            adj_y[i - 1] = mid - min_sep / 2
            adj_y[i] = mid + min_sep / 2
    for (_, blabel, color), y_pos in zip(sorted_tl, adj_y):
        ax_cost.text(
            1.01, y_pos, f"{blabel} target",
            transform=blended_transform_factory(ax_cost.transAxes, ax_cost.transData),
            fontsize=9, color=color, va="center", ha="left",
            fontweight="bold", clip_on=False,
        )

    _add_shading(ax_cost)
    ax_cost.set_ylabel("$/request", fontsize=12)
    ax_cost.set_xlabel("Prompts Routed", fontsize=12)
    ax_cost.grid(True, alpha=0.2, linewidth=0.5)
    ax_cost.set_title("(c)  Windowed Avg Cost / Request",
                      fontsize=13, fontweight="bold", pad=8)

    handles, labels = ax_gem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 3), fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.subplots_adjust(hspace=0.15)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_scenario_2(
    cfg: DemoConfig,
    train: SyntheticSplit,
    test: SyntheticSplit,
) -> Path:
    """Quality degradation and recovery (Exp 03-style 3-panel figure).

    Runs five conditions (ParetoBandit x3 budgets, Naive Bandit,
    Unconstrained) through a three-phase simulation, averaging over
    ``cfg.n_seeds`` independent seeds for smooth curves.

    Returns:
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 2: Quality Degradation & Recovery")
    print("=" * 65)

    degraded_arm = "midtier-llm/mistral-large"
    degraded_reward = 0.75
    phase_size, total_steps, window = _phase_geometry(train, test)

    budget_targets = _compute_degradation_budgets(train)
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        _BUDGET_NICE[bl] = rf"{bl.title()} ($B{{=}}\${bt:.1e}$)"
        print(f"  Budget {bl:<10s} = ${bt:.2e}/req")

    conditions: Dict[str, _AveragedCurves] = {}

    for bl in _BUDGET_LABELS:
        conditions[f"ParetoBandit ({bl})"] = _run_multi_seed_phased(
            trial_fn=_run_phased_trial,
            trial_kwargs=dict(
                train=train, test=test,
                degraded_arm=degraded_arm,
                degraded_reward=degraded_reward,
                alpha=cfg.alpha,
                forgetting_factor=cfg.forgetting_factor,
                cost_penalty=0.0,
                budget_pacer=BudgetPacer(
                    target_avg_spend_usd=budget_targets[bl],
                    mode=PacingMode.ADAPTIVE,
                ),
            ),
            n_seeds=cfg.n_seeds, base_seed=cfg.seed,
            window=window, target_arm=GEMINI_ARM,
        )

    conditions["Naive Bandit (moderate)"] = _run_multi_seed_phased(
        trial_fn=_run_phased_trial,
        trial_kwargs=dict(
            train=train, test=test,
            degraded_arm=degraded_arm,
            degraded_reward=degraded_reward,
            alpha=cfg.alpha,
            forgetting_factor=1.0,
            cost_penalty=0.0,
            budget_pacer=BudgetPacer(
                target_avg_spend_usd=budget_targets["moderate"],
                mode=PacingMode.ADAPTIVE,
            ),
        ),
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_phased_trial,
        trial_kwargs=dict(
            train=train, test=test,
            degraded_arm=degraded_arm,
            degraded_reward=degraded_reward,
            alpha=cfg.alpha,
            forgetting_factor=cfg.forgetting_factor,
            cost_penalty=0.0,
        ),
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    phase_boundaries = [phase_size, 2 * phase_size, 3 * phase_size]
    x_axis = np.arange(total_steps - window + 1) + window // 2

    # Print per-phase reward summary (from unconstrained seed 0 as reference)
    print()
    for cond_name, curves in conditions.items():
        ps = curves.phase_size
        r = curves.mean_reward
        p1 = r[: ps - window + 1]
        p2 = r[ps - window + 1: 2 * ps - window + 1]
        p3 = r[2 * ps - window + 1:]
        print(f"  {cond_name:<30s}  P1={np.mean(p1):.4f}  "
              f"P2={np.mean(p2):.4f}  P3={np.mean(p3):.4f}")

    out_path = Path(cfg.output_dir) / "scenario2_quality_degradation.png"
    _plot_phased_3panel(
        conditions, budget_targets, phase_boundaries, window, x_axis,
        phase_labels=_PHASE_LABELS_S2, shade_color=_PHASE2_SHADE,
        suptitle=(f"Quality Degradation & Recovery — {ARM_SHORT[degraded_arm]} "
                  f"reward drops to {degraded_reward:.2f} in Phase 2"),
        out_path=out_path,
    )
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — Cost Drift & Recovery (Exp 02 analogue)
# ═══════════════════════════════════════════════════════════════════════════

# In Phase 2, Gemini-Pro's pricing drops by this factor (e.g. 0.02 → 50x
# cheaper).  The BudgetPacer should exploit the cheap premium model, then
# return to the original routing mix when prices are restored in Phase 3.
_GEMINI_COST_SCALE_PHASE2 = 0.02

_PHASE_LABELS_S3 = ["Normal", "Price Drop", "Restored"]


def _run_cost_drift_trial(
    train: SyntheticSplit,
    test: SyntheticSplit,
    *,
    cost_drift_arm: str = "premium-llm/gemini-pro",
    cost_scale: float = _GEMINI_COST_SCALE_PHASE2,
    alpha: float = 0.01,
    forgetting_factor: float = 0.997,
    cost_penalty: float = 0.0,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int = 0,
) -> Tuple[List[str], List[float], List[float], int]:
    """Three-phase cost drift trial: normal -> price drop -> restored.

    Phase 2 scales the ``cost_drift_arm``'s per-request costs by
    ``cost_scale`` (e.g. 0.02 = 50x cheaper) and updates the router's
    pricing registry.  Phase 3 restores original pricing.

    Args:
        train: Online-learning data (first half for learning).
        test: Held-out evaluation data.
        cost_drift_arm: Model whose pricing changes in Phase 2.
        cost_scale: Multiplicative factor applied to costs in Phase 2.
        alpha: Exploration coefficient.
        forgetting_factor: Geometric discount.
        cost_penalty: Static cost-penalty weight.
        budget_pacer: Optional budget pacer for cost-constrained routing.
        seed: Random seed.

    Returns:
        ``(step_models, step_rewards, step_costs, phase_size)`` —
        per-step lists covering all three phases, plus the phase length.
    """
    feature_dim = train.embeddings.shape[1]
    rng = np.random.default_rng(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_router(
        feature_dim,
        alpha=alpha,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
        seed=seed,
    )

    # Learn on first half of train
    train_order = rng.permutation(train.n)
    learn_n = train.n // 2
    learn_idx = train_order[:learn_n]
    extra_eval_idx = train_order[learn_n:]

    for i in learn_idx:
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Build combined evaluation pool
    eval_emb = np.vstack([train.embeddings[extra_eval_idx], test.embeddings])
    eval_rewards = {
        a: np.concatenate([train.rewards[a][extra_eval_idx], test.rewards[a]])
        for a in ARM_ORDER
    }
    eval_costs_normal = {
        a: np.concatenate([train.costs[a][extra_eval_idx], test.costs[a]])
        for a in ARM_ORDER
    }
    # Phase 2 cost array — only the drift arm is scaled
    eval_costs_cheap = dict(eval_costs_normal)
    eval_costs_cheap[cost_drift_arm] = (
        eval_costs_normal[cost_drift_arm] * cost_scale
    )

    n_eval = eval_emb.shape[0]
    eval_order = rng.permutation(n_eval)

    phase_size = n_eval // 3
    phases = [
        eval_order[:phase_size],
        eval_order[phase_size:2 * phase_size],
        eval_order[2 * phase_size:3 * phase_size],
    ]

    # Original registry pricing for the drift arm
    orig_input = SYNTHETIC_REGISTRY[cost_drift_arm]["input_cost_per_m"]
    orig_output = SYNTHETIC_REGISTRY[cost_drift_arm]["output_cost_per_m"]

    step_models: List[str] = []
    step_rewards: List[float] = []
    step_costs: List[float] = []

    for phase_idx, phase_indices in enumerate(phases):
        # At Phase 2 boundary: apply price drop
        if phase_idx == 1:
            router.update_model_pricing(
                cost_drift_arm,
                input_cost_per_m=orig_input * cost_scale,
                output_cost_per_m=orig_output * cost_scale,
            )
        # At Phase 3 boundary: restore original pricing
        elif phase_idx == 2:
            router.update_model_pricing(
                cost_drift_arm,
                input_cost_per_m=orig_input,
                output_cost_per_m=orig_output,
            )

        cost_table = eval_costs_cheap if phase_idx == 1 else eval_costs_normal

        for i in phase_indices:
            model, log = router.route(eval_emb[i])
            reward = float(eval_rewards[model][i])
            cost = float(cost_table[model][i])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            step_models.append(model)
            step_rewards.append(reward)
            step_costs.append(cost)

    return step_models, step_rewards, step_costs, phase_size


def run_scenario_3(
    cfg: DemoConfig,
    train: SyntheticSplit,
    test: SyntheticSplit,
) -> Path:
    """Cost drift and recovery (Exp 02-style 3-panel figure).

    Simulates a Gemini-Pro price drop in Phase 2 (normal -> cheap -> restored),
    averaging over ``cfg.n_seeds`` independent seeds for smooth curves.

    Returns:
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 3: Cost Drift & Recovery (Gemini-Pro Price Drop)")
    print("=" * 65)

    cost_drift_arm = "premium-llm/gemini-pro"
    cost_scale = _GEMINI_COST_SCALE_PHASE2
    print(f"  Gemini-Pro cost multiplier in Phase 2: {cost_scale}x "
          f"(~{1/cost_scale:.0f}x cheaper)")

    phase_size, total_steps, window = _phase_geometry(train, test)

    budget_targets = _compute_degradation_budgets(train)
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        _BUDGET_NICE[bl] = rf"{bl.title()} ($B{{=}}\${bt:.1e}$)"
        print(f"  Budget {bl:<10s} = ${bt:.2e}/req")

    conditions: Dict[str, _AveragedCurves] = {}

    for bl in _BUDGET_LABELS:
        conditions[f"ParetoBandit ({bl})"] = _run_multi_seed_phased(
            trial_fn=_run_cost_drift_trial,
            trial_kwargs=dict(
                train=train, test=test,
                cost_drift_arm=cost_drift_arm,
                cost_scale=cost_scale,
                alpha=cfg.alpha,
                forgetting_factor=cfg.forgetting_factor,
                cost_penalty=0.0,
                budget_pacer=BudgetPacer(
                    target_avg_spend_usd=budget_targets[bl],
                    mode=PacingMode.ADAPTIVE,
                ),
            ),
            n_seeds=cfg.n_seeds, base_seed=cfg.seed,
            window=window, target_arm=GEMINI_ARM,
        )

    conditions["Naive Bandit (moderate)"] = _run_multi_seed_phased(
        trial_fn=_run_cost_drift_trial,
        trial_kwargs=dict(
            train=train, test=test,
            cost_drift_arm=cost_drift_arm,
            cost_scale=cost_scale,
            alpha=cfg.alpha,
            forgetting_factor=1.0,
            cost_penalty=0.0,
            budget_pacer=BudgetPacer(
                target_avg_spend_usd=budget_targets["moderate"],
                mode=PacingMode.ADAPTIVE,
            ),
        ),
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_cost_drift_trial,
        trial_kwargs=dict(
            train=train, test=test,
            cost_drift_arm=cost_drift_arm,
            cost_scale=cost_scale,
            alpha=cfg.alpha,
            forgetting_factor=cfg.forgetting_factor,
            cost_penalty=0.0,
        ),
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    phase_boundaries = [phase_size, 2 * phase_size, 3 * phase_size]
    x_axis = np.arange(total_steps - window + 1) + window // 2

    print()
    for cond_name, curves in conditions.items():
        ps = curves.phase_size
        r = curves.mean_reward
        p1 = r[: ps - window + 1]
        p2 = r[ps - window + 1: 2 * ps - window + 1]
        p3 = r[2 * ps - window + 1:]
        c = curves.mean_cost
        c1, c2, c3 = c[: ps - window + 1], c[ps - window + 1: 2 * ps - window + 1], c[2 * ps - window + 1:]
        print(f"  {cond_name:<30s}  "
              f"P1: R={np.mean(p1):.4f} C=${np.mean(c1):.2e}  "
              f"P2: R={np.mean(p2):.4f} C=${np.mean(c2):.2e}  "
              f"P3: R={np.mean(p3):.4f} C=${np.mean(c3):.2e}")

    out_path = Path(cfg.output_dir) / "scenario3_cost_drift.png"
    _plot_phased_3panel(
        conditions, budget_targets, phase_boundaries, window, x_axis,
        phase_labels=_PHASE_LABELS_S3, shade_color=CB_GREEN,
        suptitle=("Cost Drift & Recovery — Gemini-Pro pricing drops "
                  f"{1/cost_scale:.0f}x in Phase 2"),
        out_path=out_path,
    )
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4 — Configuration Comparison
# ═══════════════════════════════════════════════════════════════════════════


def run_scenario_4(
    cfg: DemoConfig,
    train: SyntheticSplit,
    test: SyntheticSplit,
) -> Path:
    """Compare key configuration knobs on the quality-cost frontier.

    Sweeps three parameters (``alpha``, ``forgetting_factor``,
    ``cost_penalty``) one at a time while holding the others at their
    defaults. Each parameter is tested at three levels.

    Returns:
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 4: Configuration Comparison")
    print("=" * 65)

    param_sweeps = {
        "alpha (exploration)": {
            "param": "alpha",
            "values": [0.001, 0.01, 0.1],
            "labels": ["α=0.001\n(exploit)", "α=0.01\n(default)", "α=0.1\n(explore)"],
            "defaults": {"forgetting_factor": cfg.forgetting_factor,
                         "cost_penalty": cfg.cost_penalty},
        },
        "forgetting_factor (adaptation)": {
            "param": "forgetting_factor",
            "values": [1.0, 0.997, 0.99],
            "labels": ["γ=1.0\n(stationary)", "γ=0.997\n(default)", "γ=0.99\n(aggressive)"],
            "defaults": {"alpha": cfg.alpha, "cost_penalty": cfg.cost_penalty},
        },
        "cost_penalty (cost aversion)": {
            "param": "cost_penalty",
            "values": [0.0, 0.3, 1.0],
            "labels": ["λ_c=0.0\n(quality only)", "λ_c=0.3\n(default)", "λ_c=1.0\n(cost focus)"],
            "defaults": {"alpha": cfg.alpha,
                         "forgetting_factor": cfg.forgetting_factor},
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.96, top=0.86, bottom=0.18)

    for ax, (sweep_name, sweep_cfg) in zip(axes, param_sweeps.items()):
        param_key = sweep_cfg["param"]
        values = sweep_cfg["values"]
        labels = sweep_cfg["labels"]
        defaults = sweep_cfg["defaults"]

        print(f"\n  Sweep: {sweep_name}")
        bar_rewards: List[float] = []
        bar_costs: List[float] = []
        bar_fracs: List[Dict[str, float]] = []

        for val in values:
            kwargs = dict(defaults)
            kwargs[param_key] = val

            seed_r: List[float] = []
            seed_c: List[float] = []
            seed_f: List[Dict[str, float]] = []

            for s in range(cfg.n_seeds):
                trial = run_trial(
                    train, test,
                    seed=cfg.seed + s,
                    **kwargs,
                )
                seed_r.append(trial.mean_reward)
                seed_c.append(trial.mean_cost)
                seed_f.append(trial.model_fractions)

            avg_r = float(np.mean(seed_r))
            avg_c = float(np.mean(seed_c))
            avg_f = {m: float(np.mean([f[m] for f in seed_f])) for m in ARM_ORDER}

            bar_rewards.append(avg_r)
            bar_costs.append(avg_c)
            bar_fracs.append(avg_f)
            print(f"    {param_key}={val:<8}  reward={avg_r:.4f}  cost=${avg_c:.6f}")

        # Stacked bar for model mix
        x_pos = np.arange(len(values))
        bottom = np.zeros(len(values))
        bar_width = 0.55

        for arm in ARM_ORDER:
            fracs = np.array([f[arm] for f in bar_fracs])
            ax.bar(
                x_pos, fracs, bar_width, bottom=bottom,
                label=ARM_SHORT[arm], color=ARM_COLORS[arm],
                edgecolor="white", linewidth=0.5,
            )
            bottom += fracs

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Selection Fraction", fontsize=11)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_title(sweep_name, fontsize=11, fontweight="bold", pad=6)

        # Annotate reward/cost on each bar
        for i in range(len(values)):
            ax.text(
                x_pos[i], 1.01,
                f"R={bar_rewards[i]:.3f}\nC=${bar_costs[i]:.5f}",
                ha="center", va="bottom", fontsize=7.5, color="0.3",
            )

    # Shared legend at top
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.99))

    fig.suptitle(
        "Configuration Comparison — How Each Knob Shapes the Model Mix",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scenario4_config_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI and Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> DemoConfig:
    """Parse CLI arguments into a DemoConfig."""
    parser = argparse.ArgumentParser(
        description="ParetoBandit interactive demo — runs on synthetic data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-prompts", type=int, default=3000,
                        help="Total synthetic prompts (default: 3000)")
    parser.add_argument("--n-features", type=int, default=25,
                        help="Feature dimensionality (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master RNG seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds per condition (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="LinUCB exploration (default: 0.01)")
    parser.add_argument("--forgetting-factor", type=float, default=0.997,
                        help="Geometric discount (default: 0.997)")
    parser.add_argument("--cost-penalty", type=float, default=0.3,
                        help="Cost penalty weight (default: 0.3)")
    parser.add_argument("--n-budget-targets", type=int, default=7,
                        help="Budget sweep points (default: 5)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "examples" / "results"),
                        help="Output directory for plots")
    parser.add_argument("--scenario", type=int, default=None, choices=[1, 2, 3, 4],
                        help="Run only this scenario (default: all)")
    args = parser.parse_args()
    return DemoConfig(
        n_prompts=args.n_prompts,
        n_features=args.n_features,
        seed=args.seed,
        n_seeds=args.n_seeds,
        alpha=args.alpha,
        forgetting_factor=args.forgetting_factor,
        cost_penalty=args.cost_penalty,
        n_budget_targets=args.n_budget_targets,
        output_dir=args.output_dir,
        scenario=args.scenario,
    )


def main() -> None:
    """Entry point: generate data, run scenarios, save plots."""
    cfg = parse_args()
    t0 = time.time()

    print("=" * 65)
    print("  ParetoBandit Interactive Demo")
    print("=" * 65)
    print(f"  Prompts:     {cfg.n_prompts}")
    print(f"  Features:    {cfg.n_features}")
    print(f"  Seeds/cond:  {cfg.n_seeds}")
    print(f"  Seed:        {cfg.seed}")
    print(f"  Output:      {cfg.output_dir}")
    print()

    train, test = generate_synthetic_dataset(
        n_prompts=cfg.n_prompts,
        n_features=cfg.n_features,
        seed=cfg.seed,
    )

    saved: List[Path] = []
    run_all = cfg.scenario is None

    if run_all or cfg.scenario == 1:
        saved.append(run_scenario_1(cfg, train, test))

    if run_all or cfg.scenario == 2:
        saved.append(run_scenario_2(cfg, train, test))

    if run_all or cfg.scenario == 3:
        saved.append(run_scenario_3(cfg, train, test))

    if run_all or cfg.scenario == 4:
        saved.append(run_scenario_4(cfg, train, test))

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  Demo complete in {elapsed:.1f}s")
    print("  Saved plots:")
    for p in saved:
        print(f"    {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()
