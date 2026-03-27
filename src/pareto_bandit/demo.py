"""ParetoBandit Interactive Demo.

Loads evaluation data from the shipped K=3 test holdout (1,824 prompts
from public benchmarks), embeds them with the library's default
SentenceTransformer + PCA pipeline, and runs four scenarios that
showcase core capabilities:

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

All plots are saved to ``<output_dir>/`` (default ``./demo_results/``).

Requires ``pip install paretobandit[demo]``.  Pass
``--encoder-model`` to swap the embedding backbone (a matching PCA
artifact is then required via ``--pca-path``, or raw embeddings are
used when omitted).

Usage::

    # Via CLI entry point (after pip install paretobandit[demo])
    paretobandit-demo

    # Use fewer prompts for a quick test
    paretobandit-demo --n-prompts 500

    # Run a single scenario
    paretobandit-demo --scenario 2

    # Use a custom JSONL reward file
    paretobandit-demo --prompts-file path/to/my_rewards.jsonl

    # Use a different SentenceTransformer encoder (raw embeddings, no PCA)
    paretobandit-demo --encoder-model all-mpnet-base-v2

    # Use a different encoder with a custom PCA artifact
    paretobandit-demo --encoder-model all-mpnet-base-v2 --pca-path my_pca.joblib

    # Via the examples/ wrapper (git clone only)
    python examples/demo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency guard: matplotlib is required only at runtime.
# The module remains importable without it so that help text and
# introspection still work; main() exits cleanly if it is missing.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    DEFAULT_MODEL_REGISTRY_PATH,
    K3_ARM_ORDER,
)
from pareto_bandit.data import get_example_holdout_path
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Visual Constants — Colorblind-safe palette (Wong, Nature Methods 2011)
# ═══════════════════════════════════════════════════════════════════════════

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_TEAL = "#56B4E9"
CB_GRAY = "#999999"

# ═══════════════════════════════════════════════════════════════════════════
# K=3 Model Definitions (real IDs — matches priors & holdout data)
# ═══════════════════════════════════════════════════════════════════════════

ARM_ORDER: list[str] = list(K3_ARM_ORDER)

ARM_SHORT: dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

ARM_COLORS: dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": CB_TEAL,
    "mistralai/mistral-large-2512": CB_ORANGE,
    "google/gemini-2.5-pro": CB_BLUE,
}

GEMINI_ARM: str = "google/gemini-2.5-pro"


def _load_k3_registry() -> dict[str, dict[str, object]]:
    """Load the K=3 model registry from the shipped ``models.json``."""
    with open(DEFAULT_MODEL_REGISTRY_PATH) as fh:
        data = json.load(fh)
    arm_set = set(ARM_ORDER)
    return {
        m["model_id"]: {
            "model_id": m["model_id"],
            "display_name": m.get("display", m["model_id"]),
            "input_cost_per_m": m["input_cost_per_m"],
            "output_cost_per_m": m["output_cost_per_m"],
        }
        for m in data["models"]
        if m["model_id"] in arm_set
    }


MODEL_REGISTRY: dict[str, dict[str, object]] = _load_k3_registry()

# Empirical mean per-request cost (USD) from the K=3 benchmark.
#   Llama  ~$2.9e-5/req  (cheapest — ~400 tokens)
#   Mistral ~$5.0e-4/req  (mid-tier — variable token count)
#   Gemini  ~$1.5e-2/req  (premium — reasoning traces yield long outputs)
_MEAN_COST_PER_REQ: dict[str, float] = {
    "meta-llama/llama-3.1-8b-instruct": 2.9e-05,
    "mistralai/mistral-large-2512": 5.0e-04,
    "google/gemini-2.5-pro": 1.5e-02,
}

# ═══════════════════════════════════════════════════════════════════════════
# Data Split
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DataSplit:
    """One split (train or test) of the evaluation dataset.

    Parameters
    ----------
    embeddings : np.ndarray
        Feature matrix of shape ``(n, d + 1)`` where *d* is the
        embedding dimension.  The last column is a constant bias term
        (always 1.0), matching ``FeatureService.for_precomputed()``.
    rewards : Dict[str, np.ndarray]
        Per-arm reward vectors ``{model_id: ndarray(n,)}``.
    costs : Dict[str, np.ndarray]
        Per-arm cost vectors ``{model_id: ndarray(n,)}``.
    """

    embeddings: np.ndarray
    rewards: dict[str, np.ndarray]
    costs: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        """Number of samples in this split."""
        return self.embeddings.shape[0]


# ═══════════════════════════════════════════════════════════════════════════
# Demo Configuration
# ═══════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def _default_holdout_path() -> str:
    """Resolve the shipped holdout JSONL path (cached, no repeated I/O)."""
    return str(get_example_holdout_path())


@dataclass
class DemoConfig:
    """Top-level configuration for the demo.

    Modify these values to explore different operating regimes.
    All parameters can also be overridden via CLI flags.
    """

    n_prompts: int = 1000
    """Prompts to sample from the data file."""

    seed: int = 42
    """Master RNG seed for full reproducibility."""

    n_seeds: int = 5
    """Independent seeds per condition (more seeds = tighter CIs, slower)."""

    alpha: float = BEST_K3_HPARAMS["alpha"]
    """LinUCB exploration coefficient (from tuned K=3 hyperparameters)."""

    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"]
    """Geometric discount on sufficient statistics (1.0 = stationary)."""

    cost_penalty: float = 0.3
    """Static cost-penalty weight in the UCB score."""

    n_budget_targets: int = 7
    """Number of log-spaced budget targets for Scenario 1."""

    output_dir: str = "demo_results"
    """Directory for saved plots (CWD-relative by default)."""

    prompts_file: str = field(default_factory=_default_holdout_path)
    """Path to a JSONL reward file.  Defaults to the shipped K=3 test
    holdout (``pareto_bandit/data/examples/test_holdout.jsonl``)."""

    encoder_model: str | None = None
    """SentenceTransformer model name.  ``None`` uses the library default
    (``all-MiniLM-L6-v2``).  A non-default model requires ``--pca-path``
    or falls back to raw (uncompressed) embeddings."""

    pca_path: str | None = None
    """Path to a PCA ``.joblib`` artifact for a non-default encoder.
    ``None`` uses the shipped ``pca_25.joblib`` (default encoder only)."""

    scenario: int | None = None
    """Run only this scenario (1–4).  ``None`` runs all four."""


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_evaluation_data(
    prompts_file: str,
    feature_service: FeatureService,
    seed: int = 42,
    n_prompts: int | None = None,
) -> tuple[DataSplit, DataSplit]:
    """Load a JSONL holdout file, embed prompts, and split train/test.

    Each JSONL record must contain a ``"prompt"`` string and an
    ``"arms"`` mapping ``{model_id: {"reward": float, "cost": float}}``.
    Per-arm rewards and costs are taken directly from the file.

    Parameters
    ----------
    prompts_file : str
        Path to a JSONL file with ``prompt`` and ``arms`` fields.
    feature_service : FeatureService
        Configured encoder for embedding prompts.
    seed : int
        RNG seed for subsampling and the train/test split.
    n_prompts : int | None
        Maximum prompts to use.  ``None`` uses all records.

    Returns
    -------
    Tuple[DataSplit, DataSplit]
        ``(train, test)`` with a 2:1 ratio.

    Raises
    ------
    FileNotFoundError
        If *prompts_file* does not exist.
    ValueError
        If the file is empty, has fewer than 50 prompts, or is
        missing the expected arm IDs.
    """
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    rng = np.random.default_rng(seed)

    records: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    if not records:
        raise ValueError(f"No records found in {path}")

    first_arms = set(records[0].get("arms", {}).keys())  # type: ignore[union-attr]
    missing = set(ARM_ORDER) - first_arms
    if missing:
        raise ValueError(
            f"JSONL records must contain arms {set(ARM_ORDER)} "
            f"but first record is missing: {missing}"
        )

    if n_prompts is not None and len(records) > n_prompts:
        idx = rng.choice(len(records), size=n_prompts, replace=False)
        records = [records[i] for i in sorted(idx)]

    min_prompts = 50
    if len(records) < min_prompts:
        raise ValueError(
            f"Need at least {min_prompts} prompts for meaningful "
            f"experiments, got {len(records)} in {path}"
        )

    raw_prompts = [str(r["prompt"]) for r in records]

    logger.info("Embedding %d prompts from %s ...", len(raw_prompts), path.name)
    X_bias = feature_service.extract_features_batch(raw_prompts)
    dim = X_bias.shape[1] - 1
    logger.info("Embedded %d prompts -> %d features (+bias)", len(raw_prompts), dim)

    n_total = len(records)
    rewards: dict[str, np.ndarray] = {a: np.empty(n_total) for a in ARM_ORDER}
    costs: dict[str, np.ndarray] = {a: np.empty(n_total) for a in ARM_ORDER}
    for i, rec in enumerate(records):
        arms = rec["arms"]  # type: ignore[index]
        for arm_id in ARM_ORDER:
            arm_data = arms[arm_id]  # type: ignore[index]
            rewards[arm_id][i] = float(arm_data["reward"])  # type: ignore[index]
            costs[arm_id][i] = float(arm_data["cost"])  # type: ignore[index]

    # Train / test split (2:1)
    n_train = int(n_total * 2 / 3)
    perm = rng.permutation(n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def _make_split(indices: np.ndarray) -> DataSplit:
        return DataSplit(
            embeddings=X_bias[indices],
            rewards={a: rewards[a][indices] for a in ARM_ORDER},
            costs={a: costs[a][indices] for a in ARM_ORDER},
        )

    train = _make_split(train_idx)
    test = _make_split(test_idx)

    logger.info(
        "Loaded %d train, %d test samples (%d features + bias)",
        train.n, test.n, dim,
    )
    for arm_id in ARM_ORDER:
        logger.info(
            "  %-28s  reward=%.3f+/-%.3f  cost=$%.6f",
            arm_id,
            float(np.mean(rewards[arm_id])),
            float(np.std(rewards[arm_id])),
            float(np.mean(costs[arm_id])),
        )
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# Router Factory
# ═══════════════════════════════════════════════════════════════════════════


def _create_router(
    feature_dim: int,
    *,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.3,
    budget_pacer: BudgetPacer | None = None,
    seed: int | None = None,
    warmup: bool = True,
) -> BanditRouter:
    """Build a K=3 router with warmup priors and optional budget pacer.

    Parameters
    ----------
    feature_dim : int
        Feature vector length (including bias column).
    alpha : float
        LinUCB exploration coefficient.
    forgetting_factor : float
        Geometric discount on sufficient statistics.
    cost_penalty : float
        Static cost-penalty weight in the UCB score.
    budget_pacer : BudgetPacer | None
        Adaptive budget pacer (``None`` = unconstrained routing).
    seed : int | None
        Bandit random seed for tie-breaking.
    warmup : bool
        If ``True``, load pre-trained (A, b) prior matrices from the
        shipped warmup artifact.

    Returns
    -------
    BanditRouter
        Configured router ready for ``route()`` / ``process_feedback()``.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=dict(MODEL_REGISTRY),
        feature_service=fs,
        context_store=store,
        priors="warmup" if warmup else "none",
        prior_n_effective=BEST_K3_HPARAMS["prior_n_effective"],
        alpha=alpha,
        forgetting_factor=forgetting_factor,
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
        bandit_seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Trial Execution
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrialMetrics:
    """Aggregate metrics from a single bandit trial.

    Parameters
    ----------
    mean_reward : float
        Average reward across evaluated prompts.
    mean_cost : float
        Average per-request cost (USD).
    model_fractions : Dict[str, float]
        Selection frequency per arm on the test split.
    per_step_models : List[str]
        Arm chosen at each evaluation step (empty unless
        ``record_steps=True``).
    per_step_rewards : List[float]
        Observed reward at each step.
    per_step_costs : List[float]
        Observed cost at each step.
    """

    mean_reward: float
    mean_cost: float
    model_fractions: dict[str, float]
    per_step_models: list[str] = field(default_factory=list)
    per_step_rewards: list[float] = field(default_factory=list)
    per_step_costs: list[float] = field(default_factory=list)


def run_trial(
    train: DataSplit,
    test: DataSplit,
    *,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.3,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
    record_steps: bool = False,
) -> TrialMetrics:
    """Run one online-learning then evaluation trial.

    The router learns on *train* (shuffled), then is evaluated on *test*
    (shuffled) while continuing to learn (standard bandit protocol).

    Parameters
    ----------
    train : DataSplit
        Online-learning data.
    test : DataSplit
        Held-out evaluation data.
    alpha : float
        Exploration coefficient.
    forgetting_factor : float
        Geometric discount (1.0 = stationary).
    cost_penalty : float
        Static cost-penalty weight.
    budget_pacer : BudgetPacer | None
        Optional adaptive budget pacer.
    seed : int
        Random seed.
    record_steps : bool
        If ``True``, populate the per-step lists in the result.

    Returns
    -------
    TrialMetrics
        Aggregate (and optionally per-step) evaluation metrics.
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

    # Online learning (train split)
    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Evaluation (test split)
    test_order = rng.permutation(test.n)
    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []
    model_counts: dict[str, int] = dict.fromkeys(ARM_ORDER, 0)
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
# Shared Helpers for Phased Scenarios (2 & 3)
# ═══════════════════════════════════════════════════════════════════════════

# Budget-level labels and per-level visual styling.
_BUDGET_LABELS: list[str] = ["tight", "moderate", "loose"]
_BUDGET_COLORS: dict[str, str] = {
    "tight": CB_RED,
    "moderate": CB_BLUE,
    "loose": CB_PURPLE,
}


def _rolling_mean(arr: list[float], window: int) -> np.ndarray:
    """Compute a simple moving average with a ``valid``-mode convolution."""
    return np.convolve(np.asarray(arr, dtype=float),
                       np.ones(window) / window, mode="valid")


def _rolling_fraction(
    models: list[str], arm: str, window: int,
) -> np.ndarray:
    """Rolling selection fraction for *arm* over a sliding window."""
    indicator = np.array([1.0 if m == arm else 0.0 for m in models])
    return np.convolve(indicator, np.ones(window) / window, mode="valid")


@dataclass
class _AveragedCurves:
    """Seed-averaged rolling curves for one condition in a phased trial."""

    frac_gemini: np.ndarray
    mean_reward: np.ndarray
    mean_cost: np.ndarray
    phase_size: int


def _phase_geometry(
    train: DataSplit, test: DataSplit,
) -> tuple[int, int, int]:
    """Compute ``(phase_size, total_steps, window)`` for phased trials.

    These depend only on data dimensions (not on seed or
    hyperparameters), so they can be computed once before any trials.
    """
    learn_n = train.n // 2
    n_eval = (train.n - learn_n) + test.n
    phase_size = n_eval // 3
    total_steps = 3 * phase_size
    window = max(20, total_steps // 30)
    return phase_size, total_steps, window


def _compute_degradation_budgets(train: DataSplit) -> dict[str, float]:
    """Three budget targets spanning the cost range for degradation trials.

    Returns tight / moderate / loose targets analogous to Experiment 03
    in the paper.
    """
    mean_costs = [float(np.mean(train.costs[m])) for m in ARM_ORDER]
    lo, hi = min(mean_costs), max(mean_costs)
    tight, moderate, loose = np.geomspace(lo * 10, hi * 0.15, num=3)
    return {"tight": tight, "moderate": moderate, "loose": loose}


PhasedTrialFn = Callable[..., tuple[list[str], list[float], list[float], int]]
"""Signature shared by ``_run_phased_trial`` and ``_run_cost_drift_trial``."""


def _run_multi_seed_phased(
    trial_fn: PhasedTrialFn,
    trial_kwargs: dict[str, object],
    n_seeds: int,
    base_seed: int,
    window: int,
    target_arm: str,
) -> _AveragedCurves:
    """Run a phased trial over multiple seeds and average rolling curves.

    Parameters
    ----------
    trial_fn : PhasedTrialFn
        Either ``_run_phased_trial`` or ``_run_cost_drift_trial``.
    trial_kwargs : dict
        Keyword arguments forwarded to *trial_fn* (excluding ``seed``).
    n_seeds : int
        Number of independent seeds.
    base_seed : int
        Starting seed (incremented by 1 per trial).
    window : int
        Rolling-window width for smoothing.
    target_arm : str
        Arm whose selection fraction is tracked (e.g. Gemini-Pro).

    Returns
    -------
    _AveragedCurves
        Seed-averaged rolling curves.
    """
    all_frac: list[np.ndarray] = []
    all_rwd: list[np.ndarray] = []
    all_cost: list[np.ndarray] = []
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


def _dollar_fmt(x: float, _pos: object = None) -> str:
    """Format a dollar amount for axis tick labels."""
    if x >= 0.01:
        return f"${x:.3f}"
    if x >= 0.001:
        return f"${x:.4f}"
    return f"${x:.5f}"


def _plot_phased_3panel(
    conditions: dict[str, _AveragedCurves],
    budget_targets: dict[str, float],
    budget_nice: dict[str, str],
    phase_boundaries: list[int],
    window: int,
    x_axis: np.ndarray,
    *,
    phase_labels: list[str],
    shade_color: str,
    suptitle: str,
    out_path: Path,
) -> Path:
    """Draw the Exp 02/03-style 3-panel stacked figure.

    Conditions keyed ``"ParetoBandit (<label>)"`` are drawn as solid
    lines colour-coded by budget label.  ``"Naive Bandit …"`` is gray
    dashed and ``"Unconstrained"`` is green dash-dot.

    Parameters
    ----------
    conditions : dict
        ``{label: _AveragedCurves}`` mapping.
    budget_targets : dict
        ``{budget_label: target_spend}`` for the cost-panel target lines.
    budget_nice : dict
        ``{budget_label: display_string}`` for legend entries.
    phase_boundaries : list of int
        ``[p1_end, p2_end, p3_end]`` step indices.
    window : int
        Rolling-window width (shown in panel title).
    x_axis : np.ndarray
        Shared x-axis array.
    phase_labels : list of str
        Three-element list of phase names.
    shade_color : str
        Fill colour for the Phase 2 band.
    suptitle : str
        Figure super-title.
    out_path : Path
        Where to save the PNG.

    Returns
    -------
    Path
        *out_path* after saving.
    """
    naive_color = CB_GRAY
    unconstrained_color = CB_GREEN

    def _add_shading(ax: plt.Axes) -> None:  # type: ignore[name-defined]
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
        for mid, lab in zip(mids, phase_labels, strict=False):
            ax.text(mid, 0.97, lab, transform=trans, ha="center",
                    va="top", fontsize=10, fontweight="bold", color="#333333")

    fig, (ax_gem, ax_rwd, ax_cost) = plt.subplots(
        3, 1, figsize=(10, 12), sharex=True,
    )

    def _plot_line(
        ax: plt.Axes,  # type: ignore[name-defined]
        series: np.ndarray,
        label: str,
        color: str,
        ls: str = "-",
        lw: float = 2.2,
        zo: int = 4,
    ) -> None:
        ax.plot(x_axis, series, color=color, linewidth=lw,
                linestyle=ls, label=label, zorder=zo)

    for cond_label, curves in conditions.items():
        if cond_label.startswith("ParetoBandit"):
            # Extract budget label from "ParetoBandit (tight)" etc.
            bl = cond_label.split("(")[1].rstrip(")")
            color = _BUDGET_COLORS[bl]
            nice = budget_nice[bl]
            _plot_line(ax_gem, curves.frac_gemini, nice, color)
            _plot_line(ax_rwd, curves.mean_reward, nice, color)
            _plot_line(ax_cost, curves.mean_cost, nice, color)
        elif cond_label.startswith("Naive"):
            _plot_line(ax_gem, curves.frac_gemini, cond_label,
                       naive_color, ls="--", lw=1.8, zo=3)
            _plot_line(ax_rwd, curves.mean_reward, cond_label,
                       naive_color, ls="--", lw=1.8, zo=3)
            _plot_line(ax_cost, curves.mean_cost, cond_label,
                       naive_color, ls="--", lw=1.8, zo=3)
        elif cond_label == "Unconstrained":
            uc_label = r"Unconstrained ($\lambda_s{=}0$)"
            _plot_line(ax_gem, curves.frac_gemini, uc_label,
                       unconstrained_color, ls="-.", lw=2.0, zo=3)
            _plot_line(ax_rwd, curves.mean_reward, uc_label,
                       unconstrained_color, ls="-.", lw=2.0, zo=3)
            _plot_line(ax_cost, curves.mean_cost, uc_label,
                       unconstrained_color, ls="-.", lw=2.0, zo=3)

    # Panel (a) — Gemini-Pro selection fraction
    _add_shading(ax_gem)
    ax_gem.set_ylabel("Fraction", fontsize=12)
    ax_gem.set_ylim(-0.02, 1.02)
    ax_gem.grid(True, alpha=0.2, linewidth=0.5)
    ax_gem.set_title("(a)  Gemini-Pro Selection Fraction",
                     fontsize=13, fontweight="bold", pad=8)
    ax_gem.tick_params(labelbottom=False)

    # Panel (b) — windowed mean reward
    _add_shading(ax_rwd)
    ax_rwd.set_ylabel("Mean Reward", fontsize=12)
    ax_rwd.grid(True, alpha=0.2, linewidth=0.5)
    ax_rwd.set_title(f"(b)  Windowed Mean Reward (window={window})",
                     fontsize=13, fontweight="bold", pad=8)
    ax_rwd.tick_params(labelbottom=False)

    # Panel (c) — windowed average cost per request + budget targets
    target_label_data: list[tuple[float, str, str]] = []
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
    for (_, blabel, color), y_pos in zip(sorted_tl, adj_y, strict=False):
        ax_cost.text(
            1.01, y_pos, f"{blabel} target",
            transform=blended_transform_factory(
                ax_cost.transAxes, ax_cost.transData,
            ),
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


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — Budget-Paced Routing
# ═══════════════════════════════════════════════════════════════════════════


def _compute_budget_targets(
    train: DataSplit, n_targets: int = 5,
) -> list[float]:
    """Log-spaced budget targets spanning arm cost extremes."""
    per_arm_means = [float(np.mean(train.costs[m])) for m in ARM_ORDER]
    lo, hi = min(per_arm_means), max(per_arm_means)
    return list(np.geomspace(lo, hi, num=n_targets))


def run_scenario_1(
    cfg: DemoConfig,
    train: DataSplit,
    test: DataSplit,
) -> Path:
    """Budget-paced routing sweep with 3-panel Pareto frontier plot.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 1: Budget-Paced LLM Routing")
    print("=" * 65)

    targets = _compute_budget_targets(train, cfg.n_budget_targets)
    target_strs = [f"${t:.2e}" if t < 1e-4 else f"${t:.5f}" for t in targets]
    print(f"  Budget targets ($/req): {target_strs}")

    # Fixed single-model baselines
    baselines: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        r = float(np.mean(test.rewards[arm]))
        c = float(np.mean(test.costs[arm]))
        baselines.append({"model_id": arm, "mean_reward": r, "mean_cost": c})
        print(f"  Baseline {ARM_SHORT[arm]:<16s}  reward={r:.4f}  cost=${c:.6f}")

    # Budget sweep (multi-seed)
    sweep_results: list[dict[str, object]] = []
    for target in targets:
        seed_rewards: list[float] = []
        seed_costs: list[float] = []
        seed_fracs: list[dict[str, float]] = []

        for s in range(cfg.n_seeds):
            pacer = BudgetPacer(
                target_avg_spend_usd=target,
                mode=PacingMode.ADAPTIVE,
            )
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
        se_r = (
            float(np.std(seed_rewards, ddof=1) / np.sqrt(cfg.n_seeds))
            if cfg.n_seeds > 1 else 0.0
        )
        se_c = (
            float(np.std(seed_costs, ddof=1) / np.sqrt(cfg.n_seeds))
            if cfg.n_seeds > 1 else 0.0
        )
        row: dict[str, object] = {
            "target_spend": target,
            "mean_reward": float(np.mean(seed_rewards)),
            "se_reward": se_r,
            "mean_cost": float(np.mean(seed_costs)),
            "se_cost": se_c,
            "model_fractions": avg_fracs,
        }
        sweep_results.append(row)
        util = float(row["mean_cost"]) / target if target > 0 else 0.0  # type: ignore[arg-type]
        print(
            f"  target=${target:.2e}  reward={row['mean_reward']:.4f}"
            f"+/-{row['se_reward']:.4f}  cost=${row['mean_cost']:.2e}"
            f"  util={util:.2f}x"
        )

    # --- 3-panel plot: Quality vs Cost, Budget Compliance, Model Mix ---
    targets_arr = np.array([r["target_spend"] for r in sweep_results])
    costs_arr = np.array([r["mean_cost"] for r in sweep_results])
    rewards_arr = np.array([r["mean_reward"] for r in sweep_results])
    se_r_arr = np.array([r["se_reward"] for r in sweep_results])
    se_c_arr = np.array([r["se_cost"] for r in sweep_results])

    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.25, 1.0, 1.0],
        wspace=0.38, left=0.05, right=0.97, top=0.86, bottom=0.15,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Panel A: Quality vs Cost
    ax_a.plot(
        costs_arr, rewards_arr,
        color=CB_BLUE, linewidth=2.5, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=CB_BLUE, markeredgewidth=2.0,
        zorder=6,
    )
    ax_a.errorbar(
        costs_arr, rewards_arr, yerr=se_r_arr,
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    util_tolerance = 0.10
    for r in sweep_results:
        ts = float(r["target_spend"])  # type: ignore[arg-type]
        util = float(r["mean_cost"]) / ts if ts > 0 else 0.0  # type: ignore[arg-type]
        if 1.0 - util_tolerance <= util <= 1.0 + util_tolerance:
            mc = CB_GREEN
        elif util < 1.0 - util_tolerance:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_a.plot(
            r["mean_cost"], r["mean_reward"],
            "o", markersize=7, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=7,
        )

    for b in baselines:
        mid = str(b["model_id"])
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

    y_lo = min(float(b["mean_reward"]) for b in baselines) - 0.02
    y_hi = max(float(r["mean_reward"]) for r in sweep_results) + 0.02
    ax_a.set_ylim(y_lo, y_hi)
    ax_a.set_xlabel("Cost per Request (USD)", fontsize=12)
    ax_a.set_ylabel("Mean Quality (Reward)", fontsize=12)
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax_a.grid(True, alpha=0.15, linewidth=0.5)
    ax_a.set_title("(a)  Quality vs. Budget", fontsize=13,
                    fontweight="bold", pad=8)

    legend_a = [
        Line2D([0], [0], color=CB_BLUE, linewidth=2.5, marker="o",
               markerfacecolor="white", markeredgecolor=CB_BLUE,
               markersize=7, label="ParetoBandit"),
        Line2D([0], [0], color="none", marker="*", markerfacecolor=CB_GRAY,
               markeredgecolor="black", markersize=12,
               label="Fixed single-model"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_GREEN,
               markeredgecolor=CB_BLUE, markersize=7,
               label="On-budget (+/-10%)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_ORANGE,
               markeredgecolor=CB_BLUE, markersize=7,
               label="Under-budget"),
    ]
    ax_a.legend(handles=legend_a, fontsize=8, loc="lower right",
                framealpha=0.9)

    # Panel B: Budget Compliance
    diag_range = np.geomspace(
        targets_arr.min() * 0.5, targets_arr.max() * 2.0, 100,
    )
    ax_b.plot(
        diag_range, diag_range,
        color=CB_GRAY, linestyle="--", linewidth=1.0, alpha=0.6,
        label="Perfect compliance", zorder=3,
    )
    ax_b.fill_between(
        diag_range,
        diag_range * (1.0 - util_tolerance),
        diag_range * (1.0 + util_tolerance),
        color=CB_GREEN, alpha=0.25, label="+/-10% band", zorder=2,
    )

    for r in sweep_results:
        ts = float(r["target_spend"])  # type: ignore[arg-type]
        util = float(r["mean_cost"]) / ts if ts > 0 else 0.0  # type: ignore[arg-type]
        if 1.0 - util_tolerance <= util <= 1.0 + util_tolerance:
            mc = CB_GREEN
        elif util < 1.0 - util_tolerance:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_b.plot(
            r["target_spend"], r["mean_cost"],
            "o", markersize=9, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=6,
        )
        ax_b.annotate(
            f"{util:.2f}x",
            xy=(float(r["target_spend"]), float(r["mean_cost"])),  # type: ignore[arg-type]
            xytext=(7, 0), textcoords="offset points",
            fontsize=9, color="0.3", ha="left", va="center",
        )

    ax_b.errorbar(
        targets_arr, costs_arr, yerr=se_c_arr,
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
    ax_b.set_title("(b)  Budget Compliance", fontsize=13,
                    fontweight="bold", pad=8)
    ax_b.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    # Panel C: Model Allocation
    x_pos = np.arange(len(sweep_results))
    bottom = np.zeros(len(sweep_results))
    bar_width = 0.65

    for arm in ARM_ORDER:
        fracs = np.array(
            [r["model_fractions"][arm] for r in sweep_results],  # type: ignore[index]
        )
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
    ax_c.set_title("(c)  Model Allocation", fontsize=13,
                    fontweight="bold", pad=8)
    ax_c.legend(fontsize=9, loc="center left", framealpha=0.9,
                bbox_to_anchor=(0.0, 0.5))

    fig.suptitle(
        "Budget-Paced LLM Routing (K=3)",
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

_DEGRADED_REWARD = 0.50
_PHASE_LABELS_S2 = ["Normal", "Mistral Failure", "Recovered"]


def _run_phased_trial(
    train: DataSplit,
    test: DataSplit,
    *,
    degraded_arm: str = "mistralai/mistral-large-2512",
    degraded_reward: float = _DEGRADED_REWARD,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.3,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
) -> tuple[list[str], list[float], list[float], int]:
    """Three-phase trial: normal -> degradation -> recovery.

    Uses the first half of *train* for online learning, then
    concatenates the second half with *test* to form a longer
    evaluation sequence (3 equal phases).

    During Phase 2, the *degraded_arm*'s rewards are replaced with a
    low constant.  Phase 3 restores normal rewards.

    Returns
    -------
    tuple
        ``(step_models, step_rewards, step_costs, phase_size)``.
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

    train_order = rng.permutation(train.n)
    learn_n = train.n // 2
    learn_idx = train_order[:learn_n]
    extra_eval_idx = train_order[learn_n:]

    for i in learn_idx:
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Combined evaluation pool (second half of train + full test)
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

    phase_size = n_eval // 3
    phases = [
        eval_order[:phase_size],
        eval_order[phase_size:2 * phase_size],
        eval_order[2 * phase_size:3 * phase_size],
    ]

    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []

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


def run_scenario_2(
    cfg: DemoConfig,
    train: DataSplit,
    test: DataSplit,
) -> Path:
    """Quality degradation and recovery (Exp 03-style 3-panel figure).

    Runs five conditions (ParetoBandit x3 budgets, Naive Bandit,
    Unconstrained) through a three-phase simulation, averaging over
    ``cfg.n_seeds`` independent seeds for smooth curves.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 2: Quality Degradation & Recovery")
    print("=" * 65)

    degraded_arm = "mistralai/mistral-large-2512"
    degraded_reward = _DEGRADED_REWARD
    phase_size, total_steps, window = _phase_geometry(train, test)

    budget_targets = _compute_degradation_budgets(train)
    budget_nice: dict[str, str] = {}
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        budget_nice[bl] = rf"{bl.title()} ($B{{=}}\${bt:.1e}$)"
        print(f"  Budget {bl:<10s} = ${bt:.2e}/req")

    conditions: dict[str, _AveragedCurves] = {}

    for bl in _BUDGET_LABELS:
        conditions[f"ParetoBandit ({bl})"] = _run_multi_seed_phased(
            trial_fn=_run_phased_trial,
            trial_kwargs={
                "train": train, "test": test,
                "degraded_arm": degraded_arm,
                "degraded_reward": degraded_reward,
                "alpha": cfg.alpha,
                "forgetting_factor": cfg.forgetting_factor,
                "cost_penalty": 0.0,
                "budget_pacer": BudgetPacer(
                    target_avg_spend_usd=budget_targets[bl],
                    mode=PacingMode.ADAPTIVE,
                ),
            },
            n_seeds=cfg.n_seeds, base_seed=cfg.seed,
            window=window, target_arm=GEMINI_ARM,
        )

    conditions["Naive Bandit (moderate)"] = _run_multi_seed_phased(
        trial_fn=_run_phased_trial,
        trial_kwargs={
            "train": train, "test": test,
            "degraded_arm": degraded_arm,
            "degraded_reward": degraded_reward,
            "alpha": cfg.alpha,
            "forgetting_factor": 1.0,
            "cost_penalty": 0.0,
            "budget_pacer": BudgetPacer(
                target_avg_spend_usd=budget_targets["moderate"],
                mode=PacingMode.ADAPTIVE,
            ),
        },
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_phased_trial,
        trial_kwargs={
            "train": train, "test": test,
            "degraded_arm": degraded_arm,
            "degraded_reward": degraded_reward,
            "alpha": cfg.alpha,
            "forgetting_factor": cfg.forgetting_factor,
            "cost_penalty": 0.0,
        },
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
        print(f"  {cond_name:<30s}  P1={np.mean(p1):.4f}  "
              f"P2={np.mean(p2):.4f}  P3={np.mean(p3):.4f}")

    out_path = Path(cfg.output_dir) / "scenario2_quality_degradation.png"
    _plot_phased_3panel(
        conditions, budget_targets, budget_nice,
        phase_boundaries, window, x_axis,
        phase_labels=_PHASE_LABELS_S2, shade_color=CB_RED,
        suptitle=(f"Quality Degradation & Recovery — {ARM_SHORT[degraded_arm]} "
                  f"reward drops to {degraded_reward:.2f} in Phase 2"),
        out_path=out_path,
    )
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — Cost Drift & Recovery (Exp 02 analogue)
# ═══════════════════════════════════════════════════════════════════════════

# In Phase 2, Gemini-Pro's pricing drops by this factor (0.02 = 50x
# cheaper).  The BudgetPacer exploits cheap premium routing, then
# returns to the original mix when prices are restored in Phase 3.
_GEMINI_COST_SCALE_PHASE2 = 0.02

_PHASE_LABELS_S3 = ["Normal", "Price Drop", "Restored"]


def _run_cost_drift_trial(
    train: DataSplit,
    test: DataSplit,
    *,
    cost_drift_arm: str = "google/gemini-2.5-pro",
    cost_scale: float = _GEMINI_COST_SCALE_PHASE2,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.0,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
) -> tuple[list[str], list[float], list[float], int]:
    """Three-phase cost-drift trial: normal -> price drop -> restored.

    Phase 2 scales the *cost_drift_arm*'s per-request costs by
    *cost_scale* (e.g. 0.02 = 50x cheaper) and updates the router's
    pricing registry.  Phase 3 restores original pricing.

    Returns
    -------
    tuple
        ``(step_models, step_rewards, step_costs, phase_size)``.
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

    train_order = rng.permutation(train.n)
    learn_n = train.n // 2
    learn_idx = train_order[:learn_n]
    extra_eval_idx = train_order[learn_n:]

    for i in learn_idx:
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Combined evaluation pool
    eval_emb = np.vstack([train.embeddings[extra_eval_idx], test.embeddings])
    eval_rewards = {
        a: np.concatenate([train.rewards[a][extra_eval_idx], test.rewards[a]])
        for a in ARM_ORDER
    }
    eval_costs_normal = {
        a: np.concatenate([train.costs[a][extra_eval_idx], test.costs[a]])
        for a in ARM_ORDER
    }
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

    orig_input = MODEL_REGISTRY[cost_drift_arm]["input_cost_per_m"]
    orig_output = MODEL_REGISTRY[cost_drift_arm]["output_cost_per_m"]

    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []

    for phase_idx, phase_indices in enumerate(phases):
        if phase_idx == 1:
            router.update_model_pricing(
                cost_drift_arm,
                input_cost_per_m=float(orig_input) * cost_scale,  # type: ignore[arg-type]
                output_cost_per_m=float(orig_output) * cost_scale,  # type: ignore[arg-type]
            )
        elif phase_idx == 2:
            router.update_model_pricing(
                cost_drift_arm,
                input_cost_per_m=float(orig_input),  # type: ignore[arg-type]
                output_cost_per_m=float(orig_output),  # type: ignore[arg-type]
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
    train: DataSplit,
    test: DataSplit,
) -> Path:
    """Cost drift and recovery (Exp 02-style 3-panel figure).

    Simulates a Gemini-Pro price drop in Phase 2, averaging over
    ``cfg.n_seeds`` independent seeds for smooth curves.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 3: Cost Drift & Recovery (Gemini-Pro Price Drop)")
    print("=" * 65)

    cost_drift_arm = "google/gemini-2.5-pro"
    cost_scale = _GEMINI_COST_SCALE_PHASE2
    print(f"  Gemini-Pro cost multiplier in Phase 2: {cost_scale}x "
          f"(~{1 / cost_scale:.0f}x cheaper)")

    phase_size, total_steps, window = _phase_geometry(train, test)

    budget_targets = _compute_degradation_budgets(train)
    budget_nice: dict[str, str] = {}
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        budget_nice[bl] = rf"{bl.title()} ($B{{=}}\${bt:.1e}$)"
        print(f"  Budget {bl:<10s} = ${bt:.2e}/req")

    conditions: dict[str, _AveragedCurves] = {}

    for bl in _BUDGET_LABELS:
        conditions[f"ParetoBandit ({bl})"] = _run_multi_seed_phased(
            trial_fn=_run_cost_drift_trial,
            trial_kwargs={
                "train": train, "test": test,
                "cost_drift_arm": cost_drift_arm,
                "cost_scale": cost_scale,
                "alpha": cfg.alpha,
                "forgetting_factor": cfg.forgetting_factor,
                "cost_penalty": 0.0,
                "budget_pacer": BudgetPacer(
                    target_avg_spend_usd=budget_targets[bl],
                    mode=PacingMode.ADAPTIVE,
                ),
            },
            n_seeds=cfg.n_seeds, base_seed=cfg.seed,
            window=window, target_arm=GEMINI_ARM,
        )

    conditions["Naive Bandit (moderate)"] = _run_multi_seed_phased(
        trial_fn=_run_cost_drift_trial,
        trial_kwargs={
            "train": train, "test": test,
            "cost_drift_arm": cost_drift_arm,
            "cost_scale": cost_scale,
            "alpha": cfg.alpha,
            "forgetting_factor": 1.0,
            "cost_penalty": 0.0,
            "budget_pacer": BudgetPacer(
                target_avg_spend_usd=budget_targets["moderate"],
                mode=PacingMode.ADAPTIVE,
            ),
        },
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_cost_drift_trial,
        trial_kwargs={
            "train": train, "test": test,
            "cost_drift_arm": cost_drift_arm,
            "cost_scale": cost_scale,
            "alpha": cfg.alpha,
            "forgetting_factor": cfg.forgetting_factor,
            "cost_penalty": 0.0,
        },
        n_seeds=cfg.n_seeds, base_seed=cfg.seed,
        window=window, target_arm=GEMINI_ARM,
    )

    phase_boundaries = [phase_size, 2 * phase_size, 3 * phase_size]
    x_axis = np.arange(total_steps - window + 1) + window // 2

    print()
    for cond_name, curves in conditions.items():
        ps = curves.phase_size
        r = curves.mean_reward
        c = curves.mean_cost
        p1_r = r[: ps - window + 1]
        p2_r = r[ps - window + 1: 2 * ps - window + 1]
        p3_r = r[2 * ps - window + 1:]
        p1_c = c[: ps - window + 1]
        p2_c = c[ps - window + 1: 2 * ps - window + 1]
        p3_c = c[2 * ps - window + 1:]
        print(f"  {cond_name:<30s}  "
              f"P1: R={np.mean(p1_r):.4f} C=${np.mean(p1_c):.2e}  "
              f"P2: R={np.mean(p2_r):.4f} C=${np.mean(p2_c):.2e}  "
              f"P3: R={np.mean(p3_r):.4f} C=${np.mean(p3_c):.2e}")

    out_path = Path(cfg.output_dir) / "scenario3_cost_drift.png"
    _plot_phased_3panel(
        conditions, budget_targets, budget_nice,
        phase_boundaries, window, x_axis,
        phase_labels=_PHASE_LABELS_S3, shade_color=CB_GREEN,
        suptitle=("Cost Drift & Recovery — Gemini-Pro pricing drops "
                  f"{1 / cost_scale:.0f}x in Phase 2"),
        out_path=out_path,
    )
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4 — Configuration Comparison
# ═══════════════════════════════════════════════════════════════════════════


def run_scenario_4(
    cfg: DemoConfig,
    train: DataSplit,
    test: DataSplit,
) -> Path:
    """Compare key configuration knobs on the quality-cost frontier.

    Sweeps ``alpha``, ``forgetting_factor``, and ``cost_penalty`` one at
    a time while holding the others at their defaults.  Each parameter
    is tested at three levels.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 4: Configuration Comparison")
    print("=" * 65)

    param_sweeps = {
        "alpha (exploration)": {
            "param": "alpha",
            "values": [0.001, 0.01, 0.1],
            "labels": [
                "a=0.001\n(exploit)",
                "a=0.01\n(default)",
                "a=0.1\n(explore)",
            ],
            "defaults": {
                "forgetting_factor": cfg.forgetting_factor,
                "cost_penalty": cfg.cost_penalty,
            },
        },
        "forgetting_factor (adaptation)": {
            "param": "forgetting_factor",
            "values": [1.0, 0.997, 0.99],
            "labels": [
                "g=1.0\n(stationary)",
                "g=0.997\n(default)",
                "g=0.99\n(aggressive)",
            ],
            "defaults": {
                "alpha": cfg.alpha,
                "cost_penalty": cfg.cost_penalty,
            },
        },
        "cost_penalty (cost aversion)": {
            "param": "cost_penalty",
            "values": [0.0, 0.3, 1.0],
            "labels": [
                "lc=0.0\n(quality only)",
                "lc=0.3\n(default)",
                "lc=1.0\n(cost focus)",
            ],
            "defaults": {
                "alpha": cfg.alpha,
                "forgetting_factor": cfg.forgetting_factor,
            },
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.subplots_adjust(
        wspace=0.35, left=0.06, right=0.96, top=0.86, bottom=0.18,
    )

    for ax, (sweep_name, sweep_cfg) in zip(axes, param_sweeps.items(), strict=False):
        param_key = sweep_cfg["param"]
        values = sweep_cfg["values"]
        labels = sweep_cfg["labels"]
        defaults = sweep_cfg["defaults"]

        print(f"\n  Sweep: {sweep_name}")
        bar_rewards: list[float] = []
        bar_costs: list[float] = []
        bar_fracs: list[dict[str, float]] = []

        for val in values:
            kwargs = dict(defaults)
            kwargs[param_key] = val

            seed_r: list[float] = []
            seed_c: list[float] = []
            seed_f: list[dict[str, float]] = []

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
            avg_f = {
                m: float(np.mean([f[m] for f in seed_f])) for m in ARM_ORDER
            }

            bar_rewards.append(avg_r)
            bar_costs.append(avg_c)
            bar_fracs.append(avg_f)
            print(f"    {param_key}={val:<8}  "
                  f"reward={avg_r:.4f}  cost=${avg_c:.6f}")

        # Stacked bar chart for model mix
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

        for i in range(len(values)):
            ax.text(
                x_pos[i], 1.01,
                f"R={bar_rewards[i]:.3f}\nC=${bar_costs[i]:.5f}",
                ha="center", va="bottom", fontsize=7.5, color="0.3",
            )

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, leg_labels, loc="upper center", ncol=3,
        fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.99),
    )

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
# CLI and Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> DemoConfig:
    """Parse CLI arguments into a :class:`DemoConfig`."""
    parser = argparse.ArgumentParser(
        description="ParetoBandit interactive demo — runs four "
                    "scenarios showcasing budget-paced LLM routing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-prompts", type=int, default=1000,
        help="Prompts to sample from the data file (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Independent seeds per condition (default: 5)",
    )
    parser.add_argument(
        "--alpha", type=float, default=BEST_K3_HPARAMS["alpha"],
        help=f"LinUCB exploration (default: {BEST_K3_HPARAMS['alpha']})",
    )
    parser.add_argument(
        "--forgetting-factor", type=float,
        default=BEST_K3_HPARAMS["forgetting_factor"],
        help="Geometric discount (default: "
             f"{BEST_K3_HPARAMS['forgetting_factor']})",
    )
    parser.add_argument(
        "--cost-penalty", type=float, default=0.3,
        help="Cost-penalty weight (default: 0.3)",
    )
    parser.add_argument(
        "--n-budget-targets", type=int, default=7,
        help="Budget sweep points for Scenario 1 (default: 7)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="demo_results",
        help="Output directory for plots (default: ./demo_results)",
    )
    parser.add_argument(
        "--prompts-file", type=str, default=_default_holdout_path(),
        help="JSONL reward file (default: shipped test holdout)",
    )
    parser.add_argument(
        "--encoder-model", type=str, default=None,
        help="SentenceTransformer model name "
             "(default: all-MiniLM-L6-v2). "
             "Requires --pca-path for a compatible PCA, "
             "or raw embeddings are used when omitted.",
    )
    parser.add_argument(
        "--pca-path", type=str, default=None,
        help="PCA .joblib artifact for a non-default encoder",
    )
    parser.add_argument(
        "--scenario", type=int, default=None, choices=[1, 2, 3, 4],
        help="Run only this scenario (default: all)",
    )
    args = parser.parse_args()
    return DemoConfig(
        n_prompts=args.n_prompts,
        seed=args.seed,
        n_seeds=args.n_seeds,
        alpha=args.alpha,
        forgetting_factor=args.forgetting_factor,
        cost_penalty=args.cost_penalty,
        n_budget_targets=args.n_budget_targets,
        output_dir=args.output_dir,
        prompts_file=args.prompts_file,
        encoder_model=args.encoder_model,
        pca_path=args.pca_path,
        scenario=args.scenario,
    )


def main() -> None:
    """Entry point: parse config, generate data, run scenarios, save plots."""
    if not _HAS_MATPLOTLIB:
        sys.exit(
            "ERROR: The ParetoBandit demo requires matplotlib.\n"
            "Install with:  pip install paretobandit[demo]"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for _noisy in (
        "pareto_bandit.router",
        "pareto_bandit.feature_service",
        "pareto_bandit.policy",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    cfg = parse_args()
    t0 = time.time()

    # --- Build FeatureService ---
    if cfg.encoder_model is not None:
        encoder_label = cfg.encoder_model
        if cfg.pca_path is not None:
            fs = FeatureService(
                encoder_model=cfg.encoder_model,
                pca_path=cfg.pca_path,
            )
        else:
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(cfg.encoder_model)
            _emb_dim = _st_model.get_sentence_embedding_dimension()
            fs = FeatureService(
                custom_encoder=lambda text, _m=_st_model: _m.encode(
                    text, normalize_embeddings=True, show_progress_bar=False,
                ),
                embedding_dim=_emb_dim,
            )
    else:
        encoder_label = "default (all-MiniLM-L6-v2 + PCA-25)"
        fs = FeatureService()

    print("=" * 65)
    print("  ParetoBandit Interactive Demo")
    print("=" * 65)
    print(f"  Data:        {cfg.prompts_file} (<={cfg.n_prompts})")
    print(f"  Encoder:     {encoder_label}")
    print(f"  Seeds/cond:  {cfg.n_seeds}")
    print(f"  Seed:        {cfg.seed}")
    print(f"  Output:      {cfg.output_dir}")
    print()

    train, test = load_evaluation_data(
        prompts_file=cfg.prompts_file,
        feature_service=fs,
        seed=cfg.seed,
        n_prompts=cfg.n_prompts,
    )

    saved: list[Path] = []
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
