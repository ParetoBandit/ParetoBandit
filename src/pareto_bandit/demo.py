"""ParetoBandit Interactive Demo.

Uses the paper's train-on-val / evaluate-on-holdout protocol (§4.1):

    * **Train split** — shipped ``val.jsonl`` (n=1,785 prompts): online
      learning only, no evaluation metrics recorded.
    * **Eval split** — shipped ``test_holdout.jsonl`` (n=1,824 prompts):
      partitioned into phases for evaluation.

Four scenarios showcase core capabilities:

    **Scenario 1 — Budget-Paced Routing** (paper §4.2, Figure 1)
        Sweeps budget targets and shows how ParetoBandit smoothly
        interpolates between cheap/low-quality and expensive/high-quality
        models while respecting an operator-set dollar budget (§3.2).

    **Scenario 2 — Quality Degradation & Recovery** (paper §4.4, Figure 3)
        3-phase (608 prompts/phase): simulates a silent quality regression
        on Mistral-Large (reward → 0.75), demonstrating that geometric
        forgetting (§3.3) detects the drop, redistributes traffic, and
        recovers when quality is restored.

    **Scenario 3 — Cost Drift & Recovery** (paper §4.3, Figure 2)
        3-phase: simulates a dramatic Gemini-Pro price drop, showing how
        the BudgetPacer (§3.2) exploits cheap premium routing during the
        drop and restores budget-compliant routing when prices are
        corrected.

    **Scenario 4 — Configuration Comparison** (demo-specific)
        Varies ``alpha`` (§3.2), ``forgetting_factor`` (§3.3), and
        ``cost_penalty`` λ_c to illustrate how each knob shapes the
        quality-cost trade-off.

All plots are saved to ``<output_dir>/`` (default ``./demo_results/``).

Requires ``pip install paretobandit[demo]``.  Pass
``--encoder-model`` to swap the embedding backbone (a matching PCA
artifact is then required via ``--pca-path``, or raw embeddings are
used when omitted).

Usage::

    # Via CLI entry point (after pip install paretobandit[demo])
    paretobandit-demo

    # Run a single scenario
    paretobandit-demo --scenario 2

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
    GEMINI_COST_DROP,
    K3_ARM_ORDER,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    K3_FAILURE_ARM,
    K3_FAILURE_REWARD,
)
from pareto_bandit.data import get_example_holdout_path, get_example_val_path
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

# Empirical mean per-request cost (USD) from the K=3 benchmark (Table 1).
#   Llama   ~$2.9e-5/req  (budget — ~400 tokens)
#   Mistral ~$5.3e-4/req  (mid-tier — variable token count)
#   Gemini  ~$1.5e-2/req  (premium — reasoning traces yield long outputs)
# 530× spread between cheapest and most expensive.
_MEAN_COST_PER_REQ: dict[str, float] = {
    "meta-llama/llama-3.1-8b-instruct": 2.9e-05,
    "mistralai/mistral-large-2512": 5.3e-04,
    "google/gemini-2.5-pro": 1.5e-02,
}

# ═══════════════════════════════════════════════════════════════════════════
# Data Split
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DataSplit:
    """One split (train or holdout) of the evaluation dataset.

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


@lru_cache(maxsize=1)
def _default_val_path() -> str:
    """Resolve the shipped validation JSONL path (cached, no repeated I/O)."""
    return str(get_example_val_path())


@dataclass
class DemoConfig:
    """Top-level configuration for the demo.

    Modify these values to explore different operating regimes.
    All parameters can also be overridden via CLI flags.
    """

    seed: int = 42
    """Master RNG seed for full reproducibility."""

    n_seeds: int = 10
    """Independent seeds per condition (paper uses 20; demo default trades
    smoothness for speed)."""

    alpha: float = BEST_K3_HPARAMS["alpha"]
    """LinUCB exploration coefficient (§3.2, Eq. 2; tuned via Appendix A)."""

    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"]
    """Geometric discount on sufficient statistics (§3.3, Eqs. 7–8).
    1.0 = stationary; 0.997 gives ~333-step effective memory."""

    cost_penalty: float = 0.3
    """Static cost-penalty weight λ_c in the UCB score (§3.2, Eq. 2).
    The BudgetPacer's adaptive λ_t provides closed-loop enforcement
    on top of this baseline preference."""

    n_budget_targets: int = 7
    """Number of log-spaced budget targets for Scenario 1."""

    output_dir: str = "demo_results"
    """Directory for saved plots (CWD-relative by default)."""

    val_file: str = field(default_factory=_default_val_path)
    """Path to the JSONL file used for **training** (online learning).
    Defaults to the shipped ``val.jsonl`` (1,785 prompts)."""

    holdout_file: str = field(default_factory=_default_holdout_path)
    """Path to the JSONL file used for **evaluation**.
    Defaults to the shipped ``test_holdout.jsonl`` (1,824 prompts)."""

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

PHASE_N: int = 608
"""Prompts per phase in three-phase scenarios (§4.1: 608 prompts/phase)."""


def _load_jsonl(
    path: Path,
    feature_service: FeatureService,
) -> DataSplit:
    """Load a single JSONL file and embed prompts.

    Each JSONL record must contain a ``"prompt"`` string and an
    ``"arms"`` mapping ``{model_id: {"reward": float, "cost": float}}``.

    Parameters
    ----------
    path : Path
        Path to a JSONL reward file.
    feature_service : FeatureService
        Configured encoder for embedding prompts.

    Returns
    -------
    DataSplit
        Embeddings, rewards, and costs for all records in the file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty or missing expected arm IDs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

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

    raw_prompts = [str(r["prompt"]) for r in records]
    logger.info("Embedding %d prompts from %s ...", len(raw_prompts), path.name)
    X_bias = feature_service.extract_features_batch(raw_prompts)
    logger.info(
        "Embedded %d prompts -> %d features (+bias)",
        len(raw_prompts), X_bias.shape[1] - 1,
    )

    n_total = len(records)
    rewards: dict[str, np.ndarray] = {a: np.empty(n_total) for a in ARM_ORDER}
    costs: dict[str, np.ndarray] = {a: np.empty(n_total) for a in ARM_ORDER}
    for i, rec in enumerate(records):
        arms = rec["arms"]  # type: ignore[index]
        for arm_id in ARM_ORDER:
            arm_data = arms[arm_id]  # type: ignore[index]
            rewards[arm_id][i] = float(arm_data["reward"])  # type: ignore[index]
            costs[arm_id][i] = float(arm_data["cost"])  # type: ignore[index]

    return DataSplit(embeddings=X_bias, rewards=rewards, costs=costs)


def load_demo_splits(
    val_file: str,
    holdout_file: str,
    feature_service: FeatureService,
) -> tuple[DataSplit, DataSplit]:
    """Load the val (train) and holdout (eval) splits separately.

    This follows the paper's protocol: the val split is used exclusively
    for online learning and the holdout split exclusively for evaluation.

    Parameters
    ----------
    val_file : str
        Path to the validation JSONL file (training data).
    holdout_file : str
        Path to the holdout JSONL file (evaluation data).
    feature_service : FeatureService
        Configured encoder for embedding prompts.

    Returns
    -------
    Tuple[DataSplit, DataSplit]
        ``(train, eval)`` — val split for training, holdout for evaluation.
    """
    train = _load_jsonl(Path(val_file), feature_service)
    holdout = _load_jsonl(Path(holdout_file), feature_service)

    logger.info(
        "Loaded %d train (val), %d eval (holdout) samples",
        train.n, holdout.n,
    )
    for arm_id in ARM_ORDER:
        all_rewards = np.concatenate([
            train.rewards[arm_id], holdout.rewards[arm_id],
        ])
        all_costs = np.concatenate([
            train.costs[arm_id], holdout.costs[arm_id],
        ])
        logger.info(
            "  %-28s  reward=%.3f+/-%.3f  cost=$%.6f",
            arm_id,
            float(np.mean(all_rewards)),
            float(np.std(all_rewards)),
            float(np.mean(all_costs)),
        )
    return train, holdout


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
    holdout: DataSplit,
    *,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.3,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
    record_steps: bool = False,
) -> TrialMetrics:
    """Run one online-learning then evaluation trial.

    The router learns on the full *train* split (val, shuffled), then is
    evaluated on *holdout* (shuffled) while continuing to learn — the
    standard bandit protocol under bandit feedback (§2.1).

    Parameters
    ----------
    train : DataSplit
        Online-learning data (val split).
    holdout : DataSplit
        Held-out evaluation data (test holdout split).
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

    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_order = rng.permutation(holdout.n)
    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []
    model_counts: dict[str, int] = dict.fromkeys(ARM_ORDER, 0)
    reward_sum = 0.0
    cost_sum = 0.0

    for i in eval_order:
        model, log = router.route(holdout.embeddings[i])
        reward = float(holdout.rewards[model][i])
        cost = float(holdout.costs[model][i])
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        model_counts[model] += 1
        reward_sum += reward
        cost_sum += cost

        if record_steps:
            step_models.append(model)
            step_rewards.append(reward)
            step_costs.append(cost)

    n_eval = len(eval_order)
    return TrialMetrics(
        mean_reward=reward_sum / n_eval,
        mean_cost=cost_sum / n_eval,
        model_fractions={m: cnt / n_eval for m, cnt in model_counts.items()},
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
    n_phases: int = 3,
    phase_n: int = PHASE_N,
) -> tuple[int, int, int]:
    """Compute ``(phase_size, total_steps, window)`` for phased trials.

    Parameters
    ----------
    n_phases : int
        Number of evaluation phases (3 for both quality-degradation
        and cost-drift scenarios).
    phase_n : int
        Prompts per phase (default: ``PHASE_N = 608``).

    Returns
    -------
    tuple of (phase_size, total_steps, window)
    """
    total_steps = n_phases * phase_n
    window = max(20, total_steps // 30)
    return phase_n, total_steps, window


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
    """Draw the Exp 02/03-style 3-panel stacked figure (Figures 2–3).

    Parameters
    ----------
    conditions : dict
        ``{label: _AveragedCurves}`` mapping.
    budget_targets : dict
        ``{budget_label: target_spend}`` for the cost-panel target lines.
    budget_nice : dict
        ``{budget_label: display_string}`` for legend entries.
    phase_boundaries : list of int
        Step indices at the end of each phase.  Length must equal
        ``len(phase_labels)``.
    window : int
        Rolling-window width (shown in panel title).
    x_axis : np.ndarray
        Shared x-axis array.
    phase_labels : list of str
        Phase names (3 elements for the three-phase layout).
    shade_color : str
        Fill colour for the perturbation phase band (Phase 2).
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
        mids = [
            phase_boundaries[0] / 2,
            (phase_boundaries[0] + phase_boundaries[1]) / 2,
            (phase_boundaries[1] + phase_boundaries[2]) / 2,
        ]
        trans = blended_transform_factory(ax.transData, ax.transAxes)
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
    ax_rwd.set_ylim(0.75, 0.95)
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
# Scenario 1 — Budget-Paced Routing (§4.2, Figure 1)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_budget_targets(
    train: DataSplit,
    holdout: DataSplit,
    n_targets: int = 5,
) -> list[float]:
    """Log-spaced budget targets spanning arm cost extremes.

    Arm means are pooled across both splits (val + holdout) for stable
    targets.  Budget targets are configuration inputs, so using all
    available data is appropriate.
    """
    per_arm_means: list[float] = []
    for m in ARM_ORDER:
        merged = np.concatenate([train.costs[m], holdout.costs[m]])
        per_arm_means.append(float(np.mean(merged)))
    lo, hi = min(per_arm_means), max(per_arm_means)
    return list(np.geomspace(lo, hi, num=n_targets))


def run_scenario_1(
    cfg: DemoConfig,
    train: DataSplit,
    holdout: DataSplit,
) -> Path:
    """Budget-paced routing sweep with 3-panel Pareto frontier plot (§4.2, Figure 1).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 1: Budget-Paced LLM Routing (§4.2)")
    print("=" * 65)

    targets = _compute_budget_targets(train, holdout, cfg.n_budget_targets)
    target_strs = [f"${t:.2e}" if t < 1e-4 else f"${t:.5f}" for t in targets]
    print(f"  Budget targets ($/req): {target_strs}")

    baselines: list[dict[str, object]] = []
    for arm in ARM_ORDER:
        r = float(np.mean(holdout.rewards[arm]))
        c = float(np.mean(holdout.costs[arm]))
        baselines.append({"model_id": arm, "mean_reward": r, "mean_cost": c})
        print(f"  Baseline {ARM_SHORT[arm]:<16s}  reward={r:.4f}  cost=${c:.6f}")

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
                train, holdout,
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
# Scenario 2 — Quality Degradation & Recovery (§4.4, Figure 3; Exp 03)
# ═══════════════════════════════════════════════════════════════════════════

_DEGRADED_REWARD: float = K3_FAILURE_REWARD
_DEGRADED_ARM: str = K3_FAILURE_ARM
_PHASE_LABELS_S2 = ["Normal", "Mistral Failure", "Recovery"]


def _run_degradation_trial(
    train: DataSplit,
    holdout: DataSplit,
    p1_idx: np.ndarray,
    p2_idx: np.ndarray,
    *,
    degraded_arm: str = _DEGRADED_ARM,
    degraded_reward: float = _DEGRADED_REWARD,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.0,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
) -> tuple[list[str], list[float], list[float], int]:
    """Three-phase quality degradation trial (§4.4, Figure 3; Exp 03).

    Phase 1 (Normal): ``p1_idx`` holdout prompts, all models healthy.
    Phase 2 (Failure): ``p2_idx`` holdout prompts, *degraded_arm*
    rewards replaced with *degraded_reward* (costs unchanged).
    Geometric forgetting (§3.3) detects the regression via the reward
    signal alone.
    Phase 3 (Recovery): Reuses ``p1_idx`` prompts with original rewards
    — controlled within-subject comparison with Phase 1 (§4.1).

    Training uses the full *train* split (no metrics recorded).

    Returns
    -------
    tuple
        ``(step_models, step_rewards, step_costs, phase_size)``.
    """
    feature_dim = train.embeddings.shape[1]
    rng = np.random.default_rng(seed)
    phase_n = len(p1_idx)

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

    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    p1_order = rng.permutation(phase_n)
    p2_order = rng.permutation(len(p2_idx))
    p3_order = rng.permutation(phase_n)

    phase_specs: list[tuple[np.ndarray, np.ndarray, bool]] = [
        (p1_idx, p1_order, False),
        (p2_idx, p2_order, True),
        (p1_idx, p3_order, False),
    ]

    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []

    for pool_idx, order, is_degraded in phase_specs:
        for o in order:
            i = pool_idx[o]
            model, log = router.route(holdout.embeddings[i])

            if is_degraded and model == degraded_arm:
                reward = degraded_reward
            else:
                reward = float(holdout.rewards[model][i])
            cost = float(holdout.costs[model][i])

            log.cost_usd = cost
            router.process_feedback(log.request_id, reward=reward)

            step_models.append(model)
            step_rewards.append(reward)
            step_costs.append(cost)

    return step_models, step_rewards, step_costs, phase_n


def run_scenario_2(
    cfg: DemoConfig,
    train: DataSplit,
    holdout: DataSplit,
) -> Path:
    """Quality degradation & recovery (§4.4, Figure 3; paper Exp 03).

    Three phases of ``PHASE_N`` (608) prompts each, matching the paper's
    non-stationary protocol (§4.1):

    * Phase 1 — Normal (608 holdout prompts)
    * Phase 2 — Mistral Failure: reward drops to ``K3_FAILURE_REWARD``
      (~18% below normal, §4.4)
    * Phase 3 — Recovery: reuses Phase 1 prompts (within-subject design)

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 2: Quality Degradation & Recovery")
    print("=" * 65)

    rng_global = np.random.default_rng(cfg.seed)
    all_idx = rng_global.permutation(holdout.n)
    p1_idx = all_idx[:PHASE_N]
    p2_idx = all_idx[PHASE_N:2 * PHASE_N]

    phase_size, total_steps, window = _phase_geometry(n_phases=3)

    mistral_normal = float(np.mean(holdout.rewards[_DEGRADED_ARM][p1_idx]))
    deg_pct = (mistral_normal - _DEGRADED_REWARD) / mistral_normal * 100
    print(f"  Degraded arm:   {ARM_SHORT[_DEGRADED_ARM]}")
    print(f"  Degraded reward: {_DEGRADED_REWARD:.2f} "
          f"(~{deg_pct:.0f}% below normal {mistral_normal:.3f})")
    print(f"  Phase size:     {phase_size} prompts x 3 phases")
    print(f"  Phase 3 reuses Phase 1 prompts (within-subject design)")

    budget_targets: dict[str, float] = dict(
        zip(K3_BUDGET_LABELS, K3_BUDGET_TARGETS, strict=True),
    )
    budget_nice: dict[str, str] = {}
    for bl in _BUDGET_LABELS:
        bt = budget_targets[bl]
        budget_nice[bl] = rf"{bl.title()} ($B{{=}}\${bt:.1e}$)"
        print(f"  Budget {bl:<10s} = ${bt:.2e}/req")

    conditions: dict[str, _AveragedCurves] = {}

    for bl in _BUDGET_LABELS:
        conditions[f"ParetoBandit ({bl})"] = _run_multi_seed_phased(
            trial_fn=_run_degradation_trial,
            trial_kwargs={
                "train": train, "holdout": holdout,
                "p1_idx": p1_idx, "p2_idx": p2_idx,
                "degraded_arm": _DEGRADED_ARM,
                "degraded_reward": _DEGRADED_REWARD,
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

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_degradation_trial,
        trial_kwargs={
            "train": train, "holdout": holdout,
            "p1_idx": p1_idx, "p2_idx": p2_idx,
            "degraded_arm": _DEGRADED_ARM,
            "degraded_reward": _DEGRADED_REWARD,
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
        valid_per_phase = ps - window + 1
        p1 = r[:valid_per_phase]
        p2 = r[valid_per_phase:2 * valid_per_phase]
        p3 = r[2 * valid_per_phase:]
        print(f"  {cond_name:<30s}  P1={np.mean(p1):.4f}  "
              f"P2={np.mean(p2):.4f}  P3={np.mean(p3):.4f}")

    out_path = Path(cfg.output_dir) / "scenario2_quality_degradation.png"
    _plot_phased_3panel(
        conditions, budget_targets, budget_nice,
        phase_boundaries, window, x_axis,
        phase_labels=_PHASE_LABELS_S2, shade_color=CB_RED,
        suptitle=(f"Quality Degradation & Recovery — {ARM_SHORT[_DEGRADED_ARM]} "
                  f"reward drops to {_DEGRADED_REWARD:.2f} in Phase 2"),
        out_path=out_path,
    )
    print(f"\n  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — Cost Drift & Recovery (§4.3, Figure 2; Exp 02)
# ═══════════════════════════════════════════════════════════════════════════

_GEMINI_NEW_INPUT: float = GEMINI_COST_DROP["new_input_cost_per_m"]
_GEMINI_NEW_OUTPUT: float = GEMINI_COST_DROP["new_output_cost_per_m"]

_PHASE_LABELS_S3 = ["Normal", "Price Drop", "Price Restored"]


def _gemini_cost_scale() -> float:
    """Effective cost multiplier for the Gemini price drop.

    Derived from the registry's original pricing and the paper's new
    pricing, exactly as ``Experiment 02`` does.
    """
    orig_input = float(MODEL_REGISTRY[GEMINI_ARM]["input_cost_per_m"])
    orig_output = float(MODEL_REGISTRY[GEMINI_ARM]["output_cost_per_m"])
    orig_avg = (orig_input + orig_output) / 2.0
    new_avg = (_GEMINI_NEW_INPUT + _GEMINI_NEW_OUTPUT) / 2.0
    return new_avg / orig_avg


def _run_cost_drift_trial(
    train: DataSplit,
    holdout: DataSplit,
    p1_idx: np.ndarray,
    p2_idx: np.ndarray,
    *,
    cost_drift_arm: str = GEMINI_ARM,
    alpha: float = BEST_K3_HPARAMS["alpha"],
    forgetting_factor: float = BEST_K3_HPARAMS["forgetting_factor"],
    cost_penalty: float = 0.0,
    budget_pacer: BudgetPacer | None = None,
    seed: int = 0,
) -> tuple[list[str], list[float], list[float], int]:
    """Three-phase cost-drift trial (§4.3, Figure 2; Exp 02).

    Phase 1 (Normal): ``p1_idx`` holdout prompts, original pricing.
    Phase 2 (Price Drop): ``p2_idx`` holdout prompts, Gemini pricing
    reduced per ``GEMINI_COST_DROP``.  Registry updated so the
    BudgetPacer (§3.2, Eqs. 3–4) reacts immediately via its EMA signal.
    Phase 3 (Price Restored): Reuses ``p1_idx`` prompts with original
    pricing restored — controlled within-subject comparison (§4.1).

    Training uses the full *train* split (no metrics recorded).

    Returns
    -------
    tuple
        ``(step_models, step_rewards, step_costs, phase_size)``.
    """
    feature_dim = train.embeddings.shape[1]
    rng = np.random.default_rng(seed)
    cost_scale = _gemini_cost_scale()
    phase_n = len(p1_idx)

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

    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        log.cost_usd = float(train.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    orig_input = float(MODEL_REGISTRY[cost_drift_arm]["input_cost_per_m"])
    orig_output = float(MODEL_REGISTRY[cost_drift_arm]["output_cost_per_m"])

    p1_order = rng.permutation(phase_n)
    p2_order = rng.permutation(len(p2_idx))
    p3_order = rng.permutation(phase_n)

    step_models: list[str] = []
    step_rewards: list[float] = []
    step_costs: list[float] = []

    # Phase 1 — Normal pricing
    for o in p1_order:
        i = p1_idx[o]
        model, log = router.route(holdout.embeddings[i])
        reward = float(holdout.rewards[model][i])
        cost = float(holdout.costs[model][i])
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)
        step_models.append(model)
        step_rewards.append(reward)
        step_costs.append(cost)

    # Phase 2 — Price drop (update registry)
    router.update_model_pricing(
        cost_drift_arm,
        input_cost_per_m=_GEMINI_NEW_INPUT,
        output_cost_per_m=_GEMINI_NEW_OUTPUT,
    )
    for o in p2_order:
        i = p2_idx[o]
        model, log = router.route(holdout.embeddings[i])
        reward = float(holdout.rewards[model][i])
        cost = float(holdout.costs[model][i])
        if model == cost_drift_arm:
            cost *= cost_scale
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)
        step_models.append(model)
        step_rewards.append(reward)
        step_costs.append(cost)

    # Phase 3 — Price restored (revert registry)
    router.update_model_pricing(
        cost_drift_arm,
        input_cost_per_m=orig_input,
        output_cost_per_m=orig_output,
    )
    for o in p3_order:
        i = p1_idx[o]
        model, log = router.route(holdout.embeddings[i])
        reward = float(holdout.rewards[model][i])
        cost = float(holdout.costs[model][i])
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)
        step_models.append(model)
        step_rewards.append(reward)
        step_costs.append(cost)

    return step_models, step_rewards, step_costs, phase_n


def run_scenario_3(
    cfg: DemoConfig,
    train: DataSplit,
    holdout: DataSplit,
) -> Path:
    """Cost drift & recovery (§4.3, Figure 2; paper Exp 02).

    Three phases of ``PHASE_N`` (608) prompts each, matching the paper's
    non-stationary protocol (§4.1):

    * Phase 1 — Normal pricing (608 holdout prompts)
    * Phase 2 — Gemini price drop (~56x cheaper, §4.3)
    * Phase 3 — Price restored (reuses Phase 1 prompts, within-subject)

    Returns
    -------
    Path
        Path to the saved figure.
    """
    print("\n" + "=" * 65)
    print("  SCENARIO 3: Cost Drift & Recovery (Gemini-Pro Price Drop)")
    print("=" * 65)

    cost_scale = _gemini_cost_scale()
    print(f"  Gemini-Pro cost multiplier in Phase 2: {cost_scale:.4f}x "
          f"(~{1 / cost_scale:.0f}x cheaper)")

    rng_global = np.random.default_rng(cfg.seed + 1000)
    all_idx = rng_global.permutation(holdout.n)
    p1_idx = all_idx[:PHASE_N]
    p2_idx = all_idx[PHASE_N:2 * PHASE_N]

    phase_size, total_steps, window = _phase_geometry(n_phases=3)

    print(f"  Phase size:     {phase_size} prompts x 3 phases")
    print(f"  Phase 3 reuses Phase 1 prompts (within-subject design)")

    budget_targets: dict[str, float] = dict(
        zip(K3_BUDGET_LABELS, K3_BUDGET_TARGETS, strict=True),
    )
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
                "train": train, "holdout": holdout,
                "p1_idx": p1_idx, "p2_idx": p2_idx,
                "cost_drift_arm": GEMINI_ARM,
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

    conditions["Unconstrained"] = _run_multi_seed_phased(
        trial_fn=_run_cost_drift_trial,
        trial_kwargs={
            "train": train, "holdout": holdout,
            "p1_idx": p1_idx, "p2_idx": p2_idx,
            "cost_drift_arm": GEMINI_ARM,
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
        valid_per_phase = ps - window + 1
        p1_r = r[:valid_per_phase]
        p2_r = r[valid_per_phase:2 * valid_per_phase]
        p3_r = r[2 * valid_per_phase:]
        p1_c = c[:valid_per_phase]
        p2_c = c[valid_per_phase:2 * valid_per_phase]
        p3_c = c[2 * valid_per_phase:]
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
                  f"{1 / cost_scale:.0f}x in Phase 2, restored in Phase 3"),
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
    holdout: DataSplit,
) -> Path:
    """Compare key configuration knobs on the quality-cost frontier.

    Demo-specific scenario (not a direct paper experiment).  Sweeps
    ``alpha`` (§3.2), ``forgetting_factor`` (§3.3), and ``cost_penalty``
    λ_c one at a time while holding the others at their defaults.  Each
    parameter is tested at three levels.

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
                    train, holdout,
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
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Independent seeds per condition (default: 10)",
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
        "--val-file", type=str, default=_default_val_path(),
        help="JSONL file for training (default: shipped val.jsonl)",
    )
    parser.add_argument(
        "--holdout-file", type=str, default=_default_holdout_path(),
        help="JSONL file for evaluation (default: shipped test_holdout.jsonl)",
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
        seed=args.seed,
        n_seeds=args.n_seeds,
        alpha=args.alpha,
        forgetting_factor=args.forgetting_factor,
        cost_penalty=args.cost_penalty,
        n_budget_targets=args.n_budget_targets,
        output_dir=args.output_dir,
        val_file=args.val_file,
        holdout_file=args.holdout_file,
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
    print(f"  Train (val):    {cfg.val_file}")
    print(f"  Eval (holdout): {cfg.holdout_file}")
    print(f"  Encoder:        {encoder_label}")
    print(f"  Seeds/cond:     {cfg.n_seeds}")
    print(f"  Seed:           {cfg.seed}")
    print(f"  Output:         {cfg.output_dir}")
    print()

    train, holdout = load_demo_splits(
        val_file=cfg.val_file,
        holdout_file=cfg.holdout_file,
        feature_service=fs,
    )

    saved: list[Path] = []
    run_all = cfg.scenario is None

    if run_all or cfg.scenario == 1:
        saved.append(run_scenario_1(cfg, train, holdout))

    if run_all or cfg.scenario == 2:
        saved.append(run_scenario_2(cfg, train, holdout))

    if run_all or cfg.scenario == 3:
        saved.append(run_scenario_3(cfg, train, holdout))

    if run_all or cfg.scenario == 4:
        saved.append(run_scenario_4(cfg, train, holdout))

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"  Demo complete in {elapsed:.1f}s")
    print("  Saved plots:")
    for p in saved:
        print(f"    {p}")
    print("=" * 65)


if __name__ == "__main__":
    main()
