#!/usr/bin/env python3
"""Figure 1 / RQ1: Cost-Quality Pareto Frontier — BanditGPT vs All Baselines.

Runs BanditGPT and a comprehensive set of baseline routers end-to-end on
real prompt data from the canonical test split, sweeping parameters to trace
each router's Pareto frontier.  This is the headline comparison for RQ1.

Baselines
---------
1. **BanditGPT** (our system): Hybrid LinUCB with warmup priors + cost penalty.
2. **Supervised (LogReg)**: Logistic regression trained on full reward labels
   for ALL models — the "full information" ceiling.  Represents the supervised
   routing paradigm (RouteLLM, RouterDC).  Sweeps classification threshold.
3. **Tabula rasa LinUCB**: Same architecture but no warmup priors and no
   Corralling — pure online contextual bandit.  Isolates the value of
   offline-to-online prior transfer.
4. **ε-greedy (contextual)**: Same features as LinUCB but ε-greedy exploration
   instead of UCB.  Isolates the value of principled exploration.
5. **Context-free UCB**: Non-contextual UCB1 with cost penalty.  Isolates
   the value of contextual (prompt-level) features.
6. **Linear Thompson Sampling (LinTS)**: Posterior sampling alternative to
   UCB.  Tests whether the exploration mechanism matters.
7. **Static random mix**: Linear interpolation between always-weak and
   always-strong endpoints (trivial baseline).

Fairness
--------
All bandit baselines share the same evaluation protocol:
  - Train on the same shuffled training split (online, partial feedback).
  - Evaluate on the same shuffled test split (online, partial feedback).
  - Same seeds, same cost_penalty grid, same number of seeds.

The supervised baseline has *strictly more information*: it observes full
reward labels for all models on all training prompts.  If BanditGPT
matches or exceeds it, that demonstrates online learning recovers the
information lost by partial observability.

Statistical treatment
---------------------
- Each seed runs an independent train→test simulation (different
  shuffle order → different online learning trajectory).
- **Pareto AUC** is computed per-seed first, then averaged.
- **CostSave@Q** 95 % CIs use a *prompt-level paired bootstrap*.
- The supervised baseline is deterministic (no seed variance)
  once trained.

Outputs
-------
``results/figure1_pareto_k2.pdf``
    Publication-quality single-panel Pareto figure with all baselines.

``results/figure1_data.json``
    Machine-readable results for downstream table generation.

Usage
-----
    python experiments/01_figure/plot_pareto_frontier.py
    python experiments/01_figure/plot_pareto_frontier.py --n-seeds 5
    python experiments/01_figure/plot_pareto_frontier.py --fast
    python experiments/01_figure/plot_pareto_frontier.py --no-baselines
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    BEST_K2_HPARAMS,
    HOLDOUT_DATA_PATH,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.pareto import (
    interpolate_pareto_cost,
    pareto_auc,
    pareto_hull,
)

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # type: ignore[misc,assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

for _noisy in (
    "bandit_gpt.router",
    "bandit_gpt.router_v2",
    "bandit_gpt.feature_service",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"
ARM_ORDER = K2_ARM_ORDER
ARM_LABELS = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}
COST_PENALTY_SWEEP = [
    0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
    0.22, 0.25, 0.28,
    0.3, 0.35, 0.4, 0.45,
    0.5, 0.6, 0.7, 0.8,
    1.0, 2.0, 5.0, 10.0,
]
COSTSAVE_THRESHOLDS = [0.90, 0.95, 0.99]

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_TEAL = "#56B4E9"
CB_YELLOW = "#F0E442"


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SplitData:
    """Pre-processed split with embeddings ready for bandit simulation."""

    prompts: List[str]
    rewards: Dict[str, np.ndarray]
    costs: Dict[str, np.ndarray]
    embeddings: np.ndarray

    @property
    def n(self) -> int:
        return len(self.prompts)


def load_split(path: Path, fs: FeatureService) -> SplitData:
    """Load a JSONL split and encode prompts into feature vectors.

    Args:
        path: Path to a JSONL file where each line contains ``prompt``
            and ``arms`` with per-model ``reward`` and ``cost``.
        fs: Feature service for encoding prompts.

    Returns:
        Fully loaded and embedded split data.
    """
    prompts: List[str] = []
    per_arm_rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            for arm_id in ARM_ORDER:
                info = r["arms"][arm_id]
                per_arm_rewards[arm_id].append(info["reward"])
                per_arm_costs[arm_id].append(info["cost"])

    rewards = {a: np.array(v) for a, v in per_arm_rewards.items()}
    costs = {a: np.array(v) for a, v in per_arm_costs.items()}

    logger.info("  Encoding %d prompts from %s ...", len(prompts), path.name)
    embeddings = fs.extract_features_batch(prompts)

    return SplitData(
        prompts=prompts, rewards=rewards, costs=costs, embeddings=embeddings,
    )


def build_model_registry() -> Dict[str, Any]:
    """Build model registry filtered to the K=2 arm set."""
    config_path = PROJECT_ROOT / "data_collection" / "config" / "models_k3.json"
    with open(config_path) as f:
        data = json.load(f)
    registry: Dict[str, Any] = {}
    for m in data["models"]:
        if m["model_id"] in ARM_ORDER:
            registry[m["model_id"]] = {
                "model_id": m["model_id"],
                "display_name": m.get("display", m["model_id"]),
                "input_cost_per_m": m["input_cost_per_m"],
                "output_cost_per_m": m["output_cost_per_m"],
            }
    return registry


def compute_normalized_costs(registry: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-model normalized costs from registry pricing.

    Uses :func:`bandit_gpt.costs.log_normalize_cost` — the same canonical
    normalization as ``BanditRouter._get_normalized_cost`` — so that a
    given ``cost_penalty`` value has identical semantics across all methods.

    Market anchors and the ``(input + output) / 2 / 1000`` blending formula
    are read from ``RouterConfig`` (single source of truth).

    Args:
        registry: Model registry with ``input_cost_per_m`` and
            ``output_cost_per_m`` entries ($/1M tokens).

    Returns:
        ``{model_id: normalized_cost}`` in [0, 1].
    """
    from bandit_gpt.costs import log_normalize_cost
    from bandit_gpt.router import RouterConfig

    cfg = RouterConfig()
    norm: Dict[str, float] = {}
    for arm in ARM_ORDER:
        meta = registry[arm]
        input_cost = meta["input_cost_per_m"]
        output_cost = meta["output_cost_per_m"]
        avg_cost_per_1k = ((input_cost + output_cost) / 2.0) / 1000.0
        norm[arm] = log_normalize_cost(
            avg_cost_per_1k,
            floor=cfg.market_cost_floor,
            ceiling=cfg.market_cost_ceiling,
        )
    return norm


# ═══════════════════════════════════════════════════════════════════════════
# BanditGPT Simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimResult:
    """Per-prompt results from a single bandit simulation run.

    Arrays are in *evaluation order* (the shuffled sequence the bandit
    processed).  ``eval_idx`` maps evaluation position ``j`` to the
    original prompt index ``i``, enabling un-shuffling back to the
    canonical prompt order for paired bootstrap resampling.
    """

    rewards: np.ndarray
    costs: np.ndarray
    choices: np.ndarray
    eval_idx: np.ndarray


def simulate_bandit(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    cost_penalty: float,
    hparams: Dict[str, Any],
    warmup_path: str,
    seed: int,
) -> SimResult:
    """Run a full train→test bandit simulation with the actual router.

    The bandit first processes the training split (updating parameters
    without recording metrics), then processes the test split where
    per-prompt rewards, costs, and arm choices are recorded.  The
    bandit continues learning during the test phase, faithfully
    reflecting online deployment conditions.

    Args:
        train: Training split data.
        test: Test split data.
        registry: Model registry (filtered to K=2 arms).
        feature_dim: Dimensionality of feature vectors.
        cost_penalty: Cost penalty weight for routing.
        hparams: Non-cost hyperparameters (alpha, policy, etc.).
        warmup_path: Path to warmup priors joblib file.
        seed: Random seed controlling data shuffle order.

    Returns:
        Per-prompt reward, cost, and arm-choice arrays for the test
        split.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    is_tabula_rasa = hparams.get("policy") == "tabula_rasa"
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="none" if is_tabula_rasa else "warmup",
        warmup_path=None if is_tabula_rasa else warmup_path,
        prior_n_effective=hparams["prior_n_effective"],
        alpha=hparams["alpha"],
        use_corralling=hparams["use_corralling"],
        cost_penalty=cost_penalty,
        forgetting_factor=hparams["forgetting_factor"],
        policy="disjoint" if is_tabula_rasa else hparams["policy"],
    )

    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}

    train_idx = rng.permutation(train.n)
    for i in train_idx:
        emb = train.embeddings[i]
        model, log = router.route(emb)
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    eval_rewards = np.zeros(test.n)
    eval_costs = np.zeros(test.n)
    eval_choices = np.zeros(test.n, dtype=np.int32)
    eval_idx = rng.permutation(test.n)
    for j, i in enumerate(eval_idx):
        emb = test.embeddings[i]
        model, log = router.route(emb)
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)
        eval_rewards[j] = reward
        eval_costs[j] = cost
        eval_choices[j] = arm_to_idx[model]

    return SimResult(
        rewards=eval_rewards, costs=eval_costs, choices=eval_choices,
        eval_idx=eval_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Routers (RQ1 Comparison)
# ═══════════════════════════════════════════════════════════════════════════


class ContextFreeUCB:
    """Non-contextual Upper Confidence Bound (UCB1) with additive cost penalty.

    Ignores prompt features entirely.  Selection rule:
        a_t = argmax_a [ μ_a + α·√(2·ln(t)/n_a) − λ·c̃_a ]

    Isolates the value of contextual features: if contextual LinUCB
    substantially outperforms this, prompt embeddings carry routing signal.

    Args:
        models: Model identifiers.
        normalized_costs: ``{model_id: normalized_cost}`` in [0, 1].
        alpha: Exploration coefficient (default 2.0 — standard UCB1).
        cost_penalty: Additive cost penalty weight λ.
    """

    def __init__(
        self,
        models: List[str],
        normalized_costs: Dict[str, float],
        alpha: float = 2.0,
        cost_penalty: float = 0.0,
    ):
        self.models = models
        self.normalized_costs = normalized_costs
        self.cost_penalty = cost_penalty
        self.alpha = alpha
        self.counts: Dict[str, int] = {m: 0 for m in models}
        self.sum_rewards: Dict[str, float] = {m: 0.0 for m in models}
        self.t = 0

    def select_model(self, context: np.ndarray) -> str:
        """Select arm via UCB1 (context is ignored)."""
        self.t += 1
        for m in self.models:
            if self.counts[m] == 0:
                return m
        scores: Dict[str, float] = {}
        for m in self.models:
            mu = self.sum_rewards[m] / self.counts[m]
            ucb_bonus = self.alpha * np.sqrt(
                2.0 * np.log(self.t) / self.counts[m]
            )
            scores[m] = (
                mu + ucb_bonus - self.cost_penalty * self.normalized_costs[m]
            )
        return max(scores, key=scores.get)

    def update(
        self, context: np.ndarray, model: str, reward: float,
    ) -> None:
        self.counts[model] += 1
        self.sum_rewards[model] += reward


class EpsilonGreedyRouter:
    """Contextual ε-greedy with online ridge regression and cost penalty.

    Uses the same prompt features as LinUCB but replaces the UCB
    exploration bonus with ε-greedy exploration:
      - With probability ε: select uniformly at random.
      - With probability 1−ε: select argmax_a [θ̂_aᵀx − λ·c̃_a].

    Isolates the value of principled (UCB) exploration: if LinUCB
    outperforms this, the confidence-based bonus meaningfully improves
    sample efficiency.

    Args:
        models: Model identifiers.
        context_dim: Dimensionality of context vectors.
        normalized_costs: ``{model_id: normalized_cost}`` in [0, 1].
        epsilon: Exploration probability (default 0.1).
        cost_penalty: Additive cost penalty weight λ.
        ridge_lambda: Regularization for the per-arm ridge regression.
    """

    def __init__(
        self,
        models: List[str],
        context_dim: int,
        normalized_costs: Dict[str, float],
        epsilon: float = 0.1,
        cost_penalty: float = 0.0,
        ridge_lambda: float = 1.0,
        rng_seed: int = 0,
    ):
        self.models = models
        self.normalized_costs = normalized_costs
        self.epsilon = epsilon
        self.cost_penalty = cost_penalty
        self.d = context_dim
        self._rng = np.random.default_rng(rng_seed)
        self.A = {m: ridge_lambda * np.eye(context_dim) for m in models}
        self.A_inv = {m: np.eye(context_dim) / ridge_lambda for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}

    def select_model(self, context: np.ndarray) -> str:
        """Select model: ε-random, (1−ε)-greedy on predicted reward − cost."""
        if self._rng.random() < self.epsilon:
            return self.models[self._rng.integers(len(self.models))]
        scores: Dict[str, float] = {}
        x = context.flatten()
        for m in self.models:
            theta = self.A_inv[m] @ self.b[m]
            expected = float(np.clip(theta @ x, -1.0, 2.0))
            scores[m] = (
                expected - self.cost_penalty * self.normalized_costs[m]
            )
        return max(scores, key=scores.get)

    def update(
        self, context: np.ndarray, model: str, reward: float,
    ) -> None:
        x = context.flatten()
        A_inv = self.A_inv[model]
        A_inv_x = A_inv @ x
        denom = 1.0 + x @ A_inv_x
        if abs(denom) > 1e-6:
            self.A_inv[model] = A_inv - np.outer(A_inv_x, A_inv_x) / denom
        else:
            self.A[model] += np.outer(x, x)
            self.A_inv[model] = np.linalg.inv(self.A[model])
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x


def simulate_lightweight_baseline(
    train: SplitData,
    test: SplitData,
    baseline_factory,
    *,
    seed: int,
) -> SimResult:
    """Simulate a lightweight baseline router through train→test.

    The baseline follows the same protocol as BanditGPT:
    1. Process training prompts in shuffled order (learning, not recorded).
    2. Process test prompts in shuffled order (learning + recording).

    The ``baseline_factory`` callable receives no arguments and returns
    a fresh router instance with ``select_model(context)`` and
    ``update(context, model, reward)`` methods.

    Args:
        train: Training split data.
        test: Test split data.
        baseline_factory: Zero-arg callable returning a fresh router.
        seed: Random seed for shuffle order.

    Returns:
        ``SimResult`` with per-prompt test outcomes and shuffle index.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    router = baseline_factory()
    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}

    for i in rng.permutation(train.n):
        model = router.select_model(train.embeddings[i])
        reward = float(train.rewards[model][i])
        router.update(train.embeddings[i], model, reward)

    eval_rewards = np.zeros(test.n)
    eval_costs = np.zeros(test.n)
    eval_choices = np.zeros(test.n, dtype=np.int32)
    eval_idx = rng.permutation(test.n)
    for j, i in enumerate(eval_idx):
        model = router.select_model(test.embeddings[i])
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        router.update(test.embeddings[i], model, reward)
        eval_rewards[j] = reward
        eval_costs[j] = cost
        eval_choices[j] = arm_to_idx[model]

    return SimResult(
        rewards=eval_rewards, costs=eval_costs, choices=eval_choices,
        eval_idx=eval_idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Hyperparameter Tuning (on val split)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TunedBaselineHparams:
    """Best hyperparameters for each baseline, selected on val Pareto AUC."""

    ucb_alpha: float = 2.0
    epsilon: float = 0.1
    lints_noise_variance: float = 0.25
    logreg_C: float = 1.0


def tune_baseline_hparams(
    train: SplitData,
    val: SplitData,
    feature_dim: int,
    registry: Dict[str, Any],
    cost_penalties: List[float],
    n_seeds: int = 3,
    seed_offset: int = 5000,
) -> TunedBaselineHparams:
    """Select baseline hyperparameters by val-split Pareto AUC.

    Each baseline's key hyperparameter is swept over a small grid.  For
    each candidate value the baseline is run through the standard
    train→val protocol and the Pareto AUC on the val split is computed.
    The value maximising val AUC is selected.

    This mirrors the tuning protocol used for BanditGPT (which was tuned
    via ``tune_hybrid_router.py`` on the same val split).

    Args:
        train: Training data (online learning phase).
        val: Validation data (evaluation phase — NOT test).
        feature_dim: Context dimensionality.
        registry: Model registry for cost normalization.
        cost_penalties: Cost-penalty grid shared across all methods.
        n_seeds: Seeds per (hparam, cost_penalty) combination.
        seed_offset: Base seed (avoid collision with test evaluation).

    Returns:
        :class:`TunedBaselineHparams` with the best value for each baseline.
    """
    norm_costs = compute_normalized_costs(registry)
    fixed_costs = [float(val.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(val.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo, cost_hi = min(fixed_costs), max(fixed_costs)

    def _val_auc(
        baseline_factory_for_cp,
    ) -> float:
        """Compute mean Pareto AUC across seeds on the val split."""
        per_seed_auc: List[float] = []
        for s in range(n_seeds):
            seed = seed_offset + s
            seed_costs: List[float] = []
            seed_rewards: List[float] = []
            for cp in cost_penalties:
                factory = baseline_factory_for_cp(cp)
                result = simulate_lightweight_baseline(
                    train, val, factory, seed=seed,
                )
                seed_costs.append(float(result.costs.mean()))
                seed_rewards.append(float(result.rewards.mean()))
            all_c = seed_costs + fixed_costs
            all_r = seed_rewards + fixed_rewards
            per_seed_auc.append(pareto_auc(all_c, all_r, cost_lo, cost_hi))
        return float(np.mean(per_seed_auc))

    # ── 1. Context-Free UCB alpha ─────────────────────────────────────
    UCB_ALPHA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_ucb_alpha, best_ucb_auc = 2.0, -1.0
    logger.info("  Tuning Context-Free UCB alpha (%d candidates) ...",
                len(UCB_ALPHA_GRID))
    for alpha in UCB_ALPHA_GRID:
        auc = _val_auc(
            lambda cp, _a=alpha: (
                lambda: ContextFreeUCB(
                    ARM_ORDER, norm_costs, alpha=_a, cost_penalty=cp,
                )
            ),
        )
        logger.info("    UCB alpha=%.1f → val AUC=%.4f", alpha, auc)
        if auc > best_ucb_auc:
            best_ucb_alpha, best_ucb_auc = alpha, auc

    # ── 2. ε-Greedy epsilon ───────────────────────────────────────────
    EPSILON_GRID = [0.01, 0.05, 0.1, 0.2, 0.3]
    best_eps, best_eps_auc = 0.1, -1.0
    logger.info("  Tuning ε-greedy epsilon (%d candidates) ...",
                len(EPSILON_GRID))
    for eps in EPSILON_GRID:
        _counter = [0]

        def _make_factory(cp: float, _eps=eps):
            _inner = [0]

            def _factory():
                _inner[0] += 1
                return EpsilonGreedyRouter(
                    ARM_ORDER, feature_dim, norm_costs,
                    epsilon=_eps, cost_penalty=cp,
                    rng_seed=seed_offset + _inner[0],
                )
            return _factory

        auc = _val_auc(_make_factory)
        logger.info("    ε=%.2f → val AUC=%.4f", eps, auc)
        if auc > best_eps_auc:
            best_eps, best_eps_auc = eps, auc

    # ── 3. LinTS noise_variance ───────────────────────────────────────
    from bandit_gpt.baselines import CostAwareLinTSRouter
    NV_GRID = [0.05, 0.1, 0.25, 0.5, 1.0]
    _lints_costs = {
        m: {"normalized_cost": norm_costs[m]} for m in ARM_ORDER
    }
    best_nv, best_nv_auc = 0.25, -1.0
    logger.info("  Tuning LinTS noise_variance (%d candidates) ...",
                len(NV_GRID))
    for nv in NV_GRID:
        auc = _val_auc(
            lambda cp, _nv=nv: (
                lambda: CostAwareLinTSRouter(
                    ARM_ORDER, feature_dim,
                    model_costs=_lints_costs,
                    cost_penalty=cp,
                    noise_variance=_nv,
                )
            ),
        )
        logger.info("    noise_var=%.2f → val AUC=%.4f", nv, auc)
        if auc > best_nv_auc:
            best_nv, best_nv_auc = nv, auc

    # ── 4. Logistic Regression C ──────────────────────────────────────
    C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C, best_C_auc = 1.0, -1.0
    if LogisticRegression is not None:
        logger.info("  Tuning LogReg C (%d candidates) ...", len(C_GRID))
        weak, strong = ARM_ORDER[0], ARM_ORDER[-1]
        y_train = (train.rewards[strong] > train.rewards[weak]).astype(int)
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
            clf.fit(train.embeddings, y_train)
            val_probs = clf.predict_proba(val.embeddings)[:, 1]
            thresholds = np.linspace(0.0, 1.0, 52)[1:-1]
            sup_costs_list: List[float] = []
            sup_rewards_list: List[float] = []
            for t in thresholds:
                route_strong = val_probs > t
                sup_rewards_list.append(float(np.where(
                    route_strong, val.rewards[strong], val.rewards[weak],
                ).mean()))
                sup_costs_list.append(float(np.where(
                    route_strong, val.costs[strong], val.costs[weak],
                ).mean()))
            all_c = sup_costs_list + fixed_costs
            all_r = sup_rewards_list + fixed_rewards
            auc = pareto_auc(all_c, all_r, cost_lo, cost_hi)
            logger.info("    C=%.2f → val AUC=%.4f", C, auc)
            if auc > best_C_auc:
                best_C, best_C_auc = C, auc

    result = TunedBaselineHparams(
        ucb_alpha=best_ucb_alpha,
        epsilon=best_eps,
        lints_noise_variance=best_nv,
        logreg_C=best_C,
    )
    logger.info(
        "  Tuned baseline hparams: UCB α=%.1f, ε=%.2f, "
        "LinTS σ²=%.2f, LogReg C=%.1f",
        result.ucb_alpha, result.epsilon,
        result.lints_noise_variance, result.logreg_C,
    )
    return result


def evaluate_supervised_frontier(
    train: SplitData,
    test: SplitData,
    n_thresholds: int = 50,
    C: float = 1.0,
) -> List[Dict[str, float]]:
    """Train a supervised logistic regression router and sweep thresholds.

    The supervised baseline receives *full* reward labels for all models
    on all training prompts — strictly more information than any bandit.
    It trains a logistic regression to predict P(strong model is better)
    from the same PCA features used by the bandit.  At test time, it
    sweeps the classification threshold to trace a Pareto frontier.

    This represents the supervised routing paradigm (RouteLLM, RouterDC):
    a frozen classifier trained offline on labeled data.

    Args:
        train: Training data with full reward labels for all models.
        test: Test data (same features; rewards used only for evaluation).
        n_thresholds: Number of threshold values to sweep.
        C: Inverse regularization strength (val-tuned).

    Returns:
        List of ``{"threshold", "mean_reward", "mean_cost", "pct_strong"}``
        dicts, one per threshold, sorted by ascending mean cost.

    Raises:
        RuntimeError: If ``scikit-learn`` is not installed.
    """
    if LogisticRegression is None:
        raise RuntimeError(
            "scikit-learn is required for the supervised baseline. "
            "Install with: pip install scikit-learn"
        )
    weak, strong = ARM_ORDER[0], ARM_ORDER[-1]

    y_train = (train.rewards[strong] > train.rewards[weak]).astype(int)
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    clf.fit(train.embeddings, y_train)

    test_probs = clf.predict_proba(test.embeddings)[:, 1]

    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    points: List[Dict[str, float]] = []
    for t in thresholds:
        route_strong = test_probs > t
        per_prompt_reward = np.where(
            route_strong, test.rewards[strong], test.rewards[weak],
        )
        per_prompt_cost = np.where(
            route_strong, test.costs[strong], test.costs[weak],
        )
        points.append({
            "threshold": float(t),
            "mean_reward": float(per_prompt_reward.mean()),
            "mean_cost": float(per_prompt_cost.mean()),
            "pct_strong": float(route_strong.mean()),
        })
    points.sort(key=lambda p: p["mean_cost"])
    return points


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Sweep Orchestration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BaselineFrontier:
    """Pareto frontier data for a single baseline router."""

    name: str
    sweep_points: List[Dict[str, float]]
    hull_costs: List[float]
    hull_rewards: List[float]
    pareto_auc: float
    n_seeds: int


def _run_bandit_baseline_sweep(
    name: str,
    train: SplitData,
    test: SplitData,
    baseline_factory_for_cp,
    cost_penalties: List[float],
    n_seeds: int,
    seed_offset: int,
) -> BaselineFrontier:
    """Sweep cost_penalty for a bandit baseline and build its Pareto frontier.

    Args:
        name: Display name for this baseline.
        train: Training split.
        test: Test split.
        baseline_factory_for_cp: Callable ``(cost_penalty) -> factory``
            where ``factory()`` returns a fresh router instance.
        cost_penalties: Grid of cost_penalty values.
        n_seeds: Number of random seeds.
        seed_offset: Base seed offset.

    Returns:
        ``BaselineFrontier`` with sweep points and Pareto hull.
    """
    fixed_costs = [float(test.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(test.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo, cost_hi = min(fixed_costs), max(fixed_costs)

    cp_rewards: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}
    cp_costs: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}

    for s in range(n_seeds):
        seed = seed_offset + s
        for cp in cost_penalties:
            factory = baseline_factory_for_cp(cp)
            result = simulate_lightweight_baseline(
                train, test, factory, seed=seed,
            )
            cp_rewards[cp].append(float(result.rewards.mean()))
            cp_costs[cp].append(float(result.costs.mean()))

    sweep_points: List[Dict[str, float]] = []
    for cp in cost_penalties:
        sweep_points.append({
            "cost_penalty": cp,
            "mean_reward": float(np.mean(cp_rewards[cp])),
            "mean_cost": float(np.mean(cp_costs[cp])),
            "std_reward": (
                float(np.std(cp_rewards[cp], ddof=1))
                if n_seeds > 1 else 0.0
            ),
        })

    all_costs = [p["mean_cost"] for p in sweep_points] + fixed_costs
    all_rewards = [p["mean_reward"] for p in sweep_points] + fixed_rewards
    hull_c, hull_r = pareto_hull(all_costs, all_rewards)
    auc = pareto_auc(all_costs, all_rewards, cost_lo, cost_hi)

    return BaselineFrontier(
        name=name,
        sweep_points=sweep_points,
        hull_costs=hull_c,
        hull_rewards=hull_r,
        pareto_auc=auc,
        n_seeds=n_seeds,
    )


def _load_scaled_warmup_priors(
    warmup_path: str,
    prior_n_effective: float,
) -> Optional[Dict[str, Any]]:
    """Load warmup priors and apply the same n_effective scaling as BanditRouter.

    Returns the scaled priors dict with keys ``A``, ``b``, ``context_dim``,
    or ``None`` if loading fails.

    Args:
        warmup_path: Path to a ``.joblib`` warmup priors file.
        prior_n_effective: Effective sample count (controls prior strength).
    """
    try:
        import joblib
        warmup_data = joblib.load(warmup_path)
        n_warmup = max(warmup_data.get("n", 20000), 1)
        scale = prior_n_effective / float(n_warmup)
        scaled: Dict[str, Any] = {
            "context_dim": warmup_data["context_dim"],
            "A": {
                m: A_m * scale
                for m, A_m in warmup_data.get("A", {}).items()
            },
            "b": {
                m: b_m * scale
                for m, b_m in warmup_data.get("b", {}).items()
            },
        }
        return scaled
    except Exception as exc:
        logger.warning("Failed to load warmup priors for baseline: %s", exc)
        return None


def run_all_baselines(
    train: SplitData,
    test: SplitData,
    feature_dim: int,
    registry: Dict[str, Any],
    cost_penalties: List[float],
    n_seeds: int,
    seed_offset: int = 1000,
    warmup_path: Optional[str] = None,
    prior_n_effective: float = 10.0,
    tuned_hparams: Optional[TunedBaselineHparams] = None,
) -> Dict[str, BaselineFrontier]:
    """Run all baseline routers and return their Pareto frontiers.

    Args:
        train: Training split.
        test: Test split.
        feature_dim: Feature vector dimensionality.
        registry: Model registry for cost normalization.
        cost_penalties: Grid of cost_penalty values.
        n_seeds: Number of seeds per baseline.
        seed_offset: Seed offset (should match BanditGPT's).
        warmup_path: Path to warmup priors (for LinTS warm-started variant).
        prior_n_effective: Effective sample count for prior scaling.
        tuned_hparams: Val-tuned hyperparameters for each baseline.
            If ``None``, falls back to literature defaults.

    Returns:
        ``{baseline_name: BaselineFrontier}`` dict.
    """
    hp = tuned_hparams or TunedBaselineHparams()
    logger.info(
        "  Baseline hparams: UCB α=%.1f, ε=%.2f, LinTS σ²=%.2f, "
        "LogReg C=%.1f%s",
        hp.ucb_alpha, hp.epsilon, hp.lints_noise_variance, hp.logreg_C,
        " (val-tuned)" if tuned_hparams else " (defaults)",
    )

    norm_costs = compute_normalized_costs(registry)
    warmup_priors = (
        _load_scaled_warmup_priors(warmup_path, prior_n_effective)
        if warmup_path else None
    )
    fixed_costs = [float(test.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(test.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo, cost_hi = min(fixed_costs), max(fixed_costs)

    baselines: Dict[str, BaselineFrontier] = {}

    # ── 1. Context-Free UCB ───────────────────────────────────────────
    logger.info("  Running Context-Free UCB (α=%.1f) ...", hp.ucb_alpha)
    baselines["Context-free UCB"] = _run_bandit_baseline_sweep(
        "Context-free UCB", train, test,
        baseline_factory_for_cp=lambda cp: (
            lambda: ContextFreeUCB(
                ARM_ORDER, norm_costs, alpha=hp.ucb_alpha, cost_penalty=cp,
            )
        ),
        cost_penalties=cost_penalties,
        n_seeds=n_seeds,
        seed_offset=seed_offset,
    )

    # ── 2. ε-Greedy (contextual) ─────────────────────────────────────
    logger.info("  Running ε-greedy (ε=%.2f) ...", hp.epsilon)
    _tuned_eps = hp.epsilon

    def _make_epsgreedy_factory(cp: float):
        _counter = [0]

        def _factory():
            _counter[0] += 1
            return EpsilonGreedyRouter(
                ARM_ORDER, feature_dim, norm_costs,
                epsilon=_tuned_eps, cost_penalty=cp,
                rng_seed=seed_offset + _counter[0],
            )
        return _factory

    baselines["ε-greedy"] = _run_bandit_baseline_sweep(
        "ε-greedy", train, test,
        baseline_factory_for_cp=_make_epsgreedy_factory,
        cost_penalties=cost_penalties,
        n_seeds=n_seeds,
        seed_offset=seed_offset,
    )

    # ── 3. Linear Thompson Sampling (with warmup priors) ────────────
    from bandit_gpt.baselines import CostAwareLinTSRouter
    _lints_costs = {
        m: {"normalized_cost": norm_costs[m]} for m in ARM_ORDER
    }
    _tuned_nv = hp.lints_noise_variance

    if warmup_priors is not None:
        logger.info("  Running LinTS (warm-started, σ²=%.2f) ...", _tuned_nv)
        baselines["LinTS"] = _run_bandit_baseline_sweep(
            "LinTS", train, test,
            baseline_factory_for_cp=lambda cp: (
                lambda: CostAwareLinTSRouter(
                    ARM_ORDER, feature_dim,
                    model_costs=_lints_costs,
                    cost_penalty=cp,
                    noise_variance=_tuned_nv,
                    warmup_priors=warmup_priors,
                )
            ),
            cost_penalties=cost_penalties,
            n_seeds=n_seeds,
            seed_offset=seed_offset,
        )
    else:
        logger.info("  Running LinTS (tabula rasa, σ²=%.2f) ...", _tuned_nv)
        baselines["LinTS"] = _run_bandit_baseline_sweep(
            "LinTS", train, test,
            baseline_factory_for_cp=lambda cp: (
                lambda: CostAwareLinTSRouter(
                    ARM_ORDER, feature_dim,
                    model_costs=_lints_costs,
                    cost_penalty=cp,
                    noise_variance=_tuned_nv,
                )
            ),
            cost_penalties=cost_penalties,
            n_seeds=n_seeds,
            seed_offset=seed_offset,
        )

    # ── 4. Supervised (Logistic Regression) ───────────────────────────
    logger.info("  Running Supervised (LogReg, C=%.1f) ...", hp.logreg_C)
    supervised_points = evaluate_supervised_frontier(
        train, test, n_thresholds=80, C=hp.logreg_C,
    )
    sup_costs = [p["mean_cost"] for p in supervised_points] + fixed_costs
    sup_rewards = [p["mean_reward"] for p in supervised_points] + fixed_rewards
    sup_hull_c, sup_hull_r = pareto_hull(sup_costs, sup_rewards)
    sup_auc = pareto_auc(sup_costs, sup_rewards, cost_lo, cost_hi)
    baselines["Supervised (LogReg)"] = BaselineFrontier(
        name="Supervised (LogReg)",
        sweep_points=supervised_points,
        hull_costs=sup_hull_c,
        hull_rewards=sup_hull_r,
        pareto_auc=sup_auc,
        n_seeds=1,
    )

    return baselines


# ═══════════════════════════════════════════════════════════════════════════
# BanditGPT Sweep + Pareto Computation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SweepPoint:
    """Aggregated statistics for a single cost_penalty value."""

    cost_penalty: float
    mean_reward: float
    std_reward: float
    mean_cost: float
    std_cost: float
    pct_weak: float
    pct_strong: float
    per_seed_rewards: List[float] = field(default_factory=list)
    per_seed_costs: List[float] = field(default_factory=list)


@dataclass
class FrontierResult:
    """Complete frontier evaluation results.

    Per-prompt arrays (``per_prompt_bandit_*``, ``baseline_per_prompt_*``)
    are stored in the *original* prompt order (matching the JSONL row
    index) so the bootstrap can pair bandit and baseline outcomes for
    the same prompt.
    """

    sweep_points: List[SweepPoint]
    pareto_auc_mean: float
    pareto_auc_std: float
    pareto_auc_per_seed: List[float]
    hull_costs: List[float]
    hull_rewards: List[float]
    baselines: Dict[str, Dict[str, float]]
    static_auc: float
    hparams: Dict[str, Any]
    n_seeds: int
    oracle_reward: float

    per_prompt_bandit_rewards: Optional[Dict[float, np.ndarray]] = None
    per_prompt_bandit_costs: Optional[Dict[float, np.ndarray]] = None
    baseline_per_prompt_rewards: Optional[Dict[str, np.ndarray]] = None
    baseline_per_prompt_costs: Optional[Dict[str, np.ndarray]] = None


def run_frontier_sweep(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    hparams: Dict[str, Any],
    warmup_path: str,
    cost_penalties: List[float],
    n_seeds: int,
    seed_offset: int = 1000,
    *,
    tabula_rasa: bool = False,
) -> FrontierResult:
    """Sweep cost_penalty and build the bandit Pareto frontier.

    For each seed independently:
      1. Run simulation for every cost_penalty value.
      2. Build the Pareto hull from sweep points + fixed-model endpoints.
      3. Compute that seed's Pareto AUC.

    Args:
        train: Training data (bandit learns, not evaluated).
        test: Test data (bandit evaluated, continues learning).
        registry: K=2 model registry.
        feature_dim: Feature vector dimensionality.
        hparams: Best hyperparameters (non-cost).
        warmup_path: Path to warmup priors.
        cost_penalties: Grid of cost_penalty values to sweep.
        n_seeds: Number of independent random seeds.
        seed_offset: Base offset added to seed index.
        tabula_rasa: If True, override to tabula-rasa mode (no warmup,
            disjoint policy, no corralling).

    Returns:
        Comprehensive frontier results with per-seed statistics and
        per-prompt arrays for bootstrap CIs.
    """
    effective_hparams = dict(hparams)
    if tabula_rasa:
        effective_hparams = {
            "alpha": hparams["alpha"],
            "prior_n_effective": 1.0,
            "policy": "tabula_rasa",
            "use_corralling": False,
            "forgetting_factor": 1.0,
        }

    n_arms = len(ARM_ORDER)
    fixed_costs = [float(test.costs[a].mean()) for a in ARM_ORDER]
    fixed_rewards = [float(test.rewards[a].mean()) for a in ARM_ORDER]
    cost_lo = min(fixed_costs)
    cost_hi = max(fixed_costs)

    oracle_per_prompt = np.maximum.reduce(
        [test.rewards[a] for a in ARM_ORDER]
    )
    oracle_reward = float(oracle_per_prompt.mean())

    baselines = {}
    for arm_id in ARM_ORDER:
        label = ARM_LABELS[arm_id]
        baselines[label] = {
            "mean_reward": float(test.rewards[arm_id].mean()),
            "mean_cost": float(test.costs[arm_id].mean()),
        }

    static_auc = pareto_auc(fixed_costs, fixed_rewards, cost_lo, cost_hi)

    cp_rewards: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}
    cp_costs: Dict[float, List[float]] = {cp: [] for cp in cost_penalties}
    cp_choices: Dict[float, List[np.ndarray]] = {
        cp: [] for cp in cost_penalties
    }

    pp_rewards_accum: Dict[float, List[np.ndarray]] = {
        cp: [] for cp in cost_penalties
    }
    pp_costs_accum: Dict[float, List[np.ndarray]] = {
        cp: [] for cp in cost_penalties
    }

    per_seed_auc: List[float] = []

    for s in range(n_seeds):
        seed = seed_offset + s
        seed_costs_list: List[float] = []
        seed_rewards_list: List[float] = []

        logger.info("  Seed %d/%d (seed=%d)", s + 1, n_seeds, seed)
        for cp in cost_penalties:
            result = simulate_bandit(
                train, test, registry, feature_dim,
                cost_penalty=cp,
                hparams=effective_hparams,
                warmup_path=warmup_path,
                seed=seed,
            )
            mr = float(result.rewards.mean())
            mc = float(result.costs.mean())
            seed_costs_list.append(mc)
            seed_rewards_list.append(mr)

            cp_rewards[cp].append(mr)
            cp_costs[cp].append(mc)
            arm_counts = np.bincount(result.choices, minlength=n_arms)
            cp_choices[cp].append(arm_counts)

            prompt_order_r = np.empty(test.n)
            prompt_order_c = np.empty(test.n)
            prompt_order_r[result.eval_idx] = result.rewards
            prompt_order_c[result.eval_idx] = result.costs
            pp_rewards_accum[cp].append(prompt_order_r)
            pp_costs_accum[cp].append(prompt_order_c)

        all_c = seed_costs_list + fixed_costs
        all_r = seed_rewards_list + fixed_rewards
        seed_auc = pareto_auc(all_c, all_r, cost_lo, cost_hi)
        per_seed_auc.append(seed_auc)
        logger.info(
            "    Seed %d AUC: %.6f (static: %.6f, Δ=%+.3f%%)",
            s + 1, seed_auc, static_auc,
            (seed_auc - static_auc) / static_auc * 100,
        )

    sweep_points: List[SweepPoint] = []
    for cp in cost_penalties:
        rewards_arr = np.array(cp_rewards[cp])
        costs_arr = np.array(cp_costs[cp])
        choices_arr = np.array([c.astype(float) for c in cp_choices[cp]])
        mean_choices = choices_arr.mean(axis=0)
        total = mean_choices.sum()
        sweep_points.append(SweepPoint(
            cost_penalty=cp,
            mean_reward=float(rewards_arr.mean()),
            std_reward=(
                float(rewards_arr.std(ddof=1)) if n_seeds > 1 else 0.0
            ),
            mean_cost=float(costs_arr.mean()),
            std_cost=(
                float(costs_arr.std(ddof=1)) if n_seeds > 1 else 0.0
            ),
            pct_weak=float(mean_choices[0] / total) if total > 0 else 0.0,
            pct_strong=(
                float(mean_choices[-1] / total) if total > 0 else 0.0
            ),
            per_seed_rewards=cp_rewards[cp],
            per_seed_costs=cp_costs[cp],
        ))

    avg_costs = [sp.mean_cost for sp in sweep_points] + fixed_costs
    avg_rewards = [sp.mean_reward for sp in sweep_points] + fixed_rewards
    hull_c, hull_r = pareto_hull(avg_costs, avg_rewards)

    auc_mean = float(np.mean(per_seed_auc))
    auc_std = (
        float(np.std(per_seed_auc, ddof=1)) if n_seeds > 1 else 0.0
    )

    pp_bandit_rewards = {
        cp: np.mean(np.stack(arrs), axis=0)
        for cp, arrs in pp_rewards_accum.items()
    }
    pp_bandit_costs = {
        cp: np.mean(np.stack(arrs), axis=0)
        for cp, arrs in pp_costs_accum.items()
    }

    return FrontierResult(
        sweep_points=sweep_points,
        pareto_auc_mean=auc_mean,
        pareto_auc_std=auc_std,
        pareto_auc_per_seed=per_seed_auc,
        hull_costs=hull_c,
        hull_rewards=hull_r,
        baselines=baselines,
        static_auc=static_auc,
        hparams=effective_hparams,
        n_seeds=n_seeds,
        oracle_reward=oracle_reward,
        per_prompt_bandit_rewards=pp_bandit_rewards,
        per_prompt_bandit_costs=pp_bandit_costs,
        baseline_per_prompt_rewards={a: test.rewards[a] for a in ARM_ORDER},
        baseline_per_prompt_costs={a: test.costs[a] for a in ARM_ORDER},
    )


# ═══════════════════════════════════════════════════════════════════════════
# CostSave@Q with Bootstrap CIs
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CostSaveResult:
    """CostSave at a single quality threshold with uncertainty."""

    threshold: float
    target_reward: float
    bandit_cost: Optional[float]
    bandit_saving_pct: Optional[float]
    baseline_cost: Optional[float]
    baseline_saving_pct: Optional[float]
    advantage_pp: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]


def _costsave_at_threshold(
    hull_c: List[float],
    hull_r: List[float],
    strong_cost: float,
    strong_reward: float,
    q: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute cost and saving percentage at quality threshold q."""
    target_r = q * strong_reward
    cost_at_q = interpolate_pareto_cost(hull_c, hull_r, target_r)
    if cost_at_q is not None:
        saving = (1.0 - cost_at_q / strong_cost) * 100
        return cost_at_q, saving
    return None, None


def compute_costsave_with_bootstrap(
    frontier: FrontierResult,
    thresholds: List[float],
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
) -> List[CostSaveResult]:
    """Compute CostSave@Q with prompt-level paired bootstrap CIs.

    For each bootstrap iteration the same resampled prompt indices
    are used for *both* the bandit sweep points and the fixed-model
    baselines (paired bootstrap), then the Pareto hulls and CostSave
    are recomputed.

    Args:
        frontier: Results from ``run_frontier_sweep``.
        thresholds: Quality fractions (e.g. [0.90, 0.95, 0.99]).
        n_bootstrap: Number of bootstrap resamples.
        bootstrap_seed: RNG seed for reproducibility.

    Returns:
        One ``CostSaveResult`` per threshold.
    """
    if frontier.per_prompt_bandit_rewards is None:
        raise ValueError(
            "FrontierResult missing per-prompt data — "
            "re-run run_frontier_sweep with the latest code."
        )

    rng = np.random.default_rng(bootstrap_seed)

    pp_rewards = frontier.per_prompt_bandit_rewards
    pp_costs = frontier.per_prompt_bandit_costs
    bl_pp_rewards = frontier.baseline_per_prompt_rewards
    bl_pp_costs = frontier.baseline_per_prompt_costs
    n_prompts = len(next(iter(pp_rewards.values())))

    strong_arm = max(
        ARM_ORDER,
        key=lambda a: frontier.baselines[ARM_LABELS[a]]["mean_reward"],
    )
    strong_label = ARM_LABELS[strong_arm]
    strong_cost = frontier.baselines[strong_label]["mean_cost"]
    strong_reward = frontier.baselines[strong_label]["mean_reward"]

    fixed_costs = [
        frontier.baselines[ARM_LABELS[a]]["mean_cost"] for a in ARM_ORDER
    ]
    fixed_rewards = [
        frontier.baselines[ARM_LABELS[a]]["mean_reward"] for a in ARM_ORDER
    ]

    bandit_hull_c, bandit_hull_r = pareto_hull(
        [sp.mean_cost for sp in frontier.sweep_points] + fixed_costs,
        [sp.mean_reward for sp in frontier.sweep_points] + fixed_rewards,
    )
    baseline_hull_c, baseline_hull_r = pareto_hull(
        fixed_costs, fixed_rewards,
    )

    observed: Dict[
        float, Tuple[Optional[float], Optional[float], Optional[float]]
    ] = {}
    for q in thresholds:
        _, b_sav = _costsave_at_threshold(
            bandit_hull_c, bandit_hull_r, strong_cost, strong_reward, q,
        )
        _, bl_sav = _costsave_at_threshold(
            baseline_hull_c, baseline_hull_r, strong_cost, strong_reward, q,
        )
        adv = None
        if b_sav is not None and bl_sav is not None:
            adv = b_sav - bl_sav
        observed[q] = (b_sav, bl_sav, adv)

    boot_advantages: Dict[float, List[float]] = {q: [] for q in thresholds}
    cost_penalties = [sp.cost_penalty for sp in frontier.sweep_points]

    for _ in range(n_bootstrap):
        idx = rng.choice(n_prompts, size=n_prompts, replace=True)

        boot_sweep_costs: List[float] = []
        boot_sweep_rewards: List[float] = []
        for cp in cost_penalties:
            boot_sweep_rewards.append(float(pp_rewards[cp][idx].mean()))
            boot_sweep_costs.append(float(pp_costs[cp][idx].mean()))

        boot_fixed_costs = [
            float(bl_pp_costs[a][idx].mean()) for a in ARM_ORDER
        ]
        boot_fixed_rewards = [
            float(bl_pp_rewards[a][idx].mean()) for a in ARM_ORDER
        ]

        boot_strong_idx = int(np.argmax(boot_fixed_rewards))
        boot_strong_cost = boot_fixed_costs[boot_strong_idx]
        boot_strong_reward = boot_fixed_rewards[boot_strong_idx]

        boot_bandit_hull_c, boot_bandit_hull_r = pareto_hull(
            boot_sweep_costs + boot_fixed_costs,
            boot_sweep_rewards + boot_fixed_rewards,
        )
        boot_baseline_hull_c, boot_baseline_hull_r = pareto_hull(
            boot_fixed_costs, boot_fixed_rewards,
        )

        for q in thresholds:
            _, b_sav = _costsave_at_threshold(
                boot_bandit_hull_c, boot_bandit_hull_r,
                boot_strong_cost, boot_strong_reward, q,
            )
            _, bl_sav = _costsave_at_threshold(
                boot_baseline_hull_c, boot_baseline_hull_r,
                boot_strong_cost, boot_strong_reward, q,
            )
            if b_sav is not None and bl_sav is not None:
                boot_advantages[q].append(b_sav - bl_sav)

    results: List[CostSaveResult] = []
    for q in thresholds:
        b_sav, bl_sav, adv = observed[q]
        target_r = q * strong_reward

        b_cost, _ = _costsave_at_threshold(
            bandit_hull_c, bandit_hull_r, strong_cost, strong_reward, q,
        )
        bl_cost, _ = _costsave_at_threshold(
            baseline_hull_c, baseline_hull_r, strong_cost, strong_reward, q,
        )

        ci_lo: Optional[float] = None
        ci_hi: Optional[float] = None
        boot_arr = boot_advantages[q]
        if len(boot_arr) >= 20:
            ci_lo = float(np.percentile(boot_arr, 2.5))
            ci_hi = float(np.percentile(boot_arr, 97.5))

        results.append(CostSaveResult(
            threshold=q,
            target_reward=round(target_r, 4),
            bandit_cost=b_cost,
            bandit_saving_pct=(
                round(b_sav, 2) if b_sav is not None else None
            ),
            baseline_cost=bl_cost,
            baseline_saving_pct=(
                round(bl_sav, 2) if bl_sav is not None else None
            ),
            advantage_pp=round(adv, 2) if adv is not None else None,
            ci_lower=round(ci_lo, 2) if ci_lo is not None else None,
            ci_upper=round(ci_hi, 2) if ci_hi is not None else None,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Gap@Oracle
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GapAtOracleResult:
    """Normalised gap between a router's quality and the per-instance oracle.

    Gap@Oracle := (R_oracle − R_router) / (R_oracle − R_weak) × 100.
    """

    cost_penalty: float
    router_reward: float
    oracle_reward: float
    weak_reward: float
    gap_pct: float


def compute_gap_at_oracle(
    frontier: FrontierResult,
) -> List[GapAtOracleResult]:
    """Compute normalised Gap@Oracle for every sweep operating point."""
    oracle_r = frontier.oracle_reward
    weak_r = min(
        frontier.baselines[ARM_LABELS[a]]["mean_reward"] for a in ARM_ORDER
    )
    quality_range = oracle_r - weak_r

    results: List[GapAtOracleResult] = []
    for sp in frontier.sweep_points:
        gap_pct = (
            (oracle_r - sp.mean_reward) / quality_range * 100
            if quality_range > 0 else 0.0
        )
        results.append(GapAtOracleResult(
            cost_penalty=sp.cost_penalty,
            router_reward=sp.mean_reward,
            oracle_reward=oracle_r,
            weak_reward=weak_r,
            gap_pct=round(gap_pct, 2),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


MARKER_STRONG = "D"
MARKER_WEAK = "o"

BASELINE_STYLES: Dict[str, Dict[str, Any]] = {
    "Supervised (LogReg)": {
        "color": CB_RED, "ls": "--", "lw": 1.8, "zorder": 4,
        "label": "Supervised (full labels)",
    },
    "Tabula rasa": {
        "color": CB_TEAL, "ls": "-.", "lw": 1.5, "zorder": 4,
        "label": "Tabula rasa LinUCB",
    },
    "ε-greedy": {
        "color": CB_ORANGE, "ls": ":", "lw": 1.6, "zorder": 3,
        "label": "ε-greedy (contextual)",
    },
    "Context-free UCB": {
        "color": CB_GREEN, "ls": ":", "lw": 1.5, "zorder": 3,
        "label": "Context-free UCB",
    },
    "LinTS": {
        "color": CB_PURPLE, "ls": "-.", "lw": 1.4, "zorder": 3,
        "label": "LinTS (Thompson Sampling)",
    },
}


def _dollar_formatter(x: float, _pos: Any) -> str:
    if x < 0.001:
        return f"${x:.1e}"
    return f"${x:.4f}"


def plot_pareto_panel_a(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    out_dir: Path,
    gap_results: Optional[List[GapAtOracleResult]] = None,
    baseline_frontiers: Optional[Dict[str, BaselineFrontier]] = None,
) -> Path:
    """Generate the K=2 Pareto frontier figure with all baselines.

    Args:
        frontier: BanditGPT sweep results.
        costsave_results: CostSave@Q metrics with CIs.
        out_dir: Directory for output files.
        gap_results: Gap@Oracle metrics per sweep point.
        baseline_frontiers: Additional baselines to overlay.

    Returns:
        Path to the saved figure.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

    weak_label = ARM_LABELS[ARM_ORDER[0]]
    strong_label = ARM_LABELS[ARM_ORDER[-1]]
    weak_c = frontier.baselines[weak_label]["mean_cost"]
    weak_r = frontier.baselines[weak_label]["mean_reward"]
    strong_c = frontier.baselines[strong_label]["mean_cost"]
    strong_r = frontier.baselines[strong_label]["mean_reward"]

    # ── Static baseline ───────────────────────────────────────────────
    ax.plot(
        [weak_c, strong_c], [weak_r, strong_r],
        "--", color=CB_GRAY, lw=2.0, zorder=3,
        label="Static random mix",
    )

    # ── Fixed-model endpoints ─────────────────────────────────────────
    ax.scatter(
        [weak_c], [weak_r],
        marker=MARKER_WEAK, c="white", edgecolors=CB_GRAY, s=80,
        linewidths=1.5, zorder=7,
    )
    ax.annotate(
        weak_label, (weak_c, weak_r),
        textcoords="offset points", xytext=(8, -2),
        fontsize=8, color=CB_GRAY, fontstyle="italic",
    )
    ax.scatter(
        [strong_c], [strong_r],
        marker=MARKER_STRONG, c="white", edgecolors=CB_GRAY, s=80,
        linewidths=1.5, zorder=7,
    )
    ax.annotate(
        strong_label, (strong_c, strong_r),
        textcoords="offset points", xytext=(0, 10),
        fontsize=8, color=CB_GRAY, fontstyle="italic", ha="center",
    )

    # ── Baseline frontiers ────────────────────────────────────────────
    if baseline_frontiers:
        for name, bf in baseline_frontiers.items():
            style = BASELINE_STYLES.get(name, {
                "color": "#888888", "ls": "--", "lw": 1.2,
                "zorder": 3, "label": name,
            })
            hc = np.array(bf.hull_costs)
            hr = np.array(bf.hull_rewards)
            ax.plot(
                hc, hr,
                ls=style["ls"], color=style["color"],
                lw=style["lw"], zorder=style["zorder"],
                label=style["label"],
            )

    # ── BanditGPT frontier (primary) ──────────────────────────────────
    hull_c = np.array(frontier.hull_costs)
    hull_r = np.array(frontier.hull_rewards)

    ax.plot(
        hull_c, hull_r, "-",
        color=CB_BLUE, lw=2.5, zorder=6,
        label="BanditGPT (ours)",
    )

    interior_mask = np.array([
        not any(
            abs(c - frontier.baselines[ARM_LABELS[a]]["mean_cost"]) < 1e-12
            and abs(r - frontier.baselines[ARM_LABELS[a]]["mean_reward"])
            < 1e-12
            for a in ARM_ORDER
        )
        for c, r in zip(hull_c, hull_r)
    ])
    if np.any(interior_mask):
        ax.scatter(
            hull_c[interior_mask], hull_r[interior_mask],
            marker="o", c=CB_BLUE, s=30, zorder=7,
            edgecolors="white", linewidths=0.5,
        )

    # ── ±1 SE band ────────────────────────────────────────────────────
    n_seeds = frontier.n_seeds
    if n_seeds > 1:
        band_c: List[float] = []
        band_upper: List[float] = []
        band_lower: List[float] = []
        for sp in sorted(frontier.sweep_points, key=lambda s: s.mean_cost):
            se = sp.std_reward / np.sqrt(n_seeds)
            band_c.append(sp.mean_cost)
            band_upper.append(sp.mean_reward + se)
            band_lower.append(sp.mean_reward - se)
        ax.fill_between(
            band_c, band_lower, band_upper,
            color=CB_BLUE, alpha=0.12, zorder=2,
            label=f"±1 SE ({n_seeds} seeds)",
        )

    # ── CostSave@Q annotations ───────────────────────────────────────
    q_colors = {0.90: CB_GREEN, 0.95: CB_ORANGE, 0.99: CB_RED}
    annotation_offsets = {
        0.90: (40, -18), 0.95: (-160, -12), 0.99: (-160, -14),
    }
    for cs in costsave_results:
        q = cs.threshold
        if cs.bandit_cost is None or cs.baseline_cost is None:
            continue
        target_r = cs.target_reward
        color = q_colors.get(q, CB_GRAY)
        ax.plot(
            [cs.baseline_cost, cs.bandit_cost], [target_r, target_r],
            "-", color=color, lw=1.5, alpha=0.7, zorder=4,
        )
        ax.scatter(
            [cs.bandit_cost], [target_r],
            marker="*", c=color, s=120, zorder=8,
            edgecolors="white", linewidths=0.3,
        )
        offset = annotation_offsets.get(q, (15, 0))
        ci_str = ""
        if cs.ci_lower is not None and cs.ci_upper is not None:
            ci_str = f"\n95% CI [{cs.ci_lower:+.1f}, {cs.ci_upper:+.1f}] pp"
        ax.annotate(
            f"CostSave@{q:.0%}: {cs.bandit_saving_pct:.1f}%\n"
            f"(Δ={cs.advantage_pp:+.1f} pp vs static"
            f" {cs.baseline_saving_pct:.1f}%){ci_str}",
            xy=(cs.bandit_cost, target_r),
            xytext=offset, textcoords="offset points",
            fontsize=6.5, color=color, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor=color, alpha=0.85, linewidth=0.5,
            ),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, ls=":"),
            zorder=9,
        )

    # ── Oracle reference line ─────────────────────────────────────────
    oracle_r = frontier.oracle_reward
    ax.axhline(oracle_r, color=CB_RED, lw=1.0, ls=":", alpha=0.5, zorder=2)
    ax.annotate(
        f"Oracle ({oracle_r:.3f})",
        xy=(strong_c * 0.40, oracle_r),
        xytext=(0, 6), textcoords="offset points",
        fontsize=7, color=CB_RED, fontstyle="italic", alpha=0.7,
    )

    # ── Pareto AUC annotation ─────────────────────────────────────────
    auc_parts = [
        f"BanditGPT AUC: {frontier.pareto_auc_mean:.4f}"
        f" (static: {frontier.static_auc:.4f},"
        f" Δ={((frontier.pareto_auc_mean - frontier.static_auc) / frontier.static_auc * 100):+.2f}%)",
    ]
    if baseline_frontiers and "Supervised (LogReg)" in baseline_frontiers:
        sup_auc = baseline_frontiers["Supervised (LogReg)"].pareto_auc
        auc_parts.append(f"Supervised AUC: {sup_auc:.4f}")
    ax.text(
        0.02, 0.02, "  |  ".join(auc_parts),
        transform=ax.transAxes, fontsize=6, color="#666666",
        verticalalignment="bottom",
    )

    # ── Axes and legend ───────────────────────────────────────────────
    ax.set_xlabel("Average Cost per Request ($)")
    ax.set_ylabel("Average Reward (Quality)")
    ax.set_title(
        "RQ1: Cost\u2013Quality Pareto Frontier (K=2)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_dollar_formatter))
    ax.grid(True, alpha=0.15, ls="--")
    ax.legend(loc="lower right", framealpha=0.92, ncol=1)

    y_lo = weak_r - 0.02
    y_hi = strong_r + 0.035
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-strong_c * 0.02, strong_c * 1.14)

    out_path = out_dir / "figure1_pareto_k2.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        out_path.with_suffix(".png"), dpi=300,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════


def export_results(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    elapsed_s: float,
    out_dir: Path,
    gap_results: Optional[List[GapAtOracleResult]] = None,
    tabula_rasa_frontier: Optional[FrontierResult] = None,
    baseline_frontiers: Optional[Dict[str, BaselineFrontier]] = None,
    tuned_baseline_hparams: Optional[TunedBaselineHparams] = None,
) -> Path:
    """Write machine-readable results for downstream consumption."""
    data: Dict[str, Any] = {
        "experiment": "RQ1 — Cost-Quality Pareto Frontier (K=2)",
        "hparams": frontier.hparams,
        "n_seeds": frontier.n_seeds,
        "oracle_reward": round(frontier.oracle_reward, 6),
        "pareto_auc": {
            "bandit_mean": round(frontier.pareto_auc_mean, 6),
            "bandit_std": round(frontier.pareto_auc_std, 6),
            "bandit_per_seed": [
                round(a, 6) for a in frontier.pareto_auc_per_seed
            ],
            "static": round(frontier.static_auc, 6),
            "delta_pct": round(
                (frontier.pareto_auc_mean - frontier.static_auc)
                / frontier.static_auc * 100, 3,
            ),
        },
        "costsave": [
            {
                "threshold": f"{cs.threshold:.0%}",
                "target_reward": cs.target_reward,
                "bandit_saving_pct": cs.bandit_saving_pct,
                "baseline_saving_pct": cs.baseline_saving_pct,
                "advantage_pp": cs.advantage_pp,
                "ci_95_lower_pp": cs.ci_lower,
                "ci_95_upper_pp": cs.ci_upper,
            }
            for cs in costsave_results
        ],
        "gap_at_oracle": [
            {
                "cost_penalty": g.cost_penalty,
                "router_reward": round(g.router_reward, 6),
                "gap_pct": g.gap_pct,
            }
            for g in (gap_results or [])
        ],
        "sweep_points": [
            {
                "cost_penalty": sp.cost_penalty,
                "mean_reward": round(sp.mean_reward, 6),
                "std_reward": round(sp.std_reward, 6),
                "mean_cost": round(sp.mean_cost, 8),
                "std_cost": round(sp.std_cost, 8),
                "pct_weak": round(sp.pct_weak * 100, 1),
                "pct_strong": round(sp.pct_strong * 100, 1),
            }
            for sp in frontier.sweep_points
        ],
        "pareto_hull": {
            "costs": [round(c, 8) for c in frontier.hull_costs],
            "rewards": [round(r, 6) for r in frontier.hull_rewards],
        },
        "baselines": frontier.baselines,
        "arm_order": ARM_ORDER,
        "elapsed_s": round(elapsed_s, 1),
    }
    if tabula_rasa_frontier is not None:
        data["tabula_rasa"] = {
            "pareto_auc_mean": round(
                tabula_rasa_frontier.pareto_auc_mean, 6,
            ),
            "pareto_auc_std": round(
                tabula_rasa_frontier.pareto_auc_std, 6,
            ),
        }
    if tuned_baseline_hparams is not None:
        data["tuned_baseline_hparams"] = {
            "ucb_alpha": tuned_baseline_hparams.ucb_alpha,
            "epsilon": tuned_baseline_hparams.epsilon,
            "lints_noise_variance": tuned_baseline_hparams.lints_noise_variance,
            "logreg_C": tuned_baseline_hparams.logreg_C,
        }
    if baseline_frontiers:
        data["baseline_pareto_aucs"] = {
            name: round(bf.pareto_auc, 6)
            for name, bf in baseline_frontiers.items()
        }

    out_path = out_dir / "figure1_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Console Summary
# ═══════════════════════════════════════════════════════════════════════════


def print_summary(
    frontier: FrontierResult,
    costsave_results: List[CostSaveResult],
    gap_results: Optional[List[GapAtOracleResult]] = None,
    tabula_rasa_frontier: Optional[FrontierResult] = None,
    baseline_frontiers: Optional[Dict[str, BaselineFrontier]] = None,
) -> None:
    """Print a concise, reviewer-friendly summary to stdout."""
    print("\n" + "=" * 72)
    print("RQ1 — COST-QUALITY PARETO FRONTIER (K=2)")
    print("=" * 72)

    print(f"\nConfig: {json.dumps(frontier.hparams)}")
    print(f"Seeds:  {frontier.n_seeds} (offset=1000)")
    print(f"Arms:   {', '.join(ARM_LABELS[a] for a in ARM_ORDER)}")

    print(f"\n{'Model':<20s}  {'Reward':>8s}  {'Cost':>12s}")
    print("-" * 44)
    for arm_id in ARM_ORDER:
        label = ARM_LABELS[arm_id]
        b = frontier.baselines[label]
        print(
            f"{label:<20s}  {b['mean_reward']:8.4f}"
            f"  ${b['mean_cost']:11.8f}"
        )
    print(f"{'Oracle (per-prompt)':<20s}  {frontier.oracle_reward:8.4f}")

    print(f"\nPareto AUC Comparison")
    print(f"  {'Router':<30s}  {'AUC':>10s}  {'vs Static':>10s}")
    print("  " + "-" * 54)
    _print_auc_row(
        "BanditGPT (ours)",
        frontier.pareto_auc_mean, frontier.static_auc,
    )
    if tabula_rasa_frontier is not None:
        _print_auc_row(
            "Tabula rasa LinUCB",
            tabula_rasa_frontier.pareto_auc_mean, frontier.static_auc,
        )
    if baseline_frontiers:
        for name, bf in baseline_frontiers.items():
            _print_auc_row(name, bf.pareto_auc, frontier.static_auc)
    _print_auc_row("Static random mix", frontier.static_auc, frontier.static_auc)

    if gap_results:
        print(
            f"\nGap@Oracle "
            f"(normalised: 0% = oracle, 100% = always-weak)"
        )
        print(f"  {'λ':<6s}  {'Reward':>8s}  {'Gap':>8s}")
        print("  " + "-" * 26)
        for g in gap_results:
            if g.cost_penalty in (0.0, 0.05, 0.1, 0.2, 0.5):
                print(
                    f"  {g.cost_penalty:<6.2f}  {g.router_reward:8.4f}"
                    f"  {g.gap_pct:7.1f}%"
                )

    n_prompts_str = ""
    if frontier.per_prompt_bandit_rewards is not None:
        n_pp = len(next(iter(frontier.per_prompt_bandit_rewards.values())))
        n_prompts_str = f" (prompt-level, n={n_pp})"
    print(f"\nCostSave{n_prompts_str}")
    print(
        f"{'Threshold':<12s}  {'Bandit':>8s}  {'Static':>8s}"
        f"  {'Advantage':>10s}  {'95% CI':>18s}"
    )
    print("-" * 62)
    for cs in costsave_results:
        b_str = (
            f"{cs.bandit_saving_pct:.1f}%"
            if cs.bandit_saving_pct is not None else "N/A"
        )
        bl_str = (
            f"{cs.baseline_saving_pct:.1f}%"
            if cs.baseline_saving_pct is not None else "N/A"
        )
        adv_str = (
            f"{cs.advantage_pp:+.1f} pp"
            if cs.advantage_pp is not None else "N/A"
        )
        ci_str = (
            f"[{cs.ci_lower:+.1f}, {cs.ci_upper:+.1f}] pp"
            if cs.ci_lower is not None and cs.ci_upper is not None
            else "N/A"
        )
        print(
            f"  @{cs.threshold:.0%}       {b_str:>8s}  {bl_str:>8s}"
            f"  {adv_str:>10s}  {ci_str:>18s}"
        )

    print("=" * 72)


def _print_auc_row(name: str, auc: float, static_auc: float) -> None:
    delta = (auc - static_auc) / static_auc * 100 if static_auc > 0 else 0.0
    print(f"  {name:<30s}  {auc:10.6f}  {delta:+9.3f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent random seeds (default: 5)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=2000,
        help="Bootstrap resamples for CostSave CIs (default: 2000)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Reduced sweep for quick testing",
    )
    parser.add_argument(
        "--no-baselines", action="store_true",
        help="Skip baseline routers (BanditGPT only)",
    )
    args = parser.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_baselines and LogisticRegression is None:
        parser.error(
            "scikit-learn is required for the supervised baseline. "
            "Install with: pip install scikit-learn  "
            "(or pass --no-baselines to skip)"
        )

    n_seeds = 3 if args.fast else args.n_seeds
    cost_penalties = (
        [0.0, 0.1, 0.5, 2.0, 10.0] if args.fast else COST_PENALTY_SWEEP
    )
    n_bootstrap = 200 if args.fast else args.n_bootstrap

    hparams = dict(BEST_K2_HPARAMS)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)

    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading data and encoding prompts ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    logger.info("  Feature dim: %d", feature_dim)

    train = load_split(TRAIN_DATA_PATH, fs)
    val = load_split(VAL_DATA_PATH, fs)
    test = load_split(HOLDOUT_DATA_PATH, fs)
    logger.info(
        "  Train: %d, Val: %d, Test: %d prompts",
        train.n, val.n, test.n,
    )

    registry = build_model_registry()
    logger.info("  Registry: %s", list(registry.keys()))

    # ── Tune baseline hyperparameters on val ──────────────────────────
    tuned_hparams: Optional[TunedBaselineHparams] = None
    if not args.no_baselines:
        logger.info("\nTuning baseline hyperparameters on val split ...")
        tuning_cp = (
            [0.0, 0.1, 0.5, 2.0, 10.0] if args.fast else cost_penalties
        )
        tuned_hparams = tune_baseline_hparams(
            train, val, feature_dim,
            registry=registry,
            cost_penalties=tuning_cp,
            n_seeds=min(n_seeds, 3),
            seed_offset=5000,
        )

    # ── BanditGPT frontier ────────────────────────────────────────────
    logger.info(
        "\nBanditGPT: %d cost_penalties × %d seeds = %d simulations",
        len(cost_penalties), n_seeds, len(cost_penalties) * n_seeds,
    )
    frontier = run_frontier_sweep(
        train, test, registry, feature_dim,
        hparams=hparams,
        warmup_path=warmup_path,
        cost_penalties=cost_penalties,
        n_seeds=n_seeds,
        seed_offset=1000,
    )

    # ── Tabula rasa LinUCB (uses BanditRouter, just no priors) ────────
    # Fairness: tabula rasa should use its own best-tuned alpha, not
    # BanditGPT's.  When BEST_K2_TABULA_RASA_HPARAMS is populated from
    # the sweep, use those.  Until then, we inherit alpha from the best
    # overall config.  Alpha=1.0 is the grid maximum (HPARAM_GRID), so
    # tabula rasa is not disadvantaged by using a suboptimally low alpha.
    from bandit_gpt.config import BEST_K2_TABULA_RASA_HPARAMS
    tabula_rasa_hparams = (
        BEST_K2_TABULA_RASA_HPARAMS if BEST_K2_TABULA_RASA_HPARAMS
        else hparams
    )
    logger.info(
        "\nTabula rasa LinUCB: %d cost_penalties × %d seeds",
        len(cost_penalties), n_seeds,
    )
    tabula_rasa_frontier = run_frontier_sweep(
        train, test, registry, feature_dim,
        hparams=tabula_rasa_hparams,
        warmup_path=warmup_path,
        cost_penalties=cost_penalties,
        n_seeds=n_seeds,
        seed_offset=1000,
        tabula_rasa=True,
    )

    # ── Other baselines ───────────────────────────────────────────────
    baseline_frontiers: Optional[Dict[str, BaselineFrontier]] = None
    if not args.no_baselines:
        baseline_cp = (
            [0.0, 0.1, 0.5, 2.0, 10.0] if args.fast else cost_penalties
        )
        logger.info(
            "\nRunning baselines: %d cost_penalties × %d seeds each",
            len(baseline_cp), n_seeds,
        )
        baseline_frontiers = run_all_baselines(
            train, test, feature_dim,
            registry=registry,
            cost_penalties=baseline_cp,
            n_seeds=n_seeds,
            seed_offset=1000,
            warmup_path=warmup_path,
            prior_n_effective=hparams.get("prior_n_effective", 10.0),
            tuned_hparams=tuned_hparams,
        )
        baseline_frontiers["Tabula rasa"] = BaselineFrontier(
            name="Tabula rasa",
            sweep_points=[
                {
                    "cost_penalty": sp.cost_penalty,
                    "mean_reward": sp.mean_reward,
                    "mean_cost": sp.mean_cost,
                }
                for sp in tabula_rasa_frontier.sweep_points
            ],
            hull_costs=tabula_rasa_frontier.hull_costs,
            hull_rewards=tabula_rasa_frontier.hull_rewards,
            pareto_auc=tabula_rasa_frontier.pareto_auc_mean,
            n_seeds=tabula_rasa_frontier.n_seeds,
        )

    # ── CostSave@Q with bootstrap CIs ────────────────────────────────
    logger.info(
        "\nCostSave@Q with %d bootstrap resamples ...", n_bootstrap,
    )
    costsave_results = compute_costsave_with_bootstrap(
        frontier, COSTSAVE_THRESHOLDS,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=42,
    )

    # ── Gap@Oracle ────────────────────────────────────────────────────
    logger.info("\nGap@Oracle ...")
    gap_results = compute_gap_at_oracle(frontier)

    # ── Generate outputs ──────────────────────────────────────────────
    elapsed = time.time() - t0

    fig_path = plot_pareto_panel_a(
        frontier, costsave_results, RESULTS_DIR,
        gap_results=gap_results,
        baseline_frontiers=baseline_frontiers,
    )
    logger.info("Figure saved to %s", fig_path)

    json_path = export_results(
        frontier, costsave_results, elapsed, RESULTS_DIR,
        gap_results=gap_results,
        tabula_rasa_frontier=tabula_rasa_frontier,
        baseline_frontiers=baseline_frontiers,
        tuned_baseline_hparams=tuned_hparams,
    )
    logger.info("Data saved to %s", json_path)

    print_summary(
        frontier, costsave_results,
        gap_results=gap_results,
        tabula_rasa_frontier=tabula_rasa_frontier,
        baseline_frontiers=baseline_frontiers,
    )

    logger.info("\nTotal wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()
