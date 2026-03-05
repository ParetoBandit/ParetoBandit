#!/usr/bin/env python3
"""
Figure 5: Corralling Enables Recovery from Catastrophic LLM Failure.

Demonstrates that the Corralling meta-learner provides automatic failover
when a production LLM degrades catastrophically.  Uses the K=3 portfolio
(Llama-3.1-8B, Gemini-2.5-Flash, GPT-4.1) where the frontier model
(GPT-4.1) failing forces the router to adapt by shifting traffic to
Gemini-2.5-Flash or Llama-3.1-8B.  Hyperparameters from Appendix H,
portfolio-specific warmup priors.

**All prompts, embeddings, and rewards are real** — drawn from the
canonical evaluation datasets.  The only synthetic element is GPT-4.1's
reward during the failure phase, which is replaced with low noise to
simulate an API outage or model regression.

Data Separation
---------------
- **Dev-train** (80% of canonical dev pool): the online routing stream.
  The router receives these prompts sequentially, routes each one, and
  gets reward feedback.
- **Dev-val** (20% of canonical dev pool): unused in this experiment
  (hyperparameters come from Appendix H).
- **Holdout**: reserved for periodic frozen evaluation.  At checkpoints,
  the router is frozen (greedy exploitation) and evaluated on the full
  holdout set to produce clean quality measurements.

Scenario
--------
The dev-train stream is cycled three times (one pass per phase):

  Phase 1 — Healthy  (pass 1):  Real rewards for all models.
  Phase 2 — Failure  (pass 2):  GPT-4.1 reward → near-zero.
  Phase 3 — Recovery (pass 3):  Real rewards restored.

Outputs (``results/``)
    catastrophic_failure_results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "03_figure"))

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K3_WARMUP_PRIORS_PATH,
    K3_MODELS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)
from utils.router_factory import create_experiment_router
from utils.model_pricing import load_model_catalog
from utils.embeddings import load_embedding_cache

from run_prequential import (
    load_rewards_from_file,
    embed_dataset,
    _split_dev_train_val,
    DEV_VAL_FRACTION,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# K=3 model catalog (Llama-3.1-8B, Gemini-2.5-Flash, GPT-4.1)
# ============================================================================

K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)

# ============================================================================
# Hyperparameters from Appendix H
# ============================================================================


def _load_hparams(path: Path, key: str) -> Dict[str, float]:
    """Load dev-val-selected hyperparameters from Appendix H."""
    if not path.exists():
        raise FileNotFoundError(
            f"Appendix H hparams not found: {path}\n"
            "Run experiments/appendix/H_alpha_neff_ablation/"
            "run_3d_grid_ablation.py first."
        )
    data = json.loads(path.read_text())
    cfg = data[key]
    return {
        "alpha": float(cfg["alpha"]),
        "prior_n_effective": float(cfg["n_eff"]),
        "forgetting_factor": float(cfg["gamma"]),
    }


HPARAMS_DIR = (
    PROJECT_ROOT / "experiments" / "appendix"
    / "H_alpha_neff_ablation" / "results"
)

# ============================================================================
# Experiment constants
# ============================================================================

N_SEEDS: int = 20
SEED_OFFSET: int = 42

FAILING_MODEL: str = "openai/gpt-4.1"
FAILURE_REWARD_MEAN: float = 0.10
FAILURE_REWARD_STD: float = 0.05

CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

# Non-stationary forgetting factor for the tabula rasa expert.
# Appendix H tunes gamma for stationary rewards (gamma ≈ 1.0), which is
# optimal when reward distributions are fixed.  In a deployment where LLM
# reliability can change, the *tabula rasa* expert should use a shorter
# memory so it adapts quickly, while the *warmup* expert retains its
# Appendix H gamma to serve as stable "institutional knowledge."
# Corralling automatically shifts weight to whichever expert is performing
# better, giving the user both stability (warmup) and adaptability (TR)
# without manual intervention.
# gamma = 0.999 → effective memory ≈ 1/(1−0.999) = 1000 steps.
TR_NS_FORGETTING_FACTOR: float = 0.999

EVAL_INTERVAL: int = 50


def _short_name(model_id: str) -> str:
    """Human-readable short name for a model."""
    return K3_CATALOG[model_id]["display"]


# ============================================================================
# Frozen holdout evaluation
# ============================================================================


def evaluate_frozen_holdout(
    router: Any,
    holdout_data: List[Dict[str, Any]],
    holdout_emb: List[np.ndarray],
    models: List[str],
    degrade_model: Optional[str] = None,
    failure_rng: Optional[np.random.RandomState] = None,
) -> float:
    """Evaluate the router on holdout without updating its state.

    Args:
        router: Trained router with ``.route()`` method.
        holdout_data: Holdout prompt records with ``"rewards"`` dicts.
        holdout_emb: Pre-computed holdout embeddings.
        models: Model IDs.
        degrade_model: If set, this model's reward is replaced with
            failure noise (simulating an ongoing outage during eval).
        failure_rng: RNG for failure noise generation.

    Returns:
        Mean reward over the holdout set.
    """
    rewards: List[float] = []
    for i, prompt in enumerate(holdout_data):
        model, _log = router.route(holdout_emb[i])
        r = prompt["rewards"][model]
        if degrade_model and model == degrade_model and failure_rng is not None:
            r = float(np.clip(
                failure_rng.normal(FAILURE_REWARD_MEAN, FAILURE_REWARD_STD),
                0.0, 1.0,
            ))
        rewards.append(r)
    return float(np.mean(rewards))


# ============================================================================
# Baselines
# ============================================================================


class StaticRouter:
    """Always routes to a fixed model."""

    def __init__(self, model: str):
        self.model = model

    def select(self, context: np.ndarray) -> str:
        return self.model


class EMATracker:
    """Epsilon-greedy router with exponential moving average reward tracking."""

    def __init__(
        self,
        models: List[str],
        alpha: float = 0.15,
        epsilon: float = 0.1,
        seed: int = 42,
    ):
        self.models = models
        self.alpha = alpha
        self.epsilon = epsilon
        self.ema: Dict[str, float] = {m: 0.5 for m in models}
        self.rng = np.random.RandomState(seed)

    def select(self, context: np.ndarray) -> str:
        if self.rng.random() < self.epsilon:
            return self.models[self.rng.randint(len(self.models))]
        return max(self.ema, key=self.ema.get)

    def update(self, model: str, reward: float) -> None:
        self.ema[model] = (1 - self.alpha) * self.ema[model] + self.alpha * reward


# ============================================================================
# Trial result container
# ============================================================================


@dataclass
class TrialResult:
    """Per-seed results for all methods."""

    seed: int
    n_steps: int = 0
    phase_boundaries: Tuple[int, int] = (0, 0)

    # Online rewards (from dev-train stream)
    online_banditgpt: List[float] = field(default_factory=list)
    online_warmup_only: List[float] = field(default_factory=list)
    online_tabula_rasa: List[float] = field(default_factory=list)
    online_static: List[float] = field(default_factory=list)
    online_ema: List[float] = field(default_factory=list)
    online_oracle: List[float] = field(default_factory=list)

    # Frozen holdout evaluations at checkpoints
    holdout_steps: List[int] = field(default_factory=list)
    holdout_banditgpt: List[float] = field(default_factory=list)
    holdout_warmup_only: List[float] = field(default_factory=list)
    holdout_tabula_rasa: List[float] = field(default_factory=list)

    expert_weights: List[np.ndarray] = field(default_factory=list)
    model_chosen_banditgpt: List[str] = field(default_factory=list)
    model_chosen_ema: List[str] = field(default_factory=list)

    failure_detection_step: Optional[int] = None
    recovery_detection_step: Optional[int] = None


# ============================================================================
# Router construction helpers
# ============================================================================


def _build_model_registry(models: List[str]) -> Dict[str, Dict]:
    """Build model registry from K3_CATALOG."""
    return {
        m: {
            "input_cost_per_m": K3_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": K3_CATALOG[m]["output_cost_per_m"],
        }
        for m in models
    }


# ============================================================================
# Single trial
# ============================================================================


def run_single_trial(
    seed: int,
    models: List[str],
    train_data: List[Dict[str, Any]],
    train_emb: List[np.ndarray],
    holdout_data: List[Dict[str, Any]],
    holdout_emb: List[np.ndarray],
    feature_dim: int,
    tuned_warmup: Dict[str, float],
    tuned_tr: Dict[str, float],
) -> TrialResult:
    """Run one seed with all methods on the same dev-train prompt stream.

    The dev-train set is cycled three times (healthy → failure → recovery).
    At periodic checkpoints, each bandit-based router is frozen and
    evaluated on the holdout set.  Baselines (static, EMA) are evaluated
    on the online stream only since they have no ``route()`` API.
    """
    rng = np.random.RandomState(seed)
    failure_rng = np.random.RandomState(seed + 10_000)
    holdout_eval_rng = np.random.RandomState(seed + 20_000)
    model_registry = _build_model_registry(models)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    n_train = len(train_data)
    phase_size = n_train
    n_steps = 3 * phase_size
    phase_boundaries = (phase_size, 2 * phase_size)

    # Build prompt order: one shuffled pass per phase
    idx_healthy = rng.permutation(n_train)
    idx_failure = rng.permutation(n_train)
    idx_recovery = rng.permutation(n_train)
    order = np.concatenate([idx_healthy, idx_failure, idx_recovery])

    result = TrialResult(
        seed=seed,
        n_steps=n_steps,
        phase_boundaries=phase_boundaries,
    )

    # Reward normalization bounds (from dev-train)
    all_r = [p["rewards"][m] for p in train_data for m in models]
    r_min = min(all_r)
    r_max = max(all_r)
    r_range = r_max - r_min

    # -- BanditGPT (Corralling + warmup) --
    np.random.seed(seed)
    router_bg = create_experiment_router(
        model_registry=model_registry,
        feature_dim=feature_dim,
        prior_n_effective=tuned_warmup["prior_n_effective"],
        alpha=tuned_warmup["alpha"],
        warmup_path=warmup_path,
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=0.0,
        forgetting_factor=tuned_warmup["forgetting_factor"],
        tabula_rasa_alpha=tuned_tr["alpha"],
        tabula_rasa_forgetting_factor=tuned_tr["forgetting_factor"],
    )

    # -- Warmup-only (no Corralling) --
    np.random.seed(seed)
    router_wo = create_experiment_router(
        model_registry=model_registry,
        feature_dim=feature_dim,
        prior_n_effective=tuned_warmup["prior_n_effective"],
        alpha=tuned_warmup["alpha"],
        warmup_path=warmup_path,
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=tuned_warmup["forgetting_factor"],
    )

    # -- Tabula rasa (no priors, no Corralling) --
    np.random.seed(seed)
    router_tr = create_experiment_router(
        model_registry=model_registry,
        feature_dim=feature_dim,
        prior_n_effective=tuned_tr["prior_n_effective"],
        alpha=tuned_tr["alpha"],
        warmup_path=None,
        use_corralling=False,
        cost_penalty=0.0,
        forgetting_factor=tuned_tr["forgetting_factor"],
    )

    static_router = StaticRouter(FAILING_MODEL)
    ema_router = EMATracker(models, alpha=0.15, epsilon=0.1, seed=seed)

    checkpoints = set(range(0, n_steps + 1, EVAL_INTERVAL))
    checkpoints.add(n_steps)

    def _get_phase(step: int) -> str:
        if step < phase_boundaries[0]:
            return "healthy"
        elif step < phase_boundaries[1]:
            return "failure"
        return "recovery"

    def _maybe_degrade(rewards: Dict[str, float], phase: str) -> Dict[str, float]:
        """Replace failing model's reward during Phase 2."""
        if phase != "failure":
            return rewards
        out = dict(rewards)
        out[FAILING_MODEL] = float(np.clip(
            failure_rng.normal(FAILURE_REWARD_MEAN, FAILURE_REWARD_STD),
            0.0, 1.0,
        ))
        return out

    # Initial holdout evaluation (step 0)
    if 0 in checkpoints:
        result.holdout_steps.append(0)
        result.holdout_banditgpt.append(
            evaluate_frozen_holdout(
                router_bg, holdout_data, holdout_emb, models,
            )
        )
        result.holdout_warmup_only.append(
            evaluate_frozen_holdout(
                router_wo, holdout_data, holdout_emb, models,
            )
        )
        result.holdout_tabula_rasa.append(
            evaluate_frozen_holdout(
                router_tr, holdout_data, holdout_emb, models,
            )
        )

    for t in range(n_steps):
        idx = order[t]
        prompt = train_data[idx]
        emb = train_emb[idx]
        phase = _get_phase(t)
        rewards = _maybe_degrade(prompt["rewards"], phase)

        # Oracle (per-step best from available rewards)
        oracle_r = max(rewards[m] for m in models)
        result.online_oracle.append(oracle_r)

        # BanditGPT
        model_bg, log_bg = router_bg.route(emb, total_steps=n_steps)
        reward_bg = rewards[model_bg]
        norm_bg = (reward_bg - r_min) / r_range if r_range > 1e-6 else 0.5
        router_bg.process_feedback(log_bg.request_id, norm_bg)
        result.online_banditgpt.append(reward_bg)
        result.model_chosen_banditgpt.append(model_bg)
        if hasattr(router_bg, "corralling_router"):
            result.expert_weights.append(
                router_bg.corralling_router.weights.copy()
            )

        # Warmup-only
        model_wo, log_wo = router_wo.route(emb, total_steps=n_steps)
        reward_wo = rewards[model_wo]
        norm_wo = (reward_wo - r_min) / r_range if r_range > 1e-6 else 0.5
        router_wo.process_feedback(log_wo.request_id, norm_wo)
        result.online_warmup_only.append(reward_wo)

        # Tabula rasa
        model_tr, log_tr = router_tr.route(emb, total_steps=n_steps)
        reward_tr = rewards[model_tr]
        norm_tr = (reward_tr - r_min) / r_range if r_range > 1e-6 else 0.5
        router_tr.process_feedback(log_tr.request_id, norm_tr)
        result.online_tabula_rasa.append(reward_tr)

        # Static
        model_st = static_router.select(emb)
        result.online_static.append(rewards[model_st])

        # EMA
        model_em = ema_router.select(emb)
        reward_em = rewards[model_em]
        ema_router.update(model_em, reward_em)
        result.online_ema.append(reward_em)
        result.model_chosen_ema.append(model_em)

        # Periodic frozen holdout evaluation
        step = t + 1
        if step in checkpoints:
            degrade = FAILING_MODEL if phase == "failure" else None
            result.holdout_steps.append(step)
            result.holdout_banditgpt.append(
                evaluate_frozen_holdout(
                    router_bg, holdout_data, holdout_emb, models,
                    degrade_model=degrade, failure_rng=holdout_eval_rng,
                )
            )
            result.holdout_warmup_only.append(
                evaluate_frozen_holdout(
                    router_wo, holdout_data, holdout_emb, models,
                    degrade_model=degrade, failure_rng=holdout_eval_rng,
                )
            )
            result.holdout_tabula_rasa.append(
                evaluate_frozen_holdout(
                    router_tr, holdout_data, holdout_emb, models,
                    degrade_model=degrade, failure_rng=holdout_eval_rng,
                )
            )

    # Failure detection: sustained drop in routing to the failing model.
    # Use a sliding window over model selections — when the fraction of
    # traffic to FAILING_MODEL drops below 50% of its healthy-phase
    # average for DETECTION_WINDOW consecutive steps, declare detection.
    DETECTION_WINDOW = 50
    choices = result.model_chosen_banditgpt
    healthy_frac = (
        sum(1 for c in choices[:phase_boundaries[0]] if c == FAILING_MODEL)
        / max(phase_boundaries[0], 1)
    )
    detection_threshold = healthy_frac * 0.50

    for t_idx in range(phase_boundaries[0], min(phase_boundaries[1], n_steps) - DETECTION_WINDOW):
        window = choices[t_idx : t_idx + DETECTION_WINDOW]
        frac = sum(1 for c in window if c == FAILING_MODEL) / DETECTION_WINDOW
        if frac < detection_threshold:
            result.failure_detection_step = t_idx
            break

    # Recovery detection: routing to FAILING_MODEL returns above 80%
    # of its healthy-phase average.
    recovery_threshold = healthy_frac * 0.80
    for t_idx in range(phase_boundaries[1], n_steps - DETECTION_WINDOW):
        window = choices[t_idx : t_idx + DETECTION_WINDOW]
        frac = sum(1 for c in window if c == FAILING_MODEL) / DETECTION_WINDOW
        if frac >= recovery_threshold:
            result.recovery_detection_step = t_idx
            break

    return result


# ============================================================================
# Statistics
# ============================================================================


def compute_statistics(
    results: List[TrialResult],
    models: List[str],
) -> Dict[str, Any]:
    """Aggregate statistics across seeds."""
    pb = results[0].phase_boundaries
    n_steps = results[0].n_steps

    detection_steps = [
        r.failure_detection_step for r in results
        if r.failure_detection_step is not None
    ]
    reaction_times = [d - pb[0] for d in detection_steps]
    recovery_steps = [
        r.recovery_detection_step for r in results
        if r.recovery_detection_step is not None
    ]

    stats: Dict[str, Any] = {
        "n_seeds": len(results),
        "K": len(models),
        "n_steps": n_steps,
        "n_train": pb[0],
        "phase_boundaries": list(pb),
        "failing_model": FAILING_MODEL,
        "detection_rate": len(detection_steps) / len(results),
        "recovery_rate": len(recovery_steps) / len(results),
    }

    if reaction_times:
        stats["reaction_mean"] = float(np.mean(reaction_times))
        stats["reaction_std"] = float(np.std(reaction_times))
        stats["reaction_median"] = float(np.median(reaction_times))
    else:
        stats["reaction_mean"] = None
        stats["reaction_std"] = None
        stats["reaction_median"] = None

    phase_slices = {
        "healthy": (0, pb[0]),
        "failure": (pb[0], pb[1]),
        "recovery": (pb[1], n_steps),
    }

    # Online reward statistics (from dev-train stream)
    for method, attr in [
        ("banditgpt", "online_banditgpt"),
        ("warmup_only", "online_warmup_only"),
        ("tabula_rasa", "online_tabula_rasa"),
        ("static", "online_static"),
        ("ema", "online_ema"),
        ("oracle", "online_oracle"),
    ]:
        for phase, (start, end) in phase_slices.items():
            vals = [np.mean(getattr(r, attr)[start:end]) for r in results]
            stats[f"{method}_{phase}_mean"] = float(np.mean(vals))
            stats[f"{method}_{phase}_std"] = float(np.std(vals))

    # Model selection fractions during failure phase
    for method, attr in [
        ("banditgpt", "model_chosen_banditgpt"),
        ("ema", "model_chosen_ema"),
    ]:
        for model in models:
            fracs = []
            for r in results:
                choices = getattr(r, attr)[pb[0]:pb[1]]
                fracs.append(
                    sum(1 for c in choices if c == model) / max(len(choices), 1)
                )
            stats[f"{method}_failure_{_short_name(model)}_frac"] = float(
                np.mean(fracs)
            )

    return stats


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:
    """Run the catastrophic failure experiment for K=3 with real prompts."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("FIGURE 5: CORRALLING RECOVERY FROM CATASTROPHIC FAILURE")
    logger.info(f"  K=3 portfolio, {N_SEEDS} seeds, real prompts")
    logger.info(f"  Failing model: {_short_name(FAILING_MODEL)}")
    logger.info("=" * 70)

    # Load hyperparameters from Appendix H
    tuned_warmup = _load_hparams(HPARAMS_DIR / "best_hparams_k3.json", "K3")
    tuned_tr = _load_hparams(
        HPARAMS_DIR / "best_hparams_k3_tabula_rasa.json", "K3",
    )
    # Heterogeneous forgetting strategy:
    #   Warmup expert  → keeps Appendix H gamma (stable, long memory)
    #   Tabula rasa    → uses TR_NS_FORGETTING_FACTOR (adaptive, short memory)
    # Corralling shifts weight to the expert that matches the current regime.
    tuned_tr["forgetting_factor"] = TR_NS_FORGETTING_FACTOR

    logger.info(
        f"  Warmup hparams: alpha={tuned_warmup['alpha']} "
        f"n_eff={tuned_warmup['prior_n_effective']} "
        f"gamma={tuned_warmup['forgetting_factor']} (Appendix H)"
    )
    logger.info(
        f"  TR hparams: alpha={tuned_tr['alpha']} "
        f"n_eff={tuned_tr['prior_n_effective']} "
        f"gamma={tuned_tr['forgetting_factor']} "
        f"(non-stationary override)"
    )

    # Validate warmup priors
    if not K3_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=3 warmup priors not found: {K3_WARMUP_PRIORS_PATH}"
        )

    # ------------------------------------------------------------------
    # Load real data with proper separation
    # ------------------------------------------------------------------
    logger.info("\n  Loading data and embeddings ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    feature_dim = pca.n_components_ + 1

    import run_prequential as _rp
    _rp._EMBEDDING_CACHE = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    dev_data = load_rewards_from_file(CANONICAL_DEV_DATA_PATH, K3_MODELS)
    holdout_data = load_rewards_from_file(
        CANONICAL_HOLDOUT_DATA_PATH, K3_MODELS,
    )

    logger.info(f"    Dev: {len(dev_data)} prompts")
    logger.info(f"    Holdout: {len(holdout_data)} prompts")

    # Embed
    dev_emb = embed_dataset(dev_data, encoder, pca)
    holdout_emb = embed_dataset(holdout_data, encoder, pca)

    # Split dev into train/val (same deterministic split as Figures 3/4)
    train_data, train_emb, val_data, val_emb = _split_dev_train_val(
        dev_data, dev_emb,
    )
    logger.info(
        f"    Dev-train: {len(train_data)} prompts "
        f"(online routing stream)"
    )
    logger.info(
        f"    Dev-val: {len(val_data)} prompts "
        f"(reserved, not used here)"
    )
    logger.info(
        f"    Holdout: {len(holdout_data)} prompts "
        f"(frozen evaluation only)"
    )
    logger.info(f"    Feature dim: {feature_dim}")

    n_train = len(train_data)
    logger.info(
        f"\n    Phases: healthy [0,{n_train}), "
        f"failure [{n_train},{2 * n_train}), "
        f"recovery [{2 * n_train},{3 * n_train})"
    )
    logger.info(f"    Total online steps: {3 * n_train}")
    logger.info(
        f"    Holdout eval every {EVAL_INTERVAL} steps "
        f"({3 * n_train // EVAL_INTERVAL + 1} checkpoints)"
    )

    # Per-model dev-train reward stats
    logger.info("\n    Per-model dev-train mean rewards:")
    for m in K3_MODELS:
        r = np.mean([p["rewards"][m] for p in train_data])
        logger.info(f"      {_short_name(m):<22} R={r:.4f}")

    # ------------------------------------------------------------------
    # Run trials
    # ------------------------------------------------------------------
    logger.info(f"\n  Running {N_SEEDS}-seed experiment ...")
    all_results: List[TrialResult] = []
    for seed_idx in range(N_SEEDS):
        seed = SEED_OFFSET + seed_idx
        result = run_single_trial(
            seed, K3_MODELS,
            train_data, train_emb,
            holdout_data, holdout_emb,
            feature_dim,
            tuned_warmup, tuned_tr,
        )
        all_results.append(result)
        if (seed_idx + 1) % 5 == 0:
            det = result.failure_detection_step
            pb = result.phase_boundaries
            det_str = (
                f"t={det} (Δ={det - pb[0]})"
                if det else "not detected"
            )
            logger.info(f"    Seed {seed_idx + 1}/{N_SEEDS}: detection={det_str}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    stats = compute_statistics(all_results, K3_MODELS)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY (online dev-train rewards)")
    logger.info("=" * 70)
    logger.info(
        f"  Detection rate: {stats['detection_rate']:.0%} "
        f"({int(stats['detection_rate'] * N_SEEDS)}/{N_SEEDS})"
    )
    if stats["reaction_mean"] is not None:
        logger.info(
            f"  Reaction time: {stats['reaction_mean']:.1f} "
            f"+/- {stats['reaction_std']:.1f} steps "
            f"(median={stats['reaction_median']:.0f})"
        )
    logger.info(f"  Recovery rate: {stats['recovery_rate']:.0%}")

    header = f"  {'Method':<28} {'Healthy':>10} {'Failure':>10} {'Recovery':>10}"
    logger.info(f"\n{header}")
    logger.info("  " + "-" * 60)
    for method, label in [
        ("oracle", "Oracle"),
        ("banditgpt", "BanditGPT (Corralling)"),
        ("warmup_only", "Warmup-only (no Corr.)"),
        ("tabula_rasa", "Tabula rasa"),
        ("ema", "EMA Tracker"),
        ("static", f"Static ({_short_name(FAILING_MODEL)})"),
    ]:
        h = f"{stats[f'{method}_healthy_mean']:.3f}"
        f_ = f"{stats[f'{method}_failure_mean']:.3f}"
        r = f"{stats[f'{method}_recovery_mean']:.3f}"
        logger.info(f"  {label:<28} {h:>10} {f_:>10} {r:>10}")

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------
    stats["data_separation"] = {
        "online_stream": "dev-train (80% of canonical dev pool)",
        "frozen_evaluation": "holdout (canonical holdout set)",
        "dev_val": "unused (hparams from Appendix H)",
        "n_dev_train": len(train_data),
        "n_dev_val": len(val_data),
        "n_holdout": len(holdout_data),
    }
    stats["base_rewards"] = {
        m: float(np.mean([p["rewards"][m] for p in train_data]))
        for m in K3_MODELS
    }
    stats["warmup_hparams"] = tuned_warmup
    stats["tabula_rasa_hparams"] = tuned_tr
    stats["models"] = K3_MODELS
    stats["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Per-seed online time series for plotting
    n_steps = all_results[0].n_steps
    stats["time_series"] = {}
    for method, attr in [
        ("banditgpt", "online_banditgpt"),
        ("warmup_only", "online_warmup_only"),
        ("tabula_rasa", "online_tabula_rasa"),
        ("static", "online_static"),
        ("ema", "online_ema"),
        ("oracle", "online_oracle"),
    ]:
        mat = np.array([getattr(r, attr) for r in all_results])
        stats["time_series"][method] = {
            "mean": mat.mean(axis=0).tolist(),
            "std": mat.std(axis=0).tolist(),
        }

    # Frozen holdout evaluation curves (bandit methods only)
    stats["holdout_eval"] = {
        "steps": all_results[0].holdout_steps,
    }
    for method, attr in [
        ("banditgpt", "holdout_banditgpt"),
        ("warmup_only", "holdout_warmup_only"),
        ("tabula_rasa", "holdout_tabula_rasa"),
    ]:
        mat = np.array([getattr(r, attr) for r in all_results])
        stats["holdout_eval"][method] = {
            "mean": mat.mean(axis=0).tolist(),
            "std": mat.std(axis=0).tolist(),
        }

    # Expert weights time series
    if all_results[0].expert_weights:
        all_w = np.array([r.expert_weights for r in all_results])
        stats["expert_weights"] = {
            "warmup_mean": all_w[:, :, 0].mean(axis=0).tolist(),
            "warmup_std": all_w[:, :, 0].std(axis=0).tolist(),
            "tabula_rasa_mean": all_w[:, :, 1].mean(axis=0).tolist(),
            "tabula_rasa_std": all_w[:, :, 1].std(axis=0).tolist(),
        }

    # Model selection fractions (BanditGPT) per step
    sel_mat: Dict[str, np.ndarray] = {m: np.zeros(n_steps) for m in K3_MODELS}
    for r in all_results:
        for t_step, m in enumerate(r.model_chosen_banditgpt):
            sel_mat[m][t_step] += 1
    stats["model_selection_banditgpt"] = {
        _short_name(m): (sel_mat[m] / N_SEEDS).tolist() for m in K3_MODELS
    }

    out_path = output_dir / "catastrophic_failure_results.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_experiment()
