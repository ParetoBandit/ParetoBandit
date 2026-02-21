"""
Figure 6 (5-Model): Catastrophic Failure Detection — Large Portfolio
====================================================================

Tests whether Corralling's contextual advantage over EMA continues to
grow with K=5 models, where post-failure routing across 4 remaining
models is a genuinely contextual problem.

Setup:
  - 5 models with different context-dependent strengths
  - GPT-4-Turbo fails catastrophically in Phase 2
  - After failure, 4 healthy models remain, each excelling in different
    context regions → optimal routing is highly contextual
  - EMA tracks 5 averages and explores ε/K = 0.1/5 = 2% per model
  - LinUCB learns context→model mapping and explores via UCB

Models:
  1. Mixtral-8x7B    (base 0.73) — direct priors from RouteLLM battles
  2. GPT-3.5-Turbo   (base 0.70) — semantic transfer from GPT-4-Turbo
  3. Claude-3-Haiku  (base 0.76) — semantic transfer from Mixtral
  4. GPT-4-Turbo     (base 0.82) — direct priors from RouteLLM battles, FAILS
  5. GPT-4o          (base 0.80) — semantic transfer from GPT-4-Turbo

Warmup priors: All 5 models get informed priors via the production system's
  semantic transfer mechanism (Section 3.3).  Models with direct RouteLLM
  priors (Mixtral, GPT-4-Turbo) are scaled; the rest inherit θ from their
  nearest known neighbor with A reset to n_eff·I (First-Child Bias Correction).
"""

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import (
    CorrallingRouter,
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
)
from bandit_gpt.config_legacy import DEFAULT_WARMUP_PRIORS_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS = [
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
]
MODEL_COSTS = {m: {"normalized_cost": 0.0} for m in MODELS}
N_STEPS = 500
N_SEEDS = 20
PHASE_BOUNDARIES = (100, 300)
CONTEXT_DIM = 33
CONTEXT_NORM = 1.15  # Match production PCA embedding norm (32 PCA components + 1 bias)
LEARNING_RATE = 0.3
GAMMA = 0.05
PRIOR_SCALE = 10.0

MODEL_SHORT = {
    "mistralai/mixtral-8x7b-instruct": "Mixtral",
    "openai/gpt-3.5-turbo": "GPT-3.5",
    "anthropic/claude-3-haiku": "Haiku",
    "openai/gpt-4-turbo": "GPT-4-Turbo",
    "openai/gpt-4o": "GPT-4o",
}

FAILING_MODEL = "openai/gpt-4-turbo"


# ============================================================================
# ENVIRONMENT
# ============================================================================

class FiveModelEnvironment:
    """
    Context-dependent reward environment with 5 models and three phases.

    Each non-failing model excels in a different region of context space,
    defined by orthogonal context directions.  This means post-failure
    routing is a 4-way contextual decision — EMA's per-model averages
    can't distinguish which of the 4 remaining models is best for a
    given prompt, while LinUCB can learn the context→model mapping.

    Base rewards are chosen so GPT-4-Turbo is the clear pre-failure
    favourite on average (0.82), making Static a strong Phase-1 baseline
    and the failure maximally disruptive.
    """

    def __init__(self, seed: int = 42, context_dim: int = CONTEXT_DIM):
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        self.t = 0

        # Create 4 orthogonal-ish context directions for non-failing models.
        # We use random directions and Gram-Schmidt to make them orthogonal,
        # ensuring each model excels in a distinct context region.
        raw_dirs = self.rng.randn(4, context_dim)
        # Simple Gram-Schmidt
        ortho_dirs = np.zeros_like(raw_dirs)
        for i in range(4):
            v = raw_dirs[i].copy()
            for j in range(i):
                v -= np.dot(v, ortho_dirs[j]) * ortho_dirs[j]
            ortho_dirs[i] = v / (np.linalg.norm(v) + 1e-8)

        # Scale: each direction contributes up to ±0.06 reward via tanh
        scale = 0.06
        non_failing = [m for m in MODELS if m != FAILING_MODEL]
        self.context_weights = {}
        for i, model in enumerate(non_failing):
            self.context_weights[model] = ortho_dirs[i] * scale

        # Failing model has weak random context sensitivity
        self.context_weights[FAILING_MODEL] = self.rng.randn(context_dim) * 0.02

        self.phase_base = {
            "healthy": {
                MODELS[0]: 0.73,  # Mixtral
                MODELS[1]: 0.70,  # GPT-3.5
                MODELS[2]: 0.76,  # Haiku
                MODELS[3]: 0.82,  # GPT-4-Turbo (best pre-failure)
                MODELS[4]: 0.80,  # GPT-4o
            },
            "failure": {
                MODELS[0]: 0.73,
                MODELS[1]: 0.70,
                MODELS[2]: 0.76,
                MODELS[3]: 0.15,  # GPT-4-Turbo CRASHES
                MODELS[4]: 0.80,
            },
            "recovery": {
                MODELS[0]: 0.73,
                MODELS[1]: 0.70,
                MODELS[2]: 0.76,
                MODELS[3]: 0.82,
                MODELS[4]: 0.80,
            },
        }

    def _get_phase(self) -> str:
        if self.t < PHASE_BOUNDARIES[0]:
            return "healthy"
        elif self.t < PHASE_BOUNDARIES[1]:
            return "failure"
        return "recovery"

    def get_reward(self, model: str, context: np.ndarray) -> float:
        self.t += 1
        phase = self._get_phase()
        base = self.phase_base[phase][model]
        ctx_norm = context / (np.linalg.norm(context) + 1e-8)
        ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)
        noise = self.rng.normal(0, 0.08)
        return float(np.clip(base + ctx_bonus + noise, 0.0, 1.0))

    def get_oracle_reward(self, context: np.ndarray) -> float:
        phase = self._get_phase()
        rewards = []
        for model in MODELS:
            base = self.phase_base[phase][model]
            ctx_norm = context / (np.linalg.norm(context) + 1e-8)
            ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)
            rewards.append(base + ctx_bonus)
        return max(rewards)

    def get_best_model(self, context: np.ndarray) -> str:
        phase = self._get_phase()
        best_r, best_m = -1, MODELS[0]
        for model in MODELS:
            base = self.phase_base[phase][model]
            ctx_norm = context / (np.linalg.norm(context) + 1e-8)
            ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)
            r = base + ctx_bonus
            if r > best_r:
                best_r, best_m = r, model
        return best_m


# ============================================================================
# BASELINES
# ============================================================================

class StaticRouter:
    """Always picks the specified model."""
    def __init__(self, model: str):
        self.model = model
    def select(self, context: np.ndarray) -> str:
        return self.model


class EMATracker:
    def __init__(self, models: List[str], alpha: float = 0.1, epsilon: float = 0.1,
                 seed: int = 42):
        self.models = models
        self.alpha = alpha
        self.epsilon = epsilon
        self.ema = {m: 0.5 for m in models}
        self.rng = np.random.RandomState(seed)

    def select(self, context: np.ndarray) -> str:
        if self.rng.random() < self.epsilon:
            return self.models[self.rng.randint(len(self.models))]
        return max(self.ema, key=self.ema.get)

    def update(self, model: str, reward: float):
        self.ema[model] = (1 - self.alpha) * self.ema[model] + self.alpha * reward


# ============================================================================
# PRIOR LOADING — with semantic transfer for unknown models
# ============================================================================

# Semantic transfer map: new_model → neighbor to transfer from.
# This matches what the production system does via register_model():
# new models inherit θ (preferences) from their closest known model,
# with A reset to n_eff·I (fresh exploration).
SEMANTIC_NEIGHBORS = {
    "openai/gpt-4o":          "openai/gpt-4-turbo",      # same family
    "openai/gpt-3.5-turbo":   "openai/gpt-4-turbo",      # same provider
    "anthropic/claude-3-haiku": "mistralai/mixtral-8x7b-instruct",  # similar tier
}


def load_and_scale_priors(target_mass: float = PRIOR_SCALE) -> Dict:
    """
    Load production warmup priors and create priors for all 5 models.

    For models with real priors (Mixtral, GPT-4-Turbo): scale to target_mass.
    For models without priors (GPT-3.5, Haiku, GPT-4o): apply semantic transfer
    from the nearest known model — exactly the "First-Child Bias Correction"
    from the production system (Section 3.3 of the paper):
        θ_new = A_neighbor^{-1} @ b_neighbor   (transfer preferences)
        A_new = n_eff · I                       (reset confidence → explore)
        b_new = n_eff · θ_new                   (encode preferences at low confidence)

    This is critical for a fair K=5 evaluation: without it, the warmup expert
    has no informed opinion about 60% of the portfolio, handicapping Corralling.
    """
    norm_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    if norm_path.exists():
        priors = joblib.load(norm_path)
        logger.info(f"  Loaded normalized priors from {norm_path.name}")
    else:
        priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        logger.info(f"  Loaded raw priors")

    scaled = copy.deepcopy(priors)
    dim = priors["context_dim"]

    # Step 1: Scale existing priors to target mass
    for m in priors["A"]:
        current_mass = np.trace(priors["A"][m]) / dim
        if current_mass > 1e-6:
            scale = target_mass / current_mass
            scaled["A"][m] = priors["A"][m] * scale
            scaled["b"][m] = priors["b"][m] * scale
            logger.info(f"  {MODEL_SHORT.get(m, m)}: direct priors, mass {current_mass:.0f} → {target_mass:.0f}")

    # Step 2: Semantic transfer for models without priors
    # (mirrors CostAwareLinUCBRouter.add_model_with_semantic_transfer)
    for new_model, neighbor in SEMANTIC_NEIGHBORS.items():
        if new_model not in scaled["A"] and neighbor in scaled["A"]:
            A_neighbor = scaled["A"][neighbor]
            b_neighbor = scaled["b"][neighbor]
            # Extract learned preferences from neighbor
            A_inv = np.linalg.inv(A_neighbor + 1e-6 * np.eye(dim))
            theta_neighbor = A_inv @ b_neighbor
            # First-Child Bias Correction: transfer θ, reset A
            scaled["A"][new_model] = target_mass * np.eye(dim)
            scaled["b"][new_model] = target_mass * theta_neighbor
            logger.info(f"  {MODEL_SHORT.get(new_model, new_model)}: semantic transfer from "
                        f"{MODEL_SHORT.get(neighbor, neighbor)} "
                        f"(||θ||={np.linalg.norm(theta_neighbor):.3f})")

    return scaled


# ============================================================================
# SINGLE TRIAL
# ============================================================================

@dataclass
class TrialResult:
    seed: int
    rewards_corralling: List[float] = field(default_factory=list)
    rewards_static: List[float] = field(default_factory=list)
    rewards_ema: List[float] = field(default_factory=list)
    oracle_rewards: List[float] = field(default_factory=list)
    expert_weights: List[np.ndarray] = field(default_factory=list)
    model_chosen_corralling: List[str] = field(default_factory=list)
    model_chosen_ema: List[str] = field(default_factory=list)
    failure_detection_step: Optional[int] = None
    recovery_detection_step: Optional[int] = None


def run_single_trial(seed: int, warmup_priors: Dict) -> TrialResult:
    rng = np.random.RandomState(seed)
    result = TrialResult(seed=seed)

    env_corralling = FiveModelEnvironment(seed=seed)
    env_static = FiveModelEnvironment(seed=seed)
    env_ema = FiveModelEnvironment(seed=seed)
    env_oracle = FiveModelEnvironment(seed=seed)

    np.random.seed(seed)
    warmup_expert = CostAwareLinUCBRouter(
        models=MODELS,
        warmup_priors=copy.deepcopy(warmup_priors),
        model_costs=MODEL_COSTS,
        alpha_start=1.0,
        alpha_end=0.1,
        cost_penalty=0.0,
    )
    tabula_rasa = CostAwareTabulaRasaRouter(
        models=MODELS,
        context_dim=CONTEXT_DIM,
        model_costs=MODEL_COSTS,
        alpha_start=1.0,
        alpha_end=0.1,
        cost_penalty=0.0,
    )
    corralling = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa],
        models=MODELS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        loss_decay=1.0,
    )

    static_router = StaticRouter(FAILING_MODEL)
    ema_router = EMATracker(MODELS, alpha=0.15, epsilon=0.1, seed=seed)

    for t in range(N_STEPS):
        context = rng.randn(CONTEXT_DIM)
        # Normalize to match production PCA embedding norms (~1.15).
        # Raw randn(33) has ||x|| ≈ 5.7, causing θᵀx predictions to overshoot
        # [0,1] and triggering PredictionMonitor warnings. The environment's
        # get_reward() already unit-normalizes context internally for the context
        # bonus, so this rescaling only affects the routers' feature space.
        context = context / np.linalg.norm(context) * CONTEXT_NORM

        # Oracle
        oracle_r = env_oracle.get_oracle_reward(context)
        _ = env_oracle.get_reward(MODELS[0], context)
        result.oracle_rewards.append(oracle_r)

        # Corralling
        model_c, token = corralling.select_model(context, total_steps=N_STEPS)
        reward_c = env_corralling.get_reward(model_c, context)
        corralling.update(context, model_c, reward_c, selection_token=token)
        result.rewards_corralling.append(reward_c)
        result.expert_weights.append(corralling.weights.copy())
        result.model_chosen_corralling.append(model_c)

        # Static
        model_s = static_router.select(context)
        reward_s = env_static.get_reward(model_s, context)
        result.rewards_static.append(reward_s)

        # EMA
        model_e = ema_router.select(context)
        reward_e = env_ema.get_reward(model_e, context)
        ema_router.update(model_e, reward_e)
        result.rewards_ema.append(reward_e)
        result.model_chosen_ema.append(model_e)

    # Detection metrics
    weights = np.array(result.expert_weights)
    for t in range(PHASE_BOUNDARIES[0], min(PHASE_BOUNDARIES[1], N_STEPS)):
        if weights[t, 0] < 0.15:
            result.failure_detection_step = t
            break
    for t in range(PHASE_BOUNDARIES[1], N_STEPS):
        if weights[t, 0] > 0.35:
            result.recovery_detection_step = t
            break

    return result


# ============================================================================
# MULTI-SEED
# ============================================================================

def run_all_seeds(warmup_priors: Dict) -> List[TrialResult]:
    results = []
    for seed in range(N_SEEDS):
        result = run_single_trial(seed, warmup_priors)
        results.append(result)
        if (seed + 1) % 5 == 0:
            det = result.failure_detection_step
            det_str = f"t={det} (Δ={det - PHASE_BOUNDARIES[0]})" if det else "not detected"
            logger.info(f"  Seed {seed:2d}: detection={det_str}")
    return results


# ============================================================================
# STATISTICS
# ============================================================================

def compute_statistics(results: List[TrialResult]) -> Dict:
    detection_steps = [r.failure_detection_step for r in results if r.failure_detection_step is not None]
    reaction_times = [d - PHASE_BOUNDARIES[0] for d in detection_steps]
    recovery_steps = [r.recovery_detection_step for r in results if r.recovery_detection_step is not None]

    stats = {
        "n_seeds": len(results),
        "detection_rate": len(detection_steps) / len(results),
        "reaction_mean": np.mean(reaction_times) if reaction_times else None,
        "reaction_std": np.std(reaction_times) if reaction_times else None,
        "reaction_median": np.median(reaction_times) if reaction_times else None,
        "reaction_min": np.min(reaction_times) if reaction_times else None,
        "reaction_max": np.max(reaction_times) if reaction_times else None,
        "recovery_rate": len(recovery_steps) / len(results),
    }

    for method_name, attr in [("corralling", "rewards_corralling"),
                               ("static", "rewards_static"),
                               ("ema", "rewards_ema"),
                               ("oracle", "oracle_rewards")]:
        for phase_name, (start, end) in [("healthy", (0, PHASE_BOUNDARIES[0])),
                                          ("failure", (PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1])),
                                          ("recovery", (PHASE_BOUNDARIES[1], N_STEPS))]:
            vals = [np.mean(getattr(r, attr)[start:end]) for r in results]
            stats[f"{method_name}_{phase_name}_mean"] = np.mean(vals)
            stats[f"{method_name}_{phase_name}_std"] = np.std(vals)

    # Model selection during failure
    for method_name, attr in [("corralling", "model_chosen_corralling"),
                               ("ema", "model_chosen_ema")]:
        for model in MODELS:
            fracs = []
            for r in results:
                choices = getattr(r, attr)[PHASE_BOUNDARIES[0]:PHASE_BOUNDARIES[1]]
                fracs.append(sum(1 for c in choices if c == model) / len(choices))
            stats[f"{method_name}_failure_{MODEL_SHORT[model]}_frac"] = np.mean(fracs)

    return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_figure(results: List[TrialResult], stats: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.04, 1, 1, 1],
        hspace=0.08,
        top=0.94,        # pull panels closer to the suptitle
        bottom=0.06,
    )
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_mid = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[3, 0], sharex=ax_top)

    t = np.arange(N_STEPS)
    window = 20

    all_corr = np.array([r.rewards_corralling for r in results])
    all_stat = np.array([r.rewards_static for r in results])
    all_ema = np.array([r.rewards_ema for r in results])
    all_orac = np.array([r.oracle_rewards for r in results])

    def running_mean(arr_2d, w):
        mean_per_seed = np.array([
            np.convolve(row, np.ones(w) / w, mode="valid") for row in arr_2d
        ])
        return np.mean(mean_per_seed, axis=0), np.std(mean_per_seed, axis=0)

    t_smooth = t[window - 1:]

    corr_mu, corr_std = running_mean(all_corr, window)
    stat_mu, stat_std = running_mean(all_stat, window)
    ema_mu, ema_std = running_mean(all_ema, window)
    orac_mu, orac_std = running_mean(all_orac, window)

    # Phase strip
    ax_phase.set_xlim(0, N_STEPS)
    ax_phase.set_ylim(0, 1)
    ax_phase.axvspan(0, PHASE_BOUNDARIES[0], color="#2ecc71", alpha=0.25)
    ax_phase.axvspan(PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1], color="#e74c3c", alpha=0.25)
    ax_phase.axvspan(PHASE_BOUNDARIES[1], N_STEPS, color="#3498db", alpha=0.25)
    ax_phase.text(50, 0.5, "All Healthy", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#1a7a3a")
    ax_phase.text(200, 0.5, "GPT-4-Turbo Fails", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#a8201a")
    ax_phase.text(400, 0.5, "GPT-4-Turbo Recovers", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#1a5276")
    ax_phase.set_axis_off()

    for ax in (ax_top, ax_mid, ax_bot):
        ax.axvspan(0, PHASE_BOUNDARIES[0], color="#2ecc71", alpha=0.04)
        ax.axvspan(PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1], color="#e74c3c", alpha=0.04)
        ax.axvspan(PHASE_BOUNDARIES[1], N_STEPS, color="#3498db", alpha=0.04)
        ax.axvline(PHASE_BOUNDARIES[0], color="gray", ls="--", lw=1, alpha=0.4)
        ax.axvline(PHASE_BOUNDARIES[1], color="gray", ls="--", lw=1, alpha=0.4)

    # Panel A: Rewards
    ax_top.plot(t_smooth, orac_mu, color="#95a5a6", lw=1.5, ls=":", label="Oracle", zorder=2)
    ax_top.fill_between(t_smooth, orac_mu - orac_std, orac_mu + orac_std,
                        color="#95a5a6", alpha=0.10)
    ax_top.plot(t_smooth, corr_mu, color="#2c3e50", lw=2.5,
                label="banditGPT (Corralling)", zorder=4)
    ax_top.fill_between(t_smooth, corr_mu - corr_std, corr_mu + corr_std,
                        color="#2c3e50", alpha=0.15)
    ax_top.plot(t_smooth, ema_mu, color="#e67e22", lw=1.8, ls="-.",
                label="EMA Tracker", zorder=3)
    ax_top.fill_between(t_smooth, ema_mu - ema_std, ema_mu + ema_std,
                        color="#e67e22", alpha=0.10)
    ax_top.plot(t_smooth, stat_mu, color="#e74c3c", lw=1.8, ls="--",
                label="Static (GPT-4-Turbo Only)", zorder=3)
    ax_top.fill_between(t_smooth, stat_mu - stat_std, stat_mu + stat_std,
                        color="#e74c3c", alpha=0.10)

    ax_top.set_ylabel(f"Reward ({window}-step running avg)", fontsize=11)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.legend(loc="center left", bbox_to_anchor=(0.01, 0.30),
                  fontsize=8.5, framealpha=0.75, edgecolor="gray", handlelength=2.0)

    fig.suptitle(
        "Catastrophic Failure Detection: 5-Model Portfolio (K=5)",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # Summary box
    fail_corr = stats["corralling_failure_mean"]
    fail_ema = stats["ema_failure_mean"]
    fail_orac = stats["oracle_failure_mean"]
    delta = fail_corr - fail_ema
    summary = (
        f'Failure phase reward:\n'
        f'  banditGPT: {fail_corr:.3f}  |  EMA: {fail_ema:.3f}  |  Oracle: {fail_orac:.3f}\n'
        f'  Δ(banditGPT − EMA) = {delta:+.3f}'
    )
    ax_top.text(0.98, 0.03, summary, transform=ax_top.transAxes,
                fontsize=8, fontfamily="monospace", va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95))

    # Panel B: Expert weights
    all_weights = np.array([r.expert_weights for r in results])
    w_warmup_mu = all_weights[:, :, 0].mean(axis=0)
    w_warmup_std = all_weights[:, :, 0].std(axis=0)
    w_tabula_mu = all_weights[:, :, 1].mean(axis=0)
    w_tabula_std = all_weights[:, :, 1].std(axis=0)

    ax_mid.plot(t, w_warmup_mu, color="#e74c3c", lw=2,
                label="Warmup Expert (LinUCB + priors)")
    ax_mid.fill_between(t, w_warmup_mu - w_warmup_std, w_warmup_mu + w_warmup_std,
                        color="#e74c3c", alpha=0.15)
    ax_mid.plot(t, w_tabula_mu, color="#27ae60", lw=2,
                label="Tabula Rasa Expert (LinUCB)")
    ax_mid.fill_between(t, w_tabula_mu - w_tabula_std, w_tabula_mu + w_tabula_std,
                        color="#27ae60", alpha=0.15)
    ax_mid.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax_mid.set_ylabel("Expert Weight $p_{i,t}$", fontsize=11)
    ax_mid.set_ylim(-0.05, 1.05)
    ax_mid.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_mid.get_xticklabels(), visible=False)
    ax_mid.legend(loc="upper left", fontsize=8.5, framealpha=0.75)

    # Panel C: Model selection during failure
    model_colors = {
        MODELS[0]: "#3498db",   # Mixtral: blue
        MODELS[1]: "#2ecc71",   # GPT-3.5: green
        MODELS[2]: "#f39c12",   # Haiku: gold
        MODELS[3]: "#e74c3c",   # GPT-4-Turbo: red
        MODELS[4]: "#9b59b6",   # GPT-4o: purple
    }

    for model in MODELS:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0 for c in r.model_chosen_corralling])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        mu = np.array(fracs_per_seed).mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=2,
                    label=f"Corralling → {MODEL_SHORT[model]}")

    for model in MODELS:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0 for c in r.model_chosen_ema])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        mu = np.array(fracs_per_seed).mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=1.5, ls="--",
                    label=f"EMA → {MODEL_SHORT[model]}")

    ax_bot.set_ylabel("Model Selection Fraction", fontsize=11)
    ax_bot.set_xlabel("Routing Step (t)", fontsize=11)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.2, ls=":")

    # Legend: 2 columns, placed in the upper-right where the data is sparse
    ax_bot.legend(loc="upper right", fontsize=7, framealpha=0.75,
                  edgecolor="gray", ncol=2, columnspacing=1.0, handlelength=2.0)

    fig.align_ylabels([ax_top, ax_mid, ax_bot])

    for fmt in ("png", "pdf"):
        out = output_dir / f"figure6_5model.{fmt}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_report(stats: Dict):
    logger.info("\n" + "=" * 70)
    logger.info("FIGURE 6 (5-MODEL) — CATASTROPHIC FAILURE DETECTION")
    logger.info("=" * 70)

    logger.info(f"\n  Seeds: {stats['n_seeds']}")
    logger.info(f"  Detection rate: {stats['detection_rate'] * 100:.0f}%"
                f"  ({int(stats['detection_rate'] * stats['n_seeds'])}/{stats['n_seeds']})")

    if stats["reaction_mean"] is not None:
        logger.info(f"  Reaction time:  {stats['reaction_mean']:.1f} ± {stats['reaction_std']:.1f} steps"
                     f"  (median={stats['reaction_median']:.0f},"
                     f" range=[{stats['reaction_min']:.0f}, {stats['reaction_max']:.0f}])")

    logger.info(f"  Recovery rate:  {stats['recovery_rate'] * 100:.0f}%")

    logger.info(f"\n  {'Method':<22} {'Healthy':>10} {'Failure':>10} {'Recovery':>10}")
    logger.info("  " + "-" * 54)
    for method in ("oracle", "corralling", "ema", "static"):
        label = {"oracle": "Oracle", "corralling": "banditGPT", "ema": "EMA Tracker",
                 "static": "Static GPT-4-Turbo"}[method]
        h = f"{stats[f'{method}_healthy_mean']:.3f}±{stats[f'{method}_healthy_std']:.3f}"
        f = f"{stats[f'{method}_failure_mean']:.3f}±{stats[f'{method}_failure_std']:.3f}"
        r = f"{stats[f'{method}_recovery_mean']:.3f}±{stats[f'{method}_recovery_std']:.3f}"
        logger.info(f"  {label:<22} {h:>10} {f:>10} {r:>10}")

    logger.info(f"\n  Model selection during failure phase:")
    logger.info(f"  {'Model':<20} {'Corralling':>12} {'EMA':>12}")
    logger.info("  " + "-" * 46)
    for model in MODELS:
        short = MODEL_SHORT[model]
        c = stats[f"corralling_failure_{short}_frac"]
        e = stats[f"ema_failure_{short}_frac"]
        logger.info(f"  {short:<20} {c:>11.1%} {e:>11.1%}")

    fail_corr = stats["corralling_failure_mean"]
    fail_ema = stats["ema_failure_mean"]
    delta = fail_corr - fail_ema
    logger.info(f"\n  KEY COMPARISON (failure phase):")
    logger.info(f"    banditGPT: {fail_corr:.3f}")
    logger.info(f"    EMA:       {fail_ema:.3f}")
    logger.info(f"    Δ:         {delta:+.3f}  {'banditGPT WINS' if delta > 0 else 'EMA wins'}")

    logger.info("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "=" * 70)
    logger.info("FIGURE 6 (5-MODEL): Catastrophic Failure with Large Portfolio")
    logger.info("  5 models • Production router • Orthogonal context strengths")
    logger.info("=" * 70)

    logger.info("\n1. Loading warmup priors...")
    warmup_priors = load_and_scale_priors(target_mass=PRIOR_SCALE)

    logger.info(f"\n2. Running {N_SEEDS}-seed experiment ({N_STEPS} steps each)...")
    results = run_all_seeds(warmup_priors)

    logger.info("\n3. Computing statistics...")
    stats = compute_statistics(results)

    print_report(stats)

    logger.info("\n4. Generating figure...")
    output_dir = Path(__file__).parent / "results"
    plot_figure(results, stats, output_dir)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
