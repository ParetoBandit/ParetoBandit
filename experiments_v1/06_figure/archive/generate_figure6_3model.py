"""
Figure 6 (3-Model): Catastrophic Failure Detection — Portfolio Advantage
========================================================================

Tests the hypothesis that Corralling's contextual routing advantage over
a simple EMA tracker *grows* with more models.

With K=2, EMA only needs to learn "model A bad, model B good" — a trivial
tracking problem.  With K=3, post-failure routing becomes *contextual*:
the optimal policy redistributes traffic across 2 remaining models based
on prompt features.  EMA can't do this; Corralling's LinUCB experts can.

Setup:
  - 3 models: Mixtral-8x7B, GPT-4-Turbo, GPT-4o
  - GPT-4-Turbo fails catastrophically in Phase 2
  - The other two models have *different* per-context strengths
    (Mixtral excels at some contexts, GPT-4o at others)
  - After failure, the optimal policy requires context-dependent routing
    between Mixtral and GPT-4o — exactly where LinUCB shines

Baselines: Oracle, Static GPT-4-Turbo, EMA Tracker, Random
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

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
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
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
]
MODEL_COSTS = {
    "mistralai/mixtral-8x7b-instruct": {"normalized_cost": 0.0},
    "openai/gpt-4-turbo": {"normalized_cost": 0.0},
    "openai/gpt-4o": {"normalized_cost": 0.0},
}
N_STEPS = 500
N_SEEDS = 20
PHASE_BOUNDARIES = (100, 300)  # failure starts, recovery starts
CONTEXT_DIM = 33  # matches production priors
LEARNING_RATE = 0.3  # eta for Corralling
GAMMA = 0.05  # exploration floor
PRIOR_SCALE = 10.0  # effective sample count for warmup priors

MODEL_SHORT = {
    "mistralai/mixtral-8x7b-instruct": "Mixtral",
    "openai/gpt-4-turbo": "GPT-4-Turbo",
    "openai/gpt-4o": "GPT-4o",
}


# ============================================================================
# ENVIRONMENT: 3-model context-dependent three-phase catastrophic failure
# ============================================================================

class ThreeModelEnvironment:
    """
    Context-dependent reward environment with three models and three phases.

    Key design: After GPT-4-Turbo fails, the optimal policy is NOT trivial.
    Mixtral and GPT-4o have *different* context-dependent strengths:
      - Mixtral excels when context[0] > 0  (e.g., formatting tasks)
      - GPT-4o excels when context[0] <= 0  (e.g., reasoning tasks)

    This means a simple EMA tracker that just picks the model with the
    highest average reward will split ~50/50 between Mixtral and GPT-4o
    (same average), while a contextual bandit (LinUCB) can learn the
    context→model mapping and achieve near-Oracle performance.

    Phase 1 (t < 100):  All 3 healthy.
        Mixtral: base=0.75, GPT-4-Turbo: base=0.82, GPT-4o: base=0.80
    Phase 2 (100 <= t < 300):  GPT-4-Turbo fails.
        Mixtral: base=0.75, GPT-4-Turbo: base=0.15, GPT-4o: base=0.80
    Phase 3 (t >= 300):  GPT-4-Turbo recovers.
        Same as Phase 1.
    """

    def __init__(self, seed: int = 42, context_dim: int = CONTEXT_DIM):
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        self.t = 0

        # Generate stable context weights per model.
        # Mixtral and GPT-4o have *anti-correlated* context sensitivities
        # so that the optimal model depends on the context direction.
        base_weights = self.rng.randn(context_dim) * 0.08
        self.context_weights = {
            MODELS[0]: +base_weights.copy(),     # Mixtral: positive projection
            MODELS[1]: self.rng.randn(context_dim) * 0.04,  # GPT-4-Turbo: weak random
            MODELS[2]: -base_weights.copy(),     # GPT-4o: anti-correlated to Mixtral
        }

        # Base rewards per phase.  Pre-failure, GPT-4-Turbo is slightly best
        # on average, making it the natural default choice.
        self.phase_base = {
            "healthy": {MODELS[0]: 0.75, MODELS[1]: 0.82, MODELS[2]: 0.80},
            "failure": {MODELS[0]: 0.75, MODELS[1]: 0.15, MODELS[2]: 0.80},
            "recovery": {MODELS[0]: 0.75, MODELS[1]: 0.82, MODELS[2]: 0.80},
        }

    def _get_phase(self) -> str:
        if self.t < PHASE_BOUNDARIES[0]:
            return "healthy"
        elif self.t < PHASE_BOUNDARIES[1]:
            return "failure"
        return "recovery"

    def get_reward(self, model: str, context: np.ndarray) -> float:
        """Return context-dependent reward for the chosen model."""
        self.t += 1
        phase = self._get_phase()
        base = self.phase_base[phase][model]

        ctx_norm = context / (np.linalg.norm(context) + 1e-8)
        ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)

        noise = self.rng.normal(0, 0.08)
        return float(np.clip(base + ctx_bonus + noise, 0.0, 1.0))

    def get_oracle_reward(self, context: np.ndarray) -> float:
        """Best achievable reward at this timestep (without incrementing t)."""
        phase = self._get_phase()
        rewards = []
        for model in MODELS:
            base = self.phase_base[phase][model]
            ctx_norm = context / (np.linalg.norm(context) + 1e-8)
            ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)
            rewards.append(base + ctx_bonus)
        return max(rewards)

    def get_best_model(self, context: np.ndarray) -> str:
        """Return the oracle's chosen model (for analysis)."""
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

class StaticGPT4Router:
    """Always picks GPT-4-Turbo."""

    def select(self, context: np.ndarray) -> str:
        return MODELS[1]


class EMATracker:
    """
    Exponential moving average reward tracker with epsilon-greedy exploration.
    With K=3 models, exploration budget is spread thinner (P(explore each) = ε/K).
    """

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


class RandomRouter:
    """Uniformly random model selection (lower bound)."""

    def __init__(self, models: List[str], seed: int = 42):
        self.models = models
        self.rng = np.random.RandomState(seed)

    def select(self, context: np.ndarray) -> str:
        return self.models[self.rng.randint(len(self.models))]


# ============================================================================
# PRIOR LOADING
# ============================================================================

def load_and_scale_priors(target_mass: float = PRIOR_SCALE) -> Dict:
    """
    Load production warmup priors and scale to target effective sample size.

    Priors exist for Mixtral and GPT-4-Turbo.  GPT-4o gets semantic transfer
    from GPT-4-Turbo (same family) via the production "First-Child Bias
    Correction": transfer θ (preferences), reset A to n_eff·I (fresh exploration).
    """
    norm_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    if norm_path.exists():
        priors = joblib.load(norm_path)
        logger.info(f"  Loaded normalized priors from {norm_path.name}")
    else:
        priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        logger.info(f"  Loaded raw priors from {DEFAULT_WARMUP_PRIORS_PATH}")

    scaled = copy.deepcopy(priors)
    dim = priors["context_dim"]

    # Scale existing priors
    for m in priors["A"]:
        current_mass = np.trace(priors["A"][m]) / dim
        if current_mass > 1e-6:
            scale = target_mass / current_mass
            scaled["A"][m] = priors["A"][m] * scale
            scaled["b"][m] = priors["b"][m] * scale
            logger.info(f"  {MODEL_SHORT.get(m, m)}: direct priors, mass {current_mass:.0f} → {target_mass:.0f}")

    # Semantic transfer for GPT-4o from GPT-4-Turbo
    neighbor = "openai/gpt-4-turbo"
    new_model = "openai/gpt-4o"
    if new_model not in scaled["A"] and neighbor in scaled["A"]:
        A_inv = np.linalg.inv(scaled["A"][neighbor] + 1e-6 * np.eye(dim))
        theta = A_inv @ scaled["b"][neighbor]
        scaled["A"][new_model] = target_mass * np.eye(dim)
        scaled["b"][new_model] = target_mass * theta
        logger.info(f"  {MODEL_SHORT.get(new_model, new_model)}: semantic transfer from "
                    f"{MODEL_SHORT.get(neighbor, neighbor)} (||θ||={np.linalg.norm(theta):.3f})")

    return scaled


# ============================================================================
# SINGLE TRIAL RUNNER
# ============================================================================

@dataclass
class TrialResult:
    """Results from a single trial."""
    seed: int
    # Per-step tracking
    rewards_corralling: List[float] = field(default_factory=list)
    rewards_static: List[float] = field(default_factory=list)
    rewards_ema: List[float] = field(default_factory=list)
    rewards_random: List[float] = field(default_factory=list)
    oracle_rewards: List[float] = field(default_factory=list)
    # Corralling internals
    expert_weights: List[np.ndarray] = field(default_factory=list)
    model_chosen_corralling: List[str] = field(default_factory=list)
    model_chosen_ema: List[str] = field(default_factory=list)
    oracle_model: List[str] = field(default_factory=list)
    # Detection metrics
    failure_detection_step: Optional[int] = None
    recovery_detection_step: Optional[int] = None


def run_single_trial(seed: int, warmup_priors: Dict) -> TrialResult:
    """Run one complete trial with all methods on the same environment."""
    rng = np.random.RandomState(seed)
    result = TrialResult(seed=seed)

    # --- Environments (one per method, same seed → same contexts) ---
    env_corralling = ThreeModelEnvironment(seed=seed)
    env_static = ThreeModelEnvironment(seed=seed)
    env_ema = ThreeModelEnvironment(seed=seed)
    env_random = ThreeModelEnvironment(seed=seed)
    env_oracle = ThreeModelEnvironment(seed=seed)

    # --- Corralling router (production components) ---
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

    # --- Baselines ---
    static_router = StaticGPT4Router()
    ema_router = EMATracker(MODELS, alpha=0.15, epsilon=0.1, seed=seed)
    random_router = RandomRouter(MODELS, seed=seed)

    # --- Run ---
    for t in range(N_STEPS):
        context = rng.randn(CONTEXT_DIM)

        # Oracle
        oracle_r = env_oracle.get_oracle_reward(context)
        oracle_m = env_oracle.get_best_model(context)
        _ = env_oracle.get_reward(MODELS[0], context)  # advance counter
        result.oracle_rewards.append(oracle_r)
        result.oracle_model.append(oracle_m)

        # Corralling (production API)
        model_c, token = corralling.select_model(context, total_steps=N_STEPS)
        reward_c = env_corralling.get_reward(model_c, context)
        corralling.update(context, model_c, reward_c, selection_token=token)
        result.rewards_corralling.append(reward_c)
        result.expert_weights.append(corralling.weights.copy())
        result.model_chosen_corralling.append(model_c)

        # Static GPT-4-Turbo
        model_s = static_router.select(context)
        reward_s = env_static.get_reward(model_s, context)
        result.rewards_static.append(reward_s)

        # EMA tracker
        model_e = ema_router.select(context)
        reward_e = env_ema.get_reward(model_e, context)
        ema_router.update(model_e, reward_e)
        result.rewards_ema.append(reward_e)
        result.model_chosen_ema.append(model_e)

        # Random
        model_r = random_router.select(context)
        reward_r = env_random.get_reward(model_r, context)
        result.rewards_random.append(reward_r)

    # --- Detection metrics ---
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
# MULTI-SEED ANALYSIS
# ============================================================================

def run_all_seeds(warmup_priors: Dict) -> List[TrialResult]:
    """Run multi-seed evaluation."""
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
    """Aggregate multi-seed statistics."""
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

    # Per-phase average rewards
    for method_name, attr in [("corralling", "rewards_corralling"),
                               ("static", "rewards_static"),
                               ("ema", "rewards_ema"),
                               ("random", "rewards_random"),
                               ("oracle", "oracle_rewards")]:
        for phase_name, (start, end) in [("healthy", (0, PHASE_BOUNDARIES[0])),
                                          ("failure", (PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1])),
                                          ("recovery", (PHASE_BOUNDARIES[1], N_STEPS))]:
            vals = [np.mean(getattr(r, attr)[start:end]) for r in results]
            stats[f"{method_name}_{phase_name}_mean"] = np.mean(vals)
            stats[f"{method_name}_{phase_name}_std"] = np.std(vals)

    # Model selection analysis (during failure phase only, for Corralling vs EMA)
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
# VISUALIZATION — Side-by-side comparison (2-model vs 3-model)
# ============================================================================

def plot_figure6_3model(results: List[TrialResult], stats: Dict, output_dir: Path):
    """
    Three-panel figure:
      Panel A — Running average reward: all methods
      Panel B — Expert weight evolution
      Panel C — Model selection during failure phase (Corralling vs EMA)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.06, 1.2, 0.85, 0.85],
        hspace=0.12,
    )
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_mid = fig.add_subplot(gs[2, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[3, 0], sharex=ax_top)

    t = np.arange(N_STEPS)
    window = 20

    # ---- Aggregate rewards ----
    all_corr = np.array([r.rewards_corralling for r in results])
    all_stat = np.array([r.rewards_static for r in results])
    all_ema = np.array([r.rewards_ema for r in results])
    all_rand = np.array([r.rewards_random for r in results])
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
    rand_mu, rand_std = running_mean(all_rand, window)
    orac_mu, orac_std = running_mean(all_orac, window)

    # ---- Phase-label strip ----
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

    # ---- Phase backgrounds ----
    for ax in (ax_top, ax_mid, ax_bot):
        ax.axvspan(0, PHASE_BOUNDARIES[0], color="#2ecc71", alpha=0.04)
        ax.axvspan(PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1], color="#e74c3c", alpha=0.04)
        ax.axvspan(PHASE_BOUNDARIES[1], N_STEPS, color="#3498db", alpha=0.04)
        ax.axvline(PHASE_BOUNDARIES[0], color="gray", ls="--", lw=1, alpha=0.4)
        ax.axvline(PHASE_BOUNDARIES[1], color="gray", ls="--", lw=1, alpha=0.4)

    # ---- Panel A: Reward trajectories ----
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

    ax_top.plot(t_smooth, rand_mu, color="#bdc3c7", lw=1.2, ls=":",
                label="Random", zorder=1)

    ax_top.set_ylabel(f"Reward ({window}-step running avg)", fontsize=11)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_top.legend(
        loc="center left", bbox_to_anchor=(0.01, 0.30),
        fontsize=8.5, framealpha=0.95, edgecolor="gray",
        handlelength=2.0,
    )

    fig.suptitle(
        "Catastrophic Failure Detection: 3-Model Portfolio",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ---- Panel B: Expert weights ----
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
    ax_mid.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # ---- Panel C: Model selection fractions (running window) ----
    model_colors = {
        MODELS[0]: "#3498db",   # Mixtral: blue
        MODELS[1]: "#e74c3c",   # GPT-4-Turbo: red
        MODELS[2]: "#9b59b6",   # GPT-4o: purple
    }

    # Corralling model selection fractions (running average)
    for model in MODELS:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0 for c in r.model_chosen_corralling])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        fracs_arr = np.array(fracs_per_seed)
        mu = fracs_arr.mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=2,
                    label=f"Corralling → {MODEL_SHORT[model]}")

    # EMA model selection fractions (running average) — dashed
    for model in MODELS:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0 for c in r.model_chosen_ema])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        fracs_arr = np.array(fracs_per_seed)
        mu = fracs_arr.mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=1.5, ls="--",
                    label=f"EMA → {MODEL_SHORT[model]}")

    ax_bot.set_ylabel("Model Selection Fraction", fontsize=11)
    ax_bot.set_xlabel("Routing Step (t)", fontsize=11)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.2, ls=":")
    ax_bot.legend(loc="center right", fontsize=7.5, framealpha=0.95, ncol=2)

    # Summary stats
    fail_corr = stats["corralling_failure_mean"]
    fail_ema = stats["ema_failure_mean"]
    fail_orac = stats["oracle_failure_mean"]
    delta = fail_corr - fail_ema
    summary = (
        f'Failure phase reward:\n'
        f'  banditGPT: {fail_corr:.3f}  |  EMA: {fail_ema:.3f}  |  Oracle: {fail_orac:.3f}\n'
        f'  Δ(banditGPT − EMA) = {delta:+.3f}'
    )
    ax_top.text(
        0.98, 0.03, summary, transform=ax_top.transAxes,
        fontsize=8, fontfamily="monospace", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95),
    )

    fig.align_ylabels([ax_top, ax_mid, ax_bot])

    for fmt in ("png", "pdf"):
        out = output_dir / f"figure6_3model.{fmt}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_report(stats: Dict):
    """Print structured summary of results."""
    logger.info("\n" + "=" * 70)
    logger.info("FIGURE 6 (3-MODEL) — CATASTROPHIC FAILURE DETECTION")
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
    for method in ("oracle", "corralling", "ema", "random", "static"):
        label = {"oracle": "Oracle", "corralling": "banditGPT", "ema": "EMA Tracker",
                 "static": "Static GPT-4-Turbo", "random": "Random"}[method]
        h = f"{stats[f'{method}_healthy_mean']:.3f}±{stats[f'{method}_healthy_std']:.3f}"
        f = f"{stats[f'{method}_failure_mean']:.3f}±{stats[f'{method}_failure_std']:.3f}"
        r = f"{stats[f'{method}_recovery_mean']:.3f}±{stats[f'{method}_recovery_std']:.3f}"
        logger.info(f"  {label:<22} {h:>10} {f:>10} {r:>10}")

    # Model selection during failure
    logger.info(f"\n  Model selection during failure phase:")
    logger.info(f"  {'Model':<20} {'Corralling':>12} {'EMA':>12}")
    logger.info("  " + "-" * 46)
    for model in MODELS:
        short = MODEL_SHORT[model]
        c_frac = stats[f"corralling_failure_{short}_frac"]
        e_frac = stats[f"ema_failure_{short}_frac"]
        logger.info(f"  {short:<20} {c_frac:>11.1%} {e_frac:>11.1%}")

    # Key comparison
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
    logger.info("FIGURE 6 (3-MODEL): Catastrophic Failure with Portfolio Advantage")
    logger.info("  3 models • Production router • Context-dependent routing")
    logger.info("=" * 70)

    # Load production warmup priors
    logger.info("\n1. Loading warmup priors...")
    warmup_priors = load_and_scale_priors(target_mass=PRIOR_SCALE)

    # Run multi-seed experiment
    logger.info(f"\n2. Running {N_SEEDS}-seed experiment ({N_STEPS} steps each)...")
    results = run_all_seeds(warmup_priors)

    # Compute statistics
    logger.info("\n3. Computing statistics...")
    stats = compute_statistics(results)

    # Print report
    print_report(stats)

    # Generate figure
    logger.info("\n4. Generating figure...")
    output_dir = Path(__file__).parent / "results"
    plot_figure6_3model(results, stats, output_dir)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
