"""
Figure 6 CORRECTED: Catastrophic Failure Detection with Production Router
=========================================================================

This experiment validates Corralling as a safety mechanism using the PRODUCTION
banditGPT router components (CostAwareLinUCBRouter, CostAwareTabulaRasaRouter,
CorrallingRouter with selection_token).

Key improvements over the original experiment:
  1. Real LinUCB experts (not hard-coded mock experts)
  2. Proper selection_token handling (meta-weights actually update)
  3. Context-dependent rewards (experts learn context→quality mapping)
  4. Multiple baselines (Oracle, Static GPT-4-Turbo, EMA Tracker)
  5. Multi-seed evaluation (20 seeds, mean ± std)
  6. Both experts can select EITHER model (outcome not predetermined)

Three-Phase Scenario:
  Phase 1 (t=0-100):   Both models healthy, equal quality
  Phase 2 (t=100-300):  GPT-4-Turbo catastrophically fails (d ≈ 5.0)
  Phase 3 (t=300-500):  GPT-4-Turbo recovers

What this tests: Can the Corralling meta-learner, coordinating two real LinUCB
experts that both explore freely, detect and route around a catastrophic model
failure faster than simpler alternatives?
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
MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
MODEL_COSTS = {
    "mistralai/mixtral-8x7b-instruct": {"normalized_cost": 0.0},
    "openai/gpt-4-turbo": {"normalized_cost": 0.0},  # cost disabled for this experiment
}
N_STEPS = 500
N_SEEDS = 20
PHASE_BOUNDARIES = (100, 300)  # failure starts, recovery starts
CONTEXT_DIM = 33  # matches production priors
LEARNING_RATE = 0.3  # eta for Corralling
GAMMA = 0.05  # exploration floor
PRIOR_SCALE = 10.0  # effective sample count for warmup priors


# ============================================================================
# ENVIRONMENT: Context-dependent three-phase catastrophic failure
# ============================================================================

class CatastrophicFailureEnvironment:
    """
    Context-dependent reward environment with three phases.

    Unlike the original experiment, rewards depend on context via a linear
    model per (phase, model).  This means the LinUCB experts can learn
    meaningful theta vectors, and the experiment tests whether Corralling
    coordinates experts that *both learn and explore freely*.

    Phase 1 (t < 100):  Both models healthy.
        Mixtral  base=0.80, GPT-4-Turbo base=0.80, context_scale=0.05
    Phase 2 (100 <= t < 300):  GPT-4-Turbo catastrophically fails.
        Mixtral  base=0.80, GPT-4-Turbo base=0.15, context_scale=0.05
    Phase 3 (t >= 300):  GPT-4-Turbo recovers.
        Mixtral  base=0.80, GPT-4-Turbo base=0.80, context_scale=0.05
    """

    def __init__(self, seed: int = 42, context_dim: int = CONTEXT_DIM):
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        self.t = 0

        # Stable model-specific context weights (persist across phases)
        self.context_weights = {
            MODELS[0]: self.rng.randn(context_dim) * 0.05,
            MODELS[1]: self.rng.randn(context_dim) * 0.05,
        }

        self.phase_base = {
            "healthy": {MODELS[0]: 0.80, MODELS[1]: 0.80},
            "failure": {MODELS[0]: 0.80, MODELS[1]: 0.15},
            "recovery": {MODELS[0]: 0.80, MODELS[1]: 0.80},
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

        # Context contribution (bounded via tanh)
        ctx_norm = context / (np.linalg.norm(context) + 1e-8)
        ctx_bonus = 0.05 * np.tanh(self.context_weights[model] @ ctx_norm)

        noise = self.rng.normal(0, 0.08)
        return float(np.clip(base + ctx_bonus + noise, 0.0, 1.0))

    def get_oracle_reward(self, context: np.ndarray) -> float:
        """Best achievable reward at this timestep (without incrementing t)."""
        phase = self._get_phase()
        rewards = []
        for model in MODELS:
            base = self.phase_base[phase][model]
            ctx_norm = context / (np.linalg.norm(context) + 1e-8)
            ctx_bonus = 0.05 * np.tanh(self.context_weights[model] @ ctx_norm)
            rewards.append(base + ctx_bonus)
        return max(rewards)


# ============================================================================
# BASELINES
# ============================================================================

class StaticGPT4Router:
    """Always picks GPT-4-Turbo — simulates a naive 'always use the best model' policy."""

    def select(self, context: np.ndarray) -> str:
        return MODELS[1]  # GPT-4-Turbo


class EMATracker:
    """
    Exponential moving average reward tracker with model switching.

    Maintains per-model running reward estimates and picks the model with
    the higher EMA.  Uses epsilon-greedy exploration.

    This is a simple, widely-used production baseline.
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


# ============================================================================
# PRIOR LOADING
# ============================================================================

def load_and_scale_priors(target_mass: float = PRIOR_SCALE) -> Dict:
    """Load production warmup priors and scale to target effective sample size."""
    # Try normalized priors first, fall back to raw
    norm_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    if norm_path.exists():
        priors = joblib.load(norm_path)
        logger.info(f"  Loaded normalized priors from {norm_path.name}")
    else:
        priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        logger.info(f"  Loaded raw priors from {DEFAULT_WARMUP_PRIORS_PATH.name}")

    scaled = copy.deepcopy(priors)
    dim = priors["context_dim"]

    for m in priors["A"]:
        current_mass = np.trace(priors["A"][m]) / dim
        if current_mass > 1e-6:
            scale = target_mass / current_mass
            scaled["A"][m] = priors["A"][m] * scale
            scaled["b"][m] = priors["b"][m] * scale
            logger.info(f"  {m}: mass {current_mass:.0f} → {target_mass:.0f} (×{scale:.4f})")

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
    oracle_rewards: List[float] = field(default_factory=list)
    # Corralling internals
    expert_weights: List[np.ndarray] = field(default_factory=list)
    model_chosen_corralling: List[str] = field(default_factory=list)
    # Detection metrics
    failure_detection_step: Optional[int] = None
    recovery_detection_step: Optional[int] = None


def run_single_trial(seed: int, warmup_priors: Dict) -> TrialResult:
    """Run one complete trial with all methods on the same environment."""
    rng = np.random.RandomState(seed)
    result = TrialResult(seed=seed)

    # --- Environment (shared across all methods) ---
    env_corralling = CatastrophicFailureEnvironment(seed=seed)
    env_static = CatastrophicFailureEnvironment(seed=seed)
    env_ema = CatastrophicFailureEnvironment(seed=seed)
    env_oracle = CatastrophicFailureEnvironment(seed=seed)

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
        loss_decay=1.0,  # stationary within each phase
    )

    # --- Baselines ---
    static_router = StaticGPT4Router()
    ema_router = EMATracker(MODELS, alpha=0.15, epsilon=0.1, seed=seed)

    # --- Run ---
    for t in range(N_STEPS):
        context = rng.randn(CONTEXT_DIM)

        # Oracle (best possible)
        oracle_r = env_oracle.get_oracle_reward(context)
        # Advance oracle env's internal counter to stay in sync
        _ = env_oracle.get_reward(MODELS[0], context)
        result.oracle_rewards.append(oracle_r)

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

    # --- Compute detection metrics ---
    weights = np.array(result.expert_weights)

    # Failure detection: warmup weight drops below 0.15
    # (warmup expert starts with real priors, not hard-coded to GPT-4-Turbo,
    #  so we look at when the meta-learner decommissions the warmup expert
    #  because its LinUCB keeps recommending the now-failing GPT-4-Turbo)
    for t in range(PHASE_BOUNDARIES[0], min(PHASE_BOUNDARIES[1], N_STEPS)):
        if weights[t, 0] < 0.15:
            result.failure_detection_step = t
            break

    # Recovery detection: warmup weight rises back above 0.35
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
                               ("oracle", "oracle_rewards")]:
        for phase_name, (start, end) in [("healthy", (0, PHASE_BOUNDARIES[0])),
                                          ("failure", (PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1])),
                                          ("recovery", (PHASE_BOUNDARIES[1], N_STEPS))]:
            vals = [np.mean(getattr(r, attr)[start:end]) for r in results]
            stats[f"{method_name}_{phase_name}_mean"] = np.mean(vals)
            stats[f"{method_name}_{phase_name}_std"] = np.std(vals)

    return stats


# ============================================================================
# VISUALIZATION — Two clean panels
# ============================================================================

def plot_figure6(results: List[TrialResult], stats: Dict, output_dir: Path):
    """
    Two-panel figure:
      Panel A — Running average reward: Corralling vs baselines vs Oracle
      Panel B — Expert weight evolution (mean ± std across seeds)

    Layout uses a 3-row gridspec: a thin phase-label strip on top, then
    the two data panels.  Legends are placed outside the data area so
    nothing overlaps the curves or confidence bands.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[0.06, 1.3, 1],
        hspace=0.08,
    )
    ax_phase = fig.add_subplot(gs[0, 0])   # thin strip for phase labels
    ax_top = fig.add_subplot(gs[1, 0])     # reward panel
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)  # weight panel

    t = np.arange(N_STEPS)
    window = 20  # running average window

    # ---- Aggregate rewards across seeds ----
    all_corr = np.array([r.rewards_corralling for r in results])
    all_stat = np.array([r.rewards_static for r in results])
    all_ema = np.array([r.rewards_ema for r in results])
    all_orac = np.array([r.oracle_rewards for r in results])

    def running_mean(arr_2d, w):
        """Running mean across axis=1 (time), averaged across axis=0 (seeds)."""
        mean_per_seed = np.array([
            np.convolve(row, np.ones(w) / w, mode="valid") for row in arr_2d
        ])
        return np.mean(mean_per_seed, axis=0), np.std(mean_per_seed, axis=0)

    t_smooth = t[window - 1:]

    corr_mu, corr_std = running_mean(all_corr, window)
    stat_mu, stat_std = running_mean(all_stat, window)
    ema_mu, ema_std = running_mean(all_ema, window)
    orac_mu, orac_std = running_mean(all_orac, window)

    # ---- Phase-label strip (no data, just coloured spans + text) ----
    ax_phase.set_xlim(0, N_STEPS)
    ax_phase.set_ylim(0, 1)
    ax_phase.axvspan(0, PHASE_BOUNDARIES[0], color="#2ecc71", alpha=0.25)
    ax_phase.axvspan(PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1], color="#e74c3c", alpha=0.25)
    ax_phase.axvspan(PHASE_BOUNDARIES[1], N_STEPS, color="#3498db", alpha=0.25)
    ax_phase.text(50, 0.5, "Both Healthy", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#1a7a3a")
    ax_phase.text(200, 0.5, "GPT-4-Turbo Fails", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#a8201a")
    ax_phase.text(400, 0.5, "GPT-4-Turbo Recovers", ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#1a5276")
    ax_phase.set_axis_off()

    # ---- Phase backgrounds on data panels (subtle) ----
    for ax in (ax_top, ax_bot):
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

    ax_top.set_ylabel(f"Reward ({window}-step running avg)", fontsize=11)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Legend in the open space during failure phase (between banditGPT ~0.45
    # and Static ~0.15, around x=200) — no data lives at y=0.25 there.
    ax_top.legend(
        loc="center left", bbox_to_anchor=(0.01, 0.28),
        fontsize=8.5, framealpha=0.95, edgecolor="gray",
        handlelength=2.0,
    )

    fig.suptitle(
        "Catastrophic Failure Detection: Production Router",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ---- Panel B: Expert weights ----
    all_weights = np.array([r.expert_weights for r in results])  # (seeds, steps, 2)
    w_warmup_mu = all_weights[:, :, 0].mean(axis=0)
    w_warmup_std = all_weights[:, :, 0].std(axis=0)
    w_tabula_mu = all_weights[:, :, 1].mean(axis=0)
    w_tabula_std = all_weights[:, :, 1].std(axis=0)

    ax_bot.plot(t, w_warmup_mu, color="#e74c3c", lw=2,
                label="Warmup Expert (LinUCB + priors)")
    ax_bot.fill_between(t, w_warmup_mu - w_warmup_std, w_warmup_mu + w_warmup_std,
                        color="#e74c3c", alpha=0.15)

    ax_bot.plot(t, w_tabula_mu, color="#27ae60", lw=2,
                label="Tabula Rasa Expert (LinUCB)")
    ax_bot.fill_between(t, w_tabula_mu - w_tabula_std, w_tabula_mu + w_tabula_std,
                        color="#27ae60", alpha=0.15)

    ax_bot.axhline(0.5, color="gray", ls=":", alpha=0.3)

    # Detection annotation — arrow points to the mean warmup weight at detection time
    if stats["reaction_mean"] is not None:
        det_t = int(PHASE_BOUNDARIES[0] + stats["reaction_mean"])
        det_t_clamped = min(det_t, len(w_warmup_mu) - 1)
        target_y = w_warmup_mu[det_t_clamped]
        ax_bot.annotate(
            f'Median detection: {stats["reaction_median"]:.0f} steps\n'
            f'(mean {stats["reaction_mean"]:.0f} ± {stats["reaction_std"]:.0f})',
            xy=(det_t, target_y),
            xytext=(260, 0.92),
            fontsize=8.5, fontweight="bold", color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#c0392b", alpha=0.95),
            zorder=5,
        )

    ax_bot.set_ylabel("Expert Weight $p_{i,t}$", fontsize=11)
    ax_bot.set_xlabel("Routing Step (t)", fontsize=11)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.2, ls=":")

    # Legend in upper-left (Phase 1 region has ~0.5 weight, plenty of room above)
    ax_bot.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Summary stats text box — lower-right corner (no data there)
    det_rate = stats["detection_rate"] * 100
    react_str = (f'{stats["reaction_mean"]:.0f} ± {stats["reaction_std"]:.0f}'
                 if stats["reaction_mean"] is not None else "N/A")
    summary = (
        f'Detection: {det_rate:.0f}% ({int(stats["detection_rate"] * stats["n_seeds"])}'
        f'/{stats["n_seeds"]} seeds)\n'
        f'Reaction:  {react_str} steps (median {stats["reaction_median"]:.0f})\n'
        f'Recovery:  {stats["recovery_rate"] * 100:.0f}%'
    )
    ax_bot.text(
        0.98, 0.03, summary, transform=ax_bot.transAxes,
        fontsize=8, fontfamily="monospace", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95),
    )

    fig.align_ylabels([ax_top, ax_bot])

    for fmt in ("png", "pdf"):
        out = output_dir / f"figure6_corrected.{fmt}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_report(stats: Dict):
    """Print structured summary of results."""
    logger.info("\n" + "=" * 70)
    logger.info("FIGURE 6 — CORRECTED EXPERIMENT RESULTS")
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

    logger.info("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "=" * 70)
    logger.info("FIGURE 6 CORRECTED: Catastrophic Failure Detection")
    logger.info("  Production router • Real LinUCB experts • Multi-seed")
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
    plot_figure6(results, stats, output_dir)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
