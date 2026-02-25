"""
Catastrophic Failure Detection — K=5 and K=10 Portfolios
=========================================================

Evaluates Corralling's automatic failover under catastrophic model failure
using the **production hybrid router** (BanditRouter with Corralling,
Hybrid LinUCB family sharing, and 43-model warmup priors).

Setup:
  - GPT-4.1 (the best model) catastrophically fails in Phase 2
  - After failure, K-1 remaining models have context-dependent strengths
  - EMA tracks K averages and explores ε/K per model
  - banditGPT uses the full production routing stack

Portfolios (matching Section 5.2):
  K=5:  Llama-3.1-8B, Mixtral-8x7B, Gemini-2.5-Flash, Claude-Sonnet-4, GPT-4.1
  K=10: K5 + Llama-4-Maverick, Gemma-3-27B, Claude-Haiku-4.5, GPT-4-Turbo, DeepSeek-V3

Base rewards derived from holdout evaluation (Section 5.2, Table 3).
"""

import sys
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.router_factory import create_experiment_router
from utils.multimodel import (
    MODEL_CATALOG, PORTFOLIO_K5, PORTFOLIO_K10,
    TARGET_NEFF, ALPHA_START, CORRALLING_LR, CORRALLING_GAMMA,
    build_model_registry, load_warmup_priors, MULTIMODEL_WARMUP_PRIORS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STEPS = 500
N_SEEDS = 20
PHASE_BOUNDARIES = (100, 300)
CONTEXT_DIM = 33
CONTEXT_NORM = 1.15

FAILING_MODEL = "openai/gpt-4.1"
FAILURE_REWARD = 0.15

HOLDOUT_REWARDS = {
    "meta-llama/llama-3.1-8b-instruct": 0.745,
    "mistralai/mixtral-8x7b-instruct": 0.823,
    "google/gemini-2.5-flash-preview-09-2025": 0.953,
    "anthropic/claude-sonnet-4": 0.975,
    "openai/gpt-4.1": 0.983,
    "meta-llama/llama-4-maverick": 0.929,
    "google/gemma-3-27b-it": 0.951,
    "anthropic/claude-haiku-4.5": 0.951,
    "openai/gpt-4-turbo": 0.812,
    "deepseek/deepseek-chat-v3-0324": 0.973,
}


def short_name(model_id: str) -> str:
    return MODEL_CATALOG[model_id]["display"]


def validate_warmup_priors(models: List[str]) -> Dict:
    """Load and validate that warmup priors exist for every portfolio model.

    Returns the K-specific subset of the 43-model priors (only the models
    in ``models`` are included).  Raises if any model is missing.
    """
    priors = load_warmup_priors(models)
    covered = set(priors["A"].keys())
    missing = [m for m in models if m not in covered]
    if missing:
        raise RuntimeError(
            f"Warmup priors missing for {len(missing)} models: "
            + ", ".join(missing)
        )
    logger.info(f"  Warmup priors: {len(covered)}/{len(models)} models covered "
                f"(dim={priors['context_dim']})")
    for m in models:
        trace_a = np.trace(priors["A"][m])
        norm_b = np.linalg.norm(priors["b"][m])
        logger.info(f"    {short_name(m):<22} tr(A)={trace_a:7.1f}  ||b||={norm_b:.3f}")
    return priors


# ============================================================================
# ENVIRONMENT
# ============================================================================

class CatastrophicFailureEnvironment:
    """
    Context-dependent reward environment with K models and three phases.

    Each non-failing model excels in a different region of context space
    (orthogonal directions via Gram-Schmidt). Post-failure routing across
    K-1 remaining models is a genuinely contextual decision.

    Base rewards match real holdout evaluation values (Section 5.2).
    """

    def __init__(self, models: List[str], failing_model: str,
                 seed: int = 42, context_dim: int = CONTEXT_DIM):
        self.models = models
        self.failing_model = failing_model
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        self.t = 0

        non_failing = [m for m in models if m != failing_model]
        n_healthy = len(non_failing)

        raw_dirs = self.rng.randn(n_healthy, context_dim)
        ortho_dirs = np.zeros_like(raw_dirs)
        for i in range(n_healthy):
            v = raw_dirs[i].copy()
            for j in range(i):
                v -= np.dot(v, ortho_dirs[j]) * ortho_dirs[j]
            ortho_dirs[i] = v / (np.linalg.norm(v) + 1e-8)

        scale = 0.06
        self.context_weights = {}
        for i, model in enumerate(non_failing):
            self.context_weights[model] = ortho_dirs[i] * scale
        self.context_weights[failing_model] = self.rng.randn(context_dim) * 0.02

        base_healthy = {m: HOLDOUT_REWARDS[m] for m in models}
        base_failure = dict(base_healthy)
        base_failure[failing_model] = FAILURE_REWARD

        self.phase_base = {
            "healthy": base_healthy,
            "failure": base_failure,
            "recovery": dict(base_healthy),
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
        for model in self.models:
            base = self.phase_base[phase][model]
            ctx_norm = context / (np.linalg.norm(context) + 1e-8)
            ctx_bonus = np.tanh(self.context_weights[model] @ ctx_norm)
            rewards.append(base + ctx_bonus)
        return max(rewards)


# ============================================================================
# BASELINES
# ============================================================================

class StaticRouter:
    def __init__(self, model: str):
        self.model = model
    def select(self, context: np.ndarray) -> str:
        return self.model


class EMATracker:
    def __init__(self, models: List[str], alpha: float = 0.1,
                 epsilon: float = 0.1, seed: int = 42):
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


def run_single_trial(seed: int, models: List[str]) -> TrialResult:
    rng = np.random.RandomState(seed)
    result = TrialResult(seed=seed)
    K = len(models)

    env_corr = CatastrophicFailureEnvironment(models, FAILING_MODEL, seed=seed)
    env_static = CatastrophicFailureEnvironment(models, FAILING_MODEL, seed=seed)
    env_ema = CatastrophicFailureEnvironment(models, FAILING_MODEL, seed=seed)
    env_oracle = CatastrophicFailureEnvironment(models, FAILING_MODEL, seed=seed)

    np.random.seed(seed)
    router = create_experiment_router(
        model_registry=build_model_registry(models),
        feature_dim=CONTEXT_DIM,
        prior_n_effective=TARGET_NEFF,
        alpha=ALPHA_START,
        warmup_path=str(MULTIMODEL_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=0.0,
    )

    static_router = StaticRouter(FAILING_MODEL)
    ema_router = EMATracker(models, alpha=0.15, epsilon=0.1, seed=seed)

    for t in range(N_STEPS):
        context = rng.randn(CONTEXT_DIM)
        context = context / np.linalg.norm(context) * CONTEXT_NORM

        oracle_r = env_oracle.get_oracle_reward(context)
        _ = env_oracle.get_reward(models[0], context)
        result.oracle_rewards.append(oracle_r)

        model_c, log = router.route(context, total_steps=N_STEPS)
        reward_c = env_corr.get_reward(model_c, context)
        router.process_feedback(log.request_id, reward_c)
        result.rewards_corralling.append(reward_c)
        result.expert_weights.append(router.corralling_router.weights.copy())
        result.model_chosen_corralling.append(model_c)

        model_s = static_router.select(context)
        reward_s = env_static.get_reward(model_s, context)
        result.rewards_static.append(reward_s)

        model_e = ema_router.select(context)
        reward_e = env_ema.get_reward(model_e, context)
        ema_router.update(model_e, reward_e)
        result.rewards_ema.append(reward_e)
        result.model_chosen_ema.append(model_e)

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

def run_all_seeds(models: List[str]) -> List[TrialResult]:
    results = []
    for seed in range(N_SEEDS):
        result = run_single_trial(seed, models)
        results.append(result)
        if (seed + 1) % 5 == 0:
            det = result.failure_detection_step
            det_str = f"t={det} (Δ={det - PHASE_BOUNDARIES[0]})" if det else "not detected"
            logger.info(f"  Seed {seed:2d}: detection={det_str}")
    return results


# ============================================================================
# STATISTICS
# ============================================================================

def compute_statistics(results: List[TrialResult], models: List[str]) -> Dict:
    detection_steps = [r.failure_detection_step for r in results
                       if r.failure_detection_step is not None]
    reaction_times = [d - PHASE_BOUNDARIES[0] for d in detection_steps]
    recovery_steps = [r.recovery_detection_step for r in results
                      if r.recovery_detection_step is not None]

    stats = {
        "n_seeds": len(results),
        "K": len(models),
        "detection_rate": len(detection_steps) / len(results),
        "reaction_mean": float(np.mean(reaction_times)) if reaction_times else None,
        "reaction_std": float(np.std(reaction_times)) if reaction_times else None,
        "reaction_median": float(np.median(reaction_times)) if reaction_times else None,
        "reaction_min": float(np.min(reaction_times)) if reaction_times else None,
        "reaction_max": float(np.max(reaction_times)) if reaction_times else None,
        "recovery_rate": len(recovery_steps) / len(results),
    }

    for method_name, attr in [("corralling", "rewards_corralling"),
                               ("static", "rewards_static"),
                               ("ema", "rewards_ema"),
                               ("oracle", "oracle_rewards")]:
        for phase_name, (start, end) in [
            ("healthy", (0, PHASE_BOUNDARIES[0])),
            ("failure", (PHASE_BOUNDARIES[0], PHASE_BOUNDARIES[1])),
            ("recovery", (PHASE_BOUNDARIES[1], N_STEPS)),
        ]:
            vals = [np.mean(getattr(r, attr)[start:end]) for r in results]
            stats[f"{method_name}_{phase_name}_mean"] = float(np.mean(vals))
            stats[f"{method_name}_{phase_name}_std"] = float(np.std(vals))

    for method_name, attr in [("corralling", "model_chosen_corralling"),
                               ("ema", "model_chosen_ema")]:
        for model in models:
            fracs = []
            for r in results:
                choices = getattr(r, attr)[PHASE_BOUNDARIES[0]:PHASE_BOUNDARIES[1]]
                fracs.append(sum(1 for c in choices if c == model) / len(choices))
            stats[f"{method_name}_failure_{short_name(model)}_frac"] = float(np.mean(fracs))

    return stats


# ============================================================================
# VISUALIZATION
# ============================================================================

MODEL_COLORS = [
    "#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#d35400", "#7f8c8d",
]


def plot_figure(results: List[TrialResult], stats: Dict, models: List[str],
                portfolio_label: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    K = len(models)

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.04, 1, 1, 1],
        hspace=0.08, top=0.94, bottom=0.06,
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
    ax_phase.text(200, 0.5, f"{short_name(FAILING_MODEL)} Fails", ha="center",
                  va="center", fontsize=9, fontweight="bold", color="#a8201a")
    ax_phase.text(400, 0.5, f"{short_name(FAILING_MODEL)} Recovers", ha="center",
                  va="center", fontsize=9, fontweight="bold", color="#1a5276")
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
                label=f"Static ({short_name(FAILING_MODEL)} Only)", zorder=3)
    ax_top.fill_between(t_smooth, stat_mu - stat_std, stat_mu + stat_std,
                        color="#e74c3c", alpha=0.10)

    ax_top.set_ylabel(f"Reward ({window}-step running avg)", fontsize=11)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.legend(loc="center left", bbox_to_anchor=(0.01, 0.30),
                  fontsize=8.5, framealpha=0.75, edgecolor="gray", handlelength=2.0)

    fig.suptitle(
        f"Catastrophic Failure Detection: {K}-Model Portfolio (K={K})",
        fontsize=13, fontweight="bold", y=0.97,
    )

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
                label="Warmup Expert (Hybrid LinUCB + priors)")
    ax_mid.fill_between(t, w_warmup_mu - w_warmup_std, w_warmup_mu + w_warmup_std,
                        color="#e74c3c", alpha=0.15)
    ax_mid.plot(t, w_tabula_mu, color="#27ae60", lw=2,
                label="Tabula Rasa Expert (Hybrid LinUCB)")
    ax_mid.fill_between(t, w_tabula_mu - w_tabula_std, w_tabula_mu + w_tabula_std,
                        color="#27ae60", alpha=0.15)
    ax_mid.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax_mid.set_ylabel("Expert Weight $p_{i,t}$", fontsize=11)
    ax_mid.set_ylim(-0.05, 1.05)
    ax_mid.grid(True, alpha=0.2, ls=":")
    plt.setp(ax_mid.get_xticklabels(), visible=False)
    ax_mid.legend(loc="upper left", fontsize=8.5, framealpha=0.75)

    # Panel C: Model selection fractions
    model_colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

    for model in models:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0
                               for c in r.model_chosen_corralling])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        mu = np.array(fracs_per_seed).mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=2,
                    label=f"Corr → {short_name(model)}")

    for model in models:
        fracs_per_seed = []
        for r in results:
            chosen = np.array([1.0 if c == model else 0.0
                               for c in r.model_chosen_ema])
            smoothed = np.convolve(chosen, np.ones(window) / window, mode="valid")
            fracs_per_seed.append(smoothed)
        mu = np.array(fracs_per_seed).mean(axis=0)
        ax_bot.plot(t_smooth, mu, color=model_colors[model], lw=1.5, ls="--",
                    label=f"EMA → {short_name(model)}")

    ax_bot.set_ylabel("Model Selection Fraction", fontsize=11)
    ax_bot.set_xlabel("Routing Step (t)", fontsize=11)
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.grid(True, alpha=0.2, ls=":")

    ncol = 2 if K <= 5 else 3
    ax_bot.legend(loc="upper right", fontsize=6, framealpha=0.75,
                  edgecolor="gray", ncol=ncol, columnspacing=1.0, handlelength=2.0)

    fig.align_ylabels([ax_top, ax_mid, ax_bot])

    for fmt in ("png", "pdf"):
        out = output_dir / f"catastrophic_{portfolio_label}.{fmt}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
    plt.close()


# ============================================================================
# REPORTING
# ============================================================================

def print_report(stats: Dict, models: List[str]):
    K = stats["K"]
    logger.info("\n" + "=" * 70)
    logger.info(f"CATASTROPHIC FAILURE DETECTION — K={K}")
    logger.info("=" * 70)

    logger.info(f"\n  Seeds: {stats['n_seeds']}")
    logger.info(f"  Detection rate: {stats['detection_rate'] * 100:.0f}%"
                f"  ({int(stats['detection_rate'] * stats['n_seeds'])}/{stats['n_seeds']})")

    if stats["reaction_mean"] is not None:
        logger.info(f"  Reaction time:  {stats['reaction_mean']:.1f} ± {stats['reaction_std']:.1f} steps"
                     f"  (median={stats['reaction_median']:.0f},"
                     f" range=[{stats['reaction_min']:.0f}, {stats['reaction_max']:.0f}])")

    logger.info(f"  Recovery rate:  {stats['recovery_rate'] * 100:.0f}%")

    logger.info(f"\n  {'Method':<30} {'Healthy':>10} {'Failure':>10} {'Recovery':>10}")
    logger.info("  " + "-" * 62)
    for method in ("oracle", "corralling", "ema", "static"):
        label = {"oracle": "Oracle", "corralling": "banditGPT",
                 "ema": "EMA Tracker",
                 "static": f"Static {short_name(FAILING_MODEL)}"}[method]
        h = f"{stats[f'{method}_healthy_mean']:.3f}±{stats[f'{method}_healthy_std']:.3f}"
        f = f"{stats[f'{method}_failure_mean']:.3f}±{stats[f'{method}_failure_std']:.3f}"
        r = f"{stats[f'{method}_recovery_mean']:.3f}±{stats[f'{method}_recovery_std']:.3f}"
        logger.info(f"  {label:<30} {h:>10} {f:>10} {r:>10}")

    logger.info(f"\n  Model selection during failure phase:")
    logger.info(f"  {'Model':<24} {'Corralling':>12} {'EMA':>12}")
    logger.info("  " + "-" * 50)
    for model in models:
        sn = short_name(model)
        c = stats[f"corralling_failure_{sn}_frac"]
        e = stats[f"ema_failure_{sn}_frac"]
        logger.info(f"  {sn:<24} {c:>11.1%} {e:>11.1%}")

    fail_corr = stats["corralling_failure_mean"]
    fail_ema = stats["ema_failure_mean"]
    delta = fail_corr - fail_ema
    logger.info(f"\n  KEY COMPARISON (failure phase):")
    logger.info(f"    banditGPT: {fail_corr:.3f}")
    logger.info(f"    EMA:       {fail_ema:.3f}")
    logger.info(f"    Δ:         {delta:+.3f}  {'banditGPT WINS' if delta > 0 else 'EMA wins'}")
    logger.info("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "=" * 70)
    logger.info("CATASTROPHIC FAILURE DETECTION — K=5 and K=10")
    logger.info("  Production hybrid router • 43-model warmup priors")
    logger.info("  Failing model: " + short_name(FAILING_MODEL))
    logger.info("=" * 70)

    output_dir = Path(__file__).parent / "results"
    all_stats = {}

    for portfolio_label, models in [("K5", PORTFOLIO_K5), ("K10", PORTFOLIO_K10)]:
        K = len(models)
        logger.info(f"\n{'='*70}")
        logger.info(f"PORTFOLIO: {portfolio_label} ({K} models)")
        logger.info(f"  Models: {', '.join(short_name(m) for m in models)}")
        logger.info("=" * 70)

        logger.info(f"\n  Validating K={K} warmup priors ...")
        validate_warmup_priors(models)

        t0 = time.time()
        logger.info(f"\n  Running {N_SEEDS}-seed experiment ({N_STEPS} steps each)...")
        results = run_all_seeds(models)
        elapsed = time.time() - t0
        logger.info(f"  Completed in {elapsed:.1f}s")

        logger.info("\n  Computing statistics...")
        stats = compute_statistics(results, models)
        all_stats[portfolio_label] = stats

        print_report(stats, models)

        logger.info("\n  Generating figure...")
        plot_figure(results, stats, models, portfolio_label, output_dir)

    # Summary comparison table
    logger.info("\n" + "=" * 70)
    logger.info("PORTFOLIO SIZE SCALING — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n  {'K':>4} {'banditGPT':>12} {'EMA':>12} {'Δ':>10} {'Detection':>12}")
    logger.info("  " + "-" * 52)
    for label in ["K5", "K10"]:
        s = all_stats[label]
        fc = s["corralling_failure_mean"]
        fe = s["ema_failure_mean"]
        delta = fc - fe
        det = s["detection_rate"]
        logger.info(f"  {s['K']:>4} {fc:>12.3f} {fe:>12.3f} {delta:>+10.3f} {det:>11.0%}")
    logger.info("=" * 70)

    # Save JSON results
    json_path = output_dir / "catastrophic_failure_results.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
