#!/usr/bin/env python3
"""
Synthetic Capability Study for banditGPT Router
================================================

Uses synthetic data with controllable parameters to characterize
when semantic transfer helps, hurts, or is neutral — using the
ACTUAL CostAwareLinUCBRouter from the production codebase.

Key controllable parameters:
  - quality_dispersion (σ): how different models' preference profiles are
  - similarity_accuracy (ρ): how well DNA similarity predicts θ similarity

Primary metric: cumulative regret during burn-in (cold-start cost).
Secondary metric: holdout eval reward (post-learning quality).

Produces:
  Part A: Dispersion sweep — regret vs σ for oracle/transfer/tabula
  Part B: Accuracy sweep — regret vs ρ
  Part C: 2D capability map — heatmap of (σ, ρ) → transfer advantage
  Part D: K scaling — transfer benefit vs portfolio size
  Part E: Learning curves — step-by-step reward in best/worst scenarios
"""

import sys
import copy
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -- Project imports: use the ACTUAL router --
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.router import CostAwareLinUCBRouter  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONTEXT_DIM = 33          # Match real router (32 PCA + 1 bias)
N_PRIOR = 500             # Samples to build existing-model priors
N_BURNIN = 200            # Online learning steps for new model
N_EVAL = 300              # Evaluation samples (noiseless expected reward)
N_SEEDS = 10
ALPHA_START = 1.0         # Moderate exploration (2.0 was too aggressive)
ALPHA_END = 0.1
N_EFF = 2.0               # Tuned from Part 4 sweep
COST_PENALTY = 0.3
REWARD_NOISE = 0.1

# Sweep ranges
DISPERSION_VALUES = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
ACCURACY_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K_VALUES = [3, 5, 10, 20, 40]

# Defaults for sweeps (when one dim is fixed)
DEFAULT_SIGMA = 0.15
DEFAULT_RHO = 0.8
DEFAULT_K = 10
DEFAULT_N_FAMILIES = 3


# =============================================================================
# SYNTHETIC ENVIRONMENT
# =============================================================================

@dataclass
class SyntheticModel:
    model_id: str
    family_id: int
    theta: np.ndarray         # True preference vector (d,)
    dna_embedding: np.ndarray  # DNA embedding (d,)
    cost: float


class SyntheticEnvironment:
    """
    Generates a synthetic routing environment with controllable parameters.

    Models are organized in families with shared preference profiles (θ).
    DNA embeddings correlate with true θ at controlled accuracy ρ.
    Rewards follow the LinUCB assumption: E[r|x,m] = clip(0.5 + x^T θ_m, 0, 1).

    Theta scaling: components ~ N(0, σ), NOT N(0, σ/√d). This ensures
    that x^T θ has std ≈ 1.4σ, giving reward variation visible above noise.
    """

    def __init__(
        self,
        n_models: int = 10,
        n_families: int = 3,
        context_dim: int = CONTEXT_DIM,
        quality_dispersion: float = 0.15,
        within_family_noise: float = 0.02,
        similarity_accuracy: float = 0.8,
        reward_noise: float = REWARD_NOISE,
        cost_range: Tuple[float, float] = (0.0001, 0.01),
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        self.reward_noise = reward_noise
        self.models = self._generate_models(
            n_models, n_families, context_dim,
            quality_dispersion, within_family_noise,
            similarity_accuracy, cost_range,
        )
        self.model_ids = [m.model_id for m in self.models]
        self.model_map = {m.model_id: m for m in self.models}

    def _generate_models(self, n_models, n_families, dim,
                         sigma_between, sigma_within, rho, cost_range):
        models = []
        models_per_family = max(2, n_models // n_families)

        # Family base thetas — NO /sqrt(dim) for meaningful reward differences
        family_thetas = [self.rng.randn(dim) * sigma_between for _ in range(n_families)]

        for j in range(n_families):
            n_in_fam = models_per_family if j < n_families - 1 else n_models - len(models)
            for i in range(n_in_fam):
                if len(models) >= n_models:
                    break
                theta = family_thetas[j] + self.rng.randn(dim) * sigma_within

                # DNA embedding: correlated with theta at level ρ
                theta_unit = theta / (np.linalg.norm(theta) + 1e-12)
                z = self.rng.randn(dim)
                z = z - np.dot(z, theta_unit) * theta_unit  # orthogonalize
                z = z / (np.linalg.norm(z) + 1e-12)
                dna = rho * theta_unit + np.sqrt(max(0, 1 - rho ** 2)) * z

                frac = j / max(n_families - 1, 1)
                log_c = np.log(cost_range[0]) + (np.log(cost_range[1]) - np.log(cost_range[0])) * (
                    frac + self.rng.uniform(-0.15, 0.15)
                )
                cost = float(np.clip(np.exp(log_c), cost_range[0], cost_range[1]))

                models.append(SyntheticModel(
                    model_id=f"fam{j}/model-{chr(65 + i)}",
                    family_id=j,
                    theta=theta,
                    dna_embedding=dna,
                    cost=cost,
                ))
        return models[:n_models]

    def sample_context(self) -> np.ndarray:
        x = self.rng.randn(self.context_dim - 1) / np.sqrt(self.context_dim)
        return np.append(x, 1.0)

    def get_reward(self, model_id: str, context: np.ndarray) -> float:
        m = self.model_map[model_id]
        expected = 0.5 + context @ m.theta
        noise = self.rng.randn() * self.reward_noise
        return float(np.clip(expected + noise, 0.0, 1.0))

    def get_expected_reward(self, model_id: str, context: np.ndarray) -> float:
        m = self.model_map[model_id]
        return float(np.clip(0.5 + context @ m.theta, 0.0, 1.0))

    def get_oracle_model(self, context: np.ndarray) -> Tuple[str, float]:
        best_m = max(self.model_ids,
                     key=lambda mid: self.get_expected_reward(mid, context))
        return best_m, self.get_expected_reward(best_m, context)

    def dna_cosine(self, m1: str, m2: str) -> float:
        d1, d2 = self.model_map[m1].dna_embedding, self.model_map[m2].dna_embedding
        return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))

    def find_dna_neighbor(self, model_id: str, candidates: List[str]) -> Tuple[str, float]:
        best_id, best_sim = None, -1.0
        for c in candidates:
            if c == model_id:
                continue
            sim = self.dna_cosine(model_id, c)
            if sim > best_sim:
                best_sim = sim
                best_id = c
        return best_id, best_sim


# =============================================================================
# PRE-GENERATE DATA (shared across modes for fair comparison)
# =============================================================================

def pregenerate_data(env: SyntheticEnvironment, n_burnin: int, n_eval: int):
    """Generate contexts and per-model rewards ONCE, shared across all modes."""
    burnin_contexts = [env.sample_context() for _ in range(n_burnin)]
    eval_contexts = [env.sample_context() for _ in range(n_eval)]

    # Noisy rewards for burn-in (what the router would observe)
    burnin_rewards = {}
    for m in env.model_ids:
        burnin_rewards[m] = [env.get_reward(m, x) for x in burnin_contexts]

    # Noiseless expected rewards for evaluation
    eval_expected = {}
    for m in env.model_ids:
        eval_expected[m] = [env.get_expected_reward(m, x) for x in eval_contexts]

    # True selection oracle: best model for each context
    oracle_burnin = []
    for t, x in enumerate(burnin_contexts):
        best_m, best_r = env.get_oracle_model(x)
        oracle_burnin.append((best_m, best_r))

    oracle_eval = []
    for t, x in enumerate(eval_contexts):
        best_m, best_r = env.get_oracle_model(x)
        oracle_eval.append((best_m, best_r))

    return {
        "burnin_contexts": burnin_contexts,
        "eval_contexts": eval_contexts,
        "burnin_rewards": burnin_rewards,
        "eval_expected": eval_expected,
        "oracle_burnin": oracle_burnin,
        "oracle_eval": oracle_eval,
    }


# =============================================================================
# BUILD PRIORS
# =============================================================================

def build_priors(env: SyntheticEnvironment, models: List[str], n_samples: int) -> Dict:
    """Build warmup priors for existing models from synthetic data."""
    dim = env.context_dim
    A = {m: np.eye(dim) for m in models}
    b = {m: np.zeros(dim) for m in models}
    for m in models:
        for _ in range(n_samples):
            x = env.sample_context()
            r = env.get_reward(m, x)
            A[m] += np.outer(x, x)
            b[m] += r * x
    return {"A": A, "b": b, "context_dim": dim}


# =============================================================================
# TRIAL RUNNER: uses real CostAwareLinUCBRouter with pre-generated data
# =============================================================================

def run_trial(
    env: SyntheticEnvironment,
    existing_models: List[str],
    new_model: str,
    mode: str,
    priors: Dict,
    data: Dict,
    trial_seed: int,
) -> Dict:
    """
    Run one onboarding trial using the REAL CostAwareLinUCBRouter.

    All modes receive the SAME contexts and rewards (from pre-generated data).
    Modes:
      - "transfer":        Current A=n_eff*I, b=n_eff*theta (reduces exploration)
      - "transfer_b_only": A=I, b=theta_neighbor (preserves full exploration bonus)
      - "transfer_full_a": A=scaled A_neighbor, b=scaled b_neighbor (preserves structure)
      - "random_transfer":  Random neighbor, A=n_eff*I, b=n_eff*theta
      - "tabula_rasa":     A=I, b=0 (baseline)
    """
    dim = env.context_dim
    all_models = existing_models + [new_model]
    rng = np.random.RandomState(trial_seed)

    # Normalize costs
    costs_raw = {m: env.model_map[m].cost for m in all_models}
    c_min, c_max = min(costs_raw.values()), max(costs_raw.values())
    c_range = c_max - c_min if c_max > c_min else 1.0
    costs_norm = {m: {"normalized_cost": (costs_raw[m] - c_min) / c_range} for m in all_models}

    # Prepare priors
    trial_priors = copy.deepcopy(priors)
    if new_model not in trial_priors["A"]:
        trial_priors["A"][new_model] = np.eye(dim)
        trial_priors["b"][new_model] = np.zeros(dim)

    # Helper: get DNA neighbor's theta
    def _get_dna_neighbor_theta():
        neighbor, sim = env.find_dna_neighbor(new_model, existing_models)
        if not neighbor:
            return None, None, None
        A_inv_n = np.linalg.inv(priors["A"][neighbor])
        theta_n = A_inv_n @ priors["b"][neighbor]
        return neighbor, sim, theta_n

    if mode == "transfer":
        neighbor, sim, theta_n = _get_dna_neighbor_theta()
        if theta_n is not None:
            trial_priors["A"][new_model] = N_EFF * np.eye(dim)
            trial_priors["b"][new_model] = N_EFF * theta_n
    elif mode == "transfer_b_only":
        # KEY INSIGHT: keep A=I (same exploration bonus as tabula rasa)
        # but inject directional prior via b=theta_neighbor
        neighbor, sim, theta_n = _get_dna_neighbor_theta()
        if theta_n is not None:
            trial_priors["A"][new_model] = np.eye(dim)
            trial_priors["b"][new_model] = theta_n
    elif mode == "transfer_full_a":
        # Transfer scaled copy of neighbor's full A and b (preserves covariance shape)
        neighbor, sim, theta_n = _get_dna_neighbor_theta()
        if neighbor is not None:
            A_nb = priors["A"][neighbor]
            b_nb = priors["b"][neighbor]
            n_neighbor = max(np.trace(A_nb) / dim, 1.0)
            scale = N_EFF / n_neighbor
            trial_priors["A"][new_model] = A_nb * scale
            trial_priors["b"][new_model] = b_nb * scale
    elif mode == "random_transfer":
        rand_idx = rng.randint(len(existing_models))
        rand_neighbor = existing_models[rand_idx]
        A_inv_r = np.linalg.inv(priors["A"][rand_neighbor])
        theta_rand = A_inv_r @ priors["b"][rand_neighbor]
        trial_priors["A"][new_model] = N_EFF * np.eye(dim)
        trial_priors["b"][new_model] = N_EFF * theta_rand
    elif mode == "tabula_rasa":
        trial_priors["A"][new_model] = np.eye(dim)
        trial_priors["b"][new_model] = np.zeros(dim)

    # Instantiate the REAL router
    router = CostAwareLinUCBRouter(
        models=all_models,
        warmup_priors=trial_priors,
        model_costs=costs_norm,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,
        cost_penalty=COST_PENALTY,
    )

    n_burnin = len(data["burnin_contexts"])
    n_eval = len(data["eval_contexts"])

    # Burn-in: online learning with pre-generated contexts/rewards
    burnin_step_rewards = []
    for t in range(n_burnin):
        x = data["burnin_contexts"][t]
        sel = router.select_model(x, total_steps=n_burnin)
        r = data["burnin_rewards"][sel][t]
        router.update(x, sel, r)
        burnin_step_rewards.append(r)

    # Evaluation: no updates, noiseless expected rewards
    eval_rewards = []
    for t in range(n_eval):
        x = data["eval_contexts"][t]
        sel = router.select_model(x, total_steps=n_burnin)
        eval_rewards.append(data["eval_expected"][sel][t])

    # Compute oracle regret during burn-in
    oracle_burnin_rewards = [data["oracle_burnin"][t][1] for t in range(n_burnin)]
    regret_per_step = [oracle_burnin_rewards[t] - burnin_step_rewards[t] for t in range(n_burnin)]

    return {
        "burnin_reward": float(np.mean(burnin_step_rewards)),
        "burnin_reward_early": float(np.mean(burnin_step_rewards[:50])),
        "burnin_reward_late": float(np.mean(burnin_step_rewards[-50:])),
        "cumulative_regret": float(np.sum(regret_per_step)),
        "eval_reward": float(np.mean(eval_rewards)),
        "burnin_curve": burnin_step_rewards,
    }


# =============================================================================
# CELL RUNNER: one (σ, ρ, K) configuration
# =============================================================================

def run_cell(
    sigma: float,
    rho: float,
    K: int = DEFAULT_K,
    n_families: int = DEFAULT_N_FAMILIES,
    n_seeds: int = N_SEEDS,
    n_burnin: int = N_BURNIN,
    n_eval: int = N_EVAL,
    base_seed: int = 42,
    return_curves: bool = False,
) -> Dict:
    modes = ["transfer", "transfer_b_only", "transfer_full_a", "random_transfer", "tabula_rasa"]
    all_results = {m: {"burnin_reward": [], "eval_reward": [], "cumulative_regret": [],
                       "burnin_reward_early": [], "burnin_reward_late": []}
                   for m in modes}
    all_curves = {m: [] for m in modes} if return_curves else None
    oracle_eval_rewards = []
    oracle_burnin_rewards = []

    for seed_idx in range(n_seeds):
        cell_seed = base_seed + seed_idx * 1000
        env = SyntheticEnvironment(
            n_models=K, n_families=n_families,
            context_dim=CONTEXT_DIM,
            quality_dispersion=sigma,
            similarity_accuracy=rho,
            seed=cell_seed,
        )

        existing = env.model_ids[:-1]
        new_model = env.model_ids[-1]

        # Build priors for existing models
        priors = build_priors(env, existing, N_PRIOR)
        priors["context_dim"] = CONTEXT_DIM

        # Pre-generate data ONCE per seed (shared across modes)
        data = pregenerate_data(env, n_burnin, n_eval)

        # Selection oracle (true ceiling, no router)
        oracle_eval_rewards.append(np.mean([r for _, r in data["oracle_eval"]]))
        oracle_burnin_rewards.append(np.mean([r for _, r in data["oracle_burnin"]]))

        for mode in modes:
            res = run_trial(env, existing, new_model, mode, priors, data,
                            trial_seed=cell_seed + hash(mode) % 10000)
            for key in ["burnin_reward", "eval_reward", "cumulative_regret",
                        "burnin_reward_early", "burnin_reward_late"]:
                all_results[mode][key].append(res[key])
            if return_curves:
                all_curves[mode].append(res["burnin_curve"])

    output = {}
    for mode in modes:
        for key in all_results[mode]:
            vals = all_results[mode][key]
            if mode not in output:
                output[mode] = {}
            output[mode][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci95": float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            }

    output["selection_oracle"] = {
        "eval_reward": {"mean": float(np.mean(oracle_eval_rewards)),
                        "std": float(np.std(oracle_eval_rewards, ddof=1)),
                        "ci95": float(1.96 * np.std(oracle_eval_rewards, ddof=1) / np.sqrt(len(oracle_eval_rewards)))},
        "burnin_reward": {"mean": float(np.mean(oracle_burnin_rewards)),
                          "std": float(np.std(oracle_burnin_rewards, ddof=1)),
                          "ci95": float(1.96 * np.std(oracle_burnin_rewards, ddof=1) / np.sqrt(len(oracle_burnin_rewards)))},
    }

    if return_curves:
        output["_curves"] = all_curves
    return output


# =============================================================================
# HELPER: extract primary metric for comparisons
# =============================================================================

def transfer_advantage(cell: Dict, mode: str = "transfer_b_only", metric: str = "eval_reward") -> float:
    return cell[mode][metric]["mean"] - cell["tabula_rasa"][metric]["mean"]


def regret_reduction(cell: Dict, mode: str = "transfer_b_only") -> float:
    """Fraction of tabula-rasa regret eliminated by a transfer mode."""
    tr_reg = cell[mode]["cumulative_regret"]["mean"]
    tb_reg = cell["tabula_rasa"]["cumulative_regret"]["mean"]
    if abs(tb_reg) < 1e-6:
        return 0.0
    return (tb_reg - tr_reg) / abs(tb_reg)


# =============================================================================
# PART A: QUALITY DISPERSION SWEEP
# =============================================================================

def run_part_a(rho: float = DEFAULT_RHO) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"PART A: QUALITY DISPERSION SWEEP (ρ={rho})")
    logger.info("=" * 70)
    logger.info(f"  {'σ':>6}  {'Oracle':>8}  {'b-only':>8}  {'Current':>8}  {'FullA':>8}  {'Tabula':>8}  "
                f"{'Δ(b-only)':>10}  {'RegRed%':>8}")
    logger.info("  " + "-" * 90)

    results = {}
    for sigma in DISPERSION_VALUES:
        t0 = time.time()
        cell = run_cell(sigma=sigma, rho=rho)
        dt = time.time() - t0

        adv = transfer_advantage(cell)
        rr = regret_reduction(cell) * 100
        logger.info(
            f"  {sigma:<6.3f}  {cell['selection_oracle']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer_b_only']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer_full_a']['eval_reward']['mean']:>8.4f}  "
            f"{cell['tabula_rasa']['eval_reward']['mean']:>8.4f}  "
            f"{adv:>+10.4f}  "
            f"{rr:>+7.1f}%  ({dt:.1f}s)"
        )
        results[sigma] = cell
    return results


def plot_part_a(results: Dict, output_dir: Path):
    sigmas = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    eval_style = [
        ("selection_oracle", "eval_reward", "Oracle (best model)", "#f1c40f", "--"),
        ("transfer_b_only", "eval_reward", "b-only Transfer", "#2ecc71", "-"),
        ("transfer_full_a", "eval_reward", "Full-A Transfer", "#9b59b6", "-"),
        ("transfer", "eval_reward", "Current Transfer", "#3498db", "-."),
        ("random_transfer", "eval_reward", "Random Transfer", "#95a5a6", "-."),
        ("tabula_rasa", "eval_reward", "Tabula Rasa", "#e74c3c", ":"),
    ]

    for mode, metric, label, color, ls in eval_style:
        means = [results[s][mode][metric]["mean"] for s in sigmas]
        ci95s = [results[s][mode][metric]["ci95"] for s in sigmas]
        means, ci95s = np.array(means), np.array(ci95s)
        ax1.plot(sigmas, means, ls, color=color, lw=2.2, label=label, marker="o", markersize=5)
        ax1.fill_between(sigmas, means - ci95s, means + ci95s, color=color, alpha=0.10)

    ax1.set_xlabel("Quality Dispersion (σ)", fontsize=12)
    ax1.set_ylabel("Eval Reward (noiseless)", fontsize=12)
    ax1.set_title("Eval Reward vs. Dispersion", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    regret_style = [
        ("transfer_b_only", "b-only Transfer", "#2ecc71", "-"),
        ("transfer_full_a", "Full-A Transfer", "#9b59b6", "-"),
        ("transfer", "Current Transfer", "#3498db", "-."),
        ("random_transfer", "Random Transfer", "#95a5a6", "-."),
        ("tabula_rasa", "Tabula Rasa", "#e74c3c", ":"),
    ]

    for mode, label, color, ls in regret_style:
        means = [results[s][mode]["cumulative_regret"]["mean"] for s in sigmas]
        ci95s = [results[s][mode]["cumulative_regret"]["ci95"] for s in sigmas]
        means, ci95s = np.array(means), np.array(ci95s)
        ax2.plot(sigmas, means, ls, color=color, lw=2.2, label=label, marker="o", markersize=5)
        ax2.fill_between(sigmas, means - ci95s, means + ci95s, color=color, alpha=0.10)

    ax2.set_xlabel("Quality Dispersion (σ)", fontsize=12)
    ax2.set_ylabel("Cumulative Regret (burn-in)", fontsize=12)
    ax2.set_title("Cold-Start Regret vs. Dispersion", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Part A: Quality Dispersion Sweep (ρ={DEFAULT_RHO}, K={DEFAULT_K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "part_a_dispersion_sweep.png", dpi=200)
    fig.savefig(output_dir / "part_a_dispersion_sweep.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part A: {output_dir / 'part_a_dispersion_sweep.png'}")


# =============================================================================
# PART B: SIMILARITY ACCURACY SWEEP
# =============================================================================

def run_part_b(sigma: float = DEFAULT_SIGMA) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"PART B: SIMILARITY ACCURACY SWEEP (σ={sigma})")
    logger.info("=" * 70)
    logger.info(f"  {'ρ':>5}  {'Oracle':>8}  {'b-only':>8}  {'Current':>8}  {'FullA':>8}  {'Tabula':>8}  "
                f"{'Δ(b-only)':>10}  {'RegRed%':>8}")
    logger.info("  " + "-" * 90)

    results = {}
    for rho in ACCURACY_VALUES:
        t0 = time.time()
        cell = run_cell(sigma=sigma, rho=rho)
        dt = time.time() - t0

        adv = transfer_advantage(cell)
        rr = regret_reduction(cell) * 100
        logger.info(
            f"  {rho:<5.2f}  {cell['selection_oracle']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer_b_only']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer']['eval_reward']['mean']:>8.4f}  "
            f"{cell['transfer_full_a']['eval_reward']['mean']:>8.4f}  "
            f"{cell['tabula_rasa']['eval_reward']['mean']:>8.4f}  "
            f"{adv:>+10.4f}  {rr:>+7.1f}%  ({dt:.1f}s)"
        )
        results[rho] = cell
    return results


def plot_part_b(results: Dict, output_dir: Path):
    rhos = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    eval_style = [
        ("selection_oracle", "eval_reward", "Oracle", "#f1c40f", "--"),
        ("transfer_b_only", "eval_reward", "b-only Transfer", "#2ecc71", "-"),
        ("transfer_full_a", "eval_reward", "Full-A Transfer", "#9b59b6", "-"),
        ("transfer", "eval_reward", "Current Transfer", "#3498db", "-."),
        ("random_transfer", "eval_reward", "Random Transfer", "#95a5a6", "-."),
        ("tabula_rasa", "eval_reward", "Tabula Rasa", "#e74c3c", ":"),
    ]

    for mode, metric, label, color, ls in eval_style:
        means = [results[r][mode][metric]["mean"] for r in rhos]
        ci95s = [results[r][mode][metric]["ci95"] for r in rhos]
        means, ci95s = np.array(means), np.array(ci95s)
        ax1.plot(rhos, means, ls, color=color, lw=2.2, label=label, marker="o", markersize=5)
        ax1.fill_between(rhos, means - ci95s, means + ci95s, color=color, alpha=0.10)

    ax1.set_xlabel("Similarity Accuracy (ρ)", fontsize=12)
    ax1.set_ylabel("Eval Reward", fontsize=12)
    ax1.set_title("Eval Reward vs. Accuracy", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    regret_style = [
        ("transfer_b_only", "b-only Transfer", "#2ecc71", "-"),
        ("transfer_full_a", "Full-A Transfer", "#9b59b6", "-"),
        ("transfer", "Current Transfer", "#3498db", "-."),
        ("random_transfer", "Random Transfer", "#95a5a6", "-."),
        ("tabula_rasa", "Tabula Rasa", "#e74c3c", ":"),
    ]

    for mode, label, color, ls in regret_style:
        means = [results[r][mode]["cumulative_regret"]["mean"] for r in rhos]
        ci95s = [results[r][mode]["cumulative_regret"]["ci95"] for r in rhos]
        means, ci95s = np.array(means), np.array(ci95s)
        ax2.plot(rhos, means, ls, color=color, lw=2.2, label=label, marker="o", markersize=5)
        ax2.fill_between(rhos, means - ci95s, means + ci95s, color=color, alpha=0.10)

    ax2.set_xlabel("Similarity Accuracy (ρ)", fontsize=12)
    ax2.set_ylabel("Cumulative Regret", fontsize=12)
    ax2.set_title("Regret vs. Accuracy", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Part B: Similarity Accuracy Sweep (σ={DEFAULT_SIGMA}, K={DEFAULT_K})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "part_b_accuracy_sweep.png", dpi=200)
    fig.savefig(output_dir / "part_b_accuracy_sweep.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part B: {output_dir / 'part_b_accuracy_sweep.png'}")


# =============================================================================
# PART C: 2D CAPABILITY MAP
# =============================================================================

def run_part_c() -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("PART C: 2D CAPABILITY MAP")
    logger.info("=" * 70)

    results = {}
    total = len(DISPERSION_VALUES) * len(ACCURACY_VALUES)
    done = 0
    for sigma in DISPERSION_VALUES:
        for rho in ACCURACY_VALUES:
            t0 = time.time()
            cell = run_cell(sigma=sigma, rho=rho, n_seeds=5)
            dt = time.time() - t0
            done += 1
            adv_bonly = transfer_advantage(cell, mode="transfer_b_only")
            adv_current = transfer_advantage(cell, mode="transfer")
            results[(sigma, rho)] = cell
            if done % 10 == 0 or done == total:
                logger.info(f"  [{done}/{total}] σ={sigma:.3f} ρ={rho:.2f}  "
                            f"Δ(b-only)={adv_bonly:+.4f}  Δ(current)={adv_current:+.4f}  ({dt:.1f}s)")
    return results


def plot_part_c(results: Dict, output_dir: Path):
    sigmas = sorted(set(k[0] for k in results.keys()))
    rhos = sorted(set(k[1] for k in results.keys()))

    # Build advantage matrices for each transfer variant
    variants = [
        ("transfer_b_only", "b-only Transfer"),
        ("transfer", "Current Transfer"),
        ("transfer_full_a", "Full-A Transfer"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    for col_idx, (mode, mode_label) in enumerate(variants):
        Z_eval = np.zeros((len(rhos), len(sigmas)))
        Z_regret = np.zeros((len(rhos), len(sigmas)))
        for i, rho in enumerate(rhos):
            for j, sigma in enumerate(sigmas):
                cell = results.get((sigma, rho))
                if cell:
                    Z_eval[i, j] = transfer_advantage(cell, mode=mode)
                    Z_regret[i, j] = regret_reduction(cell, mode=mode) * 100

        # Row 1: Eval reward advantage
        ax1 = axes[0, col_idx]
        vmax = max(abs(Z_eval.min()), abs(Z_eval.max()), 0.001)
        norm1 = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im1 = ax1.imshow(Z_eval, aspect="auto", origin="lower", cmap="RdYlGn", norm=norm1,
                         extent=[-0.5, len(sigmas) - 0.5, -0.5, len(rhos) - 0.5])
        ax1.set_xticks(range(len(sigmas)))
        ax1.set_xticklabels([f"{s:.2f}" for s in sigmas], fontsize=7)
        ax1.set_yticks(range(len(rhos)))
        ax1.set_yticklabels([f"{r:.1f}" for r in rhos], fontsize=7)
        ax1.set_xlabel("Quality Dispersion (σ)", fontsize=10)
        if col_idx == 0:
            ax1.set_ylabel("Similarity Accuracy (ρ)", fontsize=10)
        ax1.set_title(f"Eval Reward: {mode_label} − Tabula Rasa", fontsize=10)
        for i, rho in enumerate(rhos):
            for j, sigma in enumerate(sigmas):
                color = "white" if abs(Z_eval[i, j]) > vmax * 0.5 else "black"
                ax1.text(j, i, f"{Z_eval[i, j]:+.3f}", ha="center", va="center",
                         fontsize=5.5, color=color, fontweight="bold")
        fig.colorbar(im1, ax=ax1, shrink=0.75)

        # Row 2: Regret reduction %
        ax2 = axes[1, col_idx]
        vmax2 = max(abs(Z_regret.min()), abs(Z_regret.max()), 1.0)
        norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
        im2 = ax2.imshow(Z_regret, aspect="auto", origin="lower", cmap="RdYlGn", norm=norm2,
                         extent=[-0.5, len(sigmas) - 0.5, -0.5, len(rhos) - 0.5])
        ax2.set_xticks(range(len(sigmas)))
        ax2.set_xticklabels([f"{s:.2f}" for s in sigmas], fontsize=7)
        ax2.set_yticks(range(len(rhos)))
        ax2.set_yticklabels([f"{r:.1f}" for r in rhos], fontsize=7)
        ax2.set_xlabel("Quality Dispersion (σ)", fontsize=10)
        if col_idx == 0:
            ax2.set_ylabel("Similarity Accuracy (ρ)", fontsize=10)
        ax2.set_title(f"Regret Reduction: {mode_label} vs Tabula Rasa", fontsize=10)
        for i, rho in enumerate(rhos):
            for j, sigma in enumerate(sigmas):
                color = "white" if abs(Z_regret[i, j]) > vmax2 * 0.5 else "black"
                ax2.text(j, i, f"{Z_regret[i, j]:+.0f}%", ha="center", va="center",
                         fontsize=5.5, color=color, fontweight="bold")
        fig.colorbar(im2, ax=ax2, shrink=0.75)

        # Mark real-data position
        for ax in [ax1, ax2]:
            real_sigma_idx = np.interp(0.02, sigmas, range(len(sigmas)))
            real_rho_idx = np.interp(0.2, rhos, range(len(rhos)))
            ax.plot(real_sigma_idx, real_rho_idx, marker="*", color="magenta",
                    markersize=14, markeredgecolor="black", markeredgewidth=1.0, zorder=10)

    fig.suptitle("Part C: Capability Map — Transfer Variant Comparison (K=10)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "part_c_capability_map.png", dpi=200)
    fig.savefig(output_dir / "part_c_capability_map.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part C: {output_dir / 'part_c_capability_map.png'}")


# =============================================================================
# PART D: K SCALING
# =============================================================================

def run_part_d(sigma: float = DEFAULT_SIGMA, rho: float = DEFAULT_RHO) -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"PART D: K SCALING (σ={sigma}, ρ={rho})")
    logger.info("=" * 70)

    results = {}
    for K in K_VALUES:
        n_fam = max(2, K // 3)
        t0 = time.time()
        cell = run_cell(sigma=sigma, rho=rho, K=K, n_families=n_fam)
        dt = time.time() - t0

        adv_bonly = transfer_advantage(cell, mode="transfer_b_only")
        adv_current = transfer_advantage(cell, mode="transfer")
        rr_bonly = regret_reduction(cell, mode="transfer_b_only") * 100
        rr_current = regret_reduction(cell, mode="transfer") * 100
        logger.info(
            f"  K={K:<3}  Oracle={cell['selection_oracle']['eval_reward']['mean']:.4f}  "
            f"b-only={cell['transfer_b_only']['eval_reward']['mean']:.4f}  "
            f"Current={cell['transfer']['eval_reward']['mean']:.4f}  "
            f"Tabula={cell['tabula_rasa']['eval_reward']['mean']:.4f}  "
            f"Δ(b-only)={adv_bonly:+.4f}  RegRed(b-only)={rr_bonly:+.1f}%  ({dt:.1f}s)"
        )
        results[K] = cell
    return results


def plot_part_d(results: Dict, output_dir: Path):
    ks = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    eval_style = [
        ("selection_oracle", "eval_reward", "Oracle", "#f1c40f", "--"),
        ("transfer_b_only", "eval_reward", "b-only Transfer", "#2ecc71", "-"),
        ("transfer_full_a", "eval_reward", "Full-A Transfer", "#9b59b6", "-"),
        ("transfer", "eval_reward", "Current Transfer", "#3498db", "-."),
        ("random_transfer", "eval_reward", "Random Transfer", "#95a5a6", "-."),
        ("tabula_rasa", "eval_reward", "Tabula Rasa", "#e74c3c", ":"),
    ]

    for mode, metric, label, color, ls in eval_style:
        means = [results[k][mode][metric]["mean"] for k in ks]
        ci95s = [results[k][mode][metric]["ci95"] for k in ks]
        means, ci95s = np.array(means), np.array(ci95s)
        ax1.plot(ks, means, ls, color=color, lw=2.2, label=label, marker="o", markersize=6)
        ax1.fill_between(ks, means - ci95s, means + ci95s, color=color, alpha=0.10)

    ax1.set_xlabel("Portfolio Size (K)", fontsize=12)
    ax1.set_ylabel("Eval Reward", fontsize=12)
    ax1.set_title("Eval Reward vs. K", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Grouped bar chart: regret reduction for each transfer variant
    transfer_modes = [
        ("transfer_b_only", "b-only", "#2ecc71"),
        ("transfer_full_a", "Full-A", "#9b59b6"),
        ("transfer", "Current", "#3498db"),
    ]
    n_groups = len(ks)
    n_bars = len(transfer_modes)
    bar_width = 0.25
    x = np.arange(n_groups)

    for bar_idx, (mode, label, color) in enumerate(transfer_modes):
        rr_vals = [regret_reduction(results[k], mode=mode) * 100 for k in ks]
        ax2.bar(x + bar_idx * bar_width, rr_vals, bar_width, label=label,
                color=color, edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("Portfolio Size (K)", fontsize=12)
    ax2.set_ylabel("Regret Reduction %", fontsize=12)
    ax2.set_title("Transfer Benefit by Variant (>0 = helps)", fontsize=12)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Part D: K Scaling (σ={DEFAULT_SIGMA}, ρ={DEFAULT_RHO})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "part_d_k_scaling.png", dpi=200)
    fig.savefig(output_dir / "part_d_k_scaling.pdf")
    plt.close(fig)
    logger.info(f"  Saved Part D: {output_dir / 'part_d_k_scaling.png'}")


# =============================================================================
# PART E: LEARNING CURVES
# =============================================================================

def run_part_e() -> Dict:
    logger.info("\n" + "=" * 70)
    logger.info("PART E: LEARNING CURVES")
    logger.info("=" * 70)

    scenarios = {
        "Favorable\n(σ=0.2, ρ=0.9)": {"sigma": 0.2, "rho": 0.9},
        "Moderate\n(σ=0.1, ρ=0.5)": {"sigma": 0.1, "rho": 0.5},
        "Unfavorable\n(σ=0.02, ρ=0.2)": {"sigma": 0.02, "rho": 0.2},
    }

    results = {}
    for label, params in scenarios.items():
        t0 = time.time()
        cell = run_cell(
            sigma=params["sigma"], rho=params["rho"],
            n_burnin=500, n_eval=N_EVAL, n_seeds=N_SEEDS,
            return_curves=True,
        )
        dt = time.time() - t0
        results[label] = cell
        adv_bonly = transfer_advantage(cell, mode="transfer_b_only")
        adv_current = transfer_advantage(cell, mode="transfer")
        rr_bonly = regret_reduction(cell, mode="transfer_b_only") * 100
        rr_current = regret_reduction(cell, mode="transfer") * 100
        logger.info(f"  {label.replace(chr(10), ' ')}:  "
                    f"Δ(b-only)={adv_bonly:+.4f}  RR(b-only)={rr_bonly:+.1f}%  "
                    f"Δ(current)={adv_current:+.4f}  RR(current)={rr_current:+.1f}%  ({dt:.1f}s)")

    return results


def plot_part_e(results: Dict, output_dir: Path):
    scenarios = list(results.keys())
    n_panels = len(scenarios)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    curve_style = [
        ("transfer_b_only", "#2ecc71", "-", "b-only Transfer"),
        ("transfer_full_a", "#9b59b6", "-", "Full-A Transfer"),
        ("transfer", "#3498db", "-.", "Current Transfer"),
        ("random_transfer", "#95a5a6", "-.", "Random Transfer"),
        ("tabula_rasa", "#e74c3c", ":", "Tabula Rasa"),
    ]

    for ax, label in zip(axes, scenarios):
        curves = results[label].get("_curves", {})
        for mode, color, ls, mlabel in curve_style:
            if mode not in curves or not curves[mode]:
                continue
            all_c = np.array(curves[mode])
            mean_c = np.mean(all_c, axis=0)
            cum_avg = np.cumsum(mean_c) / np.arange(1, len(mean_c) + 1)
            ax.plot(cum_avg, ls, color=color, lw=1.8, label=mlabel, alpha=0.9)

        oracle_r = results[label]["selection_oracle"]["burnin_reward"]["mean"]
        ax.axhline(oracle_r, color="#f1c40f", ls="--", lw=1.5, label="Oracle", alpha=0.8)

        ax.set_xlabel("Step", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Cumulative Avg Reward", fontsize=11)
    fig.suptitle("Part E: Learning Curves — Convergence Speed", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "part_e_learning_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "part_e_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved Part E: {output_dir / 'part_e_learning_curves.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("SYNTHETIC CAPABILITY STUDY — using real CostAwareLinUCBRouter")
    logger.info("=" * 70)
    logger.info(f"  Context dim: {CONTEXT_DIM}, Default K: {DEFAULT_K}")
    logger.info(f"  N_prior={N_PRIOR}, N_burnin={N_BURNIN}, N_eval={N_EVAL}, N_seeds={N_SEEDS}")
    logger.info(f"  n_eff={N_EFF}, alpha={ALPHA_START}→{ALPHA_END}, cost_penalty={COST_PENALTY}")
    logger.info(f"  Default σ={DEFAULT_SIGMA}, Default ρ={DEFAULT_RHO}")
    t_total = time.time()

    part_a = run_part_a()
    plot_part_a(part_a, RESULTS_DIR)

    part_b = run_part_b()
    plot_part_b(part_b, RESULTS_DIR)

    part_c = run_part_c()
    plot_part_c(part_c, RESULTS_DIR)

    part_d = run_part_d()
    plot_part_d(part_d, RESULTS_DIR)

    part_e = run_part_e()
    plot_part_e(part_e, RESULTS_DIR)

    # Save JSON
    output_data = {
        "metadata": {
            "description": "Synthetic capability study using real CostAwareLinUCBRouter",
            "context_dim": CONTEXT_DIM, "n_prior": N_PRIOR,
            "n_burnin": N_BURNIN, "n_eval": N_EVAL, "n_seeds": N_SEEDS,
            "n_eff": N_EFF, "alpha_start": ALPHA_START, "alpha_end": ALPHA_END,
            "cost_penalty": COST_PENALTY, "default_sigma": DEFAULT_SIGMA,
            "default_rho": DEFAULT_RHO,
        },
    }
    with open(RESULTS_DIR / "synthetic_study_results.json", "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Final Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY — TRANSFER VARIANT COMPARISON")
    logger.info("=" * 70)

    transfer_modes_summary = [
        ("transfer_b_only", "b-only"),
        ("transfer_full_a", "Full-A"),
        ("transfer", "Current"),
    ]

    logger.info(f"\n  Part A — Dispersion Sweep (ρ={DEFAULT_RHO}):")
    header = f"    {'σ':>6}  {'Oracle':>8}  {'Tabula':>8}"
    for _, lbl in transfer_modes_summary:
        header += f"  {lbl+' Δ':>10}  {lbl+' RR%':>10}"
    logger.info(header)
    for sigma in DISPERSION_VALUES:
        c = part_a[sigma]
        line = (f"    {sigma:>6.3f}  {c['selection_oracle']['eval_reward']['mean']:>8.4f}  "
                f"{c['tabula_rasa']['eval_reward']['mean']:>8.4f}")
        for mode, _ in transfer_modes_summary:
            adv = transfer_advantage(c, mode=mode)
            rr = regret_reduction(c, mode=mode) * 100
            line += f"  {adv:>+10.4f}  {rr:>+9.1f}%"
        logger.info(line)

    logger.info(f"\n  Part B — Accuracy Sweep (σ={DEFAULT_SIGMA}):")
    header = f"    {'ρ':>5}  {'Oracle':>8}  {'Tabula':>8}"
    for _, lbl in transfer_modes_summary:
        header += f"  {lbl+' Δ':>10}  {lbl+' RR%':>10}"
    logger.info(header)
    for rho in ACCURACY_VALUES:
        c = part_b[rho]
        line = (f"    {rho:>5.2f}  {c['selection_oracle']['eval_reward']['mean']:>8.4f}  "
                f"{c['tabula_rasa']['eval_reward']['mean']:>8.4f}")
        for mode, _ in transfer_modes_summary:
            adv = transfer_advantage(c, mode=mode)
            rr = regret_reduction(c, mode=mode) * 100
            line += f"  {adv:>+10.4f}  {rr:>+9.1f}%"
        logger.info(line)

    logger.info(f"\n  Part D — K Scaling:")
    for K in K_VALUES:
        c = part_d[K]
        parts = [f"    K={K:<3}"]
        for mode, lbl in transfer_modes_summary:
            adv = transfer_advantage(c, mode=mode)
            rr = regret_reduction(c, mode=mode) * 100
            parts.append(f"{lbl} Δ={adv:+.4f} RR={rr:+.1f}%")
        logger.info("  ".join(parts))

    # Determine winner
    logger.info("\n  --- WINNER DETERMINATION ---")
    for sigma in [0.05, 0.1, 0.2]:
        if sigma in part_a:
            c = part_a[sigma]
            best_mode = max(transfer_modes_summary,
                          key=lambda x: regret_reduction(c, mode=x[0]))
            rr = regret_reduction(c, mode=best_mode[0]) * 100
            logger.info(f"    σ={sigma}: Best variant = {best_mode[1]} (RegRed={rr:+.1f}%)")

    elapsed = time.time() - t_total
    logger.info(f"\n  Total time: {elapsed:.0f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
