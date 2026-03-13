#!/usr/bin/env python3
"""
Drift threshold calibration via regret-grounded synthetic shift sweep.

A real distribution shift changes both the **embedding distribution**
(what the detector monitors) and the **reward landscape** (which arm is
best).  This script couples them: for each sweep step it both shifts the
top PCA components and boosts Llama's reward on shifted prompts, simulating
a domain change where the cheaper model becomes more competitive.

For two forgetting-factor settings (ff=1.0 vs ff=0.999), the script
measures cumulative regret.  The **crossover point** — the shift
magnitude where ff=0.999 first beats ff=1.0 — gives a principled,
regret-grounded calibration for the drift detector threshold.

The chi-squared score the detector would observe at each magnitude
maps directly to the sigma-based threshold parameter.

Outputs (``results/``)
    drift_calibration.json  — full sweep data + recommended threshold
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    BEST_K2_HPARAMS,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.storage import EphemeralContextStore
from bandit_gpt.router import BanditRouter

from utils.embeddings import load_embedding_cache, embed_dataset_cached
from utils.simulation import build_model_registry, compute_normalized_costs

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

K4_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "rewards"
    / "archive" / "k4_canonical" / "rewards.jsonl"
)
K4_MODELS_CONFIG_PATH = (
    PROJECT_ROOT / "data_collection" / "config" / "models_k4.json"
)

ARM_ORDER: List[str] = K2_ARM_ORDER
ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

HEADLINE_LAMBDA: float = 0.2
PRIOR_N_EFFECTIVE: float = 5000.0
PHASE1_N_PARETO: int = 500
ONLINE_FRACTION: float = 0.70

N_SHIFT_COMPONENTS: int = 8
SHIFT_MAGNITUDES: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
CALIBRATION_SEEDS: int = 5
SEED_OFFSET: int = 42

# Reward perturbation per unit of embedding shift.  At shift=Xσ,
# Llama reward is boosted by X * REWARD_BOOST_PER_SIGMA (clamped to
# [0, 1]).  This couples covariate shift with reward landscape change:
# a domain shift that also makes the cheap model more competitive.
REWARD_BOOST_PER_SIGMA: float = 0.04

# ============================================================================
# Data loading — reuse from run_distribution_shift
# ============================================================================

from run_distribution_shift import (
    load_k4_as_per_prompt,
    load_pareto_split,
    embed_records,
    split_online_holdout,
)


def best_arm_dist(
    records: List[Dict[str, Any]],
    arm_order: List[str],
) -> Dict[str, float]:
    """Fraction of prompts where each arm has the highest reward."""
    counts: Dict[str, int] = {a: 0 for a in arm_order}
    for rec in records:
        best = max(arm_order, key=lambda a: rec["arms"][a]["reward"])
        counts[best] += 1
    n = len(records)
    return {a: counts[a] / n for a in arm_order}


# ============================================================================
# Lightweight regret runner (no checkpointing, just final regret)
# ============================================================================


def run_final_regret(
    phase1_records: List[Dict[str, Any]],
    phase1_emb: List[np.ndarray],
    phase2_records: List[Dict[str, Any]],
    phase2_emb: List[np.ndarray],
    arm_order: List[str],
    feature_dim: int,
    normalized_costs: Dict[str, float],
    *,
    alpha: float,
    prior_n_effective: float,
    warmup_path: str,
    cost_penalty: float,
    forgetting_factor: float,
    drift_threshold: float = 0.0,
    n_seeds: int = CALIBRATION_SEEDS,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, Any]:
    """Run two-phase stream and return final cumulative regret stats.

    Stripped-down version of ``run_learning_curve`` without checkpoint
    overhead — only tracks cumulative regret for speed.

    Args:
        phase1_records: In-distribution prompts (Pareto).
        phase1_emb: Phase 1 embeddings.
        phase2_records: Cross-distribution prompts (K4 online).
        phase2_emb: Phase 2 embeddings (possibly shifted).
        arm_order: Model identifiers.
        feature_dim: Context vector dimensionality.
        normalized_costs: Normalized per-model costs.
        alpha: UCB exploration parameter.
        prior_n_effective: Prior scaling factor.
        warmup_path: Path to warmup priors.
        cost_penalty: Lambda for cost-adjusted regret.
        forgetting_factor: Forgetting factor (1.0 = stationary, <1 = decay).
        drift_threshold: Sigma threshold for the covariate drift detector.
            When > 0, the router resets to tabula rasa upon detection.
            0 = disabled (default).
        n_seeds: Number of random seeds.
        seed_offset: Base seed.

    Returns:
        Dict with mean/std cumulative regret, per-seed values, and
        the number of resets triggered (when drift_threshold > 0).
    """
    n_p1 = len(phase1_records)
    n_p2 = len(phase2_records)
    seed_regrets: List[float] = []
    seed_resets: List[int] = []

    for s in range(n_seeds):
        seed = seed_offset + s
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        registry = build_model_registry(arm_order)
        fs = FeatureService.for_precomputed(feature_dim)
        store = EphemeralContextStore()

        hp = BEST_K2_HPARAMS
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=fs,
            context_store=store,
            priors="warmup",
            warmup_path=warmup_path,
            prior_n_effective=prior_n_effective,
            alpha=alpha,
            use_corralling=False,
            cost_penalty=cost_penalty,
            forgetting_factor=forgetting_factor,
            drift_threshold=drift_threshold,
            drift_burn_in_steps=50,
            drift_ema_alpha=0.05,
            drift_confirmation_window=20,
            policy=hp.get("policy", "hybrid"),
        )

        p1_order = rng.permutation(n_p1)
        p2_order = rng.permutation(n_p2)

        all_records = (
            [phase1_records[i] for i in p1_order]
            + [phase2_records[i] for i in p2_order]
        )
        all_emb = (
            [phase1_emb[i] for i in p1_order]
            + [phase2_emb[i] for i in p2_order]
        )

        cumulative_regret = 0.0
        for t in range(len(all_records)):
            rec = all_records[t]
            emb = all_emb[t]
            model, log = router.route(emb)
            reward = rec["arms"][model]["reward"]
            router.process_feedback(log.request_id, reward=reward)

            oracle_utility = max(
                rec["arms"][a]["reward"] - cost_penalty * normalized_costs[a]
                for a in arm_order
            )
            chosen_utility = reward - cost_penalty * normalized_costs[model]
            cumulative_regret += oracle_utility - chosen_utility

        seed_regrets.append(cumulative_regret)
        seed_resets.append(getattr(router, "_n_resets", 0))

    return {
        "mean_regret": float(np.mean(seed_regrets)),
        "std_regret": float(np.std(seed_regrets)),
        "seed_regrets": seed_regrets,
        "mean_resets": float(np.mean(seed_resets)),
        "seed_resets": seed_resets,
    }


# ============================================================================
# Chi-squared score computation
# ============================================================================


def compute_chi2_for_shift(
    pareto_emb: List[np.ndarray],
    shifted_k4_emb: List[np.ndarray],
    burn_in_steps: int = 50,
) -> Dict[str, float]:
    """Simulate the DriftDetector's chi-squared computation.

    Uses the second half of the first ``burn_in_steps`` Pareto vectors
    as the reference distribution, then computes the mean chi-squared
    score over the shifted K4 vectors.

    Args:
        pareto_emb: Phase 1 embeddings (Pareto).
        shifted_k4_emb: Phase 2 embeddings (K4, possibly shifted).
        burn_in_steps: Number of burn-in observations.

    Returns:
        Dict with baseline, mean_chi2, and sigma-excess.
    """
    burn_in = np.array(pareto_emb[:burn_in_steps])
    half = burn_in_steps // 2
    second_half = burn_in[half:]

    ref_mean = second_half.mean(axis=0)
    raw_std = second_half.std(axis=0)
    ref_std = np.where(raw_std > 1e-10, raw_std, 1e-10)

    def _chi2(x: np.ndarray) -> float:
        z = (np.asarray(x) - ref_mean) / ref_std
        return float(np.mean(z ** 2))

    burn_in_scores = [_chi2(v) for v in second_half]
    baseline_mean = float(np.mean(burn_in_scores))
    baseline_std = float(np.std(burn_in_scores))
    baseline = baseline_mean + 2.0 * baseline_std

    k4_scores = [_chi2(v) for v in shifted_k4_emb]
    mean_chi2 = float(np.mean(k4_scores))

    sigma_excess = (
        (mean_chi2 - baseline) / baseline_std
        if baseline_std > 0 else 0.0
    )

    return {
        "baseline": baseline,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "mean_chi2": mean_chi2,
        "sigma_excess": sigma_excess,
    }


# ============================================================================
# Main calibration sweep
# ============================================================================


def run_calibration() -> None:
    """Run the synthetic shift sweep and output calibration data."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("DRIFT THRESHOLD CALIBRATION — REGRET-GROUNDED SWEEP")
    logger.info(f"  Shift magnitudes: {SHIFT_MAGNITUDES}")
    logger.info(f"  Components shifted: top {N_SHIFT_COMPONENTS}")
    logger.info(f"  Seeds per condition: {CALIBRATION_SEEDS}")
    logger.info(f"  λ={HEADLINE_LAMBDA}, n_eff={PRIOR_N_EFFECTIVE}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load shared resources
    # ------------------------------------------------------------------
    logger.info("\nLoading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    feature_dim = pca.n_components_ + 1

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    hp = BEST_K2_HPARAMS

    # ------------------------------------------------------------------
    # Load and embed data
    # ------------------------------------------------------------------
    logger.info("\nLoading data ...")
    pareto_all = load_pareto_split(TRAIN_DATA_PATH, ARM_ORDER)
    k4_all = load_k4_as_per_prompt(ARM_ORDER)
    logger.info(f"  Pareto: {len(pareto_all)}, K4: {len(k4_all)}")

    logger.info("Embedding ...")
    pareto_all_emb = embed_records(pareto_all, embedding_cache, encoder, pca)
    k4_all_emb = embed_records(k4_all, embedding_cache, encoder, pca)

    # Phase 1: sample Pareto prompts
    rng_split = np.random.default_rng(0)
    p1_idx = rng_split.choice(
        len(pareto_all), size=PHASE1_N_PARETO, replace=False,
    )
    phase1_records = [pareto_all[i] for i in p1_idx]
    phase1_emb = [pareto_all_emb[i] for i in p1_idx]

    # Phase 2: K4 online/holdout split
    k4_online, k4_online_emb, k4_holdout, k4_holdout_emb = split_online_holdout(
        k4_all, k4_all_emb,
    )
    logger.info(f"  Phase 1: {len(phase1_records)} Pareto")
    logger.info(f"  Phase 2: {len(k4_online)} K4 online")

    # ------------------------------------------------------------------
    # Compute reference std from Pareto (for scaling shifts)
    # ------------------------------------------------------------------
    pareto_matrix = np.array(phase1_emb)
    ref_std_per_comp = pareto_matrix.std(axis=0)
    logger.info(f"  Pareto ref_std (first 5 comps): "
                f"{ref_std_per_comp[:5].round(4).tolist()}")

    # ------------------------------------------------------------------
    # Normalized costs
    # ------------------------------------------------------------------
    registry = build_model_registry(ARM_ORDER)
    norm_costs = compute_normalized_costs(registry, ARM_ORDER)

    # ------------------------------------------------------------------
    # Sweep: coupled embedding shift + reward perturbation
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP START (coupled embedding + reward shift)")
    logger.info(f"  Reward boost per σ: {REWARD_BOOST_PER_SIGMA} "
                f"(Llama reward += shift * {REWARD_BOOST_PER_SIGMA})")
    logger.info("  Conditions: ff=1.0 (static priors), ff=0.999 (gradual "
                "decay), Reset (tabula rasa on detection)")
    logger.info("=" * 70)

    llama_arm = ARM_ORDER[0]  # meta-llama/llama-3.1-8b-instruct

    # The drift detector threshold for the "Reset" condition must be
    # below the sigma excess at the smallest non-zero shift to ensure
    # it fires.  We use a conservative 1.0σ — the sweep itself will
    # reveal what threshold is appropriate.
    DETECTOR_THRESHOLD: float = 1.0

    sweep_results: List[Dict[str, Any]] = []

    for mag in SHIFT_MAGNITUDES:
        logger.info(f"\n--- Shift = {mag:.1f}σ ---")

        # 1. Shift embeddings
        shift_vec = np.zeros(feature_dim)
        for j in range(min(N_SHIFT_COMPONENTS, feature_dim - 1)):
            shift_vec[j] = mag * ref_std_per_comp[j]

        shifted_k4_emb = [emb + shift_vec for emb in k4_online_emb]

        # 2. Perturb rewards: boost Llama by mag * REWARD_BOOST_PER_SIGMA
        reward_boost = mag * REWARD_BOOST_PER_SIGMA
        perturbed_k4 = []
        for rec in k4_online:
            new_rec = {
                "prompt": rec["prompt"],
                "arms": {},
                "source": rec.get("source", "k4_canonical"),
            }
            for arm in ARM_ORDER:
                r = rec["arms"][arm]["reward"]
                c = rec["arms"][arm]["cost"]
                if arm == llama_arm:
                    r = min(1.0, r + reward_boost)
                new_rec["arms"][arm] = {"reward": r, "cost": c}
            perturbed_k4.append(new_rec)

        orig_llama_r = np.mean([r["arms"][llama_arm]["reward"] for r in k4_online])
        new_llama_r = np.mean([r["arms"][llama_arm]["reward"] for r in perturbed_k4])
        new_best_arm = best_arm_dist(perturbed_k4, ARM_ORDER)
        logger.info(f"  Llama reward: {orig_llama_r:.4f} → {new_llama_r:.4f} "
                    f"(+{reward_boost:.3f})")
        logger.info(f"  Best-arm: Llama={new_best_arm[llama_arm]:.1%}, "
                    f"Gemini={new_best_arm[ARM_ORDER[1]]:.1%}")

        # 3. Compute chi-squared score
        chi2_info = compute_chi2_for_shift(phase1_emb, shifted_k4_emb)
        logger.info(f"  chi2: mean={chi2_info['mean_chi2']:.4f}, "
                    f"baseline={chi2_info['baseline']:.4f}, "
                    f"sigma_excess={chi2_info['sigma_excess']:.2f}")

        # 4. Run ff=1.0 (Static priors — no adaptation)
        logger.info(f"  Running ff=1.0 ({CALIBRATION_SEEDS} seeds) ...")
        ff1_result = run_final_regret(
            phase1_records, phase1_emb,
            perturbed_k4, shifted_k4_emb,
            ARM_ORDER, feature_dim, norm_costs,
            alpha=hp["alpha"],
            prior_n_effective=PRIOR_N_EFFECTIVE,
            warmup_path=warmup_path,
            cost_penalty=HEADLINE_LAMBDA,
            forgetting_factor=1.0,
        )

        # 5. Run ff=0.999 (Gradual decay)
        logger.info(f"  Running ff=0.999 ({CALIBRATION_SEEDS} seeds) ...")
        ff999_result = run_final_regret(
            phase1_records, phase1_emb,
            perturbed_k4, shifted_k4_emb,
            ARM_ORDER, feature_dim, norm_costs,
            alpha=hp["alpha"],
            prior_n_effective=PRIOR_N_EFFECTIVE,
            warmup_path=warmup_path,
            cost_penalty=HEADLINE_LAMBDA,
            forgetting_factor=0.999,
        )

        # 6. Run Reset (tabula rasa on detection)
        logger.info(f"  Running Reset@{DETECTOR_THRESHOLD:.1f}σ "
                    f"({CALIBRATION_SEEDS} seeds) ...")
        reset_result = run_final_regret(
            phase1_records, phase1_emb,
            perturbed_k4, shifted_k4_emb,
            ARM_ORDER, feature_dim, norm_costs,
            alpha=hp["alpha"],
            prior_n_effective=PRIOR_N_EFFECTIVE,
            warmup_path=warmup_path,
            cost_penalty=HEADLINE_LAMBDA,
            forgetting_factor=1.0,
            drift_threshold=DETECTOR_THRESHOLD,
        )

        delta_ff999 = ff1_result["mean_regret"] - ff999_result["mean_regret"]
        delta_reset = ff1_result["mean_regret"] - reset_result["mean_regret"]

        logger.info(f"  ff=1.0 regret:   {ff1_result['mean_regret']:.2f} "
                    f"± {ff1_result['std_regret']:.2f}")
        logger.info(f"  ff=0.999 regret: {ff999_result['mean_regret']:.2f} "
                    f"± {ff999_result['std_regret']:.2f}")
        logger.info(f"  Reset regret:    {reset_result['mean_regret']:.2f} "
                    f"± {reset_result['std_regret']:.2f}  "
                    f"(avg resets={reset_result['mean_resets']:.1f})")
        logger.info(f"  Δ(ff1 - ff999) = {delta_ff999:+.2f}  "
                    f"({'ff=0.999 WINS' if delta_ff999 > 0 else 'ff=1.0 wins'})")
        logger.info(f"  Δ(ff1 - Reset) = {delta_reset:+.2f}  "
                    f"({'RESET WINS' if delta_reset > 0 else 'ff=1.0 wins'})")

        sweep_results.append({
            "shift_magnitude": mag,
            "n_components_shifted": N_SHIFT_COMPONENTS,
            "reward_boost": reward_boost,
            "llama_mean_reward": new_llama_r,
            "best_arm_llama_frac": new_best_arm[llama_arm],
            "ff1_mean_regret": ff1_result["mean_regret"],
            "ff1_std_regret": ff1_result["std_regret"],
            "ff999_mean_regret": ff999_result["mean_regret"],
            "ff999_std_regret": ff999_result["std_regret"],
            "reset_mean_regret": reset_result["mean_regret"],
            "reset_std_regret": reset_result["std_regret"],
            "reset_mean_n_resets": reset_result["mean_resets"],
            "delta_ff999": delta_ff999,
            "delta_reset": delta_reset,
            "ff999_wins": delta_ff999 > 0,
            "reset_wins": delta_reset > 0,
            "chi2_mean": chi2_info["mean_chi2"],
            "chi2_baseline": chi2_info["baseline"],
            "chi2_baseline_std": chi2_info["baseline_std"],
            "chi2_sigma_excess": chi2_info["sigma_excess"],
        })

    # ------------------------------------------------------------------
    # Find crossover
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 70)

    crossover_reset: Optional[float] = None
    crossover_reset_chi2: Optional[float] = None
    crossover_reset_sigma: Optional[float] = None

    for entry in sweep_results:
        if entry["reset_wins"] and crossover_reset is None:
            crossover_reset = entry["shift_magnitude"]
            crossover_reset_chi2 = entry["chi2_mean"]
            crossover_reset_sigma = entry["chi2_sigma_excess"]

    header = (
        "  Shift | R_boost | Llama% |  ff=1.0 reg  | ff=0.999 reg "
        "|  Reset reg   | Δ(999) | Δ(Rst) | chi2  | σ-excess"
    )
    sep = "  " + "-" * (len(header) - 2)
    logger.info(f"\n{header}")
    logger.info(sep)
    for e in sweep_results:
        marker = " <<<" if e["shift_magnitude"] == crossover_reset else ""
        logger.info(
            f"  {e['shift_magnitude']:4.1f}σ  | {e['reward_boost']:+.3f}  | "
            f"{e['best_arm_llama_frac']:5.1%}  | "
            f"{e['ff1_mean_regret']:12.2f} | {e['ff999_mean_regret']:12.2f} | "
            f"{e['reset_mean_regret']:12.2f} | "
            f"{e['delta_ff999']:+6.1f} | {e['delta_reset']:+6.1f} | "
            f"{e['chi2_mean']:5.2f} | {e['chi2_sigma_excess']:+7.2f}{marker}"
        )

    if crossover_reset is not None:
        logger.info(f"\n  RESET CROSSOVER at shift={crossover_reset:.1f}σ")
        logger.info(f"  Chi2 at crossover: {crossover_reset_chi2:.4f}")
        logger.info(f"  Sigma excess at crossover: {crossover_reset_sigma:.2f}")
        recommended = max(1.0, math.floor(crossover_reset_sigma * 0.8))
        logger.info(f"  RECOMMENDED DRIFT_THRESHOLD: {recommended:.1f}σ")
        logger.info(f"  (80% of crossover sigma excess, floored to 1.0)")
    else:
        logger.info("\n  NO RESET CROSSOVER — ff=1.0 wins at all shift levels.")
        logger.info("  Tabula rasa reset does not help for these magnitudes.")

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Drift threshold calibration (with reset)",
            "shift_magnitudes": SHIFT_MAGNITUDES,
            "n_components_shifted": N_SHIFT_COMPONENTS,
            "calibration_seeds": CALIBRATION_SEEDS,
            "headline_lambda": HEADLINE_LAMBDA,
            "prior_n_effective": PRIOR_N_EFFECTIVE,
            "phase1_n_pareto": PHASE1_N_PARETO,
            "reward_boost_per_sigma": REWARD_BOOST_PER_SIGMA,
            "detector_threshold": DETECTOR_THRESHOLD,
            "hparams": BEST_K2_HPARAMS,
        },
        "sweep": sweep_results,
        "crossover_reset": {
            "found": crossover_reset is not None,
            "shift_magnitude": crossover_reset,
            "chi2_at_crossover": crossover_reset_chi2,
            "sigma_excess_at_crossover": crossover_reset_sigma,
        },
    }

    out_path = output_dir / "drift_calibration.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults → {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_calibration()
