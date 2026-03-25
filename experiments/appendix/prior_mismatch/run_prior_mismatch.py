#!/usr/bin/env python3
"""Appendix: Prior Mismatch Sensitivity Analysis.

Measures how increasing levels of prior mismatch affect routing quality,
and whether conservative ``n_eff`` can rescue bad priors.

Five prior-quality levels form a mismatch gradient:

  1. **Well-calibrated** — full training set (correct arm ranking and magnitude)
  2. **Random-1680** — random subsample of 1,680 training prompts (same
     sample count as GSM8K-only; isolates sample-size effects from domain
     mismatch)
  3. **MMLU-only (mild)** — correct arm ranking, domain-specific magnitudes
  4. **GSM8K-only (moderate)** — all models appear near-equal (~0.95+)
  5. **Inverted (severe)** — Llama ↔ Gemini rewards swapped in prior data

Each prior level is tested at three ``n_eff`` values (10, 100, 1000),
plus two Tabula Rasa baselines:

  - **Tabula Rasa** — independently tuned (γ=0.995), the production default.
  - **Tabula Rasa (γ-matched)** — same γ=0.996 as warmup conditions but no
    priors.  Isolates the prior's informational benefit from the forgetting-
    factor difference, which otherwise confounds the comparison.

Total: 17 conditions.  Pairwise tests are reported against both baselines.

Priors are generated inline from the canonical training split using the
same FeatureService and PCA as evaluation.  The well-calibrated condition
uses the shipped priors for exact reproducibility with the warmup ablation.

Design note — prior strength fairness:
  The router scales loaded priors by ``n_eff / n_warmup``, so the final
  effective prior matrix is ``A_final ≈ n_eff * plasticity * E[xx^T]``,
  independent of the training-set size.  Prior *strength* is therefore
  controlled by n_eff alone.  The Random-1680 control empirically verifies
  this: any gap between Random-1680 and Well-calibrated at the same n_eff
  is attributable to covariance estimation noise, not prior strength.

Evaluation: cumulative-regret protocol on the held-out test split
(n=1,824 prompts, 20 seeds), stationary environment.  Seeds match the
warmup ablation (offset=9000) for potential paired comparisons.

Usage:
    python experiments/appendix/prior_mismatch/run_prior_mismatch.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    BEST_K3_TABULA_RASA_HPARAMS,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    HOLDOUT_DATA_PATH,
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    K3_WARMUP_PRIORS_PATH,
    N_SEEDS,
    TRAIN_DATA_PATH,
)
from pareto_bandit.calibration import generate_warmup_priors
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in (
    "pareto_bandit.router",
    "pareto_bandit.feature_service",
    "pareto_bandit.policy",
    "pareto_bandit.calibration",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ======================================================================
# Constants
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
ARM_SHORT: Dict[str, str] = K3_ARM_SHORT
LLAMA = "meta-llama/llama-3.1-8b-instruct"
GEMINI = "google/gemini-2.5-pro"

SEED_OFFSET: int = 9000
RESULTS_DIR = Path(__file__).parent / "results"
PRIORS_DIR = RESULTS_DIR / "priors"

CHECKPOINT_INTERVAL: int = 25
EARLY_STEP: int = 200

ALPHA: float = BEST_K3_HPARAMS["alpha"]
GAMMA: float = BEST_K3_HPARAMS["forgetting_factor"]
TABULA_ALPHA: float = BEST_K3_TABULA_RASA_HPARAMS["alpha"]
TABULA_GAMMA: float = BEST_K3_TABULA_RASA_HPARAMS["forgetting_factor"]

N_EFF_VALUES: List[float] = [10.0, 100.0, 1000.0]
PLASTICITY: float = 0.1
PRIOR_SEED: int = 42


# ======================================================================
# Data types
# ======================================================================


@dataclass
class StepRecord:
    """Per-step metrics recorded during a trial."""

    step: int
    model: str
    reward: float
    oracle_reward: float

    @property
    def regret(self) -> float:
        return self.oracle_reward - self.reward


@dataclass
class SeedResult:
    """Aggregate metrics for one (condition, seed) trial."""

    condition: str
    seed: int
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    def total_regret(self) -> float:
        return sum(s.regret for s in self.steps)

    def regret_at(self, step: int) -> float:
        return sum(s.regret for s in self.steps[:step])

    def mean_reward(self) -> float:
        return float(np.mean([s.reward for s in self.steps]))


# ======================================================================
# Prior Generation
# ======================================================================


def _load_train_jsonl(
    path: Path,
    arm_order: List[str],
    *,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load training data in the ``{prompt, rewards}`` format.

    Parameters
    ----------
    path : Path
        Canonical ``train.jsonl``.
    arm_order : list[str]
        Model IDs to include.
    source_filter : str or None
        If provided, keep only prompts from this benchmark source.

    Returns
    -------
    list[dict]
        ``{"prompt": str, "rewards": {model_id: float}}`` entries.
    """
    arm_set = set(arm_order)
    data: List[Dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if source_filter is not None and row.get("source", "") != source_filter:
                continue
            arms = row.get("arms", {})
            if not arm_set <= set(arms.keys()):
                continue
            rewards = {m: arms[m]["reward"] for m in arm_order}
            data.append({"prompt": row["prompt"], "rewards": rewards})

    return data


def _invert_rewards(
    data: List[Dict[str, Any]],
    arm_a: str,
    arm_b: str,
) -> List[Dict[str, Any]]:
    """Swap reward columns between two arms to create adversarial priors."""
    inverted: List[Dict[str, Any]] = []
    for entry in data:
        new_rewards = dict(entry["rewards"])
        new_rewards[arm_a], new_rewards[arm_b] = (
            entry["rewards"][arm_b],
            entry["rewards"][arm_a],
        )
        inverted.append({"prompt": entry["prompt"], "rewards": new_rewards})
    return inverted


def _build_priors(
    rewards_data: List[Dict[str, Any]],
    output_path: Path,
    label: str,
) -> Path:
    """Build warmup priors from rewards data and save to disk.

    Uses the same PCA and encoder as the shipped priors for consistency.

    Parameters
    ----------
    rewards_data : list[dict]
        ``{"prompt": str, "rewards": {...}}`` entries.
    output_path : Path
        Where to save the ``.joblib`` artifact.
    label : str
        Descriptive label stored in the artifact metadata.

    Returns
    -------
    Path
        The saved artifact path.
    """
    np.random.seed(PRIOR_SEED)
    state = generate_warmup_priors(
        rewards_data=rewards_data,
        encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca=DEFAULT_PCA_PATH,
        plasticity=PLASTICITY,
        whiten_pca=True,
        output_path=output_path,
    )
    state["split_mode"] = label
    state["arm_order"] = ARM_ORDER
    joblib.dump(state, output_path)
    return output_path


RANDOM_SUBSAMPLE_N: int = 1680
"""Match GSM8K-only sample count for the sample-size control."""

RANDOM_SUBSAMPLE_SEED: int = 42
"""Deterministic seed for the random subsample (reproducible)."""


def generate_all_priors() -> Dict[str, Path]:
    """Generate (or locate) the five prior-quality levels.

    Returns a mapping from prior-quality label to ``.joblib`` path.
    The well-calibrated condition uses the shipped priors for exact
    reproducibility with the warmup ablation.
    """
    PRIORS_DIR.mkdir(parents=True, exist_ok=True)
    priors: Dict[str, Path] = {}

    # --- Well-calibrated: shipped priors (no regeneration needed) ---
    priors["Well-calibrated"] = K3_WARMUP_PRIORS_PATH
    logger.info("Well-calibrated priors: %s", K3_WARMUP_PRIORS_PATH)

    # --- Random-1680: sample-size control ---
    # Same number of prompts as GSM8K-only but drawn uniformly from the
    # full training distribution.  Any gap between this and Well-calibrated
    # at the same n_eff is due to covariance estimation noise, not domain
    # mismatch.  Any gap between this and GSM8K-only isolates domain shift.
    random_path = PRIORS_DIR / "priors_random_1680.joblib"
    if not random_path.exists():
        logger.info("Generating Random-1680 priors (sample-size control) ...")
        full_data = _load_train_jsonl(TRAIN_DATA_PATH, ARM_ORDER)
        rng_sub = np.random.default_rng(RANDOM_SUBSAMPLE_SEED)
        indices = rng_sub.choice(
            len(full_data), size=RANDOM_SUBSAMPLE_N, replace=False,
        )
        subset_data = [full_data[i] for i in indices]
        logger.info(
            "  %d / %d random training prompts",
            len(subset_data), len(full_data),
        )
        _build_priors(subset_data, random_path, "random_subsample_1680")
    else:
        logger.info("Random-1680 priors already exist: %s", random_path)
    priors["Random-1680"] = random_path

    # --- MMLU-only (mild mismatch) ---
    mmlu_path = PRIORS_DIR / "priors_mmlu_only.joblib"
    if not mmlu_path.exists():
        logger.info("Generating MMLU-only priors ...")
        mmlu_data = _load_train_jsonl(TRAIN_DATA_PATH, ARM_ORDER, source_filter="mmlu")
        logger.info("  %d MMLU training prompts", len(mmlu_data))
        _build_priors(mmlu_data, mmlu_path, "mmlu_only")
    else:
        logger.info("MMLU-only priors already exist: %s", mmlu_path)
    priors["MMLU-only"] = mmlu_path

    # --- GSM8K-only (moderate mismatch) ---
    gsm8k_path = PRIORS_DIR / "priors_gsm8k_only.joblib"
    if not gsm8k_path.exists():
        logger.info("Generating GSM8K-only priors ...")
        gsm8k_data = _load_train_jsonl(
            TRAIN_DATA_PATH, ARM_ORDER, source_filter="gsm8k",
        )
        logger.info("  %d GSM8K training prompts", len(gsm8k_data))
        _build_priors(gsm8k_data, gsm8k_path, "gsm8k_only")
    else:
        logger.info("GSM8K-only priors already exist: %s", gsm8k_path)
    priors["GSM8K-only"] = gsm8k_path

    # --- Inverted (severe mismatch): swap Llama ↔ Gemini rewards ---
    inverted_path = PRIORS_DIR / "priors_inverted.joblib"
    if not inverted_path.exists():
        logger.info("Generating inverted priors (Llama <-> Gemini swap) ...")
        full_data = _load_train_jsonl(TRAIN_DATA_PATH, ARM_ORDER)
        inverted_data = _invert_rewards(full_data, LLAMA, GEMINI)
        logger.info("  %d inverted training prompts", len(inverted_data))
        _build_priors(inverted_data, inverted_path, "inverted_llama_gemini")
    else:
        logger.info("Inverted priors already exist: %s", inverted_path)
    priors["Inverted"] = inverted_path

    return priors


# ======================================================================
# Router Factory
# ======================================================================


def _create_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    priors_path: Optional[Path] = None,
    alpha: float = ALPHA,
    prior_n_effective: float = 1000.0,
    forgetting_factor: float = GAMMA,
    seed: Optional[int] = None,
) -> BanditRouter:
    """Build a K=3 router with the specified prior configuration.

    Parameters
    ----------
    registry : dict
        Model registry.
    feature_dim : int
        Context vector dimensionality.
    priors_path : Path or None
        Path to ``.joblib`` priors.  ``None`` for tabula rasa.
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Effective pseudo-observations for priors.
    forgetting_factor : float
        Geometric forgetting factor.
    seed : int or None
        If provided, seeds the policy's internal RNG for deterministic
        tie-breaking.  Required for bit-exact reproducibility.
    """
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    use_warmup = priors_path is not None
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=str(priors_path) if use_warmup else None,
        prior_n_effective=prior_n_effective,
        alpha=alpha,
        cost_penalty=0.0,
        forgetting_factor=forgetting_factor,
    )
    if seed is not None:
        router.bandit._rng = np.random.default_rng(seed)
    return router


# ======================================================================
# Trial Runner
# ======================================================================


def _run_trial(
    *,
    condition_label: str,
    data: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    seed: int,
    priors_path: Optional[Path] = None,
    alpha: float = ALPHA,
    prior_n_effective: float = 1000.0,
    forgetting_factor: float = GAMMA,
) -> SeedResult:
    """Run one seed for one condition (cumulative-regret protocol).

    Parameters
    ----------
    condition_label : str
        Human-readable condition name.
    data : SplitData
        Held-out test split.
    registry : dict
        Model registry.
    feature_dim : int
        Context vector dimensionality.
    seed : int
        Random seed for prompt ordering.
    priors_path : Path or None
        Path to ``.joblib`` priors.  ``None`` for tabula rasa.
    alpha : float
        Exploration coefficient.
    prior_n_effective : float
        Effective pseudo-observations for priors.
    forgetting_factor : float
        Geometric forgetting factor.

    Returns
    -------
    SeedResult
        Per-step metrics for this seed.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(data.n)

    router = _create_router(
        registry,
        feature_dim,
        priors_path=priors_path,
        alpha=alpha,
        prior_n_effective=prior_n_effective,
        forgetting_factor=forgetting_factor,
        seed=seed,
    )

    result = SeedResult(condition=condition_label, seed=seed)

    for t_idx in range(data.n):
        orig_idx = order[t_idx]
        emb = data.embeddings[orig_idx]
        model, log = router.route(emb)
        reward = float(data.rewards[model][orig_idx])
        log.cost_usd = float(data.costs[model][orig_idx])
        router.process_feedback(log.request_id, reward=reward)

        oracle_reward = max(
            float(data.rewards[a][orig_idx]) for a in ARM_ORDER
        )
        result.steps.append(
            StepRecord(
                step=t_idx + 1,
                model=model,
                reward=reward,
                oracle_reward=oracle_reward,
            )
        )

    return result


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_seeds(
    seed_results: List[SeedResult],
) -> Dict[str, Any]:
    """Aggregate per-seed results into checkpoint curves and summary stats."""
    n_seeds = len(seed_results)
    n_total = seed_results[0].n

    checkpoints = sorted(set(
        [1]
        + list(range(CHECKPOINT_INTERVAL, n_total + 1, CHECKPOINT_INTERVAL))
        + [n_total]
    ))

    curves: List[Dict[str, Any]] = []
    for cp_step in checkpoints:
        cum_regrets = [
            sum(s.regret for s in sr.steps[:cp_step]) for sr in seed_results
        ]
        curves.append({
            "step": cp_step,
            "mean_cumulative_regret": float(np.mean(cum_regrets)),
            "se_cumulative_regret": float(
                np.std(cum_regrets) / np.sqrt(n_seeds)
            ),
            "per_seed_cumulative_regret": [float(r) for r in cum_regrets],
        })

    per_seed_regret = [sr.total_regret() for sr in seed_results]
    per_seed_reward = [sr.mean_reward() for sr in seed_results]
    per_seed_regret_early = [sr.regret_at(EARLY_STEP) for sr in seed_results]

    return {
        "label": seed_results[0].condition,
        "curves": curves,
        "total_regret": {
            "mean": float(np.mean(per_seed_regret)),
            "median": float(np.median(per_seed_regret)),
            "std": float(np.std(per_seed_regret)),
            "se": float(np.std(per_seed_regret) / np.sqrt(n_seeds)),
        },
        "mean_reward": {
            "mean": float(np.mean(per_seed_reward)),
            "se": float(np.std(per_seed_reward) / np.sqrt(n_seeds)),
        },
        f"regret_at_{EARLY_STEP}": {
            "mean": float(np.mean(per_seed_regret_early)),
            "median": float(np.median(per_seed_regret_early)),
            "std": float(np.std(per_seed_regret_early)),
            "se": float(np.std(per_seed_regret_early) / np.sqrt(n_seeds)),
        },
        "per_seed_regret": per_seed_regret,
        "per_seed_reward": per_seed_reward,
        f"per_seed_regret_at_{EARLY_STEP}": per_seed_regret_early,
    }


# ======================================================================
# Condition Definitions
# ======================================================================


PRIOR_QUALITY_ORDER: List[str] = [
    "Well-calibrated",
    "Random-1680",
    "MMLU-only",
    "GSM8K-only",
    "Inverted",
]

TABULA_RASA_LABEL: str = "Tabula Rasa"
GAMMA_MATCHED_TR_LABEL: str = "Tabula Rasa (γ-matched)"

BASELINE_LABELS: List[str] = [TABULA_RASA_LABEL, GAMMA_MATCHED_TR_LABEL]


def _build_conditions(
    prior_paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    """Build the 17-condition experimental design.

    Five prior-quality levels × three n_eff values + two tabula-rasa
    baselines (independently tuned and γ-matched).

    Parameters
    ----------
    prior_paths : dict
        Mapping from prior-quality label to ``.joblib`` path.

    Returns
    -------
    list[dict]
        Each dict has keys used by ``_run_trial``.
    """
    conditions: List[Dict[str, Any]] = []

    conditions.append({
        "label": TABULA_RASA_LABEL,
        "priors_path": None,
        "alpha": TABULA_ALPHA,
        "prior_n_effective": BEST_K3_TABULA_RASA_HPARAMS["prior_n_effective"],
        "forgetting_factor": TABULA_GAMMA,
    })

    conditions.append({
        "label": GAMMA_MATCHED_TR_LABEL,
        "priors_path": None,
        "alpha": ALPHA,
        "prior_n_effective": 1.0,
        "forgetting_factor": GAMMA,
    })

    for quality_label in PRIOR_QUALITY_ORDER:
        path = prior_paths[quality_label]
        for n_eff in N_EFF_VALUES:
            conditions.append({
                "label": f"{quality_label} (n_eff={int(n_eff)})",
                "priors_path": path,
                "alpha": ALPHA,
                "prior_n_effective": n_eff,
                "forgetting_factor": GAMMA,
            })

    return conditions


# ======================================================================
# Statistical Tests
# ======================================================================


CATASTROPHIC_MULTIPLIER: float = 2.0
"""A seed is 'catastrophic' if its regret exceeds this multiple of a
fixed reference median (the γ-matched Tabula Rasa baseline).  Using a
condition-independent threshold avoids the bias where high-regret
conditions inflate their own threshold and appear to have zero
catastrophic failures."""


def _paired_tests(
    per_seed_a: List[float],
    per_seed_b: List[float],
    *,
    catastrophic_ref_median: Optional[float] = None,
) -> Dict[str, Any]:
    """Sign test + Fisher exact test on paired per-seed regret.

    Computes two complementary non-parametric tests:

    - **Sign test** (exact binomial): counts seeds where the condition
      has strictly lower regret than baseline.  H0: P(condition wins)
      = 0.5.  No distributional assumptions — answers "does the
      condition have lower regret seed-by-seed?" (location).
    - **Fisher exact test** on the 2×2 catastrophic-failure table
      (condition vs baseline × catastrophic vs safe).  H0: equal
      failure rates.  Answers "does the condition reduce tail risk?"

    The sign test requires only that paired observations are
    independent — it makes no assumption about the symmetry or shape
    of the difference distribution, which is critical here because
    Tabula Rasa's per-seed regret is heavy-tailed / bimodal while
    warmup conditions are tightly clustered.

    Parameters
    ----------
    per_seed_a, per_seed_b : list[float]
        Same-length arrays of per-seed total regret, paired by seed.
        Convention: a = condition under test, b = baseline (Tabula Rasa).
        A "win" is when a < b (condition has lower regret).
    catastrophic_ref_median : float or None
        Fixed reference for the catastrophic threshold
        (``CATASTROPHIC_MULTIPLIER * ref``).  Should be the γ-matched
        Tabula Rasa median so that the threshold is condition-
        independent.  Falls back to ``median(per_seed_a)`` if ``None``.

    Returns
    -------
    dict
        Sign test wins/losses/ties, p-value, Fisher exact test p-value,
        and catastrophic failure rates for both conditions.
    """
    from scipy.stats import binomtest, fisher_exact

    a = np.array(per_seed_a)
    b = np.array(per_seed_b)
    diffs = a - b
    n_seeds = len(diffs)

    n_a_wins = int(np.sum(diffs < 0))
    n_b_wins = int(np.sum(diffs > 0))
    n_ties = int(np.sum(diffs == 0))
    n_effective = n_seeds - n_ties

    if n_effective > 0:
        result = binomtest(n_a_wins, n_effective, 0.5, alternative="two-sided")
        p_val = result.pvalue
    else:
        p_val = 1.0

    ref_median = catastrophic_ref_median if catastrophic_ref_median is not None else float(np.median(a))
    catastrophic_threshold = CATASTROPHIC_MULTIPLIER * ref_median

    a_catastrophic = int(np.sum(a > catastrophic_threshold))
    b_catastrophic = int(np.sum(b > catastrophic_threshold))

    fisher_table = [
        [a_catastrophic, n_seeds - a_catastrophic],
        [b_catastrophic, n_seeds - b_catastrophic],
    ]
    _, fisher_p = fisher_exact(fisher_table, alternative="less")

    return {
        "test": "sign_test_and_fisher",
        "n_condition_wins": n_a_wins,
        "n_baseline_wins": n_b_wins,
        "n_ties": n_ties,
        "n_effective": n_effective,
        "sign_test_p": float(p_val),
        "fisher_exact_p": float(fisher_p),
        "delta_mean": float(np.mean(diffs)),
        "delta_median": float(np.median(diffs)),
        "condition_median": float(np.median(a)),
        "baseline_median": float(np.median(b)),
        "catastrophic_threshold": catastrophic_threshold,
        "condition_catastrophic_count": a_catastrophic,
        "baseline_catastrophic_count": b_catastrophic,
        "condition_catastrophic_rate": float(a_catastrophic / n_seeds),
        "baseline_catastrophic_rate": float(b_catastrophic / n_seeds),
        "n_seeds": n_seeds,
    }


def _holm_bonferroni(
    label_to_pval: Dict[str, float],
) -> Dict[str, float]:
    """Apply Holm-Bonferroni step-down correction for multiple comparisons.

    Parameters
    ----------
    label_to_pval : dict[str, float]
        Mapping from condition label to raw (uncorrected) p-value.

    Returns
    -------
    dict[str, float]
        Same keys, with corrected p-values (capped at 1.0).
    """
    items = sorted(label_to_pval.items(), key=lambda kv: kv[1])
    m = len(items)
    corrected: Dict[str, float] = {}
    running_max = 0.0
    for rank, (label, raw_p) in enumerate(items):
        adjusted = raw_p * (m - rank)
        running_max = max(running_max, adjusted)
        corrected[label] = min(running_max, 1.0)
    return corrected


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Prior generation (cached after first run) ---
    logger.info("=" * 70)
    logger.info("PHASE 1: Prior Generation")
    logger.info("=" * 70)
    prior_paths = generate_all_priors()

    # --- Log prior diagnostics ---
    logger.info("\nPrior diagnostics:")
    for label, path in prior_paths.items():
        p = joblib.load(path)
        logger.info("  %s (n=%d prompts):", label, p.get("n", 0))
        for m in ARM_ORDER:
            theta = np.linalg.solve(p["A"][m], p["b"][m])
            bias_pred = theta[-1]
            logger.info(
                "    %-20s  bias_pred=%.3f  ||theta||=%.4f",
                ARM_SHORT[m], bias_pred, np.linalg.norm(theta),
            )

    # --- Data loading ---
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Evaluation")
    logger.info("=" * 70)
    fs = FeatureService()
    feature_dim = fs.dimension
    test_data = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    logger.info("Test split: %d prompts, K=%d arms", test_data.n, len(ARM_ORDER))

    registry = build_model_registry(ARM_ORDER)
    conditions = _build_conditions(prior_paths)

    # --- Run all conditions ---
    all_results: Dict[str, Dict[str, Any]] = {}

    for cond in conditions:
        label = cond["label"]
        logger.info("=== %s ===", label)
        seed_results: List[SeedResult] = []

        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            sr = _run_trial(
                condition_label=label,
                data=test_data,
                registry=registry,
                feature_dim=feature_dim,
                seed=seed,
                priors_path=cond["priors_path"],
                alpha=cond["alpha"],
                prior_n_effective=cond["prior_n_effective"],
                forgetting_factor=cond["forgetting_factor"],
            )
            seed_results.append(sr)
            if (s + 1) % 5 == 0:
                logger.info(
                    "  seed %d/%d  regret=%.1f  R@%d=%.1f  reward=%.4f",
                    s + 1, N_SEEDS,
                    sr.total_regret(), EARLY_STEP,
                    sr.regret_at(EARLY_STEP), sr.mean_reward(),
                )

        agg = _aggregate_seeds(seed_results)
        all_results[label] = agg
        logger.info(
            "  FINAL: regret=%.1f +/- %.1f  R@%d=%.1f +/- %.1f  reward=%.4f",
            agg["total_regret"]["mean"], agg["total_regret"]["se"],
            EARLY_STEP,
            agg[f"regret_at_{EARLY_STEP}"]["mean"],
            agg[f"regret_at_{EARLY_STEP}"]["se"],
            agg["mean_reward"]["mean"],
        )

    # --- Pairwise tests against both baselines ---
    # The γ-matched TR is the fair, controlled comparison (isolates the
    # prior effect); the original TR is the production-relevant baseline.
    gm_tr_median = all_results[GAMMA_MATCHED_TR_LABEL]["total_regret"]["median"]
    catastrophic_ref = gm_tr_median
    logger.info(
        "\nCatastrophic threshold: %.1f × %.1f (γ-matched TR median) = %.1f",
        CATASTROPHIC_MULTIPLIER, catastrophic_ref,
        CATASTROPHIC_MULTIPLIER * catastrophic_ref,
    )

    all_pairwise: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for baseline_label in BASELINE_LABELS:
        bl_seeds = all_results[baseline_label]["per_seed_regret"]
        bl_std = all_results[baseline_label]["total_regret"]["std"]
        logger.info("\n--- Statistical Tests (vs %s) ---", baseline_label)
        logger.info("  Sign test (exact binomial, H0: P(condition wins) = 0.5)")
        logger.info("  %s std=%.1f", baseline_label, bl_std)

        pairwise_tests: Dict[str, Dict[str, Any]] = {}
        for label, agg in all_results.items():
            if label in BASELINE_LABELS:
                continue
            test = _paired_tests(
                agg["per_seed_regret"],
                bl_seeds,
                catastrophic_ref_median=catastrophic_ref,
            )
            pairwise_tests[label] = test

        raw_sign_p = {k: v["sign_test_p"] for k, v in pairwise_tests.items()}
        raw_fisher_p = {k: v["fisher_exact_p"] for k, v in pairwise_tests.items()}
        corrected_sign = _holm_bonferroni(raw_sign_p)
        corrected_fisher = _holm_bonferroni(raw_fisher_p)

        for label, test in pairwise_tests.items():
            test["sign_test_p_holm"] = corrected_sign[label]
            test["fisher_exact_p_holm"] = corrected_fisher[label]
            p_corr = test["sign_test_p_holm"]
            sig = "***" if p_corr < 0.001 else (
                "**" if p_corr < 0.01 else (
                    "*" if p_corr < 0.05 else "ns"
                )
            )
            logger.info(
                "  %-35s  wins=%d/%d  sign_p=%.4g (Holm=%.4g)  Δmed=%+.1f  "
                "cat: %d/%d vs %d/%d (Fisher p=%.4g, Holm=%.4g)  %s",
                label,
                test["n_condition_wins"], test["n_effective"],
                test["sign_test_p"], test["sign_test_p_holm"],
                test["delta_median"],
                test["condition_catastrophic_count"], test["n_seeds"],
                test["baseline_catastrophic_count"], test["n_seeds"],
                test["fisher_exact_p"], test["fisher_exact_p_holm"],
                sig,
            )

        all_pairwise[baseline_label] = pairwise_tests

    # --- Baseline-vs-baseline: quantify the γ effect itself ---
    logger.info("\n--- Baseline Comparison: γ effect (TR γ=0.996 vs TR γ=0.997) ---")
    baseline_comparison = _paired_tests(
        all_results[GAMMA_MATCHED_TR_LABEL]["per_seed_regret"],
        all_results[TABULA_RASA_LABEL]["per_seed_regret"],
        catastrophic_ref_median=catastrophic_ref,
    )
    gm_stats = all_results[GAMMA_MATCHED_TR_LABEL]["total_regret"]
    tr_stats = all_results[TABULA_RASA_LABEL]["total_regret"]
    logger.info(
        "  γ-matched TR: median=%.1f, mean=%.1f, std=%.1f",
        gm_stats["median"], gm_stats["mean"], gm_stats["std"],
    )
    logger.info(
        "  Original  TR: median=%.1f, mean=%.1f, std=%.1f",
        tr_stats["median"], tr_stats["mean"], tr_stats["std"],
    )
    logger.info(
        "  γ-matched wins=%d/%d  sign_p=%.4g  Δmed=%+.1f  "
        "cat: %d/%d vs %d/%d (Fisher p=%.4g)",
        baseline_comparison["n_condition_wins"],
        baseline_comparison["n_effective"],
        baseline_comparison["sign_test_p"],
        baseline_comparison["delta_median"],
        baseline_comparison["condition_catastrophic_count"],
        baseline_comparison["n_seeds"],
        baseline_comparison["baseline_catastrophic_count"],
        baseline_comparison["n_seeds"],
        baseline_comparison["fisher_exact_p"],
    )

    # --- Save results ---
    output: Dict[str, Any] = {
        "experiment": "appendix_prior_mismatch",
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "n_prompts": test_data.n,
        "split": "test",
        "arms": ARM_ORDER,
        "arm_short": ARM_SHORT,
        "early_step": EARLY_STEP,
        "n_eff_values": N_EFF_VALUES,
        "prior_quality_levels": PRIOR_QUALITY_ORDER,
        "hparams": {
            "warmup": {"alpha": ALPHA, "forgetting_factor": GAMMA},
            "tabula_rasa": {"alpha": TABULA_ALPHA, "forgetting_factor": TABULA_GAMMA},
            "gamma_matched_tabula_rasa": {"alpha": ALPHA, "forgetting_factor": GAMMA},
            "policy": "disjoint",
            "plasticity": PLASTICITY,
        },
        "catastrophic_ref_median": catastrophic_ref,
        "catastrophic_threshold": CATASTROPHIC_MULTIPLIER * catastrophic_ref,
        "prior_diagnostics": {},
        "pairwise_tests_vs_tabula_rasa": all_pairwise.get(TABULA_RASA_LABEL, {}),
        "pairwise_tests_vs_gamma_matched_tr": all_pairwise.get(
            GAMMA_MATCHED_TR_LABEL, {},
        ),
        "baseline_comparison_gamma_effect": baseline_comparison,
        "conditions": all_results,
    }

    for label, path in prior_paths.items():
        p = joblib.load(path)
        diag: Dict[str, Any] = {"n_prompts": p.get("n", 0), "models": {}}
        for m in ARM_ORDER:
            theta = np.linalg.solve(p["A"][m], p["b"][m])
            diag["models"][ARM_SHORT[m]] = {
                "bias_pred": float(theta[-1]),
                "theta_norm": float(np.linalg.norm(theta)),
                "trace_A": float(np.trace(p["A"][m])),
            }
        output["prior_diagnostics"][label] = diag

    out_path = RESULTS_DIR / "prior_mismatch_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults written to %s", out_path)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
