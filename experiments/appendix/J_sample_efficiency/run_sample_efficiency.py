#!/usr/bin/env python3
"""
Appendix J: Sample Efficiency of Warmup Priors
===============================================

Compares the learning curves of BanditGPT (Corralling + warmup priors)
versus Tabula Rasa (single LinUCB, no priors) to isolate the sample
efficiency advantage of warmup initialization.

Both variants are evaluated using *holdout reward as a function of online
training steps* (the learning curve), which captures how quickly each
approach learns a good routing policy from limited data.  This
complements the train-then-freeze Pareto analysis in Figure 3/4, which
evaluates only the final converged policy.

Key insight: warmup priors and tabula rasa are expected to converge to
similar final reward given enough data.  The value of warmup priors is
*faster learning* — achieving a target quality level with fewer online
samples.  This experiment quantifies that advantage.

Protocol
--------
1. Data: Same canonical dev/holdout splits as Figure 3.
2. For both K=2 and K=10 portfolios:
   a. BanditGPT: Corralling + warmup priors (Appendix H-tuned hparams).
   b. Tabula Rasa: Single LinUCB, no priors (Appendix H-tuned hparams).
3. At each checkpoint (0, 10, 25, 50, ..., N), freeze the router and
   evaluate on the full holdout set (alpha=0, exploit mode).
4. Repeat for 20 random seeds; report mean reward and 95% CI.

Outputs (``results/``)
    sample_efficiency_results.json
    figure_sample_efficiency.png
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_WARMUP_FROM_MULTIMODEL_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
    K10_MODELS_PATH,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models, load_model_catalog
from utils.router_factory import create_experiment_router
from utils.embeddings import load_embedding_cache, embed_dataset_cached

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

N_SEEDS: int = 20
SEED_OFFSET: int = 42
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05
DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7

REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0

K2_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4.1",
]

_PRICES_K2 = get_prices_for_models(K2_MODELS)


def _req_cost(inp: float, out: float) -> float:
    """Per-request cost assuming 100 input + 400 output tokens."""
    return (100 * inp + 400 * out) / 1_000_000


K2_CATALOG: Dict[str, Dict] = {
    m: {
        "display": m.split("/")[-1],
        **_PRICES_K2[m],
        "cost": _req_cost(
            _PRICES_K2[m]["input_cost_per_m"],
            _PRICES_K2[m]["output_cost_per_m"],
        ),
    }
    for m in K2_MODELS
}

K10_MODELS, K10_CATALOG = load_model_catalog(K10_MODELS_PATH)


def _make_learning_curve_checkpoints(n_train: int) -> List[int]:
    """Build checkpoint list adapted to training set size.

    Dense at the start (where learning speed differences are most
    visible), sparser later.  Always includes 0 and n_train.
    """
    candidates = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500,
                  600, 700, 800, 900, 1000, 1200, 1500, 2000]
    checkpoints = [s for s in candidates if s <= n_train]
    if n_train not in checkpoints:
        checkpoints.append(n_train)
    return checkpoints


# ============================================================================
# Data loading
# ============================================================================

import gzip
from collections import defaultdict


def load_rewards_from_file(
    data_path: Path,
    models: List[str],
    prompt_filter: Optional[set] = None,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL.

    Only prompts with rewards for *all* requested models are included.
    """
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            if model_id not in model_set:
                continue
            if prompt_filter is not None and prompt not in prompt_filter:
                continue
            rewards[prompt][model_id] = extract_reward(entry)

    n_models = len(models)
    return [
        {"prompt": p, "rewards": rmap}
        for p, rmap in rewards.items()
        if len(rmap) == n_models
    ]


def build_model_registry(
    models: List[str],
    catalog: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    """Build the registry dict that ``create_experiment_router`` expects."""
    return {
        m: {
            "input_cost_per_m": catalog[m]["input_cost_per_m"],
            "output_cost_per_m": catalog[m]["output_cost_per_m"],
        }
        for m in models
    }


_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def embed_dataset(
    data: List[Dict],
    encoder: "SentenceTransformer",
    pca: Any,
) -> List[np.ndarray]:
    """Embed all prompts, using the pre-computed cache when available."""
    return embed_dataset_cached(data, _EMBEDDING_CACHE, encoder, pca)


def _split_dev_train_val(
    data: List[Dict],
    emb: List[np.ndarray],
    val_fraction: float = DEV_VAL_FRACTION,
    seed: int = DEV_VAL_SEED,
) -> Tuple[List[Dict], List[np.ndarray], List[Dict], List[np.ndarray]]:
    """Deterministically split (data, emb) into train and val portions."""
    n = len(data)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_d = [data[i] for i in range(n) if i not in val_idx]
    train_e = [emb[i] for i in range(n) if i not in val_idx]
    val_d = [data[i] for i in range(n) if i in val_idx]
    val_e = [emb[i] for i in range(n) if i in val_idx]
    return train_d, train_e, val_d, val_e


# ============================================================================
# Frozen evaluation
# ============================================================================


def _set_exploit_mode(router: Any, *, enable: bool) -> Dict[str, Any]:
    """Switch to greedy exploitation on a frozen router."""
    if not enable:
        return {}
    saved: Dict[str, Any] = {"expert_alphas": [], "meta_exploit": False}
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts"):
        for expert in cr.experts:
            saved["expert_alphas"].append((expert.alpha_start, expert.alpha_end))
            expert.alpha_start = 0.0
            expert.alpha_end = 0.0
        saved["meta_exploit"] = cr.exploit_mode
        cr.exploit_mode = True
    return saved


def _restore_exploit_mode(router: Any, saved: Dict[str, Any]) -> None:
    """Restore expert alpha values and meta exploit mode after evaluation."""
    if not saved:
        return
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts") and saved.get("expert_alphas"):
        for expert, (a_s, a_e) in zip(cr.experts, saved["expert_alphas"]):
            expert.alpha_start = a_s
            expert.alpha_end = a_e
        cr.exploit_mode = saved.get("meta_exploit", False)


def evaluate_frozen(
    router: Any,
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    costs: Dict[str, float],
    total_steps: int,
) -> Tuple[float, float]:
    """Evaluate a frozen router on the holdout set (no learning).

    Returns:
        (mean_reward, mean_cost).
    """
    saved = _set_exploit_mode(router, enable=True)
    rng_state = np.random.get_state()

    r_total = c_total = 0.0
    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=total_steps)
        r_total += p["rewards"][model]
        c_total += costs[model]

    np.random.set_state(rng_state)
    _restore_exploit_mode(router, saved)
    n = len(eval_data)
    return r_total / n, c_total / n


# ============================================================================
# Learning curve
# ============================================================================


def run_learning_curve(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: Optional[str],
    costs: Dict[str, float],
    n_trials: int,
    checkpoints: List[int],
    *,
    use_corralling: bool = True,
    cost_penalty: float = 0.0,
    alpha: float = 1.0,
    label: str = "banditGPT",
    prior_n_effective: float = 5000.0,
    forgetting_factor: float = 1.0,
) -> List[Dict]:
    """Holdout quality as a function of online training steps.

    At each checkpoint, the router is frozen (alpha=0, exploit mode) and
    evaluated on the full holdout set.  Step 0 evaluates with priors only
    (or cold initialization for tabula rasa).

    Args:
        models: Candidate model IDs.
        catalog: Model metadata catalog.
        train_data: Dev-train prompts with rewards.
        eval_data: Holdout-set prompts with rewards.
        train_emb: Pre-computed feature vectors for dev-train.
        eval_emb: Pre-computed feature vectors for holdout.
        warmup_path: Path to warmup priors (None for tabula rasa).
        costs: Per-model cost dict.
        n_trials: Number of random seeds.
        checkpoints: Training steps at which to evaluate.
        use_corralling: Whether to use Corralling meta-learner.
        cost_penalty: Lambda for cost-quality trade-off.
        alpha: Exploration coefficient for LinUCB experts.
        label: Label for the curve in output data.
        prior_n_effective: Effective sample size for prior scaling.
        forgetting_factor: Exponential decay for past observations.

    Returns:
        List of dicts, one per checkpoint, with mean/std reward.
    """
    dim = train_emb[0].shape[0]
    r_min = REWARD_THEORETICAL_MIN
    r_range = REWARD_THEORETICAL_MAX - REWARD_THEORETICAL_MIN
    burn_in = len(train_data)
    checkpoint_set = set(checkpoints)

    by_step: Dict[int, List[float]] = {s: [] for s in checkpoints}

    for trial in range(n_trials):
        seed = SEED_OFFSET + trial
        np.random.seed(seed)
        trial_rng = np.random.default_rng(seed)
        router = create_experiment_router(
            model_registry=build_model_registry(models, catalog),
            feature_dim=dim,
            prior_n_effective=prior_n_effective,
            alpha=alpha,
            warmup_path=warmup_path,
            use_corralling=use_corralling,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=cost_penalty,
            forgetting_factor=forgetting_factor,
        )

        if 0 in checkpoint_set:
            r, _ = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            by_step[0].append(r)

        order = trial_rng.permutation(len(train_data))
        for step_idx, idx in enumerate(order):
            p, x = train_data[idx], train_emb[idx]
            model, log = router.route(x, total_steps=burn_in)
            raw_reward = p["rewards"][model]
            norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
            router.process_feedback(log.request_id, norm_reward)
            current = step_idx + 1
            if current in checkpoint_set:
                r, _ = evaluate_frozen(
                    router, eval_data, eval_emb, costs, burn_in,
                )
                by_step[current].append(r)

        if (trial + 1) % 5 == 0:
            logger.info(f"    [{label}] trial {trial + 1}/{n_trials}")

    curve = []
    for s in sorted(checkpoints):
        rr = by_step[s]
        if rr:
            curve.append({
                "step": s,
                "mean_reward": float(np.mean(rr)),
                "std_reward": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
                "n_trials": len(rr),
                "label": label,
            })
    return curve


# ============================================================================
# Quantitative metrics
# ============================================================================


def compute_sample_efficiency_metrics(
    warmup_curve: List[Dict],
    tr_curve: List[Dict],
    oracle_reward: float,
    threshold_fraction: float = 0.90,
) -> Dict[str, Any]:
    """Compute sample-efficiency metrics from two learning curves.

    Args:
        warmup_curve: BanditGPT learning curve (list of checkpoint dicts).
        tr_curve: Tabula Rasa learning curve (list of checkpoint dicts).
        oracle_reward: Per-prompt oracle reward on the holdout set.
        threshold_fraction: Fraction of oracle reward to use as a target.

    Returns:
        Dict with AUC, steps-to-threshold, and final reward for both.
    """
    target = threshold_fraction * oracle_reward

    def _auc(curve: List[Dict]) -> float:
        """Trapezoidal AUC of the learning curve."""
        if len(curve) < 2:
            return 0.0
        steps = np.array([d["step"] for d in curve], dtype=float)
        rewards = np.array([d["mean_reward"] for d in curve])
        return float(np.trapz(rewards, steps))

    def _steps_to_threshold(curve: List[Dict], thresh: float) -> Optional[int]:
        """First checkpoint where mean_reward >= threshold."""
        for d in curve:
            if d["mean_reward"] >= thresh:
                return d["step"]
        return None

    warmup_steps = _steps_to_threshold(warmup_curve, target)
    tr_steps = _steps_to_threshold(tr_curve, target)

    return {
        "threshold_fraction": threshold_fraction,
        "target_reward": target,
        "oracle_reward": oracle_reward,
        "warmup": {
            "auc": _auc(warmup_curve),
            "steps_to_threshold": warmup_steps,
            "final_reward": warmup_curve[-1]["mean_reward"] if warmup_curve else None,
            "final_std": warmup_curve[-1]["std_reward"] if warmup_curve else None,
        },
        "tabula_rasa": {
            "auc": _auc(tr_curve),
            "steps_to_threshold": tr_steps,
            "final_reward": tr_curve[-1]["mean_reward"] if tr_curve else None,
            "final_std": tr_curve[-1]["std_reward"] if tr_curve else None,
        },
        "auc_advantage": (
            _auc(warmup_curve) - _auc(tr_curve) if warmup_curve and tr_curve else None
        ),
        "speedup": (
            f"{tr_steps / warmup_steps:.1f}x"
            if warmup_steps and tr_steps and warmup_steps > 0
            else "N/A"
        ),
    }


# ============================================================================
# Hyperparameter loading
# ============================================================================


def _load_tuned_hparams(
    path: Path,
    key: str,
) -> Optional[Dict[str, float]]:
    """Load dev-val-selected hyperparameters from Appendix H.

    Returns None if file is missing or unparseable.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cfg = data.get(key, {})
        return {
            "alpha": float(cfg["alpha"]),
            "prior_n_effective": float(cfg["n_eff"]),
            "forgetting_factor": float(cfg["gamma"]),
        }
    except Exception as exc:
        logger.warning(f"Failed to load hparams from {path}: {exc}")
        return None


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the sample-efficiency experiment for K=2 and K=10."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components")

    global _EMBEDDING_CACHE  # noqa: PLW0603
    _EMBEDDING_CACHE = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    # Load Appendix H hyperparameters
    hparams_dir = (
        Path(__file__).resolve().parent.parent
        / "H_alpha_neff_ablation" / "results"
    )
    tuned_k2_bg = _load_tuned_hparams(hparams_dir / "best_hparams_k2.json", "K2")
    tuned_k2_tr = _load_tuned_hparams(
        hparams_dir / "best_hparams_k2_tabula_rasa.json", "K2",
    )
    tuned_k10_bg = _load_tuned_hparams(hparams_dir / "best_hparams_k10.json", "K10")
    tuned_k10_tr = _load_tuned_hparams(
        hparams_dir / "best_hparams_k10_tabula_rasa.json", "K10",
    )

    _ablation_script = (
        "experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py"
    )
    defaults_bg = {"alpha": 1.0, "prior_n_effective": 5000.0, "forgetting_factor": 1.0}
    defaults_tr = {"alpha": 1.0, "prior_n_effective": 10.0, "forgetting_factor": 1.0}

    for label, tuned, path_name in [
        ("K=2 BanditGPT", tuned_k2_bg, "best_hparams_k2.json"),
        ("K=2 Tabula Rasa", tuned_k2_tr, "best_hparams_k2_tabula_rasa.json"),
        ("K=10 BanditGPT", tuned_k10_bg, "best_hparams_k10.json"),
        ("K=10 Tabula Rasa", tuned_k10_tr, "best_hparams_k10_tabula_rasa.json"),
    ]:
        if tuned is not None:
            logger.info(
                f"  Loaded {label}: alpha={tuned['alpha']} "
                f"n_eff={tuned['prior_n_effective']} "
                f"gamma={tuned['forgetting_factor']}"
            )
        else:
            logger.warning(
                f"  {label} hparams not found ({path_name}). "
                f"Using defaults. Run {_ablation_script} first."
            )

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Appendix J: Sample Efficiency of Warmup Priors",
            "n_seeds": N_SEEDS,
            "protocol": "learning_curve_train_then_freeze",
            "description": (
                "Compares learning curves (holdout reward vs online steps) "
                "for BanditGPT (Corralling + warmup) and Tabula Rasa "
                "(single LinUCB, no priors).  The warmup advantage is "
                "primarily in sample efficiency (learning speed), not "
                "asymptotic performance."
            ),
        },
    }

    # ==================================================================
    # K=2
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=2: Sample Efficiency Comparison")
    logger.info("=" * 70)

    costs_k2 = {m: K2_CATALOG[m]["cost"] for m in K2_MODELS}

    logger.info("  Loading K=2 data ...")
    dev_k2 = load_rewards_from_file(CANONICAL_DEV_DATA_PATH, K2_MODELS)
    holdout_k2 = load_rewards_from_file(CANONICAL_HOLDOUT_DATA_PATH, K2_MODELS)
    logger.info(f"    Dev: {len(dev_k2)}  Holdout: {len(holdout_k2)}")

    logger.info("  Embedding K=2 prompts ...")
    dev_emb_k2 = embed_dataset(dev_k2, encoder, pca)
    holdout_emb_k2 = embed_dataset(holdout_k2, encoder, pca)

    dev_train_k2, dev_train_emb_k2, _, _ = _split_dev_train_val(dev_k2, dev_emb_k2)
    logger.info(f"    Dev-train: {len(dev_train_k2)}")

    checkpoints_k2 = _make_learning_curve_checkpoints(len(dev_train_k2))
    logger.info(f"    Checkpoints: {checkpoints_k2}")

    k2_warmup_path = str(K2_WARMUP_FROM_MULTIMODEL_PATH)

    hp_bg = tuned_k2_bg or defaults_bg
    logger.info(
        f"\n  Running BanditGPT learning curve (K=2, {N_SEEDS} seeds) ..."
    )
    lc_k2_bg = run_learning_curve(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_k2, dev_train_emb_k2, holdout_emb_k2,
        k2_warmup_path, costs_k2, N_SEEDS, checkpoints_k2,
        use_corralling=True, label="BanditGPT",
        alpha=hp_bg["alpha"],
        prior_n_effective=hp_bg["prior_n_effective"],
        forgetting_factor=hp_bg["forgetting_factor"],
    )

    hp_tr = tuned_k2_tr or defaults_tr
    logger.info(
        f"\n  Running Tabula Rasa learning curve (K=2, {N_SEEDS} seeds) ..."
    )
    lc_k2_tr = run_learning_curve(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_k2, dev_train_emb_k2, holdout_emb_k2,
        None, costs_k2, N_SEEDS, checkpoints_k2,
        use_corralling=False, label="Tabula Rasa",
        alpha=hp_tr["alpha"],
        prior_n_effective=hp_tr["prior_n_effective"],
        forgetting_factor=hp_tr["forgetting_factor"],
    )

    oracle_k2 = max(
        float(np.mean([p["rewards"][m] for p in holdout_k2]))
        for m in K2_MODELS
    )
    oracle_perprompt_k2 = float(np.mean([
        max(p["rewards"][m] for m in K2_MODELS) for p in holdout_k2
    ]))
    static_weak_k2 = min(
        float(np.mean([p["rewards"][m] for p in holdout_k2]))
        for m in K2_MODELS
    )

    metrics_k2 = compute_sample_efficiency_metrics(
        lc_k2_bg, lc_k2_tr, oracle_perprompt_k2,
    )

    logger.info(f"\n  K=2 Results:")
    logger.info(f"    Oracle (per-prompt):     {oracle_perprompt_k2:.4f}")
    logger.info(f"    Target (90% oracle):     {metrics_k2['target_reward']:.4f}")
    logger.info(
        f"    BanditGPT final:         {metrics_k2['warmup']['final_reward']:.4f}"
        f" +/- {metrics_k2['warmup']['final_std']:.4f}"
    )
    logger.info(
        f"    Tabula Rasa final:       {metrics_k2['tabula_rasa']['final_reward']:.4f}"
        f" +/- {metrics_k2['tabula_rasa']['final_std']:.4f}"
    )
    logger.info(
        f"    BanditGPT steps to 90%:  {metrics_k2['warmup']['steps_to_threshold']}"
    )
    logger.info(
        f"    Tabula Rasa steps to 90%:{metrics_k2['tabula_rasa']['steps_to_threshold']}"
    )
    logger.info(f"    Speedup:                 {metrics_k2['speedup']}")
    logger.info(
        f"    AUC advantage:           {metrics_k2['auc_advantage']:.1f}"
    )

    results["K2"] = {
        "models": K2_MODELS,
        "n_dev_train": len(dev_train_k2),
        "n_holdout": len(holdout_k2),
        "oracle_per_prompt": oracle_perprompt_k2,
        "best_static": oracle_k2,
        "weak_static": static_weak_k2,
        "checkpoints": checkpoints_k2,
        "warmup_hparams": hp_bg,
        "tabula_rasa_hparams": hp_tr,
        "warmup_curve": lc_k2_bg,
        "tabula_rasa_curve": lc_k2_tr,
        "metrics": metrics_k2,
    }

    # ==================================================================
    # K=10
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=10: Sample Efficiency Comparison")
    logger.info("=" * 70)

    costs_k10 = {m: K10_CATALOG[m]["cost"] for m in K10_MODELS}

    logger.info("  Loading K=10 data ...")
    prior_train_prompts: set = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(
            f"    Excluding {len(prior_train_prompts)} prior-train prompts"
        )

    all_dev_k10 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K10_MODELS)
    train_k10 = [
        d for d in all_dev_k10 if d["prompt"] not in prior_train_prompts
    ]
    holdout_k10 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K10_MODELS,
    )
    logger.info(f"    Train (excl. prior-train): {len(train_k10)}")
    logger.info(f"    Holdout: {len(holdout_k10)}")

    logger.info("  Embedding K=10 prompts ...")
    train_emb_k10 = embed_dataset(train_k10, encoder, pca)
    holdout_emb_k10 = embed_dataset(holdout_k10, encoder, pca)

    train_train_k10, train_train_emb_k10, _, _ = _split_dev_train_val(
        train_k10, train_emb_k10,
    )
    logger.info(f"    Train-train: {len(train_train_k10)}")

    checkpoints_k10 = _make_learning_curve_checkpoints(len(train_train_k10))
    logger.info(f"    Checkpoints: {checkpoints_k10}")

    k10_warmup_path = str(MULTIMODEL_WARMUP_PRIORS_PATH)

    hp_bg_k10 = tuned_k10_bg or defaults_bg
    logger.info(
        f"\n  Running BanditGPT learning curve (K=10, {N_SEEDS} seeds) ..."
    )
    lc_k10_bg = run_learning_curve(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_k10, train_train_emb_k10, holdout_emb_k10,
        k10_warmup_path, costs_k10, N_SEEDS, checkpoints_k10,
        use_corralling=True, label="BanditGPT",
        alpha=hp_bg_k10["alpha"],
        prior_n_effective=hp_bg_k10["prior_n_effective"],
        forgetting_factor=hp_bg_k10["forgetting_factor"],
    )

    hp_tr_k10 = tuned_k10_tr or defaults_tr
    logger.info(
        f"\n  Running Tabula Rasa learning curve (K=10, {N_SEEDS} seeds) ..."
    )
    lc_k10_tr = run_learning_curve(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_k10, train_train_emb_k10, holdout_emb_k10,
        None, costs_k10, N_SEEDS, checkpoints_k10,
        use_corralling=False, label="Tabula Rasa",
        alpha=hp_tr_k10["alpha"],
        prior_n_effective=hp_tr_k10["prior_n_effective"],
        forgetting_factor=hp_tr_k10["forgetting_factor"],
    )

    oracle_perprompt_k10 = float(np.mean([
        max(p["rewards"][m] for m in K10_MODELS) for p in holdout_k10
    ]))
    best_static_k10 = max(
        float(np.mean([p["rewards"][m] for p in holdout_k10]))
        for m in K10_MODELS
    )
    weak_static_k10 = min(
        float(np.mean([p["rewards"][m] for p in holdout_k10]))
        for m in K10_MODELS
    )

    metrics_k10 = compute_sample_efficiency_metrics(
        lc_k10_bg, lc_k10_tr, oracle_perprompt_k10,
    )

    logger.info(f"\n  K=10 Results:")
    logger.info(f"    Oracle (per-prompt):     {oracle_perprompt_k10:.4f}")
    logger.info(f"    Target (90% oracle):     {metrics_k10['target_reward']:.4f}")
    logger.info(
        f"    BanditGPT final:         {metrics_k10['warmup']['final_reward']:.4f}"
        f" +/- {metrics_k10['warmup']['final_std']:.4f}"
    )
    logger.info(
        f"    Tabula Rasa final:       {metrics_k10['tabula_rasa']['final_reward']:.4f}"
        f" +/- {metrics_k10['tabula_rasa']['final_std']:.4f}"
    )
    logger.info(
        f"    BanditGPT steps to 90%:  {metrics_k10['warmup']['steps_to_threshold']}"
    )
    logger.info(
        f"    Tabula Rasa steps to 90%:{metrics_k10['tabula_rasa']['steps_to_threshold']}"
    )
    logger.info(f"    Speedup:                 {metrics_k10['speedup']}")
    logger.info(
        f"    AUC advantage:           {metrics_k10['auc_advantage']:.1f}"
    )

    results["K10"] = {
        "models": K10_MODELS,
        "n_train": len(train_train_k10),
        "n_holdout": len(holdout_k10),
        "oracle_per_prompt": oracle_perprompt_k10,
        "best_static": best_static_k10,
        "weak_static": weak_static_k10,
        "checkpoints": checkpoints_k10,
        "warmup_hparams": hp_bg_k10,
        "tabula_rasa_hparams": hp_tr_k10,
        "warmup_curve": lc_k10_bg,
        "tabula_rasa_curve": lc_k10_tr,
        "metrics": metrics_k10,
    }

    # ==================================================================
    # Save
    # ==================================================================
    out_path = output_dir / "sample_efficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Generate figure
    from plot_sample_efficiency import plot_sample_efficiency
    plot_sample_efficiency(results, output_dir)


if __name__ == "__main__":
    main()
