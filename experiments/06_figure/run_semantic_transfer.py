#!/usr/bin/env python3
"""
Figure 6: Semantic Transfer Evaluation (K=3).

Re-evaluates model-to-model theta-transfer after improvements to the
router, PCA components (15-comp), and encoder (BAAI/bge-m3).

Design (Leave-One-Out)
----------------------
For each model in the K=3 portfolio:

1. **Target**: One model is treated as the simulated newcomer (no prior).
2. **Base models**: The remaining K-1 models receive warmup priors from
   the canonical K=3 prior file.
3. **Condition A (semantic transfer)**: Add target via ``register_model()``
   with neighbor selected by within-provider tetrachoric correlation of
   binarized reward vectors.
4. **Condition B (tabula rasa)**: Add target with identity init
   (A=lambda*I, b=0) -- no transfer.
5. Both conditions use the same BanditRouter (Corralling + Hybrid LinUCB),
   same data, same seeds, same shuffled training order.

Data Separation (same as Figures 4/5)
-------------------------------------
- **Prior-train pool**: excluded from online learning (used only for
  warmup priors; defined in ``splits_three_way.json``).
- **Dev-train** (80% of online-learn pool): the online routing stream.
- **Dev-val** (20% of online-learn pool): reserved (hyperparameters
  from Appendix H).
- **Holdout**: reserved for frozen evaluation only.

Outputs (``results/``)
    semantic_transfer_results.json
    semantic_transfer_summary.txt
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy import stats as sp_stats
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
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
)
from utils.router_factory import create_experiment_router
from utils.model_pricing import load_model_catalog
from utils.embeddings import load_embedding_cache, embed_dataset_cached
from utils.transfer import (
    build_reward_vectors,
    find_tetrachoric_neighbor,
    build_filtered_warmup,
)
from utils.metrics import holm_bonferroni, cohens_d_paired

from run_prequential import (
    load_rewards_from_file,
    build_model_registry,
    _split_dev_train_val,
    _make_learning_curve_checkpoints,
    N_SEEDS,
    SEED_OFFSET,
    DEV_VAL_FRACTION,
    CORRALLING_LR,
    CORRALLING_GAMMA,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# K=3 model catalog (consistent with Figures 4/5)
# ============================================================================

K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)


# ============================================================================
# Hyperparameter loading (Appendix H)
# ============================================================================

HPARAMS_DIR = (
    PROJECT_ROOT / "experiments" / "appendix"
    / "H_alpha_neff_ablation" / "results"
)


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


# ============================================================================
# Single trial
# ============================================================================


def run_trial(
    base_models: List[str],
    target_model: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    r_min: float,
    r_max: float,
    warmup_path: Path,
    use_transfer: bool,
    seed: int,
    encoder: SentenceTransformer,
    feature_dim: int,
    tuned_warmup: Dict[str, float],
    tuned_tr: Dict[str, float],
    precomputed_neighbor: Optional[Tuple[Optional[str], float]] = None,
) -> Dict[str, Any]:
    """Run one trial for a single target model.

    Builds a router with K-1 warmup priors, adds the target via
    ``register_model()`` (condition A) or with identity init
    (condition B), trains on *train_data*, and evaluates frozen on
    *eval_data*.

    Args:
        base_models: The K-1 models receiving warmup priors.
        target_model: The simulated newcomer.
        train_data: Dev-train prompts with rewards.
        eval_data: Holdout prompts with rewards.
        train_emb: Pre-computed embeddings for dev-train.
        eval_emb: Pre-computed embeddings for holdout.
        r_min: Reward normalization lower bound.
        r_max: Reward normalization upper bound.
        warmup_path: Path to filtered warmup priors (K-1 models).
        use_transfer: If True, condition A (semantic transfer);
            if False, condition B (tabula rasa).
        seed: Random seed for this trial.
        encoder: Shared SentenceTransformer instance.
        feature_dim: Context vector dimensionality (PCA + bias).
        tuned_warmup: Appendix H hparams for warmup expert.
        tuned_tr: Appendix H hparams for tabula-rasa expert.
        precomputed_neighbor: ``(neighbor_id, similarity)`` for the
            target model's best tetrachoric neighbor, or ``None``.

    Returns:
        Dict with ``holdout_reward``, ``actually_transferred``,
        ``neighbor_used``, ``similarity``.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    burn_in = len(train_data)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    perm = rng.permutation(burn_in)

    router = create_experiment_router(
        model_registry=build_model_registry(base_models, K3_CATALOG),
        feature_dim=feature_dim,
        prior_n_effective=tuned_warmup["prior_n_effective"],
        alpha=tuned_warmup["alpha"],
        warmup_path=str(warmup_path),
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=0.0,
        forgetting_factor=tuned_warmup["forgetting_factor"],
        tabula_rasa_alpha=tuned_tr["alpha"],
        tabula_rasa_forgetting_factor=tuned_tr["forgetting_factor"],
    )
    target_registry = build_model_registry([target_model], K3_CATALOG)
    router.registry[target_model] = target_registry[target_model]

    transfer_info: Dict[str, Any] = {
        "actually_transferred": False,
        "neighbor_used": None,
        "similarity": None,
    }

    orig_admix = router.admix_theta_from_neighbors
    if not use_transfer:
        def _no_transfer(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
            bandit = router.bandit
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64),
            )
        router.admix_theta_from_neighbors = _no_transfer
    else:
        tet_nb = precomputed_neighbor

        def _tetrachoric_admix(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
            kwargs["precomputed_neighbor"] = tet_nb
            A, b = orig_admix(*args, **kwargs)
            actually_transferred = np.linalg.norm(b) > 1e-12
            transfer_info["actually_transferred"] = actually_transferred
            if tet_nb:
                transfer_info["neighbor_used"] = tet_nb[0]
                transfer_info["similarity"] = tet_nb[1]
            return A, b

        router.admix_theta_from_neighbors = _tetrachoric_admix

    router.encoder = encoder
    router.register_model(target_model, speed="balanced")
    router.encoder = None

    router.admix_theta_from_neighbors = orig_admix

    for idx in perm:
        p = train_data[idx]
        x = train_emb[idx]
        model, log = router.route(x, total_steps=burn_in)
        norm_r = (p["rewards"][model] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

    rng_state = np.random.get_state()
    total_r = 0.0
    for p, x in zip(eval_data, eval_emb):
        model, _ = router.route(x, total_steps=burn_in)
        total_r += p["rewards"][model]
    np.random.set_state(rng_state)

    return {
        "holdout_reward": total_r / len(eval_data),
        "actually_transferred": transfer_info["actually_transferred"],
        "neighbor_used": transfer_info["neighbor_used"],
        "similarity": transfer_info["similarity"],
    }


# ============================================================================
# Learning-curve trial
# ============================================================================


def run_learning_curve_trial(
    base_models: List[str],
    target_model: str,
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    r_min: float,
    r_max: float,
    warmup_path: Path,
    use_transfer: bool,
    seed: int,
    encoder: SentenceTransformer,
    feature_dim: int,
    tuned_warmup: Dict[str, float],
    tuned_tr: Dict[str, float],
    checkpoints: List[int],
    precomputed_neighbor: Optional[Tuple[Optional[str], float]] = None,
) -> List[Dict[str, Any]]:
    """Run one trial collecting holdout reward at each checkpoint.

    Returns a list of dicts, one per checkpoint, with ``step`` and
    ``holdout_reward``.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    burn_in = len(train_data)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    perm = rng.permutation(burn_in)
    checkpoint_set = set(checkpoints)

    router = create_experiment_router(
        model_registry=build_model_registry(base_models, K3_CATALOG),
        feature_dim=feature_dim,
        prior_n_effective=tuned_warmup["prior_n_effective"],
        alpha=tuned_warmup["alpha"],
        warmup_path=str(warmup_path),
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=0.0,
        forgetting_factor=tuned_warmup["forgetting_factor"],
        tabula_rasa_alpha=tuned_tr["alpha"],
        tabula_rasa_forgetting_factor=tuned_tr["forgetting_factor"],
    )
    target_registry = build_model_registry([target_model], K3_CATALOG)
    router.registry[target_model] = target_registry[target_model]

    orig_admix = router.admix_theta_from_neighbors
    if not use_transfer:
        def _no_transfer(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
            bandit = router.bandit
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64),
            )
        router.admix_theta_from_neighbors = _no_transfer
    else:
        tet_nb = precomputed_neighbor

        def _tetrachoric_admix(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
            kwargs["precomputed_neighbor"] = tet_nb
            return orig_admix(*args, **kwargs)

        router.admix_theta_from_neighbors = _tetrachoric_admix

    router.encoder = encoder
    router.register_model(target_model, speed="balanced")
    router.encoder = None
    router.admix_theta_from_neighbors = orig_admix

    results: List[Dict[str, Any]] = []

    def _evaluate_frozen() -> float:
        rng_state = np.random.get_state()
        total = 0.0
        for p, x in zip(eval_data, eval_emb):
            m, _ = router.route(x, total_steps=burn_in)
            total += p["rewards"][m]
        np.random.set_state(rng_state)
        return total / len(eval_data)

    if 0 in checkpoint_set:
        results.append({"step": 0, "holdout_reward": _evaluate_frozen()})

    for step_idx, idx in enumerate(perm, start=1):
        p = train_data[idx]
        x = train_emb[idx]
        model, log = router.route(x, total_steps=burn_in)
        norm_r = (p["rewards"][model] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

        if step_idx in checkpoint_set:
            results.append({"step": step_idx, "holdout_reward": _evaluate_frozen()})

    return results


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:
    """Run the semantic transfer evaluation for K=3."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("FIGURE 6: SEMANTIC TRANSFER EVALUATION (K=3)")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    logger.info("\n  Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    feature_dim = pca.n_components_ + 1

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    # ------------------------------------------------------------------
    # Hyperparameters from Appendix H
    # ------------------------------------------------------------------
    tuned_warmup = _load_hparams(HPARAMS_DIR / "best_hparams_k3.json", "K3")
    tuned_tr = _load_hparams(
        HPARAMS_DIR / "best_hparams_k3_tabula_rasa.json", "K3",
    )
    logger.info(
        f"  Warmup hparams: alpha={tuned_warmup['alpha']} "
        f"n_eff={tuned_warmup['prior_n_effective']} "
        f"gamma={tuned_warmup['forgetting_factor']}"
    )
    logger.info(
        f"  TR hparams: alpha={tuned_tr['alpha']} "
        f"n_eff={tuned_tr['prior_n_effective']} "
        f"gamma={tuned_tr['forgetting_factor']}"
    )

    # ------------------------------------------------------------------
    # Validate warmup priors
    # ------------------------------------------------------------------
    if not K3_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=3 warmup priors not found: {K3_WARMUP_PRIORS_PATH}\n"
            "Generate with: python scripts/extract_warmup_from_multimodel.py"
        )

    # ------------------------------------------------------------------
    # Load data with proper separation
    # ------------------------------------------------------------------
    logger.info("\n  Loading K=3 data ...")

    prior_train_prompts: set = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(
            f"    Excluding {len(prior_train_prompts)} prior-train prompts"
        )

    all_dev = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K3_MODELS)
    dev_data = [d for d in all_dev if d["prompt"] not in prior_train_prompts]
    holdout_data = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"    Dev (excl. prior-train): {len(dev_data)} prompts")
    logger.info(f"    Holdout: {len(holdout_data)} prompts")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    logger.info("  Embedding prompts ...")
    dev_emb = embed_dataset_cached(dev_data, embedding_cache, encoder, pca)
    holdout_emb = embed_dataset_cached(holdout_data, embedding_cache, encoder, pca)

    # ------------------------------------------------------------------
    # Dev train/val split
    # ------------------------------------------------------------------
    logger.info(
        f"  Splitting dev into train/val "
        f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ..."
    )
    train_data, train_emb, val_data, val_emb = _split_dev_train_val(
        dev_data, dev_emb,
    )
    logger.info(
        f"    Dev-train: {len(train_data)}  Dev-val: {len(val_data)}"
    )
    logger.info(f"    Feature dim: {feature_dim}")

    # ------------------------------------------------------------------
    # Reward normalization bounds
    # ------------------------------------------------------------------
    all_r = [p["rewards"][m] for p in train_data for m in K3_MODELS]
    r_min, r_max = min(all_r), max(all_r)

    # ------------------------------------------------------------------
    # Tetrachoric neighbor selection
    # ------------------------------------------------------------------
    logger.info("\n  Computing binarized reward vectors ...")
    reward_vectors = build_reward_vectors(train_data, K3_MODELS)

    logger.info("  Within-provider tetrachoric neighbors:")
    neighbor_map: Dict[str, Tuple[Optional[str], float]] = {}
    for m in K3_MODELS:
        base = [x for x in K3_MODELS if x != m]
        nb, sim = find_tetrachoric_neighbor(
            m, base, reward_vectors, within_provider_only=True,
        )
        neighbor_map[m] = (nb, sim)
        disp_m = K3_CATALOG[m]["display"]
        disp_nb = (
            K3_CATALOG[nb]["display"] if nb else "(no same-provider peer)"
        )
        logger.info(f"    {disp_m:<25} -> {disp_nb:<25} (r_tet={sim:.3f})")

    # ------------------------------------------------------------------
    # Leave-one-out trials
    # ------------------------------------------------------------------
    logger.info(
        f"\n  Running leave-one-out ({len(K3_MODELS)} targets x "
        f"{N_SEEDS} seeds x 2 conditions) ..."
    )

    per_target_results: Dict[str, Dict[str, Any]] = {}

    for target in K3_MODELS:
        base = [m for m in K3_MODELS if m != target]
        display = K3_CATALOG[target]["display"]
        best_nb, best_sim = neighbor_map[target]

        if best_nb:
            disp_nb = K3_CATALOG[best_nb]["display"]
        else:
            disp_nb = "(no same-provider peer)"
        logger.info(
            f"\n  Target: {display}  ->  neighbor: {disp_nb} "
            f"(r_tet={best_sim:.3f})"
        )

        warmup_filtered = build_filtered_warmup(base, K3_WARMUP_PRIORS_PATH)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = Path(f.name)
        try:
            joblib.dump(warmup_filtered, temp_path)

            transfer_rewards: List[float] = []
            tabula_rewards: List[float] = []
            transfer_received: List[bool] = []

            for t in range(N_SEEDS):
                seed = SEED_OFFSET + t
                res_t = run_trial(
                    base, target, train_data, holdout_data,
                    train_emb, holdout_emb, r_min, r_max,
                    temp_path, use_transfer=True, seed=seed,
                    encoder=encoder, feature_dim=feature_dim,
                    tuned_warmup=tuned_warmup, tuned_tr=tuned_tr,
                    precomputed_neighbor=(best_nb, best_sim) if best_nb else None,
                )
                res_b = run_trial(
                    base, target, train_data, holdout_data,
                    train_emb, holdout_emb, r_min, r_max,
                    temp_path, use_transfer=False, seed=seed,
                    encoder=encoder, feature_dim=feature_dim,
                    tuned_warmup=tuned_warmup, tuned_tr=tuned_tr,
                )
                transfer_rewards.append(res_t["holdout_reward"])
                tabula_rewards.append(res_b["holdout_reward"])
                transfer_received.append(res_t["actually_transferred"])

            mean_transfer = float(np.mean(transfer_rewards))
            mean_tabula = float(np.mean(tabula_rewards))
            std_transfer = (
                float(np.std(transfer_rewards, ddof=1))
                if N_SEEDS > 1 else 0.0
            )
            std_tabula = (
                float(np.std(tabula_rewards, ddof=1))
                if N_SEEDS > 1 else 0.0
            )
            diffs = np.array(transfer_rewards) - np.array(tabula_rewards)
            if np.std(diffs) < 1e-15:
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = sp_stats.ttest_rel(
                    transfer_rewards, tabula_rewards,
                )
            t_crit = (
                float(sp_stats.t.ppf(0.975, N_SEEDS - 1))
                if N_SEEDS > 1 else 0.0
            )
            ci_half = t_crit * float(np.std(diffs, ddof=1)) / (N_SEEDS ** 0.5)
            d = cohens_d_paired(transfer_rewards, tabula_rewards)
            n_transferred = int(sum(transfer_received))

            per_target_results[target] = {
                "display": display,
                "transfer": {
                    "mean": mean_transfer,
                    "std": std_transfer,
                    "per_trial": transfer_rewards,
                },
                "tabula_rasa": {
                    "mean": mean_tabula,
                    "std": std_tabula,
                    "per_trial": tabula_rewards,
                },
                "delta_mean": float(np.mean(diffs)),
                "delta_ci95": float(ci_half),
                "p_value_uncorrected": float(p_val),
                "t_statistic": float(t_stat),
                "cohens_d": d,
                "n_trials_with_actual_transfer": n_transferred,
                "n_trials_total": N_SEEDS,
                "tetrachoric_neighbor": best_nb,
                "tetrachoric_neighbor_display": disp_nb,
                "tetrachoric_similarity": best_sim,
            }

            sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
            logger.info(
                f"    Transfer:  {mean_transfer:.4f} +/- {std_transfer:.4f}  |  "
                f"Tabula: {mean_tabula:.4f} +/- {std_tabula:.4f}  |  "
                f"Delta={np.mean(diffs):+.4f}  p={p_val:.4f}{sig}  d={d:.3f}  "
                f"({n_transferred}/{N_SEEDS} xfer)"
            )
        finally:
            temp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Holm-Bonferroni correction
    # ------------------------------------------------------------------
    targets_ordered = list(per_target_results.keys())
    raw_ps = [per_target_results[t]["p_value_uncorrected"] for t in targets_ordered]
    adjusted_ps = holm_bonferroni(raw_ps)
    for i, t in enumerate(targets_ordered):
        per_target_results[t]["p_value_holm"] = adjusted_ps[i]

    # ------------------------------------------------------------------
    # Aggregate portfolio-level test
    # ------------------------------------------------------------------
    all_transfer: List[float] = []
    all_tabula: List[float] = []
    for t in targets_ordered:
        all_transfer.extend(per_target_results[t]["transfer"]["per_trial"])
        all_tabula.extend(per_target_results[t]["tabula_rasa"]["per_trial"])
    agg_diffs = np.array(all_transfer) - np.array(all_tabula)
    if np.std(agg_diffs) < 1e-15:
        agg_t, agg_p = 0.0, 1.0
    else:
        agg_t, agg_p = sp_stats.ttest_rel(all_transfer, all_tabula)
    agg_d = cohens_d_paired(all_transfer, all_tabula)

    aggregate = {
        "t_statistic": float(agg_t),
        "p_value": float(agg_p),
        "cohens_d": agg_d,
        "n_pairs": len(all_transfer),
        "mean_delta": float(np.mean(agg_diffs)),
    }

    logger.info(
        f"\n  Aggregate: Delta={aggregate['mean_delta']:+.4f}  "
        f"p={agg_p:.4f}  d={agg_d:.3f}  (N={len(all_transfer)} pairs)"
    )

    # ------------------------------------------------------------------
    # Learning curves for targets that received transfer
    # ------------------------------------------------------------------
    transferred_targets = [
        t for t in targets_ordered
        if per_target_results[t]["n_trials_with_actual_transfer"] > 0
    ]

    learning_curves: Dict[str, Dict[str, Any]] = {}
    if transferred_targets:
        checkpoints = _make_learning_curve_checkpoints(len(train_data))
        logger.info(
            f"\n  Learning curves for {len(transferred_targets)} "
            f"transferred targets, checkpoints: {checkpoints}"
        )

        for target in transferred_targets:
            base = [m for m in K3_MODELS if m != target]
            display = K3_CATALOG[target]["display"]
            best_nb, best_sim = neighbor_map[target]

            warmup_filtered = build_filtered_warmup(
                base, K3_WARMUP_PRIORS_PATH,
            )
            with tempfile.NamedTemporaryFile(
                suffix=".joblib", delete=False,
            ) as f:
                temp_path = Path(f.name)
            try:
                joblib.dump(warmup_filtered, temp_path)

                lc_transfer: Dict[int, List[float]] = {
                    s: [] for s in checkpoints
                }
                lc_tabula: Dict[int, List[float]] = {
                    s: [] for s in checkpoints
                }

                for t in range(N_SEEDS):
                    seed = SEED_OFFSET + t
                    curve_t = run_learning_curve_trial(
                        base, target, train_data, holdout_data,
                        train_emb, holdout_emb, r_min, r_max,
                        temp_path, use_transfer=True, seed=seed,
                        encoder=encoder, feature_dim=feature_dim,
                        tuned_warmup=tuned_warmup, tuned_tr=tuned_tr,
                        checkpoints=checkpoints,
                        precomputed_neighbor=(best_nb, best_sim),
                    )
                    curve_b = run_learning_curve_trial(
                        base, target, train_data, holdout_data,
                        train_emb, holdout_emb, r_min, r_max,
                        temp_path, use_transfer=False, seed=seed,
                        encoder=encoder, feature_dim=feature_dim,
                        tuned_warmup=tuned_warmup, tuned_tr=tuned_tr,
                        checkpoints=checkpoints,
                    )
                    for pt in curve_t:
                        lc_transfer[pt["step"]].append(pt["holdout_reward"])
                    for pt in curve_b:
                        lc_tabula[pt["step"]].append(pt["holdout_reward"])

                learning_curves[target] = {
                    "display": display,
                    "neighbor": K3_CATALOG[best_nb]["display"],
                    "tetrachoric_similarity": best_sim,
                    "checkpoints": checkpoints,
                    "transfer": {
                        s: {
                            "mean": float(np.mean(lc_transfer[s])),
                            "std": float(np.std(lc_transfer[s], ddof=1))
                            if len(lc_transfer[s]) > 1 else 0.0,
                        }
                        for s in checkpoints
                    },
                    "tabula_rasa": {
                        s: {
                            "mean": float(np.mean(lc_tabula[s])),
                            "std": float(np.std(lc_tabula[s], ddof=1))
                            if len(lc_tabula[s]) > 1 else 0.0,
                        }
                        for s in checkpoints
                    },
                }

                logger.info(f"    {display}: done")
            finally:
                temp_path.unlink(missing_ok=True)
    else:
        logger.info("\n  No targets received transfer -- skipping learning curves.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    header = (
        f"  {'Target':<22} {'Neighbor':<22} {'r_tet':>5} "
        f"{'Delta':>8} {'CI95':>8} {'p(raw)':>8} {'p(Holm)':>8} "
        f"{'d':>7} {'Xfer':>5}"
    )
    logger.info(header)
    logger.info("  " + "-" * 100)
    for t in targets_ordered:
        r = per_target_results[t]
        logger.info(
            f"  {r['display']:<22} {r['tetrachoric_neighbor_display']:<22} "
            f"{r['tetrachoric_similarity']:5.3f} "
            f"{r['delta_mean']:+8.4f} {r['delta_ci95']:8.4f} "
            f"{r['p_value_uncorrected']:8.4f} {r['p_value_holm']:8.4f} "
            f"{r['cohens_d']:+7.3f} "
            f"{r['n_trials_with_actual_transfer']:>2}/{r['n_trials_total']}"
        )
    logger.info(
        f"  {'AGGREGATE':<22} {'':22} {'':>5} "
        f"{aggregate['mean_delta']:+8.4f} {'':8} "
        f"{aggregate['p_value']:8.4f} {'---':>8} "
        f"{aggregate['cohens_d']:+7.3f} "
        f"{aggregate['n_pairs']:>4}p"
    )

    # ------------------------------------------------------------------
    # Serialize
    # ------------------------------------------------------------------
    results_all = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Figure 6: Semantic Transfer (K=3)",
            "n_seeds": N_SEEDS,
            "n_models": len(K3_MODELS),
            "feature_dim": feature_dim,
            "pca_components": pca.n_components_,
            "encoder": DEFAULT_SENTENCE_TRANSFORMER,
            "warmup_priors_path": str(K3_WARMUP_PRIORS_PATH),
            "warmup_hparams": tuned_warmup,
            "tabula_rasa_hparams": tuned_tr,
            "neighbor_selection": "within_provider_tetrachoric_correlation",
        },
        "data_separation": {
            "online_stream": "dev-train (80% of online-learn pool, excl. prior-train)",
            "frozen_evaluation": "holdout (canonical holdout set)",
            "dev_val": "unused (hparams from Appendix H)",
            "n_prior_train_excluded": len(prior_train_prompts),
            "n_dev_train": len(train_data),
            "n_dev_val": len(val_data),
            "n_holdout": len(holdout_data),
        },
        "models": K3_MODELS,
        "model_displays": {m: K3_CATALOG[m]["display"] for m in K3_MODELS},
        "per_target": per_target_results,
        "aggregate": aggregate,
        "learning_curves": learning_curves,
    }

    out_json = output_dir / "semantic_transfer_results.json"
    with open(out_json, "w") as f:
        json.dump(results_all, f, indent=2)

    # Text summary
    summary_lines = [
        "Semantic Transfer Evaluation: Leave-One-Out (K=3)",
        "=" * 60,
        f"Design: Paired, {N_SEEDS} seeds/target, shuffled training order",
        f"Encoder: {DEFAULT_SENTENCE_TRANSFORMER}",
        f"PCA: {pca.n_components_} components (feature_dim={feature_dim})",
        "Neighbor: Within-provider tetrachoric correlation",
        "Stat: Paired t-test (uncorrected + Holm-Bonferroni), Cohen's d",
        "",
        header.strip(),
        "-" * 100,
    ]
    for t in targets_ordered:
        r = per_target_results[t]
        summary_lines.append(
            f"{r['display']:<22} {r['tetrachoric_neighbor_display']:<22} "
            f"{r['tetrachoric_similarity']:5.3f} "
            f"{r['delta_mean']:+8.4f} {r['delta_ci95']:8.4f} "
            f"{r['p_value_uncorrected']:8.4f} {r['p_value_holm']:8.4f} "
            f"{r['cohens_d']:+7.3f} "
            f"{r['n_trials_with_actual_transfer']:>2}/{r['n_trials_total']}"
        )
    summary_lines.append(
        f"{'AGGREGATE':<22} {'':22} {'':>5} "
        f"{aggregate['mean_delta']:+8.4f} {'':8} "
        f"{aggregate['p_value']:8.4f} {'---':>8} "
        f"{aggregate['cohens_d']:+7.3f} "
        f"{aggregate['n_pairs']:>4}p"
    )

    summary_text = "\n".join(summary_lines)
    out_summary = output_dir / "semantic_transfer_summary.txt"
    with open(out_summary, "w") as f:
        f.write(summary_text)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_json}")
    logger.info(f"Summary -> {out_summary}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_experiment()
