#!/usr/bin/env python3
"""
K=3 Multi-Model Pareto Frontier Evaluation.

Evaluates BanditGPT's routing performance across a 3-model portfolio
comparing against tabula rasa (plain LinUCB) and standard baselines.
This experiment tests whether BanditGPT's architecture (Corralling over
heterogeneous LinUCB experts with family-based parameter sharing) scales
beyond the K=2 setting.

Protocol
--------
1. **Data source.**
   Training data comes from the online-learn pool of a three-way
   stratified split; the holdout is the canonical holdout set.
   Rewards are ground-truth multi-judge scores via ``extract_reward()``
   (mean of vote x confidence).

2. **Train-then-freeze evaluation.**
   BanditGPT trains on the dev-train split (80% of the online-learn
   pool) with oracle rewards, then is frozen for holdout evaluation.
   Greedy exploitation (alpha=0) during evaluation.

3. **Dev-selected deployable Pareto frontier.**
   The dev set is split 80/20 into train/val.  The Pareto hull is
   built from (dev_val_cost, dev_val_reward); holdout performance
   of dev-optimal hyperparameters is the primary metric.

4. **Full-dev retrain pass.**
   After dev-val selection identifies the Pareto-optimal lambda values,
   a second training pass retrains the router on the **full** dev set
   for those lambda values only.  Holdout metrics are updated in place
   while preserving the dev-val selection (no holdout leakage).  This
   reduces estimator variance.

5. **Baselines.**
   Oracle, best-static, best-static-plus-noise, UCB1 (non-contextual),
   random, and tabula rasa (BanditGPT without priors or Corralling).

6. **Statistical reporting.**
   Paired bootstrap CI for dev-selected Pareto AUC difference
   (1,000 holdout resamples; dev indices fixed before bootstrapping).

7. **Tuned hyperparameters.**
   If Appendix H ablation results are available, dev-val-selected
   (alpha, prior_n_effective, forgetting_factor) are loaded and used.
   Otherwise, module-level defaults from ``run_prequential`` apply.

Outputs (``results/``)
    multimodel_pareto_results.json

This script imports shared evaluation functions and model catalogs from
``experiments/03_figure/run_prequential.py`` to avoid code duplication.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "03_figure"))

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    K3_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
)

from utils.pareto import (
    pareto_auc,
    dev_pareto_indices,
    dev_selected_pareto_auc,
    bootstrap_pareto_auc_difference,
    extract_dev_optimal_per_prompt,
)

from run_prequential import (
    load_rewards_from_file,
    oracle_route,
    static_route,
    random_route,
    best_static_noisy_route,
    ucb1_online_route,
    run_pareto_sweep,
    _split_dev_train_val,
    K3_MODELS,
    K3_CATALOG,
    LAMBDA_VALUES_K3,
    N_SEEDS,
    TARGET_NEFF,
    ALPHA_START,
    CORRALLING_LR,
    CORRALLING_GAMMA,
    DEV_VAL_FRACTION,
    DEV_VAL_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Main K=3 experiment
# ============================================================================


def _load_tuned_hparams(key: str = "K3") -> Optional[Dict[str, float]]:
    """Load dev-val-selected hyperparameters from Appendix H ablation.

    Args:
        key: Top-level key in the JSON file (``"K2"`` or ``"K3"``).

    Returns:
        Dict with ``alpha``, ``prior_n_effective``, ``forgetting_factor``,
        or ``None`` if the file is missing or malformed.
    """
    hparams_path = (
        Path(__file__).resolve().parent.parent / "appendix"
        / "H_alpha_neff_ablation" / "results" / "best_hparams_k3.json"
    )
    if not hparams_path.exists():
        return None
    try:
        data = json.loads(hparams_path.read_text())
        cfg = data.get(key, {})
        return {
            "alpha": float(cfg["alpha"]),
            "prior_n_effective": float(cfg["n_eff"]),
            "forgetting_factor": float(cfg["gamma"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to load hparams from %s: %s", hparams_path, exc)
        return None


def run_k3_experiment() -> None:
    """Run the K=3 multi-model Pareto frontier evaluation."""
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

    from utils.embeddings import load_embedding_cache, embed_dataset_cached
    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    # ------------------------------------------------------------------
    # Tuned hyperparameters (Appendix H, dev-val-selected)
    # ------------------------------------------------------------------
    tuned = _load_tuned_hparams("K3")
    if tuned is not None:
        logger.info(
            f"Loaded K=3 tuned hparams (dev-val-selected): "
            f"alpha={tuned['alpha']} n_eff={tuned['prior_n_effective']} "
            f"forgetting_factor={tuned['forgetting_factor']}"
        )
    else:
        logger.warning(
            "Appendix H results not found. Falling back to module-level "
            f"defaults (alpha={ALPHA_START}, n_eff={TARGET_NEFF}). Run "
            "experiments/appendix/H_alpha_neff_ablation/"
            "run_3d_grid_ablation.py first for tuned hyperparameters."
        )
    k3_alpha = tuned["alpha"] if tuned is not None else ALPHA_START
    k3_neff = tuned["prior_n_effective"] if tuned is not None else TARGET_NEFF
    k3_forgetting = tuned["forgetting_factor"] if tuned is not None else 1.0

    results_all: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": N_SEEDS,
            "reward_source": "extract_reward(mean_vote_x_confidence)",
            "protocol": "train_then_freeze_with_full_dev_retrain",
            "split_protocol": "three_way_split",
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
            "hparams": {
                "alpha": k3_alpha,
                "prior_n_effective": k3_neff,
                "forgetting_factor": k3_forgetting,
                "corralling_lr": CORRALLING_LR,
                "corralling_gamma": CORRALLING_GAMMA,
                "source": "appendix_H" if tuned is not None else "module_defaults",
            },
        },
    }

    # ==================================================================
    # K=3 — Multi-model Pareto frontier
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=3: Multi-Model Pareto Frontier")
    logger.info("=" * 70)

    costs_k3 = {m: K3_CATALOG[m]["cost"] for m in K3_MODELS}

    # --- Load K=3 data ------------------------------------------------
    logger.info("\n  Loading K=3 data ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits_3way = json.load(f)
    online_prompts = set(splits_3way["online_learn_pool"])

    train_data_k3 = load_rewards_from_file(
        DEV_DATA_PATH_ALL_MODELS, K3_MODELS,
        prompt_filter=online_prompts,
    )
    holdout_data_k3 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"    Train (online-learn): {len(train_data_k3)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k3)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info("  Embedding K=3 prompts ...")
    train_emb_k3 = embed_dataset_cached(train_data_k3, embedding_cache, encoder, pca)
    holdout_emb_k3 = embed_dataset_cached(holdout_data_k3, embedding_cache, encoder, pca)

    # --- Dev train/val split -------------------------------------------
    logger.info(f"  Splitting K=3 train into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    train_train_k3, train_train_emb_k3, train_val_k3, train_val_emb_k3 = (
        _split_dev_train_val(train_data_k3, train_emb_k3)
    )
    logger.info(
        f"    Train-train: {len(train_train_k3)}  "
        f"Train-val: {len(train_val_k3)}"
    )

    # --- Baselines -----------------------------------------------------
    logger.info("\n  Computing K=3 baselines ...")
    oracle_r_k3, oracle_c_k3 = oracle_route(
        holdout_data_k3, K3_MODELS, costs_k3,
    )
    logger.info(f"    Oracle: R={oracle_r_k3:.4f}  C=${oracle_c_k3:.6f}")

    static_k3: Dict[str, Dict] = {}
    for m in K3_MODELS:
        sr, sc = static_route(holdout_data_k3, m, costs_k3)
        static_k3[m] = {"reward": sr, "cost": sc}
        logger.info(
            f"    Static {K3_CATALOG[m]['display']:<22}: R={sr:.4f}  C=${sc:.6f}"
        )

    random_k3 = random_route(
        holdout_data_k3, K3_MODELS, costs_k3, N_SEEDS * 4,
    )
    logger.info(f"    Random: R={random_k3['reward']:.4f}")

    eg_k3 = best_static_noisy_route(
        train_train_k3, holdout_data_k3, K3_MODELS, costs_k3,
        n_trials=N_SEEDS * 4,
    )
    logger.info(f"    Best-static+noise: R={eg_k3['reward']:.4f}")

    ucb1_k3 = ucb1_online_route(
        train_train_k3, holdout_data_k3, K3_MODELS, costs_k3,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(
        f"    UCB1 (non-contextual): R={ucb1_k3['reward']:.4f} "
        f"+/-{ucb1_k3['std_reward']:.4f}"
    )

    # --- BanditGPT Pareto sweep ----------------------------------------
    logger.info(
        f"\n  BanditGPT K=3 Pareto sweep "
        f"({len(LAMBDA_VALUES_K3)} lambda x {N_SEEDS} seeds) ..."
    )
    bandit_pareto_k3 = run_pareto_sweep(
        K3_MODELS, K3_CATALOG,
        train_train_k3, holdout_data_k3, train_train_emb_k3, holdout_emb_k3,
        str(K3_WARMUP_PRIORS_PATH), costs_k3, LAMBDA_VALUES_K3,
        N_SEEDS, use_corralling=True, label="banditGPT",
        dev_val_data=train_val_k3, dev_val_emb=train_val_emb_k3,
        alpha=k3_alpha,
        prior_n_effective=k3_neff,
        forgetting_factor=k3_forgetting,
    )

    # Tabula rasa ablation (no priors, no Corralling)
    logger.info(
        f"\n  Tabula rasa K=3 ablation "
        f"({len(LAMBDA_VALUES_K3)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k3 = run_pareto_sweep(
        K3_MODELS, K3_CATALOG,
        train_train_k3, holdout_data_k3, train_train_emb_k3, holdout_emb_k3,
        None, costs_k3, LAMBDA_VALUES_K3,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
        dev_val_data=train_val_k3, dev_val_emb=train_val_emb_k3,
        alpha=k3_alpha,
        prior_n_effective=k3_neff,
        forgetting_factor=k3_forgetting,
    )

    # --- Dev-selected Pareto AUC ----------------------------------------
    best_static_m = max(static_k3, key=lambda m: static_k3[m]["reward"])

    bg_dev_costs_k3 = [p["dev_mean_cost"] for p in bandit_pareto_k3]
    tr_dev_costs_k3 = [p["dev_mean_cost"] for p in tabula_pareto_k3]
    cost_lo_k3 = max(min(bg_dev_costs_k3), min(tr_dev_costs_k3))
    cost_hi_k3 = min(max(bg_dev_costs_k3), max(tr_dev_costs_k3))

    bg_ds_auc_k3, _, _, bg_dev_idx_k3 = dev_selected_pareto_auc(
        bandit_pareto_k3, cost_lo_k3, cost_hi_k3,
    )
    tr_ds_auc_k3, _, _, tr_dev_idx_k3 = dev_selected_pareto_auc(
        tabula_pareto_k3, cost_lo_k3, cost_hi_k3,
    )

    # Oracle envelope AUC (reference)
    bg_costs_k3 = [p["mean_cost"] for p in bandit_pareto_k3]
    tr_costs_k3 = [p["mean_cost"] for p in tabula_pareto_k3]
    oracle_cost_lo_k3 = max(min(bg_costs_k3), min(tr_costs_k3))
    oracle_cost_hi_k3 = min(max(bg_costs_k3), max(tr_costs_k3))
    bg_oracle_auc_k3 = pareto_auc(
        bg_costs_k3,
        [p["mean_reward"] for p in bandit_pareto_k3],
        oracle_cost_lo_k3, oracle_cost_hi_k3,
    )
    tr_oracle_auc_k3 = pareto_auc(
        tr_costs_k3,
        [p["mean_reward"] for p in tabula_pareto_k3],
        oracle_cost_lo_k3, oracle_cost_hi_k3,
    )

    # Paired bootstrap CI for K=3 AUC difference
    logger.info("  Computing K=3 bootstrap CI ...")
    bg_pp_r_k3: Dict[float, np.ndarray] = {}
    bg_pp_c_k3: Dict[float, np.ndarray] = {}
    for p in bandit_pareto_k3:
        if p.get("per_seed_per_prompt_rewards") is not None:
            bg_pp_r_k3[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            bg_pp_c_k3[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )
    tr_pp_r_k3: Dict[float, np.ndarray] = {}
    tr_pp_c_k3: Dict[float, np.ndarray] = {}
    for p in tabula_pareto_k3:
        if p.get("per_seed_per_prompt_rewards") is not None:
            tr_pp_r_k3[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            tr_pp_c_k3[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )

    bg_boot_pp_r_k3, bg_boot_pp_c_k3 = extract_dev_optimal_per_prompt(
        bandit_pareto_k3, bg_dev_idx_k3,
        bg_pp_r_k3, bg_pp_c_k3, "lambda",
    )
    tr_boot_pp_r_k3, tr_boot_pp_c_k3 = extract_dev_optimal_per_prompt(
        tabula_pareto_k3, tr_dev_idx_k3,
        tr_pp_r_k3, tr_pp_c_k3, "lambda",
    )
    bootstrap_k3 = bootstrap_pareto_auc_difference(
        bg_boot_pp_r_k3, bg_boot_pp_c_k3,
        tr_boot_pp_r_k3, tr_boot_pp_c_k3,
        cost_lo=cost_lo_k3, cost_hi=cost_hi_k3,
        n_holdout=len(holdout_data_k3), n_bootstrap=1_000,
    )

    logger.info(f"\n  K=3 SUMMARY (dev-selected Pareto AUC primary):")
    logger.info(f"    Oracle:       {oracle_r_k3:.4f}")
    logger.info(
        f"    Dev-selected AUC: BanditGPT={bg_ds_auc_k3:.4f} vs "
        f"Tabula rasa={tr_ds_auc_k3:.4f} "
        f"(adv: {bg_ds_auc_k3 - tr_ds_auc_k3:+.4f})"
    )
    logger.info(
        f"    Bootstrap 95% CI: [{bootstrap_k3['ci_95_lower']:+.4f}, "
        f"{bootstrap_k3['ci_95_upper']:+.4f}] "
        f"p={bootstrap_k3['p_value']:.4g}"
    )
    logger.info(
        f"    Oracle envelope (ref): BanditGPT={bg_oracle_auc_k3:.4f} vs "
        f"Tabula rasa={tr_oracle_auc_k3:.4f}"
    )
    logger.info(
        f"    Best static:  {static_k3[best_static_m]['reward']:.4f} "
        f"({K3_CATALOG[best_static_m]['display']})"
    )
    logger.info(f"    Best-static+noise: {eg_k3['reward']:.4f}")
    logger.info(f"    UCB1 (non-ctx):    {ucb1_k3['reward']:.4f}")
    logger.info(f"    Random:            {random_k3['reward']:.4f}")

    results_all["K3"] = {
        "models": [{"id": m, **K3_CATALOG[m]} for m in K3_MODELS],
        "n_train": len(train_data_k3),
        "n_holdout": len(holdout_data_k3),
        "oracle": {"reward": oracle_r_k3, "cost": oracle_c_k3},
        "static": {m: static_k3[m] for m in K3_MODELS},
        "best_static": {
            "model": best_static_m,
            "reward": static_k3[best_static_m]["reward"],
            "cost": static_k3[best_static_m]["cost"],
        },
        "random": random_k3,
        "best_static_noisy": eg_k3,
        "ucb1": ucb1_k3,
        "banditgpt_pareto": bandit_pareto_k3,
        "tabula_rasa_pareto": tabula_pareto_k3,
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k3, cost_hi_k3],
            "banditgpt": bg_ds_auc_k3,
            "tabula_rasa": tr_ds_auc_k3,
            "advantage": bg_ds_auc_k3 - tr_ds_auc_k3,
            "bootstrap_ci": bootstrap_k3,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward).  Deployed = holdout performance of dev-optimal "
                "hyperparameters.  Bootstrap CI with fixed dev indices."
            ),
        },
        "pareto_auc_oracle_envelope": {
            "cost_range": [oracle_cost_lo_k3, oracle_cost_hi_k3],
            "banditgpt": bg_oracle_auc_k3,
            "tabula_rasa": tr_oracle_auc_k3,
            "advantage": bg_oracle_auc_k3 - tr_oracle_auc_k3,
            "note": "Oracle envelope — holdout-selected hyperparameters (reference only).",
        },
        "n_trials": N_SEEDS,
    }

    # ==================================================================
    # Serialise (strip bulky per-prompt arrays to keep JSON manageable)
    # ==================================================================
    def _strip_per_prompt(obj: Any) -> Any:
        """Recursively drop per_prompt_rewards / per_seed_per_prompt_rewards."""
        if isinstance(obj, dict):
            return {
                k: _strip_per_prompt(v) for k, v in obj.items()
                if k not in (
                    "per_prompt_rewards", "per_prompt_costs",
                    "per_seed_per_prompt_rewards", "per_seed_per_prompt_costs",
                )
            }
        if isinstance(obj, list):
            return [_strip_per_prompt(v) for v in obj]
        return obj

    out_path = output_dir / "multimodel_pareto_results.json"
    with open(out_path, "w") as f:
        json.dump(_strip_per_prompt(results_all), f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_k3_experiment()
