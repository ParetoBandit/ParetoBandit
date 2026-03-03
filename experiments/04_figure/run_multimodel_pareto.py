#!/usr/bin/env python3
"""
K=10 Multi-Model Pareto Frontier Evaluation.

Evaluates BanditGPT's routing performance across a 10-model portfolio
spanning four cost tiers, comparing against tabula rasa (plain LinUCB)
and standard baselines.  RouteLLM does not natively support K > 2, so
this experiment tests whether BanditGPT's architecture (Corralling over
heterogeneous LinUCB experts with family-based parameter sharing) scales
to larger portfolios.

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

4. **Baselines.**
   Oracle, best-static, best-static-plus-noise, UCB1 (non-contextual),
   random, and tabula rasa (BanditGPT without priors or Corralling).

5. **Statistical reporting.**
   Paired bootstrap CI for dev-selected Pareto AUC difference
   (1,000 holdout resamples; dev indices fixed before bootstrapping).

Outputs (``results/``)
    multimodel_pareto_results.json

This script imports shared evaluation functions from
``experiments/03_figure/run_prequential.py`` to avoid code duplication.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "03_figure"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
)

from run_prequential import (
    _req_cost,
    load_rewards_from_file,
    build_model_registry,
    embed_dataset,
    oracle_route,
    static_route,
    random_route,
    best_static_noisy_route,
    ucb1_online_route,
    run_pareto_sweep,
    _split_dev_train_val,
    _pareto_hull,
    pareto_auc,
    dev_selected_pareto_auc,
    bootstrap_pareto_auc_difference,
    _extract_dev_optimal_per_prompt,
    N_SEEDS,
    SEED_OFFSET,
    TARGET_NEFF,
    ALPHA_START,
    CORRALLING_LR,
    CORRALLING_GAMMA,
    DEV_VAL_FRACTION,
    DEV_VAL_SEED,
)
from utils.model_pricing import get_prices_for_models

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# K=10 Model catalog
# ============================================================================

K10_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "google/gemma-3-27b-it",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-flash-preview-09-2025",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4.1",
]

_PRICES_K10 = get_prices_for_models(K10_MODELS)

K10_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES_K10["meta-llama/llama-3.1-8b-instruct"],
        "cost": _req_cost(
            _PRICES_K10["meta-llama/llama-3.1-8b-instruct"]["input_cost_per_m"],
            _PRICES_K10["meta-llama/llama-3.1-8b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        **_PRICES_K10["mistralai/mixtral-8x7b-instruct"],
        "cost": _req_cost(
            _PRICES_K10["mistralai/mixtral-8x7b-instruct"]["input_cost_per_m"],
            _PRICES_K10["mistralai/mixtral-8x7b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        **_PRICES_K10["google/gemma-3-27b-it"],
        "cost": _req_cost(
            _PRICES_K10["google/gemma-3-27b-it"]["input_cost_per_m"],
            _PRICES_K10["google/gemma-3-27b-it"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        **_PRICES_K10["anthropic/claude-haiku-4.5"],
        "cost": _req_cost(
            _PRICES_K10["anthropic/claude-haiku-4.5"]["input_cost_per_m"],
            _PRICES_K10["anthropic/claude-haiku-4.5"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        **_PRICES_K10["deepseek/deepseek-chat-v3-0324"],
        "cost": _req_cost(
            _PRICES_K10["deepseek/deepseek-chat-v3-0324"]["input_cost_per_m"],
            _PRICES_K10["deepseek/deepseek-chat-v3-0324"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        **_PRICES_K10["google/gemini-2.5-flash-preview-09-2025"],
        "cost": _req_cost(
            _PRICES_K10["google/gemini-2.5-flash-preview-09-2025"]["input_cost_per_m"],
            _PRICES_K10["google/gemini-2.5-flash-preview-09-2025"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        **_PRICES_K10["meta-llama/llama-4-maverick"],
        "cost": _req_cost(
            _PRICES_K10["meta-llama/llama-4-maverick"]["input_cost_per_m"],
            _PRICES_K10["meta-llama/llama-4-maverick"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        **_PRICES_K10["anthropic/claude-sonnet-4"],
        "cost": _req_cost(
            _PRICES_K10["anthropic/claude-sonnet-4"]["input_cost_per_m"],
            _PRICES_K10["anthropic/claude-sonnet-4"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        **_PRICES_K10["openai/gpt-4-turbo"],
        "cost": _req_cost(
            _PRICES_K10["openai/gpt-4-turbo"]["input_cost_per_m"],
            _PRICES_K10["openai/gpt-4-turbo"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES_K10["openai/gpt-4.1"],
        "cost": _req_cost(
            _PRICES_K10["openai/gpt-4.1"]["input_cost_per_m"],
            _PRICES_K10["openai/gpt-4.1"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
}

LAMBDA_VALUES_K10: List[float] = [
    0.0, 0.01, 0.03, 0.05, 0.07, 0.08, 0.09, 0.095,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
    0.185, 0.19, 0.192, 0.195, 0.198, 0.2, 0.202, 0.205,
    0.208, 0.21, 0.215, 0.22, 0.25, 0.3, 0.5, 1.0,
]


# ============================================================================
# Main K=10 experiment
# ============================================================================


def run_k10_experiment() -> None:
    """Run the K=10 multi-model Pareto frontier evaluation."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components")

    results_all: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": N_SEEDS,
            "reward_source": "extract_reward(mean_vote_x_confidence)",
            "protocol": "train_then_freeze",
            "split_protocol": "three_way_split",
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
        },
    }

    # ==================================================================
    # K=10 — Multi-model Pareto frontier
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=10: Multi-Model Pareto Frontier")
    logger.info("=" * 70)

    costs_k10 = {m: K10_CATALOG[m]["cost"] for m in K10_MODELS}

    # --- Load K=10 data ------------------------------------------------
    logger.info("\n  Loading K=10 data ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits_3way = json.load(f)
    online_prompts = set(splits_3way["online_learn_pool"])

    train_data_k10 = load_rewards_from_file(
        DEV_DATA_PATH_ALL_MODELS, K10_MODELS,
        prompt_filter=online_prompts,
    )
    holdout_data_k10 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K10_MODELS,
    )
    logger.info(f"    Train (online-learn): {len(train_data_k10)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k10)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info("  Embedding K=10 prompts ...")
    train_emb_k10 = embed_dataset(train_data_k10, encoder, pca)
    holdout_emb_k10 = embed_dataset(holdout_data_k10, encoder, pca)

    # --- Dev train/val split -------------------------------------------
    logger.info(f"  Splitting K=10 train into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    train_train_k10, train_train_emb_k10, train_val_k10, train_val_emb_k10 = (
        _split_dev_train_val(train_data_k10, train_emb_k10)
    )
    logger.info(
        f"    Train-train: {len(train_train_k10)}  "
        f"Train-val: {len(train_val_k10)}"
    )

    # --- Baselines -----------------------------------------------------
    logger.info("\n  Computing K=10 baselines ...")
    oracle_r_k10, oracle_c_k10 = oracle_route(
        holdout_data_k10, K10_MODELS, costs_k10,
    )
    logger.info(f"    Oracle: R={oracle_r_k10:.4f}  C=${oracle_c_k10:.6f}")

    static_k10: Dict[str, Dict] = {}
    for m in K10_MODELS:
        sr, sc = static_route(holdout_data_k10, m, costs_k10)
        static_k10[m] = {"reward": sr, "cost": sc}
        logger.info(
            f"    Static {K10_CATALOG[m]['display']:<22}: R={sr:.4f}  C=${sc:.6f}"
        )

    random_k10 = random_route(
        holdout_data_k10, K10_MODELS, costs_k10, N_SEEDS * 4,
    )
    logger.info(f"    Random: R={random_k10['reward']:.4f}")

    eg_k10 = best_static_noisy_route(
        train_train_k10, holdout_data_k10, K10_MODELS, costs_k10,
        n_trials=N_SEEDS * 4,
    )
    logger.info(f"    Best-static+noise: R={eg_k10['reward']:.4f}")

    ucb1_k10 = ucb1_online_route(
        train_train_k10, holdout_data_k10, K10_MODELS, costs_k10,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(
        f"    UCB1 (non-contextual): R={ucb1_k10['reward']:.4f} "
        f"+/-{ucb1_k10['std_reward']:.4f}"
    )

    # --- BanditGPT Pareto sweep ----------------------------------------
    logger.info(
        f"\n  BanditGPT K=10 Pareto sweep "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    bandit_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_data_k10, train_train_emb_k10, holdout_emb_k10,
        str(MULTIMODEL_WARMUP_PRIORS_PATH), costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=True, label="banditGPT",
        dev_val_data=train_val_k10, dev_val_emb=train_val_emb_k10,
    )

    # Tabula rasa ablation (no priors, no Corralling)
    logger.info(
        f"\n  Tabula rasa K=10 ablation "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_data_k10, train_train_emb_k10, holdout_emb_k10,
        None, costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
        dev_val_data=train_val_k10, dev_val_emb=train_val_emb_k10,
    )

    # --- Dev-selected Pareto AUC ----------------------------------------
    best_static_m = max(static_k10, key=lambda m: static_k10[m]["reward"])

    bg_dev_costs_k10 = [p["dev_mean_cost"] for p in bandit_pareto_k10]
    tr_dev_costs_k10 = [p["dev_mean_cost"] for p in tabula_pareto_k10]
    cost_lo_k10 = max(min(bg_dev_costs_k10), min(tr_dev_costs_k10))
    cost_hi_k10 = min(max(bg_dev_costs_k10), max(tr_dev_costs_k10))

    bg_ds_auc_k10, _, _, bg_dev_idx_k10 = dev_selected_pareto_auc(
        bandit_pareto_k10, cost_lo_k10, cost_hi_k10,
    )
    tr_ds_auc_k10, _, _, tr_dev_idx_k10 = dev_selected_pareto_auc(
        tabula_pareto_k10, cost_lo_k10, cost_hi_k10,
    )

    # Oracle envelope AUC (reference)
    bg_costs_k10 = [p["mean_cost"] for p in bandit_pareto_k10]
    tr_costs_k10 = [p["mean_cost"] for p in tabula_pareto_k10]
    oracle_cost_lo_k10 = max(min(bg_costs_k10), min(tr_costs_k10))
    oracle_cost_hi_k10 = min(max(bg_costs_k10), max(tr_costs_k10))
    bg_oracle_auc_k10 = pareto_auc(
        bg_costs_k10,
        [p["mean_reward"] for p in bandit_pareto_k10],
        oracle_cost_lo_k10, oracle_cost_hi_k10,
    )
    tr_oracle_auc_k10 = pareto_auc(
        tr_costs_k10,
        [p["mean_reward"] for p in tabula_pareto_k10],
        oracle_cost_lo_k10, oracle_cost_hi_k10,
    )

    # Paired bootstrap CI for K=10 AUC difference
    logger.info("  Computing K=10 bootstrap CI ...")
    bg_pp_r_k10: Dict[float, np.ndarray] = {}
    bg_pp_c_k10: Dict[float, np.ndarray] = {}
    for p in bandit_pareto_k10:
        if p.get("per_seed_per_prompt_rewards") is not None:
            bg_pp_r_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            bg_pp_c_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )
    tr_pp_r_k10: Dict[float, np.ndarray] = {}
    tr_pp_c_k10: Dict[float, np.ndarray] = {}
    for p in tabula_pareto_k10:
        if p.get("per_seed_per_prompt_rewards") is not None:
            tr_pp_r_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            tr_pp_c_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )

    bg_boot_pp_r_k10, bg_boot_pp_c_k10 = _extract_dev_optimal_per_prompt(
        bandit_pareto_k10, bg_dev_idx_k10,
        bg_pp_r_k10, bg_pp_c_k10, "lambda",
    )
    tr_boot_pp_r_k10, tr_boot_pp_c_k10 = _extract_dev_optimal_per_prompt(
        tabula_pareto_k10, tr_dev_idx_k10,
        tr_pp_r_k10, tr_pp_c_k10, "lambda",
    )
    bootstrap_k10 = bootstrap_pareto_auc_difference(
        bg_boot_pp_r_k10, bg_boot_pp_c_k10,
        tr_boot_pp_r_k10, tr_boot_pp_c_k10,
        cost_lo=cost_lo_k10, cost_hi=cost_hi_k10,
        n_holdout=len(holdout_data_k10), n_bootstrap=1_000,
    )

    logger.info(f"\n  K=10 SUMMARY (dev-selected Pareto AUC primary):")
    logger.info(f"    Oracle:       {oracle_r_k10:.4f}")
    logger.info(
        f"    Dev-selected AUC: BanditGPT={bg_ds_auc_k10:.4f} vs "
        f"Tabula rasa={tr_ds_auc_k10:.4f} "
        f"(adv: {bg_ds_auc_k10 - tr_ds_auc_k10:+.4f})"
    )
    logger.info(
        f"    Bootstrap 95% CI: [{bootstrap_k10['ci_95_lower']:+.4f}, "
        f"{bootstrap_k10['ci_95_upper']:+.4f}] "
        f"p={bootstrap_k10['p_value']:.4g}"
    )
    logger.info(
        f"    Oracle envelope (ref): BanditGPT={bg_oracle_auc_k10:.4f} vs "
        f"Tabula rasa={tr_oracle_auc_k10:.4f}"
    )
    logger.info(
        f"    Best static:  {static_k10[best_static_m]['reward']:.4f} "
        f"({K10_CATALOG[best_static_m]['display']})"
    )
    logger.info(f"    Best-static+noise: {eg_k10['reward']:.4f}")
    logger.info(f"    UCB1 (non-ctx):    {ucb1_k10['reward']:.4f}")
    logger.info(f"    Random:            {random_k10['reward']:.4f}")

    results_all["K10"] = {
        "models": [{"id": m, **K10_CATALOG[m]} for m in K10_MODELS],
        "n_train": len(train_data_k10),
        "n_holdout": len(holdout_data_k10),
        "oracle": {"reward": oracle_r_k10, "cost": oracle_c_k10},
        "static": {m: static_k10[m] for m in K10_MODELS},
        "best_static": {
            "model": best_static_m,
            "reward": static_k10[best_static_m]["reward"],
            "cost": static_k10[best_static_m]["cost"],
        },
        "random": random_k10,
        "best_static_noisy": eg_k10,
        "ucb1": ucb1_k10,
        "banditgpt_pareto": bandit_pareto_k10,
        "tabula_rasa_pareto": tabula_pareto_k10,
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k10, cost_hi_k10],
            "banditgpt": bg_ds_auc_k10,
            "tabula_rasa": tr_ds_auc_k10,
            "advantage": bg_ds_auc_k10 - tr_ds_auc_k10,
            "bootstrap_ci": bootstrap_k10,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward).  Deployed = holdout performance of dev-optimal "
                "hyperparameters.  Bootstrap CI with fixed dev indices."
            ),
        },
        "pareto_auc_oracle_envelope": {
            "cost_range": [oracle_cost_lo_k10, oracle_cost_hi_k10],
            "banditgpt": bg_oracle_auc_k10,
            "tabula_rasa": tr_oracle_auc_k10,
            "advantage": bg_oracle_auc_k10 - tr_oracle_auc_k10,
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
    run_k10_experiment()
