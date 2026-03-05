#!/usr/bin/env python3
"""
Figure 4: The Value of Warmup Priors.

Demonstrates that warmup priors are a key architectural ingredient of
BanditGPT by comparing:

    - **BanditGPT** (Corralling + warmup priors): the full system.
    - **Tabula Rasa** (single LinUCB, no priors, no Corralling): ablation.

Both variants are evaluated on the K=3 portfolio under identical
conditions: same data, same dev/holdout splits, same seeds.  Supervised
baselines (KNN, SVM, MLP) from the LLMRouter literature provide
reference anchors.

Outputs
-------
(a) Pareto frontier comparison — how warmup priors shift the entire
    cost--quality frontier upward.
(b) Learning curve — holdout reward vs online training steps,
    demonstrating that warmup priors deliver supervised-baseline-quality
    routing from step 0 while tabula rasa needs many interactions.

Protocol
--------
Same train-then-freeze protocol as Figure 3.  Hyperparameters loaded
from Appendix H (per-expert tuning).  Portfolio-specific warmup priors.

Outputs (``results/``)
    warmup_ablation_results.json
"""

import json
import logging
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
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "03_figure"))

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K3_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
    K3_MODELS_PATH,
)

from utils.pareto import (
    dev_selected_pareto_auc,
    bootstrap_pareto_auc_difference,
    extract_dev_optimal_per_prompt,
)
from utils.embeddings import load_embedding_cache

from run_prequential import (
    load_rewards_from_file,
    embed_dataset,
    oracle_route,
    static_route,
    random_route,
    ucb1_online_route,
    run_pareto_sweep,
    run_learning_curve,
    _split_dev_train_val,
    _make_learning_curve_checkpoints,
    K3_MODELS,
    K3_CATALOG,
    LAMBDA_VALUES_K3,
    N_SEEDS,
    SEED_OFFSET,
    DEV_VAL_FRACTION,
)
from utils.supervised_baselines import (
    run_supervised_baseline,
    run_supervised_learning_curve,
    tune_supervised_hparams,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameter loading (Appendix H only)
# ============================================================================


def _load_hparams(path: Path, key: str) -> Dict[str, float]:
    """Load dev-val-selected hyperparameters from Appendix H.

    Raises FileNotFoundError if the file is missing (no fallback defaults).
    """
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
# Sample efficiency metrics
# ============================================================================


def compute_sample_efficiency(
    warmup_curve: List[Dict],
    tr_curve: List[Dict],
    reference_rewards: Dict[str, float],
) -> Dict[str, Any]:
    """Compute sample efficiency metrics against supervised baselines.

    For each reference baseline, reports the number of online steps each
    variant needs to reach that baseline's quality.

    Args:
        warmup_curve: BanditGPT learning curve (checkpoint dicts).
        tr_curve: Tabula rasa learning curve (checkpoint dicts).
        reference_rewards: {name: holdout_reward} for each supervised
            baseline to use as a threshold.

    Returns:
        Dict with per-baseline steps-to-match and overall AUC.
    """

    def _steps_to_threshold(
        curve: List[Dict], thresh: float,
    ) -> Optional[int]:
        for d in curve:
            if d["mean_reward"] >= thresh:
                return d["step"]
        return None

    def _auc(curve: List[Dict]) -> float:
        if len(curve) < 2:
            return 0.0
        steps = np.array([d["step"] for d in curve], dtype=float)
        rewards = np.array([d["mean_reward"] for d in curve])
        return float(np.trapz(rewards, steps))

    results: Dict[str, Any] = {
        "warmup_auc": _auc(warmup_curve),
        "tabula_rasa_auc": _auc(tr_curve),
        "auc_advantage": _auc(warmup_curve) - _auc(tr_curve),
        "warmup_step0": warmup_curve[0]["mean_reward"] if warmup_curve else None,
        "tabula_rasa_step0": tr_curve[0]["mean_reward"] if tr_curve else None,
        "warmup_final": warmup_curve[-1]["mean_reward"] if warmup_curve else None,
        "tabula_rasa_final": tr_curve[-1]["mean_reward"] if tr_curve else None,
        "baselines": {},
    }

    for name, thresh in reference_rewards.items():
        w_steps = _steps_to_threshold(warmup_curve, thresh)
        t_steps = _steps_to_threshold(tr_curve, thresh)
        results["baselines"][name] = {
            "threshold": thresh,
            "warmup_steps": w_steps,
            "tabula_rasa_steps": t_steps,
            "speedup": (
                f"{t_steps / w_steps:.1f}x"
                if w_steps is not None and t_steps is not None and w_steps > 0
                else ("instant" if w_steps == 0 else "N/A")
            ),
        }

    return results


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:
    """Run the warmup prior ablation for K=3."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    import run_prequential as _rp
    _rp._EMBEDDING_CACHE = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    # ------------------------------------------------------------------
    # Hyperparameters from Appendix H
    # ------------------------------------------------------------------
    hparams_dir = (
        PROJECT_ROOT / "experiments" / "appendix"
        / "H_alpha_neff_ablation" / "results"
    )
    tuned_warmup = _load_hparams(hparams_dir / "best_hparams_k3.json", "K3")
    tuned_tr = _load_hparams(
        hparams_dir / "best_hparams_k3_tabula_rasa.json", "K3",
    )
    logger.info(
        f"  Warmup hparams: alpha={tuned_warmup['alpha']} "
        f"n_eff={tuned_warmup['prior_n_effective']} "
        f"gamma={tuned_warmup['forgetting_factor']}"
    )
    logger.info(
        f"  Tabula rasa hparams: alpha={tuned_tr['alpha']} "
        f"n_eff={tuned_tr['prior_n_effective']} "
        f"gamma={tuned_tr['forgetting_factor']}"
    )

    # ------------------------------------------------------------------
    # Validate warmup priors exist
    # ------------------------------------------------------------------
    if not K3_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=3 warmup priors not found: {K3_WARMUP_PRIORS_PATH}\n"
            "Generate with: python scripts/extract_warmup_from_multimodel.py"
        )
    k3_warmup_path = str(K3_WARMUP_PRIORS_PATH)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("\n  Loading K=3 data ...")
    costs_k3 = {m: K3_CATALOG[m]["cost"] for m in K3_MODELS}

    prior_train_prompts: set = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(
            f"    Excluding {len(prior_train_prompts)} prior-train prompts"
        )

    all_dev_k3 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K3_MODELS)
    train_data = [
        d for d in all_dev_k3 if d["prompt"] not in prior_train_prompts
    ]
    holdout_data = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"    Train (excl. prior-train): {len(train_data)} prompts")
    logger.info(f"    Holdout: {len(holdout_data)} prompts")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    logger.info("  Embedding prompts ...")
    train_emb = embed_dataset(train_data, encoder, pca)
    holdout_emb = embed_dataset(holdout_data, encoder, pca)

    # ------------------------------------------------------------------
    # Dev train/val split
    # ------------------------------------------------------------------
    logger.info(
        f"  Splitting dev into train/val "
        f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ..."
    )
    train_train, train_train_emb, train_val, train_val_emb = (
        _split_dev_train_val(train_data, train_emb)
    )
    logger.info(
        f"    Dev-train: {len(train_train)}  Dev-val: {len(train_val)}"
    )

    # ------------------------------------------------------------------
    # Oracle and baselines
    # ------------------------------------------------------------------
    oracle_r, oracle_c = oracle_route(holdout_data, K3_MODELS, costs_k3)
    logger.info(f"    Oracle: R={oracle_r:.4f}  C=${oracle_c:.6f}")

    static_results: Dict[str, Dict] = {}
    for m in K3_MODELS:
        sr, sc = static_route(holdout_data, m, costs_k3)
        static_results[m] = {"reward": sr, "cost": sc}
    best_static_m = max(
        static_results, key=lambda m: static_results[m]["reward"],
    )
    logger.info(
        f"    Best static: {K3_CATALOG[best_static_m]['display']} "
        f"R={static_results[best_static_m]['reward']:.4f}"
    )

    random_res = random_route(holdout_data, K3_MODELS, costs_k3, N_SEEDS * 4)
    logger.info(f"    Random: R={random_res['reward']:.4f}")

    ucb1_res = ucb1_online_route(
        train_train, holdout_data, K3_MODELS, costs_k3,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(f"    UCB1: R={ucb1_res['reward']:.4f}")

    # ------------------------------------------------------------------
    # Supervised baselines (reference anchors)
    # ------------------------------------------------------------------
    logger.info("\n  Training supervised baselines ...")
    supervised: Dict[str, Dict] = {}
    supervised_tuning: Dict[str, Dict] = {}

    for kind in ("knn", "svm", "mlp"):
        tuning = tune_supervised_hparams(
            kind, train_train, train_train_emb,
            train_val, train_val_emb,
            K3_MODELS, costs_k3,
        )
        supervised_tuning[kind] = tuning
        res = run_supervised_baseline(
            kind, K3_MODELS, costs_k3,
            train_train, train_train_emb,
            holdout_data, holdout_emb,
            n_trials=N_SEEDS,
            hparams=tuning["best_hparams"],
        )
        supervised[kind] = res
        logger.info(
            f"    {kind.upper()}: R={res['reward']:.4f} +/-{res['std_reward']:.4f}"
        )

    # ------------------------------------------------------------------
    # BanditGPT Pareto sweep (with warmup priors)
    # ------------------------------------------------------------------
    logger.info(
        f"\n  BanditGPT Pareto sweep "
        f"({len(LAMBDA_VALUES_K3)} lambda x {N_SEEDS} seeds) ...",
    )
    pareto_warmup = run_pareto_sweep(
        K3_MODELS, K3_CATALOG,
        train_train, holdout_data, train_train_emb, holdout_emb,
        k3_warmup_path, costs_k3, LAMBDA_VALUES_K3,
        N_SEEDS, use_corralling=True, label="banditGPT_warmup",
        dev_val_data=train_val, dev_val_emb=train_val_emb,
        alpha=tuned_warmup["alpha"],
        prior_n_effective=tuned_warmup["prior_n_effective"],
        forgetting_factor=tuned_warmup["forgetting_factor"],
        tabula_rasa_alpha=tuned_tr["alpha"],
        tabula_rasa_forgetting_factor=tuned_tr["forgetting_factor"],
    )

    # ------------------------------------------------------------------
    # Tabula rasa Pareto sweep (no priors, no Corralling)
    # ------------------------------------------------------------------
    logger.info(
        f"\n  Tabula rasa Pareto sweep "
        f"({len(LAMBDA_VALUES_K3)} lambda x {N_SEEDS} seeds) ...",
    )
    pareto_tr = run_pareto_sweep(
        K3_MODELS, K3_CATALOG,
        train_train, holdout_data, train_train_emb, holdout_emb,
        None, costs_k3, LAMBDA_VALUES_K3,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
        dev_val_data=train_val, dev_val_emb=train_val_emb,
        alpha=tuned_tr["alpha"],
        prior_n_effective=tuned_tr["prior_n_effective"],
        forgetting_factor=tuned_tr["forgetting_factor"],
    )

    # ------------------------------------------------------------------
    # Learning curves
    # ------------------------------------------------------------------
    checkpoints = _make_learning_curve_checkpoints(len(train_train))
    logger.info(f"\n  Learning curve checkpoints: {checkpoints}")

    logger.info(
        f"  BanditGPT learning curve ({N_SEEDS} seeds) ...",
    )
    lc_warmup = run_learning_curve(
        K3_MODELS, K3_CATALOG,
        train_train, holdout_data, train_train_emb, holdout_emb,
        k3_warmup_path, costs_k3, N_SEEDS, checkpoints,
        use_corralling=True, cost_penalty=0.0,
        alpha=tuned_warmup["alpha"],
        prior_n_effective=tuned_warmup["prior_n_effective"],
        forgetting_factor=tuned_warmup["forgetting_factor"],
        tabula_rasa_alpha=tuned_tr["alpha"],
        tabula_rasa_forgetting_factor=tuned_tr["forgetting_factor"],
        label="BanditGPT",
    )

    logger.info(
        f"  Tabula rasa learning curve ({N_SEEDS} seeds) ...",
    )
    lc_tr = run_learning_curve(
        K3_MODELS, K3_CATALOG,
        train_train, holdout_data, train_train_emb, holdout_emb,
        None, costs_k3, N_SEEDS, checkpoints,
        use_corralling=False, cost_penalty=0.0,
        alpha=tuned_tr["alpha"],
        prior_n_effective=tuned_tr["prior_n_effective"],
        forgetting_factor=tuned_tr["forgetting_factor"],
        label="Tabula Rasa",
    )

    # ------------------------------------------------------------------
    # Supervised learning curves (for reference)
    # ------------------------------------------------------------------
    best_sv_kind = max(supervised, key=lambda k: supervised[k]["reward"])
    logger.info(
        f"  Supervised learning curve ({best_sv_kind.upper()}, "
        f"{N_SEEDS} seeds) ...",
    )
    sv_lc = run_supervised_learning_curve(
        best_sv_kind, K3_MODELS, costs_k3,
        train_train, train_train_emb,
        holdout_data, holdout_emb,
        checkpoints,
        n_trials=N_SEEDS,
        hparams=supervised_tuning[best_sv_kind]["best_hparams"],
    )

    # ------------------------------------------------------------------
    # Sample efficiency metrics
    # ------------------------------------------------------------------
    reference_rewards = {k: v["reward"] for k, v in supervised.items()}
    reference_rewards["best_static"] = static_results[best_static_m]["reward"]
    sample_eff = compute_sample_efficiency(
        lc_warmup, lc_tr, reference_rewards,
    )

    # ------------------------------------------------------------------
    # Pareto AUC comparison
    # ------------------------------------------------------------------
    bg_dev_costs = [p["dev_mean_cost"] for p in pareto_warmup]
    tr_dev_costs = [p["dev_mean_cost"] for p in pareto_tr]
    cost_lo = max(min(bg_dev_costs), min(tr_dev_costs))
    cost_hi = min(max(bg_dev_costs), max(tr_dev_costs))

    bg_auc, _, _, bg_dev_idx = dev_selected_pareto_auc(
        pareto_warmup, cost_lo, cost_hi,
    )
    tr_auc, _, _, tr_dev_idx = dev_selected_pareto_auc(
        pareto_tr, cost_lo, cost_hi,
    )

    # Bootstrap CI
    logger.info("  Computing bootstrap CI for Pareto AUC difference ...")
    bg_pp_r: Dict[float, np.ndarray] = {}
    bg_pp_c: Dict[float, np.ndarray] = {}
    for p in pareto_warmup:
        if p.get("per_seed_per_prompt_rewards") is not None:
            bg_pp_r[p["lambda"]] = np.array(p["per_seed_per_prompt_rewards"])
            bg_pp_c[p["lambda"]] = np.array(p["per_seed_per_prompt_costs"])
    tr_pp_r: Dict[float, np.ndarray] = {}
    tr_pp_c: Dict[float, np.ndarray] = {}
    for p in pareto_tr:
        if p.get("per_seed_per_prompt_rewards") is not None:
            tr_pp_r[p["lambda"]] = np.array(p["per_seed_per_prompt_rewards"])
            tr_pp_c[p["lambda"]] = np.array(p["per_seed_per_prompt_costs"])

    bg_boot_r, bg_boot_c = extract_dev_optimal_per_prompt(
        pareto_warmup, bg_dev_idx, bg_pp_r, bg_pp_c, "lambda",
    )
    tr_boot_r, tr_boot_c = extract_dev_optimal_per_prompt(
        pareto_tr, tr_dev_idx, tr_pp_r, tr_pp_c, "lambda",
    )
    bootstrap = bootstrap_pareto_auc_difference(
        bg_boot_r, bg_boot_c,
        tr_boot_r, tr_boot_c,
        cost_lo=cost_lo, cost_hi=cost_hi,
        n_holdout=len(holdout_data), n_bootstrap=1_000,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    warmup_lam0 = next(
        p for p in pareto_warmup if p["lambda"] == 0.0
    )
    tr_lam0 = next(p for p in pareto_tr if p["lambda"] == 0.0)

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Oracle:                {oracle_r:.4f}")
    logger.info(
        f"  BanditGPT (lam=0):     {warmup_lam0['mean_reward']:.4f} "
        f"+/-{warmup_lam0['std_reward']:.4f}"
    )
    logger.info(
        f"  Tabula rasa (lam=0):   {tr_lam0['mean_reward']:.4f} "
        f"+/-{tr_lam0['std_reward']:.4f}"
    )
    logger.info(
        f"  Best supervised (SVM): {supervised['svm']['reward']:.4f}"
    )
    logger.info(
        f"  Pareto AUC: BanditGPT={bg_auc:.4f} vs "
        f"TR={tr_auc:.4f} (adv={bg_auc - tr_auc:+.4f})"
    )
    logger.info(
        f"  Bootstrap 95% CI: [{bootstrap['ci_95_lower']:+.4f}, "
        f"{bootstrap['ci_95_upper']:+.4f}] p={bootstrap['p_value']:.4g}"
    )
    logger.info(f"\n  Sample efficiency:")
    for name, info in sample_eff["baselines"].items():
        logger.info(
            f"    To match {name} (R={info['threshold']:.4f}): "
            f"warmup={info['warmup_steps']}, "
            f"TR={info['tabula_rasa_steps']}, "
            f"speedup={info['speedup']}"
        )
    logger.info(
        f"  Step-0: warmup={sample_eff['warmup_step0']:.4f}, "
        f"TR={sample_eff['tabula_rasa_step0']:.4f} "
        f"(advantage={sample_eff['warmup_step0'] - sample_eff['tabula_rasa_step0']:+.4f})"
    )

    # ------------------------------------------------------------------
    # Serialise
    # ------------------------------------------------------------------
    def _strip_per_prompt(obj: Any) -> Any:
        """Recursively drop per_prompt / per_seed_per_prompt arrays."""
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

    results_all = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": "Figure 4: Value of Warmup Priors (K=3)",
            "n_seeds": N_SEEDS,
            "protocol": "train_then_freeze",
            "dev_val_fraction": DEV_VAL_FRACTION,
            "warmup_hparams": tuned_warmup,
            "tabula_rasa_hparams": tuned_tr,
            "warmup_priors_path": str(K3_WARMUP_PRIORS_PATH),
        },
        "K3": {
            "models": K3_MODELS,
            "n_train": len(train_train),
            "n_holdout": len(holdout_data),
            "oracle": {"reward": oracle_r, "cost": oracle_c},
            "static": {m: static_results[m] for m in K3_MODELS},
            "best_static": {
                "model": best_static_m,
                "reward": static_results[best_static_m]["reward"],
                "cost": static_results[best_static_m]["cost"],
            },
            "random": random_res,
            "ucb1": ucb1_res,
            "supervised": supervised,
            "supervised_tuning": supervised_tuning,
            "warmup_pareto": pareto_warmup,
            "tabula_rasa_pareto": pareto_tr,
            "warmup_learning_curve": lc_warmup,
            "tabula_rasa_learning_curve": lc_tr,
            "supervised_learning_curve": {
                "kind": best_sv_kind,
                "curve": sv_lc,
            },
            "pareto_auc": {
                "cost_range": [cost_lo, cost_hi],
                "warmup": bg_auc,
                "tabula_rasa": tr_auc,
                "advantage": bg_auc - tr_auc,
                "bootstrap_ci": bootstrap,
            },
            "sample_efficiency": sample_eff,
        },
    }

    out_path = output_dir / "warmup_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(_strip_per_prompt(results_all), f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    run_experiment()
