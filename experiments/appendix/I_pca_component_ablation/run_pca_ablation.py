#!/usr/bin/env python3
"""
PCA Component-Count Ablation for K=3

Sweeps the number of PCA components used for contextual features in the K=3
multi-model routing setting.  For each component count, we:

  1. Truncate the production 32-component PCA to the target dimensionality.
  2. Re-embed the K=3 online-learn and holdout data through the truncated PCA.
  3. Generate warmup priors at that dimensionality (from cached 43-model priors
     or by re-accumulating sufficient statistics from the prior-train pool).
  4. Run the BanditGPT Pareto sweep (lambda x seeds) with the reduced features.
  5. Compute the UCB1 (non-contextual) baseline for reference.
  6. Report the dev-holdout gap and UCB1 comparison for each component count.

The primary diagnostic is the dev-holdout reward gap as a function of the
samples-per-feature ratio.  A healthy contextual bandit should show a gap
near zero; a large positive gap indicates overfitting.

Usage:
    cd experiments/appendix/I_pca_component_ablation
    python run_pca_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEV_DATA_PATH_ALL_MODELS,
    FULL_PCA_PATH,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    K3_MODELS_PATH,
    MULTIMODEL_WARMUP_PRIORS_PATH_32,
    THREE_WAY_SPLITS_PATH,
    ARTIFACTS_DIR,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

COMPONENT_COUNTS: List[int] = [4, 6, 8, 12, 16, 24, 32]

N_SEEDS: int = 20
SEED_OFFSET: int = 42
DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

LAMBDA_VALUES: List[float] = [
    0.0, 0.03, 0.07, 0.1, 0.13, 0.16, 0.19, 0.2, 0.22, 0.25, 0.3, 0.5, 1.0,
]


def _req_cost(inp: float, out: float) -> float:
    return (100 * inp + 400 * out) / 1_000_000


def _load_k3_portfolio() -> Tuple[List[str], Dict[str, Dict]]:
    """Load K=3 model list and catalog from ``models_k3.json``."""
    with open(K3_MODELS_PATH) as f:
        k3_cfg = json.load(f)
    models = [m["model_id"] for m in k3_cfg["models"]]
    prices = get_prices_for_models(models)
    catalog: Dict[str, Dict] = {}
    for m_entry in k3_cfg["models"]:
        mid = m_entry["model_id"]
        catalog[mid] = {
            "display": m_entry.get("display", mid.split("/")[-1]),
            **prices[mid],
            "cost": _req_cost(prices[mid]["input_cost_per_m"],
                              prices[mid]["output_cost_per_m"]),
        }
    return models, catalog

K3_MODELS, K3_CATALOG = _load_k3_portfolio()


# ============================================================================
# Data loading (reused from 03_figure)
# ============================================================================

def load_rewards_from_file(
    data_path: Path,
    models: List[str],
    prompt_filter: Optional[set] = None,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL."""
    import gzip
    from collections import defaultdict

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

    return [
        {"prompt": p, "rewards": rmap}
        for p, rmap in rewards.items()
        if len(rmap) == len(models)
    ]


def build_model_registry(
    models: List[str], catalog: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    return {
        m: {
            "input_cost_per_m": catalog[m]["input_cost_per_m"],
            "output_cost_per_m": catalog[m]["output_cost_per_m"],
        }
        for m in models
    }


# ============================================================================
# PCA truncation
# ============================================================================

def truncate_pca(pca_full: PCA, n_components: int) -> PCA:
    """Create a reduced-dimension PCA by taking the first *n_components*."""
    if n_components >= pca_full.n_components_:
        return pca_full
    pca = PCA(n_components=n_components, whiten=pca_full.whiten)
    pca.components_ = pca_full.components_[:n_components]
    pca.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca_full.singular_values_[:n_components]
    pca.mean_ = pca_full.mean_
    pca.n_components_ = n_components
    pca.n_features_in_ = pca_full.n_features_in_
    pca.n_samples_ = pca_full.n_samples_
    pca.noise_variance_ = pca_full.noise_variance_
    return pca


def truncate_warmup_priors(
    priors_path: Path, n_components: int,
) -> Dict[str, Any]:
    """Truncate the sufficient statistics in a warmup priors file.

    The priors contain A (d x d) and b (d,) per model where d = pca_components + 1.
    Truncating to fewer PCA components means slicing out the first n rows/cols of A
    and first n entries of b, plus the bias (last) row/col.
    """
    priors = joblib.load(priors_path)
    old_dim = priors["context_dim"]
    new_dim = n_components + 1

    if new_dim >= old_dim:
        return priors

    keep_idx = list(range(n_components)) + [old_dim - 1]  # first n PCA + bias

    new_A = {}
    new_b = {}
    for m in priors["models"]:
        A_full = priors["A"][m]
        b_full = priors["b"][m]
        new_A[m] = A_full[np.ix_(keep_idx, keep_idx)]
        new_b[m] = b_full[keep_idx]

    truncated = dict(priors)
    truncated["A"] = new_A
    truncated["b"] = new_b
    truncated["context_dim"] = new_dim
    truncated["pca_components"] = n_components
    return truncated


# ============================================================================
# Embedding
# ============================================================================

def embed_dataset(
    data: List[Dict], encoder: SentenceTransformer, pca: PCA,
) -> List[np.ndarray]:
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


# ============================================================================
# Training and evaluation (streamlined from 03_figure)
# ============================================================================

def _compute_reward_normalization(
    data: List[Dict], models: List[str],
) -> Tuple[float, float]:
    all_r = [d["rewards"][m] for d in data for m in models if m in d["rewards"]]
    return float(np.min(all_r)), float(np.max(all_r))


def train_bandit(
    router, data: List[Dict], emb: List[np.ndarray],
    models: List[str], r_min: float, r_range: float,
    *, rng: np.random.Generator,
) -> None:
    order = rng.permutation(len(data))
    for idx in order:
        prompt_data = data[idx]
        features = emb[idx]
        model_id, route_log = router.route(
            features, total_steps=len(data),
        )
        reward = prompt_data["rewards"][model_id]
        normalized = (reward - r_min) / max(r_range, 1e-12)
        router.process_feedback(route_log.request_id, normalized)


def _set_exploit_mode(router) -> Dict[Any, Any]:
    saved: Dict[Any, Any] = {}
    # Standard BanditRouter: corralling_router at top level
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "exploit_mode"):
        saved["_corralling_exploit"] = cr.exploit_mode
        cr.exploit_mode = True
    for exp in getattr(router, "_experts", [router]):
        saved[id(exp)] = getattr(exp, "_alpha", None)
        exp._alpha = 0.0
        if hasattr(exp, "_corralling_router") and exp._corralling_router is not None:
            cr = exp._corralling_router
            saved[id(cr)] = getattr(cr, "_alpha", None)
            cr._alpha = 0.0
            if hasattr(cr, "exploit_mode"):
                saved["_cr_exploit_" + str(id(cr))] = cr.exploit_mode
                cr.exploit_mode = True
    return saved


def _restore_exploit_mode(router, saved: Dict) -> None:
    if not saved:
        return
    cr = getattr(router, "corralling_router", None)
    if cr is not None and "_corralling_exploit" in saved:
        cr.exploit_mode = saved["_corralling_exploit"]
    for exp in getattr(router, "_experts", [router]):
        if id(exp) in saved and saved[id(exp)] is not None:
            exp._alpha = saved[id(exp)]
        if hasattr(exp, "_corralling_router") and exp._corralling_router is not None:
            cr = exp._corralling_router
            if id(cr) in saved and saved[id(cr)] is not None:
                cr._alpha = saved[id(cr)]
            if hasattr(cr, "exploit_mode") and ("_cr_exploit_" + str(id(cr))) in saved:
                cr.exploit_mode = saved["_cr_exploit_" + str(id(cr))]


def evaluate_frozen(
    router, data: List[Dict], emb: List[np.ndarray],
    costs: Dict[str, float], burn_in: int,
) -> Tuple[float, float]:
    saved = _set_exploit_mode(router)
    rng_state = np.random.get_state()
    np.random.seed(0)

    r_total, c_total = 0.0, 0.0
    for i, prompt_data in enumerate(data):
        features = emb[i]
        model_id, _ = router.route(features, total_steps=burn_in + len(data))
        r_total += prompt_data["rewards"][model_id]
        c_total += costs.get(model_id, 0.0)

    np.random.set_state(rng_state)
    _restore_exploit_mode(router, saved)
    n = len(data)
    return r_total / n, c_total / n


def ucb1_online_route(
    train_data: List[Dict], holdout_data: List[Dict],
    models: List[str], costs: Dict[str, float],
    *, n_trials: int,
) -> Dict[str, Any]:
    """Non-contextual UCB1 baseline."""
    all_rewards = []
    all_costs = []
    greedy_arms = []

    for trial in range(n_trials):
        rng = np.random.default_rng(SEED_OFFSET + trial)
        counts = {m: 0 for m in models}
        sums = {m: 0.0 for m in models}
        order = rng.permutation(len(train_data))

        for idx in order:
            total = sum(counts.values())
            if total < len(models):
                arm = models[total]
            else:
                ucb = {
                    m: (sums[m] / counts[m])
                    + np.sqrt(2 * np.log(total) / counts[m])
                    for m in models
                }
                arm = max(ucb, key=lambda m: ucb[m])  # type: ignore[arg-type]
            r = train_data[idx]["rewards"][arm]
            counts[arm] += 1
            sums[arm] += r

        greedy = max(models, key=lambda m: sums[m] / max(counts[m], 1))
        greedy_arms.append(greedy)

        trial_r = np.mean([d["rewards"][greedy] for d in holdout_data])
        trial_c = costs.get(greedy, 0.0)
        all_rewards.append(trial_r)
        all_costs.append(trial_c)

    from collections import Counter
    return {
        "reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards, ddof=1)),
        "cost": float(np.mean(all_costs)),
        "std_cost": float(np.std(all_costs, ddof=1)),
        "n_trials": n_trials,
        "greedy_arm": Counter(greedy_arms).most_common(1)[0][0],
    }


def _split_dev_train_val(
    data: List[Dict], emb: List[np.ndarray],
) -> Tuple[List[Dict], List[np.ndarray], List[Dict], List[np.ndarray]]:
    n = len(data)
    rng = np.random.default_rng(DEV_VAL_SEED)
    indices = rng.permutation(n)
    n_val = max(1, int(n * DEV_VAL_FRACTION))
    val_idx = set(indices[:n_val].tolist())
    train_d = [data[i] for i in range(n) if i not in val_idx]
    train_e = [emb[i] for i in range(n) if i not in val_idx]
    val_d = [data[i] for i in range(n) if i in val_idx]
    val_e = [emb[i] for i in range(n) if i in val_idx]
    return train_d, train_e, val_d, val_e


# ============================================================================
# Main ablation loop
# ============================================================================

def run_ablation_for_n_components(
    n_comp: int,
    *,
    pca_truncated: PCA,
    warmup_priors: Dict[str, Any],
    encoder: SentenceTransformer,
    train_data: List[Dict],
    holdout_data: List[Dict],
    costs: Dict[str, float],
    alpha: float,
    n_eff: float,
    forgetting_factor: float,
) -> Dict[str, Any]:
    """Run the full BanditGPT sweep for one component count."""
    feat_dim = n_comp + 1
    logger.info(f"\n  Embedding with {n_comp}-component PCA ...")
    train_emb = embed_dataset(train_data, encoder, pca_truncated)
    holdout_emb = embed_dataset(holdout_data, encoder, pca_truncated)

    train_train, train_train_emb, train_val, train_val_emb = (
        _split_dev_train_val(train_data, train_emb)
    )
    logger.info(f"    Train-train: {len(train_train)}, Train-val: {len(train_val)}")

    r_min, r_max = _compute_reward_normalization(train_data, K3_MODELS)
    r_range = r_max - r_min

    # Save truncated priors to a temp file for create_experiment_router
    tmp_priors = Path(__file__).parent / "results" / f"_tmp_priors_{n_comp}comp.joblib"
    joblib.dump(warmup_priors, tmp_priors)

    results = []
    for lam in LAMBDA_VALUES:
        trial_r, trial_c = [], []
        trial_dev_r, trial_dev_c = [], []
        for trial in range(N_SEEDS):
            seed = SEED_OFFSET + trial
            np.random.seed(seed)
            trial_rng = np.random.default_rng(seed)
            router = create_experiment_router(
                model_registry=build_model_registry(K3_MODELS, K3_CATALOG),
                feature_dim=feat_dim,
                prior_n_effective=n_eff,
                alpha=alpha,
                warmup_path=str(tmp_priors),
                use_corralling=True,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
                forgetting_factor=forgetting_factor,
            )
            train_bandit(
                router, train_train, train_train_emb, K3_MODELS,
                r_min, r_range, rng=trial_rng,
            )
            r, c = evaluate_frozen(
                router, holdout_data, holdout_emb, costs, len(train_train),
            )
            trial_r.append(r)
            trial_c.append(c)
            dev_r, dev_c = evaluate_frozen(
                router, train_val, train_val_emb, costs, len(train_train),
            )
            trial_dev_r.append(dev_r)
            trial_dev_c.append(dev_c)

        results.append({
            "lambda": lam,
            "holdout_reward_mean": float(np.mean(trial_r)),
            "holdout_reward_std": float(np.std(trial_r, ddof=1)) if N_SEEDS > 1 else 0.0,
            "holdout_cost_mean": float(np.mean(trial_c)),
            "dev_reward_mean": float(np.mean(trial_dev_r)),
            "dev_cost_mean": float(np.mean(trial_dev_c)),
        })
        logger.info(
            f"    lambda={lam:<6.3f}  hld_R={np.mean(trial_r):.4f}  "
            f"dev_R={np.mean(trial_dev_r):.4f}  "
            f"gap={np.mean(trial_dev_r) - np.mean(trial_r):+.4f}"
        )

    tmp_priors.unlink(missing_ok=True)

    best = max(results, key=lambda x: x["holdout_reward_mean"])
    avg_gap = np.mean([
        r["dev_reward_mean"] - r["holdout_reward_mean"] for r in results
    ])

    return {
        "n_components": n_comp,
        "feature_dim": feat_dim,
        "variance_explained": float(np.sum(pca_truncated.explained_variance_ratio_)),
        "samples_per_arm": len(train_train) / len(K3_MODELS),
        "samples_per_feature_ratio": len(train_train) / (len(K3_MODELS) * feat_dim),
        "best_holdout_reward": best["holdout_reward_mean"],
        "best_holdout_reward_std": best["holdout_reward_std"],
        "best_holdout_cost": best["holdout_cost_mean"],
        "best_lambda": best["lambda"],
        "avg_dev_holdout_gap": float(avg_gap),
        "sweep": results,
    }


def main() -> None:
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load shared resources
    logger.info("Loading encoder and full 32-component PCA ...")
    pca32 = joblib.load(FULL_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca32.n_components_} components (whiten={pca32.whiten})")

    # Load tuned hyperparameters
    hparams_path = (
        PROJECT_ROOT / "experiments" / "appendix"
        / "H_alpha_neff_ablation" / "results" / "best_hparams_k3.json"
    )
    alpha = 0.1
    n_eff = 5000.0
    forgetting_factor = 1.0
    if hparams_path.exists():
        hp = json.loads(hparams_path.read_text())
        cfg = hp.get("K3", {})
        alpha = float(cfg.get("alpha", alpha))
        n_eff = float(cfg.get("n_eff", n_eff))
        forgetting_factor = float(cfg.get("gamma", forgetting_factor))
        logger.info(f"  Loaded hparams: alpha={alpha}, n_eff={n_eff}, gamma={forgetting_factor}")

    # Load K=3 data
    logger.info("\nLoading K=3 data ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits_3way = json.load(f)
    online_prompts = set(splits_3way["online_learn_pool"])

    train_data = load_rewards_from_file(
        DEV_DATA_PATH_ALL_MODELS, K3_MODELS, prompt_filter=online_prompts,
    )
    holdout_data = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"  Train (online-learn): {len(train_data)} prompts")
    logger.info(f"  Holdout: {len(holdout_data)} prompts")

    costs = {m: K3_CATALOG[m]["cost"] for m in K3_MODELS}

    # UCB1 baseline (computed once — independent of feature dim)
    logger.info("\nComputing UCB1 baseline ...")
    train_train_data, _, _, _ = _split_dev_train_val(
        train_data, [np.zeros(1)] * len(train_data),
    )
    ucb1 = ucb1_online_route(
        train_train_data, holdout_data, K3_MODELS, costs, n_trials=N_SEEDS,
    )
    logger.info(
        f"  UCB1: reward={ucb1['reward']:.4f} +/-{ucb1['std_reward']:.4f} "
        f"(greedy: {ucb1['greedy_arm']})"
    )

    # Main ablation loop
    logger.info(f"\n{'='*70}")
    logger.info(f"PCA Component-Count Ablation (K=3)")
    logger.info(f"  Components: {COMPONENT_COUNTS}")
    logger.info(f"  Lambda values: {len(LAMBDA_VALUES)}")
    logger.info(f"  Seeds: {N_SEEDS}")
    logger.info(f"{'='*70}")

    all_results = []
    for n_comp in COMPONENT_COUNTS:
        logger.info(f"\n{'='*50}")
        logger.info(f"  n_components = {n_comp} "
                     f"(feat_dim={n_comp+1}, "
                     f"var={np.sum(pca32.explained_variance_ratio_[:n_comp]):.1%})")
        logger.info(f"{'='*50}")

        pca_trunc = truncate_pca(pca32, n_comp)
        priors_trunc = truncate_warmup_priors(MULTIMODEL_WARMUP_PRIORS_PATH_32, n_comp)

        result = run_ablation_for_n_components(
            n_comp,
            pca_truncated=pca_trunc,
            warmup_priors=priors_trunc,
            encoder=encoder,
            train_data=train_data,
            holdout_data=holdout_data,
            costs=costs,
            alpha=alpha,
            n_eff=n_eff,
            forgetting_factor=forgetting_factor,
        )
        all_results.append(result)

        beats_ucb1 = result["best_holdout_reward"] > ucb1["reward"]
        logger.info(
            f"\n  Summary: best_R={result['best_holdout_reward']:.4f}, "
            f"gap={result['avg_dev_holdout_gap']:+.4f}, "
            f"beats_UCB1={'YES' if beats_ucb1 else 'NO'} "
            f"({result['best_holdout_reward']:.4f} vs {ucb1['reward']:.4f})"
        )

    # Save results
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "component_counts": COMPONENT_COUNTS,
            "n_seeds": N_SEEDS,
            "n_lambda": len(LAMBDA_VALUES),
            "n_train": len(train_data),
            "n_holdout": len(holdout_data),
            "models": K3_MODELS,
            "alpha": alpha,
            "n_eff": n_eff,
            "forgetting_factor": forgetting_factor,
            "elapsed_seconds": time.time() - t0,
        },
        "ucb1_baseline": ucb1,
        "ablation_results": all_results,
    }

    out_path = output_dir / "pca_component_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # Final summary table
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(
        f"{'Comp':>5s}  {'Dim':>4s}  {'Var%':>6s}  "
        f"{'s/f ratio':>9s}  {'Best R':>7s}  {'Gap':>7s}  "
        f"{'vs UCB1':>8s}"
    )
    logger.info("-" * 60)
    for r in all_results:
        delta = r["best_holdout_reward"] - ucb1["reward"]
        logger.info(
            f"{r['n_components']:5d}  {r['feature_dim']:4d}  "
            f"{r['variance_explained']:5.1%}  "
            f"{r['samples_per_feature_ratio']:9.1f}  "
            f"{r['best_holdout_reward']:7.4f}  "
            f"{r['avg_dev_holdout_gap']:+7.4f}  "
            f"{delta:+8.4f}"
        )
    logger.info(f"\nUCB1 baseline: {ucb1['reward']:.4f}")
    logger.info(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
