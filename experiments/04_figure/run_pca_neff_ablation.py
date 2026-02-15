#!/usr/bin/env python3
"""
PCA Dimensionality and neff (Prior Strength) Ablation
======================================================

Two separate one-at-a-time sweeps at λ=0:
  1. PCA dimensions: {4, 8, 16, 32, 64, 128} (default=32)
  2. neff (prior strength): {1, 3, 5, 10, 20, 50, 100} (default=10)

Protocol: 20 seeds per configuration, same hyperparameters as Figure 4.
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split,
    normalize_prior_strength,
)
from bandit_gpt.router import (
    CorrallingRouter,
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
)
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
)
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
ALPHA_END = 0.1
LAMBDA = 0.0

PCA_DIMS = [4, 8, 16, 32, 64, 128]
NEFF_VALUES = [1, 3, 5, 10, 20, 50, 100]


def embed_with_pca(prompt_text, encoder, pca_model):
    """Encode prompt and apply PCA + bias."""
    emb = encoder.encode(prompt_text, normalize_embeddings=True)
    reduced = pca_model.transform(emb.reshape(1, -1))[0]
    return np.append(reduced, 1.0)  # bias term


def run_banditgpt(
    train_data, eval_data, train_emb, eval_emb,
    warmup_priors, model_costs, neff=10.0,
):
    """Run banditGPT-Hybrid with given priors and pre-computed embeddings."""
    scaled_priors = normalize_prior_strength(warmup_priors, neff)
    models = list(train_data[0]["rewards"].keys())
    dim = scaled_priors["context_dim"]

    warmup_expert = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    tabula_rasa = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=ALPHA_START, alpha_end=ALPHA_END, cost_penalty=LAMBDA,
    )
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa],
        models=models, learning_rate=1.0,
    )

    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    for i, p in enumerate(train_data):
        x = train_emb[i]
        sel, token = router.select_model(x, total_steps=len(train_data))
        norm_r = (p["rewards"][sel] - r_min) / r_range
        router.update(x, sel, norm_r, selection_token=token)

    total_reward = 0.0
    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        sel, _ = router.select_model(x, total_steps=len(train_data))
        total_reward += p["rewards"][sel]

    return total_reward / len(eval_data)


def retrain_pca(encoder, train_data, n_components):
    """Retrain PCA with a different number of components."""
    raw_embeddings = encoder.encode(
        [p["prompt"] for p in train_data], normalize_embeddings=True, show_progress_bar=False
    )
    pca_model = PCA(n_components=n_components)
    pca_model.fit(raw_embeddings)
    return pca_model


def precompute_embeddings(data, encoder, pca_model):
    """Compute embeddings with given PCA model."""
    results = []
    for p in data:
        emb = encoder.encode(p["prompt"], normalize_embeddings=True)
        reduced = pca_model.transform(emb.reshape(1, -1))[0]
        results.append(np.append(reduced, 1.0))
    return results


def retrain_priors_for_dim(original_priors, n_components):
    """Adapt priors to new PCA dimensionality.
    
    Since we can't re-derive the priors from scratch (that would require the
    original 80K dataset), we truncate/pad the existing A and b matrices.
    For dimensions smaller than 32, we truncate.
    For dimensions larger than 32, we pad with identity/zeros.
    """
    import copy
    new_priors = copy.deepcopy(original_priors)
    orig_dim = original_priors["context_dim"]  # 33 (32 PCA + bias)
    new_dim = n_components + 1  # +1 for bias

    new_priors["context_dim"] = new_dim

    for model_id in new_priors["models"]:
        old_A = np.array(new_priors["A"][model_id])
        old_b = np.array(new_priors["b"][model_id])

        if new_dim <= orig_dim:
            new_priors["A"][model_id] = old_A[:new_dim, :new_dim]
            new_priors["b"][model_id] = old_b[:new_dim]
        else:
            ridge = old_A[0, 0] if old_A.shape[0] > 0 else 1.0
            new_A = np.eye(new_dim) * ridge
            new_A[:orig_dim, :orig_dim] = old_A
            new_priors["A"][model_id] = new_A

            new_b = np.zeros(new_dim)
            new_b[:orig_dim] = old_b
            new_priors["b"][model_id] = new_b

    return new_priors


def sweep(name, values, run_fn):
    results = {}
    for val in values:
        rewards = []
        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            r = run_fn(val)
            rewards.append(r)

        avg = np.mean(rewards)
        std = np.std(rewards, ddof=1) if N_TRIALS > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(N_TRIALS)
        results[val] = {"mean": avg, "std": std, "ci95": ci95}
        marker = " <-- default" if (name == "PCA_dim" and val == 32) or (name == "neff" and val == 10) else ""
        logger.info(f"  {name}={val:<5}  Reward={avg:.4f} ± {ci95:.4f}{marker}")
    return results


def main():
    logger.info("=" * 70)
    logger.info("PCA DIMENSIONALITY & NEFF ABLATION")
    logger.info("=" * 70)

    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    default_pca = joblib.load(DEFAULT_PCA_PATH)

    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_priors = joblib.load(
        sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH
    )

    models = list(eval_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    normalized_costs = {
        m: {
            "cost": model_costs[m]["cost"],
            "normalized_cost": (model_costs[m]["cost"] - min_cost) / cost_range
            if cost_range > 0 else 0.0,
        }
        for m in models
    }

    # Pre-encode raw embeddings (we'll apply PCA with different dims)
    logger.info("\n--- Encoding raw embeddings ---")
    t0 = time.time()
    raw_train = encoder.encode(
        [p["prompt"] for p in train_data], normalize_embeddings=True, show_progress_bar=False
    )
    raw_eval = encoder.encode(
        [p["prompt"] for p in eval_data], normalize_embeddings=True, show_progress_bar=False
    )
    logger.info(f"  {len(raw_train)}+{len(raw_eval)} in {time.time()-t0:.1f}s")

    t_start = time.time()

    # ====== SWEEP 1: PCA dimensionality ======
    logger.info(f"\n[1/2] PCA dimensionality sweep (neff=10 fixed)")

    pca_results = {}
    for n_comp in PCA_DIMS:
        logger.info(f"\n  Retraining PCA with {n_comp} components...")
        pca_model = PCA(n_components=n_comp)
        pca_model.fit(raw_train)

        # Compute embeddings with this PCA
        train_emb = [np.append(pca_model.transform(raw_train[i:i+1])[0], 1.0) for i in range(len(raw_train))]
        eval_emb = [np.append(pca_model.transform(raw_eval[i:i+1])[0], 1.0) for i in range(len(raw_eval))]

        # Adapt priors to new dim
        adapted_priors = retrain_priors_for_dim(warmup_priors, n_comp)

        rewards = []
        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            r = run_banditgpt(
                train_data, eval_data, train_emb, eval_emb,
                adapted_priors, normalized_costs, neff=10.0,
            )
            rewards.append(r)

        avg = np.mean(rewards)
        std = np.std(rewards, ddof=1)
        ci95 = 1.96 * std / np.sqrt(N_TRIALS)
        pca_results[n_comp] = {"mean": avg, "std": std, "ci95": ci95}
        marker = " <-- default" if n_comp == 32 else ""
        logger.info(f"  PCA_dim={n_comp:<5}  Reward={avg:.4f} ± {ci95:.4f}{marker}")

    # ====== SWEEP 2: neff (prior strength) ======
    logger.info(f"\n[2/2] neff sweep (PCA_dim=32 fixed)")

    # Use default PCA embeddings
    train_emb_default = [np.append(default_pca.transform(raw_train[i:i+1])[0], 1.0)
                         for i in range(len(raw_train))]
    eval_emb_default = [np.append(default_pca.transform(raw_eval[i:i+1])[0], 1.0)
                        for i in range(len(raw_eval))]

    neff_results = sweep(
        "neff",
        NEFF_VALUES,
        lambda neff: run_banditgpt(
            train_data, eval_data, train_emb_default, eval_emb_default,
            warmup_priors, normalized_costs, neff=neff,
        ),
    )

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pca_neff_ablation.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "n_trials": N_TRIALS,
                    "lambda": LAMBDA,
                    "n_eval": len(eval_data),
                    "n_train": len(train_data),
                },
                "pca_sweep": {str(k): v for k, v in pca_results.items()},
                "neff_sweep": {str(k): v for k, v in neff_results.items()},
            },
            f, indent=2,
        )

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)

    logger.info("\nPCA dimensionality (neff=10):")
    for d in PCA_DIMS:
        r = pca_results[d]
        marker = " <-- default" if d == 32 else ""
        logger.info(f"  d={d:<5}  {r['mean']:.4f} ± {r['ci95']:.4f}{marker}")

    logger.info("\nneff (PCA_dim=32):")
    for n in NEFF_VALUES:
        r = neff_results[n]
        marker = " <-- default" if n == 10 else ""
        logger.info(f"  neff={n:<5}  {r['mean']:.4f} ± {r['ci95']:.4f}{marker}")

    logger.info(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
