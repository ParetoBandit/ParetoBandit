#!/usr/bin/env python3
"""
Real-Data Leave-One-Out Transfer: Hybrid vs Disjoint LinUCB
============================================================

Directly addresses the "Semantic Transfer: A Null Result" finding from
Section 5.2.2 of the paper, which reported that one-shot semantic transfer
(admix_theta_from_neighbors) provided **no statistically significant
advantage** over tabula rasa (p > 0.07).

This experiment tests whether **continuous family-shared parameter learning**
via Hybrid LinUCB resolves the cold-start transfer problem.

Protocol (leave-one-out with real data):
  1. Load all-models dev/holdout data (43 models, ~1,871 prompts each).
  2. Select 14 models spanning 5 multi-member families + 2 singletons.
  3. Embed prompts with SentenceTransformer + PCA (dim=33).
  4. For each newcomer (model with at least one same-family sibling):
     a. Pre-train Hybrid and Disjoint policies on K-1 models (dev data).
     b. Add the newcomer.
     c. Evaluate on holdout data — measure:
        - Prediction error (MAE) for the newcomer at early horizons
        - Overall routing reward (UCB arm selection)
  5. Aggregate across newcomers and seeds, report paired t-tests.

Expected result: Hybrid LinUCB shows **statistically significant improvement**
for same-family newcomers due to continuous parameter sharing, while
different-family newcomers show no difference (control).

Usage
-----
    python experiments/05_hybrid_ablation/run_real_data_transfer_ablation.py
"""

import sys
import json
import gzip
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.router import (
    DisjointLinUCBPolicy,
    HybridLinUCBPolicy,
    infer_model_family,
)
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
ALPHA = 0.5
INIT_LAMBDA = 1.0
N_SEEDS = 10
SEED_OFFSET = 42
EARLY_HORIZON = 20

SELECTED_MODELS = [
    # OpenAI GPT-5 family (3 members)
    "openai/gpt-5", "openai/gpt-5-chat", "openai/gpt-5.1",
    # Meta LLaMA-3 family (3 members: 405b, 70b, 8b)
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    # OpenAI GPT-4 family (2 members)
    "openai/gpt-4-turbo", "openai/gpt-4.1",
    # Anthropic Claude-sonnet-4 family (2 members)
    "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.5",
    # X.AI Grok-3 family (2 members)
    "x-ai/grok-3", "x-ai/grok-3-mini",
    # Singletons (cross-family control)
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-4o",
]

NEWCOMERS = [
    # Same-family newcomers (expect hybrid benefit)
    ("openai/gpt-5.1",                     "same-family"),
    ("meta-llama/llama-3.1-405b-instruct",  "same-family"),
    ("openai/gpt-4.1",                      "same-family"),
    ("anthropic/claude-sonnet-4.5",          "same-family"),
    ("x-ai/grok-3-mini",                    "same-family"),
    # Cross-family controls (expect no benefit)
    ("deepseek/deepseek-chat-v3-0324",       "singleton"),
    ("openai/gpt-4o",                        "singleton"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pivot_data(filepath: Path, selected: List[str]) -> List[Dict]:
    """Load JSONL data and pivot to {prompt: str, rewards: {model: score}}."""
    prompt_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(filepath, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok") and entry["model_id"] in selected:
                prompt_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]
    data = []
    for prompt, rewards in prompt_rewards.items():
        if len(rewards) == len(selected):
            data.append({"prompt": prompt, "rewards": rewards})
    return data


def precompute_embeddings(data: List[Dict], encoder, pca) -> List[np.ndarray]:
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    newcomer_id: str,
    established: List[str],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    seed: int,
    dim: int,
) -> Dict:
    """Run one leave-one-out trial for a single newcomer."""
    rng = np.random.default_rng(seed)

    family_map = {m: infer_model_family(m) for m in established}
    disjoint = DisjointLinUCBPolicy(
        established, dim=dim, alpha=ALPHA, init_lambda=INIT_LAMBDA,
    )
    hybrid = HybridLinUCBPolicy(
        established, dim=dim, alpha=ALPHA, init_lambda=INIT_LAMBDA,
        family_map=family_map,
    )

    # Shuffle training data for this seed
    idx = rng.permutation(len(train_data))

    # Phase 1: Pre-train on established models (update all arms per prompt)
    for i in idx:
        x = train_emb[i]
        rewards = train_data[i]["rewards"]
        for model in established:
            if model in rewards:
                disjoint.update(model, x, rewards[model])
                hybrid.update(model, x, rewards[model])

    # Phase 2: Add newcomer
    disjoint.add_arm(newcomer_id)
    hybrid.add_arm(newcomer_id, family=infer_model_family(newcomer_id))

    # Phase 3: Evaluate on holdout — measure newcomer prediction quality
    eval_idx = rng.permutation(len(eval_data))
    mae_disjoint = []
    mae_hybrid = []
    selected_disjoint = []
    selected_hybrid = []

    for step, i in enumerate(eval_idx):
        x = eval_emb[i]
        actual_reward = eval_data[i]["rewards"].get(newcomer_id)
        if actual_reward is None:
            continue

        # Disjoint prediction for newcomer
        theta_d = disjoint.A_inv[newcomer_id] @ disjoint.b[newcomer_id]
        pred_d = float(x @ theta_d)

        # Hybrid prediction for newcomer
        fam = hybrid.family_map.get(newcomer_id, newcomer_id)
        beta_h = (
            hybrid.A0_inv[fam] @ hybrid.b0[fam]
            if fam in hybrid.A0_inv
            else np.zeros(dim)
        )
        theta_h = hybrid.A_inv[newcomer_id] @ hybrid.b[newcomer_id]
        pred_h = float(x @ (beta_h + theta_h))

        mae_disjoint.append(abs(actual_reward - pred_d))
        mae_hybrid.append(abs(actual_reward - pred_h))

        # UCB arm selection (does the newcomer get selected?)
        all_models = established + [newcomer_id]
        d_arm, _ = disjoint.select_arm(x, all_models)
        h_arm, _ = hybrid.select_arm(x, all_models)
        selected_disjoint.append(d_arm == newcomer_id)
        selected_hybrid.append(h_arm == newcomer_id)

        # Update both policies with newcomer's reward
        disjoint.update(newcomer_id, x, actual_reward)
        hybrid.update(newcomer_id, x, actual_reward)

        # Also update established models (they're still being used)
        for model in established:
            r = eval_data[i]["rewards"].get(model)
            if r is not None:
                sel_d, _ = disjoint.select_arm(x, [model])
                disjoint.update(model, x, r)
                hybrid.update(model, x, r)

    return {
        "mae_disjoint": mae_disjoint,
        "mae_hybrid": mae_hybrid,
        "newcomer_selected_disjoint": selected_disjoint,
        "newcomer_selected_hybrid": selected_hybrid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 72)
    logger.info("REAL-DATA LEAVE-ONE-OUT: Hybrid vs Disjoint LinUCB")
    logger.info("=" * 72)
    logger.info(
        "\nPaper reference: Section 5.2.2 'Semantic Transfer: A Null Result'"
        "\n  Previous finding: p > 0.07, no significant transfer benefit"
        "\n  This experiment: Continuous family sharing via Hybrid LinUCB"
    )

    # Load data
    logger.info("\n--- Loading data ---")
    selected_set = set(SELECTED_MODELS)
    train_data = load_pivot_data(DEV_DATA_PATH_ALL_MODELS, selected_set)
    eval_data = load_pivot_data(HOLDOUT_DATA_PATH_ALL_MODELS, selected_set)
    logger.info(f"  Dev prompts (all {len(SELECTED_MODELS)} models): {len(train_data)}")
    logger.info(f"  Holdout prompts: {len(eval_data)}")

    if len(train_data) < 50 or len(eval_data) < 50:
        logger.error("Insufficient data — check model coverage in dataset")
        return

    # Embed
    logger.info("\n--- Loading encoder + PCA ---")
    t0 = time.time()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    dim = pca.n_components_ + 1  # PCA dims + bias

    logger.info(f"  Context dimension: {dim}")
    logger.info("  Embedding prompts...")
    train_emb = precompute_embeddings(train_data, encoder, pca)
    eval_emb = precompute_embeddings(eval_data, encoder, pca)
    logger.info(f"  Done in {time.time() - t0:.1f}s")

    # Family info
    logger.info("\n--- Family assignments ---")
    family_map = {m: infer_model_family(m) for m in SELECTED_MODELS}
    fam_groups = defaultdict(list)
    for m, f in family_map.items():
        fam_groups[f].append(m)
    for fam, members in sorted(fam_groups.items()):
        logger.info(f"  {fam}: {members}")

    # Run leave-one-out trials
    logger.info(f"\n--- Running {len(NEWCOMERS)} newcomers x {N_SEEDS} seeds ---")
    t_start = time.time()

    all_results = {}
    for newcomer_id, newcomer_type in NEWCOMERS:
        established = [m for m in SELECTED_MODELS if m != newcomer_id]
        logger.info(f"\n  Newcomer: {newcomer_id} ({newcomer_type})")
        logger.info(f"    Family: {infer_model_family(newcomer_id)}")
        logger.info(f"    Established: {len(established)} models")

        seed_results = []
        for trial in range(N_SEEDS):
            seed = SEED_OFFSET + trial
            r = run_trial(
                newcomer_id, established,
                train_data, eval_data, train_emb, eval_emb,
                seed, dim,
            )
            seed_results.append(r)

        # Aggregate
        all_mae_d = [np.mean(r["mae_disjoint"]) for r in seed_results]
        all_mae_h = [np.mean(r["mae_hybrid"]) for r in seed_results]
        early_mae_d = [
            np.mean(r["mae_disjoint"][:EARLY_HORIZON]) for r in seed_results
        ]
        early_mae_h = [
            np.mean(r["mae_hybrid"][:EARLY_HORIZON]) for r in seed_results
        ]
        sel_d = [np.mean(r["newcomer_selected_disjoint"]) for r in seed_results]
        sel_h = [np.mean(r["newcomer_selected_hybrid"]) for r in seed_results]

        # Paired t-test (hybrid improvement over disjoint)
        t_full, p_full = sp_stats.ttest_rel(all_mae_d, all_mae_h)
        t_early, p_early = sp_stats.ttest_rel(early_mae_d, early_mae_h)

        improvement_full = (
            100 * (np.mean(all_mae_d) - np.mean(all_mae_h)) / np.mean(all_mae_d)
            if np.mean(all_mae_d) > 1e-8 else 0.0
        )
        improvement_early = (
            100 * (np.mean(early_mae_d) - np.mean(early_mae_h)) / np.mean(early_mae_d)
            if np.mean(early_mae_d) > 1e-8 else 0.0
        )

        result = {
            "newcomer_type": newcomer_type,
            "family": infer_model_family(newcomer_id),
            "full_mae_disjoint": round(float(np.mean(all_mae_d)), 4),
            "full_mae_hybrid": round(float(np.mean(all_mae_h)), 4),
            "full_mae_ci95_disjoint": round(
                1.96 * float(np.std(all_mae_d, ddof=1)) / np.sqrt(N_SEEDS), 4
            ),
            "full_mae_ci95_hybrid": round(
                1.96 * float(np.std(all_mae_h, ddof=1)) / np.sqrt(N_SEEDS), 4
            ),
            "full_improvement_pct": round(improvement_full, 1),
            "full_p_value": round(float(p_full), 4),
            "early_mae_disjoint": round(float(np.mean(early_mae_d)), 4),
            "early_mae_hybrid": round(float(np.mean(early_mae_h)), 4),
            "early_mae_ci95_disjoint": round(
                1.96 * float(np.std(early_mae_d, ddof=1)) / np.sqrt(N_SEEDS), 4
            ),
            "early_mae_ci95_hybrid": round(
                1.96 * float(np.std(early_mae_h, ddof=1)) / np.sqrt(N_SEEDS), 4
            ),
            "early_improvement_pct": round(improvement_early, 1),
            "early_p_value": round(float(p_early), 4),
            "selection_rate_disjoint": round(float(np.mean(sel_d)), 4),
            "selection_rate_hybrid": round(float(np.mean(sel_h)), 4),
        }
        all_results[newcomer_id] = result

        sig = "***" if p_early < 0.001 else ("**" if p_early < 0.01 else ("*" if p_early < 0.05 else "n.s."))
        logger.info(
            f"    Full MAE:   D={np.mean(all_mae_d):.4f}  H={np.mean(all_mae_h):.4f}  "
            f"Δ={improvement_full:+.1f}%  p={p_full:.4f}"
        )
        logger.info(
            f"    Early-{EARLY_HORIZON} MAE: D={np.mean(early_mae_d):.4f}  H={np.mean(early_mae_h):.4f}  "
            f"Δ={improvement_early:+.1f}%  p={p_early:.4f} {sig}"
        )
        logger.info(
            f"    Selection:  D={np.mean(sel_d):.3f}  H={np.mean(sel_h):.3f}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\n--- Complete in {elapsed:.0f}s ---")

    # Summary table
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY: Early Prediction Error (first {} prompts)".format(EARLY_HORIZON))
    logger.info("=" * 90)
    logger.info(
        f"{'Newcomer':<42} {'Type':<12} {'Disjoint':<12} {'Hybrid':<12} "
        f"{'Δ%':<8} {'p-value':<8} {'Sig'}"
    )
    logger.info("-" * 90)

    same_fam_improvements = []
    singleton_improvements = []

    for newcomer_id, newcomer_type in NEWCOMERS:
        r = all_results[newcomer_id]
        p = r["early_p_value"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        logger.info(
            f"  {newcomer_id:<40} {newcomer_type:<12} "
            f"{r['early_mae_disjoint']:.4f}       {r['early_mae_hybrid']:.4f}       "
            f"{r['early_improvement_pct']:+.1f}%   {p:.4f}   {sig}"
        )
        if newcomer_type == "same-family":
            same_fam_improvements.append(r["early_improvement_pct"])
        else:
            singleton_improvements.append(r["early_improvement_pct"])

    logger.info("-" * 90)
    if same_fam_improvements:
        logger.info(
            f"  Same-family average improvement:  {np.mean(same_fam_improvements):+.1f}%"
        )
    if singleton_improvements:
        logger.info(
            f"  Singleton average improvement:    {np.mean(singleton_improvements):+.1f}%"
        )

    logger.info(
        "\nPaper's previous finding: p > 0.07, no significant transfer (Section 5.2.2)"
    )

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_data_transfer_ablation.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "description": (
                        "Real-data leave-one-out: Hybrid vs Disjoint LinUCB. "
                        "Addresses Section 5.2.2 null result."
                    ),
                    "n_seeds": N_SEEDS,
                    "alpha": ALPHA,
                    "early_horizon": EARLY_HORIZON,
                    "n_train_prompts": len(train_data),
                    "n_eval_prompts": len(eval_data),
                    "n_selected_models": len(SELECTED_MODELS),
                    "n_newcomers": len(NEWCOMERS),
                },
                "results": all_results,
                "aggregate": {
                    "same_family_mean_improvement_pct": (
                        round(float(np.mean(same_fam_improvements)), 1)
                        if same_fam_improvements else None
                    ),
                    "singleton_mean_improvement_pct": (
                        round(float(np.mean(singleton_improvements)), 1)
                        if singleton_improvements else None
                    ),
                },
            },
            f,
            indent=2,
        )
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
