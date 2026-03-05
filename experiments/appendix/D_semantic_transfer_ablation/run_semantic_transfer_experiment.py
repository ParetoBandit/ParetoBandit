#!/usr/bin/env python3
"""
Semantic Transfer Ablation: Leave-One-Out Evaluation
====================================================

Quantifies the value of per-model semantic transfer when bootstrapping a
"new" model from its nearest neighbor in the bandit router.

Design:
  - Leave-one-out: For each model in the K=3 portfolio, treat it as the
    target (simulated newcomer). The other K-1 models receive warmup priors
    from the canonical K=3 prior file.
  - Condition A (semantic transfer): Add target via register_model(), with
    neighbor selected by **within-provider tetrachoric correlation** of
    reward vectors (reward-aware matching, same-provider only — matching
    the library's compute_correlation_families design).
  - Condition B (tabula rasa): Add target with identity init (no transfer).
  - Both conditions use the same BanditRouter (Corralling + Hybrid LinUCB),
    same data, same seeds, same shuffled training order.
    The only treatment variable is initialization.

Neighbor selection:
  - Previous version used model DNA embedding cosine similarity, which is a
    name-based proxy.  Cross-provider sim was often < 0.5, causing 0/20
    transfers in small portfolios.
  - This version computes **within-provider tetrachoric correlation** from
    binarized reward vectors (the same metric and scope used for data-driven
    family assignment in compute_correlation_families).  Only same-provider
    models are eligible as neighbors — matching production behaviour where
    family sharing is provider-scoped.  Models that are the sole
    representative of their provider receive no transfer (no same-provider
    peer), testing the realistic cold-start scenario.

Statistical rigor:
  - Training data order is shuffled per seed (np.random.permutation).
  - Paired t-test per target; Holm-Bonferroni correction across targets.
  - Cohen's d effect size reported alongside p-values.
  - Aggregate portfolio-level paired test across all (target × seed) pairs.
  - Encoder loaded once and shared across trials.

Data:
  - Real LMSYS Arena prompts and judge-scored rewards from the 43-model
    evaluation dataset.
  - Warmup priors: K3_WARMUP_PRIORS_PATH (K=3 portfolio).
  - Splits: THREE_WAY_SPLITS_PATH (online-learn pool + holdout).

Output:
  - results/semantic_transfer_ablation.json
  - results/semantic_transfer_ablation_summary.txt
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from scipy import stats as sp_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "experiments")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    K3_MODELS_PATH,
    K3_WARMUP_PRIORS_PATH,
)
from experiments.utils.multimodel import (
    N_TRIALS,
    SEED_OFFSET,
    TARGET_NEFF,
    ALPHA_START,
    CORRALLING_LR,
    CORRALLING_GAMMA,
    load_multimodel_data,
)
from experiments.utils.model_pricing import load_model_catalog
from experiments.utils.router_factory import create_experiment_router
from experiments.utils.transfer import (
    build_reward_vectors,
    find_tetrachoric_neighbor,
    build_filtered_warmup,
)
from experiments.utils.metrics import holm_bonferroni, cohens_d_paired

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

LAMBDA = 0.0  # Quality-focused (no cost penalty)

K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)
PORTFOLIOS = {"K3": K3_MODELS}


def build_model_registry(models: list[str]) -> dict[str, dict]:
    """Build model registry from K3_CATALOG for BanditRouter."""
    return {
        m: {
            "input_cost_per_m": K3_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": K3_CATALOG[m]["output_cost_per_m"],
        }
        for m in models
    }


# Re-export MODEL_CATALOG for display name lookups throughout the script
MODEL_CATALOG = K3_CATALOG


# =============================================================================
# TRIAL RUNNER
# =============================================================================

def run_trial(
    base_models: list[str],
    target_model: str,
    train_data: list,
    eval_data: list,
    train_emb: list,
    eval_emb: list,
    r_min: float,
    r_max: float,
    warmup_path: Path,
    use_transfer: bool,
    seed: int,
    encoder: SentenceTransformer,
    precomputed_neighbor: tuple[str | None, float] | None = None,
) -> dict:
    """Run one trial. Returns dict with holdout reward and transfer metadata."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    dim = len(train_emb[0])
    burn_in = len(train_data)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0

    perm = rng.permutation(burn_in)

    router = create_experiment_router(
        model_registry=build_model_registry(base_models),
        feature_dim=dim,
        prior_n_effective=TARGET_NEFF,
        alpha=ALPHA_START,
        warmup_path=str(warmup_path),
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=LAMBDA,
    )
    router.registry[target_model] = build_model_registry([target_model])[target_model]

    transfer_info = {"actually_transferred": False, "neighbor_used": None, "similarity": None}

    orig_admix = router.admix_theta_from_neighbors
    if not use_transfer:
        def _no_transfer(*args, **kwargs):
            bandit = router.bandit
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64),
            )
        router.admix_theta_from_neighbors = _no_transfer
    else:
        # Wrap admix to inject tetrachoric-based neighbor and capture metadata.
        # Use *args/**kwargs to transparently match register_model()'s call signature.
        tet_nb = precomputed_neighbor  # capture in closure

        def _tetrachoric_admix(*args, **kwargs):
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

    # Train (shuffled order)
    for idx in perm:
        p = train_data[idx]
        x = train_emb[idx]
        model, log = router.route(x, total_steps=burn_in)
        norm_r = (p["rewards"][model] - r_min) / r_range
        router.process_feedback(log.request_id, norm_r)

    # Eval (frozen)
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("SEMANTIC TRANSFER ABLATION (Tetrachoric Neighbors)")
    logger.info("=" * 70)

    if not K3_WARMUP_PRIORS_PATH.exists():
        logger.error(
            f"Warmup priors not found at {K3_WARMUP_PRIORS_PATH}. "
            "Run: python scripts/generate_multimodel_warmup_priors.py"
        )
        return 1

    logger.info("Loading SentenceTransformer (shared across all trials)...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    all_results = {"portfolios": {}, "_meta": {
        "lambda": LAMBDA,
        "n_trials": N_TRIALS,
        "neighbor_selection": "within_provider_tetrachoric_correlation",
    }}

    for portfolio_name, models in PORTFOLIOS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"PORTFOLIO: {portfolio_name} ({len(models)} models)")
        logger.info("=" * 70)

        train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max = load_multimodel_data(
            models
        )
        logger.info(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

        # Compute binarized reward vectors for tetrachoric correlation
        reward_vectors = build_reward_vectors(train_data, models)

        logger.info("  Within-provider tetrachoric neighbors (reward-aware):")
        for m in models:
            nb, sim = find_tetrachoric_neighbor(m, models, reward_vectors, within_provider_only=True)
            disp_m = MODEL_CATALOG.get(m, {}).get("display", m.split("/")[-1])
            disp_nb = MODEL_CATALOG.get(nb, {}).get("display", nb.split("/")[-1]) if nb else "(no same-provider peer)"
            logger.info(f"    {disp_m:<25} -> {disp_nb:<30} (r_tet={sim:.3f})")

        portfolio_results = {}
        for target in models:
            base = [m for m in models if m != target]
            display = MODEL_CATALOG.get(target, {}).get("display", target.split("/")[-1])

            # Tetrachoric neighbor among base models, within provider only
            best_nb, best_sim = find_tetrachoric_neighbor(
                target, base, reward_vectors, within_provider_only=True
            )

            if best_nb:
                disp_nb = MODEL_CATALOG.get(best_nb, {}).get("display", best_nb.split("/")[-1])
            else:
                disp_nb = "(no same-provider peer)"
            logger.info(f"\n  Target: {display}  ->  neighbor: {disp_nb} (r_tet={best_sim:.3f})")

            warmup_filtered = build_filtered_warmup(base, K3_WARMUP_PRIORS_PATH)
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                temp_path = Path(f.name)
            try:
                joblib.dump(warmup_filtered, temp_path)

                transfer_rewards = []
                tabula_rewards = []
                transfer_received = []
                for t in range(N_TRIALS):
                    seed = SEED_OFFSET + t
                    res_t = run_trial(
                        base, target, train_data, eval_data,
                        train_emb, eval_emb, r_min, r_max,
                        temp_path, use_transfer=True, seed=seed,
                        encoder=encoder,
                        precomputed_neighbor=(best_nb, best_sim) if best_nb else None,
                    )
                    res_b = run_trial(
                        base, target, train_data, eval_data,
                        train_emb, eval_emb, r_min, r_max,
                        temp_path, use_transfer=False, seed=seed,
                        encoder=encoder,
                    )
                    transfer_rewards.append(res_t["holdout_reward"])
                    tabula_rewards.append(res_b["holdout_reward"])
                    transfer_received.append(res_t["actually_transferred"])

                mean_transfer = float(np.mean(transfer_rewards))
                mean_tabula = float(np.mean(tabula_rewards))
                std_transfer = float(np.std(transfer_rewards, ddof=1)) if N_TRIALS > 1 else 0.0
                std_tabula = float(np.std(tabula_rewards, ddof=1)) if N_TRIALS > 1 else 0.0
                diffs = np.array(transfer_rewards) - np.array(tabula_rewards)
                if np.std(diffs) < 1e-15:
                    t_stat, p_val = 0.0, 1.0
                else:
                    t_stat, p_val = sp_stats.ttest_rel(transfer_rewards, tabula_rewards)
                t_crit = sp_stats.t.ppf(0.975, N_TRIALS - 1) if N_TRIALS > 1 else 0
                ci_transfer = t_crit * std_transfer / (N_TRIALS ** 0.5) if N_TRIALS > 1 else 0
                ci_tabula = t_crit * std_tabula / (N_TRIALS ** 0.5) if N_TRIALS > 1 else 0
                d = cohens_d_paired(transfer_rewards, tabula_rewards)
                n_transferred = int(sum(transfer_received))

                portfolio_results[target] = {
                    "transfer": {
                        "mean": mean_transfer,
                        "std": std_transfer,
                        "ci95": float(ci_transfer),
                        "per_trial": transfer_rewards,
                    },
                    "tabula_rasa": {
                        "mean": mean_tabula,
                        "std": std_tabula,
                        "ci95": float(ci_tabula),
                        "per_trial": tabula_rewards,
                    },
                    "delta_mean": mean_transfer - mean_tabula,
                    "p_value_uncorrected": float(p_val),
                    "t_statistic": float(t_stat),
                    "cohens_d": d,
                    "n_trials_with_actual_transfer": n_transferred,
                    "n_trials_total": N_TRIALS,
                    "tetrachoric_neighbor": best_nb,
                    "tetrachoric_similarity": best_sim,
                }

                sig = "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                xfer_flag = f"({n_transferred}/{N_TRIALS} xfer)"
                logger.info(
                    f"    Transfer:  {mean_transfer:.4f} ± {ci_transfer:.4f}  |  "
                    f"Tabula: {mean_tabula:.4f} ± {ci_tabula:.4f}  |  "
                    f"Δ={mean_transfer - mean_tabula:+.4f}  p={p_val:.4f}{sig}  d={d:.3f}  "
                    f"{xfer_flag}"
                )
            finally:
                temp_path.unlink(missing_ok=True)

        # Holm-Bonferroni correction across targets within this portfolio
        targets_ordered = list(portfolio_results.keys())
        raw_ps = [portfolio_results[t]["p_value_uncorrected"] for t in targets_ordered]
        adjusted_ps = holm_bonferroni(raw_ps)
        for i, t in enumerate(targets_ordered):
            portfolio_results[t]["p_value_holm"] = adjusted_ps[i]

        # Aggregate portfolio-level test (all target × seed pairs)
        all_transfer = []
        all_tabula = []
        for t in targets_ordered:
            all_transfer.extend(portfolio_results[t]["transfer"]["per_trial"])
            all_tabula.extend(portfolio_results[t]["tabula_rasa"]["per_trial"])
        agg_diffs = np.array(all_transfer) - np.array(all_tabula)
        if np.std(agg_diffs) < 1e-15:
            agg_t, agg_p = 0.0, 1.0
        else:
            agg_t, agg_p = sp_stats.ttest_rel(all_transfer, all_tabula)
        agg_d = cohens_d_paired(all_transfer, all_tabula)

        portfolio_summary = {
            "aggregate_paired_test": {
                "t_statistic": float(agg_t),
                "p_value": float(agg_p),
                "cohens_d": agg_d,
                "n_pairs": len(all_transfer),
                "mean_delta": float(np.mean(agg_diffs)),
            },
        }

        logger.info(f"\n  Portfolio {portfolio_name} aggregate: "
                     f"Δ={portfolio_summary['aggregate_paired_test']['mean_delta']:+.4f}  "
                     f"p={agg_p:.4f}  d={agg_d:.3f}  (N={len(all_transfer)} pairs)")

        all_results["portfolios"][portfolio_name] = {
            "per_target": portfolio_results,
            "portfolio_aggregate": portfolio_summary,
        }

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "semantic_transfer_ablation.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_json}")

    # Summary
    summary_lines = [
        "Semantic Transfer Ablation: Leave-One-Out (Within-Provider Tetrachoric)",
        "=" * 68,
        "Design: Paired, 20 seeds/target, shuffled training order",
        "Neighbor: Within-provider tetrachoric correlation of binarized rewards",
        "Stat: Paired t-test (uncorrected + Holm-Bonferroni), Cohen's d",
        "",
    ]
    for pname, presults in all_results["portfolios"].items():
        summary_lines.append(f"\n{pname}:")
        summary_lines.append(
            f"  {'Target':<25} {'Neighbor':<22} {'r_tet':>5} {'Δ':>8} "
            f"{'p(raw)':>8} {'p(Holm)':>8} {'d':>7} {'Xfer':>5}"
        )
        summary_lines.append("  " + "-" * 90)
        per_target = presults["per_target"]
        for target, r in per_target.items():
            disp = MODEL_CATALOG.get(target, {}).get("display", target.split("/")[-1])
            nb = r.get("tetrachoric_neighbor", "")
            disp_nb = MODEL_CATALOG.get(nb, {}).get("display", nb.split("/")[-1])[:20] if nb else "---"
            r_tet = r.get("tetrachoric_similarity", 0)
            delta = r["delta_mean"]
            p_raw = r["p_value_uncorrected"]
            p_holm = r["p_value_holm"]
            cd = r["cohens_d"]
            xf = f"{r['n_trials_with_actual_transfer']}/{r['n_trials_total']}"
            sig = "**" if p_holm < 0.01 else "*" if p_holm < 0.05 else ""
            summary_lines.append(
                f"  {disp:<25} {disp_nb:<22} {r_tet:.3f} {delta:+.4f}  "
                f"{p_raw:.4f}   {p_holm:.4f}{sig:>2}  {cd:+.3f}  {xf:>5}"
            )
        agg = presults["portfolio_aggregate"]["aggregate_paired_test"]
        summary_lines.append(
            f"  {'AGGREGATE':<25} {'':22} {'':>5} {agg['mean_delta']:+.4f}  "
            f"{agg['p_value']:.4f}   {'---':>6}   {agg['cohens_d']:+.3f}  "
            f"{agg['n_pairs']:>4}p"
        )

    summary_text = "\n".join(summary_lines)
    with open(out_dir / "semantic_transfer_ablation_summary.txt", "w") as f:
        f.write(summary_text)
    logger.info(f"\n{summary_text}")
    logger.info(f"\nSummary: {out_dir / 'semantic_transfer_ablation_summary.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
