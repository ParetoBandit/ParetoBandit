#!/usr/bin/env python3
"""
K-Scaling Experiment: Hybrid vs. Disjoint LinUCB Sample Efficiency
===================================================================

Tests whether family parameter sharing in Hybrid LinUCB converges faster
than Disjoint LinUCB, and whether the advantage grows with portfolio size K.

Hypothesis:
  With K models and N training prompts, each model sees ~N/K observations.
  In Hybrid LinUCB, the family-level beta is updated by ALL family members'
  observations, pooling data across similar models.  As K grows, per-model
  observations shrink while family-level observations remain substantial,
  making the shared component increasingly valuable.

Design:
  - Three portfolio sizes: K=5, K=10, K=20
  - Condition A (Hybrid):  Data-driven family assignments via tetrachoric
    correlation on holdout rewards (within-provider, threshold r_tet >= 0.6)
  - Condition B (Disjoint): family_map=None (all independent arms)
  - Both start tabula rasa (no warmup priors) to isolate sharing effect
  - Common dataset across all K (N=888 dev, 750 holdout) for fair comparison
  - Metrics: holdout reward at checkpoints + cumulative online reward
  - 20 seeds per condition, paired t-test

Data:
  Real prompts and judge-scored rewards from the 43-model dataset.
  All K configs use the SAME set of prompts (intersection of K=20 models)
  to ensure dataset size is not a confound in cross-K comparisons.

Output:
  - results/k_scaling_results.json
  - results/k_scaling_figure.png (2-row × 3-col figure)
"""

import sys
import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.router import (
    CostAwareTabulaRasaRouter,
    tetrachoric_corr,
    compute_correlation_families,
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

# ── Experiment parameters ─────────────────────────────────────────────
N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 1.0
ALPHA_END = 0.01
EVAL_INTERVAL = 25
CONTEXT_DIM = 33            # 32 PCA + 1 bias
ROLLING_WINDOW = 50
TETRACHORIC_THRESHOLD = 0.6

# Family assignments are computed data-driven via tetrachoric correlation
# at runtime.  Comments below show the *syntactic* family from
# infer_model_family() for reference only — actual families may differ.
K_CONFIGS = {
    5: [
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "anthropic/claude-3.5-sonnet",
    ],
    10: [
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "openai/gpt-5",
        "openai/gpt-5.1",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "x-ai/grok-3",
        "x-ai/grok-3-mini",
        "anthropic/claude-3.5-sonnet",
    ],
    20: [
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "openai/gpt-5",
        "openai/gpt-5.1",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.2-1b-instruct",
        "x-ai/grok-3",
        "x-ai/grok-3-mini",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "mistralai/ministral-3b",
        "mistralai/ministral-8b",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-pro-preview-06-05",
        "deepseek/deepseek-chat-v3-0324",
        "microsoft/phi-4",
    ],
}


# ── Data loading ──────────────────────────────────────────────────────

def _prompt_key(text: str) -> str:
    """Deterministic, process-stable prompt identifier."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt_data(path, needed_models):
    """Load JSONL.GZ into prompt-centric records with rewards per model.

    Only includes entries with ok=True and prompts where ALL needed models
    have a reward.
    """
    prompt_rewards = defaultdict(lambda: {"prompt": None, "rewards": {}})
    with gzip.open(path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok", True):
                continue
            mid = entry["model_id"]
            if mid not in needed_models:
                continue
            sid = entry.get("sample_id", _prompt_key(entry["prompt"]))
            prompt_rewards[sid]["prompt"] = entry["prompt"]
            prompt_rewards[sid]["rewards"][mid] = entry["raw_score"]

    data = []
    for sid in sorted(prompt_rewards.keys()):
        rec = prompt_rewards[sid]
        if len(rec["rewards"]) == len(needed_models):
            data.append(rec)
    return data


def load_common_dataset():
    """Load one shared dataset using the K=20 superset of models.

    All K configs use the same prompts so that dataset size is controlled
    across the K-scaling comparison.
    """
    all_models = set(K_CONFIGS[max(K_CONFIGS)])
    logger.info(f"Loading common dataset for {len(all_models)} models (K=20 superset)...")
    dev = load_prompt_data(DEV_DATA_PATH_ALL_MODELS, all_models)
    holdout = load_prompt_data(HOLDOUT_DATA_PATH_ALL_MODELS, all_models)
    logger.info(f"  Dev: {len(dev)} prompts, Holdout: {len(holdout)} prompts")
    return dev, holdout


def precompute_embeddings(data, encoder, pca):
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


# ── Within-provider reward correlations (tetrachoric + phi) ───────────

def compute_provider_correlations(data, models):
    """Pairwise tetrachoric and phi correlations within each provider."""
    reward_vecs = {m: np.array([p["rewards"][m] for p in data]) for m in models}

    providers = defaultdict(list)
    for m in sorted(models):
        prov = m.split("/")[0] if "/" in m else "__none__"
        providers[prov].append(m)

    results = {}
    for prov, members in sorted(providers.items()):
        if len(members) < 2:
            continue
        pair_results = []
        for m1, m2 in combinations(members, 2):
            phi, p_val = sp_stats.pearsonr(reward_vecs[m1], reward_vecs[m2])
            r_tet = tetrachoric_corr(reward_vecs[m1], reward_vecs[m2])
            pair_results.append({
                "pair": (m1, m2), "phi": float(phi),
                "r_tet": float(r_tet), "p": float(p_val),
            })
        results[prov] = pair_results
    return results, reward_vecs


# ── Router construction ──────────────────────────────────────────────

def build_family_map_data_driven(reward_vecs, models, threshold=TETRACHORIC_THRESHOLD):
    """Data-driven family map using tetrachoric correlation.

    Only models present in *models* are included; reward_vecs may contain
    a superset (the K=20 holdout vectors).
    """
    subset = {m: reward_vecs[m] for m in models if m in reward_vecs}
    return compute_correlation_families(subset, threshold=threshold)


def make_router(models, family_map=None):
    """Tabula rasa LinUCB: Hybrid if family_map given, Disjoint otherwise."""
    dummy_costs = {m: {"normalized_cost": 0.0} for m in models}
    return CostAwareTabulaRasaRouter(
        models=models,
        context_dim=CONTEXT_DIM,
        model_costs=dummy_costs,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END,
        cost_penalty=0.0,
        ridge_lambda=1.0,
        family_map=family_map,
    )


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_holdout(router, holdout_data, holdout_emb, models):
    """Near-greedy holdout evaluation using the router's own select_model.

    Uses total_steps=0 which yields alpha=alpha_end (0.01), effectively
    greedy while retaining minimal tie-breaking exploration.
    """
    rewards = []
    for i, p in enumerate(holdout_data):
        model = router.select_model(holdout_emb[i], total_steps=0,
                                    candidates=models)
        rewards.append(p["rewards"].get(model, 0.0))
    return float(np.mean(rewards))


def compute_oracle(holdout_data, models):
    return float(np.mean([
        max(p["rewards"][m] for m in models) for p in holdout_data
    ]))


# ── Single trial ──────────────────────────────────────────────────────

def run_trial(models, family_map, dev_data, dev_emb, holdout_data,
              holdout_emb, seed, total_steps):
    """Returns (holdout_curve, online_rewards)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(dev_data))

    router = make_router(models, family_map)
    checkpoints = set(range(0, total_steps + 1, EVAL_INTERVAL))
    checkpoints.add(total_steps)

    holdout_curve = {}
    holdout_curve[0] = evaluate_holdout(router, holdout_data, holdout_emb, models)

    online_rewards = []
    for step_i in range(total_steps):
        i = idx[step_i]
        emb = dev_emb[i]
        model = router.select_model(emb, total_steps=total_steps)
        reward = dev_data[i]["rewards"].get(model, 0.0)
        router.update(emb, model, reward)
        online_rewards.append(reward)

        step = step_i + 1
        if step in checkpoints:
            holdout_curve[step] = evaluate_holdout(
                router, holdout_data, holdout_emb, models
            )

    return holdout_curve, online_rewards


# ── Main experiment loop ──────────────────────────────────────────────

def run_experiment():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("K-SCALING EXPERIMENT: Hybrid vs Disjoint LinUCB")
    logger.info("=" * 70)

    # One shared dataset for all K — eliminates dataset-size confound
    common_dev, common_holdout = load_common_dataset()

    logger.info("\nLoading encoder and PCA...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    dev_emb = precompute_embeddings(common_dev, encoder, pca)
    holdout_emb = precompute_embeddings(common_holdout, encoder, pca)
    logger.info(f"  Embedded {len(dev_emb)} dev + {len(holdout_emb)} holdout prompts")

    # Within-provider tetrachoric + phi correlations (full K=20 set)
    logger.info("\n── Within-provider reward correlations (holdout, tetrachoric) ──")
    prov_corr, all_reward_vecs = compute_provider_correlations(
        common_holdout, K_CONFIGS[max(K_CONFIGS)]
    )
    for prov, pairs in prov_corr.items():
        for pc in pairs:
            m1, m2 = pc["pair"]
            logger.info(f"  {prov}: {m1.split('/')[-1]} vs {m2.split('/')[-1]}  "
                         f"phi={pc['phi']:.3f}  r_tet={pc['r_tet']:.3f}")

    all_results = {"_meta": {
        "common_dev_prompts": len(common_dev),
        "common_holdout_prompts": len(common_holdout),
        "tetrachoric_threshold": TETRACHORIC_THRESHOLD,
        "provider_correlations": {
            prov: [{"pair": pc["pair"], "phi": pc["phi"],
                    "r_tet": pc["r_tet"], "p": pc["p"]}
                   for pc in pairs]
            for prov, pairs in prov_corr.items()
        },
    }}

    total_steps = len(common_dev)

    for K, models in sorted(K_CONFIGS.items()):
        logger.info(f"\n{'='*70}")
        logger.info(f"K = {K}")
        logger.info(f"{'='*70}")

        fmap = build_family_map_data_driven(all_reward_vecs, models)
        families = defaultdict(list)
        for m, f in fmap.items():
            families[f].append(m)
        n_shared = sum(1 for ms in families.values() if len(ms) > 1)
        members_in_shared = sum(len(ms) for ms in families.values() if len(ms) > 1)

        oracle = compute_oracle(common_holdout, models)
        logger.info(f"  Models: {K}")
        logger.info(f"  Families: {len(families)} ({n_shared} shared, "
                     f"{len(families)-n_shared} singletons)")
        logger.info(f"  Models in shared families: {members_in_shared}/{K}")
        logger.info(f"  Dev prompts: {total_steps}, Holdout: {len(common_holdout)}")
        logger.info(f"  Oracle reward: {oracle:.4f}")
        logger.info(f"  Obs/model (avg): {total_steps/K:.0f}")

        hybrid_holdout = []
        disjoint_holdout = []
        hybrid_online = []
        disjoint_online = []

        for trial in range(N_TRIALS):
            seed = SEED_OFFSET + trial
            logger.info(f"  Trial {trial+1}/{N_TRIALS} (seed={seed})...")

            hc, h_on = run_trial(models, fmap, common_dev, dev_emb,
                                 common_holdout, holdout_emb, seed, total_steps)
            dc, d_on = run_trial(models, None, common_dev, dev_emb,
                                 common_holdout, holdout_emb, seed, total_steps)
            hybrid_holdout.append(hc)
            disjoint_holdout.append(dc)
            hybrid_online.append(h_on)
            disjoint_online.append(d_on)

        # ── Holdout aggregation ───────────────────────────────────────
        all_steps = sorted(hybrid_holdout[0].keys())
        hybrid_hmat = np.array([[c[s] for s in all_steps] for c in hybrid_holdout])
        disjoint_hmat = np.array([[c[s] for s in all_steps] for c in disjoint_holdout])
        hybrid_final = hybrid_hmat[:, -1]
        disjoint_final = disjoint_hmat[:, -1]
        t_final, p_final = sp_stats.ttest_rel(hybrid_final, disjoint_final)

        # ── Online reward aggregation ─────────────────────────────────
        h_online_mat = np.array(hybrid_online)
        d_online_mat = np.array(disjoint_online)

        def rolling_mean(mat, w=ROLLING_WINDOW):
            kernel = np.ones(w) / w
            return np.array([np.convolve(row, kernel, mode="valid")
                             for row in mat])

        h_rolling = rolling_mean(h_online_mat)
        d_rolling = rolling_mean(d_online_mat)
        rolling_steps = np.arange(ROLLING_WINDOW, total_steps + 1)

        h_cum_final = np.sum(h_online_mat, axis=1)
        d_cum_final = np.sum(d_online_mat, axis=1)
        t_cum, p_cum = sp_stats.ttest_rel(h_cum_final, d_cum_final)
        cum_diff = h_cum_final - d_cum_final
        cohens_d_cum = float(np.mean(cum_diff) / (np.std(cum_diff, ddof=1) + 1e-12))

        logger.info(f"\n  Results for K={K}:")
        logger.info(f"    Oracle:               {oracle:.4f}")
        logger.info(f"    Hybrid holdout final: {np.mean(hybrid_final):.4f} "
                     f"+/- {1.96*np.std(hybrid_final)/np.sqrt(N_TRIALS):.4f}")
        logger.info(f"    Disjoint holdout final: {np.mean(disjoint_final):.4f} "
                     f"+/- {1.96*np.std(disjoint_final)/np.sqrt(N_TRIALS):.4f}")
        logger.info(f"    Holdout t-test:  t={t_final:.3f}, p={p_final:.6f}")
        logger.info(f"    Cumulative online reward:")
        logger.info(f"      Hybrid:  {np.mean(h_cum_final):.1f} "
                     f"+/- {1.96*np.std(h_cum_final)/np.sqrt(N_TRIALS):.1f}")
        logger.info(f"      Disjoint: {np.mean(d_cum_final):.1f} "
                     f"+/- {1.96*np.std(d_cum_final)/np.sqrt(N_TRIALS):.1f}")
        logger.info(f"      t={t_cum:.3f}, p={p_cum:.6f}, d={cohens_d_cum:.3f}")

        all_results[K] = {
            "models": models,
            "family_map": fmap,
            "families": {f: ms for f, ms in families.items()},
            "n_families": len(families),
            "n_shared_families": n_shared,
            "members_in_shared": members_in_shared,
            "dev_prompts": total_steps,
            "holdout_prompts": len(common_holdout),
            "obs_per_model_avg": total_steps / K,
            "oracle": oracle,
            "eval_steps": all_steps,
            "hybrid_holdout_mean": np.mean(hybrid_hmat, axis=0).tolist(),
            "hybrid_holdout_ci95": (1.96 * np.std(hybrid_hmat, axis=0)
                                    / np.sqrt(N_TRIALS)).tolist(),
            "disjoint_holdout_mean": np.mean(disjoint_hmat, axis=0).tolist(),
            "disjoint_holdout_ci95": (1.96 * np.std(disjoint_hmat, axis=0)
                                      / np.sqrt(N_TRIALS)).tolist(),
            "hybrid_final": float(np.mean(hybrid_final)),
            "hybrid_final_ci": float(1.96 * np.std(hybrid_final) / np.sqrt(N_TRIALS)),
            "disjoint_final": float(np.mean(disjoint_final)),
            "disjoint_final_ci": float(1.96 * np.std(disjoint_final) / np.sqrt(N_TRIALS)),
            "final_t_stat": float(t_final),
            "final_p_value": float(p_final),
            "rolling_steps": rolling_steps.tolist(),
            "hybrid_rolling_mean": np.mean(h_rolling, axis=0).tolist(),
            "hybrid_rolling_ci95": (1.96 * np.std(h_rolling, axis=0)
                                    / np.sqrt(N_TRIALS)).tolist(),
            "disjoint_rolling_mean": np.mean(d_rolling, axis=0).tolist(),
            "disjoint_rolling_ci95": (1.96 * np.std(d_rolling, axis=0)
                                      / np.sqrt(N_TRIALS)).tolist(),
            "hybrid_cum_mean": float(np.mean(h_cum_final)),
            "hybrid_cum_ci": float(1.96 * np.std(h_cum_final) / np.sqrt(N_TRIALS)),
            "disjoint_cum_mean": float(np.mean(d_cum_final)),
            "disjoint_cum_ci": float(1.96 * np.std(d_cum_final) / np.sqrt(N_TRIALS)),
            "cum_t_stat": float(t_cum),
            "cum_p_value": float(p_cum),
            "cum_cohens_d": cohens_d_cum,
        }

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "k_scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_dir / 'k_scaling_results.json'}")

    generate_figure(all_results, out_dir)
    return all_results


# ── Figure generation ─────────────────────────────────────────────────

def generate_figure(results, out_dir):
    """2-row x 3-col: rolling online reward (top) + holdout (bottom)."""
    k_keys = sorted(k for k in results if k != "_meta")
    fig, axes = plt.subplots(2, len(k_keys), figsize=(5 * len(k_keys), 8))
    if len(k_keys) == 1:
        axes = axes.reshape(2, 1)

    for col, K in enumerate(k_keys):
        r = results[K]
        n_shared = r["n_shared_families"]
        n_fam = r["n_families"]

        # Top: rolling online reward
        ax_top = axes[0, col]
        rs = np.array(r["rolling_steps"])
        h_rm = np.array(r["hybrid_rolling_mean"])
        h_rc = np.array(r["hybrid_rolling_ci95"])
        d_rm = np.array(r["disjoint_rolling_mean"])
        d_rc = np.array(r["disjoint_rolling_ci95"])

        ax_top.fill_between(rs, h_rm - h_rc, h_rm + h_rc, alpha=0.15, color="C0")
        ax_top.fill_between(rs, d_rm - d_rc, d_rm + d_rc, alpha=0.15, color="C1")
        ax_top.plot(rs, h_rm, "C0-", lw=1.5, label=f"Hybrid ({n_shared} shared fam.)")
        ax_top.plot(rs, d_rm, "C1--", lw=1.5, label=f"Disjoint ({K} indep. arms)")

        p_cum = r["cum_p_value"]
        sig = "***" if p_cum < 0.001 else "**" if p_cum < 0.01 else "*" if p_cum < 0.05 else "ns"
        cum_gap = r["hybrid_cum_mean"] - r["disjoint_cum_mean"]
        ax_top.set_title(
            f"K={K}  ({n_fam} fam, {n_shared} shared)\n"
            f"Cum. reward gap: {cum_gap:+.1f} ({sig}, d={r['cum_cohens_d']:.2f})",
            fontsize=10)
        if col == 0:
            ax_top.set_ylabel(f"Rolling online reward\n(window={ROLLING_WINDOW})")
        ax_top.legend(fontsize=7, loc="lower right")
        ax_top.grid(True, alpha=0.3)

        # Bottom: holdout reward
        ax_bot = axes[1, col]
        steps = np.array(r["eval_steps"])
        h_hm = np.array(r["hybrid_holdout_mean"])
        h_hc = np.array(r["hybrid_holdout_ci95"])
        d_hm = np.array(r["disjoint_holdout_mean"])
        d_hc = np.array(r["disjoint_holdout_ci95"])

        ax_bot.fill_between(steps, h_hm - h_hc, h_hm + h_hc, alpha=0.15, color="C0")
        ax_bot.fill_between(steps, d_hm - d_hc, d_hm + d_hc, alpha=0.15, color="C1")
        ax_bot.plot(steps, h_hm, "C0-", lw=1.5, label=f"Hybrid ({n_shared} shared fam.)")
        ax_bot.plot(steps, d_hm, "C1--", lw=1.5, label=f"Disjoint ({K} indep. arms)")
        ax_bot.axhline(r["oracle"], color="gray", ls=":", lw=1,
                       label=f"Oracle ({r['oracle']:.3f})")

        p_f = r["final_p_value"]
        sig_f = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < 0.05 else "ns"
        gap_f = r["hybrid_final"] - r["disjoint_final"]
        ax_bot.set_title(f"Final holdout gap: {gap_f:+.4f} ({sig_f})", fontsize=9)
        ax_bot.set_xlabel("Training steps")
        if col == 0:
            ax_bot.set_ylabel("Holdout reward (greedy)")
        ax_bot.legend(fontsize=7, loc="lower right")
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "k_scaling_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_experiment()
