#!/usr/bin/env python3
"""
Figure G1: Model Preference Heterogeneity (K=3)

Establishes the empirical motivation for contextual routing:
  1. Model preference varies by prompt (not all prompts favor the same model)
  2. The router's PCA features predict this variation
  3. This prediction exceeds what any random projection achieves

This directly motivates BanditGPT's design: because features predict reward,
a contextual bandit can learn to route. If preference were uniform, a static
policy would be optimal and learned routing would be unnecessary.

Methodology:
  - Uses the SAME feature pipeline as router.py (FeatureService)
  - Holdout only (N=750, no dev contamination)
  - PCA trained on independent dataset (~46K LMSYS arena prompts)
  - Reward signal: mean(vote × confidence) from judge panel
  - Primary metric: Spearman rank correlation (PC1 vs oracle gain)
  - Null baseline: 100 random orthonormal projections (QR-decomposed)

Panel A: Per-model reward vs PC1 — running means show preference crossings
Panel B: Best-model distribution by PC1 quintile — routing opportunity varies

Usage:
    python3 experiments/appendix/G_preference_heterogeneity/plot_figure_g1.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
from collections import defaultdict
from typing import Dict, List, Tuple

from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)

sys.path.insert(0, str(project_root / "experiments"))
from utils.rewards import extract_reward


# ══════════════════════════════════════════════════════════════════════════
#  K=3 PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════

PORTFOLIO_K3: List[Dict] = [
    {"id": "meta-llama/llama-3.1-8b-instruct",              "display": "Llama-3.1-8B",     "color": "#e41a1c", "tier": "cheap"},
    {"id": "google/gemini-2.5-flash",                        "display": "Gemini-2.5-Flash", "color": "#4daf4a", "tier": "mid"},
    {"id": "openai/gpt-4.1",                                "display": "GPT-4.1",          "color": "#1f78b4", "tier": "expensive"},
]

MODEL_IDS = [m["id"] for m in PORTFOLIO_K3]
MODEL_DISPLAY = {m["id"]: m["display"] for m in PORTFOLIO_K3}
MODEL_COLORS = {m["id"]: m["color"] for m in PORTFOLIO_K3}


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_holdout_k3(
    holdout_file: Path,
    model_ids: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Load holdout rewards for the K=3 portfolio.

    Reads gzipped JSONL, filters to the target models, and computes
    ``mean(vote × confidence)`` via :func:`extract_reward`.  Only
    prompts with rewards for all K models are retained.

    Args:
        holdout_file: Path to the gzipped JSONL holdout data.
        model_ids: List of model identifiers in the portfolio.

    Returns:
        prompts: Ordered list of prompt strings.
        rewards: ``{prompt: {model_id: reward}}`` for complete prompts.
    """
    model_set = set(model_ids)
    K = len(model_ids)
    prompt_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(holdout_file, "rt") as f:
        for line in f:
            entry = json.loads(line)
            mid = entry.get("model_id", "")
            if mid not in model_set:
                continue
            prompt = entry.get("prompt", "").strip()
            if not prompt:
                continue
            prompt_rewards[prompt][mid] = extract_reward(entry)

    complete = {
        p: r for p, r in prompt_rewards.items() if len(r) == K
    }
    prompts = sorted(complete.keys())
    return prompts, {p: complete[p] for p in prompts}


# ══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def per_model_spearman(
    pc1: np.ndarray,
    prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    model_ids: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Spearman(PC1, reward) for each model.

    Returns:
        Dict mapping model_id to (rho, p-value).
    """
    result = {}
    for mid in model_ids:
        r_vec = np.array([rewards[p][mid] for p in prompts])
        rho, pval = spearmanr(pc1, r_vec)
        result[mid] = (rho, pval)
    return result


def oracle_gain(
    prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    model_ids: List[str],
) -> np.ndarray:
    """Per-prompt oracle gain: max(reward) - mean(reward) across models.

    Measures the per-prompt routing opportunity: how much better the oracle
    router does compared to uniform random selection.
    """
    gains = np.empty(len(prompts))
    for i, p in enumerate(prompts):
        vals = [rewards[p][m] for m in model_ids]
        gains[i] = max(vals) - np.mean(vals)
    return gains


def best_model_per_prompt(
    prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    model_ids: List[str],
) -> np.ndarray:
    """Return the model_id of the best model for each prompt."""
    return np.array([
        max(model_ids, key=lambda m: rewards[p][m]) for p in prompts
    ])


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap 95% CI for Spearman rho via case resampling.

    Args:
        x: First variable.
        y: Second variable.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        ci_low: Lower bound.
        ci_high: Upper bound.
        boot_rhos: All bootstrap rho values.
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_rhos = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_rhos[i], _ = spearmanr(x[idx], y[idx])
    alpha = 1 - ci
    return (
        float(np.percentile(boot_rhos, 100 * alpha / 2)),
        float(np.percentile(boot_rhos, 100 * (1 - alpha / 2))),
        boot_rhos,
    )


def random_projection_null(
    embeddings: np.ndarray,
    target: np.ndarray,
    n_seeds: int = 100,
) -> np.ndarray:
    """Null distribution: |Spearman(random_proj_dim1, target)| for N seeds.

    Each projection is a QR-decomposed random matrix (384 → 2).
    """
    rhos = []
    for seed in range(n_seeds):
        rng_i = np.random.RandomState(seed)
        mat_i, _ = np.linalg.qr(rng_i.randn(embeddings.shape[1], 2))
        proj_dim1 = (embeddings @ mat_i)[:, 0]
        rho_i, _ = spearmanr(proj_dim1, target)
        rhos.append(abs(rho_i))
    return np.array(rhos)


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def running_mean(x: np.ndarray, y: np.ndarray, window: int = 50):
    """Compute a running mean of y sorted by x."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    half_w = window // 2
    x_out, y_out = [], []
    for i in range(half_w, len(x_s) - half_w):
        x_out.append(x_s[i])
        y_out.append(np.mean(y_s[i - half_w:i + half_w]))
    return np.array(x_out), np.array(y_out)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    K = len(MODEL_IDS)

    # ── Load holdout data ─────────────────────────────────────────────────
    print(f"Loading holdout rewards for K={K} portfolio ...")
    prompts, rewards = load_holdout_k3(
        HOLDOUT_DATA_PATH_ALL_MODELS, MODEL_IDS,
    )
    N = len(prompts)
    print(f"  {N} prompts with complete K={K} coverage")

    # ── Embed prompts ────────────────────────────────────────────────────
    print("Embedding prompts ...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True,
    )

    # ── Load Router PCA ──────────────────────────────────────────────────
    if not DEFAULT_PCA_PATH.exists():
        raise FileNotFoundError(
            f"Router PCA not found at {DEFAULT_PCA_PATH}. "
            f"Run: python3 scripts/train_pca_from_routellm.py --n-components 32"
        )
    router_pca = joblib.load(DEFAULT_PCA_PATH)
    X_pca = router_pca.transform(embeddings)
    pc1 = X_pca[:, 0]

    # ── Per-model Spearman ───────────────────────────────────────────────
    model_rhos = per_model_spearman(pc1, prompts, rewards, MODEL_IDS)
    print("\n  Per-model Spearman(PC1, reward):")
    for mid in sorted(MODEL_IDS, key=lambda m: abs(model_rhos[m][0]), reverse=True):
        rho, pval = model_rhos[mid]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"    {MODEL_DISPLAY[mid]:<22} ρ={rho:+.4f}  p={pval:.2e}  {sig}")

    # ── Oracle gain analysis ─────────────────────────────────────────────
    og = oracle_gain(prompts, rewards, MODEL_IDS)
    rho_og, p_og = spearmanr(pc1, og)
    ci_low, ci_high, boot_rhos = bootstrap_spearman_ci(pc1, og)
    p_str = "p < 0.0001" if p_og < 0.0001 else f"p = {p_og:.4f}"
    print(f"\n  Oracle gain ~ PC1:  ρ = {rho_og:+.4f}  {p_str}")
    print(f"    95% CI [{ci_low:.3f}, {ci_high:.3f}]")

    # ── Null baseline ────────────────────────────────────────────────────
    N_RANDOM = 100
    random_rhos = random_projection_null(embeddings, og, n_seeds=N_RANDOM)
    rho_abs = abs(rho_og)
    n_exceed = int(np.sum(random_rhos >= rho_abs))
    signal_ratio = rho_abs / np.median(random_rhos) if np.median(random_rhos) > 0 else float("inf")
    print(f"    Null: median |ρ| = {np.median(random_rhos):.4f}, "
          f"max = {np.max(random_rhos):.4f}")
    print(f"    Router |ρ| exceeds {N_RANDOM - n_exceed}/{N_RANDOM} random projections "
          f"({signal_ratio:.1f}× median)")

    # ── Best-model distribution ──────────────────────────────────────────
    best_models = best_model_per_prompt(prompts, rewards, MODEL_IDS)
    print("\n  Best-model distribution:")
    for mid in MODEL_IDS:
        cnt = int(np.sum(best_models == mid))
        print(f"    {MODEL_DISPLAY[mid]:<22} {cnt:>4} ({cnt / N * 100:5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    #  FIGURE
    # ══════════════════════════════════════════════════════════════════════

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.5),
        gridspec_kw={"width_ratios": [1.2, 1.3], "wspace": 0.32},
    )

    # ── Panel A: Per-model reward running means vs PC1 ───────────────────
    WINDOW = 80
    tier_styles = {"cheap": "--", "mid": "-", "expensive": "-"}
    tier_widths = {"cheap": 1.8, "mid": 2.2, "expensive": 2.8}

    for m_info in PORTFOLIO_K3:
        mid = m_info["id"]
        r_vec = np.array([rewards[p][mid] for p in prompts])
        rm_x, rm_y = running_mean(pc1, r_vec, window=WINDOW)
        ax1.plot(
            rm_x, rm_y,
            color=m_info["color"],
            linewidth=tier_widths[m_info["tier"]],
            linestyle=tier_styles[m_info["tier"]],
            label=m_info["display"],
            alpha=0.85,
        )

    ax1.set_xlabel("PC1 (router PCA)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean reward  [mean(vote × conf)]",
                    fontsize=12, fontweight="bold")
    ax1.set_title("(A)  Per-Model Quality Varies with Features",
                   fontsize=14, fontweight="bold", pad=8)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8,
    )
    ax1.grid(alpha=0.15, linestyle="--", linewidth=0.5)
    ax1.set_xlim(pc1.min() - 0.02, pc1.max() + 0.02)

    # ── Panel B: Best-model stacked bars by PC1 quintile ─────────────────
    n_bins = 5
    bin_edges = np.percentile(pc1, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    fracs = {mid: [] for mid in MODEL_IDS}
    bin_ns = []

    for i in range(n_bins):
        mask = (pc1 >= bin_edges[i]) & (pc1 < bin_edges[i + 1])
        n_bin = int(mask.sum())
        bin_ns.append(n_bin)
        best_in_bin = best_models[mask]
        for mid in MODEL_IDS:
            fracs[mid].append(np.sum(best_in_bin == mid) / n_bin * 100)

    x_pos = np.arange(n_bins)
    bar_width = 0.65
    bottom = np.zeros(n_bins)

    for m_info in PORTFOLIO_K3:
        mid = m_info["id"]
        vals = np.array(fracs[mid])
        ax2.bar(
            x_pos, vals, bar_width,
            bottom=bottom,
            color=m_info["color"],
            alpha=0.88,
            label=m_info["display"],
            edgecolor="white",
            linewidth=0.4,
        )
        for i in range(n_bins):
            if vals[i] >= 7:
                ax2.text(
                    x_pos[i], bottom[i] + vals[i] / 2,
                    f"{vals[i]:.0f}%",
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white",
                )
        bottom += vals

    x_labels = [f"Q{i + 1}\n(n={bin_ns[i]})" for i in range(n_bins)]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=10)

    ax2.set_xlabel("PC1 quintile", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Best-model share (%)", fontsize=12, fontweight="bold")
    ax2.set_title("(B)  Routing Opportunity by Feature Region",
                   fontsize=14, fontweight="bold", pad=8)
    ax2.set_ylim(0, 105)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        fontsize=8.5,
        framealpha=0.95,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.15, linestyle="--", linewidth=0.5)

    # ── Annotation: primary statistic ────────────────────────────────────
    fig.text(
        0.5, 0.96,
        f"Spearman(PC1, oracle gain): ρ = {rho_og:+.3f}  "
        f"[{ci_low:.3f}, {ci_high:.3f}]  {p_str}   |   "
        f"N = {N} prompts, K = {K} models",
        ha="center", va="bottom", fontsize=10.5,
        fontstyle="italic", color="#333333",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.28, top=0.92)
    out_300 = output_dir / "figure1_k3_heterogeneity.png"
    fig.savefig(out_300, dpi=300, bbox_inches="tight", facecolor="white")
    out_600 = output_dir / "figure1_k3_heterogeneity_hires.png"
    fig.savefig(out_600, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure saved to {out_300}")
    print("Done.")


if __name__ == "__main__":
    main()
