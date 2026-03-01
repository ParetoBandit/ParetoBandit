#!/usr/bin/env python3
"""
Figure 1 (contextual): Model Win Probability Shifts with Prompt Features (K=10)

Demonstrates that per-model reward is a function of prompt context (PC1),
with heterogeneous slopes across models — the core premise for contextual
bandit routing.

Panel A: Per-model OLS regression of reward on standardised PC1, with 95%
         confidence bands.  A likelihood-ratio test compares the per-model-
         slope model against a shared-slope null.

Panel B: Forest plot of per-model contextual slopes (γ_m) with bootstrap
         95% CIs.  Stars mark models whose CI excludes zero.

Methodology:
  - Holdout only (N=750 prompts, no dev contamination)
  - Reward: mean(vote × confidence) from judge panel
  - PCA trained on independent dataset (80K RouteLLM battles)
  - LR test uses 6 PCA components; Panel A visualises PC1 only
  - Bootstrap CIs: 10 000 case-resamples

Usage:
    python3 experiments/01_figure/plot_figure1_contextual.py
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
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from scipy.stats import chi2
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
#  K=10 PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════

PORTFOLIO_K10: List[Dict] = [
    {"id": "meta-llama/llama-3.1-8b-instruct",              "display": "Llama-3.1-8B",     "color": "#e41a1c", "ls": "--",  "marker": "o"},
    {"id": "mistralai/mixtral-8x7b-instruct",               "display": "Mixtral-8x7B",     "color": "#ff7f00", "ls": "--",  "marker": "s"},
    {"id": "google/gemma-3-27b-it",                         "display": "Gemma-3-27B",      "color": "#a65628", "ls": ":",   "marker": "D"},
    {"id": "anthropic/claude-haiku-4.5",                    "display": "Haiku-4.5",        "color": "#984ea3", "ls": "-",   "marker": "^"},
    {"id": "deepseek/deepseek-chat-v3-0324",                "display": "DeepSeek-V3",      "color": "#377eb8", "ls": "-",   "marker": "v"},
    {"id": "google/gemini-2.5-flash-preview-09-2025",       "display": "Gemini-2.5-Flash", "color": "#4daf4a", "ls": "-",   "marker": "p"},
    {"id": "meta-llama/llama-4-maverick",                   "display": "Llama-4-Maverick", "color": "#f781bf", "ls": "-",   "marker": "h"},
    {"id": "anthropic/claude-sonnet-4",                     "display": "Claude-Sonnet-4",  "color": "#6a3d9a", "ls": "-",   "marker": "P"},
    {"id": "openai/gpt-4-turbo",                            "display": "GPT-4-Turbo",      "color": "#b15928", "ls": "--",  "marker": "*"},
    {"id": "openai/gpt-4.1",                                "display": "GPT-4.1",          "color": "#1f78b4", "ls": "-",   "marker": "X"},
]

MODEL_IDS = [m["id"] for m in PORTFOLIO_K10]
_MODEL_MAP = {m["id"]: m for m in PORTFOLIO_K10}


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_holdout_k10(
    holdout_file: Path,
    model_ids: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Load holdout rewards for the K=10 portfolio.

    Returns:
        prompts: Ordered list of prompt strings (only prompts with all K).
        rewards: ``{prompt: {model_id: float}}`` for complete prompts.
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

    complete = {p: r for p, r in prompt_rewards.items() if len(r) == K}
    prompts = sorted(complete.keys())
    return prompts, {p: complete[p] for p in prompts}


# ══════════════════════════════════════════════════════════════════════════
#  REGRESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Simple OLS: y = alpha + gamma * x.

    Returns:
        alpha: Intercept.
        gamma: Slope.
        sigma: Residual standard deviation.
    """
    n = len(x)
    x_bar = x.mean()
    y_bar = y.mean()
    ss_xx = np.sum((x - x_bar) ** 2)
    ss_xy = np.sum((x - x_bar) * (y - y_bar))
    gamma = ss_xy / ss_xx if ss_xx > 0 else 0.0
    alpha = y_bar - gamma * x_bar
    residuals = y - (alpha + gamma * x)
    sigma = np.sqrt(np.sum(residuals ** 2) / (n - 2)) if n > 2 else 0.0
    return alpha, gamma, sigma


def ols_prediction_band(
    x_grid: np.ndarray,
    x_data: np.ndarray,
    alpha: float,
    gamma: float,
    sigma: float,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence band for the regression line (mean prediction).

    Returns:
        y_hat: Predicted values on x_grid.
        y_lo: Lower CI bound.
        y_hi: Upper CI bound.
    """
    from scipy.stats import t as t_dist

    n = len(x_data)
    x_bar = x_data.mean()
    ss_xx = np.sum((x_data - x_bar) ** 2)
    y_hat = alpha + gamma * x_grid
    t_crit = t_dist.ppf((1 + ci) / 2, df=n - 2)
    se = sigma * np.sqrt(1.0 / n + (x_grid - x_bar) ** 2 / ss_xx)
    return y_hat, y_hat - t_crit * se, y_hat + t_crit * se


def likelihood_ratio_test(
    pc_features: np.ndarray,
    prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    model_ids: List[str],
    n_pcs: int = 6,
) -> Tuple[float, int, float]:
    """LR test: per-model slopes vs shared slopes on n_pcs PCA features.

    H0: reward_{i,m} = α_m + Σ_j β_j · PC_j_i + ε   (shared slopes)
    H1: reward_{i,m} = α_m + Σ_j β_{m,j} · PC_j_i + ε (per-model slopes)

    Returns:
        chi2_stat: LR test statistic.
        df: Degrees of freedom ((K-1) * n_pcs).
        p_value: P-value from chi-squared distribution.
    """
    N = len(prompts)
    K = len(model_ids)
    n_obs = N * K
    d = min(n_pcs, pc_features.shape[1])

    Y = np.empty(n_obs)
    model_dummies = np.zeros((n_obs, K))
    pc_shared = np.zeros((n_obs, d))
    pc_interaction = np.zeros((n_obs, K * d))

    for i, p in enumerate(prompts):
        for j, mid in enumerate(model_ids):
            row = i * K + j
            Y[row] = rewards[p][mid]
            model_dummies[row, j] = 1.0
            pc_shared[row, :] = pc_features[i, :d]
            pc_interaction[row, j * d:(j + 1) * d] = pc_features[i, :d]

    X_h0 = np.hstack([model_dummies, pc_shared])
    X_h1 = np.hstack([model_dummies, pc_interaction])

    def _rss(X, Y):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        return np.sum((Y - X @ beta) ** 2)

    rss_h0 = _rss(X_h0, Y)
    rss_h1 = _rss(X_h1, Y)

    df = (K - 1) * d
    lr_stat = n_obs * np.log(rss_h0 / rss_h1) if rss_h1 > 0 else 0.0
    p_value = 1.0 - chi2.cdf(lr_stat, df)
    return lr_stat, df, p_value


def bootstrap_slopes(
    pc1_std: np.ndarray,
    prompts: List[str],
    rewards: Dict[str, Dict[str, float]],
    model_ids: List[str],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap 95% CIs for per-model OLS slopes on standardised PC1.

    Resamples at the *prompt* level (preserves within-prompt correlation).

    Returns:
        Dict mapping model_id to (gamma_hat, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    N = len(prompts)
    K = len(model_ids)

    reward_matrix = np.empty((N, K))
    for i, p in enumerate(prompts):
        for j, mid in enumerate(model_ids):
            reward_matrix[i, j] = rewards[p][mid]

    point_gammas = np.empty(K)
    for j in range(K):
        _, point_gammas[j], _ = ols_fit(pc1_std, reward_matrix[:, j])

    boot_gammas = np.empty((n_bootstrap, K))
    for b in range(n_bootstrap):
        idx = rng.randint(0, N, size=N)
        pc1_b = pc1_std[idx]
        for j in range(K):
            _, boot_gammas[b, j], _ = ols_fit(pc1_b, reward_matrix[idx, j])

    result = {}
    for j, mid in enumerate(model_ids):
        lo = float(np.percentile(boot_gammas[:, j], 2.5))
        hi = float(np.percentile(boot_gammas[:, j], 97.5))
        result[mid] = (float(point_gammas[j]), lo, hi)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    K = len(MODEL_IDS)

    # ── Load holdout ──────────────────────────────────────────────────────
    print(f"Loading holdout rewards for K={K} portfolio ...")
    prompts, rewards = load_holdout_k10(HOLDOUT_DATA_PATH_ALL_MODELS, MODEL_IDS)
    N = len(prompts)
    print(f"  {N} prompts with complete K={K} coverage")

    # ── Embed & PCA ───────────────────────────────────────────────────────
    print("Embedding prompts ...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True,
    )
    router_pca = joblib.load(DEFAULT_PCA_PATH)
    X_pca = router_pca.transform(embeddings)

    pc1_raw = X_pca[:, 0]
    pc1_mean, pc1_std_val = pc1_raw.mean(), pc1_raw.std()
    pc1_std = (pc1_raw - pc1_mean) / pc1_std_val

    # ── LR test (6 PCs) ──────────────────────────────────────────────────
    N_PCS_LR = 6
    pc_std_all = (X_pca[:, :N_PCS_LR] - X_pca[:, :N_PCS_LR].mean(0)) / X_pca[:, :N_PCS_LR].std(0)
    chi2_stat, df, p_val = likelihood_ratio_test(
        pc_std_all, prompts, rewards, MODEL_IDS, n_pcs=N_PCS_LR,
    )
    print(f"\n  LR test: χ² = {chi2_stat:.1f}, df = {df}, p = {p_val:.2e}")

    # ── Per-model OLS on PC1 ─────────────────────────────────────────────
    print("\n  Per-model OLS (reward ~ PC1_std):")
    fits = {}
    for mid in MODEL_IDS:
        r_vec = np.array([rewards[p][mid] for p in prompts])
        alpha, gamma, sigma = ols_fit(pc1_std, r_vec)
        fits[mid] = (alpha, gamma, sigma)
        print(f"    {_MODEL_MAP[mid]['display']:<22} "
              f"α={alpha:.4f}  γ={gamma:+.4f}  σ={sigma:.4f}")

    # ── Bootstrap slopes ──────────────────────────────────────────────────
    print("\n  Bootstrap slopes (10 000 resamples) ...")
    slope_cis = bootstrap_slopes(pc1_std, prompts, rewards, MODEL_IDS)
    n_sig = 0
    print("  Per-model γ_m [95% CI]:")
    for mid in sorted(MODEL_IDS, key=lambda m: abs(slope_cis[m][0]), reverse=True):
        g, lo, hi = slope_cis[mid]
        sig = lo > 0 or hi < 0
        if sig:
            n_sig += 1
        star = " *" if sig else ""
        print(f"    {_MODEL_MAP[mid]['display']:<22} γ={g:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")
    print(f"  {n_sig}/{K} significant (bootstrap 95% CI excl. 0)")

    # ══════════════════════════════════════════════════════════════════════
    #  FIGURE
    # ══════════════════════════════════════════════════════════════════════

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.5),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.35},
    )

    # ── Panel A: Regression lines with CI bands ──────────────────────────
    x_grid = np.linspace(pc1_std.min() - 0.1, pc1_std.max() + 0.1, 200)

    for m_info in PORTFOLIO_K10:
        mid = m_info["id"]
        alpha_m, gamma_m, sigma_m = fits[mid]
        r_vec = np.array([rewards[p][mid] for p in prompts])

        y_hat, y_lo, y_hi = ols_prediction_band(
            x_grid, pc1_std, alpha_m, gamma_m, sigma_m,
        )

        ax1.fill_between(
            x_grid, y_lo * 100, y_hi * 100,
            alpha=0.12, color=m_info["color"], linewidth=0,
        )
        ax1.plot(
            x_grid, y_hat * 100,
            color=m_info["color"],
            linewidth=2.0,
            linestyle=m_info["ls"],
            label=m_info["display"],
        )

    lr_text = f"LR test: χ² = {chi2_stat:.1f}, df = {df}, p = {p_val:.1e}"
    ax1.text(
        0.03, 0.97, lr_text,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax1.set_xlabel("PC1 (router PCA)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean reward  (% scale)", fontsize=12, fontweight="bold")
    ax1.set_title("(A)  Model Reward Shifts with Prompt Features",
                   fontsize=14, fontweight="bold", pad=8)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
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

    # ── Panel B: Forest plot of contextual slopes ────────────────────────
    sorted_mids = sorted(MODEL_IDS, key=lambda m: slope_cis[m][0])
    y_positions = np.arange(K)

    for i, mid in enumerate(sorted_mids):
        g, lo, hi = slope_cis[mid]
        m_info = _MODEL_MAP[mid]
        sig = lo > 0 or hi < 0

        ax2.errorbar(
            g, i,
            xerr=[[g - lo], [hi - g]],
            fmt=m_info["marker"],
            color=m_info["color"],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            ecolor=m_info["color"],
            zorder=3,
        )
        if sig:
            ax2.annotate(
                "*",
                xy=(hi + 0.003, i),
                fontsize=14, fontweight="bold",
                color=m_info["color"],
                va="center",
            )

    ax2.axvline(x=0, color="#555555", linestyle=":", linewidth=1.0, zorder=1)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(
        [_MODEL_MAP[mid]["display"] for mid in sorted_mids],
        fontsize=10,
    )
    ax2.set_xlabel(
        "γ$_m$\n(contextual slope on standardised PC1)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_title("(B)  Contextual Sensitivity per Model",
                   fontsize=14, fontweight="bold", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.15, linestyle="--", linewidth=0.5)

    ax2.text(
        0.97, 0.04,
        f"{n_sig}/{K} significant\n(bootstrap 95% CI excl. 0)",
        transform=ax2.transAxes,
        fontsize=9, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7",
                  edgecolor="#cccccc", alpha=0.9),
    )

    # ── Save ──────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.24, top=0.93)
    out_300 = output_dir / "figure1_k10_contextual.png"
    fig.savefig(out_300, dpi=300, bbox_inches="tight", facecolor="white")
    out_600 = output_dir / "figure1_k10_contextual_hires.png"
    fig.savefig(out_600, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure saved to {out_300}")
    print("Done.")


if __name__ == "__main__":
    main()
