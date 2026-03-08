#!/usr/bin/env python3
"""
Figure 1 (contextual): Model Win Probability Shifts with Prompt Features

Demonstrates that per-model reward is a function of prompt context (PC1),
with heterogeneous slopes across models — the core premise for contextual
bandit routing.

Panel A: Per-model OLS regression of reward on standardised PC1, with 95%
         confidence bands.  A likelihood-ratio test compares the per-model-
         slope model against a shared-slope null.

Panel B: Forest plot of per-model contextual slopes (γ_m) with bootstrap
         95% CIs.  Stars mark models whose CI excludes zero.

Feature pipeline (must match production router):
  Sentence embedding → PCA → whitening → standardisation.
  Whitening scales each PCA coordinate by 1/√(explained_variance), so
  components have roughly unit variance under the PCA training distribution.
  This matches the FeatureService / embed_prompt() pipeline used by all
  downstream experiments and the production router.

Methodology:
  - Holdout only (no dev contamination)
  - Reward: weighted rubric composite from judge panel
  - PCA trained on independent dataset (~46K LMSYS arena prompts)
  - PCA features are whitened to match production router pipeline
  - LR test uses min(6, n_components) PCA components; Panel A visualises PC1
  - Bootstrap CIs: 10 000 case-resamples

Usage:
    python3 experiments/01_figure/plot_figure1.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from collections import defaultdict
from typing import Dict, List, Tuple

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    K4_HOLDOUT_DATA_PATH,
    K3_MODELS_PATH,
)

sys.path.insert(0, str(project_root / "experiments"))
from utils.embeddings import load_raw_embedding_cache, get_raw_embeddings_for_prompts

sys.path.insert(0, str(project_root / "experiments"))
from utils.rewards import extract_reward


# ══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO (loaded from canonical config)
# ══════════════════════════════════════════════════════════════════════════

# Visual properties cycled across models in portfolio order.
_COLORS = ["#e41a1c", "#4daf4a", "#1f78b4", "#984ea3", "#ff7f00",
           "#a65628", "#f781bf", "#377eb8", "#6a3d9a", "#b15928"]
_LINESTYLES = ["--", "-", "-", ":", "-", "--", "-", "-", "--", ":"]
_MARKERS = ["o", "p", "X", "^", "s", "D", "v", "h", "P", "*"]


def _load_portfolio(models_path: Path) -> List[Dict]:
    """Load model portfolio from a canonical JSON config file.

    Each entry gets visual metadata (color, linestyle, marker) assigned
    by position so the figure remains deterministic.

    Args:
        models_path: Path to a JSON file with a top-level ``"models"``
            list, each entry having at least ``"model_id"`` and
            ``"display"`` keys.

    Returns:
        List of dicts with keys ``id``, ``display``, ``color``, ``ls``,
        ``marker``.
    """
    with open(models_path) as f:
        raw = json.load(f)["models"]
    portfolio: List[Dict] = []
    for i, m in enumerate(raw):
        portfolio.append({
            "id": m["model_id"],
            "display": m["display"],
            "color": _COLORS[i % len(_COLORS)],
            "ls": _LINESTYLES[i % len(_LINESTYLES)],
            "marker": _MARKERS[i % len(_MARKERS)],
        })
    return portfolio


PORTFOLIO: List[Dict] = _load_portfolio(K3_MODELS_PATH)
MODEL_IDS: List[str] = [m["id"] for m in PORTFOLIO]
_MODEL_MAP: Dict[str, Dict] = {m["id"]: m for m in PORTFOLIO}


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_holdout_rewards(
    holdout_file: Path,
    model_ids: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Load holdout rewards for the target portfolio.

    Args:
        holdout_file: Path to a gzipped JSONL reward file.
        model_ids: Model IDs to include; only prompts with coverage
            across all ``len(model_ids)`` models are retained.

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
    prompts, rewards = load_holdout_rewards(K4_HOLDOUT_DATA_PATH, MODEL_IDS)
    N = len(prompts)
    print(f"  {N} prompts with complete K={K} coverage")

    # ── Embed & PCA (whitened) ────────────────────────────────────────────
    print("Loading embeddings from cache ...")
    raw_cache = load_raw_embedding_cache()
    embeddings = get_raw_embeddings_for_prompts(prompts, raw_cache)
    router_pca = joblib.load(DEFAULT_PCA_PATH)
    X_pca = router_pca.transform(embeddings)

    # Whitening: scale by 1/√(explained_variance) per component.
    # Matches embed_prompt(whiten_pca=True) and FeatureService behaviour.
    pca_has_builtin_whitening = bool(getattr(router_pca, "whiten", False))
    if not pca_has_builtin_whitening:
        ev = getattr(router_pca, "explained_variance_", None)
        if ev is not None:
            whitening_scale = 1.0 / np.sqrt(np.maximum(
                np.asarray(ev, dtype=np.float64), 1e-12,
            ))
            X_pca = X_pca * whitening_scale
            print(f"  Applied external whitening (PCA artifact whiten=False)")
        else:
            print("  WARNING: PCA artifact lacks explained_variance_; "
                  "cannot apply whitening")
    else:
        print(f"  PCA artifact has builtin whitening (whiten=True)")

    print(f"  PCA components: {router_pca.n_components_}, "
          f"explained variance: "
          f"{np.sum(router_pca.explained_variance_ratio_):.1%}")

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
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )

    # ── Panel A: Regression lines with CI bands ──────────────────────────
    x_grid = np.linspace(pc1_std.min() - 0.1, pc1_std.max() + 0.1, 200)

    for m_info in PORTFOLIO:
        mid = m_info["id"]
        alpha_m, gamma_m, sigma_m = fits[mid]
        r_vec = np.array([rewards[p][mid] for p in prompts])

        y_hat, y_lo, y_hi = ols_prediction_band(
            x_grid, pc1_std, alpha_m, gamma_m, sigma_m,
        )

        y_hat_pct = np.clip(y_hat * 100, 0, 100)
        y_lo_pct = np.clip(y_lo * 100, 0, 100)
        y_hi_pct = np.clip(y_hi * 100, 0, 100)

        ax1.fill_between(
            x_grid, y_lo_pct, y_hi_pct,
            alpha=0.12, color=m_info["color"], linewidth=0,
        )
        ax1.plot(
            x_grid, y_hat_pct,
            color=m_info["color"],
            linewidth=2.0,
            linestyle=m_info["ls"],
            label=m_info["display"],
        )

    lr_text = (
        f"LR test: χ² = {chi2_stat:.1f}, df = {df}, p = {p_val:.1e}"
        f"\nN = {N} prompts, K = {K} models"
    )
    ax1.text(
        0.03, 0.03, lr_text,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    ax1.set_xlabel("PC1 (router PCA, whitened)", fontsize=11)
    ax1.set_ylabel("Mean reward (%)", fontsize=11)
    ax1.set_title("(a)  Model Reward Shifts with Prompt Features",
                   fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylim(0, 100)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=9)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=K,
        fontsize=9,
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=True,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=1.0,
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
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=4,
            capthick=1.3,
            elinewidth=1.3,
            ecolor=m_info["color"],
            zorder=3,
        )
        if sig:
            ax2.annotate(
                "*",
                xy=(hi + 0.003, i),
                fontsize=13, fontweight="bold",
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
        "γ$_m$ (contextual slope on standardised whitened PC1)",
        fontsize=11,
    )
    ax2.set_title("(b)  Contextual Sensitivity per Model",
                   fontsize=13, fontweight="bold", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)
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
    out_300 = output_dir / "figure1_k3_contextual.png"
    fig.savefig(out_300, dpi=300, bbox_inches="tight", facecolor="white")
    out_600 = output_dir / "figure1_k3_contextual_hires.png"
    fig.savefig(out_600, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\nFigure saved to {out_300}")
    print("Done.")


if __name__ == "__main__":
    main()
