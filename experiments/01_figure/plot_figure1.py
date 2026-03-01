#!/usr/bin/env python3
"""
Figure 1: Model Preference Heterogeneity

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
  - PCA trained on independent dataset (80K RouteLLM battles)
  - Primary metric: Spearman rank correlation (PC1 vs reward gap)
  - Null baseline: 100 random orthonormal projections (QR-decomposed)

Panel A: PC1 vs Reward Gap — shows features predict model preference
Panel B: Signal vs Null — Router PCA rho vs 100 random projections

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
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, mannwhitneyu
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)

sys.path.insert(0, str(project_root / "experiments"))
from utils.rewards import extract_reward


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_holdout_only(holdout_file: Path):
    """Load holdout data ONLY (no dev contamination)."""
    prompt_rewards = {}
    with gzip.open(holdout_file, 'rt') as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '').strip()
                model_id = entry.get('model_id', '')
                if not prompt or not entry.get('ok'):
                    continue
                reward = extract_reward(entry)
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = reward
                elif 'gpt-4-turbo' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = reward
            except Exception:
                continue

    prompts, reward_gaps = [], []
    for prompt, rewards in prompt_rewards.items():
        if 'mixtral' in rewards and 'gpt4' in rewards:
            prompts.append(prompt)
            reward_gaps.append(rewards['gpt4'] - rewards['mixtral'])

    return prompts, np.array(reward_gaps)


# ══════════════════════════════════════════════════════════════════════════
#  PRIMARY ANALYSIS: SPEARMAN CORRELATION
# ══════════════════════════════════════════════════════════════════════════

def compute_spearman(pc1, reward_gaps):
    """Compute Spearman rank correlation between PC1 and reward gap.

    This is the primary metric: does the router's first principal component
    predict which model will perform better?

    Returns:
        rho: Spearman rank correlation coefficient
        p: two-sided p-value
    """
    rho, p = spearmanr(pc1, reward_gaps)
    return rho, p


def bootstrap_spearman_ci(pc1, reward_gaps, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for Spearman rho.

    Uses case resampling (resample paired observations with replacement)
    to estimate the sampling distribution of Spearman rho.

    Args:
        pc1: array of PC1 values
        reward_gaps: array of reward gaps
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (default 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        ci_low: lower bound of CI
        ci_high: upper bound of CI
        boot_rhos: array of bootstrap rho estimates
    """
    rng = np.random.RandomState(seed)
    n = len(pc1)
    boot_rhos = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_rhos[i], _ = spearmanr(pc1[idx], reward_gaps[idx])

    alpha = 1 - ci
    ci_low = np.percentile(boot_rhos, 100 * alpha / 2)
    ci_high = np.percentile(boot_rhos, 100 * (1 - alpha / 2))
    return ci_low, ci_high, boot_rhos


def random_projection_distribution(embeddings, reward_gaps, n_seeds=100):
    """Compute Spearman rho for N random orthonormal projections.

    Each projection is a QR-decomposed random matrix (384 -> 2).
    We compute |rho| between the first projected dimension and reward gap.

    Returns:
        rhos: array of |rho| values for each valid projection
    """
    rhos = []
    for seed in range(n_seeds):
        rng_i = np.random.RandomState(seed)
        mat_i, _ = np.linalg.qr(rng_i.randn(embeddings.shape[1], 2))
        pc1_i = (embeddings @ mat_i)[:, 0]
        rho_i, _ = spearmanr(pc1_i, reward_gaps)
        rhos.append(abs(rho_i))
    return np.array(rhos)


# ══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY ANALYSIS: OUTCOME HETEROGENEITY
# ══════════════════════════════════════════════════════════════════════════

OUTCOME_ORDER = ['GPT-4T wins', 'Tie', 'Mixtral wins']


def categorize_gap(g, eps=1e-9):
    """Map reward gap to discrete outcome."""
    if g > eps:
        return 'GPT-4T wins'
    elif g < -eps:
        return 'Mixtral wins'
    else:
        return 'Tie'


def outcome_summary(reward_gaps):
    """Compute outcome distribution (supplementary)."""
    pass


def supplementary_clustering(pc1, reward_gaps, prompts):
    """Run threshold-based clustering analysis (supplementary, not primary).

    This provides the contingency table view for readers who want it,
    but is NOT the basis for any claims in the paper.
    """
    pass


def _tfidf_summary(prompts_low, prompts_high, top_n=10):
    """Brief TF-IDF keyword analysis (supplementary)."""
    pass


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

def running_mean(x, y, window=50):
    """Compute a running mean of y sorted by x."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    half_w = window // 2
    x_out, y_out = [], []
    for i in range(half_w, len(x_sorted) - half_w):
        x_out.append(x_sorted[i])
        y_out.append(np.mean(y_sorted[i - half_w:i + half_w]))
    return np.array(x_out), np.array(y_out)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load holdout data ─────────────────────────────────────────────────
    prompts, reward_gaps = load_holdout_only(CANONICAL_HOLDOUT_DATA_PATH)
    outcome_summary(reward_gaps)

    # ── Embed prompts ────────────────────────────────────────────────────
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True
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

    # ── Primary analysis: Spearman correlation ───────────────────────────
    rho, p_rho = compute_spearman(pc1, reward_gaps)
    rho_abs = abs(rho)

    ci_low, ci_high, boot_rhos = bootstrap_spearman_ci(pc1, reward_gaps)
    ci_str = f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    p_str = f"p < 0.0001" if p_rho < 0.0001 else f"p = {p_rho:.4f}"

    # ── Null baseline: 100 random projections ────────────────────────────
    N_RANDOM = 100
    random_rhos = random_projection_distribution(
        embeddings, reward_gaps, n_seeds=N_RANDOM
    )

    rho_median = np.median(random_rhos)
    rho_p25, rho_p75 = np.percentile(random_rhos, [25, 75])
    rho_max = np.max(random_rhos)
    n_exceed = int(np.sum(random_rhos >= rho_abs))
    signal_ratio = rho_abs / rho_median if rho_median > 0 else float('inf')

    # ── Supplementary: Mann-Whitney U ────────────────────────────────────
    median_pc1 = np.median(pc1)
    low_mask = pc1 < median_pc1
    high_mask = pc1 >= median_pc1
    mw_stat, mw_p = mannwhitneyu(
        reward_gaps[low_mask], reward_gaps[high_mask], alternative='two-sided'
    )

    supplementary_clustering(pc1, reward_gaps, prompts)

    # ── Create figure ────────────────────────────────────────────────────

    blue, red, grey = '#4575b4', '#d73027', '#888888'

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1.2, 1.3], 'wspace': 0.30}
    )

    # ── Panel A: PC1 vs Reward Gap ───────────────────────────────────────
    outcomes = np.array([categorize_gap(g) for g in reward_gaps])
    for outcome, color, marker_label in [
        ('Tie', grey, 'Tie'),
        ('GPT-4T wins', blue, 'GPT-4T wins'),
        ('Mixtral wins', red, 'Mixtral wins'),
    ]:
        mask = outcomes == outcome
        ax1.scatter(
            pc1[mask], reward_gaps[mask],
            c=color, s=16, alpha=0.45, edgecolors='none',
            rasterized=True, zorder=2,
            label=f'{marker_label} ({int(mask.sum())})'
        )

    # Running mean trend line
    rm_x, rm_y = running_mean(pc1, reward_gaps, window=60)
    ax1.plot(rm_x, rm_y, color='black', linewidth=2.5, zorder=4,
             label='Running mean (w=60)')

    ax1.axhline(y=0, color=grey, linestyle=':', linewidth=1.0,
                alpha=0.6, zorder=1)

    ax1.set_xlabel('PC1 (router PCA, trained on RouteLLM battles)',
                    fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward gap  (GPT-4-Turbo \u2212 Mixtral)',
                    fontsize=12, fontweight='bold')
    ax1.set_title('(A)  Features Predict Model Preference',
                   fontsize=14, fontweight='bold', pad=8)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.26),
        ncol=4,
        fontsize=10,
        framealpha=0.95,
        edgecolor='#cccccc',
        fancybox=True,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8
    )
    ax1.grid(alpha=0.15, linestyle='--', linewidth=0.5)
    ax1.set_xlim(pc1.min() - 0.03, pc1.max() + 0.03)
    ax1.set_ylim(-1.35, 1.35)

    # ── Panel B: Conditional Win Rates by PC1 Quintile ─────────────────
    n_bins = 5
    bin_edges = np.percentile(pc1, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    gpt4_fracs, tie_fracs, mixtral_fracs, bin_ns = [], [], [], []

    for i in range(n_bins):
        mask = (pc1 >= bin_edges[i]) & (pc1 < bin_edges[i + 1])
        gaps_bin = reward_gaps[mask]
        n_bin = len(gaps_bin)
        bin_ns.append(n_bin)
        cats = [categorize_gap(g) for g in gaps_bin]
        gpt4_fracs.append(cats.count('GPT-4T wins') / n_bin * 100)
        tie_fracs.append(cats.count('Tie') / n_bin * 100)
        mixtral_fracs.append(cats.count('Mixtral wins') / n_bin * 100)

    gpt4_arr = np.array(gpt4_fracs)
    tie_arr = np.array(tie_fracs)
    mixtral_arr = np.array(mixtral_fracs)

    x_pos = np.arange(n_bins)
    bar_width = 0.6

    # Stacked bars
    ax2.bar(x_pos, gpt4_arr, bar_width, color=blue, alpha=0.85,
            label='GPT-4T wins', edgecolor='white', linewidth=0.5)
    ax2.bar(x_pos, tie_arr, bar_width, bottom=gpt4_arr,
            color='#d0d0d0', alpha=0.85,
            label='Tie', edgecolor='white', linewidth=0.5)
    ax2.bar(x_pos, mixtral_arr, bar_width,
            bottom=gpt4_arr + tie_arr, color=red, alpha=0.85,
            label='Mixtral wins', edgecolor='white', linewidth=0.5)

    # Percentage labels (only if segment is tall enough: >= 8%)
    for i in range(n_bins):
        if gpt4_arr[i] >= 8:
            ax2.text(x_pos[i], gpt4_arr[i] / 2,
                     f'{gpt4_arr[i]:.0f}%', ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white')
        tie_y = gpt4_arr[i] + tie_arr[i] / 2
        if tie_arr[i] >= 15:
            ax2.text(x_pos[i], tie_y,
                     f'{tie_arr[i]:.0f}%', ha='center', va='center',
                     fontsize=9, fontweight='bold', color='#444444')
        if mixtral_arr[i] >= 8:
            mix_y = gpt4_arr[i] + tie_arr[i] + mixtral_arr[i] / 2
            ax2.text(x_pos[i], mix_y,
                     f'{mixtral_arr[i]:.0f}%', ha='center', va='center',
                     fontsize=9, fontweight='bold', color='white')

    # Clean single-line x-tick labels: "Q1 (n=150)"
    x_labels = [f'Q{i+1} (n={bin_ns[i]})' for i in range(n_bins)]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=10.5)

    # Directional annotation below x-axis
    ax2.annotate(
        '', xy=(n_bins - 0.7, -8), xytext=(-0.3, -8),
        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
        annotation_clip=False
    )
    ax2.text(
        (n_bins - 1) / 2, -13,
        'PC1 increasing  \u2192  Mixtral preference grows',
        ha='center', va='top', fontsize=14, color='#555555',
        fontstyle='italic', fontweight='bold', clip_on=False
    )

    ax2.set_xlabel('')  # Arrow + text serve as x-axis label
    ax2.set_ylabel('Outcome proportion (%)',
                    fontsize=12, fontweight='bold')
    ax2.set_title('(B)  Routing Opportunity by Feature Region',
                   fontsize=14, fontweight='bold', pad=8)
    ax2.set_ylim(0, 105)
    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.26),
        ncol=3,
        fontsize=10,
        framealpha=0.95,
        edgecolor='#cccccc',
        fancybox=True,
        borderpad=0.4,
        handletextpad=0.5,
        labelspacing=0.3,
        columnspacing=0.8
    )
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)

    # ── Save ──────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.30, top=0.93)
    out_300 = output_dir / "figure1_lmsys_holdout_pca.png"
    fig.savefig(out_300, dpi=300, bbox_inches='tight', facecolor='white')
    out_600 = output_dir / "figure1_lmsys_holdout_pca_hires.png"
    fig.savefig(out_600, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    main()
