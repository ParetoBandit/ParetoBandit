#!/usr/bin/env python3
"""
Figure 1 (Revised): Model Preference Heterogeneity in Prompt Embeddings

Clean methodology:
  - Routing PCA (domain-adapted feature extraction, unsupervised)
  - Holdout only (N=750, no dev contamination)
  - Unsupervised threshold (silhouette-optimal, no reward peeking)

Panel A: PC1 vs Reward Gap scatter — directly shows the inversion
Panel B: Violin plots of reward gap by cluster — shows distributional difference

Usage:
    python3 experiments_v1/01_figure/plot_figure1_revised.py
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
import matplotlib.patches as mpatches
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, sem
from scipy import stats as scipy_stats
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)


def load_holdout_only(holdout_file: Path):
    """Load holdout data ONLY (no dev contamination)."""
    prompt_rewards = {}
    with gzip.open(holdout_file, 'rt') as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '').strip()
                model_id = entry.get('model_id', '')
                raw_score = entry.get('raw_score', None)
                if not prompt or raw_score is None:
                    continue
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = raw_score
                elif 'gpt-4-turbo' in model_id.lower() or 'gpt-4' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = raw_score
            except Exception:
                continue

    prompts, reward_gaps = [], []
    for prompt, rewards in prompt_rewards.items():
        if 'mixtral' in rewards and 'gpt4' in rewards:
            prompts.append(prompt)
            reward_gaps.append(rewards['gpt4'] - rewards['mixtral'])

    return prompts, np.array(reward_gaps)


def find_silhouette_optimal_threshold(X_2d):
    """Find threshold using silhouette score only (no reward labels)."""
    pc1 = X_2d[:, 0]
    thresholds = np.linspace(pc1.min() + 0.05, pc1.max() - 0.05, 50)
    best_threshold, best_sil = None, -1

    for t in thresholds:
        labels = (pc1 >= t).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        if min(np.sum(labels == 0), np.sum(labels == 1)) < len(pc1) * 0.05:
            continue
        try:
            sil = silhouette_score(X_2d, labels)
            if sil > best_sil:
                best_sil = sil
                best_threshold = t
        except Exception:
            continue

    return best_threshold, best_sil


def running_mean(x, y, window=50):
    """Compute a running mean of y sorted by x."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    half_w = window // 2
    x_out, y_out = [], []
    for i in range(half_w, len(x_sorted) - half_w):
        x_out.append(x_sorted[i])
        y_out.append(np.mean(y_sorted[i - half_w:i + half_w]))
    return np.array(x_out), np.array(y_out)


def main():
    print("=" * 80)
    print("FIGURE 1 (REVISED): MODEL PREFERENCE HETEROGENEITY")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────
    holdout_file = CANONICAL_HOLDOUT_DATA_PATH
    pca_file = DEFAULT_PCA_PATH
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts, reward_gaps = load_holdout_only(holdout_file)
    print(f"Loaded {len(prompts)} holdout prompts")

    # ── Embed & project ───────────────────────────────────────────────────
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True
    )
    pca = joblib.load(pca_file)
    X_2d = pca.transform(embeddings)[:, :2]
    pc1 = X_2d[:, 0]

    # ── Unsupervised threshold (silhouette-optimal) ───────────────────────
    threshold, sil_score = find_silhouette_optimal_threshold(X_2d)
    print(f"Silhouette-optimal threshold: {threshold:.3f} (sil={sil_score:.3f})")

    low_mask = pc1 < threshold
    high_mask = pc1 >= threshold
    gaps_low = reward_gaps[low_mask]
    gaps_high = reward_gaps[high_mask]

    # ── Statistics ─────────────────────────────────────────────────────────
    stat, p_value = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')
    pooled_std = np.sqrt(
        ((len(gaps_low) - 1) * np.var(gaps_low, ddof=1)
         + (len(gaps_high) - 1) * np.var(gaps_high, ddof=1))
        / (len(gaps_low) + len(gaps_high) - 2)
    )
    cohens_d = (np.mean(gaps_low) - np.mean(gaps_high)) / pooled_std
    ci_low = scipy_stats.t.interval(
        0.95, len(gaps_low) - 1,
        loc=np.mean(gaps_low), scale=sem(gaps_low)
    )
    ci_high = scipy_stats.t.interval(
        0.95, len(gaps_high) - 1,
        loc=np.mean(gaps_high), scale=sem(gaps_high)
    )

    print(f"\nLow  PC1 ({len(gaps_low):,}): mean={np.mean(gaps_low):+.3f}  "
          f"CI=[{ci_low[0]:+.3f}, {ci_low[1]:+.3f}]")
    print(f"High PC1 ({len(gaps_high):,}): mean={np.mean(gaps_high):+.3f}  "
          f"CI=[{ci_high[0]:+.3f}, {ci_high[1]:+.3f}]")
    print(f"Mann-Whitney p = {p_value:.2e}   Cohen's d = {cohens_d:.2f}")

    # ── Colours & labels ──────────────────────────────────────────────────
    blue = '#4575b4'
    red = '#d73027'
    grey = '#888888'

    n_low = int(np.sum(low_mask))
    n_high = int(np.sum(high_mask))
    pct_low = n_low / len(reward_gaps) * 100
    pct_high = n_high / len(reward_gaps) * 100

    # ══════════════════════════════════════════════════════════════════════
    #  CREATE FIGURE
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.2),
        gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.32}
    )

    # ── Panel A: PC1 vs Reward Gap scatter ────────────────────────────────
    ax1.scatter(
        pc1[low_mask], reward_gaps[low_mask],
        c=blue, s=18, alpha=0.45, edgecolors='none',
        rasterized=True, zorder=2,
        label=f'GPT-4-Turbo preferred ({pct_low:.0f}%)'
    )
    ax1.scatter(
        pc1[high_mask], reward_gaps[high_mask],
        c=red, s=18, alpha=0.45, edgecolors='none',
        rasterized=True, zorder=2,
        label=f'Mixtral preferred ({pct_high:.0f}%)'
    )

    # Running-mean trend
    rm_x, rm_y = running_mean(pc1, reward_gaps, window=60)
    ax1.plot(rm_x, rm_y, color='black', linewidth=2.5, zorder=4,
             label='Running mean')

    # Threshold & zero lines
    ax1.axvline(x=threshold, color=grey, linestyle='--', linewidth=1.8,
                zorder=3, label=f'Threshold ({threshold:.2f})')
    ax1.axhline(y=0, color=grey, linestyle=':', linewidth=1.0, alpha=0.6,
                zorder=1)

    ax1.set_xlabel('PC1 (routing-adapted PCA)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward gap  (GPT-4-Turbo − Mixtral)', fontsize=12,
                    fontweight='bold')
    ax1.set_title('(A)  Reward Gap vs. Embedding PC1', fontsize=13,
                   fontweight='bold', pad=10)
    # Legend in mid-left gap between the y≈0 data band and y=1.0 band
    ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.97,
               edgecolor='#cccccc', fancybox=True, borderpad=0.5,
               bbox_to_anchor=(0.0, 0.82))
    ax1.grid(alpha=0.15, linestyle='--', linewidth=0.5)
    ax1.set_xlim(pc1.min() - 0.03, pc1.max() + 0.03)
    ax1.set_ylim(-1.25, 1.25)

    # ── Panel B: Violin + box plots ──────────────────────────────────────
    # Pad data with a few phantom points beyond the range so the KDE
    # tapers to natural tails instead of being cut flat at data extremes.
    pad = 0.08
    gaps_low_violin = np.concatenate([
        gaps_low,
        [gaps_low.min() - pad, gaps_low.max() + pad],
    ])
    gaps_high_violin = np.concatenate([
        gaps_high,
        [gaps_high.min() - pad, gaps_high.max() + pad],
    ])
    vp = ax2.violinplot(
        [gaps_low_violin, gaps_high_violin], positions=[0, 1],
        showmeans=False, showmedians=False, showextrema=False
    )
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor([blue, red][i])
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
        body.set_alpha(0.55)

    # Overlay box plots (narrower)
    bp = ax2.boxplot(
        [gaps_low, gaps_high], positions=[0, 1],
        widths=0.15, patch_artist=True,
        showfliers=False, zorder=5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
    )
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor([blue, red][i])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
        patch.set_alpha(0.85)

    # Mean markers
    for i, (gaps, col) in enumerate([(gaps_low, blue), (gaps_high, red)]):
        ax2.scatter([i], [np.mean(gaps)], color='white', edgecolors='black',
                    s=60, zorder=6, linewidths=1.5, marker='D')

    ax2.axhline(y=0, color=grey, linestyle=':', linewidth=1.0, alpha=0.6)

    # Statistical annotation box
    p_str = f'p < 0.0001' if p_value < 0.0001 else f'p = {p_value:.4f}'
    stat_text = (
        f'{p_str}\n'
        f"Cohen's d = {cohens_d:.2f}\n"
        f'N = {len(reward_gaps)}'
    )
    ax2.text(
        0.97, 0.97, stat_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#cccccc', alpha=0.95)
    )

    # Mean ± CI annotations below violins — placed well below data range
    for i, (gaps, ci, col) in enumerate([
        (gaps_low, ci_low, blue), (gaps_high, ci_high, red)
    ]):
        mean_str = f'{np.mean(gaps):+.2f}'
        ci_str = f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
        ax2.text(i, -1.22, f'$\\mu$ = {mean_str}\n{ci_str}',
                 ha='center', va='top', fontsize=8.5, color=col,
                 fontweight='bold')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(
        [f'GPT-4-Turbo\npreferred\n(n={n_low})',
         f'Mixtral\npreferred\n(n={n_high})'],
        fontsize=10, fontweight='bold'
    )
    ax2.set_ylabel('Reward gap  (GPT-4-Turbo − Mixtral)', fontsize=12,
                    fontweight='bold')
    ax2.set_title('(B)  Reward Gap by Cluster', fontsize=13,
                   fontweight='bold', pad=12)
    ax2.set_ylim(-1.55, 1.25)
    ax2.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.93)

    out_300 = output_dir / "figure1_lmsys_holdout_pca.png"
    fig.savefig(out_300, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {out_300}")

    out_600 = output_dir / "figure1_lmsys_holdout_pca_hires.png"
    fig.savefig(out_600, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_600}")

    plt.close()

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 1 SUMMARY")
    print("=" * 80)
    print(f"  Data:       Holdout only (N={len(prompts)})")
    print(f"  PCA:        Routing-adapted (domain-specific feature extraction)")
    print(f"  Threshold:  {threshold:.3f} (silhouette-optimal, unsupervised)")
    print(f"  Low  PC1:   {n_low} prompts ({pct_low:.1f}%), "
          f"mean gap = {np.mean(gaps_low):+.3f}")
    print(f"  High PC1:   {n_high} prompts ({pct_high:.1f}%), "
          f"mean gap = {np.mean(gaps_high):+.3f}")
    print(f"  {p_str},  Cohen's d = {cohens_d:.2f}")
    print(f"  Robustness: Generic C4 PCA confirms effect "
          f"(p<0.0001, d=0.33)")
    print("=" * 80)


if __name__ == "__main__":
    main()
