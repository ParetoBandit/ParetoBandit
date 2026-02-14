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
    python3 experiments_v1/01_figure/plot_figure1.py
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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)


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
                raw_score = entry.get('raw_score', None)
                if not prompt or raw_score is None:
                    continue
                if prompt not in prompt_rewards:
                    prompt_rewards[prompt] = {}
                if 'mixtral' in model_id.lower():
                    prompt_rewards[prompt]['mixtral'] = raw_score
                elif 'gpt-4-turbo' in model_id.lower():
                    prompt_rewards[prompt]['gpt4'] = raw_score
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
    """Print outcome distribution (supplementary)."""
    cats = [categorize_gap(g) for g in reward_gaps]
    n = len(cats)
    print(f"\n── Outcome Distribution (N={n}) ──")
    for outcome in OUTCOME_ORDER:
        count = cats.count(outcome)
        print(f"  {outcome:<15s}: {count:4d} ({count/n*100:.1f}%)")


def supplementary_clustering(pc1, reward_gaps, prompts):
    """Run threshold-based clustering analysis (supplementary, not primary).

    This provides the contingency table view for readers who want it,
    but is NOT the basis for any claims in the paper.
    """
    print("\n── Supplementary: Threshold-Based Clustering ──")
    print("  (For exploratory context only — primary metric is Spearman rho)")

    # Simple median split
    threshold = np.median(pc1)
    low_mask = pc1 < threshold
    high_mask = pc1 >= threshold
    gaps_low = reward_gaps[low_mask]
    gaps_high = reward_gaps[high_mask]

    print(f"\n  Median split at PC1 = {threshold:.3f}")
    print(f"  Low PC1:  n={int(low_mask.sum())}")
    print(f"  High PC1: n={int(high_mask.sum())}")

    # Outcome proportions per group
    for label, gaps in [("Low PC1", gaps_low), ("High PC1", gaps_high)]:
        cats = [categorize_gap(g) for g in gaps]
        n = len(cats)
        parts = [f"{o}: {cats.count(o)/n*100:.1f}%" for o in OUTCOME_ORDER]
        print(f"  {label}: {', '.join(parts)}")

    # TF-IDF content analysis
    prompts_arr = np.array(prompts)
    _tfidf_summary(list(prompts_arr[low_mask]), list(prompts_arr[high_mask]))


def _tfidf_summary(prompts_low, prompts_high, top_n=10):
    """Brief TF-IDF keyword analysis (supplementary)."""
    all_prompts = prompts_low + prompts_high
    labels_arr = np.array([0] * len(prompts_low) + [1] * len(prompts_high))

    tfidf = TfidfVectorizer(
        max_features=5000, stop_words='english',
        ngram_range=(1, 2), min_df=3, max_df=0.8,
    )
    X = tfidf.fit_transform(all_prompts)
    feature_names = np.array(tfidf.get_feature_names_out())
    mean_low = np.asarray(X[labels_arr == 0].mean(axis=0)).ravel()
    mean_high = np.asarray(X[labels_arr == 1].mean(axis=0)).ravel()
    eps = 1e-8
    ratio_high = mean_high / (mean_low + eps)
    top_high = np.argsort(ratio_high)[::-1][:top_n]

    print(f"\n  Top {top_n} distinctive terms (High PC1):")
    for i, idx in enumerate(top_high, 1):
        print(f"    {i:2d}. {feature_names[idx]}")


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
    print("=" * 80)
    print("FIGURE 1: MODEL PREFERENCE HETEROGENEITY")
    print("=" * 80)
    print("\nQuestion: Do the router's features predict model preference?")
    print("Method:   Spearman correlation (PC1 vs reward gap)")
    print("Null:     100 random orthonormal projections")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load holdout data ─────────────────────────────────────────────────
    prompts, reward_gaps = load_holdout_only(CANONICAL_HOLDOUT_DATA_PATH)
    print(f"\nLoaded {len(prompts)} holdout prompts")
    print(f"Unique reward gap values: {sorted(np.unique(reward_gaps))}")
    outcome_summary(reward_gaps)

    # ── Embed prompts ────────────────────────────────────────────────────
    print(f"\nEncoding prompts with {DEFAULT_SENTENCE_TRANSFORMER} ...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True
    )
    print(f"Embedding shape: {embeddings.shape}")

    # ── Load Router PCA ──────────────────────────────────────────────────
    if not DEFAULT_PCA_PATH.exists():
        print(f"\nERROR: Router PCA not found at {DEFAULT_PCA_PATH}")
        print(f"Run: python3 scripts/train_pca_from_routellm.py --n-components 32")
        sys.exit(1)

    router_pca = joblib.load(DEFAULT_PCA_PATH)
    X_pca = router_pca.transform(embeddings)
    pc1 = X_pca[:, 0]
    print(f"\nRouter PCA loaded: {router_pca.n_components_} components")
    print(f"PC1 variance explained: {router_pca.explained_variance_ratio_[0]:.2%}")

    # ══════════════════════════════════════════════════════════════════════
    #  PRIMARY ANALYSIS: SPEARMAN CORRELATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("PRIMARY ANALYSIS: SPEARMAN RANK CORRELATION")
    print("=" * 80)

    rho, p_rho = compute_spearman(pc1, reward_gaps)
    rho_abs = abs(rho)

    p_str = f"p < 0.0001" if p_rho < 0.0001 else f"p = {p_rho:.4f}"
    print(f"\n  Router PCA (PC1 vs reward gap):")
    print(f"    Spearman rho = {rho:.3f}  ({p_str})")
    print(f"    |rho|         = {rho_abs:.3f}")
    print(f"    N             = {len(reward_gaps)}")

    # ── Null baseline: 100 random projections ────────────────────────────
    N_RANDOM = 100
    print(f"\n  Null baseline: {N_RANDOM} random orthonormal projections")
    random_rhos = random_projection_distribution(
        embeddings, reward_gaps, n_seeds=N_RANDOM
    )

    rho_median = np.median(random_rhos)
    rho_p25, rho_p75 = np.percentile(random_rhos, [25, 75])
    rho_max = np.max(random_rhos)
    n_exceed = int(np.sum(random_rhos >= rho_abs))
    signal_ratio = rho_abs / rho_median if rho_median > 0 else float('inf')

    print(f"    |rho| distribution:")
    print(f"      Median: {rho_median:.3f}")
    print(f"      IQR:    [{rho_p25:.3f}, {rho_p75:.3f}]")
    print(f"      Max:    {rho_max:.3f}")
    print(f"    Router PCA |rho| = {rho_abs:.3f}")
    print(f"    Signal ratio (vs median): {signal_ratio:.1f}x")
    print(f"    Random projections >= Router PCA: {n_exceed}/{len(random_rhos)}")
    if n_exceed == 0:
        print(f"    --> Router PCA exceeds ALL {len(random_rhos)} random projections.")

    # ── Supplementary: Mann-Whitney U ────────────────────────────────────
    # Split at median PC1 for a simple group comparison
    median_pc1 = np.median(pc1)
    low_mask = pc1 < median_pc1
    high_mask = pc1 >= median_pc1
    mw_stat, mw_p = mannwhitneyu(
        reward_gaps[low_mask], reward_gaps[high_mask], alternative='two-sided'
    )
    mw_str = "p < 0.0001" if mw_p < 0.0001 else f"p = {mw_p:.4f}"
    print(f"\n  Supplementary: Mann-Whitney U (median split)")
    print(f"    U = {mw_stat:.0f}, {mw_str}")

    # ── Supplementary: Clustering analysis (console only) ────────────────
    supplementary_clustering(pc1, reward_gaps, prompts)

    # ══════════════════════════════════════════════════════════════════════
    #  CREATE FIGURE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CREATING FIGURE")
    print("=" * 80)

    blue, red, grey = '#4575b4', '#d73027', '#888888'
    green = '#2ca02c'

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.2),
        gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.32}
    )

    # ── Panel A: PC1 vs Reward Gap ───────────────────────────────────────
    # Color by outcome for visual clarity
    outcomes = np.array([categorize_gap(g) for g in reward_gaps])
    for outcome, color, marker_label in [
        ('Tie', grey, 'Tie'),
        ('GPT-4T wins', blue, 'GPT-4T wins'),
        ('Mixtral wins', red, 'Mixtral wins'),
    ]:
        mask = outcomes == outcome
        ax1.scatter(
            pc1[mask], reward_gaps[mask],
            c=color, s=18, alpha=0.5, edgecolors='none',
            rasterized=True, zorder=2,
            label=f'{marker_label} ({int(mask.sum())})'
        )

    # Running mean trend line
    rm_x, rm_y = running_mean(pc1, reward_gaps, window=60)
    ax1.plot(rm_x, rm_y, color='black', linewidth=2.5, zorder=4,
             label='Running mean')

    ax1.axhline(y=0, color=grey, linestyle=':', linewidth=1.0,
                alpha=0.6, zorder=1)

    # Annotation: Spearman rho
    rho_label = (
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"{p_str}\n"
        f"N = {len(reward_gaps)}"
    )
    ax1.text(
        0.97, 0.97, rho_label, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#cccccc', alpha=0.95),
        fontweight='bold'
    )

    ax1.text(
        0.98, 0.02,
        'Outcomes are discrete (win=+1, tie=0, loss=\u22121);\n'
        'running mean smooths proportions.',
        transform=ax1.transAxes, fontsize=6.5, color='#666666',
        verticalalignment='bottom', horizontalalignment='right',
        fontstyle='italic',
    )

    ax1.set_xlabel('PC1 (router PCA, trained on RouteLLM battles)',
                    fontsize=11, fontweight='bold')
    ax1.set_ylabel('Reward gap  (GPT-4-Turbo \u2212 Mixtral)',
                    fontsize=11, fontweight='bold')
    ax1.set_title('(A)  Features Predict Model Preference',
                   fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.97,
               edgecolor='#cccccc', fancybox=True, borderpad=0.5,
               bbox_to_anchor=(0.0, 0.78))
    ax1.grid(alpha=0.15, linestyle='--', linewidth=0.5)
    ax1.set_xlim(pc1.min() - 0.03, pc1.max() + 0.03)
    ax1.set_ylim(-1.25, 1.25)

    # ── Panel B: Router PCA rho vs Random Projection Distribution ────────
    ax2.hist(random_rhos, bins=20, color=grey, alpha=0.7,
             edgecolor='white', linewidth=0.8,
             label=f'Random projections (N={N_RANDOM})')

    ax2.axvline(x=rho_abs, color=red, linewidth=3, linestyle='-',
                label=f'Router PCA |$\\rho$| = {rho_abs:.3f}', zorder=5)

    # Mark the median of random
    ax2.axvline(x=rho_median, color=grey, linewidth=1.5, linestyle='--',
                label=f'Random median = {rho_median:.3f}', zorder=4)

    ax2.set_xlabel('|Spearman $\\rho$|  (PC1 vs reward gap)',
                    fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('(B)  Signal vs. Null Baseline',
                   fontsize=13, fontweight='bold', pad=12)

    # Annotation: signal ratio
    n_exceeds_label = len(random_rhos) if n_exceed == 0 else len(random_rhos) - n_exceed
    ratio_text = (
        f"Signal: {signal_ratio:.1f}x median\n"
        f"Exceeds {n_exceeds_label}/{len(random_rhos)} random"
    )
    ax2.text(
        0.03, 0.97, ratio_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0',
                  edgecolor=red, alpha=0.95),
        fontweight='bold'
    )

    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.95,
               edgecolor='#cccccc', fancybox=True)
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

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 1 SUMMARY")
    print("=" * 80)
    print(f"  Data:   Holdout only (N={len(prompts)})")
    print(f"  PCA:    Router PCA ({router_pca.n_components_} components, "
          f"trained on 80K RouteLLM battles)")
    print(f"\n  CLAIM 1: Features predict model preference")
    print(f"    Spearman rho (PC1 vs reward gap) = {rho:.3f}, {p_str}")
    print(f"\n  CLAIM 2: Signal exceeds null baseline")
    print(f"    Router PCA |rho| = {rho_abs:.3f}")
    print(f"    Random projection |rho|: median = {rho_median:.3f}, "
          f"max = {rho_max:.3f}")
    print(f"    Signal ratio: {signal_ratio:.1f}x (vs median)")
    print(f"    Router PCA exceeds {len(random_rhos) - n_exceed}/"
          f"{len(random_rhos)} random projections")
    print(f"\n  INTERPRETATION:")
    print(f"    Model preference varies by prompt — it is not uniform.")
    print(f"    The router's PCA features predict this variation")
    print(f"    ({signal_ratio:.1f}x better than random linear projections).")
    print(f"    This is the necessary condition for contextual routing:")
    print(f"    because features predict reward, the bandit can learn to route.")
    print(f"    Whether it does so effectively is tested in Table 2.")
    print("=" * 80)


if __name__ == "__main__":
    main()
