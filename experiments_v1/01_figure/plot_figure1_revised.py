#!/usr/bin/env python3
"""
Figure 1 (Revised): Model Preference Heterogeneity in Prompt Embeddings

Clean methodology:
  - Routing PCA (domain-adapted feature extraction, unsupervised)
  - Holdout only (N=750, no dev contamination)
  - Unsupervised threshold (silhouette-optimal, no reward peeking)
  - Categorical statistics appropriate for discrete reward data

Panel A: PC1 vs Reward Gap scatter — directly shows the inversion
Panel B: Outcome proportions by cluster — grouped bar chart (win/tie/loss)

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, sem, chi2_contingency
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


def threshold_stability_analysis(X_2d, reward_gaps, primary_threshold):
    """Sweep thresholds and report effect size stability (not just p-values).

    For discrete reward data, report Cramer's V across a range of thresholds
    to show that significance is not an artifact of a particular threshold.
    """
    pc1 = X_2d[:, 0]
    lo, hi = pc1.min() + 0.05, pc1.max() - 0.05
    thresholds = np.linspace(lo, hi, 40)

    print("\n── Threshold Stability Analysis ──")
    print(f"{'Threshold':>10} {'n_high':>7} {'%high':>6} "
          f"{'chi2':>8} {'p_chi2':>10} {'V':>6} {'d':>6}")
    print("-" * 65)

    results = []
    for t in thresholds:
        low_m = pc1 < t
        high_m = pc1 >= t
        g_low = reward_gaps[low_m]
        g_high = reward_gaps[high_m]
        n_h = int(high_m.sum())

        # Skip degenerate splits
        if n_h < 15 or len(g_low) < 15:
            continue

        # Categorical analysis
        def _cat(g):
            if g > 1e-9:
                return 0
            elif g < -1e-9:
                return 2
            return 1

        cats_l = [_cat(g) for g in g_low]
        cats_h = [_cat(g) for g in g_high]
        ct = np.array([[cats_l.count(k) for k in range(3)],
                       [cats_h.count(k) for k in range(3)]])

        # Skip if any expected count < 1 (chi2 unreliable)
        _, _, _, expected = chi2_contingency(ct)
        if expected.min() < 1:
            continue

        chi2_val, p_val, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2_val / (len(reward_gaps) * (min(ct.shape) - 1)))

        # Cohen's d (approximate)
        ps = np.sqrt(((len(g_low) - 1) * np.var(g_low, ddof=1)
                      + (len(g_high) - 1) * np.var(g_high, ddof=1))
                     / (len(g_low) + len(g_high) - 2))
        d = (np.mean(g_low) - np.mean(g_high)) / ps if ps > 0 else 0

        pct_h = n_h / len(reward_gaps) * 100
        marker = "  <-- primary" if abs(t - primary_threshold) < 0.02 else ""
        print(f"{t:10.3f} {n_h:7d} {pct_h:5.1f}% "
              f"{chi2_val:8.1f} {p_val:10.2e} {v:6.3f} {d:6.2f}{marker}")

        results.append({'threshold': t, 'n_high': n_h, 'chi2': chi2_val,
                        'p': p_val, 'V': v, 'd': d})

    if results:
        vs = [r['V'] for r in results]
        ds = [r['d'] for r in results]
        print(f"\nCramer's V range: [{min(vs):.3f}, {max(vs):.3f}]  "
              f"(median {np.median(vs):.3f})")
        print(f"Cohen's d range:  [{min(ds):.2f}, {max(ds):.2f}]  "
              f"(median {np.median(ds):.2f})")
        sig_count = sum(1 for r in results if r['p'] < 0.001)
        print(f"Thresholds with p < 0.001: {sig_count}/{len(results)}")

    return results


def power_analysis_chi2(contingency, alpha=0.05, n_simulations=10_000):
    """Monte-Carlo power analysis for chi-squared test on discrete outcomes.

    Estimates the statistical power of the chi-squared test given the
    observed contingency table proportions and sample sizes.  This is more
    appropriate than parametric power formulae for small, discrete tables.

    Parameters
    ----------
    contingency : ndarray, shape (2, 3)
        Observed contingency table (clusters x outcomes).
    alpha : float
        Significance level.
    n_simulations : int
        Number of Monte-Carlo replications.
    """
    print("\n── Power Analysis (Monte-Carlo, chi-squared) ──")

    n_per_row = contingency.sum(axis=1)  # [n_low, n_high]
    row_props = contingency / contingency.sum(axis=1, keepdims=True)

    # --- Power under the ALTERNATIVE (observed proportions) ---
    reject_alt = 0
    for _ in range(n_simulations):
        # Simulate from the observed per-cluster proportions
        sim_rows = []
        for k in range(contingency.shape[0]):
            sim_rows.append(np.random.multinomial(int(n_per_row[k]),
                                                  row_props[k]))
        sim_ct = np.array(sim_rows)
        # Guard against zero rows/cols
        if sim_ct.min() < 0 or sim_ct.sum() == 0:
            continue
        try:
            _, p, _, exp = chi2_contingency(sim_ct)
            if exp.min() >= 1:
                reject_alt += int(p < alpha)
        except Exception:
            continue
    power = reject_alt / n_simulations

    # --- False-positive rate under the NULL (pooled proportions) ---
    pooled_props = contingency.sum(axis=0) / contingency.sum()
    reject_null = 0
    for _ in range(n_simulations):
        sim_rows = []
        for k in range(contingency.shape[0]):
            sim_rows.append(np.random.multinomial(int(n_per_row[k]),
                                                  pooled_props))
        sim_ct = np.array(sim_rows)
        if sim_ct.min() < 0 or sim_ct.sum() == 0:
            continue
        try:
            _, p, _, exp = chi2_contingency(sim_ct)
            if exp.min() >= 1:
                reject_null += int(p < alpha)
        except Exception:
            continue
    fpr = reject_null / n_simulations

    print(f"  Sample sizes:       n_low={int(n_per_row[0])}, "
          f"n_high={int(n_per_row[1])}")
    print(f"  Observed proportions (Low):  {row_props[0]}")
    print(f"  Observed proportions (High): {row_props[1]}")
    print(f"  Significance level:  alpha = {alpha}")
    print(f"  Simulations:         {n_simulations:,}")
    print(f"  Power (alternative): {power:.3f}  "
          f"({reject_alt}/{n_simulations} rejections)")
    print(f"  FPR (null):          {fpr:.3f}  "
          f"({reject_null}/{n_simulations} rejections)")

    # --- Minimum detectable effect size (Cramer's V) ---
    # Sweep smaller and smaller effect sizes to find the threshold
    print("\n  Detectable effect sizes at 80% power:")
    for target_v in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        # Perturb pooled proportions by target_v to create a synthetic
        # alternative.  For a 2x3 table, shift mass between cells.
        n_total = int(n_per_row.sum())
        # Use the chi2 non-centrality parameter:
        #   V = sqrt(chi2 / (n * min(r-1,c-1)))
        #   => chi2 = V^2 * n * min(r-1,c-1)
        k_min = min(contingency.shape) - 1
        ncp = target_v ** 2 * n_total * k_min
        dof_val = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
        crit = scipy_stats.chi2.ppf(1 - alpha, dof_val)
        power_approx = 1 - scipy_stats.ncx2.cdf(crit, dof_val, ncp)
        marker = " <-- ~80%" if abs(power_approx - 0.80) < 0.05 else ""
        print(f"    V = {target_v:.2f}:  power ≈ {power_approx:.3f}{marker}")

    return power, fpr


def cluster_content_analysis(prompts_low, prompts_high, top_n=15):
    """Systematic TF-IDF keyword analysis of cluster contents.

    Computes terms that are most distinctive to each cluster relative to the
    other, replacing anecdotal 'qualitative inspection' with reproducible
    evidence.
    """
    print("\n── Cluster Content Analysis (TF-IDF) ──")

    all_prompts = prompts_low + prompts_high
    labels = [0] * len(prompts_low) + [1] * len(prompts_high)

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
    )
    X = tfidf.fit_transform(all_prompts)
    feature_names = np.array(tfidf.get_feature_names_out())

    # Mean TF-IDF per cluster
    labels_arr = np.array(labels)
    mean_low = np.asarray(X[labels_arr == 0].mean(axis=0)).ravel()
    mean_high = np.asarray(X[labels_arr == 1].mean(axis=0)).ravel()

    # Distinctive terms: ratio of cluster mean to other cluster mean
    eps = 1e-8
    ratio_high = mean_high / (mean_low + eps)
    ratio_low = mean_low / (mean_high + eps)

    top_high_idx = np.argsort(ratio_high)[::-1][:top_n]
    top_low_idx = np.argsort(ratio_low)[::-1][:top_n]

    print(f"\nTop {top_n} distinctive terms — High PC1 cluster "
          f"(n={len(prompts_high)}):")
    for i, idx in enumerate(top_high_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s}  "
              f"(ratio={ratio_high[idx]:.1f}, "
              f"tf-idf_high={mean_high[idx]:.4f}, "
              f"tf-idf_low={mean_low[idx]:.4f})")

    print(f"\nTop {top_n} distinctive terms — Low PC1 cluster "
          f"(n={len(prompts_low)}):")
    for i, idx in enumerate(top_low_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s}  "
              f"(ratio={ratio_low[idx]:.1f}, "
              f"tf-idf_low={mean_low[idx]:.4f}, "
              f"tf-idf_high={mean_high[idx]:.4f})")

    # Average prompt length per cluster
    len_low = np.mean([len(p.split()) for p in prompts_low])
    len_high = np.mean([len(p.split()) for p in prompts_high])
    print(f"\nMean prompt length — Low PC1: {len_low:.1f} words, "
          f"High PC1: {len_high:.1f} words")


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
    print(f"  (Note: Cohen's d is approximate — reward gaps are discrete)")

    # ── Categorical analysis (appropriate for discrete reward data) ───────
    unique_vals = sorted(np.unique(reward_gaps))
    print(f"\nUnique reward gap values: {unique_vals}")

    def categorize_gap(g, eps=1e-9):
        if g > eps:
            return 'GPT-4T wins'
        elif g < -eps:
            return 'Mixtral wins'
        else:
            return 'Tie'

    cats_low = [categorize_gap(g) for g in gaps_low]
    cats_high = [categorize_gap(g) for g in gaps_high]
    outcome_order = ['GPT-4T wins', 'Tie', 'Mixtral wins']

    counts_low = np.array([cats_low.count(c) for c in outcome_order])
    counts_high = np.array([cats_high.count(c) for c in outcome_order])
    props_low = counts_low / counts_low.sum() * 100
    props_high = counts_high / counts_high.sum() * 100

    contingency = np.array([counts_low, counts_high])
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (len(reward_gaps) * (min(contingency.shape) - 1)))

    print(f"\nContingency Table (cluster x outcome):")
    print(f"{'':>15} {'GPT-4T wins':>12} {'Tie':>8} {'Mixtral wins':>14}")
    print(f"{'Low PC1':>15} {counts_low[0]:>12} {counts_low[1]:>8} {counts_low[2]:>14}")
    print(f"{'High PC1':>15} {counts_high[0]:>12} {counts_high[1]:>8} {counts_high[2]:>14}")
    print(f"\nProportions:")
    print(f"  Low PC1:  GPT-4T wins {props_low[0]:.1f}%, "
          f"Tie {props_low[1]:.1f}%, Mixtral wins {props_low[2]:.1f}%")
    print(f"  High PC1: GPT-4T wins {props_high[0]:.1f}%, "
          f"Tie {props_high[1]:.1f}%, Mixtral wins {props_high[2]:.1f}%")
    print(f"\nChi-squared: chi2 = {chi2:.1f}, df = {dof}, "
          f"p = {p_chi2:.2e}")
    print(f"Cramer's V = {cramers_v:.3f}")

    # ── Power analysis ────────────────────────────────────────────────────
    power_analysis_chi2(contingency)

    # ── Threshold stability (effect size across thresholds) ───────────────
    threshold_stability_analysis(X_2d, reward_gaps, threshold)

    # ── Systematic content analysis (TF-IDF) ─────────────────────────────
    prompts_arr = np.array(prompts)
    cluster_content_analysis(
        list(prompts_arr[low_mask]),
        list(prompts_arr[high_mask]),
    )

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
        label=f'Low PC1 ({pct_low:.0f}%)'
    )
    ax1.scatter(
        pc1[high_mask], reward_gaps[high_mask],
        c=red, s=18, alpha=0.45, edgecolors='none',
        rasterized=True, zorder=2,
        label=f'High PC1 ({pct_high:.0f}%)'
    )

    # Running-mean trend (smooths discrete outcomes for visual clarity)
    rm_x, rm_y = running_mean(pc1, reward_gaps, window=60)
    ax1.plot(rm_x, rm_y, color='black', linewidth=2.5, zorder=4,
             label='Running mean (w=60)')

    # Annotate that underlying data is discrete
    ax1.text(
        0.98, 0.02,
        'Note: reward gaps are discrete\n(win=+1, tie=0, loss=−1);\nrunning mean smooths proportions.',
        transform=ax1.transAxes, fontsize=6.5, color='#666666',
        verticalalignment='bottom', horizontalalignment='right',
        fontstyle='italic',
    )

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

    # ── Panel B: Outcome proportions by cluster (grouped bar chart) ─────
    bar_width = 0.32
    x_pos = np.arange(len(outcome_order))

    # Cluster bars side by side
    bars_low = ax2.bar(
        x_pos - bar_width / 2, props_low, bar_width,
        label=f'Low PC1 (n={n_low})', color=blue, alpha=0.75,
        edgecolor='black', linewidth=0.8
    )
    bars_high = ax2.bar(
        x_pos + bar_width / 2, props_high, bar_width,
        label=f'High PC1 (n={n_high})', color=red, alpha=0.75,
        edgecolor='black', linewidth=0.8
    )

    # Value labels on bars
    for bars in [bars_low, bars_high]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax2.text(bar.get_x() + bar.get_width() / 2., h + 1.2,
                         f'{h:.0f}%', ha='center', va='bottom', fontsize=8,
                         fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        ['GPT-4T\nwins', 'Tie', 'Mixtral\nwins'],
        fontsize=10, fontweight='bold'
    )
    ax2.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B)  Outcome Proportions by PC1 Region', fontsize=13,
                   fontweight='bold', pad=12)
    ax2.set_ylim(0, max(props_low.max(), props_high.max()) + 15)
    ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
               edgecolor='#cccccc', fancybox=True)
    ax2.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Statistical annotation box — include both PCA effect sizes
    p_str_chi2 = 'p < 0.0001' if p_chi2 < 0.0001 else f'p = {p_chi2:.4f}'
    p_str_mw = 'p < 0.0001' if p_value < 0.0001 else f'p = {p_value:.4f}'
    stat_text = (
        f"$\\chi^2$ = {chi2:.1f}, {p_str_chi2}\n"
        f"Cram\u00e9r's V = {cramers_v:.2f}\n"
        f"Cohen's d = 0.33 (generic PCA)\n"
        f"Cohen's d = {cohens_d:.2f} (domain PCA)\n"
        f'N = {len(reward_gaps)}'
    )
    ax2.text(
        0.97, 0.97, stat_text, transform=ax2.transAxes,
        fontsize=7.5, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#cccccc', alpha=0.95)
    )

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
    print(f"  Low  PC1:   {n_low} prompts ({pct_low:.1f}%)")
    print(f"    GPT-4T wins: {props_low[0]:.1f}%, "
          f"Tie: {props_low[1]:.1f}%, "
          f"Mixtral wins: {props_low[2]:.1f}%")
    print(f"    Mean gap = {np.mean(gaps_low):+.3f}")
    print(f"  High PC1:   {n_high} prompts ({pct_high:.1f}%)")
    print(f"    GPT-4T wins: {props_high[0]:.1f}%, "
          f"Tie: {props_high[1]:.1f}%, "
          f"Mixtral wins: {props_high[2]:.1f}%")
    print(f"    Mean gap = {np.mean(gaps_high):+.3f}")
    print(f"  Chi-squared: chi2={chi2:.1f}, {p_str_chi2}, "
          f"Cramer's V={cramers_v:.2f}")
    print(f"  Mann-Whitney: {p_str_mw}")
    print(f"  Primary effect size (unbiased):  Generic C4 PCA "
          f"Cohen's d = 0.33 (small, p<0.0001)")
    print(f"  Amplified effect size:           Domain-adapted PCA "
          f"Cohen's d = {cohens_d:.2f} (approx; data is discrete)")
    print(f"  NOTE: The generic PCA result (d=0.33) is the conservative,")
    print(f"        unbiased baseline. Domain-adapted PCA amplifies via")
    print(f"        task-relevant feature extraction (4.6x).")
    print(f"  Cluster proportion: {pct_high:.1f}% in holdout; "
          f"may differ in production (cf. 1M dataset: ~5.9%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
