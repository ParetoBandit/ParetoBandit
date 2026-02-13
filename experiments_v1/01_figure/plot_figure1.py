#!/usr/bin/env python3
"""
Figure 1: Routing Signal Validation — Model Preference Heterogeneity

Tests whether the router's PCA feature extraction captures genuine model
preference signal, compared against baselines.

Methodology:
  - Uses the SAME feature extraction pipeline as router.py (FeatureService)
  - Holdout only (N=750, no dev contamination)
  - PCA trained on independent dataset (80K RouteLLM battles, separate from holdout)
  - Unsupervised threshold (silhouette-optimal, no reward peeking)
  - Categorical statistics for discrete win/tie/loss outcomes
  - Three-condition comparison:
      1. Router PCA (domain-adapted, trained on RouteLLM battles)
      2. Generic PCA (C4 web text baseline)
      3. Random projection (null baseline)
  - Permutation test for distribution-free p-values

Panel A: PC1 vs Reward Gap scatter (primary PCA condition)
Panel B: Outcome proportions by cluster (grouped bar chart)

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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu, sem, chi2_contingency, fisher_exact
from scipy import stats as scipy_stats
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    GENERIC_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
)

# Outcome categories (order matters for contingency table)
OUTCOME_ORDER = ['GPT-4T wins', 'Tie', 'Mixtral wins']


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
#  CATEGORICAL EFFECT SIZES (appropriate for discrete outcomes)
# ══════════════════════════════════════════════════════════════════════════

def categorize_gap(g, eps=1e-9):
    """Map continuous reward gap to discrete outcome."""
    if g > eps:
        return 'GPT-4T wins'
    elif g < -eps:
        return 'Mixtral wins'
    else:
        return 'Tie'


def build_contingency(gaps_low, gaps_high):
    """Build 2x3 contingency table from reward gaps."""
    cats_low = [categorize_gap(g) for g in gaps_low]
    cats_high = [categorize_gap(g) for g in gaps_high]
    counts_low = np.array([cats_low.count(c) for c in OUTCOME_ORDER])
    counts_high = np.array([cats_high.count(c) for c in OUTCOME_ORDER])
    return np.array([counts_low, counts_high])


def compute_categorical_effects(contingency):
    """Compute all categorical effect sizes for a 2x3 contingency table.

    Returns dict with:
      - chi2, p_chi2, dof: chi-squared test
      - cramers_v: primary categorical effect size
      - odds_ratio_mixtral: odds of Mixtral winning in High vs Low PC1
      - risk_diff_mixtral: difference in Mixtral win probability (High - Low)
      - risk_diff_gpt4t: difference in GPT-4T win probability (High - Low)
    """
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    n = contingency.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k))

    # Proportions per cluster
    props_low = contingency[0] / contingency[0].sum()
    props_high = contingency[1] / contingency[1].sum()

    # Risk difference: P(Mixtral wins | High) - P(Mixtral wins | Low)
    # Mixtral wins is index 2 in OUTCOME_ORDER
    risk_diff_mixtral = props_high[2] - props_low[2]
    risk_diff_gpt4t = props_high[0] - props_low[0]

    # Odds ratio for Mixtral wins (High vs Low)
    # OR = [P(Mixtral|High)/P(not Mixtral|High)] / [P(Mixtral|Low)/P(not Mixtral|Low)]
    a = contingency[1, 2]  # Mixtral wins in High
    b = contingency[1, 0] + contingency[1, 1]  # not Mixtral wins in High
    c = contingency[0, 2]  # Mixtral wins in Low
    d = contingency[0, 0] + contingency[0, 1]  # not Mixtral wins in Low
    # Add 0.5 continuity correction to avoid division by zero
    odds_ratio_mixtral = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

    return {
        'chi2': chi2, 'p_chi2': p_chi2, 'dof': dof,
        'cramers_v': cramers_v,
        'odds_ratio_mixtral': odds_ratio_mixtral,
        'risk_diff_mixtral': risk_diff_mixtral,
        'risk_diff_gpt4t': risk_diff_gpt4t,
        'props_low': props_low * 100,
        'props_high': props_high * 100,
        'contingency': contingency,
    }


# ══════════════════════════════════════════════════════════════════════════
#  UNSUPERVISED THRESHOLD & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
#  PERMUTATION TEST (distribution-free significance)
# ══════════════════════════════════════════════════════════════════════════

def permutation_test_cramers_v(
    reward_gaps, cluster_labels, n_permutations=10_000, seed=42
):
    """Distribution-free test: is observed Cramer's V larger than chance?

    Permutes reward gap labels (breaking any real association) and computes
    Cramer's V on each permuted sample. The p-value is the fraction of
    *valid* permutations with V >= observed V.

    Returns:
        v_obs: Observed Cramer's V
        p_perm: Permutation p-value (count_ge / n_valid)
        n_permutations: Total permutations attempted
        n_valid: Number of valid permutations (expected counts >= 1)
        n_skipped: Number of skipped permutations
    """
    rng = np.random.RandomState(seed)
    low_mask = cluster_labels == 0
    high_mask = cluster_labels == 1

    # Observed
    ct_obs = build_contingency(reward_gaps[low_mask], reward_gaps[high_mask])
    chi2_obs, _, _, _ = chi2_contingency(ct_obs)
    n = ct_obs.sum()
    k = min(ct_obs.shape) - 1
    v_obs = np.sqrt(chi2_obs / (n * k))

    # Permutation distribution
    count_ge = 0
    n_valid = 0
    n_skipped = 0
    for _ in range(n_permutations):
        perm = rng.permutation(reward_gaps)
        ct_perm = build_contingency(perm[low_mask], perm[high_mask])
        try:
            chi2_p, _, _, exp = chi2_contingency(ct_perm)
            if exp.min() < 1:
                n_skipped += 1
                continue
            n_valid += 1
            v_perm = np.sqrt(chi2_p / (n * k))
            if v_perm >= v_obs:
                count_ge += 1
        except Exception:
            n_skipped += 1
            continue

    p_perm = count_ge / n_valid if n_valid > 0 else 1.0
    return v_obs, p_perm, n_permutations, n_valid, n_skipped


# ══════════════════════════════════════════════════════════════════════════
#  PCA CONDITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def analyze_pca_condition(
    embeddings, reward_gaps, pca_or_matrix, condition_name, is_random=False
):
    """Run the full analysis for one PCA condition.

    Args:
        embeddings: (N, 384) raw embeddings
        reward_gaps: (N,) reward gap array
        pca_or_matrix: sklearn PCA object or (384, 2) projection matrix
        condition_name: label for printing
        is_random: if True, pca_or_matrix is a raw matrix, not sklearn PCA

    Returns:
        dict with all results for this condition
    """
    print(f"\n{'─' * 60}")
    print(f"  Condition: {condition_name}")
    print(f"{'─' * 60}")

    # Project
    if is_random:
        X_2d = embeddings @ pca_or_matrix
    else:
        X_2d = pca_or_matrix.transform(embeddings)[:, :2]

    pc1 = X_2d[:, 0]

    # Unsupervised threshold
    threshold, sil_score = find_silhouette_optimal_threshold(X_2d)
    if threshold is None:
        print(f"  No valid threshold found — skipping condition")
        return None
    print(f"  Silhouette-optimal threshold: {threshold:.3f} (sil={sil_score:.3f})")

    low_mask = pc1 < threshold
    high_mask = pc1 >= threshold
    gaps_low = reward_gaps[low_mask]
    gaps_high = reward_gaps[high_mask]
    n_low, n_high = int(low_mask.sum()), int(high_mask.sum())

    # Contingency table and categorical effects
    contingency = build_contingency(gaps_low, gaps_high)
    effects = compute_categorical_effects(contingency)

    # Mann-Whitney U (supplementary ordinal test)
    mw_stat, mw_p = mannwhitneyu(gaps_low, gaps_high, alternative='two-sided')

    # Cohen's d (approximate — data is discrete, included for comparability only)
    pooled_std = np.sqrt(
        ((len(gaps_low) - 1) * np.var(gaps_low, ddof=1)
         + (len(gaps_high) - 1) * np.var(gaps_high, ddof=1))
        / (len(gaps_low) + len(gaps_high) - 2)
    )
    cohens_d = (np.mean(gaps_low) - np.mean(gaps_high)) / pooled_std if pooled_std > 0 else 0

    # Permutation test (distribution-free)
    cluster_labels = np.where(low_mask, 0, 1)
    v_obs, p_perm, n_perm, n_valid, n_skipped = permutation_test_cramers_v(
        reward_gaps, cluster_labels
    )

    # Print results
    print(f"\n  Cluster sizes: Low={n_low} ({n_low/len(reward_gaps)*100:.1f}%), "
          f"High={n_high} ({n_high/len(reward_gaps)*100:.1f}%)")
    print(f"  Contingency table:")
    print(f"    {'':>12} {'GPT-4T win':>11} {'Tie':>6} {'Mixtral win':>12}")
    print(f"    {'Low PC1':>12} {contingency[0,0]:>11} {contingency[0,1]:>6} {contingency[0,2]:>12}")
    print(f"    {'High PC1':>12} {contingency[1,0]:>11} {contingency[1,1]:>6} {contingency[1,2]:>12}")
    print(f"\n  PRIMARY EFFECT SIZES (categorical):")
    print(f"    Cramer's V = {effects['cramers_v']:.3f}")
    print(f"    Odds ratio (Mixtral win, High vs Low) = {effects['odds_ratio_mixtral']:.1f}")
    print(f"    Risk diff (Mixtral win) = {effects['risk_diff_mixtral']:+.1%}")
    print(f"    Risk diff (GPT-4T win)  = {effects['risk_diff_gpt4t']:+.1%}")
    print(f"\n  SIGNIFICANCE:")
    p_str = 'p < 0.0001' if effects['p_chi2'] < 0.0001 else f"p = {effects['p_chi2']:.4f}"
    print(f"    Chi-squared: chi2={effects['chi2']:.1f}, {p_str}")
    mw_str = 'p < 0.0001' if mw_p < 0.0001 else f'p = {mw_p:.4f}'
    print(f"    Mann-Whitney U: {mw_str}")
    perm_str = 'p < 0.0001' if p_perm < 0.0001 else f'p = {p_perm:.4f}'
    skip_pct = 100 * n_skipped / n_perm if n_perm > 0 else 0
    print(f"    Permutation test (n={n_perm:,}, valid={n_valid:,}, "
          f"skipped={n_skipped:,} [{skip_pct:.1f}%]): {perm_str}")
    print(f"\n  SUPPLEMENTARY (approximate — data is discrete {-1, 0, +1}):")
    print(f"    Cohen's d = {cohens_d:.2f}")

    return {
        'name': condition_name,
        'X_2d': X_2d, 'pc1': pc1,
        'threshold': threshold, 'sil_score': sil_score,
        'low_mask': low_mask, 'high_mask': high_mask,
        'n_low': n_low, 'n_high': n_high,
        'effects': effects,
        'mw_p': mw_p, 'cohens_d': cohens_d,
        'p_perm': p_perm,
        'perm_n_valid': n_valid,
        'perm_n_skipped': n_skipped,
    }


# ══════════════════════════════════════════════════════════════════════════
#  THRESHOLD STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def threshold_stability_analysis(X_2d, reward_gaps, primary_threshold):
    """Sweep thresholds and report Cramer's V stability (not just p-values)."""
    pc1 = X_2d[:, 0]
    lo, hi = pc1.min() + 0.05, pc1.max() - 0.05
    thresholds = np.linspace(lo, hi, 40)

    print("\n── Threshold Stability Analysis ──")
    print(f"{'Threshold':>10} {'n_high':>7} {'%high':>6} "
          f"{'chi2':>8} {'p_chi2':>10} {'V':>6}")
    print("-" * 55)

    results = []
    for t in thresholds:
        low_m = pc1 < t
        high_m = pc1 >= t
        n_h = int(high_m.sum())
        if n_h < 15 or int(low_m.sum()) < 15:
            continue

        ct = build_contingency(reward_gaps[low_m], reward_gaps[high_m])
        _, _, _, expected = chi2_contingency(ct)
        if expected.min() < 1:
            continue

        chi2_val, p_val, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2_val / (len(reward_gaps) * (min(ct.shape) - 1)))
        pct_h = n_h / len(reward_gaps) * 100
        marker = "  <-- primary" if abs(t - primary_threshold) < 0.02 else ""
        print(f"{t:10.3f} {n_h:7d} {pct_h:5.1f}% "
              f"{chi2_val:8.1f} {p_val:10.2e} {v:6.3f}{marker}")
        results.append({'threshold': t, 'n_high': n_h, 'chi2': chi2_val,
                        'p': p_val, 'V': v})

    if results:
        vs = [r['V'] for r in results]
        print(f"\nCramer's V range: [{min(vs):.3f}, {max(vs):.3f}]  "
              f"(median {np.median(vs):.3f})")
        sig_count = sum(1 for r in results if r['p'] < 0.001)
        print(f"Thresholds with p < 0.001: {sig_count}/{len(results)}")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  POWER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def power_analysis_chi2(contingency, alpha=0.05, n_simulations=10_000):
    """Monte-Carlo power analysis for chi-squared test."""
    print("\n── Power Analysis (Monte-Carlo, chi-squared) ──")

    n_per_row = contingency.sum(axis=1)
    row_props = contingency / contingency.sum(axis=1, keepdims=True)

    reject_alt = 0
    for _ in range(n_simulations):
        sim_rows = [np.random.multinomial(int(n_per_row[k]), row_props[k])
                     for k in range(contingency.shape[0])]
        sim_ct = np.array(sim_rows)
        if sim_ct.min() < 0 or sim_ct.sum() == 0:
            continue
        try:
            _, p, _, exp = chi2_contingency(sim_ct)
            if exp.min() >= 1:
                reject_alt += int(p < alpha)
        except Exception:
            continue
    power = reject_alt / n_simulations

    pooled_props = contingency.sum(axis=0) / contingency.sum()
    reject_null = 0
    for _ in range(n_simulations):
        sim_rows = [np.random.multinomial(int(n_per_row[k]), pooled_props)
                     for k in range(contingency.shape[0])]
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

    print(f"  Sample sizes:       n_low={int(n_per_row[0])}, n_high={int(n_per_row[1])}")
    print(f"  Observed proportions (Low):  {row_props[0]}")
    print(f"  Observed proportions (High): {row_props[1]}")
    print(f"  Power (alternative): {power:.3f}  ({reject_alt}/{n_simulations})")
    print(f"  FPR (null):          {fpr:.3f}  ({reject_null}/{n_simulations})")
    return power, fpr


# ══════════════════════════════════════════════════════════════════════════
#  CLUSTER CONTENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def cluster_content_analysis(prompts_low, prompts_high, top_n=15):
    """TF-IDF keyword analysis of cluster contents."""
    print("\n── Cluster Content Analysis (TF-IDF) ──")
    all_prompts = prompts_low + prompts_high
    labels = [0] * len(prompts_low) + [1] * len(prompts_high)

    tfidf = TfidfVectorizer(
        max_features=5000, stop_words='english',
        ngram_range=(1, 2), min_df=3, max_df=0.8,
    )
    X = tfidf.fit_transform(all_prompts)
    feature_names = np.array(tfidf.get_feature_names_out())
    labels_arr = np.array(labels)
    mean_low = np.asarray(X[labels_arr == 0].mean(axis=0)).ravel()
    mean_high = np.asarray(X[labels_arr == 1].mean(axis=0)).ravel()
    eps = 1e-8
    ratio_high = mean_high / (mean_low + eps)
    ratio_low = mean_low / (mean_high + eps)
    top_high_idx = np.argsort(ratio_high)[::-1][:top_n]
    top_low_idx = np.argsort(ratio_low)[::-1][:top_n]

    print(f"\nTop {top_n} distinctive terms -- High PC1 (n={len(prompts_high)}):")
    for i, idx in enumerate(top_high_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s}  "
              f"(ratio={ratio_high[idx]:.1f}, "
              f"high={mean_high[idx]:.4f}, low={mean_low[idx]:.4f})")

    print(f"\nTop {top_n} distinctive terms -- Low PC1 (n={len(prompts_low)}):")
    for i, idx in enumerate(top_low_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s}  "
              f"(ratio={ratio_low[idx]:.1f}, "
              f"low={mean_low[idx]:.4f}, high={mean_high[idx]:.4f})")

    len_low = np.mean([len(p.split()) for p in prompts_low])
    len_high = np.mean([len(p.split()) for p in prompts_high])
    print(f"\nMean prompt length -- Low: {len_low:.1f} words, High: {len_high:.1f} words")


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
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
    print("FIGURE 1: ROUTING SIGNAL VALIDATION")
    print("=" * 80)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load holdout data ─────────────────────────────────────────────────
    prompts, reward_gaps = load_holdout_only(CANONICAL_HOLDOUT_DATA_PATH)
    print(f"Loaded {len(prompts)} holdout prompts")
    print(f"Unique reward gap values: {sorted(np.unique(reward_gaps))}")

    # ── Embed prompts (once, reused across conditions) ────────────────────
    print(f"\nEncoding prompts with {DEFAULT_SENTENCE_TRANSFORMER} ...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        prompts, normalize_embeddings=True, show_progress_bar=True,
        batch_size=64, convert_to_numpy=True
    )
    print(f"Embedding shape: {embeddings.shape}")

    # ── Load PCA artifact ────────────────────────────────────────────────
    # The router PCA is trained on 80K RouteLLM battles — an independent
    # dataset from the holdout (LMSYS general prompts), so there is no
    # contamination concern.
    if not DEFAULT_PCA_PATH.exists():
        print(f"\nERROR: Router PCA not found at {DEFAULT_PCA_PATH}")
        print(f"Run: python3 scripts/train_pca_from_routellm.py --n-components 32")
        sys.exit(1)
    router_pca_path = DEFAULT_PCA_PATH
    router_pca_label = "Router PCA (domain-adapted)"

    # ── Run all conditions ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MULTI-CONDITION ANALYSIS")
    print("=" * 80)

    results = {}

    # Condition 1: Router PCA (the PCA used by router.py / FeatureService)
    router_pca = joblib.load(router_pca_path)
    results['router'] = analyze_pca_condition(
        embeddings, reward_gaps, router_pca, router_pca_label
    )

    # Condition 2: Generic PCA (C4 web text — no routing connection)
    if GENERIC_PCA_PATH.exists():
        generic_pca = joblib.load(GENERIC_PCA_PATH)
        results['generic'] = analyze_pca_condition(
            embeddings, reward_gaps, generic_pca, "Generic PCA (C4 web text)"
        )
    else:
        print(f"\n  [SKIP] Generic PCA not found at {GENERIC_PCA_PATH}")
        print(f"  Run: python3 scripts/train_pca_generic.py --n-components 32")
        results['generic'] = None

    # Condition 3: Random projection (null baseline — single seed for figure)
    rng = np.random.RandomState(42)
    random_matrix = rng.randn(embeddings.shape[1], 2)
    random_matrix /= np.linalg.norm(random_matrix, axis=0, keepdims=True)
    results['random'] = analyze_pca_condition(
        embeddings, reward_gaps, random_matrix,
        "Random projection (null baseline)", is_random=True
    )

    # ── Multi-seed random projection distribution ─────────────────────────
    # Run 100 random projections to characterize V_random distribution,
    # so the signal ratio is against a robust denominator (not a single seed).
    N_RANDOM_SEEDS = 100
    print(f"\n{'─' * 60}")
    print(f"  Random Projection Distribution (N={N_RANDOM_SEEDS} seeds)")
    print(f"{'─' * 60}")

    random_vs = []
    for seed in range(N_RANDOM_SEEDS):
        rng_i = np.random.RandomState(seed)
        mat_i = rng_i.randn(embeddings.shape[1], 2)
        mat_i /= np.linalg.norm(mat_i, axis=0, keepdims=True)
        X_2d_i = embeddings @ mat_i
        thr_i, _ = find_silhouette_optimal_threshold(X_2d_i)
        if thr_i is None:
            continue
        pc1_i = X_2d_i[:, 0]
        low_i = pc1_i < thr_i
        high_i = pc1_i >= thr_i
        if low_i.sum() < 15 or high_i.sum() < 15:
            continue
        ct_i = build_contingency(reward_gaps[low_i], reward_gaps[high_i])
        try:
            chi2_i, _, _, exp_i = chi2_contingency(ct_i)
            if exp_i.min() < 1:
                continue
            v_i = np.sqrt(chi2_i / (ct_i.sum() * (min(ct_i.shape) - 1)))
            random_vs.append(v_i)
        except Exception:
            continue

    random_vs = np.array(random_vs)
    v_median = np.median(random_vs)
    v_mean = np.mean(random_vs)
    v_p25, v_p75 = np.percentile(random_vs, [25, 75])
    v_max = np.max(random_vs)

    router_v = results['router']['effects']['cramers_v']
    ratio_median = router_v / v_median if v_median > 0 else float('inf')
    ratio_max = router_v / v_max if v_max > 0 else float('inf')
    # How many random projections exceed router PCA?
    n_exceed = int(np.sum(random_vs >= router_v))

    print(f"  Valid projections: {len(random_vs)}/{N_RANDOM_SEEDS}")
    print(f"  V_random distribution:")
    print(f"    Median: {v_median:.3f}  Mean: {v_mean:.3f}")
    print(f"    IQR:    [{v_p25:.3f}, {v_p75:.3f}]")
    print(f"    Max:    {v_max:.3f}")
    print(f"  Router PCA V = {router_v:.3f}")
    print(f"  Signal ratio (vs median): {ratio_median:.1f}x")
    print(f"  Signal ratio (vs max):    {ratio_max:.1f}x")
    print(f"  Random projections >= router PCA: {n_exceed}/{len(random_vs)}")
    results['random_distribution'] = {
        'n_seeds': N_RANDOM_SEEDS,
        'n_valid': len(random_vs),
        'vs': random_vs,
        'median': float(v_median),
        'mean': float(v_mean),
        'iqr': (float(v_p25), float(v_p75)),
        'max': float(v_max),
        'ratio_vs_median': float(ratio_median),
        'ratio_vs_max': float(ratio_max),
        'n_exceed_router': n_exceed,
    }

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITION COMPARISON")
    print("=" * 80)
    print(f"{'Condition':<45} {'V':>6} {'OR':>6} {'RD':>7} {'perm-p':>8}")
    print("-" * 75)
    for key in ['router', 'generic', 'random']:
        r = results.get(key)
        if r is None:
            continue
        e = r['effects']
        p_str = '<.0001' if r['p_perm'] < 0.0001 else f"{r['p_perm']:.4f}"
        print(f"  {r['name']:<43} {e['cramers_v']:6.3f} "
              f"{e['odds_ratio_mixtral']:6.1f} "
              f"{e['risk_diff_mixtral']:+6.1%} {p_str:>8}")
    print(f"  {'Random projection (median of 100 seeds)':<43} {v_median:6.3f} "
          f"{'—':>6} {'—':>7} {'—':>8}")

    # ── Use router PCA condition for figure ────────────────────────────────
    primary = results['router']
    if primary is None:
        print("ERROR: Router PCA analysis failed, cannot generate figure.")
        sys.exit(1)

    eff = primary['effects']

    # ── Additional validation on primary condition ────────────────────────
    threshold_stability_analysis(primary['X_2d'], reward_gaps, primary['threshold'])
    power_analysis_chi2(eff['contingency'])

    prompts_arr = np.array(prompts)
    cluster_content_analysis(
        list(prompts_arr[primary['low_mask']]),
        list(prompts_arr[primary['high_mask']]),
    )

    # ══════════════════════════════════════════════════════════════════════
    #  CREATE FIGURE
    # ══════════════════════════════════════════════════════════════════════
    blue, red, grey = '#4575b4', '#d73027', '#888888'
    pc1 = primary['pc1']
    low_mask = primary['low_mask']
    high_mask = primary['high_mask']
    threshold = primary['threshold']
    n_low, n_high = primary['n_low'], primary['n_high']
    pct_low = n_low / len(reward_gaps) * 100
    pct_high = n_high / len(reward_gaps) * 100

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
    rm_x, rm_y = running_mean(pc1, reward_gaps, window=60)
    ax1.plot(rm_x, rm_y, color='black', linewidth=2.5, zorder=4,
             label='Running mean (w=60)')
    ax1.text(
        0.98, 0.02,
        'Note: outcomes are discrete\n(win=+1, tie=0, loss=\u22121);\n'
        'running mean smooths proportions.',
        transform=ax1.transAxes, fontsize=6.5, color='#666666',
        verticalalignment='bottom', horizontalalignment='right',
        fontstyle='italic',
    )
    ax1.axvline(x=threshold, color=grey, linestyle='--', linewidth=1.8,
                zorder=3, label=f'Threshold ({threshold:.2f})')
    ax1.axhline(y=0, color=grey, linestyle=':', linewidth=1.0, alpha=0.6, zorder=1)

    ax1.set_xlabel('PC1 (domain-adapted PCA)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward gap  (GPT-4-Turbo \u2212 Mixtral)', fontsize=12,
                    fontweight='bold')
    ax1.set_title('(A)  Reward Gap vs. Embedding PC1', fontsize=13,
                   fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.97,
               edgecolor='#cccccc', fancybox=True, borderpad=0.5,
               bbox_to_anchor=(0.0, 0.82))
    ax1.grid(alpha=0.15, linestyle='--', linewidth=0.5)
    ax1.set_xlim(pc1.min() - 0.03, pc1.max() + 0.03)
    ax1.set_ylim(-1.25, 1.25)

    # ── Panel B: Outcome proportions by cluster (grouped bar chart) ──────
    bar_width = 0.32
    x_pos = np.arange(len(OUTCOME_ORDER))
    bars_low = ax2.bar(
        x_pos - bar_width / 2, eff['props_low'], bar_width,
        label=f'Low PC1 (n={n_low})', color=blue, alpha=0.75,
        edgecolor='black', linewidth=0.8
    )
    bars_high = ax2.bar(
        x_pos + bar_width / 2, eff['props_high'], bar_width,
        label=f'High PC1 (n={n_high})', color=red, alpha=0.75,
        edgecolor='black', linewidth=0.8
    )
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
    ax2.set_ylim(0, max(eff['props_low'].max(), eff['props_high'].max()) + 15)
    ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
               edgecolor='#cccccc', fancybox=True)
    ax2.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Statistical annotation — primary categorical metrics
    p_str = 'p < 0.0001' if eff['p_chi2'] < 0.0001 else f"p = {eff['p_chi2']:.4f}"
    perm_str = 'p < 0.0001' if primary['p_perm'] < 0.0001 else f"p = {primary['p_perm']:.4f}"
    stat_text = (
        f"Cram\u00e9r's V = {eff['cramers_v']:.2f}\n"
        f"OR(Mixtral win) = {eff['odds_ratio_mixtral']:.1f}\n"
        f"Risk diff = {eff['risk_diff_mixtral']:+.0%}\n"
        f"$\\chi^2$ {p_str}\n"
        f"Permutation {perm_str}\n"
        f"N = {len(reward_gaps)}"
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

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 1 SUMMARY")
    print("=" * 80)
    print(f"  Data:       Holdout only (N={len(prompts)})")
    print(f"  PCA:        {router_pca_label}")
    print(f"  Threshold:  {threshold:.3f} (silhouette-optimal, unsupervised)")
    print(f"  Low  PC1:   {n_low} prompts ({pct_low:.1f}%)")
    print(f"    GPT-4T wins: {eff['props_low'][0]:.1f}%, "
          f"Tie: {eff['props_low'][1]:.1f}%, "
          f"Mixtral wins: {eff['props_low'][2]:.1f}%")
    print(f"  High PC1:   {n_high} prompts ({pct_high:.1f}%)")
    print(f"    GPT-4T wins: {eff['props_high'][0]:.1f}%, "
          f"Tie: {eff['props_high'][1]:.1f}%, "
          f"Mixtral wins: {eff['props_high'][2]:.1f}%")
    print(f"\n  PRIMARY EFFECT SIZES (categorical):")
    print(f"    Cramer's V = {eff['cramers_v']:.3f}")
    print(f"    OR(Mixtral win, High vs Low) = {eff['odds_ratio_mixtral']:.1f}")
    print(f"    Risk diff (Mixtral win) = {eff['risk_diff_mixtral']:+.1%}")
    print(f"\n  SIGNIFICANCE:")
    print(f"    Chi-squared: {p_str}")
    print(f"    Permutation test: {perm_str}")
    mw_str = 'p < 0.0001' if primary['mw_p'] < 0.0001 else f"p = {primary['mw_p']:.4f}"
    print(f"    Mann-Whitney U: {mw_str}")
    print(f"\n  SUPPLEMENTARY (approximate, discrete data):")
    print(f"    Cohen's d = {primary['cohens_d']:.2f}")
    print(f"\n  INTERPRETATION:")
    print(f"    A minority of prompts (~{pct_high:.0f}% in holdout) show strong")
    print(f"    Mixtral preference. The majority (~{pct_low:.0f}%) are dominated by")
    print(f"    ties, with weak GPT-4-Turbo advantage. This heterogeneity is a")
    print(f"    necessary condition for routing; whether it is sufficient for")
    print(f"    practical gains is tested in the routing evaluation (Table 2).")
    if results.get('random_distribution'):
        rd = results['random_distribution']
        router_v = eff['cramers_v']
        print(f"\n  NULL BASELINE (N={rd['n_seeds']} random projections):")
        print(f"    V_random: median={rd['median']:.3f}, "
              f"IQR=[{rd['iqr'][0]:.3f}, {rd['iqr'][1]:.3f}], max={rd['max']:.3f}")
        print(f"    Router PCA V = {router_v:.3f}")
        print(f"    Signal ratio vs median: {rd['ratio_vs_median']:.1f}x")
        print(f"    Signal ratio vs max:    {rd['ratio_vs_max']:.1f}x")
        print(f"    Random projections >= router PCA: {rd['n_exceed_router']}/{rd['n_valid']}")
        if rd['n_exceed_router'] == 0:
            print(f"    Router PCA exceeds ALL {rd['n_valid']} random projections.")
    elif results.get('random') and results['random']['effects']['cramers_v'] > 0:
        rand_v = results['random']['effects']['cramers_v']
        router_v = eff['cramers_v']
        print(f"\n  NULL BASELINE (single seed):")
        print(f"    Random projection V = {rand_v:.3f} vs Router PCA V = {router_v:.3f}")
        if router_v > 2 * rand_v:
            print(f"    Router PCA captures {router_v/rand_v:.1f}x more signal than chance.")
        else:
            print(f"    WARNING: Router PCA signal is weak relative to random baseline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
