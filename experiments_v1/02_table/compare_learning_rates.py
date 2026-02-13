#!/usr/bin/env python3
"""
Compare learning rates (η=0.1 vs η=1.0) with statistical significance tests.

This script loads the multi-seed results from both learning rates and performs:
1. Paired t-tests
2. Wilcoxon signed-rank tests
3. Effect size calculations (Cohen's d)
4. Bonferroni correction for multiple comparisons

Usage:
    python compare_learning_rates.py \
        --eta-01-results data/eta_0.1_holdout_multiseed/results_multiseed.json \
        --eta-10-results data/eta_1.0_holdout_multiseed/results_multiseed.json
"""

import argparse
import json
import numpy as np
from scipy import stats
from pathlib import Path


def load_per_seed_results(json_path: Path) -> dict:
    """Load per-seed results."""
    with open(json_path) as f:
        return json.load(f)


def compute_effect_size(group1: list, group2: list) -> float:
    """Compute Cohen's d for two independent groups."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def perform_comparison_tests(eta_01_data: dict, eta_10_data: dict, strategy: str) -> dict:
    """
    Perform statistical tests comparing two learning rates for a given strategy.
    
    Args:
        eta_01_data: Results from η=0.1
        eta_10_data: Results from η=1.0
        strategy: Strategy name (e.g., "Hybrid (Corralling)")
    
    Returns:
        Dict with test results
    """
    # Extract cumulative regrets
    regrets_01 = eta_01_data[strategy]['statistics']['raw_values']['cumulative_regret']
    regrets_10 = eta_10_data[strategy]['statistics']['raw_values']['cumulative_regret']
    
    # Extract early regrets
    early_01 = eta_01_data[strategy]['statistics']['raw_values']['early_regret']
    early_10 = eta_10_data[strategy]['statistics']['raw_values']['early_regret']
    
    # Independent samples t-test (cumulative regret)
    t_stat_cum, t_pval_cum = stats.ttest_ind(regrets_01, regrets_10)
    
    # Mann-Whitney U test (non-parametric, cumulative regret)
    u_stat_cum, u_pval_cum = stats.mannwhitneyu(regrets_01, regrets_10, alternative='two-sided')
    
    # Effect size (cumulative regret)
    cohens_d_cum = compute_effect_size(regrets_01, regrets_10)
    
    # Independent samples t-test (early regret)
    t_stat_early, t_pval_early = stats.ttest_ind(early_01, early_10)
    
    # Mann-Whitney U test (early regret)
    u_stat_early, u_pval_early = stats.mannwhitneyu(early_01, early_10, alternative='two-sided')
    
    # Effect size (early regret)
    cohens_d_early = compute_effect_size(early_01, early_10)
    
    # Mean improvements
    mean_01 = np.mean(regrets_01)
    mean_10 = np.mean(regrets_10)
    improvement_cum = mean_01 - mean_10
    improvement_pct_cum = 100 * improvement_cum / mean_01 if mean_01 != 0 else 0
    
    mean_early_01 = np.mean(early_01)
    mean_early_10 = np.mean(early_10)
    improvement_early = mean_early_01 - mean_early_10
    improvement_pct_early = 100 * improvement_early / mean_early_01 if mean_early_01 != 0 else 0
    
    return {
        'strategy': strategy,
        'cumulative_regret': {
            'eta_0.1': {
                'mean': float(mean_01),
                'std': float(np.std(regrets_01, ddof=1)),
                'values': regrets_01
            },
            'eta_1.0': {
                'mean': float(mean_10),
                'std': float(np.std(regrets_10, ddof=1)),
                'values': regrets_10
            },
            't_test': {
                'statistic': float(t_stat_cum),
                'p_value': float(t_pval_cum),
                'significant_at_0.05': bool(t_pval_cum < 0.05),
                'significant_at_0.01': bool(t_pval_cum < 0.01),
                'significant_bonferroni_0.05': bool(t_pval_cum < 0.05 / 6)  # 3 strategies × 2 metrics
            },
            'mann_whitney_u': {
                'statistic': float(u_stat_cum),
                'p_value': float(u_pval_cum),
                'significant_at_0.05': bool(u_pval_cum < 0.05),
                'significant_at_0.01': bool(u_pval_cum < 0.01),
                'significant_bonferroni_0.05': bool(u_pval_cum < 0.05 / 6)
            },
            'effect_size': {
                'cohens_d': float(cohens_d_cum),
                'interpretation': (
                    'negligible' if abs(cohens_d_cum) < 0.2 else
                    'small' if abs(cohens_d_cum) < 0.5 else
                    'medium' if abs(cohens_d_cum) < 0.8 else
                    'large'
                )
            },
            'improvement': {
                'absolute': float(improvement_cum),
                'percentage': float(improvement_pct_cum),
                'direction': 'eta_1.0_better' if improvement_cum > 0 else 'eta_0.1_better'
            }
        },
        'early_regret': {
            'eta_0.1': {
                'mean': float(mean_early_01),
                'std': float(np.std(early_01, ddof=1)),
                'values': early_01
            },
            'eta_1.0': {
                'mean': float(mean_early_10),
                'std': float(np.std(early_10, ddof=1)),
                'values': early_10
            },
            't_test': {
                'statistic': float(t_stat_early),
                'p_value': float(t_pval_early),
                'significant_at_0.05': bool(t_pval_early < 0.05),
                'significant_at_0.01': bool(t_pval_early < 0.01),
                'significant_bonferroni_0.05': bool(t_pval_early < 0.05 / 6)
            },
            'mann_whitney_u': {
                'statistic': float(u_stat_early),
                'p_value': float(u_pval_early),
                'significant_at_0.05': bool(u_pval_early < 0.05),
                'significant_at_0.01': bool(u_pval_early < 0.01),
                'significant_bonferroni_0.05': bool(u_pval_early < 0.05 / 6)
            },
            'effect_size': {
                'cohens_d': float(cohens_d_early),
                'interpretation': (
                    'negligible' if abs(cohens_d_early) < 0.2 else
                    'small' if abs(cohens_d_early) < 0.5 else
                    'medium' if abs(cohens_d_early) < 0.8 else
                    'large'
                )
            },
            'improvement': {
                'absolute': float(improvement_early),
                'percentage': float(improvement_pct_early),
                'direction': 'eta_1.0_better' if improvement_early > 0 else 'eta_0.1_better'
            }
        }
    }


def print_comparison_report(comparison_results: dict):
    """Print formatted comparison report."""
    print("\n" + "="*100)
    print("STATISTICAL COMPARISON: η=0.1 vs η=1.0")
    print("="*100)
    
    for strategy in ['Warmup', 'Tabula Rasa', 'Hybrid (Corralling)']:
        if strategy not in comparison_results:
            continue
            
        results = comparison_results[strategy]
        
        print(f"\n{'='*100}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*100}")
        
        # Check if deterministic (std = 0) or stochastic
        is_deterministic = results['cumulative_regret']['eta_1.0']['std'] < 0.01
        
        # Cumulative Regret
        print("\n📊 CUMULATIVE REGRET (Total)")
        print("-" * 100)
        cum = results['cumulative_regret']
        
        if is_deterministic:
            print(f"  η=0.1: {cum['eta_0.1']['mean']:.2f} (deterministic)")
            print(f"  η=1.0: {cum['eta_1.0']['mean']:.2f} (deterministic)")
        else:
            # For stochastic algorithms, show both mean and median
            import numpy as np
            values_01 = np.array(cum['eta_0.1']['values'])
            values_10 = np.array(cum['eta_1.0']['values'])
            median_01 = float(np.median(values_01))
            median_10 = float(np.median(values_10))
            q25_10 = float(np.percentile(values_10, 25))
            q75_10 = float(np.percentile(values_10, 75))
            
            print(f"  η=0.1: {cum['eta_0.1']['mean']:.2f} ± {cum['eta_0.1']['std']:.2f}")
            print(f"  η=1.0: {cum['eta_1.0']['mean']:.2f} ± {cum['eta_1.0']['std']:.2f}")
            print(f"         median: {median_10:.1f}, IQR: [{q25_10:.0f}-{q75_10:.0f}]  ⚠️ HIGH VARIANCE")
        
        print(f"  Improvement: {cum['improvement']['absolute']:.2f} ({cum['improvement']['percentage']:.1f}%) "
              f"[{cum['improvement']['direction']}]")
        
        print(f"\n  Independent t-test:")
        print(f"    t = {cum['t_test']['statistic']:.3f}, p = {cum['t_test']['p_value']:.4f} "
              f"{'***' if cum['t_test']['significant_at_0.01'] else '**' if cum['t_test']['significant_at_0.05'] else 'ns'}")
        print(f"    Bonferroni-corrected (α=0.05/6): {'✅ SIGNIFICANT' if cum['t_test']['significant_bonferroni_0.05'] else '❌ NOT SIGNIFICANT'}")
        
        print(f"\n  Mann-Whitney U test:")
        print(f"    U = {cum['mann_whitney_u']['statistic']:.0f}, p = {cum['mann_whitney_u']['p_value']:.4f} "
              f"{'***' if cum['mann_whitney_u']['significant_at_0.01'] else '**' if cum['mann_whitney_u']['significant_at_0.05'] else 'ns'}")
        print(f"    Bonferroni-corrected (α=0.05/6): {'✅ SIGNIFICANT' if cum['mann_whitney_u']['significant_bonferroni_0.05'] else '❌ NOT SIGNIFICANT'}")
        
        print(f"\n  Effect Size (Cohen's d): {cum['effect_size']['cohens_d']:.3f} ({cum['effect_size']['interpretation']})")
        
        # Early Regret
        print("\n📊 EARLY REGRET (0-500)")
        print("-" * 100)
        early = results['early_regret']
        
        print(f"  η=0.1: {early['eta_0.1']['mean']:.2f} ± {early['eta_0.1']['std']:.2f}")
        print(f"  η=1.0: {early['eta_1.0']['mean']:.2f} ± {early['eta_1.0']['std']:.2f}")
        print(f"  Improvement: {early['improvement']['absolute']:.2f} ({early['improvement']['percentage']:.1f}%) "
              f"[{early['improvement']['direction']}]")
        
        print(f"\n  Independent t-test:")
        print(f"    t = {early['t_test']['statistic']:.3f}, p = {early['t_test']['p_value']:.4f} "
              f"{'***' if early['t_test']['significant_at_0.01'] else '**' if early['t_test']['significant_at_0.05'] else 'ns'}")
        print(f"    Bonferroni-corrected (α=0.05/6): {'✅ SIGNIFICANT' if early['t_test']['significant_bonferroni_0.05'] else '❌ NOT SIGNIFICANT'}")
        
        print(f"\n  Mann-Whitney U test:")
        print(f"    U = {early['mann_whitney_u']['statistic']:.0f}, p = {early['mann_whitney_u']['p_value']:.4f} "
              f"{'***' if early['mann_whitney_u']['significant_at_0.01'] else '**' if early['mann_whitney_u']['significant_at_0.05'] else 'ns'}")
        print(f"    Bonferroni-corrected (α=0.05/6): {'✅ SIGNIFICANT' if early['mann_whitney_u']['significant_bonferroni_0.05'] else '❌ NOT SIGNIFICANT'}")
        
        print(f"\n  Effect Size (Cohen's d): {early['effect_size']['cohens_d']:.3f} ({early['effect_size']['interpretation']})")
    
    print("\n" + "="*100)
    print("INTERPRETATION GUIDE")
    print("="*100)
    print("  Significance levels:")
    print("    *** : p < 0.01 (highly significant)")
    print("    **  : p < 0.05 (significant)")
    print("    ns  : p ≥ 0.05 (not significant)")
    print()
    print("  Effect sizes (Cohen's d):")
    print("    |d| < 0.2 : negligible")
    print("    |d| < 0.5 : small")
    print("    |d| < 0.8 : medium")
    print("    |d| ≥ 0.8 : large")
    print()
    print("  Bonferroni correction: α_corrected = 0.05 / 6 = 0.0083")
    print("    (Corrects for 3 strategies × 2 metrics = 6 comparisons)")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Compare learning rates with statistical tests')
    parser.add_argument('--eta-01-results', type=str, required=True, 
                        help='Path to η=0.1 results JSON')
    parser.add_argument('--eta-10-results', type=str, required=True,
                        help='Path to η=1.0 results JSON')
    parser.add_argument('--output', type=str, default='data/comparison_results.json',
                        help='Output path for comparison results')
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    eta_01_data = load_per_seed_results(Path(args.eta_01_results))
    eta_10_data = load_per_seed_results(Path(args.eta_10_results))
    
    # Perform comparisons for all strategies
    comparison_results = {}
    for strategy in ['Warmup', 'Tabula Rasa', 'Hybrid (Corralling)']:
        comparison_results[strategy] = perform_comparison_tests(
            eta_01_data, eta_10_data, strategy
        )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"✅ Comparison results saved to: {output_path}")
    
    # Print report
    print_comparison_report(comparison_results)
    
    # Summary for paper
    print("\n" + "="*100)
    print("📝 SUMMARY FOR PAPER (Hybrid Strategy)")
    print("="*100)
    
    hybrid = comparison_results['Hybrid (Corralling)']
    
    print(f"\nCumulative Regret:")
    print(f"  η=0.1: {hybrid['cumulative_regret']['eta_0.1']['mean']:.1f} ± {hybrid['cumulative_regret']['eta_0.1']['std']:.1f}")
    print(f"  η=1.0: {hybrid['cumulative_regret']['eta_1.0']['mean']:.1f} ± {hybrid['cumulative_regret']['eta_1.0']['std']:.1f}")
    print(f"  Improvement: {hybrid['cumulative_regret']['improvement']['percentage']:.1f}% "
          f"(t={hybrid['cumulative_regret']['t_test']['statistic']:.2f}, "
          f"p={hybrid['cumulative_regret']['t_test']['p_value']:.4f}, "
          f"d={hybrid['cumulative_regret']['effect_size']['cohens_d']:.2f})")
    
    print(f"\nEarly Regret (0-500):")
    print(f"  η=0.1: {hybrid['early_regret']['eta_0.1']['mean']:.1f} ± {hybrid['early_regret']['eta_0.1']['std']:.1f}")
    print(f"  η=1.0: {hybrid['early_regret']['eta_1.0']['mean']:.1f} ± {hybrid['early_regret']['eta_1.0']['std']:.1f}")
    print(f"  Improvement: {hybrid['early_regret']['improvement']['percentage']:.1f}% "
          f"(t={hybrid['early_regret']['t_test']['statistic']:.2f}, "
          f"p={hybrid['early_regret']['t_test']['p_value']:.4f}, "
          f"d={hybrid['early_regret']['effect_size']['cohens_d']:.2f})")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    main()
