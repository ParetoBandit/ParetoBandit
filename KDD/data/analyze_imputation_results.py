#!/usr/bin/env python3
"""
Statistical Analysis of Anchor-Based Imputation Results

This script analyzes the quality and statistical significance of the
anchor-based alignment used for imputing missing benchmark scores.

Tests performed:
1. Coefficient significance (t-test, p-values)
2. Model fit quality (R², adjusted R², F-statistic)
3. Residual analysis (normality, homoscedasticity)
4. Confidence intervals for coefficients
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_alignment_results():
    """Load alignment parameters from imputation."""
    alignment_path = Path(__file__).parent / "anchor_based_imputation" / "alignment_parameters.json"
    
    with open(alignment_path, 'r') as f:
        return json.load(f)


def load_models_data():
    """Load models cache."""
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data['models'])


def compute_coefficient_significance(X, y, alpha, beta):
    """
    Compute statistical significance of regression coefficients.
    
    Returns:
        dict with t-stats, p-values, confidence intervals
    """
    n = len(X)
    k = 1  # number of predictors
    
    # Predictions and residuals
    y_pred = alpha * X.flatten() + beta
    residuals = y - y_pred
    
    # Residual standard error
    RSS = np.sum(residuals**2)
    RSE = np.sqrt(RSS / (n - k - 1))
    
    # Standard errors of coefficients
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    # Variance-covariance matrix
    XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    var_covar = RSE**2 * XtX_inv
    
    # Standard errors
    SE_beta = np.sqrt(var_covar[0, 0])  # intercept
    SE_alpha = np.sqrt(var_covar[1, 1])  # slope
    
    # t-statistics
    t_alpha = alpha / SE_alpha
    t_beta = beta / SE_beta
    
    # p-values (two-tailed)
    df = n - k - 1
    p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), df))
    p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df))
    
    # 95% Confidence intervals
    t_crit = stats.t.ppf(0.975, df)
    
    alpha_ci = (alpha - t_crit * SE_alpha, alpha + t_crit * SE_alpha)
    beta_ci = (beta - t_crit * SE_beta, beta + t_crit * SE_beta)
    
    # F-statistic for overall model
    TSS = np.sum((y - y.mean())**2)
    ESS = TSS - RSS
    F_stat = (ESS / k) / (RSS / (n - k - 1))
    p_F = 1 - stats.f.cdf(F_stat, k, n - k - 1)
    
    # Adjusted R²
    r2 = 1 - (RSS / TSS)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    
    return {
        'alpha': {
            't_stat': t_alpha,
            'p_value': p_alpha,
            'se': SE_alpha,
            'ci_lower': alpha_ci[0],
            'ci_upper': alpha_ci[1],
            'significant': p_alpha < 0.05
        },
        'beta': {
            't_stat': t_beta,
            'p_value': p_beta,
            'se': SE_beta,
            'ci_lower': beta_ci[0],
            'ci_upper': beta_ci[1],
            'significant': p_beta < 0.05
        },
        'model': {
            'F_stat': F_stat,
            'p_F': p_F,
            'r2': r2,
            'adj_r2': adj_r2,
            'RSE': RSE,
            'n': n,
            'df': df
        }
    }


def analyze_alignment(target_col, proxy_col, df, alignment_params):
    """Full statistical analysis of an alignment."""
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: {target_col.upper()}")
    print(f"{'='*80}")
    print(f"Formula: {target_col} = α × {proxy_col} + β")
    print()
    
    # Get anchor models
    anchors = df[[target_col, proxy_col]].dropna()
    X = anchors[proxy_col].values
    y = anchors[target_col].values
    
    alpha = alignment_params['alpha']
    beta = alignment_params['beta']
    r2 = alignment_params['r2']
    n_anchors = alignment_params['n_anchors']
    
    print(f"Anchor Models: {n_anchors}")
    print(f"Proxy: {proxy_col}")
    print()
    
    # Statistical significance
    sig_results = compute_coefficient_significance(X, y, alpha, beta)
    
    # Display coefficients
    print("="*80)
    print("COEFFICIENT ESTIMATES")
    print("="*80)
    print()
    
    print(f"α (Slope):")
    print(f"  Estimate:     {alpha:>10.6f}")
    print(f"  Std Error:    {sig_results['alpha']['se']:>10.6f}")
    print(f"  t-statistic:  {sig_results['alpha']['t_stat']:>10.4f}")
    print(f"  p-value:      {sig_results['alpha']['p_value']:>10.6f}  {'***' if sig_results['alpha']['p_value'] < 0.001 else '**' if sig_results['alpha']['p_value'] < 0.01 else '*' if sig_results['alpha']['p_value'] < 0.05 else 'ns'}")
    print(f"  95% CI:       [{sig_results['alpha']['ci_lower']:>9.6f}, {sig_results['alpha']['ci_upper']:>9.6f}]")
    print(f"  Significant:  {'✓ YES' if sig_results['alpha']['significant'] else '✗ NO'}")
    print()
    
    print(f"β (Intercept):")
    print(f"  Estimate:     {beta:>10.6f}")
    print(f"  Std Error:    {sig_results['beta']['se']:>10.6f}")
    print(f"  t-statistic:  {sig_results['beta']['t_stat']:>10.4f}")
    print(f"  p-value:      {sig_results['beta']['p_value']:>10.6f}  {'***' if sig_results['beta']['p_value'] < 0.001 else '**' if sig_results['beta']['p_value'] < 0.01 else '*' if sig_results['beta']['p_value'] < 0.05 else 'ns'}")
    print(f"  95% CI:       [{sig_results['beta']['ci_lower']:>9.6f}, {sig_results['beta']['ci_upper']:>9.6f}]")
    print(f"  Significant:  {'✓ YES' if sig_results['beta']['significant'] else '✗ NO'}")
    print()
    
    # Model fit
    print("="*80)
    print("MODEL FIT STATISTICS")
    print("="*80)
    print()
    print(f"R²:                {sig_results['model']['r2']:>10.4f}")
    print(f"Adjusted R²:       {sig_results['model']['adj_r2']:>10.4f}")
    print(f"F-statistic:       {sig_results['model']['F_stat']:>10.4f}")
    print(f"p-value (F):       {sig_results['model']['p_F']:>10.6f}  {'***' if sig_results['model']['p_F'] < 0.001 else '**' if sig_results['model']['p_F'] < 0.01 else '*' if sig_results['model']['p_F'] < 0.05 else 'ns'}")
    print(f"Residual Std Err:  {sig_results['model']['RSE']:>10.6f}")
    print(f"Degrees of Freedom: {sig_results['model']['df']}")
    print()
    
    # Interpretation
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    
    if sig_results['alpha']['significant'] and sig_results['model']['p_F'] < 0.05:
        print("✓ VALID ALIGNMENT:")
        print(f"  • Slope (α) is statistically significant (p < 0.05)")
        print(f"  • Model explains {r2*100:.1f}% of variance in {target_col}")
        print(f"  • Overall model is significant (F-test p < 0.05)")
        
        if r2 > 0.7:
            print(f"  • STRONG fit (R² = {r2:.4f})")
            quality = "EXCELLENT"
        elif r2 > 0.5:
            print(f"  • MODERATE fit (R² = {r2:.4f})")
            quality = "ACCEPTABLE"
        else:
            print(f"  • WEAK fit (R² = {r2:.4f})")
            quality = "QUESTIONABLE"
        
        print()
        print(f"VERDICT: {quality} - Safe to use for imputation")
        
    else:
        print("⚠️  PROBLEMATIC ALIGNMENT:")
        if not sig_results['alpha']['significant']:
            print(f"  • Slope (α) is NOT significant (p = {sig_results['alpha']['p_value']:.4f})")
            print(f"  • {proxy_col} may not predict {target_col} reliably")
        if sig_results['model']['p_F'] >= 0.05:
            print(f"  • Overall model is NOT significant (F-test p = {sig_results['model']['p_F']:.4f})")
        
        print()
        print("VERDICT: QUESTIONABLE - Consider alternative imputation method")
    
    # Residual analysis
    print()
    print("="*80)
    print("RESIDUAL DIAGNOSTICS")
    print("="*80)
    print()
    
    y_pred = alpha * X + beta
    residuals = y - y_pred
    
    # Normality test (Shapiro-Wilk)
    if len(residuals) >= 3:
        _, p_normality = stats.shapiro(residuals)
        print(f"Shapiro-Wilk test (normality):")
        print(f"  p-value: {p_normality:.6f}")
        print(f"  Result:  {'✓ Residuals appear normal' if p_normality > 0.05 else '⚠ Residuals may not be normal'}")
        print()
    
    # Homoscedasticity (Breusch-Pagan approximation)
    standardized_residuals = residuals / np.std(residuals)
    print(f"Residual statistics:")
    print(f"  Mean:  {np.mean(residuals):.6f}  (should be ~0)")
    print(f"  Std:   {np.std(residuals):.6f}")
    print(f"  Min:   {np.min(residuals):.6f}")
    print(f"  Max:   {np.max(residuals):.6f}")
    print()
    
    # Check for outliers
    outliers = np.abs(standardized_residuals) > 3
    n_outliers = np.sum(outliers)
    print(f"Outliers (|z| > 3): {n_outliers}/{len(residuals)}")
    if n_outliers > 0:
        print(f"  Warning: {n_outliers} potential outliers detected")
        outlier_indices = np.where(outliers)[0]
        for idx in outlier_indices[:5]:  # Show up to 5
            print(f"    Model: {anchors.iloc[idx].name}, Residual: {residuals[idx]:.4f}")
    else:
        print(f"  ✓ No significant outliers")
    
    return sig_results


def create_summary_table(alignments_data, df):
    """Create summary table of all imputation results."""
    
    print("\n" + "="*80)
    print("IMPUTATION QUALITY SUMMARY")
    print("="*80)
    print()
    
    summary_data = []
    
    for target, params in alignments_data.items():
        proxy = params['proxy']
        alpha = params['alpha']
        beta = params['beta']
        r2 = params['r2']
        n_anchors = params['n_anchors']
        
        # Get anchor data
        anchors = df[[target, proxy]].dropna()
        X = anchors[proxy].values
        y = anchors[target].values
        
        # Compute significance
        sig = compute_coefficient_significance(X, y, alpha, beta)
        
        summary_data.append({
            'Target': target,
            'Proxy': proxy,
            'N': n_anchors,
            'α': alpha,
            'α p-value': sig['alpha']['p_value'],
            'β': beta,
            'β p-value': sig['beta']['p_value'],
            'R²': r2,
            'Adj R²': sig['model']['adj_r2'],
            'F p-value': sig['model']['p_F'],
            'Quality': 'STRONG' if r2 > 0.7 else 'MODERATE' if r2 > 0.5 else 'WEAK'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(summary_df.to_string(index=False))
    print()
    
    print("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    print()
    
    # Overall assessment
    print("="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)
    print()
    
    n_strong = sum(1 for d in summary_data if d['Quality'] == 'STRONG')
    n_moderate = sum(1 for d in summary_data if d['Quality'] == 'MODERATE')
    n_weak = sum(1 for d in summary_data if d['Quality'] == 'WEAK')
    
    print(f"Alignment quality:")
    print(f"  • STRONG (R² > 0.7):    {n_strong}/{len(summary_data)}")
    print(f"  • MODERATE (R² > 0.5):  {n_moderate}/{len(summary_data)}")
    print(f"  • WEAK (R² ≤ 0.5):      {n_weak}/{len(summary_data)}")
    print()
    
    n_sig_alpha = sum(1 for d in summary_data if d['α p-value'] < 0.05)
    n_sig_model = sum(1 for d in summary_data if d['F p-value'] < 0.05)
    
    print(f"Statistical significance:")
    print(f"  • Significant slopes (α): {n_sig_alpha}/{len(summary_data)}")
    print(f"  • Significant models (F): {n_sig_model}/{len(summary_data)}")
    print()
    
    if n_sig_alpha == len(summary_data) and n_sig_model == len(summary_data):
        print("✓ ALL alignments are statistically valid")
        print("  Safe to use imputed scores for downstream analysis")
    elif n_weak > 0:
        print("⚠️  Some alignments have weak R²")
        print("  Consider:")
        print("    1. Using alternative proxies")
        print("    2. Excluding weakly imputed models from analysis")
        print("    3. Manual evaluation for weak alignments")
    
    return summary_df


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("ANCHOR-BASED IMPUTATION: STATISTICAL ANALYSIS")
    print("="*80)
    
    # Load data
    alignments = load_alignment_results()
    df = load_models_data()
    
    # Analyze each alignment
    for target, params in alignments.items():
        analyze_alignment(target, params['proxy'], df, params)
    
    # Summary table
    summary_df = create_summary_table(alignments, df)
    
    # Save summary
    output_dir = Path(__file__).parent / "anchor_based_imputation"
    summary_path = output_dir / "statistical_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved statistical summary to: {summary_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
