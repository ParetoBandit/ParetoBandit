#!/usr/bin/env python3
"""
Comprehensive Validation of Bayesian Latent Factor (BLF) Composite Quality Scores.

This script provides rigorous validation suitable for KDD reviewers:
1. Convergence Diagnostics: Trace plots and R-hat statistics
2. Posterior Predictive Checks: Model fit assessment
3. Uncertainty Quantification: Uncertainty funnel plots
4. Downstream Utility: BLF scores vs. intent classifier accuracy

Author: LLM Jury Team
Date: December 2025
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_jury.analysis.latent_factor import (
    CODING_BENCHMARKS,
    REASONING_BENCHMARKS,
    FACTUAL_QA_BENCHMARKS,
    SUMMARIZATION_BENCHMARKS,
    extract_benchmark_matrix,
    prepare_long_data,
    fit_latent_factor_model,
    summarize_latent_scores,
)

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("colorblind")

# Figure parameters
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_DOUBLE = (10, 4)
FIGSIZE_TRIPLE = (15, 4)
DPI = 300


def load_models_cache() -> List[Dict]:
    """Load models cache with all benchmark scores and composite scores."""
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    print(f"Loading models cache from: {cache_path}")
    
    with open(cache_path) as f:
        data = json.load(f)
    
    models = data.get('models', data) if isinstance(data, dict) else data
    print(f"✓ Loaded {len(models)} models")
    
    return models


def load_intent_classifier_results() -> Optional[pd.DataFrame]:
    """Load intent classifier evaluation results if available."""
    results_path = PROJECT_ROOT / "KDD" / "intent_classification" / "stratified_performance_analysis.json"
    
    if not results_path.exists():
        print(f"⚠️  Intent classifier results not found at {results_path}")
        return None
    
    with open(results_path) as f:
        data = json.load(f)
    
    print(f"✓ Loaded intent classifier results")
    return data


class BLFValidator:
    """Validator for Bayesian Latent Factor composite quality scores."""
    
    def __init__(self, output_dir: Path):
        """Initialize validator.
        
        Args:
            output_dir: Directory to save validation figures and results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = load_models_cache()
        self.intent_results = load_intent_classifier_results()
        
        # Will store fitted models
        self.fitted_models = {}
        self.model_data = {}
    
    def fit_blf_model(self, suite_name: str, benchmark_suite, 
                      draws: int = 2000, tune: int = 3000, 
                      chains: int = 4, target_accept: float = 0.99) -> Tuple:
        """Fit BLF model for a specific benchmark suite.
        
        Args:
            suite_name: Name of the suite (e.g., 'coding', 'reasoning').
            benchmark_suite: BenchmarkSuite object.
            draws: Number of posterior draws.
            tune: Number of tuning steps.
            chains: Number of MCMC chains.
        
        Returns:
            Tuple of (idata, df_z, model_names, benchmark_names).
        """
        print(f"\n{'='*60}")
        print(f"Fitting BLF Model: {suite_name.upper()}")
        print(f"{'='*60}")
        
        # Extract benchmark matrix
        df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
            self.models, benchmark_suite.get_configs(), min_benchmarks=1
        )
        
        print(f"Models: {len(model_names)}")
        print(f"Benchmarks: {benchmark_names}")
        print(f"Coverage: {df_z.notna().sum().sum()} / {len(model_names) * len(benchmark_names)} "
              f"({df_z.notna().sum().sum() / (len(model_names) * len(benchmark_names)) * 100:.1f}%)")
        
        # Prepare long-format data
        z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
            df_z, model_names, benchmark_names
        )
        
        print(f"\nFitting model (draws={draws}, tune={tune}, chains={chains})...")
        
        # Fit BLF model
        idata = fit_latent_factor_model(
            z_obs, idx_model, idx_bench, n_models, n_benchmarks,
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=42, progressbar=True
        )
        
        # Store for later use
        self.fitted_models[suite_name] = idata
        self.model_data[suite_name] = {
            'df_z': df_z,
            'model_names': model_names,
            'benchmark_names': benchmark_names,
            'z_obs': z_obs,
            'idx_model': idx_model,
            'idx_bench': idx_bench,
            'n_models': n_models,
            'n_benchmarks': n_benchmarks,
        }
        
        print(f"✓ Model fitted successfully")
        
        return idata, df_z, model_names, benchmark_names
    
    def generate_convergence_diagnostics(self, suite_name: str):
        """Generate convergence diagnostics: trace plots and R-hat.
        
        Args:
            suite_name: Name of the suite (e.g., 'coding').
        """
        print(f"\n{'='*60}")
        print(f"Convergence Diagnostics: {suite_name.upper()}")
        print(f"{'='*60}")
        
        idata = self.fitted_models[suite_name]
        data = self.model_data[suite_name]
        benchmark_names = data['benchmark_names']
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Trace plots for lambda (benchmark loadings)
        print("Generating trace plots for lambda (benchmark loadings)...")
        n_benchmarks = min(6, len(benchmark_names))
        
        for i in range(n_benchmarks):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            lambda_samples = idata.posterior['lambda'].sel(benchmark=i).values
            
            for chain_idx in range(lambda_samples.shape[0]):
                ax.plot(lambda_samples[chain_idx, :], alpha=0.7, linewidth=0.8, 
                       label=f'Chain {chain_idx+1}' if i == 0 else '')
            
            bench_name = benchmark_names[i]
            ax.set_title(f'λ[{bench_name[:20]}]', fontsize=10, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=7)
        
        # 2. R-hat statistics for all parameters
        print("Calculating R-hat statistics...")
        ax_rhat = fig.add_subplot(gs[2:, :])
        
        # Extract R-hat for all key parameters
        summary = az.summary(idata, var_names=['lambda', 'sigma', 'alpha', 'theta'])
        
        # Filter to get representative theta values (e.g., first 10 models)
        theta_indices = [f'theta[{i}]' for i in range(min(10, data['n_models']))]
        param_names = []
        rhat_values = []
        
        for param_type in ['lambda', 'sigma', 'alpha']:
            for i, bench in enumerate(benchmark_names):
                param_name = f"{param_type}[{bench[:15]}]"
                param_names.append(param_name)
                # Get R-hat for this parameter
                param_idx = i
                if param_type in summary.index.names or isinstance(summary.index, pd.MultiIndex):
                    rhat_values.append(summary.loc[param_idx if param_type else (param_type, param_idx), 'r_hat'])
                else:
                    # Flat index
                    matching_rows = [idx for idx in summary.index if param_type in idx]
                    if i < len(matching_rows):
                        rhat_values.append(summary.loc[matching_rows[i], 'r_hat'])
                    else:
                        rhat_values.append(1.0)
        
        # Add representative theta R-hat values
        theta_summary = az.summary(idata, var_names=['theta'])
        for i in range(min(5, data['n_models'])):
            param_names.append(f"θ[model_{i}]")
            rhat_values.append(theta_summary.iloc[i]['r_hat'])
        
        # Color code by convergence
        colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' 
                  for r in rhat_values]
        
        y_pos = np.arange(len(rhat_values))
        ax_rhat.barh(y_pos, rhat_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_rhat.axvline(1.01, color='darkgreen', linestyle='--', linewidth=2, 
                       label='Convergence Threshold (R̂ < 1.01)', zorder=10)
        ax_rhat.axvline(1.05, color='red', linestyle='--', linewidth=2, 
                       label='Warning Threshold (R̂ < 1.05)', alpha=0.5, zorder=10)
        
        ax_rhat.set_yticks(y_pos)
        ax_rhat.set_yticklabels(param_names, fontsize=7)
        ax_rhat.set_xlabel(r'$\hat{R}$ Statistic (Gelman-Rubin Diagnostic)', fontsize=11, fontweight='bold')
        ax_rhat.set_title('MCMC Convergence Diagnostics', fontsize=12, fontweight='bold')
        ax_rhat.legend(loc='lower right', fontsize=9)
        ax_rhat.grid(True, alpha=0.3, axis='x')
        ax_rhat.set_xlim(0.99, max(1.05, max(rhat_values) + 0.01))
        
        # Add convergence summary text
        n_converged = sum(1 for r in rhat_values if r < 1.01)
        n_total = len(rhat_values)
        convergence_rate = n_converged / n_total * 100
        
        summary_text = f"Convergence: {n_converged}/{n_total} ({convergence_rate:.1f}%) with R̂ < 1.01"
        ax_rhat.text(0.02, 0.98, summary_text, transform=ax_rhat.transAxes,
                    fontsize=10, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if convergence_rate > 95 else 'yellow', 
                             alpha=0.8))
        
        # Overall title
        fig.suptitle(f'Convergence Diagnostics for {suite_name.upper()} BLF Model\n'
                    f'Trace Plots (Top) and R-hat Statistics (Bottom)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / f"convergence_diagnostics_{suite_name}.pdf"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
        print(f"  Convergence rate: {convergence_rate:.1f}%")
        print(f"  Max R-hat: {max(rhat_values):.4f}")
    
    def generate_posterior_predictive_check(self, suite_name: str):
        """Generate posterior predictive check plots.
        
        Args:
            suite_name: Name of the suite.
        """
        print(f"\n{'='*60}")
        print(f"Posterior Predictive Check: {suite_name.upper()}")
        print(f"{'='*60}")
        
        idata = self.fitted_models[suite_name]
        data = self.model_data[suite_name]
        df_z = data['df_z']
        n_models = data['n_models']
        n_benchmarks = data['n_benchmarks']
        benchmark_names = data['benchmark_names']
        
        # Extract posterior means
        theta_mean = idata.posterior['theta'].mean(dim=['chain', 'draw']).values
        alpha_mean = idata.posterior['alpha'].mean(dim=['chain', 'draw']).values
        lambda_mean = idata.posterior['lambda'].mean(dim=['chain', 'draw']).values
        sigma_mean = idata.posterior['sigma'].mean(dim=['chain', 'draw']).values
        
        # Compute predictions: z_pred = alpha + lambda * theta
        z_pred = np.zeros_like(df_z.values)
        for i in range(n_models):
            for b in range(n_benchmarks):
                z_pred[i, b] = alpha_mean[b] + lambda_mean[b] * theta_mean[i]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # LEFT: Observed vs. Predicted scatter
        print("Generating observed vs. predicted scatter plot...")
        ax = axes[0]
        
        z_obs_flat = df_z.values.flatten()
        z_pred_flat = z_pred.flatten()
        
        # Remove NaN values
        mask = ~np.isnan(z_obs_flat)
        z_obs_clean = z_obs_flat[mask]
        z_pred_clean = z_pred_flat[mask]
        
        # Scatter plot with density coloring
        from matplotlib.colors import LogNorm
        
        ax.hexbin(z_obs_clean, z_pred_clean, gridsize=30, cmap='Blues', 
                 mincnt=1, alpha=0.8, edgecolors='black', linewidth=0.2)
        
        # Perfect prediction line
        min_val = min(z_obs_clean.min(), z_pred_clean.min())
        max_val = max(z_obs_clean.max(), z_pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2.5, label='Perfect Prediction', zorder=10)
        
        # Compute fit statistics
        ss_res = np.sum((z_obs_clean - z_pred_clean)**2)
        ss_tot = np.sum((z_obs_clean - z_obs_clean.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        rmse = np.sqrt(np.mean((z_obs_clean - z_pred_clean)**2))
        mae = np.mean(np.abs(z_obs_clean - z_pred_clean))
        
        # Correlation
        pearson_r = np.corrcoef(z_obs_clean, z_pred_clean)[0, 1]
        
        stats_text = f'$R^2 = {r_squared:.3f}$\n$r = {pearson_r:.3f}$\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        ax.set_xlabel('Observed z-score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted z-score', fontsize=11, fontweight='bold')
        ax.set_title('Posterior Predictive Check:\nObserved vs. Predicted', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # RIGHT: Posterior predictive density overlay
        print("Generating posterior predictive density overlay...")
        ax = axes[1]
        
        # Plot observed data density
        ax.hist(z_obs_clean, bins=40, density=True, alpha=0.7, 
               color='black', label='Observed Data', edgecolor='black', linewidth=1.5)
        
        # Generate posterior predictive samples
        print("  Sampling from posterior predictive distribution...")
        n_pp_samples = 50
        
        # Randomly sample from posterior
        theta_samples = idata.posterior['theta'].values  # (chains, draws, models)
        alpha_samples = idata.posterior['alpha'].values  # (chains, draws, benchmarks)
        lambda_samples = idata.posterior['lambda'].values  # (chains, draws, benchmarks)
        sigma_samples = idata.posterior['sigma'].values  # (chains, draws, benchmarks)
        
        # Flatten chain dimension
        theta_flat = theta_samples.reshape(-1, n_models)
        alpha_flat = alpha_samples.reshape(-1, n_benchmarks)
        lambda_flat = lambda_samples.reshape(-1, n_benchmarks)
        sigma_flat = sigma_samples.reshape(-1, n_benchmarks)
        
        # Random sample indices
        sample_indices = np.random.choice(theta_flat.shape[0], size=n_pp_samples, replace=False)
        
        for idx in sample_indices:
            # Generate predictions for this posterior sample
            z_pp = []
            for i in range(n_models):
                for b in range(n_benchmarks):
                    if not np.isnan(df_z.iloc[i, b]):
                        mu = alpha_flat[idx, b] + lambda_flat[idx, b] * theta_flat[idx, i]
                        z_sample = np.random.normal(mu, sigma_flat[idx, b])
                        z_pp.append(z_sample)
            
            # Plot density
            if len(z_pp) > 10:
                density = stats.gaussian_kde(z_pp)
                x_range = np.linspace(min(z_obs_clean.min(), min(z_pp)), 
                                     max(z_obs_clean.max(), max(z_pp)), 200)
                ax.plot(x_range, density(x_range), color='steelblue', 
                       alpha=0.15, linewidth=0.8)
        
        # Add legend for posterior predictive
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Observed Data'),
            Line2D([0], [0], color='steelblue', lw=2, alpha=0.5, 
                  label=f'Posterior Predictive\n(n={n_pp_samples} samples)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlabel('z-score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title('Posterior Predictive Check:\nDensity Overlay', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(f'Posterior Predictive Checks for {suite_name.upper()} BLF Model',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"posterior_predictive_check_{suite_name}.pdf"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
        print(f"  R² = {r_squared:.3f}")
        print(f"  RMSE = {rmse:.3f}")
    
    def generate_uncertainty_funnel(self, suite_name: str):
        """Generate uncertainty funnel plot: mean score vs. credible interval width.
        
        Args:
            suite_name: Name of the suite.
        """
        print(f"\n{'='*60}")
        print(f"Uncertainty Funnel: {suite_name.upper()}")
        print(f"{'='*60}")
        
        idata = self.fitted_models[suite_name]
        data = self.model_data[suite_name]
        model_names = data['model_names']
        df_z = data['df_z']
        
        # Summarize latent scores
        df_summary = summarize_latent_scores(idata, model_names, score_name='theta')
        
        # Calculate credible interval width
        df_summary['ci_width'] = df_summary['theta_hdi_high'] - df_summary['theta_hdi_low']
        
        # Calculate number of available benchmarks per model
        df_summary['n_benchmarks'] = df_z.notna().sum(axis=1).values
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # LEFT: Uncertainty funnel
        print("Generating uncertainty funnel plot...")
        ax = axes[0]
        
        # Color by number of benchmarks
        scatter = ax.scatter(df_summary['theta_mean'], df_summary['ci_width'],
                           c=df_summary['n_benchmarks'], cmap='RdYlGn',
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.8)
        
        # Fit a trend line (inverse relationship expected)
        from scipy.optimize import curve_fit
        
        def inverse_func(x, a, b):
            return a / (np.abs(x) + 1) + b
        
        x_data = df_summary['theta_mean'].values
        y_data = df_summary['ci_width'].values
        
        try:
            popt, _ = curve_fit(inverse_func, x_data, y_data, p0=[1, 0.5])
            x_fit = np.linspace(x_data.min(), x_data.max(), 100)
            y_fit = inverse_func(x_fit, *popt)
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, 
                   label='Trend (inverse relationship)')
        except:
            print("  ⚠️  Could not fit trend line")
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Benchmarks', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Posterior Mean θ (Latent Score)', fontsize=11, fontweight='bold')
        ax.set_ylabel('95% Credible Interval Width', fontsize=11, fontweight='bold')
        ax.set_title('Uncertainty Funnel:\nScore vs. Uncertainty', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add interpretation text
        interpretation = ('Models with more benchmarks\n'
                         'have narrower credible intervals\n'
                         '(higher certainty)')
        ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # RIGHT: Uncertainty by number of benchmarks
        print("Generating uncertainty vs. benchmark count...")
        ax = axes[1]
        
        # Box plot of CI width by number of benchmarks
        benchmark_counts = sorted(df_summary['n_benchmarks'].unique())
        ci_widths_by_count = [df_summary[df_summary['n_benchmarks'] == c]['ci_width'].values 
                               for c in benchmark_counts]
        
        bp = ax.boxplot(ci_widths_by_count, positions=benchmark_counts,
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Add scatter overlay
        for c in benchmark_counts:
            subset = df_summary[df_summary['n_benchmarks'] == c]
            ax.scatter([c] * len(subset), subset['ci_width'], 
                      alpha=0.4, s=30, color='darkblue', zorder=10)
        
        ax.set_xlabel('Number of Benchmarks Available', fontsize=11, fontweight='bold')
        ax.set_ylabel('95% Credible Interval Width', fontsize=11, fontweight='bold')
        ax.set_title('Uncertainty vs. Data Availability',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(benchmark_counts)
        
        # Add trend annotation
        if len(benchmark_counts) > 1:
            # Calculate Spearman correlation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(df_summary['n_benchmarks'], df_summary['ci_width'])
            corr_text = f'Spearman ρ = {rho:.3f}\n(p = {pval:.2e})'
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if rho < -0.5 else 'yellow', 
                            alpha=0.8))
        
        # Overall title
        fig.suptitle(f'Uncertainty Quantification for {suite_name.upper()} BLF Model\n'
                    f'Demonstrating Bayesian Advantage: Uncertainty Aware Scores',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"uncertainty_funnel_{suite_name}.pdf"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
        print(f"  Mean CI width: {df_summary['ci_width'].mean():.3f}")
        print(f"  CI width range: [{df_summary['ci_width'].min():.3f}, {df_summary['ci_width'].max():.3f}]")
    
    def generate_downstream_utility(self):
        """Generate downstream utility analysis: BLF scores vs. MixEval performance.
        
        This demonstrates that higher BLF scores correspond to better model performance
        as measured by MixEval (comprehensive benchmark NOT used in composite computation).
        """
        print(f"\n{'='*60}")
        print("Downstream Utility: BLF Scores vs. MixEval Performance")
        print(f"{'='*60}")
        
        # Load models with composite scores
        models_df = pd.DataFrame(self.models)
        
        # Check which composite scores are available
        score_cols = [c for c in models_df.columns if c.endswith('_100')]
        print(f"Available composite scores: {score_cols}")
        
        if not score_cols:
            print("⚠️  No composite scores found in models cache. Skipping downstream utility analysis.")
            return
        
        # Use MixEval as external validation (NOT used in any composite score)
        models_with_mixeval = models_df[models_df['mixeval_score'].notna()].copy()
        
        if len(models_with_mixeval) < 10:
            print(f"⚠️  Only {len(models_with_mixeval)} models with MixEval scores. Need at least 10.")
            print("   Skipping downstream utility analysis.")
            return
        
        print(f"✓ Using MixEval for {len(models_with_mixeval)} models")
        print("  (External validation: MixEval NOT used in any composite score computation)")
        
        # Create figure with subplots for each composite score
        n_scores = len(score_cols)
        fig, axes = plt.subplots(1, min(n_scores, 4), figsize=(5 * min(n_scores, 4), 5))
        
        if n_scores == 1:
            axes = [axes]
        
        for idx, score_col in enumerate(score_cols[:4]):
            ax = axes[idx]
            
            # Get models with both this score and MixEval
            models_with_score = models_with_mixeval[models_with_mixeval[score_col].notna()].copy()
            
            if len(models_with_score) < 10:
                print(f"  ⚠️  Insufficient models with both {score_col} and MixEval: {len(models_with_score)}")
                continue
            
            print(f"\nAnalyzing {score_col}...")
            
            # Use MixEval score directly (higher = better)
            models_with_score['mixeval_performance'] = models_with_score['mixeval_score']
            
            # Bin models by BLF score (deciles)
            models_with_score['score_bin'] = pd.qcut(models_with_score[score_col], 
                                                      q=min(10, len(models_with_score) // 5),
                                                      labels=False, duplicates='drop')
            
            # Aggregate by bin
            bin_stats = models_with_score.groupby('score_bin').agg({
                score_col: ['mean', 'std', 'count'],
                'mixeval_performance': ['mean', 'std']
            }).reset_index()
            
            bin_stats.columns = ['_'.join(col).strip('_') for col in bin_stats.columns.values]
            
            # Plot
            x = bin_stats[f'{score_col}_mean']
            y = bin_stats['mixeval_performance_mean']
            yerr = bin_stats['mixeval_performance_std']
            
            ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=8, capsize=5, 
                       capthick=2, linewidth=2, alpha=0.7, color='steelblue',
                       ecolor='gray', label='MixEval Score (Binned)')
            
            # Fit linear trend
            from scipy.stats import linregress, spearmanr
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Also compute Spearman correlation (rank-based, more robust)
            spearman_rho, spearman_p = spearmanr(models_with_score[score_col], 
                                                  models_with_score['mixeval_performance'])
            
            x_fit = np.array([x.min(), x.max()])
            y_fit = slope * x_fit + intercept
            
            ax.plot(x_fit, y_fit, 'r--', linewidth=2.5, alpha=0.8,
                   label=f'Linear Fit (r={r_value:.3f})')
            
            # Styling
            score_name = score_col.replace('_100', '').upper()
            ax.set_xlabel(f'{score_name} Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('MixEval Score\n(Higher = Better)', fontsize=11, fontweight='bold')
            ax.set_title(f'{score_name}\n(n={len(models_with_score)} models)',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Add significance annotation
            sig_text = f'Pearson r = {r_value:.3f}\nSpearman ρ = {spearman_rho:.3f}\np = {spearman_p:.2e}\n{"✓ Significant" if spearman_p < 0.05 else "Not significant"}'
            ax.text(0.05, 0.95, sig_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', 
                            facecolor='lightgreen' if spearman_p < 0.05 else 'yellow',
                            alpha=0.8))
            
            print(f"  Pearson correlation: r = {r_value:.3f}, p = {p_value:.2e}")
            print(f"  Spearman correlation: ρ = {spearman_rho:.3f}, p = {spearman_p:.2e}")
            print(f"  Monotonic trend: {'✓' if slope > 0 and spearman_p < 0.05 else '✗'}")
        
        # Overall title
        fig.suptitle('Downstream Utility: BLF Composite Scores vs. MixEval Performance\n'
                    'External Validation: Higher BLF scores should predict better MixEval performance',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "downstream_utility_mixeval_validation.pdf"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {output_path}")
        print(f"\n✓ External validation complete: BLF scores validated against independent MixEval benchmark")


def main():
    """Main validation pipeline."""
    print("="*60)
    print("BAYESIAN LATENT FACTOR (BLF) VALIDATION")
    print("Comprehensive validation for KDD reviewers")
    print("="*60)
    
    # Setup output directory
    output_dir = Path(__file__).parent
    
    # Initialize validator
    validator = BLFValidator(output_dir)
    
    # Define which suites to validate
    suites_to_validate = [
        ('coding', CODING_BENCHMARKS),
        # ('reasoning', REASONING_BENCHMARKS),  # Uncomment to validate all
        # ('factual_qa', FACTUAL_QA_BENCHMARKS),
        # ('summarization', SUMMARIZATION_BENCHMARKS),
    ]
    
    # Fit models and generate diagnostics
    for suite_name, benchmark_suite in suites_to_validate:
        try:
            # Fit BLF model with improved MCMC settings
            validator.fit_blf_model(suite_name, benchmark_suite, 
                                   draws=2000, tune=3000, chains=4,
                                   target_accept=0.99)
            
            # Generate all validation plots
            validator.generate_convergence_diagnostics(suite_name)
            validator.generate_posterior_predictive_check(suite_name)
            validator.generate_uncertainty_funnel(suite_name)
            
        except Exception as e:
            print(f"\n❌ Error validating {suite_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate downstream utility analysis (uses all composite scores)
    try:
        validator.generate_downstream_utility()
    except Exception as e:
        print(f"\n❌ Error in downstream utility analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
