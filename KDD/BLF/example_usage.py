#!/usr/bin/env python3
"""
Example usage of the Bayesian Latent Factor model.

Demonstrates:
1. Loading model data
2. Configuring benchmark suites
3. Fitting the BLF model
4. Extracting composite scores
5. Visualizing results
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_jury.analysis.latent_factor import (
    BenchmarkSuite,
    CODING_BENCHMARKS,
    REASONING_BENCHMARKS,
    extract_benchmark_matrix,
    prepare_long_data,
    fit_latent_factor_model,
    summarize_latent_scores,
    compute_weighted_zscore,
    transform_to_0_100,
    get_benchmark_diagnostics,
)

sns.set_style('whitegrid')


def example1_basic_usage():
    """Example 1: Basic usage with default coding benchmarks."""
    print("\n" + "="*60)
    print("Example 1: Basic BLF Model for Coding Quality")
    print("="*60)
    
    # Load models
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    
    print(f"\nLoaded {len(models)} models")
    
    # Extract benchmark matrix
    print("\nExtracting benchmarks...")
    df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
        models, CODING_BENCHMARKS.get_configs(), min_benchmarks=2
    )
    
    print(f"Models with sufficient data: {len(model_names)}")
    print(f"Benchmarks: {benchmark_names}")
    
    # Prepare for Bayesian inference
    z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
        df_z, model_names, benchmark_names
    )
    
    print(f"\nObservations: {len(z_obs)}")
    print(f"Models: {n_models}, Benchmarks: {n_benchmarks}")
    
    # Fit BLF model
    print("\nFitting Bayesian Latent Factor model...")
    print("(This may take 3-5 minutes)")
    
    idata = fit_latent_factor_model(
        z_obs, idx_model, idx_bench, n_models, n_benchmarks,
        draws=1000,  # Reduced for example
        tune=1000,
        chains=2,
        random_seed=42
    )
    
    print("✓ Model fitted successfully")
    
    # Extract scores
    df_result = summarize_latent_scores(idata, model_names, score_name='ccs')
    df_result = transform_to_0_100(
        df_result, 
        mean_col='ccs_mean',
        output_col='ccs_100',
        hdi_low_col='ccs_hdi_low',
        hdi_high_col='ccs_hdi_high'
    )
    
    # Sort by score
    df_result = df_result.sort_values('ccs_mean', ascending=False)
    
    # Print top 10
    print("\n" + "-"*60)
    print("Top 10 Models by Composite Coding Score (CCS)")
    print("-"*60)
    print(f"{'Rank':<5} {'Model':<35} {'CCS (0-100)':<12} {'95% HDI':<15}")
    print("-"*60)
    
    for rank, (_, row) in enumerate(df_result.head(10).iterrows(), 1):
        model = row['model'][:34]
        score = row['ccs_100']
        hdi_low = row['ccs_100_hdi_low']
        hdi_high = row['ccs_100_hdi_high']
        print(f"{rank:<5} {model:<35} {score:>6.1f}       [{hdi_low:.1f}, {hdi_high:.1f}]")
    
    return df_result, idata, model_names, benchmark_names


def example2_custom_benchmarks():
    """Example 2: Custom benchmark configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Benchmark Suite")
    print("="*60)
    
    # Create custom suite
    suite = BenchmarkSuite(
        name="custom_coding",
        description="Custom coding benchmark suite",
        score_prefix="custom_cs",
    )
    
    # Add benchmarks with custom weights
    suite.add_benchmark('humaneval_score', 'HumanEval', scale=1, weight=0.5)
    suite.add_benchmark('mbpp_score', 'MBPP', scale=1, weight=0.3)
    suite.add_benchmark('livecodebench', 'LiveCodeBench', scale=100, weight=0.2)
    
    print("\nCustom benchmark configuration:")
    weights = suite.get_weights()
    for name, cfg in suite.benchmarks.items():
        print(f"  - {name}: weight={weights[name]:.1%}, scale={cfg.scale}")
    
    # Save configuration
    config_path = Path(__file__).parent / "custom_config.json"
    suite.to_json(str(config_path))
    print(f"\n✓ Configuration saved to {config_path}")
    
    # Load it back
    suite_loaded = BenchmarkSuite.from_json(str(config_path))
    print(f"✓ Configuration loaded: {suite_loaded.name}")
    
    return suite


def example3_diagnostics(idata, benchmark_names):
    """Example 3: Convergence diagnostics."""
    print("\n" + "="*60)
    print("Example 3: Model Diagnostics")
    print("="*60)
    
    # Get diagnostics
    diagnostics = get_benchmark_diagnostics(idata, benchmark_names)
    
    print("\nBenchmark Loadings (λ):")
    print(f"{'Benchmark':<20} {'Mean':<8} {'SD':<8} {'R̂':<8}")
    print("-"*50)
    for bench in benchmark_names:
        row = diagnostics['lambda'].loc[bench]
        print(f"{bench:<20} {row['mean']:>7.3f} {row['sd']:>7.3f} {row['r_hat']:>7.3f}")
    
    print("\nBenchmark Noise (σ):")
    print(f"{'Benchmark':<20} {'Mean':<8} {'SD':<8} {'R̂':<8}")
    print("-"*50)
    for bench in benchmark_names:
        row = diagnostics['sigma'].loc[bench]
        print(f"{bench:<20} {row['mean']:>7.3f} {row['sd']:>7.3f} {row['r_hat']:>7.3f}")
    
    # Check convergence
    all_rhat = diagnostics['lambda']['r_hat'].values
    max_rhat = all_rhat.max()
    
    if max_rhat < 1.01:
        print(f"\n✓ Convergence achieved: max R̂ = {max_rhat:.4f}")
    else:
        print(f"\n⚠️  Convergence issues: max R̂ = {max_rhat:.4f} (should be < 1.01)")


def example4_comparison():
    """Example 4: Compare BLF with weighted z-score."""
    print("\n" + "="*60)
    print("Example 4: BLF vs. Weighted Z-Score")
    print("="*60)
    
    # Load models
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    
    # Extract benchmark matrix
    df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
        models, CODING_BENCHMARKS.get_configs(), min_benchmarks=4  # Require complete data
    )
    
    if len(model_names) == 0:
        print("⚠️  No models with complete benchmark data. Skipping comparison.")
        return
    
    print(f"\nComparing on {len(model_names)} models with complete data")
    
    # Method 1: Weighted Z-Score
    weights = CODING_BENCHMARKS.get_weights()
    df_wzs = compute_weighted_zscore(
        df_z, model_names, weights, score_name='wzs'
    )
    
    # Method 2: BLF (using precomputed if available)
    df_blf = pd.DataFrame({
        'model': model_names,
        'blf_score': [
            next((m['ccs_100'] for m in models if m['name'] == name), np.nan)
            for name in model_names
        ]
    })
    
    # Merge
    df_comp = df_wzs.merge(df_blf, on='model')
    df_comp = df_comp.dropna()
    
    if len(df_comp) == 0:
        print("⚠️  No overlapping scores. Skipping comparison.")
        return
    
    # Correlation
    from scipy.stats import spearmanr, pearsonr
    rho_spearman, p_spearman = spearmanr(df_comp['wzs_mean'], df_comp['blf_score'])
    rho_pearson, p_pearson = pearsonr(df_comp['wzs_mean'], df_comp['blf_score'])
    
    print(f"\nCorrelation between methods:")
    print(f"  Spearman ρ: {rho_spearman:.3f} (p={p_spearman:.2e})")
    print(f"  Pearson r:  {rho_pearson:.3f} (p={p_pearson:.2e})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_comp['wzs_mean'], df_comp['blf_score'], alpha=0.6, s=50)
    ax.plot([df_comp['wzs_mean'].min(), df_comp['wzs_mean'].max()],
            [df_comp['wzs_mean'].min(), df_comp['wzs_mean'].max()],
            'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Weighted Z-Score')
    ax.set_ylabel('BLF Score (CCS)')
    ax.set_title(f'Method Comparison (ρ={rho_spearman:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = Path(__file__).parent / "method_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()


def main():
    """Run all examples."""
    print("="*60)
    print("BAYESIAN LATENT FACTOR MODEL: EXAMPLE USAGE")
    print("="*60)
    
    try:
        # Example 1: Basic usage
        df_result, idata, model_names, benchmark_names = example1_basic_usage()
        
        # Example 2: Custom benchmarks
        suite = example2_custom_benchmarks()
        
        # Example 3: Diagnostics
        example3_diagnostics(idata, benchmark_names)
        
        # Example 4: Comparison
        example4_comparison()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
