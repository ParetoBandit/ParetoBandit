#!/usr/bin/env python3
"""
Compute a Composite Reasoning Score (CRS) per model.

This script reads from the models_cache.json and computes a statistically
principled reasoning score using Bayesian latent factor modeling.

Default benchmarks:
- math_500: Mathematical problem solving (MATH-500)
- gpqa: Graduate-level science questions
- hle: Humanity's Last Exam
- aime: Competition-level mathematics
- math_index: Artificial Analysis math composite

Custom benchmarks can be specified via --benchmarks or --config.

Usage:
    # Use defaults
    python scripts/compute_reasoning_score.py
    python scripts/compute_reasoning_score.py --bayesian
    
    # Custom benchmarks
    python scripts/compute_reasoning_score.py --benchmarks math_500:100:0.4 gpqa:1:0.3 aime:1:0.3
    
    # From config file
    python scripts/compute_reasoning_score.py --config my_benchmarks.json
    
    # Save config for later
    python scripts/compute_reasoning_score.py --save-config reasoning_config.json

Requirements:
    Default: pandas numpy
    Bayesian: pip install pymc arviz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import from refactored module
from llm_jury.analysis.latent_factor import (
    BenchmarkSuite,
    REASONING_BENCHMARKS,
    extract_benchmark_matrix,
    prepare_long_data,
    fit_latent_factor_model,
    summarize_latent_scores,
    compute_weighted_zscore,
    transform_to_0_100,
    update_models_cache,
    get_benchmark_diagnostics,
    parse_benchmark_args,
    add_benchmark_args,
)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "models_cache.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Composite Reasoning Score (CRS).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default benchmarks
  python compute_reasoning_score.py --bayesian
  
  # Custom benchmarks (format: field:scale:weight)
  python compute_reasoning_score.py --benchmarks math_500:100:0.4 gpqa:1:0.3 aime:1:0.3
  
  # Load from config file
  python compute_reasoning_score.py --config reasoning_config.json
  
  # Save default config
  python compute_reasoning_score.py --save-config reasoning_config.json
"""
    )
    
    # Add benchmark arguments
    add_benchmark_args(parser)
    
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show results without updating cache"
    )
    parser.add_argument(
        "--bayesian", action="store_true",
        help="Use Bayesian latent factor model (requires pymc/arviz)"
    )
    parser.add_argument(
        "--draws", type=int, default=2000,
        help="Number of posterior draws (per chain) for MCMC (Bayesian only)."
    )
    parser.add_argument(
        "--tune", type=int, default=2000,
        help="Number of tuning steps for MCMC (Bayesian only)."
    )
    parser.add_argument(
        "--chains", type=int, default=4,
        help="Number of MCMC chains (Bayesian only)."
    )
    parser.add_argument(
        "--target_accept", type=float, default=0.97,
        help="Target acceptance rate for NUTS sampler (Bayesian only)."
    )
    parser.add_argument(
        "--random_seed", type=int, default=123,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--min-benchmarks", type=int, default=3,
        help="Minimum number of benchmarks required for a model to be included."
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Print detailed convergence diagnostics (Bayesian only)."
    )
    parser.add_argument(
        "--winsorize", type=float, default=None,
        help="Winsorize z-scores at +/- this value (e.g., 3.0 to cap at ±3)."
    )
    parser.add_argument(
        "--score-prefix", type=str, default=None,
        help="Override the score prefix (default: from suite config, e.g., 'crs')."
    )
    return parser.parse_args()


def load_models_cache():
    """Load models from the cache file."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    return data, models


def print_benchmark_suite(suite: BenchmarkSuite):
    """Print benchmark suite configuration."""
    print(f"\nBenchmark Suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Score prefix: {suite.score_prefix}")
    print("\nBenchmarks:")
    weights = suite.get_weights()
    for name, cfg in suite.benchmarks.items():
        print(f"  - {name}:")
        print(f"      Description: {cfg.description}")
        print(f"      Scale: {cfg.scale}")
        print(f"      Invert: {cfg.invert}")
        print(f"      Weight: {weights[name]:.2%}")


def print_diagnostics(diagnostics, benchmark_names):
    """Print convergence diagnostics for the Bayesian model."""
    print("\n" + "=" * 60)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 60)
    
    print("\nBenchmark Loadings (lambda):")
    for bench in benchmark_names:
        row = diagnostics['lambda'].loc[bench]
        print(f"  {bench}: {row['mean']:.3f} ± {row['sd']:.3f} "
              f"[{row['hdi_2.5%']:.3f}, {row['hdi_97.5%']:.3f}] "
              f"R̂={row['r_hat']:.3f}")
    
    print("\nBenchmark Noise (sigma):")
    for bench in benchmark_names:
        row = diagnostics['sigma'].loc[bench]
        print(f"  {bench}: {row['mean']:.3f} ± {row['sd']:.3f} "
              f"[{row['hdi_2.5%']:.3f}, {row['hdi_97.5%']:.3f}] "
              f"R̂={row['r_hat']:.3f}")
    
    print("\nBenchmark Intercepts (alpha):")
    for bench in benchmark_names:
        row = diagnostics['alpha'].loc[bench]
        print(f"  {bench}: {row['mean']:.3f} ± {row['sd']:.3f} "
              f"[{row['hdi_2.5%']:.3f}, {row['hdi_97.5%']:.3f}] "
              f"R̂={row['r_hat']:.3f}")


def main():
    args = parse_args()
    
    # Handle --list-benchmarks
    if args.list_benchmarks:
        print("Default Reasoning Benchmarks (CRS):")
        print_benchmark_suite(REASONING_BENCHMARKS)
        return
    
    # Parse benchmark configuration
    suite = parse_benchmark_args(
        benchmark_args=args.benchmarks,
        config_path=args.config,
        default_suite=REASONING_BENCHMARKS,
    )
    
    # Override score prefix if specified
    if args.score_prefix:
        suite.score_prefix = args.score_prefix
    
    # Handle --save-config
    if args.save_config:
        suite.to_json(args.save_config)
        print(f"Saved benchmark config to {args.save_config}")
        return
    
    score_prefix = suite.score_prefix
    
    print("=" * 60)
    print(f"COMPOSITE SCORE ({score_prefix.upper()}) COMPUTATION")
    print("=" * 60)
    
    # Show benchmark configuration
    print_benchmark_suite(suite)
    
    # Load data
    print("\nLoading models cache...")
    cache_data, models = load_models_cache()
    print(f"Loaded {len(models)} models.")
    
    # Show benchmark coverage
    print("\nBenchmark coverage:")
    for name in suite.benchmarks:
        count = sum(1 for m in models if m.get(name) is not None)
        print(f"  - {name}: {count} models")
    
    # Extract benchmark matrix
    print(f"\nExtracting benchmarks (min {args.min_benchmarks} required)...")
    df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
        models, suite.get_configs(), min_benchmarks=args.min_benchmarks
    )
    print(f"Models with sufficient data: {len(model_names)}")
    
    if len(model_names) == 0:
        print("\nERROR: No models have sufficient benchmark data.")
        sys.exit(1)
    
    # Apply winsorization if requested
    if args.winsorize is not None:
        print(f"\nApplying winsorization at ±{args.winsorize}...")
        df_z = df_z.clip(lower=-args.winsorize, upper=args.winsorize)
    
    # Compute composite score
    method = "bayesian" if args.bayesian else "weighted_zscore"
    
    if args.bayesian:
        print("\n" + "-" * 60)
        print("BAYESIAN LATENT FACTOR MODEL")
        print("-" * 60)
        
        # Prepare long-format data
        z_obs, idx_model, idx_bench, n_models, n_benchmarks = prepare_long_data(
            df_z, model_names, benchmark_names
        )
        print(f"Observations: {len(z_obs)} (models={n_models}, benchmarks={n_benchmarks})")
        
        # Fit model
        print(f"\nFitting model (draws={args.draws}, tune={args.tune}, chains={args.chains})...")
        idata = fit_latent_factor_model(
            z_obs, idx_model, idx_bench, n_models, n_benchmarks,
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=args.random_seed
        )
        
        # Print diagnostics if requested
        if args.diagnostics:
            diagnostics = get_benchmark_diagnostics(idata, benchmark_names)
            print_diagnostics(diagnostics, benchmark_names)
        
        # Summarize scores
        df_result = summarize_latent_scores(idata, model_names, score_name=score_prefix)
    else:
        print("\n" + "-" * 60)
        print("WEIGHTED Z-SCORE METHOD")
        print("-" * 60)
        df_result = compute_weighted_zscore(
            df_z, model_names, suite.get_weights(), score_name=score_prefix
        )
    
    # Transform to 0-100 scale
    df_result = transform_to_0_100(
        df_result, 
        mean_col=f"{score_prefix}_mean", 
        output_col=f"{score_prefix}_100",
        hdi_low_col=f"{score_prefix}_hdi_low",
        hdi_high_col=f"{score_prefix}_hdi_high"
    )
    
    # Sort by score
    df_result = df_result.sort_values(f'{score_prefix}_mean', ascending=False)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"COMPOSITE SCORE ({score_prefix.upper()}) RESULTS")
    print("=" * 60)
    print(f"\nTop 20 models (method: {method}):\n")
    
    print(f"{'Rank':<5} {'Model':<45} {f'{score_prefix.upper()} (z)':<12} {f'{score_prefix.upper()} (0-100)':<14} {'95% HDI':<18}")
    print("-" * 100)
    
    for rank, (_, row) in enumerate(df_result.head(20).iterrows(), 1):
        model = row['model'][:44]
        score_z = f"{row[f'{score_prefix}_mean']:.3f}" if not pd.isna(row[f'{score_prefix}_mean']) else "N/A"
        score_100 = f"{row[f'{score_prefix}_100']:.1f}" if not pd.isna(row.get(f'{score_prefix}_100')) else "N/A"
        hdi_low = row.get(f'{score_prefix}_hdi_low')
        hdi_high = row.get(f'{score_prefix}_hdi_high')
        hdi = f"[{hdi_low:.2f}, {hdi_high:.2f}]" if hdi_low is not None and not pd.isna(hdi_low) else "N/A"
        print(f"{rank:<5} {model:<45} {score_z:<12} {score_100:<14} {hdi:<18}")
    
    # Update cache
    if not args.dry_run:
        print("\nUpdating models cache...")
        cache_data, count = update_models_cache(cache_data, df_result, score_prefix, method=method)
        print(f"Updated {count} models with {score_prefix.upper()} scores.")
        
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved to {CACHE_PATH}")
    else:
        print("\n[DRY RUN] Cache not updated.")
    
    # Save detailed results
    output_path = PROJECT_ROOT / "data" / f"{score_prefix}_scores_detailed.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
