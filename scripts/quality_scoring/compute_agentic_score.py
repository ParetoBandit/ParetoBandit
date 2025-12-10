#!/usr/bin/env python3
"""
Compute a Composite Agentic Execution Score (CAE) per model.

This script reads from the models_cache.json and computes a statistically
principled agentic execution score using Bayesian latent factor modeling.

Default benchmarks (from Artificial Analysis API):
- tau2: TAU-bench 2.0 (multi-turn agent tasks in retail/airline domains)
- terminalbench_hard: TerminalBench-Hard (terminal command recovery tasks)

These benchmarks evaluate models on realistic agentic capabilities including:
- Multi-turn dialogue with tool/API calls
- Task completion in constrained environments
- Real-world scenario handling (booking, shopping, etc.)

Usage:
    # Use defaults (weighted z-score method)
    python scripts/quality_scoring/compute_agentic_score.py
    
    # Use Bayesian latent factor model
    python scripts/quality_scoring/compute_agentic_score.py --bayesian
    
    # Custom benchmarks
    python scripts/quality_scoring/compute_agentic_score.py --benchmarks tau2:1:0.7 terminalbench_hard:1:0.3
    
    # Dry run (don't update cache)
    python scripts/quality_scoring/compute_agentic_score.py --dry-run

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
    BenchmarkConfig,
    extract_benchmark_matrix,
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

# Define Agentic Execution Benchmarks
AGENTIC_BENCHMARKS = BenchmarkSuite(
    name="Agentic Execution",
    description="Composite score for multi-turn agent tasks and tool use",
    score_prefix="cae",
    benchmarks={
        "tau2": BenchmarkConfig(
            name="tau2",
            description="TAU-bench 2.0: Multi-turn agent tasks (retail, airline scenarios)",
            scale=1.0,
            invert=False,
            weight=0.7,
            is_auxiliary=False
        ),
        "terminalbench_hard": BenchmarkConfig(
            name="terminalbench_hard",
            description="TerminalBench-Hard: Terminal command recovery tasks",
            scale=1.0,
            invert=False,
            weight=0.3,
            is_auxiliary=False
        ),
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Composite Agentic Execution Score (CAE).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default benchmarks (weighted z-score)
  python compute_agentic_score.py
  
  # Use Bayesian latent factor model
  python compute_agentic_score.py --bayesian
  
  # Custom benchmarks (format: field:scale:weight)
  python compute_agentic_score.py --benchmarks tau2:1:0.8 terminalbench_hard:1:0.2
  
  # Dry run (don't update cache)
  python compute_agentic_score.py --dry-run
  
  # Save config for later
  python compute_agentic_score.py --save-config agentic_config.json
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
        "--target_accept", type=float, default=0.9,
        help="Target acceptance rate for NUTS sampler (Bayesian only)."
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--min-benchmarks", type=int, default=1,
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
        help="Override the score prefix (default: 'cae')."
    )
    return parser.parse_args()


def load_models_cache():
    """Load models from the cache file, extracting from raw_data.evaluations."""
    with open(CACHE_PATH) as f:
        data = json.load(f)
    models = data.get('models', data) if isinstance(data, dict) else data
    
    # Extract tau2 and terminalbench_hard from raw_data.evaluations to top level
    print("\nExtracting agentic benchmarks from raw_data.evaluations...")
    extracted_count = 0
    for model in models:
        raw_data = model.get('raw_data', {})
        if isinstance(raw_data, dict):
            evaluations = raw_data.get('evaluations', {})
            if isinstance(evaluations, dict):
                # Extract tau2
                if 'tau2' in evaluations and 'tau2' not in model:
                    model['tau2'] = evaluations['tau2']
                    extracted_count += 1
                # Extract terminalbench_hard
                if 'terminalbench_hard' in evaluations and 'terminalbench_hard' not in model:
                    model['terminalbench_hard'] = evaluations['terminalbench_hard']
    
    if extracted_count > 0:
        print(f"  Extracted benchmarks for {extracted_count} models from raw_data")
    
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
        print("Default Agentic Execution Benchmarks (CAE):")
        print_benchmark_suite(AGENTIC_BENCHMARKS)
        return
    
    # Parse benchmark configuration
    suite = parse_benchmark_args(
        benchmark_args=args.benchmarks,
        config_path=args.config,
        default_suite=AGENTIC_BENCHMARKS,
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
    print(f"COMPOSITE AGENTIC EXECUTION SCORE ({score_prefix.upper()}) COMPUTATION")
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
        print(f"  - {name}: {count} models ({100*count/len(models):.1f}%)")
    
    # Extract benchmark matrix
    print(f"\nExtracting benchmarks (min {args.min_benchmarks} required)...")
    df_scores, df_z, model_names, benchmark_names = extract_benchmark_matrix(
        models, suite.get_configs(), min_benchmarks=args.min_benchmarks
    )
    print(f"Models with sufficient data: {len(model_names)}")
    
    if len(model_names) == 0:
        print("\nERROR: No models have sufficient benchmark data.")
        print("\nTIP: Try lowering --min-benchmarks to 1")
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
        
        try:
            # Import prepare_long_data
            from llm_jury.analysis.latent_factor import prepare_long_data
            
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
            
        except ImportError as e:
            print(f"\nERROR: Bayesian method requires pymc and arviz:")
            print(f"  pip install pymc arviz")
            print(f"  Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR during Bayesian inference: {e}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to weighted z-score method...")
            method = "weighted_zscore"
            args.bayesian = False
    
    if not args.bayesian:
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
    print(f"COMPOSITE AGENTIC EXECUTION SCORE ({score_prefix.upper()}) RESULTS")
    print("=" * 60)
    print(f"\nTop 15 models (method: {method}):\n")
    
    print(f"{'Rank':<5} {'Model':<50} {f'{score_prefix.upper()} (z)':<12} {f'{score_prefix.upper()} (0-100)':<14}")
    print("-" * 90)
    
    for rank, (_, row) in enumerate(df_result.head(15).iterrows(), 1):
        model = row['model'][:49]
        score_z = f"{row[f'{score_prefix}_mean']:.3f}" if not pd.isna(row[f'{score_prefix}_mean']) else "N/A"
        score_100 = f"{row[f'{score_prefix}_100']:.1f}" if not pd.isna(row.get(f'{score_prefix}_100')) else "N/A"
        print(f"{rank:<5} {model:<50} {score_z:<12} {score_100:<14}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SCORE STATISTICS")
    print("=" * 60)
    print(f"  Models scored: {len(df_result)}")
    print(f"  Mean (z-score): {df_result[f'{score_prefix}_mean'].mean():.3f}")
    print(f"  Std (z-score): {df_result[f'{score_prefix}_mean'].std():.3f}")
    print(f"  Range (z-score): [{df_result[f'{score_prefix}_mean'].min():.3f}, {df_result[f'{score_prefix}_mean'].max():.3f}]")
    if f'{score_prefix}_100' in df_result.columns:
        print(f"  Range (0-100): [{df_result[f'{score_prefix}_100'].min():.2f}, {df_result[f'{score_prefix}_100'].max():.2f}]")
    
    # Update models cache
    if not args.dry_run:
        print("\nUpdating models cache...")
        
        # Apply updates
        cache_data, updated_count = update_models_cache(
            cache_data, df_result, score_prefix, method=method
        )
        print(f"✓ Updated {updated_count} models with {score_prefix} scores")
        
        # Save backup
        backup_path = CACHE_PATH.parent / f"models_cache_backup_{score_prefix}.json"
        import shutil
        shutil.copy(CACHE_PATH, backup_path)
        print(f"✓ Backup saved to {backup_path}")
        
        # Write updated cache
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Wrote updated cache to {CACHE_PATH}")
    else:
        print("\n⚠ DRY RUN: Cache not updated")
    
    print("\n" + "=" * 60)
    print(f"✓ {score_prefix.upper()} COMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
