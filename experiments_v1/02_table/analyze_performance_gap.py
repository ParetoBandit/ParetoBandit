#!/usr/bin/env python3
"""
Analysis script for Table 2: The Performance Gap
Compares η=1.0 (aggressive) vs η=0.1 (baseline) performance
Demonstrates 1.26x near-optimal regret on real-world data.
"""

import json
import os
from pathlib import Path


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_metrics(data):
    """Calculate key performance metrics."""
    warmup = data.get('Warmup', {})
    tabula_rasa = data.get('Tabula Rasa', {})
    hybrid = data.get('Hybrid (Corralling)', {})
    
    # Extract regret values
    warmup_regret = warmup.get('cumulative_regret', 0)
    tr_regret = tabula_rasa.get('cumulative_regret', 0)
    hybrid_regret = hybrid.get('cumulative_regret', 0)
    
    # Calculate improvement metrics
    improvement_vs_warmup = ((warmup_regret - hybrid_regret) / warmup_regret) * 100
    gap_vs_optimal = ((hybrid_regret - tr_regret) / tr_regret) * 100
    multiplier_vs_optimal = hybrid_regret / tr_regret if tr_regret > 0 else 0
    
    # Model usage
    hybrid_usage = hybrid.get('model_usage', {})
    tr_usage = tabula_rasa.get('model_usage', {})
    
    gpt4_hybrid = hybrid_usage.get('openai/gpt-4-turbo', 0)
    gpt4_tr = tr_usage.get('openai/gpt-4-turbo', 0)
    total_samples = hybrid.get('total_samples', 1)
    
    return {
        'warmup_regret': warmup_regret,
        'tr_regret': tr_regret,
        'hybrid_regret': hybrid_regret,
        'improvement_vs_warmup': improvement_vs_warmup,
        'gap_vs_optimal': gap_vs_optimal,
        'multiplier_vs_optimal': multiplier_vs_optimal,
        'gpt4_usage_hybrid_pct': (gpt4_hybrid / total_samples) * 100,
        'gpt4_usage_tr_pct': (gpt4_tr / total_samples) * 100,
        'total_samples': total_samples
    }


def generate_comparison_table(eta_01_metrics, eta_10_metrics):
    """Generate comparison table showing the performance gap."""
    
    print("=" * 80)
    print("TABLE 2: THE PERFORMANCE GAP")
    print("η=1.0 (Aggressive) vs η=0.1 (Conservative) Baseline")
    print("=" * 80)
    print()
    
    # Main results table
    print("PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<40} {'η=0.1':<15} {'η=1.0':<15} {'Change':<10}")
    print("-" * 80)
    
    # Regret metrics
    print(f"{'Cumulative Regret':<40} "
          f"{eta_01_metrics['hybrid_regret']:<15.1f} "
          f"{eta_10_metrics['hybrid_regret']:<15.1f} "
          f"{eta_10_metrics['hybrid_regret'] - eta_01_metrics['hybrid_regret']:>9.1f}")
    
    print(f"{'vs Optimal (multiplier)':<40} "
          f"{eta_01_metrics['multiplier_vs_optimal']:<15.2f}× "
          f"{eta_10_metrics['multiplier_vs_optimal']:<15.2f}× "
          f"{eta_10_metrics['multiplier_vs_optimal'] - eta_01_metrics['multiplier_vs_optimal']:>9.2f}×")
    
    print(f"{'vs Optimal (% gap)':<40} "
          f"{eta_01_metrics['gap_vs_optimal']:<15.1f}% "
          f"{eta_10_metrics['gap_vs_optimal']:<15.1f}% "
          f"{eta_10_metrics['gap_vs_optimal'] - eta_01_metrics['gap_vs_optimal']:>9.1f}pp")
    
    print(f"{'Improvement vs Warmup':<40} "
          f"{eta_01_metrics['improvement_vs_warmup']:<15.1f}% "
          f"{eta_10_metrics['improvement_vs_warmup']:<15.1f}% "
          f"{eta_10_metrics['improvement_vs_warmup'] - eta_01_metrics['improvement_vs_warmup']:>9.1f}pp")
    
    print()
    print("MODEL USAGE (GPT-4-Turbo %)")
    print("-" * 80)
    print(f"{'Strategy':<40} {'η=0.1':<15} {'η=1.0':<15} {'Optimal':<10}")
    print("-" * 80)
    print(f"{'Hybrid (Corralling)':<40} "
          f"{eta_01_metrics['gpt4_usage_hybrid_pct']:<15.1f} "
          f"{eta_10_metrics['gpt4_usage_hybrid_pct']:<15.1f} "
          f"{eta_10_metrics['gpt4_usage_tr_pct']:<10.1f}")
    print()
    
    # Key insights
    print("KEY INSIGHTS")
    print("-" * 80)
    
    improvement_pct = ((eta_01_metrics['hybrid_regret'] - eta_10_metrics['hybrid_regret']) / 
                      eta_01_metrics['hybrid_regret']) * 100
    
    print(f"1. η=1.0 achieves {eta_10_metrics['multiplier_vs_optimal']:.2f}× near-optimal regret")
    print(f"   (only {eta_10_metrics['gap_vs_optimal']:.1f}% worse than oracle)")
    print()
    print(f"2. {improvement_pct:.1f}% better than conservative baseline (η=0.1)")
    print(f"   Reduces regret from {eta_01_metrics['hybrid_regret']:.0f} → {eta_10_metrics['hybrid_regret']:.0f}")
    print()
    print(f"3. {eta_10_metrics['improvement_vs_warmup']:.1f}% improvement over harmful warmup priors")
    print(f"   Prevents catastrophic failure (warmup regret: {eta_10_metrics['warmup_regret']:.0f})")
    print()
    print(f"4. Near-optimal model selection: {eta_10_metrics['gpt4_usage_hybrid_pct']:.1f}% GPT-4 usage")
    print(f"   vs optimal {eta_10_metrics['gpt4_usage_tr_pct']:.1f}%")
    print()
    print("=" * 80)
    print()


def generate_latex_data(eta_01_metrics, eta_10_metrics):
    """Generate LaTeX-ready data for Table 2."""
    
    latex_data = {
        'eta_01': {
            'regret': eta_01_metrics['hybrid_regret'],
            'vs_optimal_mult': eta_01_metrics['multiplier_vs_optimal'],
            'vs_optimal_pct': eta_01_metrics['gap_vs_optimal'],
            'improvement_warmup': eta_01_metrics['improvement_vs_warmup'],
            'gpt4_usage': eta_01_metrics['gpt4_usage_hybrid_pct']
        },
        'eta_10': {
            'regret': eta_10_metrics['hybrid_regret'],
            'vs_optimal_mult': eta_10_metrics['multiplier_vs_optimal'],
            'vs_optimal_pct': eta_10_metrics['gap_vs_optimal'],
            'improvement_warmup': eta_10_metrics['improvement_vs_warmup'],
            'gpt4_usage': eta_10_metrics['gpt4_usage_hybrid_pct']
        },
        'optimal': {
            'regret': eta_10_metrics['tr_regret'],
            'gpt4_usage': eta_10_metrics['gpt4_usage_tr_pct']
        },
        'warmup': {
            'regret': eta_10_metrics['warmup_regret']
        }
    }
    
    # Save to JSON
    output_path = Path(__file__).parent / 'data' / 'performance_gap_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(latex_data, f, indent=2)
    
    print(f"✓ Saved LaTeX data to {output_path}")
    
    return latex_data


def main():
    """Main analysis function."""
    script_dir = Path(__file__).parent
    
    # Load results from both learning rates
    eta_01_path = script_dir / 'data' / 'results.json'  # η=0.1 baseline
    eta_10_path = script_dir / 'data' / 'eta_1.0' / 'results.json'  # η=1.0 aggressive
    
    print("\nLoading results...")
    print(f"  η=0.1: {eta_01_path}")
    print(f"  η=1.0: {eta_10_path}")
    print()
    
    eta_01_data = load_results(eta_01_path)
    eta_10_data = load_results(eta_10_path)
    
    # Calculate metrics
    print("Calculating metrics...")
    eta_01_metrics = calculate_metrics(eta_01_data)
    eta_10_metrics = calculate_metrics(eta_10_data)
    print()
    
    # Generate comparison table
    generate_comparison_table(eta_01_metrics, eta_10_metrics)
    
    # Generate LaTeX data
    latex_data = generate_latex_data(eta_01_metrics, eta_10_metrics)
    
    print("\n✓ Analysis complete!")
    print(f"  Total samples: {eta_10_metrics['total_samples']}")
    print(f"  Key finding: {latex_data['eta_10']['vs_optimal_mult']:.2f}× near-optimal regret with η=1.0")
    print()


if __name__ == '__main__':
    main()

