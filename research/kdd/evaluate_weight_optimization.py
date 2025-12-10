#!/usr/bin/env python3
"""
Evaluate Lagrangian Trust Weight Optimization.

This script demonstrates that the trust_score weight in quality scoring
is a shadow price of the hallucination constraint.

Key insight:
    Quality(m) = (1 - λ) × IntentScore(m) + λ × TrustScore(m)
    
    Where λ is the shadow price of the constraint:
    "Top-k models must have avg hallucination ≤ H_max"

Usage:
    python evaluate_weight_optimization.py
    python evaluate_weight_optimization.py --hallucination 5.0
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.optimization.lagrangian_weights import (
    LagrangianWeightOptimizer,
    Intent,
    INTENT_BASE_WEIGHTS,
)


def load_models(cache_path: str = "data/models_cache.json"):
    """Load models from cache."""
    with open(cache_path) as f:
        data = json.load(f)
    return data.get("models", data)


def plot_trust_weights(results: Dict, constraint: float, output_path: str):
    """Plot trust weights (λ) across intents."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        intents = list(results.keys())
        lambdas = [results[i].lambda_safety for i in intents]
        achieved_halluc = [results[i].achieved_hallucination for i in intents]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Trust weights by intent
        x = np.arange(len(intents))
        bars = ax1.bar(x, lambdas, color='steelblue', alpha=0.8)
        
        ax1.set_ylabel('Trust Weight (λ)', fontsize=12)
        ax1.set_xlabel('Intent Category', fontsize=12)
        ax1.set_title(f'Shadow Price of Safety Constraint\n(H_max = {constraint}%)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([i.upper() for i in intents], rotation=45, ha='right')
        ax1.set_ylim(0, max(lambdas) * 1.2 if max(lambdas) > 0 else 0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, lambdas):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Achieved hallucination vs constraint
        ax2.bar(x, achieved_halluc, color='coral', alpha=0.8, label='Achieved')
        ax2.axhline(y=constraint, color='red', linestyle='--', linewidth=2, label=f'Constraint ({constraint}%)')
        
        ax2.set_ylabel('Hallucination Rate (%)', fontsize=12)
        ax2.set_xlabel('Intent Category', fontsize=12)
        ax2.set_title('Achieved Hallucination vs Constraint', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([i.upper() for i in intents], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nTrust weights plot saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot")


def plot_lambda_vs_constraint(optimizer, output_path: str):
    """Plot how λ changes as hallucination constraint varies."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Test different constraint values
        constraints = np.linspace(3, 20, 15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for intent in [Intent.CODING, Intent.REASONING, Intent.FACTUAL_QA]:
            lambdas = []
            for h_max in constraints:
                result = optimizer.compute_trust_weight(
                    intent=intent.value,
                    hallucination_constraint=h_max,
                    verbose=False,
                )
                lambdas.append(result.lambda_safety)
            
            ax.plot(constraints, lambdas, 'o-', linewidth=2, markersize=6, label=intent.value.upper())
        
        ax.set_xlabel('Hallucination Constraint H_max (%)', fontsize=12)
        ax.set_ylabel('Trust Weight λ (Shadow Price)', fontsize=12)
        ax.set_title('Shadow Price Increases as Safety Constraint Tightens', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Lower constraint = tighter
        
        # Add annotation
        ax.annotate(
            "Stricter constraint →\nHigher trust weight",
            xy=(8, 0.15),
            fontsize=11,
            ha='center',
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Lambda vs constraint plot saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lagrangian trust weight optimization")
    parser.add_argument("--cache-path", default="data/models_cache.json")
    parser.add_argument("--hallucination", type=float, default=10.0, help="Max hallucination %")
    parser.add_argument("--intent", type=str, default=None, help="Single intent to optimize")
    parser.add_argument("--output-dir", default="kdd_paper/figures")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LAGRANGIAN TRUST WEIGHT OPTIMIZATION")
    print("=" * 70)
    print("""
We derive the trust_score weight via Lagrangian Dual optimization.

Quality Formula:
    Quality(m) = (1 - λ) × IntentScore(m) + λ × TrustScore(m)

Where:
    - IntentScore = Σ w_i × Benchmark_i  (FIXED by domain knowledge)
    - TrustScore = 100 - HallucinationRate
    - λ = shadow price of safety constraint (OPTIMIZED)

Constraint:
    Average hallucination of top-5 models ≤ H_max
""")
    
    print(f"Safety Constraint: {args.hallucination}% max hallucination")
    
    # Load models
    print("\nLoading models...")
    models = load_models(args.cache_path)
    print(f"  Loaded {len(models)} models")
    
    # Initialize optimizer
    optimizer = LagrangianWeightOptimizer(models)
    print(f"  {optimizer.n_models} models with valid benchmarks")
    
    # Determine intents
    if args.intent:
        intents = [args.intent]
    else:
        intents = [i.value for i in Intent if i in INTENT_BASE_WEIGHTS]
    
    results = {}
    
    for intent in intents:
        print(f"\n{'='*60}")
        print(f"INTENT: {intent.upper()}")
        print(f"{'='*60}")
        
        # Show base weights
        base_weights = INTENT_BASE_WEIGHTS.get(Intent(intent), {})
        print("\nBase Weights (from domain knowledge):")
        for bench, w in sorted(base_weights.items(), key=lambda x: -x[1])[:5]:
            if w > 0.01:
                print(f"  {bench:<20} {w:.2f}")
        
        # Optimize trust weight
        result = optimizer.compute_trust_weight(
            intent=intent,
            hallucination_constraint=args.hallucination,
            verbose=args.verbose,
        )
        
        results[intent] = result
        
        # Print results
        print(f"\n--- SHADOW PRICE (Trust Weight) ---")
        print(f"  λ = {result.lambda_safety:.4f}")
        print(f"\n  Interpretation:")
        print(f"    → {result.lambda_safety*100:.1f}% of quality comes from trust_score")
        print(f"    → {(1-result.lambda_safety)*100:.1f}% comes from intent benchmarks")
        
        print(f"\n--- CONSTRAINT SATISFACTION ---")
        print(f"  Target: ≤{args.hallucination:.1f}% hallucination")
        print(f"  Achieved: {result.achieved_hallucination:.1f}% hallucination")
        
        print(f"\n--- FINAL WEIGHTS (top 5) ---")
        for bench, w in sorted(result.final_weights.items(), key=lambda x: -x[1])[:5]:
            print(f"  {bench:<20} {w:.3f}")
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_trust_weights(results, args.hallucination, str(output_dir / "trust_weights_by_intent.png"))
    plot_lambda_vs_constraint(optimizer, str(output_dir / "lambda_vs_hallucination_constraint.png"))
    
    # Summary
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    
    print(f"""
METHODOLOGY:
────────────────────────────────────────────────────────────────────

"The trust_score weight in our quality formula is derived via Lagrangian 
Dual optimization. For each intent, we solve:

    Quality(m) = (1 - λ) × IntentScore(m) + λ × TrustScore(m)
    
    Where λ is the shadow price of: E[Halluc(top_k)] ≤ H_max

The shadow price λ has economic interpretation: it represents the quality-
hallucination trade-off. Higher λ means we're willing to sacrifice more
benchmark performance for lower hallucination.

KEY FINDINGS (H_max = {args.hallucination}%):
────────────────────────────────────────────────────────────────────
""")
    
    for intent, result in results.items():
        print(f"\n{intent.upper()}:")
        print(f"  {result.get_paper_statement()}")


if __name__ == "__main__":
    main()

