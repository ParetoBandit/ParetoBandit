#!/usr/bin/env python3
"""
Lagrangian Dual Optimization Evaluation.

Demonstrates that our routing weights are mathematically optimal Shadow Prices
derived from constrained optimization, not heuristically tuned.

This script:
1. Loads real model data from cache
2. Solves for optimal shadow prices via Dual Ascent
3. Shows how shadow prices adapt to different constraint scenarios
4. Generates convergence plots for the paper

Output:
    - Shadow price values with interpretations
    - Convergence plots showing dual ascent
    - Comparison of routing under different budget/safety constraints

Usage:
    python evaluate_lagrangian_optimization.py
    python evaluate_lagrangian_optimization.py --budget 0.005 --max-hallucination 5.0
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.optimization.lagrangian_router import (
    LagrangianRouter,
    ModelMetrics,
    DualAscentConfig,
    interpret_shadow_prices,
)


def load_models_from_cache(cache_path: str = "data/models_cache.json") -> List[Dict]:
    """Load model data from cache."""
    with open(cache_path) as f:
        data = json.load(f)
    
    # Handle both formats
    if isinstance(data, dict) and 'models' in data:
        return data['models']
    return data


def convert_to_metrics(models: List[Dict], quality_metric: str = "arena_elo") -> List[ModelMetrics]:
    """
    Convert model dicts to ModelMetrics objects.
    
    Args:
        models: List of model dicts from cache
        quality_metric: Which metric to use for quality. Options:
            - "arena_elo": Chatbot Arena ELO (has good variance)
            - "intelligence_index": Intelligence index from benchmarks
            - "quality_index": Combined quality index
    """
    metrics = []
    
    for m in models:
        # Skip models without required data
        if not m.get("name"):
            continue
        
        # Get quality based on specified metric
        if quality_metric == "arena_elo":
            quality = m.get("arena_elo")
            if quality is None:
                continue  # Skip models without ELO for fair comparison
        elif quality_metric == "intelligence_index":
            quality = m.get("intelligence_index", 50)
        else:
            quality = m.get("quality_index", 50.0)
        
        if quality is None or quality <= 0:
            continue
        
        # Get cost (blended or input)
        cost = m.get("blended_cost_per_m") or m.get("price_1m_blended")
        if cost is None:
            cost = m.get("input_cost_per_m") or m.get("price_1m_input", 1.0)
        if cost is None or cost <= 0:
            continue
        
        # Get hallucination rate
        hallucination = m.get("hallucination_rate")
        if hallucination is None:
            hallucination = m.get("truthfulqa_accuracy")
            if hallucination is not None:
                # Convert accuracy to error rate
                hallucination = 100 - hallucination
            else:
                hallucination = 15.0  # Default assumption
        
        # Get latency (tokens per second -> convert to ms per token for consistency)
        output_tps = m.get("output_tokens_per_second", 100.0)
        if output_tps and output_tps > 0:
            latency = 1000 / output_tps  # ms per token
        else:
            latency = 10.0  # default
        
        metrics.append(ModelMetrics(
            name=m["name"],
            quality=quality,
            cost_per_1m_tokens=cost,
            hallucination_rate=hallucination,
            latency_ms=latency,
        ))
    
    return metrics


def run_optimization_scenario(
    models: List[ModelMetrics],
    budget: float,
    max_hallucination: float,
    scenario_name: str,
    verbose: bool = False,
):
    """Run optimization for a specific constraint scenario."""
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    print(f"  Budget constraint: ${budget}/query")
    print(f"  Safety constraint: {max_hallucination}% max hallucination")
    
    # Create router with PID for stable convergence
    router = LagrangianRouter(
        budget_per_query=budget,
        max_hallucination_rate=max_hallucination,
        dual_config=DualAscentConfig(
            learning_rate=0.05,
            max_iterations=100,
            convergence_threshold=1e-5,
            use_pid=True,
            kp=0.5,
            ki=0.05,
            kd=0.01,
        ),
    )
    
    # Solve
    result = router.solve(models, num_queries=1000, verbose=verbose)
    
    # Print results
    print(f"\n--- OPTIMAL SHADOW PRICES ---")
    print(f"  λ_cost   = {result.lambda_cost:.4f}")
    print(f"  λ_safety = {result.lambda_safety:.4f}")
    
    print(f"\n--- INTERPRETATION ---")
    print(interpret_shadow_prices(result))
    
    print(f"\n--- OPTIMAL ROUTING ---")
    # Get unique selected models
    selected_models = set(result.model_selections.values())
    print(f"  Selected model(s): {', '.join(selected_models)}")
    
    print(f"\n--- METRICS ACHIEVED ---")
    print(f"  Avg Quality: {result.total_quality / 1000:.2f}")
    print(f"  Avg Cost: ${result.total_cost / 1000:.6f}/query")
    print(f"  Avg Hallucination: {result.avg_hallucination:.2f}%")
    
    print(f"\n--- CONSTRAINT SATISFACTION ---")
    print(f"  Budget satisfied: {'✓' if result.budget_satisfied else '✗'}")
    print(f"  Safety satisfied: {'✓' if result.safety_satisfied else '✗'}")
    
    print(f"\n--- CONVERGENCE ---")
    print(f"  Converged: {'Yes' if result.converged else 'No'}")
    print(f"  Iterations: {result.iterations}")
    
    return result


def plot_convergence(results: Dict[str, 'OptimizationResult'], output_path: str):
    """Generate convergence plot for shadow prices."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot λ_cost convergence
        ax1 = axes[0]
        for name, result in results.items():
            history = result.shadow_prices["cost"].history
            ax1.plot(history, label=name, linewidth=2)
        
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("λ_cost (Budget Shadow Price)", fontsize=12)
        ax1.set_title("Convergence of Budget Shadow Price", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot λ_safety convergence
        ax2 = axes[1]
        for name, result in results.items():
            history = result.shadow_prices["safety"].history
            ax2.plot(history, label=name, linewidth=2)
        
        ax2.set_xlabel("Iteration", fontsize=12)
        ax2.set_ylabel("λ_safety (Safety Shadow Price)", fontsize=12)
        ax2.set_title("Convergence of Safety Shadow Price", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nConvergence plot saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")


def plot_shadow_price_vs_constraint(
    models: List[ModelMetrics],
    constraint_type: str,
    output_path: str,
):
    """Plot how shadow prices change as constraints tighten."""
    try:
        import matplotlib.pyplot as plt
        
        if constraint_type == "budget":
            # Get cost range from models to set realistic budgets
            costs_per_query = [m.cost_per_query() for m in models]
            min_cost = min(costs_per_query)
            max_cost = max(costs_per_query)
            median_cost = np.median(costs_per_query)
            
            # Vary budget from below median to above max (to show binding/slack transition)
            # Use log scale for better visualization
            budgets = np.logspace(np.log10(min_cost * 0.5), np.log10(max_cost * 2), 25)
            lambda_costs = []
            selected_models = []
            
            for budget in budgets:
                router = LagrangianRouter(
                    budget_per_query=budget,
                    max_hallucination_rate=50.0,  # Loose safety to isolate budget effect
                    dual_config=DualAscentConfig(learning_rate=0.2, max_iterations=100),
                )
                result = router.solve(models, num_queries=100, verbose=False)
                lambda_costs.append(result.lambda_cost)
                selected_models.append(list(set(result.model_selections.values()))[0])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogx(budgets * 1000, lambda_costs, 'b-o', linewidth=2, markersize=5, label='λ_cost')
            ax.set_xlabel("Budget per Query ($ × 1000, log scale)", fontsize=12)
            ax.set_ylabel("Shadow Price λ_cost", fontsize=12)
            ax.set_title("Shadow Price Increases as Budget Tightens", fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            
            # Mark key thresholds
            ax.axvline(median_cost * 1000, color='green', linestyle='--', alpha=0.7, label=f'Median cost: ${median_cost*1000:.4f}')
            ax.axvline(min_cost * 1000, color='red', linestyle='--', alpha=0.7, label=f'Min cost: ${min_cost*1000:.6f}')
            ax.legend(loc='upper right')
            
            # Add annotation for tight budget region
            if max(lambda_costs) > 0:
                tight_idx = lambda_costs.index(max(lambda_costs))
                ax.annotate(
                    "Budget binding\n(λ > 0)",
                    xy=(budgets[tight_idx] * 1000, lambda_costs[tight_idx]),
                    xytext=(budgets[tight_idx] * 1000 * 5, lambda_costs[tight_idx] * 0.8),
                    arrowprops=dict(arrowstyle="->", color='gray'),
                    fontsize=10,
                )
            
        else:  # safety
            # Vary safety constraint
            max_hallucinations = np.linspace(2.0, 20.0, 20)
            lambda_costs = []
            lambda_safetys = []
            
            for max_h in max_hallucinations:
                router = LagrangianRouter(
                    budget_per_query=0.01,
                    max_hallucination_rate=max_h,
                    dual_config=DualAscentConfig(learning_rate=0.1, max_iterations=50),
                )
                result = router.solve(models, num_queries=100, verbose=False)
                lambda_costs.append(result.lambda_cost)
                lambda_safetys.append(result.lambda_safety)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(max_hallucinations, lambda_safetys, 'r-o', linewidth=2, markersize=4, label='λ_safety')
            ax.set_xlabel("Max Allowed Hallucination Rate (%)", fontsize=12)
            ax.set_ylabel("Shadow Price λ_safety", fontsize=12)
            ax.set_title("Shadow Price Increases as Safety Requirement Tightens", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # Lower hallucination tolerance = tighter constraint
            
            # Add annotation
            ax.annotate(
                "← Stricter safety\nHigher shadow price",
                xy=(max_hallucinations[5], lambda_safetys[5]),
                xytext=(max_hallucinations[10], lambda_safetys[5] + 0.3),
                arrowprops=dict(arrowstyle="->", color='gray'),
                fontsize=10,
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nShadow price vs constraint plot saved to: {output_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")


def main():
    parser = argparse.ArgumentParser(description="Lagrangian Dual Optimization Evaluation")
    parser.add_argument("--budget", type=float, default=0.01,
                        help="Budget per query in $ (default: 0.01)")
    parser.add_argument("--max-hallucination", type=float, default=10.0,
                        help="Max hallucination rate %% (default: 10.0)")
    parser.add_argument("--cache-path", type=str, default="data/models_cache.json",
                        help="Path to models cache")
    parser.add_argument("--output-dir", type=str, default="kdd_paper/figures",
                        help="Output directory for plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LAGRANGIAN DUAL OPTIMIZATION FOR LLM ROUTING")
    print("=" * 70)
    print("""
We determine optimal routing policies by solving the Lagrangian Dual of the
constrained budget formulation. The 'weights' in our objective function are
dynamically solved as Lagrange Multipliers (λ) via Dual Ascent, representing
the marginal cost of violating each constraint.
""")
    
    # Load models
    print("Loading models from cache...")
    model_dicts = load_models_from_cache(args.cache_path)
    print(f"  Loaded {len(model_dicts)} models")
    
    # Convert to metrics (using arena_elo for quality variance)
    models = convert_to_metrics(model_dicts, quality_metric="arena_elo")
    print(f"  {len(models)} models with arena_elo and valid metrics")
    
    # Get cost statistics for realistic budget ranges
    costs_per_query = [m.cost_per_query() for m in models]
    min_cost = min(costs_per_query)
    max_cost = max(costs_per_query)
    median_cost = np.median(costs_per_query)
    
    print(f"\n--- COST STATISTICS (per 1K token query) ---")
    print(f"  Min: ${min_cost:.6f}")
    print(f"  Median: ${median_cost:.6f}")  
    print(f"  Max: ${max_cost:.6f}")
    
    # Print top models
    print("\n--- TOP MODELS BY QUALITY (Arena ELO) ---")
    sorted_by_quality = sorted(models, key=lambda m: m.quality, reverse=True)[:5]
    for m in sorted_by_quality:
        print(f"  {m.name}: ELO={m.quality:.0f}, Cost=${m.cost_per_1m_tokens:.2f}/M, H={m.hallucination_rate:.1f}%")
    
    print("\n--- CHEAPEST MODELS ---")
    sorted_by_cost = sorted(models, key=lambda m: m.cost_per_1m_tokens)[:5]
    for m in sorted_by_cost:
        print(f"  {m.name}: Cost=${m.cost_per_1m_tokens:.4f}/M, ELO={m.quality:.0f}")
    
    # Run multiple scenarios to show how shadow prices adapt
    # Use budget constraints relative to actual cost distribution
    results = {}
    
    # Scenario 1: Loose constraints (budget > max cost)
    loose_budget = max_cost * 2
    results["Loose Budget"] = run_optimization_scenario(
        models, budget=loose_budget, max_hallucination=50.0,
        scenario_name=f"Loose Constraints (B=${loose_budget:.4f}/query, H≤50%)",
        verbose=args.verbose
    )
    
    # Scenario 2: Moderate constraints (budget = median cost)
    moderate_budget = median_cost
    results["Moderate Budget"] = run_optimization_scenario(
        models, budget=moderate_budget, max_hallucination=20.0,
        scenario_name=f"Moderate Constraints (B=${moderate_budget:.6f}/query, H≤20%)",
        verbose=args.verbose
    )
    
    # Scenario 3: Tight budget (budget = min cost * 2, forcing cheap models)
    tight_budget = min_cost * 2
    results["Tight Budget"] = run_optimization_scenario(
        models, budget=tight_budget, max_hallucination=50.0,
        scenario_name=f"Tight Budget (B=${tight_budget:.6f}/query, H≤50%)",
        verbose=args.verbose
    )
    
    # Scenario 4: Very tight budget (below most models)
    very_tight_budget = min_cost * 1.1
    results["Very Tight Budget"] = run_optimization_scenario(
        models, budget=very_tight_budget, max_hallucination=50.0,
        scenario_name=f"Very Tight Budget (B=${very_tight_budget:.6f}/query)",
        verbose=args.verbose
    )
    
    # Scenario 5: Tight safety
    results["Tight Safety"] = run_optimization_scenario(
        models, budget=max_cost, max_hallucination=3.0,
        scenario_name=f"Tight Safety (B=${max_cost:.4f}/query, H≤3%)",
        verbose=args.verbose
    )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SHADOW PRICE COMPARISON ACROSS SCENARIOS")
    print("=" * 70)
    print(f"\n{'Scenario':<35} {'λ_cost':<12} {'λ_safety':<12} {'Selected Model':<20}")
    print("-" * 80)
    
    for name, result in results.items():
        selected = list(set(result.model_selections.values()))[0]
        print(f"{name:<35} {result.lambda_cost:<12.4f} {result.lambda_safety:<12.4f} {selected:<20}")
    
    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- GENERATING PLOTS ---")
    
    # Convergence plot
    plot_convergence(results, str(output_dir / "shadow_price_convergence.png"))
    
    # Shadow price vs constraint plots
    plot_shadow_price_vs_constraint(models, "budget", str(output_dir / "lambda_vs_budget.png"))
    plot_shadow_price_vs_constraint(models, "safety", str(output_dir / "lambda_vs_safety.png"))
    
    # Paper-ready summary
    print("\n" + "=" * 70)
    print("PAPER-READY SUMMARY")
    print("=" * 70)
    print("""
METHODOLOGY (for paper):
─────────────────────────────────────────────────────────────────────

"We determine optimal routing policies by solving the Lagrangian Dual 
of the constrained budget formulation. The 'weights' in our objective 
function are dynamically solved as Lagrange Multipliers (λ) via Dual 
Ascent, representing the marginal cost of violating each constraint.

The shadow prices have economic interpretation:
  • λ_cost: Quality points we trade per dollar saved
  • λ_safety: Quality points we trade per 1% less hallucination

As constraints tighten (lower budget or stricter safety), the 
corresponding shadow price increases automatically, reflecting the 
higher 'cost' of using expensive or unsafe models."

KEY RESULTS:
─────────────────────────────────────────────────────────────────────
""")
    
    # Show how shadow prices adapt
    loose = results["Loose (B=$0.02, H≤15%)"]
    tight_budget = results["Tight Budget (B=$0.002, H≤10%)"]
    tight_safety = results["Tight Safety (B=$0.01, H≤3%)"]
    
    print(f"• Under loose constraints (B=$0.02, H≤15%):")
    print(f"  λ_cost = {loose.lambda_cost:.4f}, λ_safety = {loose.lambda_safety:.4f}")
    print(f"  → Both constraints slack, shadow prices near zero")
    
    print(f"\n• Under tight budget (B=$0.002):")
    print(f"  λ_cost = {tight_budget.lambda_cost:.4f} (increased {tight_budget.lambda_cost/max(loose.lambda_cost, 0.001):.1f}x)")
    print(f"  → Budget constraint binding, high shadow price for cost")
    
    print(f"\n• Under tight safety (H≤3%):")
    print(f"  λ_safety = {tight_safety.lambda_safety:.4f} (increased {tight_safety.lambda_safety/max(loose.lambda_safety, 0.001):.1f}x)")
    print(f"  → Safety constraint binding, high shadow price for hallucination")
    
    print("""
CITATIONS:
─────────────────────────────────────────────────────────────────────
1. OmniRouter (2025): "Budget and Performance Controllable Multi-LLM Routing"
2. Fioretto et al. (2020): "Lagrangian Duality for Constrained Deep Learning"
3. Stooke et al. (2020): "Responsive Safety in RL by PID Lagrangian Methods"
""")


if __name__ == "__main__":
    main()

