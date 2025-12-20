#!/usr/bin/env python3
"""
Pareto Frontier Experiment: Finding the Optimal Quality-Cost Trade-off

This script explores the efficient frontier by varying the cost penalty (lambda_cost).
It demonstrates that the router can operate at different points on the quality-cost
spectrum, from "Max Quality" to "Cost Saver" mode.

Key Insight:
- Low lambda_cost → Router prioritizes accuracy (routes to GPT-4o)
- High lambda_cost → Router prioritizes cost savings (routes to cheaper models)
- The "sweet spot" achieves near-GPT-4o accuracy at near-DeepSeek cost

Prerequisites:
    pip install datasets numpy pandas matplotlib sentence-transformers

Usage:
    # First, run calibration
    python kdd_paper/scripts/calibrate_multi_domain.py
    
    # Then, run Pareto experiment
    python kdd_paper/scripts/run_pareto_experiment.py

Output:
    - results/figures/pareto_frontier.pdf - The efficient frontier plot
    - results/experiment/pareto_results.json - Raw data
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.core.bandit_router import BanditRouter

# Import from calibration script
from kdd_paper.scripts.calibrate_multi_domain import (
    BenchmarkLoader,
    Verifier,
    mock_generate,
    MODEL_REGISTRY,
)


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

PRIORS_PATH = Path("results/multi_domain/multi_domain_priors.npz")
OUTPUT_DIR = Path("results/figures")
TEST_SET_SIZE = 100  # Samples per domain

# Lambda values to explore (cost penalty in "quality points per dollar")
LAMBDA_VALUES = [0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]

# Prior strength values to test
PRIOR_STRENGTH_VALUES = [10.0, 50.0, 100.0, 200.0]


@dataclass
class EvalResult:
    """Result from a single evaluation run."""
    lambda_cost: float
    prior_strength: float
    accuracy: float
    avg_cost: float
    model_distribution: Dict[str, float]
    domain_accuracy: Dict[str, float]
    n_samples: int


# ==============================================================================
# 2. EVALUATION FUNCTION
# ==============================================================================

def run_evaluation(
    dataset: List[Dict[str, Any]],
    router: BanditRouter,
    lambda_cost: float,
) -> EvalResult:
    """
    Run evaluation with a specific lambda_cost value.
    """
    model_counts: Dict[str, int] = {}
    domain_correct: Dict[str, List[float]] = {}
    total_cost = 0.0
    total_correct = 0
    
    for task in dataset:
        domain = task['domain']
        
        # Route with specific lambda_cost
        model_id, log = router.route(
            task['prompt'],
            exploration="safe",
            lambda_cost=lambda_cost,
        )
        
        # Track model selection
        model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        # Generate and verify
        output = mock_generate(model_id, domain, task['ground_truth'])
        is_correct = Verifier.check(domain, output, task['ground_truth'])
        
        # Track results
        if domain not in domain_correct:
            domain_correct[domain] = []
        domain_correct[domain].append(float(is_correct))
        total_correct += is_correct
        
        # Calculate cost
        cost = MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
        total_cost += cost
    
    # Compute metrics
    n = len(dataset)
    accuracy = total_correct / n if n else 0
    avg_cost = total_cost / n if n else 0
    
    model_distribution = {
        model: count / n for model, count in model_counts.items()
    }
    
    domain_accuracy = {
        domain: sum(results) / len(results) if results else 0
        for domain, results in domain_correct.items()
    }
    
    return EvalResult(
        lambda_cost=lambda_cost,
        prior_strength=0.0,  # Set by caller
        accuracy=accuracy,
        avg_cost=avg_cost,
        model_distribution=model_distribution,
        domain_accuracy=domain_accuracy,
        n_samples=n,
    )


def run_baseline(
    dataset: List[Dict[str, Any]],
    model_id: str,
) -> Tuple[float, float]:
    """Run a static baseline and return (accuracy, cost)."""
    total_correct = 0
    total_cost = 0.0
    
    for task in dataset:
        output = mock_generate(model_id, task['domain'], task['ground_truth'])
        is_correct = Verifier.check(task['domain'], output, task['ground_truth'])
        total_correct += is_correct
        total_cost += MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
    
    n = len(dataset)
    return total_correct / n, total_cost / n


# ==============================================================================
# 3. PARETO FRONTIER PLOTTING
# ==============================================================================

def plot_pareto_frontier(
    results: List[EvalResult],
    baselines: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    """
    Generate the Pareto Frontier plot showing accuracy vs cost trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract router data points
    costs = [r.avg_cost for r in results]
    accs = [r.accuracy for r in results]
    lambdas = [r.lambda_cost for r in results]
    
    # Plot router frontier
    # Sort by cost for connected line
    sorted_indices = np.argsort(costs)
    sorted_costs = [costs[i] for i in sorted_indices]
    sorted_accs = [accs[i] for i in sorted_indices]
    sorted_lambdas = [lambdas[i] for i in sorted_indices]
    
    ax.plot(sorted_costs, sorted_accs, 'go-', linewidth=2, markersize=8,
            label='Router Efficient Frontier', zorder=3)
    
    # Annotate key points with lambda values
    for i, (cost, acc, lam) in enumerate(zip(sorted_costs, sorted_accs, sorted_lambdas)):
        if lam in [0.0, 1.0, 10.0, 50.0, 200.0]:
            ax.annotate(
                f'λ={lam}',
                xy=(cost, acc),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
            )
    
    # Plot baselines
    baseline_colors = {'GPT-4o': 'red', 'DeepSeek': 'blue', 'Nova-Lite': 'orange'}
    baseline_markers = {'GPT-4o': 's', 'DeepSeek': '^', 'Nova-Lite': 'd'}
    
    for name, (acc, cost) in baselines.items():
        ax.scatter(
            cost, acc,
            marker=baseline_markers.get(name, 'x'),
            s=150,
            c=baseline_colors.get(name, 'gray'),
            label=f'{name} (Static)',
            zorder=4,
            edgecolors='black',
            linewidths=1,
        )
    
    # Find and highlight the "sweet spot" (best accuracy/cost ratio)
    ratios = [acc / cost if cost > 0 else 0 for acc, cost in zip(accs, costs)]
    best_idx = np.argmax(ratios)
    best_result = results[best_idx]
    
    ax.scatter(
        best_result.avg_cost, best_result.accuracy,
        marker='*', s=300, c='gold', edgecolors='black',
        label=f'Sweet Spot (λ={best_result.lambda_cost})',
        zorder=5,
    )
    
    # Formatting
    ax.set_xlabel('Average Cost per Query ($)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Pareto Frontier: Quality vs Cost Trade-off\n(Router can operate at any point on the green line)', 
                 fontsize=11)
    ax.set_xscale('log')
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    
    # Add annotations for regions
    ax.annotate(
        'Quality\nPriority',
        xy=(0.003, 0.92),
        fontsize=10,
        ha='center',
        color='darkgreen',
        alpha=0.7,
    )
    ax.annotate(
        'Cost\nPriority',
        xy=(0.0003, 0.75),
        fontsize=10,
        ha='center',
        color='darkgreen',
        alpha=0.7,
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_model_distribution(
    results: List[EvalResult],
    output_path: Path,
) -> None:
    """
    Show how model selection changes with lambda_cost.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all unique models
    all_models = set()
    for r in results:
        all_models.update(r.model_distribution.keys())
    models = sorted(all_models)
    
    # Prepare data
    lambdas = [r.lambda_cost for r in results]
    x = np.arange(len(lambdas))
    width = 0.8
    
    # Stack bars
    bottom = np.zeros(len(lambdas))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for model, color in zip(models, colors):
        heights = [r.model_distribution.get(model, 0) * 100 for r in results]
        ax.bar(x, heights, width, bottom=bottom, label=model.split('/')[-1], color=color)
        bottom += heights
    
    ax.set_xlabel('Lambda (Cost Penalty)', fontsize=12)
    ax.set_ylabel('Model Selection (%)', fontsize=12)
    ax.set_title('How Cost Penalty Changes Model Selection', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'λ={l}' for l in lambdas], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_lambda(
    results: List[EvalResult],
    baselines: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> None:
    """
    Show accuracy degradation as lambda increases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    lambdas = [r.lambda_cost for r in results]
    accs = [r.accuracy for r in results]
    costs = [r.avg_cost for r in results]
    
    # Panel A: Accuracy vs Lambda
    ax1.plot(lambdas, accs, 'go-', linewidth=2, markersize=8, label='Router')
    
    # Add baseline lines
    for name, (acc, _) in baselines.items():
        ax1.axhline(y=acc, linestyle='--', alpha=0.7, label=f'{name} Baseline')
    
    ax1.set_xlabel('Lambda (Cost Penalty)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('(A) Accuracy vs Cost Penalty', fontsize=11)
    ax1.set_xscale('log')
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel B: Cost vs Lambda
    ax2.plot(lambdas, costs, 'bo-', linewidth=2, markersize=8, label='Router')
    
    for name, (_, cost) in baselines.items():
        ax2.axhline(y=cost, linestyle='--', alpha=0.7, label=f'{name} Baseline')
    
    ax2.set_xlabel('Lambda (Cost Penalty)', fontsize=12)
    ax2.set_ylabel('Average Cost ($)', fontsize=12)
    ax2.set_title('(B) Cost vs Cost Penalty', fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Effect of Cost Penalty on Router Behavior', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main() -> int:
    print("=" * 70)
    print(" PARETO FRONTIER EXPERIMENT")
    print("=" * 70)
    print("Finding the optimal quality-cost trade-off by varying lambda_cost")
    print("")
    print("Key Insight:")
    print("  Low λ → Maximize accuracy (routes to GPT-4o)")
    print("  High λ → Minimize cost (routes to cheaper models)")
    print("=" * 70)
    
    # 1. Load Test Data
    print("\n[1] LOADING TEST DATA...")
    loader = BenchmarkLoader()
    dataset = loader.get_calibration_batch(n=TEST_SET_SIZE)
    
    if not dataset:
        print("ERROR: No test data loaded!")
        return 1
    
    # 2. Run Baselines
    print("\n[2] RUNNING BASELINES...")
    baselines = {}
    
    for model_id, name in [
        ("openai/gpt-4o", "GPT-4o"),
        ("deepseek/deepseek-chat-v3-0324", "DeepSeek"),
        ("amazon/nova-lite-v1", "Nova-Lite"),
    ]:
        acc, cost = run_baseline(dataset, model_id)
        baselines[name] = (acc, cost)
        print(f"  {name}: Accuracy={acc:.1%}, Cost=${cost:.5f}")
    
    # 3. Create Router
    print("\n[3] INITIALIZING ROUTER...")
    if PRIORS_PATH.exists():
        print(f"  Loading priors from: {PRIORS_PATH}")
        router = BanditRouter.create(
            model_registry=MODEL_REGISTRY,
            exploration="safe",
            priors="bundled",
            bundled_priors_path=PRIORS_PATH,
            prior_strength=100.0,  # Boosted confidence
        )
    else:
        print(f"  WARNING: No priors found, using cold start")
        router = BanditRouter(MODEL_REGISTRY, exploration="safe")
    
    # 4. Run Pareto Experiment
    print("\n[4] EXPLORING PARETO FRONTIER...")
    print(f"  Testing {len(LAMBDA_VALUES)} lambda values...")
    print("")
    print(f"{'Lambda':>10} | {'Accuracy':>10} | {'Cost':>12} | {'GPT-4o %':>10} | {'DeepSeek %':>10}")
    print("-" * 65)
    
    results = []
    for lam in LAMBDA_VALUES:
        result = run_evaluation(dataset, router, lambda_cost=lam)
        result.prior_strength = 100.0
        results.append(result)
        
        gpt_pct = result.model_distribution.get('openai/gpt-4o', 0) * 100
        ds_pct = result.model_distribution.get('deepseek/deepseek-chat-v3-0324', 0) * 100
        
        print(f"{lam:>10.1f} | {result.accuracy:>10.1%} | ${result.avg_cost:>10.5f} | {gpt_pct:>9.1f}% | {ds_pct:>9.1f}%")
    
    # 5. Generate Plots
    print("\n[5] GENERATING PLOTS...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_pareto_frontier(results, baselines, OUTPUT_DIR / "pareto_frontier.pdf")
    plot_model_distribution(results, OUTPUT_DIR / "model_distribution_by_lambda.pdf")
    plot_accuracy_vs_lambda(results, baselines, OUTPUT_DIR / "accuracy_cost_vs_lambda.pdf")
    
    # 6. Save Results
    results_json = {
        "experiment": "pareto_frontier",
        "test_set_size": TEST_SET_SIZE,
        "baselines": {name: {"accuracy": acc, "cost": cost} for name, (acc, cost) in baselines.items()},
        "results": [
            {
                "lambda_cost": r.lambda_cost,
                "accuracy": r.accuracy,
                "avg_cost": r.avg_cost,
                "model_distribution": r.model_distribution,
                "domain_accuracy": r.domain_accuracy,
            }
            for r in results
        ],
    }
    
    output_json = OUTPUT_DIR.parent / "experiment" / "pareto_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {output_json}")
    
    # 7. Find Sweet Spot
    print("\n" + "=" * 70)
    print(" ANALYSIS: Finding the Sweet Spot")
    print("=" * 70)
    
    gpt_acc, gpt_cost = baselines['GPT-4o']
    ds_acc, ds_cost = baselines['DeepSeek']
    
    # Find best trade-off points
    for r in results:
        cost_reduction = (gpt_cost - r.avg_cost) / gpt_cost * 100
        acc_drop = (gpt_acc - r.accuracy) / gpt_acc * 100
        
        if cost_reduction > 50 and acc_drop < 10:
            print(f"\n  ★ SWEET SPOT FOUND at λ={r.lambda_cost}:")
            print(f"    Accuracy: {r.accuracy:.1%} (only {acc_drop:.1f}% drop from GPT-4o)")
            print(f"    Cost: ${r.avg_cost:.5f} ({cost_reduction:.1f}% savings)")
            print(f"    Model Mix: GPT-4o={r.model_distribution.get('openai/gpt-4o', 0):.0%}, "
                  f"DeepSeek={r.model_distribution.get('deepseek/deepseek-chat-v3-0324', 0):.0%}")
    
    # Summary table
    print("\n  Operating Point Recommendations:")
    print("-" * 60)
    print(f"  {'Profile':<20} | {'Lambda':>8} | {'Accuracy':>10} | {'Cost':>10}")
    print("-" * 60)
    
    # Find best for each profile
    quality_best = max(results, key=lambda r: r.accuracy)
    cost_best = min(results, key=lambda r: r.avg_cost)
    balanced = max(results, key=lambda r: r.accuracy - r.avg_cost * 100)
    
    print(f"  {'Max Quality':<20} | {quality_best.lambda_cost:>8.1f} | {quality_best.accuracy:>10.1%} | ${quality_best.avg_cost:>8.5f}")
    print(f"  {'Balanced':<20} | {balanced.lambda_cost:>8.1f} | {balanced.accuracy:>10.1%} | ${balanced.avg_cost:>8.5f}")
    print(f"  {'Cost Saver':<20} | {cost_best.lambda_cost:>8.1f} | {cost_best.accuracy:>10.1%} | ${cost_best.avg_cost:>8.5f}")
    
    print("-" * 60)
    print("\n" + "=" * 70)
    print(" COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
