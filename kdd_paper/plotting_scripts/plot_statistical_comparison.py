#!/usr/bin/env python3
"""
Statistical Comparison: Notched Box Plots for Cost & Latency

This script generates publication-quality notched box plots that visually
prove statistical significance between the Router and baselines.

Key Visual Claims:
- No Overlap in Notches = Statistically Significant Difference (p < 0.05)
- Overlap = Indistinguishable Medians

The notch represents the 95% Confidence Interval of the Median.

Prerequisites:
    pip install matplotlib numpy pandas scipy datasets sentence-transformers

Usage:
    # First, run calibration to generate priors
    python kdd_paper/scripts/calibrate_multi_domain.py
    
    # Then, generate statistical plots
    python kdd_paper/scripts/plot_statistical_comparison.py

Output:
    - results/figures/cost_distribution_boxplot.pdf
    - results/figures/latency_distribution_boxplot.pdf
    - results/figures/combined_statistical_comparison.pdf
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
TEST_SET_SIZE = 100  # Larger sample for statistical power


@dataclass
class ExperimentResult:
    """Raw experiment data for statistical analysis."""
    name: str
    costs: List[float]
    latencies: List[float]
    accuracies: List[float]
    model_selections: List[str]
    domains: List[str]


# ==============================================================================
# 2. RAW DATA COLLECTION
# ==============================================================================

def run_baseline_raw(
    dataset: List[Dict[str, Any]],
    model_id: str,
    name: str,
) -> ExperimentResult:
    """
    Run a static baseline and collect RAW per-query data.
    """
    print(f"  Collecting raw data: {name}...")
    
    costs = []
    latencies = []
    accuracies = []
    domains = []
    
    for task in dataset:
        domain = task['domain']
        
        # Measure latency
        start = time.perf_counter()
        output = mock_generate(model_id, domain, task['ground_truth'])
        latency = time.perf_counter() - start
        
        # Verify correctness
        is_correct = Verifier.check(domain, output, task['ground_truth'])
        
        # Calculate cost (per query, normalized)
        cost = MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
        
        costs.append(cost)
        latencies.append(latency)
        accuracies.append(float(is_correct))
        domains.append(domain)
    
    return ExperimentResult(
        name=name,
        costs=costs,
        latencies=latencies,
        accuracies=accuracies,
        model_selections=[model_id] * len(dataset),
        domains=domains,
    )


def run_router_raw(
    dataset: List[Dict[str, Any]],
    router: BanditRouter,
) -> ExperimentResult:
    """
    Run the bandit router and collect RAW per-query data.
    """
    print(f"  Collecting raw data: Bandit Router...")
    
    costs = []
    latencies = []
    accuracies = []
    model_selections = []
    domains = []
    
    for task in dataset:
        domain = task['domain']
        
        # Measure routing + generation latency
        start = time.perf_counter()
        model_id, log = router.route(task['prompt'], exploration="safe")
        output = mock_generate(model_id, domain, task['ground_truth'])
        latency = time.perf_counter() - start
        
        # Verify correctness
        is_correct = Verifier.check(domain, output, task['ground_truth'])
        
        # Calculate cost
        cost = MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
        
        costs.append(cost)
        latencies.append(latency)
        accuracies.append(float(is_correct))
        model_selections.append(model_id)
        domains.append(domain)
    
    return ExperimentResult(
        name="Bandit Router",
        costs=costs,
        latencies=latencies,
        accuracies=accuracies,
        model_selections=model_selections,
        domains=domains,
    )


# ==============================================================================
# 3. STATISTICAL ANALYSIS
# ==============================================================================

def compute_statistics(results: List[ExperimentResult]) -> Dict[str, Any]:
    """
    Compute statistical summaries and pairwise comparisons.
    """
    stats_summary = {}
    
    for r in results:
        costs = np.array(r.costs)
        latencies = np.array(r.latencies)
        accuracies = np.array(r.accuracies)
        
        stats_summary[r.name] = {
            "cost": {
                "mean": np.mean(costs),
                "median": np.median(costs),
                "std": np.std(costs),
                "q25": np.percentile(costs, 25),
                "q75": np.percentile(costs, 75),
            },
            "latency": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "std": np.std(latencies),
                "q25": np.percentile(latencies, 25),
                "q75": np.percentile(latencies, 75),
            },
            "accuracy": {
                "mean": np.mean(accuracies),
                "ci_95": 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)),
            },
            "n": len(costs),
        }
    
    # Pairwise Mann-Whitney U tests (non-parametric)
    if len(results) >= 2:
        router = next((r for r in results if "Router" in r.name), None)
        teacher = next((r for r in results if "Teacher" in r.name), None)
        
        if router and teacher:
            # Cost comparison
            u_cost, p_cost = stats.mannwhitneyu(
                router.costs, teacher.costs, alternative='less'
            )
            stats_summary["pairwise_tests"] = {
                "router_vs_teacher_cost": {
                    "U_statistic": u_cost,
                    "p_value": p_cost,
                    "significant": p_cost < 0.05,
                }
            }
    
    return stats_summary


# ==============================================================================
# 4. PLOTTING FUNCTIONS
# ==============================================================================

def plot_cost_boxplot(
    results: List[ExperimentResult],
    output_path: Path,
) -> None:
    """
    Generate a notched box plot for cost distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [np.array(r.costs) for r in results]
    labels = [r.name for r in results]
    
    # Create notched box plot
    box = ax.boxplot(
        data,
        notch=True,
        bootstrap=1000,
        patch_artist=True,
        tick_labels=labels,
    )
    
    # Styling
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    for patch, color in zip(box['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Median lines in black for visibility
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Cost per Query ($)', fontsize=12)
    ax.set_title('Cost Distribution: Notched Box Plot\n(Non-overlapping notches = significant difference at p<0.05)', 
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Log scale for cost (large range)
    ax.set_yscale('log')
    
    # Add annotations
    for i, r in enumerate(results):
        median = np.median(r.costs)
        ax.annotate(
            f'${median:.5f}',
            xy=(i + 1, median),
            xytext=(i + 1.3, median),
            fontsize=9,
            ha='left',
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_latency_boxplot(
    results: List[ExperimentResult],
    output_path: Path,
) -> None:
    """
    Generate a notched box plot for latency distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to milliseconds for readability
    data = [np.array(r.latencies) * 1000 for r in results]
    labels = [r.name for r in results]
    
    # Create notched box plot
    box = ax.boxplot(
        data,
        notch=True,
        bootstrap=1000,
        patch_artist=True,
        tick_labels=labels,
    )
    
    # Styling
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    for patch, color in zip(box['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Latency per Query (ms)', fontsize=12)
    ax.set_title('Latency Distribution: Notched Box Plot\n(Includes routing overhead for Bandit Router)', 
                 fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Add annotations
    for i, r in enumerate(results):
        median = np.median(r.latencies) * 1000
        ax.annotate(
            f'{median:.2f}ms',
            xy=(i + 1, median),
            xytext=(i + 1.3, median),
            fontsize=9,
            ha='left',
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_accuracy_with_ci(
    results: List[ExperimentResult],
    output_path: Path,
) -> None:
    """
    Generate a bar plot with 95% confidence intervals for accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r.name for r in results]
    accuracies = [np.mean(r.accuracies) for r in results]
    cis = [1.96 * np.std(r.accuracies) / np.sqrt(len(r.accuracies)) for r in results]
    
    x = np.arange(len(names))
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    
    bars = ax.bar(x, accuracies, yerr=cis, capsize=8, color=colors[:len(results)], alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy with 95% Confidence Intervals', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Add value labels
    for i, (acc, ci) in enumerate(zip(accuracies, cis)):
        ax.annotate(
            f'{acc:.1%}±{ci:.1%}',
            xy=(i, acc + ci + 0.02),
            ha='center',
            fontsize=10,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_combined_figure(
    results: List[ExperimentResult],
    output_path: Path,
) -> None:
    """
    Generate a combined figure with all statistical plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- Panel A: Cost Distribution ---
    ax = axes[0]
    data = [np.array(r.costs) for r in results]
    labels = [r.name.replace(' (', '\n(') for r in results]
    
    box = ax.boxplot(data, notch=True, bootstrap=1000, patch_artist=True, tick_labels=labels)
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    for patch, color in zip(box['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Cost per Query ($)')
    ax.set_title('(A) Cost Distribution')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # --- Panel B: Latency Distribution ---
    ax = axes[1]
    data = [np.array(r.latencies) * 1000 for r in results]
    
    box = ax.boxplot(data, notch=True, bootstrap=1000, patch_artist=True, tick_labels=labels)
    for patch, color in zip(box['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('(B) Latency Distribution')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # --- Panel C: Accuracy with CI ---
    ax = axes[2]
    names = [r.name.replace(' (', '\n(') for r in results]
    accuracies = [np.mean(r.accuracies) for r in results]
    cis = [1.96 * np.std(r.accuracies) / np.sqrt(len(r.accuracies)) for r in results]
    
    x = np.arange(len(names))
    bars = ax.bar(x, accuracies, yerr=cis, capsize=8, color=colors[:len(results)], alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('(C) Accuracy (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    for i, (acc, ci) in enumerate(zip(accuracies, cis)):
        ax.annotate(f'{acc:.0%}', xy=(i, acc + ci + 0.03), ha='center', fontsize=10)
    
    plt.suptitle('Statistical Comparison: Bandit Router vs Baselines\n(Notches show 95% CI of median)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_router_selection_breakdown(
    router_result: ExperimentResult,
    output_path: Path,
) -> None:
    """
    Show how the router distributes selections across models and domains.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Panel A: Model Selection Distribution ---
    ax = axes[0]
    from collections import Counter
    selections = Counter(router_result.model_selections)
    models = list(selections.keys())
    counts = [selections[m] for m in models]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    ax.pie(counts, labels=models, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('(A) Router Model Selection')
    
    # --- Panel B: Domain-wise Cost Distribution ---
    ax = axes[1]
    domains = list(set(router_result.domains))
    domain_costs = {d: [] for d in domains}
    
    for cost, domain in zip(router_result.costs, router_result.domains):
        domain_costs[domain].append(cost)
    
    data = [domain_costs[d] for d in sorted(domains)]
    labels = [d.capitalize() for d in sorted(domains)]
    
    box = ax.boxplot(data, notch=True, bootstrap=1000, patch_artist=True, tick_labels=labels)
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(box['boxes'], colors[:len(domains)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Cost per Query ($)')
    ax.set_title('(B) Cost by Domain')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    plt.suptitle('Router Behavior Analysis', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main() -> int:
    print("=" * 70)
    print(" STATISTICAL COMPARISON: Notched Box Plots")
    print("=" * 70)
    print("Generating publication-quality statistical visualizations")
    print("")
    print("Key Visual Claim:")
    print("  Non-overlapping notches = statistically significant (p < 0.05)")
    print("=" * 70)
    
    # 1. Load Test Data
    print("\n[1] LOADING TEST DATA...")
    loader = BenchmarkLoader()
    dataset = loader.get_calibration_batch(n=TEST_SET_SIZE)
    
    if not dataset:
        print("ERROR: No test data loaded!")
        return 1
    
    # 2. Collect Raw Data
    print("\n[2] COLLECTING RAW DATA...")
    
    results = []
    
    # Baseline A: Teacher
    res_gpt4 = run_baseline_raw(dataset, "openai/gpt-4o", "GPT-4o (Teacher)")
    results.append(res_gpt4)
    
    # Baseline B: Specialist
    res_deepseek = run_baseline_raw(dataset, "deepseek/deepseek-chat-v3-0324", "DeepSeek (Specialist)")
    results.append(res_deepseek)
    
    # Baseline C: Budget
    res_nova = run_baseline_raw(dataset, "amazon/nova-lite-v1", "Nova-Lite (Budget)")
    results.append(res_nova)
    
    # Router
    print("\n[3] RUNNING ROUTER...")
    if PRIORS_PATH.exists():
        print(f"  Loading priors from: {PRIORS_PATH}")
        router = BanditRouter.create(
            model_registry=MODEL_REGISTRY,
            exploration="safe",
            priors="bundled",
            bundled_priors_path=PRIORS_PATH,
            prior_strength=50.0,
        )
    else:
        print(f"  WARNING: No priors found, using cold start")
        router = BanditRouter(MODEL_REGISTRY, exploration="safe")
    
    res_router = run_router_raw(dataset, router)
    results.append(res_router)
    
    # 3. Compute Statistics
    print("\n[4] COMPUTING STATISTICS...")
    stats_summary = compute_statistics(results)
    
    # Print summary
    print("\n  Summary Statistics:")
    print("-" * 60)
    for name, s in stats_summary.items():
        if name == "pairwise_tests":
            continue
        print(f"\n  {name}:")
        print(f"    Cost:     median=${s['cost']['median']:.5f}, std=${s['cost']['std']:.5f}")
        print(f"    Latency:  median={s['latency']['median']*1000:.2f}ms")
        print(f"    Accuracy: {s['accuracy']['mean']:.1%} ± {s['accuracy']['ci_95']:.1%}")
    
    if "pairwise_tests" in stats_summary:
        test = stats_summary["pairwise_tests"]["router_vs_teacher_cost"]
        print(f"\n  Mann-Whitney U (Router vs Teacher Cost):")
        print(f"    U={test['U_statistic']:.0f}, p={test['p_value']:.4f}")
        print(f"    Significant (p<0.05): {test['significant']}")
    
    # 4. Generate Plots
    print("\n[5] GENERATING PLOTS...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_cost_boxplot(results, OUTPUT_DIR / "cost_distribution_boxplot.pdf")
    plot_latency_boxplot(results, OUTPUT_DIR / "latency_distribution_boxplot.pdf")
    plot_accuracy_with_ci(results, OUTPUT_DIR / "accuracy_comparison.pdf")
    plot_combined_figure(results, OUTPUT_DIR / "combined_statistical_comparison.pdf")
    plot_router_selection_breakdown(res_router, OUTPUT_DIR / "router_behavior_analysis.pdf")
    
    # 5. Save raw data for reproducibility
    raw_data = {
        name: {
            "costs": r.costs,
            "latencies": r.latencies,
            "accuracies": r.accuracies,
            "model_selections": r.model_selections,
            "domains": r.domains,
        }
        for name, r in [(r.name, r) for r in results]
    }
    
    with open(OUTPUT_DIR / "raw_experiment_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'raw_experiment_data.json'}")
    
    print("\n" + "=" * 70)
    print(" COMPLETE")
    print("=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("")
    print("Key Figures for Paper:")
    print("  1. combined_statistical_comparison.pdf - Main comparison figure")
    print("  2. cost_distribution_boxplot.pdf - Cost analysis (log scale)")
    print("  3. router_behavior_analysis.pdf - Routing decisions breakdown")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
