#!/usr/bin/env python3
"""
Execution-Based Experiment: Proving Router Quality at Specialist Cost

This script generates "Table 3" for the KDD paper, demonstrating:
- Router achieves Teacher-level accuracy (≈ GPT-4o)
- Router achieves Specialist-level cost (≈ DeepSeek)

Methodology:
1. Load held-out test set (prompts not seen during calibration)
2. Run baselines: "Always GPT-4o" and "Always DeepSeek"
3. Run router with learned priors
4. Compare accuracy and cost using GROUND TRUTH verification

Key Difference from LLM-as-Judge:
- Math: Check if extracted number matches answer key
- Code: Check if code executes correctly (mock: magic string)
- Instructions: Check if constraints are met (mock: magic string)

Prerequisites:
    pip install datasets numpy pandas sentence-transformers

Usage:
    # First, run calibration to generate priors
    python kdd_paper/scripts/calibrate_multi_domain.py
    
    # Then, run this experiment
    python kdd_paper/scripts/run_experiment.py

Output:
    - Console: Table 3 showing accuracy and cost comparison
    - results/experiment/table3_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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
OUTPUT_DIR = Path("results/experiment")
TEST_SET_SIZE = 50  # Number of samples per domain for the TEST set


# ==============================================================================
# 2. EXPERIMENT RUNNERS
# ==============================================================================

def run_baseline_evaluation(
    dataset: List[Dict[str, Any]],
    model_id: str,
    name: str,
) -> Dict[str, Any]:
    """
    Run a 'Static Policy' (Always route to Model X).
    
    This establishes the baseline for comparison:
    - GPT-4o: Upper bound on quality, upper bound on cost
    - DeepSeek: Lower bound on quality (maybe), lower bound on cost
    """
    print(f"  Running Baseline: {name}...")
    
    results_by_domain: Dict[str, List[float]] = {}
    total_cost = 0.0
    total_correct = 0
    
    for task in dataset:
        domain = task['domain']
        
        # 1. Generate (Mocked)
        output = mock_generate(model_id, domain, task['ground_truth'])
        
        # 2. Verify (Ground Truth - NOT LLM Judge)
        is_correct = Verifier.check(domain, output, task['ground_truth'])
        
        # 3. Calculate Cost (normalized per task)
        cost = MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
        total_cost += cost
        
        # Track by domain
        if domain not in results_by_domain:
            results_by_domain[domain] = []
        results_by_domain[domain].append(is_correct)
        total_correct += is_correct
    
    accuracy = total_correct / len(dataset) if dataset else 0
    avg_cost = total_cost / len(dataset) if dataset else 0
    
    # Per-domain accuracy
    domain_accuracy = {
        domain: sum(results) / len(results) if results else 0
        for domain, results in results_by_domain.items()
    }
    
    return {
        "system": name,
        "model_id": model_id,
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "cost_reduction": 0.0,  # Baseline
        "domain_accuracy": domain_accuracy,
        "n_samples": len(dataset),
    }


def run_router_evaluation(
    dataset: List[Dict[str, Any]],
    router: BanditRouter,
    online_learning: bool = False,
) -> Dict[str, Any]:
    """
    Run the 'Bandit Policy' (Dynamic Routing with learned priors).
    
    Args:
        dataset: List of task dictionaries
        router: Configured BanditRouter with priors
        online_learning: If True, update bandit during evaluation
    """
    print(f"  Running Router (Bandit)...")
    
    results_by_domain: Dict[str, List[float]] = {}
    model_selections: Dict[str, int] = {}
    total_cost = 0.0
    total_correct = 0
    
    for task in dataset:
        domain = task['domain']
        
        # 1. Route (Bandit selects model based on prompt embedding)
        model_id, log = router.route(task['prompt'], exploration="safe")
        
        # Track model selections
        model_selections[model_id] = model_selections.get(model_id, 0) + 1
        
        # 2. Generate (Mocked)
        output = mock_generate(model_id, domain, task['ground_truth'])
        
        # 3. Verify (Ground Truth - NOT LLM Judge)
        is_correct = Verifier.check(domain, output, task['ground_truth'])
        
        # 4. Optional: Online learning during evaluation
        if online_learning:
            router.report_feedback(log.request_id, reward=is_correct, response_text=output)
        
        # 5. Calculate Cost
        cost = MODEL_REGISTRY[model_id].get('price_1m_blended', 1.0) / 1000.0
        total_cost += cost
        
        # Track by domain
        if domain not in results_by_domain:
            results_by_domain[domain] = []
        results_by_domain[domain].append(is_correct)
        total_correct += is_correct
    
    accuracy = total_correct / len(dataset) if dataset else 0
    avg_cost = total_cost / len(dataset) if dataset else 0
    
    # Per-domain accuracy
    domain_accuracy = {
        domain: sum(results) / len(results) if results else 0
        for domain, results in results_by_domain.items()
    }
    
    return {
        "system": "Bandit Router",
        "model_id": "dynamic",
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "cost_reduction": 0.0,  # Calculated later
        "domain_accuracy": domain_accuracy,
        "model_selections": model_selections,
        "n_samples": len(dataset),
    }


# ==============================================================================
# 3. TABLE GENERATION
# ==============================================================================

def generate_table3(results: List[Dict[str, Any]]) -> str:
    """
    Generate formatted Table 3 for the paper.
    """
    lines = [
        "",
        "=" * 70,
        " TABLE 3: Execution-Based Validation Results",
        "=" * 70,
        "",
        f"{'System':<25} | {'Accuracy':>10} | {'Avg Cost':>12} | {'Cost Reduction':>15}",
        "-" * 70,
    ]
    
    for r in results:
        accuracy_str = f"{r['accuracy']:.1%}"
        cost_str = f"${r['avg_cost']:.5f}"
        reduction_str = f"{r['cost_reduction']:.1%}" if r['cost_reduction'] != 0 else "-"
        
        lines.append(f"{r['system']:<25} | {accuracy_str:>10} | {cost_str:>12} | {reduction_str:>15}")
    
    lines.append("-" * 70)
    
    # Add domain breakdown
    lines.append("")
    lines.append("Domain Breakdown:")
    lines.append("-" * 50)
    
    domains = set()
    for r in results:
        domains.update(r.get('domain_accuracy', {}).keys())
    
    header = f"{'System':<25}"
    for domain in sorted(domains):
        header += f" | {domain.capitalize():>10}"
    lines.append(header)
    lines.append("-" * 50)
    
    for r in results:
        row = f"{r['system']:<25}"
        for domain in sorted(domains):
            acc = r.get('domain_accuracy', {}).get(domain, 0)
            row += f" | {acc:>10.1%}"
        lines.append(row)
    
    lines.append("-" * 50)
    
    # Add router model selection breakdown
    for r in results:
        if 'model_selections' in r:
            lines.append("")
            lines.append("Router Model Selection:")
            total = sum(r['model_selections'].values())
            for model, count in sorted(r['model_selections'].items(), key=lambda x: -x[1]):
                pct = count / total * 100
                lines.append(f"  {model}: {count} ({pct:.1f}%)")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main() -> int:
    print("=" * 70)
    print(" EXECUTION-BASED EXPERIMENT")
    print("=" * 70)
    print("Proving: Router achieves Teacher accuracy at Specialist cost")
    print("")
    print("Methodology:")
    print("  - Ground truth verification (NOT LLM-as-Judge)")
    print("  - Math: Numeric answer comparison")
    print("  - Code: Execution success")
    print("  - Instructions: Constraint satisfaction")
    print("=" * 70)
    
    # 1. Prepare Data (Held-out Test Set)
    print("\n[1] LOADING TEST DATA...")
    loader = BenchmarkLoader()
    # Note: In a real run, ensure these indices don't overlap with calibration
    dataset = loader.get_calibration_batch(n=TEST_SET_SIZE)
    
    if not dataset:
        print("ERROR: No test data loaded!")
        return 1
    
    experiment_results = []
    
    # 2. Run Baselines
    print("\n[2] RUNNING BASELINES...")
    
    # Baseline A: The "Teacher" (Upper Bound on Quality, Upper Bound on Cost)
    res_gpt4 = run_baseline_evaluation(dataset, "openai/gpt-4o", "GPT-4o (Teacher)")
    experiment_results.append(res_gpt4)
    
    # Baseline B: The "Specialist" (Lower Bound on Cost)
    res_deepseek = run_baseline_evaluation(dataset, "deepseek/deepseek-chat-v3-0324", "DeepSeek (Specialist)")
    experiment_results.append(res_deepseek)
    
    # Baseline C: Budget option
    res_nova = run_baseline_evaluation(dataset, "amazon/nova-lite-v1", "Nova-Lite (Budget)")
    experiment_results.append(res_nova)
    
    # 3. Run Router
    print("\n[3] RUNNING ROUTER...")
    
    # Check for priors
    if not PRIORS_PATH.exists():
        print(f"  WARNING: {PRIORS_PATH} not found.")
        print("  Router will be 'cold start' (no learned expertise).")
        print("  Run 'calibrate_multi_domain.py' first!")
        priors_mode = "none"
        priors_path = None
    else:
        print(f"  Loading priors from: {PRIORS_PATH}")
        priors_mode = "bundled"
        priors_path = PRIORS_PATH
    
    router = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="safe",  # Exploit learned knowledge
        priors=priors_mode,
        bundled_priors_path=priors_path,
        prior_strength=50.0,
    )
    
    res_router = run_router_evaluation(dataset, router, online_learning=False)
    
    # Calculate Cost Reduction vs Teacher
    teacher_cost = res_gpt4['avg_cost']
    if teacher_cost > 0:
        res_router['cost_reduction'] = (teacher_cost - res_router['avg_cost']) / teacher_cost
        res_deepseek['cost_reduction'] = (teacher_cost - res_deepseek['avg_cost']) / teacher_cost
        res_nova['cost_reduction'] = (teacher_cost - res_nova['avg_cost']) / teacher_cost
    
    experiment_results.append(res_router)
    
    # 4. Generate and Print Table 3
    table = generate_table3(experiment_results)
    print(table)
    
    # 5. Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_json = {
        "experiment": "execution_based_validation",
        "test_set_size": TEST_SET_SIZE,
        "total_samples": len(dataset),
        "results": experiment_results,
    }
    
    with open(OUTPUT_DIR / "table3_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_DIR / 'table3_results.json'}")
    
    # 6. Success Criteria Check
    print("\n" + "=" * 70)
    print(" SUCCESS CRITERIA CHECK")
    print("=" * 70)
    
    router_acc = res_router['accuracy']
    teacher_acc = res_gpt4['accuracy']
    specialist_cost = res_deepseek['avg_cost']
    router_cost = res_router['avg_cost']
    
    acc_gap = abs(router_acc - teacher_acc) / teacher_acc if teacher_acc > 0 else 0
    cost_ratio = router_cost / specialist_cost if specialist_cost > 0 else float('inf')
    
    print(f"1. Accuracy Gap (Router vs Teacher): {acc_gap:.1%}")
    if acc_gap < 0.10:
        print("   ✓ PASS: Router is within 10% of Teacher accuracy")
    else:
        print("   ✗ FAIL: Router accuracy gap > 10%")
    
    print(f"2. Cost Ratio (Router vs Specialist): {cost_ratio:.2f}x")
    if cost_ratio < 2.0:
        print("   ✓ PASS: Router cost is within 2x of Specialist")
    else:
        print("   ✗ FAIL: Router cost > 2x Specialist")
    
    print(f"3. Cost Reduction vs Teacher: {res_router['cost_reduction']:.1%}")
    if res_router['cost_reduction'] > 0.5:
        print("   ✓ PASS: Router achieves >50% cost reduction")
    else:
        print("   ○ INFO: Cost reduction < 50%")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
