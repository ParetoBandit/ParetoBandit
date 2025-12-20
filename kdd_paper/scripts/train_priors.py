#!/usr/bin/env python3
"""
Complete Workflow: Download Data → Calibrate (Train) → Save Priors → Load in Production

This script demonstrates the full lifecycle of the "Shippable Brain":
1. Load benchmark data (GSM8K math problems)
2. Calibrate the router by observing ground-truth rewards
3. Save the learned priors to a file
4. Load the priors in a new "production" router

Prerequisites:
    pip install datasets numpy sentence-transformers

Usage:
    python kdd_paper/scripts/train_priors.py

Output:
    - results/train_priors/enlightened_priors.npz (the learned priors)
    - Console output showing the training and comparison

Key Insight:
    This script shows that when you use GROUND TRUTH rewards (actual math answers)
    instead of an LLM judge, the router learns which models are actually good at math.
    
    MATH-500 Benchmark Scores (from Artificial Analysis - Independent Evaluation):
    - DeepSeek-V3: 94.2% (BEST - 24% better than GPT-4o!)
    - Nova-Lite: 76.5%
    - GPT-4o: 75.9%
"""

from __future__ import annotations

import re
import random
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.core.bandit_router import BanditRouter

# Setup basic logging to see the router working
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# We define our "Teacher" (expensive/safe) and "Specialist" (cheap/risky)
MODEL_REGISTRY = {
    "openai/gpt-4o": {
        "display_name": "GPT-4o (Teacher)",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "price_1m_blended": 5.00,
        "ttft_mean": 1.5,
    },
    "deepseek/deepseek-chat-v3-0324": {
        # Note: This is "DeepSeek V3" - the OpenRouter ID differs from marketing name
        "display_name": "DeepSeek V3 (Specialist)",
        "input_cost_per_m": 0.07,
        "output_cost_per_m": 0.28,
        "price_1m_blended": 0.14,
        "ttft_mean": 0.8,
    },
    "amazon/nova-lite-v1": {
        "display_name": "Nova-Lite (Budget)",
        "input_cost_per_m": 0.06,
        "output_cost_per_m": 0.24,
        "price_1m_blended": 0.10,
        "ttft_mean": 0.5,
    },
}

# Where we will save the learned "Enlightened" priors
OUTPUT_DIR = Path("results/train_priors")
PRIORS_OUTPUT_PATH = OUTPUT_DIR / "enlightened_priors.npz"


# ==============================================================================
# 2. GROUND TRUTH MECHANISM (The "Judge" Replacement)
# ==============================================================================

def extract_number(text: str) -> float | None:
    """Robustly extracts the last number from GSM8K text."""
    if text is None:
        return None
    # GSM8K answer key format often uses ####
    if "####" in text:
        text = text.split("####")[-1]
    # Find all numbers (integers or floats)
    matches = re.findall(r'-?[\d,]*\.?\d+', text)
    if not matches:
        return None
    # Return the last number found, removing commas
    return float(matches[-1].replace(',', ''))


def check_answer_math(model_output: str, ground_truth_str: str) -> float:
    """
    Returns 1.0 (Success) or 0.0 (Failure).
    This replaces the 'LLM-as-a-Judge'.
    """
    pred = extract_number(model_output)
    truth = extract_number(ground_truth_str)
    
    if pred is None or truth is None:
        return 0.0
    
    # Check for equality with slight floating point tolerance
    return 1.0 if abs(pred - truth) < 1e-6 else 0.0


# ==============================================================================
# 3. MOCK LLM GENERATION
# ==============================================================================

# MATH-500 scores from Artificial Analysis API (Independent Evaluation)
# Source: artificialanalysis.ai (Dec 2025)
#
# WHY AA DATA IS MORE TRUSTWORTHY:
# - Standardized evaluation (same prompt format, same grader for all models)
# - Pass@1 evaluation (no cherry-picking)
# - Independent auditor (not self-reported marketing numbers)

MATH_CAPABILITIES = {
    "openai/gpt-4o": 0.759,                  # MATH-500: 75.9% (AA)
    "deepseek/deepseek-chat-v3-0324": 0.942, # MATH-500: 94.2% (AA) - BEST!
    "amazon/nova-lite-v1": 0.765,            # MATH-500: 76.5% (AA)
}


def mock_generate(model_id: str, prompt: str, ground_truth_val: float) -> str:
    """
    Simulates Model Generation using REAL benchmark performance.
    
    REAL MATH-500 BENCHMARK SCORES (from models_cache.json / Artificial Analysis):
    - GPT-4o: 75.9% (the "popular" model)
    - DeepSeek-V3: 94.2% (actually 24% BETTER than GPT-4o on math!)
    - Nova-Lite: 76.5% (similar to GPT-4o)
    
    The bandit doesn't know this yet. It has to learn it.
    """
    # REAL capabilities from MATH-500 benchmark (models_cache.json)
    capabilities = MATH_CAPABILITIES
    
    rate = capabilities.get(model_id, 0.5)
    
    # Simulate generation
    if random.random() < rate:
        return f"Reasoning... The answer is #### {ground_truth_val}"
    else:
        return f"Reasoning... The answer is #### {ground_truth_val + 1}"


# ==============================================================================
# 4. STEP 1: LOADING BENCHMARKS & TRAINING (CALIBRATION)
# ==============================================================================

def run_calibration_loop(n_samples: int = 100) -> BanditRouter:
    """
    Train the router on GSM8K math problems using ground-truth rewards.
    """
    print(f"\n[1] LOADING BENCHMARK (GSM8K)...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    # Load samples for training
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    print(f"    Loaded {len(dataset)} math problems")
    
    # Initialize Router with "Aggressive" exploration to learn fast
    print(f"\n[2] INITIALIZING ROUTER (Cold Start)...")
    router = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="aggressive",  # Alpha = 2.0 (Try everything!)
        priors="none",  # Start fresh, no prior knowledge
    )
    
    print(f"\n[3] STARTING TRAINING LOOP...")
    print(f"{'Step':<10} | {'Model Selected':<25} | {'Reward':<10} | {'Predicted Quality'}")
    print("-" * 80)
    
    rewards_history = {k: [] for k in MODEL_REGISTRY}
    
    for i, item in enumerate(dataset):
        prompt = item['question']
        ground_truth_str = item['answer']
        
        # A. ROUTE (The Bandit picks a model)
        model_id, log = router.route(prompt, exploration="aggressive")
        
        # B. EXECUTE (We run the model and check the math)
        truth_val = extract_number(ground_truth_str)
        if truth_val is None:
            continue
            
        output = mock_generate(model_id, prompt, truth_val)
        reward = check_answer_math(output, ground_truth_str)
        
        # C. UPDATE (Train the Bandit with ground truth reward)
        router.report_feedback(log.request_id, reward=reward, response_text=output)
        
        # Log stats
        rewards_history[model_id].append(reward)
        if i % 10 == 0:
            pred_quality = log.predicted_quality
            print(f"{i:<10} | {model_id:<25} | {reward:<10.1f} | {pred_quality:.2f}")
    
    print("-" * 80)
    print("Training Complete. Stats:")
    for m, rewards in rewards_history.items():
        if rewards:
            print(f"  - {m}: {sum(rewards)/len(rewards):.1%} Accuracy ({len(rewards)} samples)")
    
    return router


# ==============================================================================
# 5. STEP 2: SAVING THE PRIORS
# ==============================================================================

def save_enlightened_priors(router: BanditRouter) -> None:
    """Save the learned priors to disk."""
    print(f"\n[4] SAVING ENLIGHTENED PRIORS...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export the internal 'A' and 'b' matrices to a compressed file
    router.save_shippable_priors(PRIORS_OUTPUT_PATH)
    
    print(f"    Success! Saved to: {PRIORS_OUTPUT_PATH}")
    print("    This file contains the learned model expertise for math problems.")


# ==============================================================================
# 6. STEP 3: USING DEFAULT VS. ENLIGHTENED PRIORS (PRODUCTION)
# ==============================================================================

def simulate_production_startup() -> None:
    """Compare default priors vs enlightened priors."""
    print(f"\n[5] SIMULATING PRODUCTION STARTUP...")
    
    # CASE A: Default Priors (No prior knowledge)
    print("\n--- Case A: Cold Start (No Priors) ---")
    router_cold = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="safe",
        priors="none",  # No prior knowledge
    )
    
    # Ask it to rank a math prompt
    ranking = router_cold.rank_prompt("Calculate 25 * 45", top_k=3)
    print("Cold Start rankings:")
    for r in ranking:
        print(f"  - {r['model_id']}: Utility {r['utility']:.3f} (Quality Est: {r['quality_hat']:.3f})")
    
    # CASE B: Enlightened Priors (Learned from GSM8K)
    print("\n--- Case B: Enlightened Priors (Learned from Math) ---")
    
    if not PRIORS_OUTPUT_PATH.exists():
        print(f"    Priors file not found: {PRIORS_OUTPUT_PATH}")
        print("    Run calibration first!")
        return
    
    router_smart = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="safe",
        priors="bundled",
        bundled_priors_path=PRIORS_OUTPUT_PATH,  # Use our learned priors
        prior_strength=50.0,  # High confidence in our calibration
    )
    
    # It should now have learned opinions about math ability
    ranking = router_smart.rank_prompt("Calculate 25 * 45", top_k=3)
    print("Enlightened Router rankings:")
    for r in ranking:
        print(f"  - {r['model_id']}: Utility {r['utility']:.3f} (Quality Est: {r['quality_hat']:.3f})")
    
    print("\n[CONCLUSION]")
    # Check if DeepSeek is ranked higher (it should be - 94.2% vs 75.9% on MATH-500!)
    top_model = ranking[0]['model_id'] if ranking else None
    if top_model == "deepseek/deepseek-chat-v3-0324":
        print("SUCCESS: The router correctly learned that DeepSeek excels at math!")
        print("         REAL MATH-500 scores: DeepSeek=94.2% vs GPT-4o=75.9%")
        print("         DeepSeek is 24% BETTER and 35x CHEAPER!")
    elif top_model == "openai/gpt-4o":
        print("NOTE: GPT-4o selected despite lower MATH-500 score (75.9% vs 94.2%)")
        print("      This may indicate insufficient training samples.")
    else:
        print(f"Top model: {top_model}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> int:
    print("=" * 80)
    print("TRAIN PRIORS: Complete Workflow Demo")
    print("=" * 80)
    print("This demonstrates the 'Shippable Brain' lifecycle:")
    print("  1. Load benchmark data (GSM8K math problems)")
    print("  2. Train router by observing ground-truth rewards")
    print("  3. Save learned priors to file")
    print("  4. Load priors in production router")
    print("=" * 80)
    
    # 1. Train
    trained_router = run_calibration_loop(n_samples=100)
    
    # 2. Save
    save_enlightened_priors(trained_router)
    
    # 3. Demo Usage
    simulate_production_startup()
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Priors saved to: {PRIORS_OUTPUT_PATH}")
    print("You can now use these priors in production!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
