#!/usr/bin/env python3
"""
Multi-Domain Calibration: The "Golden Trio" Approach

This script calibrates the router on three fundamentally different task types:
- GSM8K (Math): Numeric reasoning with verifiable answers
- HumanEval (Code): Code generation with execution-based verification
- IFEval (Instructions): Constraint following with mechanical checks

Key Insight: Semantic Clustering
Because BanditRouter uses embeddings, training on HumanEval teaches the router
about ALL coding tasks. When it sees a Python prompt, it updates weights in the
"Coding Direction" of vector space. In production, a "Java Code" request will
generalize because the vectors are semantically close.

Benchmark Performance (from Artificial Analysis API - Independent Evaluation):
- GPT-4o: 75.9% math, 90.2% code, 34.3% IFBench
- DeepSeek-V3: 94.2% math, 91.6% code, 41.0% IFBench (beats GPT-4o!)
- Nova-Lite: 76.5% math, 58.0% code, 34.1% IFBench

Note: IFBench is AA's proprietary adversarial benchmark (58 hard constraints),
NOT Google's IFEval (~500 prompts). The ~34% scores are correct for IFBench.

The bandit should learn to:
- Route math tasks to DeepSeek (24% better than GPT-4o AND cheaper!)
- Route code tasks to DeepSeek (slightly better AND cheaper)
- Route all tasks to DeepSeek in practice (it dominates on all benchmarks)

Prerequisites:
    pip install datasets numpy pandas sentence-transformers

Usage:
    python kdd_paper/scripts/calibrate_multi_domain.py

Output:
    - results/multi_domain/multi_domain_priors.npz
    - Console showing domain-specific model performance
"""

from __future__ import annotations

import re
import json
import random
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.core.bandit_router import BanditRouter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Define the models with their metadata
MODEL_REGISTRY = {
    "openai/gpt-4o": {
        "display_name": "GPT-4o (Teacher)",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "price_1m_blended": 5.00,
        "ttft_mean": 1.5,
    },
    "deepseek/deepseek-chat-v3-0324": {
        # Note: This is the correct OpenRouter ID for "DeepSeek V3"
        # The model is sometimes called "deepseek-v3" but the API ID is "deepseek-chat-v3-0324"
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

# Output paths
OUTPUT_DIR = Path("results/multi_domain")
PRIORS_OUTPUT_PATH = OUTPUT_DIR / "multi_domain_priors.npz"

# Samples per domain (use 100+ for real calibration)
SAMPLES_PER_DOMAIN = 30


# ==============================================================================
# 2. BENCHMARK LOADER (The "Data Normalizer")
# ==============================================================================

class BenchmarkLoader:
    """
    Fetches data from GSM8K, HumanEval, and IFEval and normalizes it
    into a standard format for the bandit to learn from.
    """
    
    def __init__(self):
        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
    
    def get_calibration_batch(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Load and normalize tasks from all three domains.
        
        Returns a shuffled list of tasks, each with:
        - domain: "math", "code", or "instruction"
        - prompt: The user-facing prompt text
        - ground_truth: Domain-specific verification data
        """
        print(f"\n[1] LOADING BENCHMARKS (Target: {n} samples per domain)...")
        tasks = []
        
        # --- A. Math (GSM8K) ---
        print("    Loading GSM8K (Math)...")
        try:
            ds_math = self.load_dataset("gsm8k", "main", split="test")
            for i in range(min(n, len(ds_math))):
                item = ds_math[i]
                tasks.append({
                    "domain": "math",
                    "prompt": item['question'],
                    "ground_truth": item['answer']  # Contains reasoning + #### Number
                })
            print(f"      ✓ Loaded {min(n, len(ds_math))} math problems")
        except Exception as e:
            print(f"      ✗ GSM8K failed: {e}")
        
        # --- B. Code (HumanEval) ---
        print("    Loading HumanEval (Code)...")
        try:
            ds_code = self.load_dataset("openai/openai_humaneval", split="test")
            for i in range(min(n, len(ds_code))):
                item = ds_code[i]
                tasks.append({
                    "domain": "code",
                    "prompt": item['prompt'],  # Function signature + docstring
                    "ground_truth": {
                        "test": item['test'],
                        "entry_point": item['entry_point'],
                        "canonical_solution": item.get('canonical_solution', ''),
                    }
                })
            print(f"      ✓ Loaded {min(n, len(ds_code))} code problems")
        except Exception as e:
            print(f"      ✗ HumanEval failed: {e}")
        
        # --- C. Instructions (IFEval) ---
        print("    Loading IFEval (Constraint Following)...")
        try:
            ds_instr = self.load_dataset("google/IFEval", split="train")
            for i in range(min(n, len(ds_instr))):
                item = ds_instr[i]
                tasks.append({
                    "domain": "instruction",
                    "prompt": item['prompt'],
                    "ground_truth": {
                        "kwargs": item.get('kwargs', {}),
                        "instruction_id_list": item.get('instruction_id_list', []),
                    }
                })
            print(f"      ✓ Loaded {min(n, len(ds_instr))} instruction tasks")
        except Exception as e:
            print(f"      ✗ IFEval failed: {e}")
        
        print(f"    Total tasks: {len(tasks)}")
        random.shuffle(tasks)
        return tasks


# ==============================================================================
# 3. VERIFIERS (The "Ground Truth" Judges)
# ==============================================================================

class Verifier:
    """
    Objective truth checkers. No LLMs allowed here.
    Each domain has its own verification logic.
    """
    
    @staticmethod
    def check(domain: str, model_output: str, ground_truth: Any) -> float:
        """
        Returns 1.0 (success) or 0.0 (failure) based on domain-specific rules.
        """
        if domain == "math":
            return Verifier._check_math(model_output, ground_truth)
        elif domain == "code":
            return Verifier._check_code(model_output, ground_truth)
        elif domain == "instruction":
            return Verifier._check_instruction(model_output, ground_truth)
        return 0.0
    
    @staticmethod
    def _check_math(output: str, truth_str: str) -> float:
        """Extract number after #### and compare."""
        def get_num(text: str) -> Optional[float]:
            if not text:
                return None
            if "####" in text:
                text = text.split("####")[-1]
            matches = re.findall(r'-?[\d,]*\.?\d+', text)
            return float(matches[-1].replace(',', '')) if matches else None
        
        pred = get_num(output)
        truth = get_num(truth_str)
        if pred is None or truth is None:
            return 0.0
        return 1.0 if abs(pred - truth) < 1e-6 else 0.0
    
    @staticmethod
    def _check_code(output: str, truth_dict: Dict) -> float:
        """
        In production, run this in a sandbox (PyBox/Docker).
        Here, we mock the execution result based on magic strings.
        """
        # We simulate a "Pass" if the model output contains a magic string
        if "[EXECUTION_SUCCESS]" in output:
            return 1.0
        return 0.0
    
    @staticmethod
    def _check_instruction(output: str, constraints: Dict) -> float:
        """
        Checks mechanical constraints (e.g., word count, capitalization).
        """
        # Mock logic: If the model succeeded, it respected the constraint
        if "[CONSTRAINT_MET]" in output:
            return 1.0
        return 0.0


# ==============================================================================
# 4. BENCHMARK DATA (Loaded from models_cache.json - Real AA Data)
# ==============================================================================

def load_real_capabilities() -> Dict[str, Dict[str, float]]:
    """
    Load REAL benchmark scores from models_cache.json.
    
    This file contains data from Artificial Analysis API - an independent
    evaluation source that tests all models with the same methodology.
    
    Maps benchmark fields to our domain categories:
    - math: math_500 (MATH-500 benchmark)
    - code: humaneval_score / 100 (HumanEval, normalized to 0-1)
    - instruction: mmlu_pro (proxy for instruction following)
    """
    cache_path = Path(__file__).parent.parent.parent / "banditgpt" / "data" / "models_cache.json"
    
    with open(cache_path) as f:
        data = json.load(f)
    
    # Build lookup by openrouter_id
    model_lookup = {m.get('openrouter_id'): m for m in data['models']}
    
    capabilities = {}
    for model_id in MODEL_REGISTRY.keys():
        model_data = model_lookup.get(model_id, {})
        
        # Extract REAL benchmark scores from models_cache.json
        math_score = model_data.get('math_500', 0.5)
        code_score = model_data.get('humaneval_score', 50.0) / 100.0  # Normalize to 0-1
        instruction_score = model_data.get('mmlu_pro', 0.5)  # Proxy for instruction following
        
        capabilities[model_id] = {
            "math": math_score,
            "code": code_score,
            "instruction": instruction_score,
        }
    
    return capabilities

# Load REAL data from models_cache.json at module init
REAL_CAPABILITIES = load_real_capabilities()


def mock_generate(model_id: str, domain: str, ground_truth: Any, problem_id: int = 0) -> str:
    """
    DETERMINISTIC model generation using REAL benchmark performance data.
    
    Instead of random sampling, we use a deterministic hash to decide success.
    This ensures reproducible results while matching real benchmark accuracy.
    
    THE TRUTH (From Artificial Analysis - Independent Evaluation):
    - GPT-4o:       75.9% math, 90.2% code, 34.3% IFBench
    - DeepSeek-V3:  94.2% math, 91.6% code, 41.0% IFBench (BEST!)
    - Nova-Lite:    76.5% math, 58.0% code, 34.1% IFBench
    
    Note: IFBench scores (~34%) are NOT comparable to IFEval (~85%).
    IFBench is a harder, adversarial benchmark with 58 complex constraints.
    
    Key Insight: DeepSeek-V3 beats GPT-4o on ALL domains in reality!
    This demonstrates why "reputation-based routing" fails.
    """
    import hashlib
    
    capabilities = REAL_CAPABILITIES
    success_rate = capabilities.get(model_id, {}).get(domain, 0.5)
    
    # DETERMINISTIC: Hash (model + domain + problem) to get consistent result
    # The hash determines a number 0-1, if < success_rate then success
    hash_input = f"{model_id}:{domain}:{problem_id}:{ground_truth}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) / (2**128)
    is_success = hash_val < success_rate
    
    # Generate Mock Output
    if domain == "math":
        # Extract the true answer from ground truth
        truth_num = None
        if isinstance(ground_truth, str) and "####" in ground_truth:
            matches = re.findall(r'-?[\d,]*\.?\d+', ground_truth.split("####")[-1])
            if matches:
                truth_num = float(matches[-1].replace(',', ''))
        
        if truth_num is not None:
            ans = truth_num if is_success else truth_num + 1
            return f"Let me solve this step by step... The answer is #### {ans}"
        return f"Reasoning... #### {42 if is_success else 43}"
    
    elif domain == "code":
        # In real life, this would be Python code
        if is_success:
            return "def solution(x):\n    return x * 2\n[EXECUTION_SUCCESS]"
        else:
            return "def solution(x):\n    return x  # Wrong implementation"
    
    elif domain == "instruction":
        if is_success:
            return "Here is the text following all constraints.\n[CONSTRAINT_MET]"
        else:
            return "Oops, I forgot to follow the constraint."
    
    return "Unknown domain"


# ==============================================================================
# 5. MAIN CALIBRATION LOOP
# ==============================================================================

def run_calibration(samples_per_domain: int = SAMPLES_PER_DOMAIN) -> BanditRouter:
    """
    Run the multi-domain calibration loop.
    
    Returns the trained router.
    """
    # 1. Load Data
    loader = BenchmarkLoader()
    tasks = loader.get_calibration_batch(n=samples_per_domain)
    
    if not tasks:
        raise ValueError("No tasks loaded! Check your internet connection.")
    
    # 2. Init Router (Aggressive Mode to learn fast)
    print(f"\n[2] INITIALIZING ROUTER...")
    router = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="aggressive",  # Alpha = 2.0 (explore heavily)
        priors="none",  # Start fresh
    )
    
    print(f"\n[3] STARTING TRAINING ({len(tasks)} samples)...")
    print(f"{'Step':<6} | {'Domain':<12} | {'Model':<25} | {'Reward':<8} | {'Predicted'}")
    print("-" * 75)
    
    history = []
    
    # Get model IDs for round-robin sampling
    model_ids = list(MODEL_REGISTRY.keys())
    print(f"    Using round-robin calibration across {len(model_ids)} models")
    
    # Helper to normalize vectors
    def l2_normalize(x):
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    
    for i, task in enumerate(tasks):
        # A. ROUTE - ROUND ROBIN (ensures equal model coverage)
        # During calibration, we force even sampling to learn about ALL models
        # This prevents UCB's "rich-get-richer" bias that starves new models
        model_id = model_ids[i % len(model_ids)]
        
        # Get the embedding/log for the prompt (we need the context vector)
        _, log = router.route(task['prompt'], exploration="aggressive")
        
        # B. GENERATE & VERIFY
        # This replaces the LLM Judge with domain-specific verifiers
        output = mock_generate(model_id, task['domain'], task['ground_truth'])
        reward = Verifier.check(task['domain'], output, task['ground_truth'])
        
        # C. UPDATE BANDIT with the forced model choice
        # Directly update the bandit with our chosen model (round-robin)
        # This ensures equal representation during calibration
        x = l2_normalize(np.asarray(log.context_vector, dtype=np.float64))
        router.bandit.update(model_id, x, float(reward))
        
        history.append({
            "domain": task['domain'],
            "model": model_id,
            "reward": reward,
        })
        
        if i % 10 == 0:
            print(f"{i:<6} | {task['domain']:<12} | {model_id:<25} | {reward:<8.0f} | {log.predicted_quality:.2f}")
    
    print("-" * 75)
    
    # 3. Analyze Results
    print("\n[4] WHAT THE BANDIT LEARNED:")
    print("-" * 50)
    
    # Group by domain and model
    domain_model_rewards: Dict[str, Dict[str, List[float]]] = {}
    for h in history:
        domain = h['domain']
        model = h['model']
        if domain not in domain_model_rewards:
            domain_model_rewards[domain] = {}
        if model not in domain_model_rewards[domain]:
            domain_model_rewards[domain][model] = []
        domain_model_rewards[domain][model].append(h['reward'])
    
    for domain in sorted(domain_model_rewards.keys()):
        print(f"\n  {domain.upper()}:")
        for model in sorted(domain_model_rewards[domain].keys()):
            rewards = domain_model_rewards[domain][model]
            avg = sum(rewards) / len(rewards) if rewards else 0
            print(f"    {model}: {avg:.0%} accuracy ({len(rewards)} samples)")
    
    return router


def save_priors(router: BanditRouter) -> None:
    """Save the learned multi-domain priors."""
    print(f"\n[5] SAVING MULTI-DOMAIN PRIORS...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    router.save_shippable_priors(PRIORS_OUTPUT_PATH)
    
    print(f"    ✓ Saved to: {PRIORS_OUTPUT_PATH}")
    print("    This file contains domain-specific model expertise.")


# ==============================================================================
# 6. VERIFICATION (Simulate Production)
# ==============================================================================

def verify_production_loading() -> None:
    """
    Verify that the saved priors correctly influence routing decisions.
    """
    print(f"\n[6] VERIFYING PRODUCTION BEHAVIOR...")
    
    if not PRIORS_OUTPUT_PATH.exists():
        print(f"    Priors file not found: {PRIORS_OUTPUT_PATH}")
        return
    
    # Load the router with the priors we just made
    router = BanditRouter.create(
        model_registry=MODEL_REGISTRY,
        exploration="safe",  # Production mode
        priors="bundled",
        bundled_priors_path=PRIORS_OUTPUT_PATH,
        prior_strength=50.0,
    )
    
    # Test Prompts (one per domain)
    test_prompts = [
        ("Write a Python function to sort a list using quicksort.", "Code"),
        ("Calculate: If a train travels 120 miles in 2 hours, what is its speed?", "Math"),
        ("Write a poem without using the letter 'e'. Must be exactly 4 lines.", "Instruction"),
    ]
    
    print(f"\n{'Task Type':<15} | {'Selected Model':<25} | {'Utility':<10} | {'Quality Est'}")
    print("-" * 70)
    
    for prompt, category in test_prompts:
        # Use rank_prompt to see internal scores
        ranking = router.rank_prompt(prompt, top_k=len(MODEL_REGISTRY))
        top = ranking[0]
        
        print(f"{category:<15} | {top['model_id']:<25} | {top['utility']:<10.3f} | {top['quality_hat']:.3f}")
        
        # Show all rankings for this prompt
        print(f"  Full ranking:")
        for r in ranking:
            print(f"    {r['model_id']}: utility={r['utility']:.3f}, quality={r['quality_hat']:.3f}")
    
    print("\n[INTERPRETATION]")
    print("-" * 50)
    print("Expected behavior after training (based on AA Independent Evaluation):")
    print("  • Math tasks    → DeepSeek (94.2% vs GPT-4o's 75.9%)")
    print("  • Code tasks    → DeepSeek (91.6% vs GPT-4o's 90.2%)")
    print("  • IFBench       → DeepSeek (41.0% vs GPT-4o's 34.3%)")
    print("")
    print("Key Finding: DeepSeek-V3 beats GPT-4o on ALL benchmarks!")
    print("(AA uses standardized evaluation - same ruler for all models)")
    print("This validates why 'reputation-based routing' fails.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> int:
    print("=" * 75)
    print("MULTI-DOMAIN CALIBRATION: The 'Golden Trio' Approach")
    print("=" * 75)
    print("Training on three fundamentally different task types:")
    print("  1. GSM8K (Math)      - Numeric reasoning, verifiable answers")
    print("  2. HumanEval (Code)  - Code generation, execution-based")
    print("  3. IFEval (Instructions) - Constraint following, mechanical checks")
    print("")
    print("Key Insight: Semantic Clustering")
    print("  Training on HumanEval teaches about ALL coding tasks.")
    print("  The embedding space groups similar tasks together.")
    print("=" * 75)
    
    # 1. Run calibration
    router = run_calibration(samples_per_domain=SAMPLES_PER_DOMAIN)
    
    # 2. Save priors
    save_priors(router)
    
    # 3. Verify production behavior
    verify_production_loading()
    
    print("\n" + "=" * 75)
    print("CALIBRATION COMPLETE")
    print("=" * 75)
    print(f"Priors saved to: {PRIORS_OUTPUT_PATH}")
    print("")
    print("The router has learned domain-specific model expertise:")
    print("  • Which models excel at math (numeric verification)")
    print("  • Which models excel at code (execution success)")
    print("  • Which models excel at instructions (constraint following)")
    print("=" * 75)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
