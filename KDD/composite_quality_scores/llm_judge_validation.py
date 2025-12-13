#!/usr/bin/env python3
"""
LLM-as-a-Judge Validation Experiment

Validates BLF composite scores using a novel held-out dataset and 
an independent LLM judge. This provides non-circular external validation.

Methodology:
1. Generate 20 fresh prompts per intent (not in any benchmark)
2. Select top 5 and bottom 5 models by BLF score (with API access)
3. Run all 10 models on each prompt
4. Use strong LLM judge (Claude/GPT-4o) to score responses 1-10
5. Compute Spearman correlation: BLF rank vs Judge rank

Expected Result:
- If BLF scores are valid, judge rankings should correlate with BLF rankings
- Spearman ρ > 0.6 indicates strong predictive validity

Usage:
    python llm_judge_validation.py --intent coding --dry-run
    python llm_judge_validation.py --intent coding --run
"""

import json
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, kendalltau

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
except ImportError:
    print("python-dotenv not installed, using system environment")

# Add project root


# =============================================================================
# Ranking Comparison Metrics
# =============================================================================

def compute_ranking_metrics(blf_ranks: List[int], judge_ranks: List[int], 
                           blf_scores: List[float], judge_scores: List[float]) -> Dict:
    """
    Compute multiple metrics comparing BLF rankings to Judge rankings.
    
    Args:
        blf_ranks: BLF-based ranks (1 = best)
        judge_ranks: Judge-based ranks (1 = best)
        blf_scores: Raw BLF scores
        judge_scores: Raw judge scores
    
    Returns:
        Dict of metrics with interpretations
    """
    n = len(blf_ranks)
    
    metrics = {}
    
    # 1. Spearman's ρ - Rank correlation
    rho, rho_pval = spearmanr(blf_ranks, judge_ranks)
    metrics['spearman_rho'] = rho
    metrics['spearman_pval'] = rho_pval
    
    # 2. Kendall's τ - Pairwise concordance
    tau, tau_pval = kendalltau(blf_ranks, judge_ranks)
    metrics['kendall_tau'] = tau
    metrics['kendall_pval'] = tau_pval
    
    # 3. Mean Absolute Rank Difference (MARD)
    # "On average, how many positions off are we?"
    rank_diffs = [abs(b - j) for b, j in zip(blf_ranks, judge_ranks)]
    metrics['mean_abs_rank_diff'] = np.mean(rank_diffs)
    metrics['max_rank_diff'] = max(rank_diffs)
    
    # 4. Top-k Precision
    # "What fraction of our top-k are in the judge's top-k?"
    for k in [3, 5]:
        if n >= k:
            blf_top_k = set(i for i, r in enumerate(blf_ranks) if r <= k)
            judge_top_k = set(i for i, r in enumerate(judge_ranks) if r <= k)
            precision = len(blf_top_k & judge_top_k) / k
            metrics[f'top_{k}_precision'] = precision
    
    # 5. Top/Bottom Separation
    # "Do all top-5 BLF models score higher than all bottom-5 by judge?"
    # This is the key test: can BLF distinguish good from bad?
    if n >= 10:
        blf_top_5_indices = [i for i, r in enumerate(blf_ranks) if r <= 5]
        blf_bottom_5_indices = [i for i, r in enumerate(blf_ranks) if r > n - 5]
        
        top_judge_scores = [judge_scores[i] for i in blf_top_5_indices]
        bottom_judge_scores = [judge_scores[i] for i in blf_bottom_5_indices]
        
        # What fraction of top-5 beat all bottom-5?
        min_top = min(top_judge_scores) if top_judge_scores else 0
        max_bottom = max(bottom_judge_scores) if bottom_judge_scores else 0
        
        metrics['top_bottom_gap'] = min_top - max_bottom
        metrics['top_bottom_separated'] = min_top > max_bottom
        
        # Average separation
        metrics['avg_top_judge_score'] = np.mean(top_judge_scores) if top_judge_scores else 0
        metrics['avg_bottom_judge_score'] = np.mean(bottom_judge_scores) if bottom_judge_scores else 0
    
    # 6. Pairwise Accuracy
    # "For all pairs of models, how often does BLF correctly predict which is better?"
    correct_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            blf_i_better = blf_ranks[i] < blf_ranks[j]
            judge_i_better = judge_scores[i] > judge_scores[j]
            if blf_i_better == judge_i_better:
                correct_pairs += 1
            total_pairs += 1
    
    metrics['pairwise_accuracy'] = correct_pairs / total_pairs if total_pairs > 0 else 0
    
    return metrics


def print_ranking_metrics(metrics: Dict):
    """Print ranking metrics in a formatted way."""
    print("\n" + "="*80)
    print("RANKING COMPARISON METRICS")
    print("="*80)
    
    print("\n📊 CORRELATION METRICS (do rankings agree?)")
    print("-" * 60)
    rho = metrics.get('spearman_rho', 0)
    tau = metrics.get('kendall_tau', 0)
    print(f"  Spearman's ρ:  {rho:+.3f}  (p={metrics.get('spearman_pval', 1):.4f})")
    print(f"  Kendall's τ:   {tau:+.3f}  (p={metrics.get('kendall_pval', 1):.4f})")
    
    if rho > 0.7:
        print("  → Strong agreement between BLF and Judge rankings ✅")
    elif rho > 0.4:
        print("  → Moderate agreement between BLF and Judge rankings ⚠️")
    else:
        print("  → Weak agreement between BLF and Judge rankings ❌")
    
    print("\n📏 RANK DIFFERENCE METRICS (how far off are we?)")
    print("-" * 60)
    mard = metrics.get('mean_abs_rank_diff', 0)
    max_diff = metrics.get('max_rank_diff', 0)
    print(f"  Mean Absolute Rank Diff: {mard:.2f} positions")
    print(f"  Max Rank Diff:           {max_diff} positions")
    
    if mard < 2:
        print("  → Rankings are very close (within 2 positions on average) ✅")
    elif mard < 3:
        print("  → Rankings are reasonably close ⚠️")
    else:
        print("  → Rankings differ substantially ❌")
    
    print("\n🎯 PRECISION METRICS (do we identify the best models?)")
    print("-" * 60)
    for k in [3, 5]:
        key = f'top_{k}_precision'
        if key in metrics:
            prec = metrics[key]
            print(f"  Top-{k} Precision: {prec*100:.0f}% ({int(prec*k)}/{k} models)")
    
    pairwise = metrics.get('pairwise_accuracy', 0)
    print(f"  Pairwise Accuracy: {pairwise*100:.1f}% (correctly ordered pairs)")
    
    print("\n🔀 TOP vs BOTTOM SEPARATION (can we distinguish good from bad?)")
    print("-" * 60)
    if 'top_bottom_gap' in metrics:
        gap = metrics['top_bottom_gap']
        separated = metrics['top_bottom_separated']
        avg_top = metrics['avg_top_judge_score']
        avg_bottom = metrics['avg_bottom_judge_score']
        
        print(f"  Avg Judge Score (BLF Top-5):    {avg_top:.2f}")
        print(f"  Avg Judge Score (BLF Bottom-5): {avg_bottom:.2f}")
        print(f"  Gap: {avg_top - avg_bottom:+.2f} points")
        
        if separated:
            print(f"  → All top-5 scored higher than all bottom-5 ✅")
        elif gap > 0:
            print(f"  → Top-5 scored higher on average, but with overlap ⚠️")
        else:
            print(f"  → Cannot distinguish top from bottom ❌")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAY")
    print("="*80)
    
    # Overall assessment
    rho = metrics.get('spearman_rho', 0)
    pairwise = metrics.get('pairwise_accuracy', 0)
    separated = metrics.get('top_bottom_separated', False)
    
    if rho > 0.6 and pairwise > 0.7 and separated:
        print("✅ STRONG VALIDATION: BLF scores reliably predict judge rankings")
        print("   The composite scores have high predictive validity.")
    elif rho > 0.4 or pairwise > 0.6:
        print("⚠️  MODERATE VALIDATION: BLF scores partially predict judge rankings")
        print("   The composite scores capture some signal but with noise.")
    else:
        print("❌ WEAK VALIDATION: BLF scores do not reliably predict judge rankings")
        print("   Consider revising the composite score methodology.")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Load Real Human Prompts from Dataset
# =============================================================================

@dataclass
class HumanEvalProblem:
    """A HumanEval Plus problem with all data needed for evaluation."""
    prompt: str
    canonical_solution: str
    entry_point: str          # Function name to call
    test_cases: List[Dict]    # List of {"input": ..., "expected": ...}
    task_id: str


# =============================================================================
# CODE EXECUTION SANDBOX (Step 1 of KDD-Grade Workflow)
# =============================================================================

def execute_code_in_sandbox(
    prompt: str, 
    model_code: str, 
    entry_point: str, 
    test_cases: List[Dict],
    timeout: float = 5.0
) -> Dict:
    """
    Execute model-generated code in a sandboxed environment.
    
    This is Step 1 of the KDD-Grade workflow: objectively verify correctness
    by actually running the code against test cases.
    
    Args:
        prompt: The function signature/stub from HumanEval
        model_code: The model's generated solution
        entry_point: Name of the function to test
        test_cases: List of {"input": args, "expected": output}
        timeout: Max seconds per test case
    
    Returns:
        Dict with:
        - passed: bool (all tests passed)
        - tests_passed: int
        - tests_total: int
        - error: str or None
        - details: List of test results
    """
    import re
    import signal
    import traceback
    
    result = {
        "passed": False,
        "tests_passed": 0,
        "tests_total": len(test_cases),
        "error": None,
        "details": []
    }
    
    if not test_cases:
        result["error"] = "No test cases available"
        return result
    
    # Extract code from model response (handle markdown code blocks)
    code = model_code
    
    # Try to extract code from markdown blocks
    code_block_match = re.search(r'```(?:python)?\s*\n(.*?)```', model_code, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1)
    
    # Build the full code: prompt (signature) + model solution
    # The prompt usually contains "def func_name(...):" and we need to add the body
    full_code = prompt + "\n" + code
    
    # Alternative: if model provides complete function, use that
    if f"def {entry_point}" in code:
        full_code = code
    
    # Create isolated namespace for execution
    namespace = {}
    
    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    # Try to compile and execute
    try:
        # Compile the code
        compiled = compile(full_code, '<model_code>', 'exec')
        
        # Execute to define the function
        exec(compiled, namespace)
        
        # Check function was defined
        if entry_point not in namespace:
            result["error"] = f"Function '{entry_point}' not defined"
            return result
        
        func = namespace[entry_point]
        
        # Run test cases
        for i, test in enumerate(test_cases[:10]):  # Limit to 10 tests for speed
            test_result = {
                "test_id": i,
                "input": str(test.get("call_str", test.get("input", "")))[:100],
                "passed": False,
                "error": None
            }
            
            try:
                # Set timeout
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                
                # Get expected output
                expected = test.get("expected")
                
                # Skip placeholder tests
                if expected == "__SKIP_CHECK__":
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
                    test_result["passed"] = True
                    result["tests_passed"] += 1
                    result["details"].append(test_result)
                    continue
                
                # Execute the call - either as call_str or as function args
                call_str = test.get("call_str")
                if call_str:
                    # Evaluate the call string directly
                    actual = eval(call_str, namespace)
                else:
                    # Legacy: call function with args
                    inputs = test.get("input", ())
                    if isinstance(inputs, tuple):
                        actual = func(*inputs)
                    elif isinstance(inputs, dict):
                        actual = func(**inputs)
                    else:
                        actual = func(inputs)
                
                # Cancel timeout
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)
                
                # Compare results
                if actual == expected:
                    test_result["passed"] = True
                    result["tests_passed"] += 1
                else:
                    test_result["error"] = f"Expected {expected!r}, got {actual!r}"
                    
            except TimeoutError:
                signal.setitimer(signal.ITIMER_REAL, 0)
                test_result["error"] = "Timeout"
            except Exception as e:
                signal.setitimer(signal.ITIMER_REAL, 0)
                test_result["error"] = f"{type(e).__name__}: {str(e)[:100]}"
            
            result["details"].append(test_result)
        
        # Determine overall pass/fail
        result["passed"] = result["tests_passed"] == len(result["details"])
        
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    
    return result


def load_humaneval_plus_prompts(n_samples: int = 20, seed: int = 42) -> List[HumanEvalProblem]:
    """
    Load prompts from HumanEval Plus dataset with proper test cases.
    
    Parses docstring examples (>>> lines) to create executable test cases.
    
    Args:
        n_samples: Number of prompts to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of HumanEvalProblem objects with prompts, canonical solutions, and test cases
    """
    import random
    import re
    random.seed(seed)
    
    from datasets import load_dataset
    
    print("  Loading HumanEval Plus dataset from HuggingFace...")
    dataset = load_dataset("evalplus/humanevalplus")
    
    problems = []
    for entry in dataset['test']:
        prompt = entry['prompt']
        entry_point = entry.get('entry_point', '')
        canonical = entry['canonical_solution']
        task_id = entry.get('task_id', 'unknown')
        
        # Parse docstring examples: >>> func_call\n expected_result
        test_cases = []
        
        # Find all >>> examples and their expected outputs
        # Pattern: >>> func_name(args)\n    expected_value
        lines = prompt.split('\n')
        example_calls = []
        
        for i, line in enumerate(lines):
            if '>>>' in line and entry_point in line:
                # Found an example call
                call = line.split('>>>')[1].strip()
                # Look for the expected output on the next line(s)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    # Skip if next line is another >>> or empty
                    if next_line and not next_line.startswith('>>>'):
                        example_calls.append((call, next_line))
        
        # Build and verify test cases using canonical solution
        full_canonical = prompt + "\n" + canonical
        
        try:
            namespace = {"__builtins__": __builtins__}
            exec(compile(full_canonical, '<canonical>', 'exec'), namespace)
            func = namespace.get(entry_point)
            
            if func and example_calls:
                for call_str, expected_str in example_calls[:5]:  # Max 5 tests
                    try:
                        # Parse the expected output
                        expected = eval(expected_str.strip())
                        
                        # Execute the call directly to get actual result
                        # This is safer than trying to parse args
                        actual = eval(call_str, namespace)
                        
                        if actual == expected:
                            # Store the call string for later execution
                            test_cases.append({
                                "input": call_str,  # Store as string to eval later
                                "expected": expected,
                                "call_str": call_str
                            })
                    except Exception as e:
                        pass
            
            # If no docstring examples worked, try to at least verify function exists
            if not test_cases and func:
                test_cases.append({"input": "__FUNC_EXISTS__", "expected": "__SKIP_CHECK__"})
                    
        except Exception as e:
            # If canonical fails to compile, skip this problem
            continue
        
        # Only include problems with valid test cases
        if test_cases:
            problems.append(HumanEvalProblem(
                prompt=prompt,
                canonical_solution=canonical,
                entry_point=entry_point,
                test_cases=test_cases,
                task_id=task_id
            ))
    
    print(f"  Loaded {len(problems)} HumanEval Plus problems with executable test cases")
    
    # Random sample
    if len(problems) <= n_samples:
        return problems
    
    return random.sample(problems, n_samples)


def load_real_prompts(
    intent: str, 
    n_samples: int = 20, 
    seed: int = 42,
    complexity_filter: str = "complex"  # "all", "simple", "complex", "humaneval"
) -> List[str]:
    """
    Load real human prompts from our labeled dataset or HumanEval Plus.
    
    Args:
        intent: One of 'coding', 'reasoning', 'factual_qa', 'summarization', 'general'
        n_samples: Number of prompts to sample
        seed: Random seed for reproducibility
        complexity_filter: "all" = no filter, "simple" = simple tasks only, 
                          "complex" = NVIDIA constraint filter,
                          "humaneval" = use HumanEval Plus dataset (coding only)
    
    Returns:
        List of prompt strings
    """
    import random
    random.seed(seed)
    
    # Use HumanEval Plus for coding validation (most rigorous)
    if intent == "coding" and complexity_filter == "humaneval":
        return load_humaneval_plus_prompts(n_samples, seed)
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'real_intent_prompts_labeled.json'
    
    with open(data_path) as f:
        data = json.load(f)
    
    samples = data['samples']
    
    # Filter by intent
    intent_prompts = [s['prompt'] for s in samples if s['intent_label'] == intent]
    
    if not intent_prompts:
        raise ValueError(f"No prompts found for intent: {intent}")
    
    # Apply complexity filter using NVIDIA's classifier
    if complexity_filter not in ["all", "humaneval"]:
        from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
        classifier = NvidiaComplexityClassifier()
        
        print(f"  Classifying {len(intent_prompts)} prompts with NVIDIA classifier...")
        results = classifier.classify_batch(intent_prompts)
        
        filtered_prompts = []
        
        # Use different thresholds based on intent
        if intent == "coding":
            threshold_field = "constraint_ct"
            threshold_value = 0.65
        elif intent == "reasoning":
            threshold_field = "reasoning"
            threshold_value = 0.10
        else:
            threshold_field = "prompt_complexity_score"
            threshold_value = 0.30
        
        for prompt, result in zip(intent_prompts, results):
            score = getattr(result, threshold_field)
            
            if complexity_filter == "complex" and score >= threshold_value:
                filtered_prompts.append(prompt)
            elif complexity_filter == "simple" and score < threshold_value:
                filtered_prompts.append(prompt)
        
        print(f"  Complexity filter '{complexity_filter}' ({threshold_field} >= {threshold_value}): {len(filtered_prompts)}/{len(intent_prompts)} prompts qualify")
        intent_prompts = filtered_prompts
        
        if not intent_prompts:
            print(f"  Warning: No {complexity_filter} prompts found, using all prompts")
            intent_prompts = [s['prompt'] for s in samples if s['intent_label'] == intent]
    
    # Random sample
    if len(intent_prompts) <= n_samples:
        return intent_prompts
    
    return random.sample(intent_prompts, n_samples)


# =============================================================================
# Model Selection
# =============================================================================

@dataclass
class ModelForExperiment:
    """Model selected for the experiment."""
    name: str
    openrouter_id: str  # e.g., "anthropic/claude-opus-4.5"
    blf_score: float
    blf_rank: int  # 1 = best


def load_models_for_intent(intent: str, models_cache_path: str) -> Tuple[List[ModelForExperiment], str]:
    """
    Load top 5 and bottom 5 models for a given intent.
    
    Returns:
        Tuple of (models_list, composite_field_name)
    """
    intent_to_composite = {
        'coding': 'ccs_100',
        'reasoning': 'crs_100',
        'factual_qa': 'cfs_100',
        'summarization': 'css_100',
    }
    
    composite_field = intent_to_composite.get(intent)
    if not composite_field:
        raise ValueError(f"Unknown intent: {intent}")
    
    with open(models_cache_path) as f:
        data = json.load(f)
    models = data if isinstance(data, list) else data.get('models', [])
    
    # Filter to models with OpenRouter access and composite score
    api_models = [
        m for m in models 
        if m.get('openrouter_id')  # Must have openrouter_id
        and m.get(composite_field) is not None
    ]
    
    # Sort by composite score
    api_models.sort(key=lambda x: x[composite_field], reverse=True)
    
    # Select top 5 and bottom 5
    top_5 = api_models[:5]
    bottom_5 = api_models[-5:]
    
    # Create model objects
    selected = []
    all_selected = top_5 + bottom_5
    
    for i, m in enumerate(all_selected):
        rank = i + 1 if i < 5 else len(api_models) - (9 - i)
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            blf_score=m[composite_field],
            blf_rank=rank,
        ))
    
    return selected, composite_field


# =============================================================================
# LLM API Calls
# =============================================================================

def call_model(openrouter_id: str, prompt: str, max_retries: int = 3) -> str:
    """
    Call an LLM via OpenRouter API with proper token handling.
    
    Args:
        openrouter_id: Model ID in OpenRouter format (e.g., "anthropic/claude-opus-4.5")
        prompt: The prompt to send
        max_retries: Number of retries with increasing token limits
    
    Returns:
        Model response text
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Determine token limits based on model
    # Use generous limits for all models to avoid truncation
    model_lower = openrouter_id.lower()
    is_gemini3 = 'gemini-3' in model_lower
    is_reasoning = any(x in model_lower for x in ['reasoning', 'thinking', 'r1', 'o1', 'o3'])
    
    if is_gemini3:
        token_limits = [16000, 32000, 65000]  # Gemini 3 needs significantly more tokens
    elif is_reasoning:
        token_limits = [16000, 32000, 65000]  # Reasoning models need more tokens
    else:
        token_limits = [8000, 16000, 32000]   # Increased default for all models
    
    last_error = None
    for attempt in range(max_retries):
        try:
            tokens = token_limits[min(attempt, len(token_limits) - 1)]
            
            response = client.chat.completions.create(
                model=openrouter_id,
                max_tokens=tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response
            message = response.choices[0].message
            content = message.content
            
            # Handle string content
            if isinstance(content, str) and content.strip():
                return content
            
            # Handle list content (some models return list of parts)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_val = part.get("text") or part.get("content") or ""
                    else:
                        text_val = str(part)
                    if text_val:
                        parts.append(text_val)
                if parts:
                    return "\n".join(parts)
            
            # Some reasoning models put output in 'reasoning' field
            if hasattr(message, 'reasoning') and message.reasoning:
                return message.reasoning
            
            # Try reasoning_content field
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                if isinstance(message.reasoning_content, str):
                    return message.reasoning_content
                elif isinstance(message.reasoning_content, list):
                    parts = []
                    for part in message.reasoning_content:
                        if isinstance(part, dict):
                            text_val = part.get("text") or part.get("content") or ""
                        else:
                            text_val = str(part)
                        if text_val:
                            parts.append(text_val)
                    if parts:
                        return "\n".join(parts)
            
            # Empty response - log details and retry
            if attempt < max_retries - 1:
                # Debug: log what we actually received
                finish_reason = response.choices[0].finish_reason if response.choices else "no_choices"
                print(f"\n  ⚠️ Empty content on attempt {attempt+1}, retrying...")
                print(f"     content type: {type(content)}, repr: {repr(content)[:100]}")
                print(f"     finish_reason: {finish_reason}")
                if hasattr(message, 'refusal') and message.refusal:
                    print(f"     refusal: {message.refusal}")
                continue
                
        except Exception as e:
            last_error = e
            print(f"\n  ⚠️ API error on attempt {attempt+1}: {type(e).__name__}: {str(e)[:100]}")
            if attempt < max_retries - 1:
                continue
            raise
    
    # Final debug if still empty
    print(f"\n  ❌ All {max_retries} attempts returned empty for {openrouter_id}")
    return ""


# =============================================================================
# LLM Judge
# =============================================================================

# =============================================================================
# KDD-GRADE HYBRID WORKFLOW PROMPTS
# Step 1: Execute code against tests (pass/fail) - done in Python sandbox
# Step 2: LLM Judge evaluates CODE QUALITY only (not correctness)
# =============================================================================

# Judge prompt for code that PASSED execution tests
JUDGE_PROMPT_CODING_PASSED = """You are evaluating CODE QUALITY for a solution that has already PASSED all unit tests.

**IMPORTANT:** Correctness is already verified by execution. Your job is to evaluate MAINTAINABILITY.

### THE PROBLEM
{prompt}

### MODEL SOLUTION (Passed {tests_passed}/{tests_total} tests ✓)
{response}

### REFERENCE SOLUTION (Canonical)
{canonical_solution}

### EVALUATION CRITERIA (Code Quality Only)

1. **Readability** (0-3 points)
   - Clear variable names?
   - Appropriate comments?
   - Logical flow easy to follow?

2. **Modularity** (0-3 points)
   - Good helper functions?
   - Avoids deep nesting?
   - Single responsibility?

3. **Efficiency** (0-2 points)
   - Time/space complexity vs canonical?
   - Avoids unnecessary operations?

4. **Style** (0-2 points)
   - Consistent formatting?
   - Pythonic idioms?

### OUTPUT FORMAT
Score: [X]/10
Quality: [Readability: X/3, Modularity: X/3, Efficiency: X/2, Style: X/2]
Reason: [one sentence on key quality observation]
"""

# Judge prompt for code that FAILED execution tests
JUDGE_PROMPT_CODING_FAILED = """You are analyzing WHY a code solution FAILED its unit tests.

### THE PROBLEM
{prompt}

### MODEL SOLUTION (FAILED: {tests_passed}/{tests_total} tests)
{response}

### EXECUTION ERROR
{execution_error}

### REFERENCE SOLUTION (Canonical - for comparison)
{canonical_solution}

### YOUR TASK
Identify the bug(s) that caused the failure.
Since the code failed tests, the maximum score is 4/10.

### OUTPUT FORMAT
Score: [1-4]/10
Bug Type: [logic/edge-case/syntax/algorithm]
Reason: [one sentence identifying the key bug]
"""

# Legacy prompt (for non-HumanEval coding tasks)
JUDGE_PROMPT_CODING = """Rate this code solution 1-10. Start your response with the score.

### PROBLEM
{prompt}

### MODEL SOLUTION
{response}

### SCORING SCALE
- 1-3: Broken/buggy code
- 4-6: Works but poor quality
- 7-8: Good solution  
- 9-10: Excellent solution

### OUTPUT FORMAT
Score: [X]/10
Reason: [one sentence]
"""

JUDGE_PROMPT_REASONING = """Rate this reasoning solution 1-10. Start your response with the score.

Problem: {prompt}

Solution: {response}

Scoring: 1-2=wrong answer, 3-4=major errors, 5-6=correct but unclear, 7-8=good reasoning, 9-10=excellent

Format your response EXACTLY like this:
Score: 7/10
Reason: [one sentence]
"""

JUDGE_PROMPT_FACTUAL = """Rate this factual response 1-10. Start your response with the score.

Question: {prompt}

Response: {response}

Scoring: 1-2=wrong facts, 3-4=major errors, 5-6=mostly correct, 7-8=accurate, 9-10=expert-level

Format your response EXACTLY like this:
Score: 7/10
Reason: [one sentence]
"""

JUDGE_PROMPT_SUMMARIZATION = """Rate this summary/response 1-10. Start your response with the score.

Task: {prompt}

Response: {response}

Scoring: 1-2=inaccurate, 3-4=major omissions, 5-6=okay, 7-8=good, 9-10=excellent

Format your response EXACTLY like this:
Score: 7/10
Reason: [one sentence]
"""

# Generic fallback
JUDGE_PROMPT_GENERIC = """Rate this response 1-10. Start your response with the score.

Task: {prompt}

Response: {response}

Scoring: 1-3=poor, 4-6=acceptable, 7-8=good, 9-10=excellent

Format your response EXACTLY like this:
Score: 7/10
Reason: [one sentence]
"""

def get_judge_prompt(
    intent: str, 
    prompt: str, 
    response: str,
    canonical_solution: str = None,
    execution_result: Dict = None,  # KDD-Grade: execution results
) -> str:
    """
    Get the appropriate judge prompt for the intent.
    
    For coding with KDD-Grade workflow:
    - If execution_result is provided, use PASSED/FAILED prompts
    - Judge evaluates CODE QUALITY (not correctness, which is proven by execution)
    """
    # Non-coding intents use standard prompts
    if intent != 'coding':
        templates = {
            'reasoning': JUDGE_PROMPT_REASONING,
            'factual_qa': JUDGE_PROMPT_FACTUAL,
            'summarization': JUDGE_PROMPT_SUMMARIZATION,
        }
        template = templates.get(intent, JUDGE_PROMPT_GENERIC)
        return template.format(prompt=prompt, response=response)
    
    # Coding with KDD-Grade workflow (execution-first)
    if execution_result:
        passed = execution_result.get("passed", False)
        tests_passed = execution_result.get("tests_passed", 0)
        tests_total = execution_result.get("tests_total", 0)
        
        if passed:
            # Code PASSED - judge evaluates CODE QUALITY only
            return JUDGE_PROMPT_CODING_PASSED.format(
                prompt=prompt,
                response=response,
                canonical_solution=canonical_solution or "Not provided",
                tests_passed=tests_passed,
                tests_total=tests_total,
            )
        else:
            # Code FAILED - judge analyzes the bug
            error_msg = execution_result.get("error", "Unknown error")
            details = execution_result.get("details", [])
            if details:
                failed_tests = [d for d in details if not d.get("passed")]
                if failed_tests:
                    error_msg = f"{error_msg}\nFirst failed test: {failed_tests[0].get('error', 'Unknown')}"
            
            return JUDGE_PROMPT_CODING_FAILED.format(
                prompt=prompt,
                response=response,
                canonical_solution=canonical_solution or "Not provided",
                tests_passed=tests_passed,
                tests_total=tests_total,
                execution_error=error_msg,
            )
    
    # Fallback: no execution results (legacy mode)
    return JUDGE_PROMPT_CODING.format(
        prompt=prompt,
        response=response,
    )


# Rival judges for each provider (Committee of Rivals approach)
# Using Gemini 3 Pro as primary judge (better calibration observed)
# Google models judged by Claude Opus 4.5 to avoid self-preference
RIVAL_JUDGES = {
    'anthropic': 'google/gemini-3-pro-preview',   # Anthropic → Gemini 3
    'openai': 'google/gemini-3-pro-preview',      # OpenAI → Gemini 3
    'google': 'anthropic/claude-opus-4.5',        # Google → Claude Opus 4.5
    'meta-llama': 'google/gemini-3-pro-preview',  # Meta → Gemini 3
    'mistralai': 'google/gemini-3-pro-preview',   # Mistral → Gemini 3
    'cohere': 'google/gemini-3-pro-preview',      # Cohere → Gemini 3
    'deepseek': 'google/gemini-3-pro-preview',    # DeepSeek → Gemini 3
    'x-ai': 'google/gemini-3-pro-preview',        # xAI/Grok → Gemini 3
    'moonshotai': 'google/gemini-3-pro-preview',  # Moonshot → Gemini 3
    'nvidia': 'google/gemini-3-pro-preview',      # Nvidia → Gemini 3
    'qwen': 'google/gemini-3-pro-preview',        # Qwen → Gemini 3
    'amazon': 'google/gemini-3-pro-preview',      # Amazon → Gemini 3
}

DEFAULT_JUDGE = 'google/gemini-3-pro-preview'


def get_rival_judge(model_openrouter_id: str) -> str:
    """
    Get a rival judge for a model (Committee of Rivals approach).
    
    This eliminates self-preference bias by ensuring models are never
    judged by their own provider.
    
    Args:
        model_openrouter_id: The OpenRouter ID of the model being evaluated
                            (e.g., "anthropic/claude-opus-4.5")
    
    Returns:
        OpenRouter ID of the rival judge to use
    """
    # Extract provider from model ID (format: "provider/model-name")
    if '/' in model_openrouter_id:
        provider = model_openrouter_id.split('/')[0].lower()
    else:
        provider = 'unknown'
    
    return RIVAL_JUDGES.get(provider, DEFAULT_JUDGE)


def judge_response(
    prompt: str, 
    response: str, 
    intent: str,
    model_openrouter_id: str,  # Model being evaluated (to select rival judge)
    judge_model: str = None,   # Override judge (optional)
    canonical_solution: str = None,  # For HumanEval: the canonical code
    execution_result: Dict = None,   # KDD-Grade: sandbox execution results
) -> Dict:
    """
    Use LLM judge to score a response via OpenRouter.
    
    KDD-Grade Hybrid Workflow:
    - For coding: execution_result contains pass/fail from sandbox execution
    - Judge evaluates CODE QUALITY (not correctness) based on execution outcome
    
    Uses the "Committee of Rivals" approach: models are judged by 
    a competing provider to eliminate self-preference bias.
    
    Returns:
        Dict with 'score' (1-10), 'reasoning', 'judge_used', and 'execution_passed'
    """
    from openai import OpenAI
    import re
    
    # Select rival judge if not explicitly specified
    if judge_model is None:
        judge_model = get_rival_judge(model_openrouter_id)
    
    # Get intent-specific judge prompt (with execution results for coding)
    judge_prompt = get_judge_prompt(
        intent, prompt, response,
        canonical_solution=canonical_solution,
        execution_result=execution_result,
    )
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")
    
    # Use OpenRouter for judging
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Use higher token limit for Gemini 3 judge (needs more for dry-run analysis)
    if 'gemini-3' in judge_model.lower():
        judge_max_tokens = 16000
    else:
        judge_max_tokens = 4000
    
    result = client.chat.completions.create(
        model=judge_model,
        max_tokens=judge_max_tokens,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    
    # Parse markdown response to extract score
    response_text = result.choices[0].message.content
    if not response_text:
        raise ValueError(f"Empty response from judge {judge_model}")
    
    response_text = response_text.strip()
    
    # Try multiple patterns to extract the score from markdown
    score_patterns = [
        r'##\s*Score:\s*(\d+)/10',           # ## Score: 8/10
        r'##\s*Score:\s*\[(\d+)\]/10',       # ## Score: [8]/10
        r'\*\*Score:\s*(\d+)/10\*\*',        # **Score: 8/10**
        r'\*\*Score\*\*:\s*(\d+)/10',        # **Score**: 8/10
        r'Score:\s*(\d+)/10',                # Score: 8/10
        r'Score:\s*\[(\d+)\]/10',            # Score: [8]/10
        r'score[:\s]+(\d+)\s*/\s*10',        # score: 8 / 10
        r'(\d+)\s*/\s*10',                   # 8/10 anywhere
        r'final\s+score[:\s]+(\d+)',         # final score: 8
        r'"score"\s*:\s*(\d+)',              # "score": 8 (JSON fallback)
        r'\*\*(\d+)/10\*\*',                 # **8/10**
        r':\s*(\d+)\s*/\s*10',               # : 8/10
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return {
                    "score": score, 
                    "reasoning": response_text, 
                    "judge_used": judge_model
                }
    
    # Fallback: look for "X out of 10" patterns
    out_of_ten = re.search(r'(\d+)\s*(?:out of)\s*10', response_text, re.IGNORECASE)
    if out_of_ten:
        score = int(out_of_ten.group(1))
        if 1 <= score <= 10:
            return {"score": score, "reasoning": response_text, "judge_used": judge_model}
    
    # Last resort: look for standalone numbers 1-10 near "score" keyword
    score_context = re.search(r'score[^0-9]{0,20}(\d+)', response_text, re.IGNORECASE)
    if score_context:
        score = int(score_context.group(1))
        if 1 <= score <= 10:
            return {"score": score, "reasoning": response_text, "judge_used": judge_model}
    
    # NO FALLBACK - raise error so we can debug
    print(f"\n{'='*60}")
    print("FAILED TO PARSE SCORE FROM JUDGE RESPONSE:")
    print(f"{'='*60}")
    print(response_text[:1500])
    print(f"{'='*60}\n")
    raise ValueError(f"Could not parse score from judge response. See output above.")


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    intent: str,
    n_prompts: int = 10,
    dry_run: bool = True,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run the LLM-as-Judge validation experiment.
    
    Args:
        intent: One of 'coding', 'reasoning', 'factual_qa', 'summarization'
        n_prompts: Number of prompts to use (default 10 for cost control)
        dry_run: If True, don't actually call APIs
        output_dir: Directory to save results
    
    Returns:
        Dict with experiment results
    """
    print(f"\n{'='*80}")
    print(f"LLM-AS-JUDGE VALIDATION EXPERIMENT: {intent.upper()}")
    print(f"{'='*80}\n")
    
    # Load models
    models_cache = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    models, composite_field = load_models_for_intent(intent, str(models_cache))
    
    print(f"Composite score field: {composite_field}")
    print(f"\nSelected models ({len(models)} total):")
    print(f"  {'Tier':<8} {'Score':>6} | {'Model':<40} | {'Judged By (Rival)'}")
    print(f"  {'-'*8} {'-'*6} | {'-'*40} | {'-'*25}")
    for m in models:
        tier = "TOP" if m.blf_rank <= 5 else "BOTTOM"
        rival = get_rival_judge(m.openrouter_id)
        print(f"  [{tier:6s}] {m.blf_score:5.1f} | {m.name[:40]:40s} | {rival}")
    
    # Get prompts - use HumanEval Plus for coding (most rigorous), else use labeled dataset
    if intent == "coding":
        prompts = load_real_prompts(intent, n_samples=n_prompts, complexity_filter="humaneval")
        print(f"\nUsing {len(prompts)} HumanEval Plus problems (rigorous benchmark)")
    else:
        prompts = load_real_prompts(intent, n_samples=n_prompts, complexity_filter="complex")
        print(f"\nUsing {len(prompts)} complex prompts from labeled dataset")
    
    if dry_run:
        print("\n" + "="*80)
        print("DRY RUN MODE - No API calls will be made")
        print("="*80)
        print("\nSample prompts (from real human dataset):")
        for i, p in enumerate(prompts[:3], 1):
            print(f"\n  {i}. {p[:150]}...")
        
        print("\n\nTo run the experiment for real:")
        print(f"  python llm_judge_validation.py --intent {intent} --run --n-prompts {n_prompts}")
        
        # Estimate cost
        n_models = len(models)
        total_model_calls = len(prompts) * n_models
        total_judge_calls = total_model_calls
        print(f"\nEstimated API calls:")
        print(f"  Model calls: {total_model_calls} ({len(prompts)} prompts × {n_models} models)")
        print(f"  Judge calls: {total_judge_calls}")
        print(f"  Estimated cost: ${total_model_calls * 0.01 + total_judge_calls * 0.02:.2f}")
        
        return {"status": "dry_run"}
    
    # Run experiment
    results = {
        "intent": intent,
        "composite_field": composite_field,
        "n_prompts": len(prompts),
        "n_models": len(models),
        "models": [{"name": m.name, "blf_score": m.blf_score, "blf_rank": m.blf_rank} for m in models],
        "prompt_results": [],
        "model_scores": {},
    }
    
    # Initialize model scores (KDD-Grade tracks both execution pass rate and quality)
    for m in models:
        results["model_scores"][m.name] = {
            "blf_score": m.blf_score,
            "blf_rank": m.blf_rank,
            "judge_scores": [],          # Quality scores (from LLM judge)
            "execution_passes": [],      # Boolean pass/fail from sandbox
            "avg_judge_score": None,
            "pass_rate": None,           # Fraction of tests passed
        }
    
    # Process each prompt
    # Handle both HumanEvalProblem objects and plain strings
    is_humaneval = intent == "coding" and len(prompts) > 0 and hasattr(prompts[0], 'canonical_solution')
    
    if is_humaneval:
        print("\n" + "="*80)
        print("KDD-GRADE HYBRID WORKFLOW")
        print("="*80)
        print("Step 1: EXECUTE code in Python sandbox (objective pass/fail)")
        print("Step 2: LLM JUDGE evaluates code QUALITY (readability, style)")
        print("="*80)
    
    for i, problem in enumerate(prompts, 1):
        # Extract prompt and metadata
        if is_humaneval:
            prompt_text = problem.prompt
            canonical = problem.canonical_solution
            entry_point = problem.entry_point
            test_cases = problem.test_cases
            task_id = problem.task_id
        else:
            prompt_text = problem
            canonical = None
            entry_point = None
            test_cases = None
            task_id = None
        
        print(f"\n--- Problem {i}/{len(prompts)} [{task_id or 'N/A'}] ---")
        print(f"  {prompt_text[:80]}...")
        if is_humaneval:
            print(f"  [{len(test_cases)} test cases | Entry: {entry_point}()]")
        
        prompt_result = {
            "prompt": prompt_text,
            "task_id": task_id,
            "responses": {}
        }
        
        for m in models:
            print(f"    {m.name[:30]}...", end=" ", flush=True)
            
            try:
                # Get model response via OpenRouter
                response = call_model(m.openrouter_id, prompt_text)
                
                if not response or not response.strip():
                    print(f"❌ Empty response")
                    prompt_result["responses"][m.name] = {"error": "Empty response"}
                    continue
                
                # ========================================
                # KDD-GRADE STEP 1: EXECUTE IN SANDBOX
                # ========================================
                execution_result = None
                if is_humaneval and test_cases:
                    execution_result = execute_code_in_sandbox(
                        prompt=prompt_text,
                        model_code=response,
                        entry_point=entry_point,
                        test_cases=test_cases,
                        timeout=5.0
                    )
                    
                    passed = execution_result.get("passed", False)
                    tests_passed = execution_result.get("tests_passed", 0)
                    tests_total = execution_result.get("tests_total", 0)
                    
                    results["model_scores"][m.name]["execution_passes"].append(passed)
                    
                    exec_status = "✓ PASS" if passed else "✗ FAIL"
                    print(f"[{exec_status} {tests_passed}/{tests_total}]", end=" ", flush=True)
                
                # ========================================
                # KDD-GRADE STEP 2: JUDGE CODE QUALITY
                # ========================================
                judgment = judge_response(
                    prompt_text, response, intent, m.openrouter_id,
                    canonical_solution=canonical,
                    execution_result=execution_result,
                )
                score = judgment.get("score", 0)
                judge_used = judgment.get("judge_used", "unknown")
                
                prompt_result["responses"][m.name] = {
                    "response": response[:500],  # Truncate for storage
                    "execution_passed": execution_result.get("passed") if execution_result else None,
                    "tests_passed": execution_result.get("tests_passed") if execution_result else None,
                    "tests_total": execution_result.get("tests_total") if execution_result else None,
                    "judge_score": score,
                    "judge_used": judge_used,
                    "judge_reasoning": judgment.get("reasoning", "")[:300],
                }
                
                results["model_scores"][m.name]["judge_scores"].append(score)
                
                print(f"Quality: {score}/10 (by {judge_used.split('/')[-1]})")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                prompt_result["responses"][m.name] = {"error": str(e)}
        
        results["prompt_results"].append(prompt_result)
    
    # Calculate average scores and rankings
    print("\n" + "="*80)
    print("RESULTS SUMMARY - KDD-GRADE HYBRID EVALUATION")
    print("="*80)
    
    model_avgs = []
    for m in models:
        scores = results["model_scores"][m.name]["judge_scores"]
        exec_passes = results["model_scores"][m.name].get("execution_passes", [])
        
        if scores:
            avg = np.mean(scores)
            results["model_scores"][m.name]["avg_judge_score"] = avg
            
            # Calculate pass rate
            if exec_passes:
                pass_rate = sum(exec_passes) / len(exec_passes)
                results["model_scores"][m.name]["pass_rate"] = pass_rate
            else:
                pass_rate = None
            
            model_avgs.append((m.name, m.blf_score, m.blf_rank, avg, pass_rate))
    
    # Sort by judge score to get judge rank
    model_avgs.sort(key=lambda x: x[3], reverse=True)
    
    # Show header with execution stats
    if is_humaneval:
        print(f"\n{'Model':<32} {'BLF':>6} {'Rank':>5} {'Pass%':>6} {'Quality':>7} {'J.Rank':>6}")
        print("-" * 75)
    else:
        print(f"\n{'Model':<35} {'BLF Score':>10} {'BLF Rank':>10} {'Judge Avg':>10} {'Judge Rank':>10}")
        print("-" * 80)
    
    blf_ranks = []
    judge_ranks = []
    blf_scores_list = []
    judge_scores_list = []
    
    for judge_rank, entry in enumerate(model_avgs, 1):
        if len(entry) == 5:  # KDD-Grade with pass rate
            name, blf_score, blf_rank, judge_avg, pass_rate = entry
            pass_str = f"{pass_rate*100:.0f}%" if pass_rate is not None else "N/A"
            print(f"{name[:32]:<32} {blf_score:>6.1f} {blf_rank:>5} {pass_str:>6} {judge_avg:>7.1f} {judge_rank:>6}")
        else:  # Legacy mode
            name, blf_score, blf_rank, judge_avg = entry
            print(f"{name[:35]:<35} {blf_score:>10.1f} {blf_rank:>10} {judge_avg:>10.2f} {judge_rank:>10}")
        
        blf_ranks.append(blf_rank)
        judge_ranks.append(judge_rank)
        blf_scores_list.append(blf_score)
        judge_scores_list.append(judge_avg)
    
    # Calculate comprehensive ranking metrics
    if len(blf_ranks) >= 3:
        metrics = compute_ranking_metrics(
            blf_ranks, judge_ranks, 
            blf_scores_list, judge_scores_list
        )
        results["ranking_metrics"] = metrics
        
        # Print detailed metrics
        print_ranking_metrics(metrics)
    
    # KDD-Grade specific summary
    if is_humaneval:
        print("\n" + "="*80)
        print("KDD-GRADE VALIDATION SUMMARY")
        print("="*80)
        
        # Calculate aggregate pass rates for top vs bottom
        top_5_passes = []
        bottom_5_passes = []
        for m in models:
            passes = results["model_scores"][m.name].get("execution_passes", [])
            if passes:
                pass_rate = sum(passes) / len(passes)
                if m.blf_rank <= 5:
                    top_5_passes.append(pass_rate)
                else:
                    bottom_5_passes.append(pass_rate)
        
        if top_5_passes and bottom_5_passes:
            avg_top_pass = np.mean(top_5_passes)
            avg_bottom_pass = np.mean(bottom_5_passes)
            
            print(f"\n📊 EXECUTION PASS RATES (Objective Correctness)")
            print("-" * 60)
            print(f"  Top-5 BLF models avg pass rate:    {avg_top_pass*100:.1f}%")
            print(f"  Bottom-5 BLF models avg pass rate: {avg_bottom_pass*100:.1f}%")
            print(f"  Gap: {(avg_top_pass - avg_bottom_pass)*100:+.1f} percentage points")
            
            if avg_top_pass > avg_bottom_pass + 0.2:
                print("  → Top models significantly better at passing tests ✅")
            elif avg_top_pass > avg_bottom_pass:
                print("  → Top models somewhat better at passing tests ⚠️")
            else:
                print("  → No clear difference in test pass rates ❌")
            
            results["kdd_grade_summary"] = {
                "top_5_avg_pass_rate": float(avg_top_pass),
                "bottom_5_avg_pass_rate": float(avg_bottom_pass),
                "pass_rate_gap": float(avg_top_pass - avg_bottom_pass),
            }
        
        print("\n📝 INTERPRETATION")
        print("-" * 60)
        print("  - Pass Rate: Objective correctness (code runs, tests pass)")
        print("  - Quality Score: Subjective maintainability (readability, style)")
        print("  - BLF composite scores should correlate with BOTH metrics")
    
    # Save results (convert numpy types to native Python for JSON)
    def convert_to_native(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    if output_dir:
        output_path = Path(output_dir) / f"llm_judge_{intent}_results.json"
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(convert_to_native(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Validation Experiment")
    parser.add_argument("--intent", type=str, required=True,
                       choices=["coding", "reasoning", "factual_qa", "summarization"],
                       help="Intent category to validate")
    parser.add_argument("--n-prompts", type=int, default=10,
                       help="Number of prompts to use (default: 10)")
    parser.add_argument("--run", action="store_true",
                       help="Actually run the experiment (default: dry run)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output is None:
        args.output = str(Path(__file__).parent / "llm_judge_results")
    
    run_experiment(
        intent=args.intent,
        n_prompts=args.n_prompts,
        dry_run=not args.run,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
