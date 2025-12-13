#!/usr/bin/env python3
"""
LeetCode Easy vs Hard Validation Script

This script validates that BLF-derived Composite Coding Scores (CCS) differentiate
between simple coding problems (Easy) and algorithmic reasoning (Hard).

Key Hypothesis:
- High CCS models: Small gap between Easy and Hard accuracy
- Low CCS models: Large gap (good on Easy, bad on Hard)

LeetCode Dataset:
- Easy: 540 problems - basic syntax, simple logic
- Medium: 1,281 problems - data structures, standard algorithms
- Hard: 538 problems - dynamic programming, graph theory, advanced algorithms
"""

import os
import sys
import json
import random
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time

# Try to load .env file
try:
    from dotenv import load_dotenv
    for env_path in ['.env', '../.env', '../../.env', '../../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LeetCodeProblem:
    """A LeetCode problem (Easy, Medium, or Hard)."""
    prompt: str
    title: str
    difficulty: str  # "Easy", "Medium", or "Hard"
    problem_id: str
    has_solution: bool = False


@dataclass
class ModelForExperiment:
    """Model selected for the experiment."""
    name: str
    openrouter_id: str
    ccs_score: float
    ccs_rank: int


@dataclass 
class ModelResult:
    """Results for a single model."""
    name: str
    openrouter_id: str
    ccs_score: float
    ccs_rank: int
    easy_correct: int = 0
    easy_total: int = 0
    medium_correct: int = 0
    medium_total: int = 0
    hard_correct: int = 0
    hard_total: int = 0
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0
    easy_hard_gap: float = 0.0  # Easy - Hard (positive = struggles on hard)
    responses: List[Dict] = field(default_factory=list)


# =============================================================================
# LeetCode Dataset Loading
# =============================================================================

def load_leetcode_problems(n_easy: int = 100, n_medium: int = 100, n_hard: int = 100, seed: int = 42) -> Tuple[List[LeetCodeProblem], List[LeetCodeProblem], List[LeetCodeProblem]]:
    """
    Load randomly sampled problems from LeetCode Easy, Medium, and Hard.
    
    Args:
        n_easy: Number of easy problems to sample
        n_medium: Number of medium problems to sample
        n_hard: Number of hard problems to sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (easy_problems, medium_problems, hard_problems)
    """
    from datasets import load_dataset
    
    random.seed(seed)
    
    print("\n📚 Loading LeetCode Dataset from HuggingFace...")
    
    # Load dataset
    leetcode = load_dataset("greengerong/leetcode", split="train")
    
    # Filter by difficulty
    easy_all = [ex for ex in leetcode if ex.get('difficulty') == 'Easy' and ex.get('content')]
    medium_all = [ex for ex in leetcode if ex.get('difficulty') == 'Medium' and ex.get('content')]
    hard_all = [ex for ex in leetcode if ex.get('difficulty') == 'Hard' and ex.get('content')]
    
    print(f"   Easy problems: {len(easy_all)}")
    print(f"   Medium problems: {len(medium_all)}")
    print(f"   Hard problems: {len(hard_all)}")
    
    def format_problems(problems: List, difficulty: str, n_samples: int) -> List[LeetCodeProblem]:
        """Format LeetCode problems with coding prompt."""
        # Random sample
        sampled = random.sample(problems, min(n_samples, len(problems)))
        formatted = []
        
        for item in sampled:
            title = item.get('title', 'Unknown')
            content = item.get('content', '')
            problem_id = item.get('id', item.get('slug', 'unknown'))
            
            # Check if we have a reference solution
            has_solution = bool(item.get('python') or item.get('java') or item.get('c++'))
            
            # Clean up HTML if present
            content = re.sub(r'<[^>]+>', '', content)
            content = content.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>')
            content = content.replace('&amp;', '&').replace('&#39;', "'").replace('&quot;', '"')
            
            # Truncate very long problems
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            # Create coding prompt
            prompt = f"""You are solving a coding problem. Provide a complete, working solution.

**Problem: {title}**
**Difficulty: {difficulty}**

{content}

**Instructions:**
1. Analyze the problem and identify the algorithm/approach needed.
2. Consider edge cases.
3. Write a complete, working solution in Python.
4. Include the function signature and implementation.

Provide your solution:"""

            formatted.append(LeetCodeProblem(
                prompt=prompt.strip(),
                title=title,
                difficulty=difficulty,
                problem_id=f"LC-{problem_id}",
                has_solution=has_solution
            ))
        
        return formatted
    
    easy_problems = format_problems(easy_all, "Easy", n_easy)
    medium_problems = format_problems(medium_all, "Medium", n_medium)
    hard_problems = format_problems(hard_all, "Hard", n_hard)
    
    print(f"   Sampled: {len(easy_problems)} Easy + {len(medium_problems)} Medium + {len(hard_problems)} Hard")
    
    return easy_problems, medium_problems, hard_problems


# =============================================================================
# Model Selection
# =============================================================================

def select_models_for_experiment(
    cache_path: str = "../../data/models_cache.json",
    n_top: int = 10,
    n_bottom: int = 10
) -> List[ModelForExperiment]:
    """
    Select top-N and bottom-N models by CCS for validation.
    """
    # Try multiple paths
    paths_to_try = [
        cache_path,
        "data/models_cache.json",
        "../data/models_cache.json",
        "../../data/models_cache.json",
        "/Users/annette/repostitories/llm_jury/data/models_cache.json"
    ]
    
    data = None
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            break
    
    if data is None:
        raise FileNotFoundError(f"Could not find models_cache.json")
    
    models = data if isinstance(data, list) else data.get('models', [])
    
    # Filter to models with OpenRouter access and CCS score
    api_models = [
        m for m in models 
        if m.get('openrouter_id')
        and m.get('ccs') is not None
    ]
    
    # Sort by CCS (descending)
    api_models.sort(key=lambda m: m['ccs'], reverse=True)
    
    print(f"\n🤖 Model Selection:")
    print(f"   {len(api_models)} models with OpenRouter access and CCS scores")
    
    # Select top and bottom
    top_models = api_models[:n_top]
    bottom_models = api_models[-n_bottom:]
    
    selected = []
    
    # Add top models
    for i, m in enumerate(top_models):
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            ccs_score=m['ccs'],
            ccs_rank=i + 1,
        ))
    
    # Add bottom models
    for i, m in enumerate(bottom_models):
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            ccs_score=m['ccs'],
            ccs_rank=len(api_models) - n_bottom + i + 1,
        ))
    
    print(f"   Selected: Top {n_top} (CCS {top_models[0]['ccs']:.2f} to {top_models[-1]['ccs']:.2f})")
    print(f"   Selected: Bottom {n_bottom} (CCS {bottom_models[0]['ccs']:.2f} to {bottom_models[-1]['ccs']:.2f})")
    
    return selected


# =============================================================================
# LLM API Calls
# =============================================================================

def call_model(openrouter_id: str, prompt: str, max_retries: int = 3) -> str:
    """Call an LLM via OpenRouter API."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Token limits - coding needs more for solutions
    model_lower = openrouter_id.lower()
    is_reasoning_model = any(x in model_lower for x in ['reasoning', 'thinking', 'r1', 'o1', 'o3'])
    
    if is_reasoning_model:
        token_limits = [16000, 32000]
    else:
        token_limits = [8000, 16000]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            tokens = token_limits[min(attempt, len(token_limits) - 1)]
            
            response = client.chat.completions.create(
                model=openrouter_id,
                max_tokens=tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            message = response.choices[0].message
            content = message.content
            
            if isinstance(content, str) and content.strip():
                return content
            
            # Handle list content
            if isinstance(content, list):
                parts = [str(p.get("text", p.get("content", p))) if isinstance(p, dict) else str(p) for p in content]
                if parts:
                    return "\n".join(parts)
            
            # Check reasoning fields
            for field in ['reasoning', 'reasoning_content']:
                if hasattr(message, field) and getattr(message, field):
                    val = getattr(message, field)
                    if isinstance(val, str):
                        return val
            
            if attempt < max_retries - 1:
                continue
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
    
    return ""


# =============================================================================
# Code Evaluation (Simple heuristic-based)
# =============================================================================

def evaluate_code_response(response: str, problem: LeetCodeProblem) -> Dict:
    """
    Evaluate code response using heuristics.
    
    Since we can't execute code, we use:
    1. Code presence check
    2. Function definition check
    3. Common pattern detection
    4. LLM judge for quality assessment
    """
    result = {
        "has_code": False,
        "has_function": False,
        "appears_complete": False,
        "quality_score": 0,
    }
    
    if not response:
        return result
    
    # Check for code block
    code_patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'def \w+\(.*?\):',
        r'class \w+:',
    ]
    
    has_code = any(re.search(p, response, re.DOTALL | re.IGNORECASE) for p in code_patterns)
    result["has_code"] = has_code
    
    # Check for function definition
    has_function = bool(re.search(r'def \w+\(.*?\):', response))
    result["has_function"] = has_function
    
    # Check for common solution patterns
    solution_indicators = [
        r'return\s+',
        r'for\s+\w+\s+in\s+',
        r'while\s+',
        r'if\s+.*:',
        r'\[.*for.*in.*\]',  # list comprehension
    ]
    
    indicator_count = sum(1 for p in solution_indicators if re.search(p, response))
    result["appears_complete"] = has_function and indicator_count >= 2
    
    # Quality score (0-10) based on heuristics
    score = 0
    if has_code:
        score += 3
    if has_function:
        score += 2
    score += min(indicator_count, 5)  # Up to 5 points for patterns
    
    result["quality_score"] = score
    
    return result


def judge_code_quality(response: str, problem: LeetCodeProblem) -> Tuple[int, str]:
    """
    Use LLM judge to evaluate code quality.
    Returns (score 1-10, rationale).
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return 5, "No API key for judging"
    
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    
    judge_prompt = f"""You are a coding interview evaluator. Rate the following solution attempt.

**Problem:** {problem.title} ({problem.difficulty})

**Solution Attempt:**
{response[-3000:]}

**Scoring Criteria:**
- 1-3: No solution or completely wrong approach
- 4-5: Partial solution with major issues
- 6-7: Working approach but may have bugs or inefficiencies
- 8-9: Good solution with correct approach
- 10: Optimal, clean, production-quality code

Rate the solution from 1-10. Reply with ONLY a number."""

    try:
        result = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        score_text = result.choices[0].message.content.strip()
        # Extract number
        match = re.search(r'(\d+)', score_text)
        if match:
            score = int(match.group(1))
            return min(max(score, 1), 10), "LLM judged"
        return 5, "Could not parse score"
        
    except Exception as e:
        return 5, f"Judge error: {str(e)[:50]}"


# =============================================================================
# Main Validation Loop
# =============================================================================

def run_leetcode_validation(
    models: List[ModelForExperiment],
    easy_problems: List[LeetCodeProblem],
    medium_problems: List[LeetCodeProblem],
    hard_problems: List[LeetCodeProblem],
    output_dir: str = "llm_judge_results",
    use_llm_judge: bool = True
) -> Dict:
    """
    Run LeetCode Easy vs Medium vs Hard validation for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_problems = len(easy_problems) + len(medium_problems) + len(hard_problems)
    total_evals = len(models) * total_problems
    
    print(f"\n{'='*100}")
    print(f"LEETCODE EASY vs MEDIUM vs HARD VALIDATION")
    print(f"{'='*100}")
    print(f"Models: {len(models)}")
    print(f"Easy problems: {len(easy_problems)}")
    print(f"Medium problems: {len(medium_problems)}")
    print(f"Hard problems: {len(hard_problems)}")
    print(f"Total API calls: {total_evals:,}")
    print(f"LLM Judge: {'Enabled' if use_llm_judge else 'Disabled'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_models": len(models),
            "n_easy": len(easy_problems),
            "n_medium": len(medium_problems),
            "n_hard": len(hard_problems),
            "total_evals": total_evals,
            "use_llm_judge": use_llm_judge,
        },
        "models": []
    }
    
    model_results: List[ModelResult] = []
    
    completed_evals = 0
    start_time = time.time()
    
    # Process all problems together for better progress tracking
    all_problems = [("easy", p) for p in easy_problems] + [("medium", p) for p in medium_problems] + [("hard", p) for p in hard_problems]
    
    for prob_idx, (difficulty, problem) in enumerate(all_problems):
        elapsed = time.time() - start_time
        if completed_evals > 0:
            avg_time = elapsed / completed_evals
            remaining = (total_evals - completed_evals) * avg_time
            eta = f"{int(remaining//3600)}:{int((remaining%3600)//60):02d}:{int(remaining%60):02d}"
        else:
            eta = "calculating..."
        
        pct = (prob_idx / len(all_problems)) * 100
        
        print(f"\n{'─'*100}")
        print(f"Problem {prob_idx+1}/{len(all_problems)} [{difficulty.upper()}] ({pct:.1f}% complete, ETA: {eta})")
        print(f"Title: {problem.title}")
        print(f"{'─'*100}")
        
        for model in models:
            # Find or create model result
            model_result = None
            for mr in model_results:
                if mr.name == model.name:
                    model_result = mr
                    break
            
            if model_result is None:
                model_result = ModelResult(
                    name=model.name,
                    openrouter_id=model.openrouter_id,
                    ccs_score=model.ccs_score,
                    ccs_rank=model.ccs_rank
                )
                model_results.append(model_result)
            
            # Get model response
            try:
                response = call_model(model.openrouter_id, problem.prompt)
                
                # Evaluate response
                eval_result = evaluate_code_response(response, problem)
                
                # Use LLM judge if enabled
                if use_llm_judge and response:
                    judge_score, _ = judge_code_quality(response, problem)
                else:
                    judge_score = eval_result["quality_score"]
                
                # Consider "correct" if judge score >= 7
                is_correct = judge_score >= 7
                
                # Update counts
                if difficulty == "easy":
                    model_result.easy_total += 1
                    if is_correct:
                        model_result.easy_correct += 1
                elif difficulty == "medium":
                    model_result.medium_total += 1
                    if is_correct:
                        model_result.medium_correct += 1
                else:
                    model_result.hard_total += 1
                    if is_correct:
                        model_result.hard_correct += 1
                
                # Store response
                model_result.responses.append({
                    "problem_id": problem.problem_id,
                    "title": problem.title,
                    "difficulty": difficulty,
                    "judge_score": judge_score,
                    "is_correct": is_correct,
                    "has_code": eval_result["has_code"],
                })
                
                # Progress indicator
                symbol = "✓" if is_correct else "✗"
                current_easy = f"{model_result.easy_correct}/{model_result.easy_total}" if model_result.easy_total else "-"
                current_med = f"{model_result.medium_correct}/{model_result.medium_total}" if model_result.medium_total else "-"
                current_hard = f"{model_result.hard_correct}/{model_result.hard_total}" if model_result.hard_total else "-"
                print(f"  {model.name[:35]:<35} {symbol} score={judge_score} (E:{current_easy} M:{current_med} H:{current_hard})")
                
            except Exception as e:
                print(f"  {model.name[:35]:<35} ❌ Error: {str(e)[:40]}")
                
                if difficulty == "easy":
                    model_result.easy_total += 1
                elif difficulty == "medium":
                    model_result.medium_total += 1
                else:
                    model_result.hard_total += 1
            
            completed_evals += 1
    
    # Calculate final statistics
    print(f"\n{'='*100}")
    print("CALCULATING FINAL RESULTS...")
    print(f"{'='*100}\n")
    
    for mr in model_results:
        if mr.easy_total > 0:
            mr.easy_accuracy = mr.easy_correct / mr.easy_total * 100
        if mr.medium_total > 0:
            mr.medium_accuracy = mr.medium_correct / mr.medium_total * 100
        if mr.hard_total > 0:
            mr.hard_accuracy = mr.hard_correct / mr.hard_total * 100
        mr.easy_hard_gap = mr.easy_accuracy - mr.hard_accuracy
    
    # Sort by CCS for display
    model_results.sort(key=lambda x: x.ccs_score, reverse=True)
    
    # Display results
    print(f"{'Model':<35} {'CCS':>6} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'E-H Gap':>8}")
    print("-" * 85)
    
    for mr in model_results:
        gap_symbol = "📉" if mr.easy_hard_gap > 20 else "📊" if mr.easy_hard_gap > 10 else "✅"
        print(f"{mr.name[:33]:<35} {mr.ccs_score:>6.2f} {mr.easy_accuracy:>6.1f}% {mr.medium_accuracy:>6.1f}% {mr.hard_accuracy:>6.1f}% {mr.easy_hard_gap:>+6.1f}% {gap_symbol}")
    
    # Summary statistics
    top_half = [mr for mr in model_results if mr.ccs_rank <= len(model_results) // 2]
    bottom_half = [mr for mr in model_results if mr.ccs_rank > len(model_results) // 2]
    
    def avg_metric(models_list, attr):
        vals = [getattr(m, attr) for m in models_list if getattr(m, attr) is not None]
        return sum(vals) / len(vals) if vals else 0
    
    print(f"\n{'='*100}")
    print("SUMMARY BY CCS GROUP")
    print(f"{'='*100}")
    print(f"\n{'Group':<20} {'Easy':>10} {'Medium':>10} {'Hard':>10} {'E-H Gap':>10}")
    print("-" * 65)
    print(f"{'High CCS (Top 10)':<20} {avg_metric(top_half, 'easy_accuracy'):>8.1f}% {avg_metric(top_half, 'medium_accuracy'):>8.1f}% {avg_metric(top_half, 'hard_accuracy'):>8.1f}% {avg_metric(top_half, 'easy_hard_gap'):>+8.1f}%")
    print(f"{'Low CCS (Bottom 10)':<20} {avg_metric(bottom_half, 'easy_accuracy'):>8.1f}% {avg_metric(bottom_half, 'medium_accuracy'):>8.1f}% {avg_metric(bottom_half, 'hard_accuracy'):>8.1f}% {avg_metric(bottom_half, 'easy_hard_gap'):>+8.1f}%")
    
    # Correlation analysis
    try:
        from scipy.stats import spearmanr
        
        ccs_scores = [mr.ccs_score for mr in model_results]
        gaps = [mr.easy_hard_gap for mr in model_results]
        hard_accs = [mr.hard_accuracy for mr in model_results]
        medium_accs = [mr.medium_accuracy for mr in model_results]
        
        rho_gap, p_gap = spearmanr(ccs_scores, gaps)
        rho_hard, p_hard = spearmanr(ccs_scores, hard_accs)
        rho_medium, p_medium = spearmanr(ccs_scores, medium_accs)
        
        print(f"\n{'='*100}")
        print("CORRELATION ANALYSIS")
        print(f"{'='*100}")
        print(f"\nSpearman's ρ (CCS vs Easy-Hard Gap): {rho_gap:+.3f} (p={p_gap:.4f})")
        print(f"  → {'✅ SIGNIFICANT' if p_gap < 0.05 else '❌ Not significant'}: ", end="")
        if rho_gap < -0.3:
            print("Higher CCS = smaller gap (better on hard problems)")
        else:
            print("No strong relationship")
        
        print(f"\nSpearman's ρ (CCS vs Medium Accuracy): {rho_medium:+.3f} (p={p_medium:.4f})")
        print(f"  → {'✅ SIGNIFICANT' if p_medium < 0.05 else '❌ Not significant'}")
        
        print(f"\nSpearman's ρ (CCS vs Hard Accuracy): {rho_hard:+.3f} (p={p_hard:.4f})")
        print(f"  → {'✅ SIGNIFICANT' if p_hard < 0.05 else '❌ Not significant'}: ", end="")
        if rho_hard > 0.3:
            print("Higher CCS = better hard problem performance")
        else:
            print("No strong relationship")
            
    except ImportError:
        print("\n(scipy not available for correlation analysis)")
    
    # Save results
    results["models"] = [
        {
            "name": mr.name,
            "openrouter_id": mr.openrouter_id,
            "ccs_score": mr.ccs_score,
            "ccs_rank": mr.ccs_rank,
            "easy_correct": mr.easy_correct,
            "easy_total": mr.easy_total,
            "easy_accuracy": mr.easy_accuracy,
            "medium_correct": mr.medium_correct,
            "medium_total": mr.medium_total,
            "medium_accuracy": mr.medium_accuracy,
            "hard_correct": mr.hard_correct,
            "hard_total": mr.hard_total,
            "hard_accuracy": mr.hard_accuracy,
            "easy_hard_gap": mr.easy_hard_gap,
            "responses": mr.responses,
        }
        for mr in model_results
    ]
    
    results["summary"] = {
        "high_ccs_easy_acc": avg_metric(top_half, 'easy_accuracy'),
        "high_ccs_medium_acc": avg_metric(top_half, 'medium_accuracy'),
        "high_ccs_hard_acc": avg_metric(top_half, 'hard_accuracy'),
        "high_ccs_gap": avg_metric(top_half, 'easy_hard_gap'),
        "low_ccs_easy_acc": avg_metric(bottom_half, 'easy_accuracy'),
        "low_ccs_medium_acc": avg_metric(bottom_half, 'medium_accuracy'),
        "low_ccs_hard_acc": avg_metric(bottom_half, 'hard_accuracy'),
        "low_ccs_gap": avg_metric(bottom_half, 'easy_hard_gap'),
    }
    
    # Save JSON
    output_path = os.path.join(output_dir, "leetcode_difficulty_validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, "leetcode_difficulty_validation_summary.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,CCS,CCS_Rank,Easy_Correct,Easy_Total,Easy_Acc,Medium_Correct,Medium_Total,Medium_Acc,Hard_Correct,Hard_Total,Hard_Acc,Gap\n")
        for mr in model_results:
            f.write(f'"{mr.name}",{mr.ccs_score:.3f},{mr.ccs_rank},{mr.easy_correct},{mr.easy_total},{mr.easy_accuracy:.1f},{mr.medium_correct},{mr.medium_total},{mr.medium_accuracy:.1f},{mr.hard_correct},{mr.hard_total},{mr.hard_accuracy:.1f},{mr.easy_hard_gap:.1f}\n')
    print(f"💾 CSV saved to: {csv_path}")
    
    elapsed_total = time.time() - start_time
    print(f"\n⏱️  Total time: {int(elapsed_total//60)} minutes {int(elapsed_total%60)} seconds")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LeetCode Easy vs Medium vs Hard Validation")
    parser.add_argument("--n-easy", type=int, default=100, help="Number of easy problems")
    parser.add_argument("--n-medium", type=int, default=100, help="Number of medium problems")
    parser.add_argument("--n-hard", type=int, default=100, help="Number of hard problems")
    parser.add_argument("--n-top", type=int, default=10, help="Number of top CCS models")
    parser.add_argument("--n-bottom", type=int, default=10, help="Number of bottom CCS models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="llm_judge_results", help="Output directory")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge (use heuristics only)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print("LEETCODE EASY vs MEDIUM vs HARD VALIDATION")
    print("Testing whether CCS differentiates coding ability")
    print(f"{'='*100}")
    
    # Load problems
    easy_problems, medium_problems, hard_problems = load_leetcode_problems(
        n_easy=args.n_easy,
        n_medium=args.n_medium,
        n_hard=args.n_hard,
        seed=args.seed
    )
    
    # Select models
    models = select_models_for_experiment(
        n_top=args.n_top,
        n_bottom=args.n_bottom
    )
    
    # Run validation
    results = run_leetcode_validation(
        models=models,
        easy_problems=easy_problems,
        medium_problems=medium_problems,
        hard_problems=hard_problems,
        output_dir=args.output_dir,
        use_llm_judge=not args.no_judge
    )
    
    print(f"\n{'='*100}")
    print("VALIDATION COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
