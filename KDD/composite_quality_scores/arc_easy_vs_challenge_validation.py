#!/usr/bin/env python3
"""
ARC Easy vs Challenge Validation Script

This script validates that BLF-derived Composite Reasoning Scores (CRS) differentiate
between simple knowledge recall (ARC-Easy) and deep reasoning (ARC-Challenge).

Key Hypothesis:
- High CRS models: Small gap between Easy and Challenge accuracy
- Low CRS models: Large gap (good on Easy, bad on Challenge)

ARC Dataset (AI2 Reasoning Challenge):
- ARC-Easy: 2,376 questions - simple science facts
- ARC-Challenge: 1,172 questions - multi-step reasoning required
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
class ARCProblem:
    """An ARC problem (Easy or Challenge)."""
    prompt: str
    correct_letter: str  # A, B, C, or D
    question: str
    difficulty: str  # "easy" or "challenge"
    problem_id: str


@dataclass
class ModelForExperiment:
    """Model selected for the experiment."""
    name: str
    openrouter_id: str
    crs_score: float
    crs_rank: int


@dataclass 
class ModelResult:
    """Results for a single model."""
    name: str
    openrouter_id: str
    crs_score: float
    crs_rank: int
    easy_correct: int = 0
    easy_total: int = 0
    challenge_correct: int = 0
    challenge_total: int = 0
    easy_accuracy: float = 0.0
    challenge_accuracy: float = 0.0
    accuracy_gap: float = 0.0  # Easy - Challenge (positive = struggles on hard)
    responses: List[Dict] = field(default_factory=list)


# =============================================================================
# ARC Dataset Loading
# =============================================================================

def load_arc_problems(n_easy: int = 50, n_challenge: int = 50, seed: int = 42) -> Tuple[List[ARCProblem], List[ARCProblem]]:
    """
    Load randomly sampled problems from ARC-Easy and ARC-Challenge.
    
    Args:
        n_easy: Number of easy problems to sample
        n_challenge: Number of challenge problems to sample
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (easy_problems, challenge_problems)
    """
    from datasets import load_dataset
    
    random.seed(seed)
    
    print("\n📚 Loading ARC Dataset from HuggingFace...")
    
    # Load both splits
    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    print(f"   ARC-Easy: {len(arc_easy):,} questions")
    print(f"   ARC-Challenge: {len(arc_challenge):,} questions")
    
    def format_problems(dataset, difficulty: str, n_samples: int) -> List[ARCProblem]:
        """Format ARC problems with Chain-of-Thought prompt."""
        # Random sample
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        problems = []
        
        for idx in indices:
            item = dataset[idx]
            question = item['question']
            choices = item['choices']
            answer_key = item['answerKey']
            
            # Format options
            options_text = ""
            for label, text in zip(choices['label'], choices['text']):
                options_text += f"{label}. {text}\n"
            
            # Chain-of-Thought prompt
            prompt = f"""You are answering a science question. Think step-by-step before giving your answer.

**Question:**
{question}

**Options:**
{options_text}

**Instructions:**
1. Consider what scientific concepts are relevant.
2. Analyze each option carefully.
3. State your final answer in the format: "Answer: X" (where X is A, B, C, or D)

Think through this problem before answering."""

            problems.append(ARCProblem(
                prompt=prompt.strip(),
                correct_letter=answer_key,
                question=question[:150],
                difficulty=difficulty,
                problem_id=f"ARC-{difficulty.upper()}/{item['id']}"
            ))
        
        return problems
    
    easy_problems = format_problems(arc_easy, "easy", n_easy)
    challenge_problems = format_problems(arc_challenge, "challenge", n_challenge)
    
    print(f"   Sampled: {len(easy_problems)} Easy + {len(challenge_problems)} Challenge")
    
    return easy_problems, challenge_problems


# =============================================================================
# Model Selection
# =============================================================================

def select_models_for_experiment(
    cache_path: str = "../../data/models_cache.json",
    n_top: int = 10,
    n_bottom: int = 10
) -> List[ModelForExperiment]:
    """
    Select top-N and bottom-N models by CRS for validation.
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
    
    # Filter to models with OpenRouter access and CRS score
    api_models = [
        m for m in models 
        if m.get('openrouter_id')
        and m.get('crs') is not None
    ]
    
    # Sort by CRS (descending)
    api_models.sort(key=lambda m: m['crs'], reverse=True)
    
    print(f"\n🤖 Model Selection:")
    print(f"   {len(api_models)} models with OpenRouter access and CRS scores")
    
    # Select top and bottom
    top_models = api_models[:n_top]
    bottom_models = api_models[-n_bottom:]
    
    selected = []
    
    # Add top models
    for i, m in enumerate(top_models):
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            crs_score=m['crs'],
            crs_rank=i + 1,
        ))
    
    # Add bottom models
    for i, m in enumerate(bottom_models):
        selected.append(ModelForExperiment(
            name=m['name'],
            openrouter_id=m['openrouter_id'],
            crs_score=m['crs'],
            crs_rank=len(api_models) - n_bottom + i + 1,
        ))
    
    print(f"   Selected: Top {n_top} (CRS {top_models[0]['crs']:.2f} to {top_models[-1]['crs']:.2f})")
    print(f"   Selected: Bottom {n_bottom} (CRS {bottom_models[0]['crs']:.2f} to {bottom_models[-1]['crs']:.2f})")
    
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
    
    # Token limits
    model_lower = openrouter_id.lower()
    is_reasoning_model = any(x in model_lower for x in ['reasoning', 'thinking', 'r1', 'o1', 'o3'])
    
    if is_reasoning_model:
        token_limits = [16000, 32000]
    else:
        token_limits = [4000, 8000]
    
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
# Answer Extraction
# =============================================================================

def extract_letter_answer(response: str) -> Optional[str]:
    """Extract multiple choice letter (A, B, C, D) from response."""
    if not response:
        return None
    
    response_upper = response.upper()
    
    # Pattern 1: Explicit "Answer: X" format (preferred)
    patterns = [
        r'ANSWER:\s*\(?([A-D])\)?',
        r'FINAL ANSWER:\s*\(?([A-D])\)?',
        r'THE ANSWER IS\s*\(?([A-D])\)?',
        r'CORRECT ANSWER IS\s*\(?([A-D])\)?',
        r'I (?:CHOOSE|SELECT|PICK)\s*\(?([A-D])\)?',
        r'OPTION\s*([A-D])\s*IS (?:CORRECT|THE ANSWER)',
        r'\*\*([A-D])\*\*\s*$',  # Bold letter at end
        r'THEREFORE[,\s]+([A-D])\b',
        r'SO[,\s]+THE ANSWER IS\s*([A-D])\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    # Fallback: Look for isolated letter near the end
    last_300 = response_upper[-300:] if len(response_upper) > 300 else response_upper
    match = re.search(r'\b([A-D])\b[.\s]*$', last_300)
    if match:
        return match.group(1)
    
    # Last resort: any A, B, C, D in final sentence
    match = re.search(r'\b([A-D])\b', last_300)
    if match:
        return match.group(1)
    
    return None


# =============================================================================
# Main Validation Loop
# =============================================================================

def run_arc_validation(
    models: List[ModelForExperiment],
    easy_problems: List[ARCProblem],
    challenge_problems: List[ARCProblem],
    output_dir: str = "llm_judge_results"
) -> Dict:
    """
    Run ARC Easy vs Challenge validation for all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_problems = len(easy_problems) + len(challenge_problems)
    total_evals = len(models) * total_problems
    
    print(f"\n{'='*100}")
    print(f"ARC EASY vs CHALLENGE VALIDATION")
    print(f"{'='*100}")
    print(f"Models: {len(models)}")
    print(f"Easy problems: {len(easy_problems)}")
    print(f"Challenge problems: {len(challenge_problems)}")
    print(f"Total API calls: {total_evals:,}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_models": len(models),
            "n_easy": len(easy_problems),
            "n_challenge": len(challenge_problems),
            "total_evals": total_evals,
        },
        "models": []
    }
    
    model_results: List[ModelResult] = []
    
    completed_evals = 0
    start_time = time.time()
    
    # Process all problems together for better progress tracking
    all_problems = [("easy", p) for p in easy_problems] + [("challenge", p) for p in challenge_problems]
    
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
        print(f"Q: {problem.question[:80]}...")
        print(f"Correct: {problem.correct_letter}")
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
                    crs_score=model.crs_score,
                    crs_rank=model.crs_rank
                )
                model_results.append(model_result)
            
            # Get model response
            try:
                response = call_model(model.openrouter_id, problem.prompt)
                extracted = extract_letter_answer(response)
                is_correct = extracted == problem.correct_letter
                
                # Update counts
                if difficulty == "easy":
                    model_result.easy_total += 1
                    if is_correct:
                        model_result.easy_correct += 1
                else:
                    model_result.challenge_total += 1
                    if is_correct:
                        model_result.challenge_correct += 1
                
                # Store response
                model_result.responses.append({
                    "problem_id": problem.problem_id,
                    "difficulty": difficulty,
                    "correct_letter": problem.correct_letter,
                    "extracted": extracted,
                    "is_correct": is_correct,
                })
                
                # Progress indicator
                symbol = "✓" if is_correct else "✗"
                current_easy = f"{model_result.easy_correct}/{model_result.easy_total}" if model_result.easy_total else "-"
                current_chal = f"{model_result.challenge_correct}/{model_result.challenge_total}" if model_result.challenge_total else "-"
                print(f"  {model.name[:35]:<35} {symbol} (E:{current_easy} C:{current_chal})")
                
            except Exception as e:
                print(f"  {model.name[:35]:<35} ❌ Error: {str(e)[:40]}")
                
                if difficulty == "easy":
                    model_result.easy_total += 1
                else:
                    model_result.challenge_total += 1
            
            completed_evals += 1
    
    # Calculate final statistics
    print(f"\n{'='*100}")
    print("CALCULATING FINAL RESULTS...")
    print(f"{'='*100}\n")
    
    for mr in model_results:
        if mr.easy_total > 0:
            mr.easy_accuracy = mr.easy_correct / mr.easy_total * 100
        if mr.challenge_total > 0:
            mr.challenge_accuracy = mr.challenge_correct / mr.challenge_total * 100
        mr.accuracy_gap = mr.easy_accuracy - mr.challenge_accuracy
    
    # Sort by CRS for display
    model_results.sort(key=lambda x: x.crs_score, reverse=True)
    
    # Display results
    print(f"{'Model':<40} {'CRS':>6} {'Easy':>10} {'Challenge':>10} {'Gap':>8}")
    print("-" * 80)
    
    for mr in model_results:
        gap_symbol = "📉" if mr.accuracy_gap > 10 else "📊" if mr.accuracy_gap > 5 else "✅"
        print(f"{mr.name[:38]:<40} {mr.crs_score:>6.2f} {mr.easy_accuracy:>8.1f}% {mr.challenge_accuracy:>8.1f}% {mr.accuracy_gap:>+6.1f}% {gap_symbol}")
    
    # Summary statistics
    top_half = [mr for mr in model_results if mr.crs_rank <= len(model_results) // 2]
    bottom_half = [mr for mr in model_results if mr.crs_rank > len(model_results) // 2]
    
    def avg_metric(models_list, attr):
        vals = [getattr(m, attr) for m in models_list if getattr(m, attr) is not None]
        return sum(vals) / len(vals) if vals else 0
    
    print(f"\n{'='*100}")
    print("SUMMARY BY CRS GROUP")
    print(f"{'='*100}")
    print(f"\n{'Group':<20} {'Easy Acc':>12} {'Challenge Acc':>14} {'Avg Gap':>10}")
    print("-" * 60)
    print(f"{'High CRS (Top 10)':<20} {avg_metric(top_half, 'easy_accuracy'):>10.1f}% {avg_metric(top_half, 'challenge_accuracy'):>12.1f}% {avg_metric(top_half, 'accuracy_gap'):>+8.1f}%")
    print(f"{'Low CRS (Bottom 10)':<20} {avg_metric(bottom_half, 'easy_accuracy'):>10.1f}% {avg_metric(bottom_half, 'challenge_accuracy'):>12.1f}% {avg_metric(bottom_half, 'accuracy_gap'):>+8.1f}%")
    
    # Correlation analysis
    try:
        from scipy.stats import spearmanr, pearsonr
        
        crs_scores = [mr.crs_score for mr in model_results]
        gaps = [mr.accuracy_gap for mr in model_results]
        challenge_accs = [mr.challenge_accuracy for mr in model_results]
        
        rho_gap, p_gap = spearmanr(crs_scores, gaps)
        rho_chal, p_chal = spearmanr(crs_scores, challenge_accs)
        
        print(f"\n{'='*100}")
        print("CORRELATION ANALYSIS")
        print(f"{'='*100}")
        print(f"\nSpearman's ρ (CRS vs Accuracy Gap): {rho_gap:+.3f} (p={p_gap:.4f})")
        print(f"  → {'✅ SIGNIFICANT' if p_gap < 0.05 else '❌ Not significant'}: ", end="")
        if rho_gap < -0.3:
            print("Higher CRS = smaller gap (better reasoning)")
        else:
            print("No strong relationship")
        
        print(f"\nSpearman's ρ (CRS vs Challenge Accuracy): {rho_chal:+.3f} (p={p_chal:.4f})")
        print(f"  → {'✅ SIGNIFICANT' if p_chal < 0.05 else '❌ Not significant'}: ", end="")
        if rho_chal > 0.3:
            print("Higher CRS = better challenge performance")
        else:
            print("No strong relationship")
            
    except ImportError:
        print("\n(scipy not available for correlation analysis)")
    
    # Save results
    results["models"] = [
        {
            "name": mr.name,
            "openrouter_id": mr.openrouter_id,
            "crs_score": mr.crs_score,
            "crs_rank": mr.crs_rank,
            "easy_correct": mr.easy_correct,
            "easy_total": mr.easy_total,
            "easy_accuracy": mr.easy_accuracy,
            "challenge_correct": mr.challenge_correct,
            "challenge_total": mr.challenge_total,
            "challenge_accuracy": mr.challenge_accuracy,
            "accuracy_gap": mr.accuracy_gap,
            "responses": mr.responses,
        }
        for mr in model_results
    ]
    
    results["summary"] = {
        "high_crs_easy_acc": avg_metric(top_half, 'easy_accuracy'),
        "high_crs_challenge_acc": avg_metric(top_half, 'challenge_accuracy'),
        "high_crs_gap": avg_metric(top_half, 'accuracy_gap'),
        "low_crs_easy_acc": avg_metric(bottom_half, 'easy_accuracy'),
        "low_crs_challenge_acc": avg_metric(bottom_half, 'challenge_accuracy'),
        "low_crs_gap": avg_metric(bottom_half, 'accuracy_gap'),
    }
    
    # Save JSON
    output_path = os.path.join(output_dir, "arc_easy_vs_challenge_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, "arc_easy_vs_challenge_summary.csv")
    with open(csv_path, 'w') as f:
        f.write("Model,CRS,CRS_Rank,Easy_Correct,Easy_Total,Easy_Accuracy,Challenge_Correct,Challenge_Total,Challenge_Accuracy,Gap\n")
        for mr in model_results:
            f.write(f'"{mr.name}",{mr.crs_score:.3f},{mr.crs_rank},{mr.easy_correct},{mr.easy_total},{mr.easy_accuracy:.1f},{mr.challenge_correct},{mr.challenge_total},{mr.challenge_accuracy:.1f},{mr.accuracy_gap:.1f}\n')
    print(f"💾 CSV saved to: {csv_path}")
    
    elapsed_total = time.time() - start_time
    print(f"\n⏱️  Total time: {int(elapsed_total//60)} minutes {int(elapsed_total%60)} seconds")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ARC Easy vs Challenge Validation")
    parser.add_argument("--n-easy", type=int, default=50, help="Number of easy problems")
    parser.add_argument("--n-challenge", type=int, default=50, help="Number of challenge problems")
    parser.add_argument("--n-top", type=int, default=10, help="Number of top CRS models")
    parser.add_argument("--n-bottom", type=int, default=10, help="Number of bottom CRS models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="llm_judge_results", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print("ARC EASY vs CHALLENGE VALIDATION")
    print("Testing whether CRS differentiates reasoning ability")
    print(f"{'='*100}")
    
    # Load problems
    easy_problems, challenge_problems = load_arc_problems(
        n_easy=args.n_easy,
        n_challenge=args.n_challenge,
        seed=args.seed
    )
    
    # Select models
    models = select_models_for_experiment(
        n_top=args.n_top,
        n_bottom=args.n_bottom
    )
    
    # Run validation
    results = run_arc_validation(
        models=models,
        easy_problems=easy_problems,
        challenge_problems=challenge_problems,
        output_dir=args.output_dir
    )
    
    print(f"\n{'='*100}")
    print("VALIDATION COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
