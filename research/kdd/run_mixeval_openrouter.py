#!/usr/bin/env python3
"""
Run MixEval on LLM Jury Models via OpenRouter.

This script evaluates models with hallucination data and full benchmark coverage
using MixEval through OpenRouter API.

Requirements:
    - OPENROUTER_API_KEY environment variable set
    - OPENAI_API_KEY for model parsing (judging responses)

Usage:
    # Run on all qualified models
    python kdd_paper/run_mixeval_openrouter.py --all
    
    # Run on specific models
    python kdd_paper/run_mixeval_openrouter.py --models openai/gpt-4o anthropic/claude-3.5-sonnet
    
    # Dry run (show what would be evaluated)
    python kdd_paper/run_mixeval_openrouter.py --dry-run
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MIXEVAL_PATH = PROJECT_ROOT / "external" / "MixEval"
DATA_PATH = PROJECT_ROOT / "data"

sys.path.insert(0, str(MIXEVAL_PATH))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    openrouter_id: str
    hallucination_rate: float
    
    @property
    def mixeval_name(self) -> str:
        """Convert to MixEval-compatible name."""
        return self.openrouter_id.replace("/", "_").replace("-", "_").replace(".", "_")


def load_qualified_models(benchmark: str = "mixeval") -> List[ModelConfig]:
    """Load models with hallucination data and benchmarks.
    
    Args:
        benchmark: "mixeval" or "mixeval-hard" - filters out models that already have scores
        
    Returns:
        List of models needing evaluation for the specified benchmark
    """
    cache_path = DATA_PATH / "models_cache.json"
    
    # Load scores for the requested benchmark
    if benchmark == "mixeval":
        scores_path = DATA_PATH / "mixeval_scores.json"
    else:
        scores_path = DATA_PATH / "mixeval_hard_scores.json"
    
    with open(cache_path) as f:
        data = json.load(f)
    
    # Load existing scores (or empty dict if file doesn't exist)
    existing_scores = {}
    if scores_path.exists():
        with open(scores_path) as f:
            existing_scores = json.load(f)
    
    models = data.get("models", data)
    benchmarks_required = ['intelligence_index', 'math_index', 'mmlu_pro', 'gpqa', 'livecodebench']
    
    qualified = []
    skipped_have_score = 0
    
    for m in models:
        # Must have hallucination rate
        halluc = m.get('hallucination_rate')
        if not halluc or float(halluc) <= 0:
            continue
        
        # Must have OpenRouter ID
        openrouter_id = m.get('openrouter_id', '')
        if not openrouter_id:
            continue
        
        # Skip if already has score for this benchmark
        if openrouter_id in existing_scores:
            skipped_have_score += 1
            continue
        
        # Must have most benchmarks
        bench_count = sum(1 for b in benchmarks_required if m.get(b) and float(m.get(b, 0)) > 0)
        if bench_count < 4:
            continue
        
        qualified.append(ModelConfig(
            name=m.get('name', ''),
            openrouter_id=openrouter_id,
            hallucination_rate=float(halluc),
        ))
    
    if skipped_have_score > 0:
        print(f"  (Skipped {skipped_have_score} models that already have {benchmark} scores)")
    
    return qualified


def setup_openrouter_env():
    """Ensure OpenRouter API key is set."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Get one from https://openrouter.ai/keys"
        )
    
    # Also need OpenAI key for parsing
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set - needed for response parsing")
    
    return api_key


def run_mixeval_for_model(
    model: ModelConfig,
    benchmark: str = "mixeval",
    version: str = "2024-08-11",
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    max_samples: int = 100,
) -> Optional[Dict]:
    """
    Run MixEval on a single model via OpenRouter.
    
    Returns:
        Dict with results or None if failed
    """
    if output_dir is None:
        output_dir = MIXEVAL_PATH / "mix_eval" / "data" / "model_responses"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        print(f"  [DRY RUN] Would evaluate {model.openrouter_id}")
        return {"status": "dry_run", "model": model.openrouter_id}
    
    try:
        from openai import OpenAI
        
        # Create OpenRouter client
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        
        # Load benchmark data - try different version paths
        benchmark_path = None
        for v in [version, "2024-08-11", "2024-06-01"]:
            test_path = MIXEVAL_PATH / "mix_eval" / "data" / f"mixeval-{v}"
            if test_path.exists():
                # Select mixeval or mixeval-hard subfolder
                bench_folder = "mixeval" if benchmark == "mixeval" else "mixeval-hard"
                benchmark_path = test_path / bench_folder
                if benchmark_path.exists():
                    break
                benchmark_path = None
        
        if not benchmark_path:
            print(f"  ❌ Benchmark data not found in {MIXEVAL_PATH / 'mix_eval' / 'data'}")
            return None
        
        # Run evaluation
        results = evaluate_model_openrouter(
            client=client,
            model_id=model.openrouter_id,
            benchmark_path=benchmark_path,
            output_dir=output_dir / model.mixeval_name,
            max_samples=max_samples,
        )
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model_openrouter(
    client,
    model_id: str,
    benchmark_path: Path,
    output_dir: Path,
    max_samples: int = 100,  # Limit for cost control (100 = quick test, 500 = full)
) -> Dict:
    """
    Evaluate a model using OpenRouter API.
    
    This is a simplified evaluation that:
    1. Loads MixEval questions
    2. Gets model responses via OpenRouter
    3. Saves responses for later scoring
    """
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_id": model_id,
        "responses": [],
        "errors": [],
    }
    
    # Load free-form questions
    # Try both possible paths (mixeval vs mixeval-hard subfolder)
    ff_path = benchmark_path / "free-form.json"
    mc_path = benchmark_path / "multiple-choice.json"
    
    # Fallback to subfolder structure
    if not ff_path.exists():
        ff_path = benchmark_path / "mixeval" / "free-form.json"
        mc_path = benchmark_path / "mixeval" / "multiple-choice.json"
    
    questions = []
    
    if ff_path.exists():
        with open(ff_path) as f:
            ff_data = json.load(f)
            # Handle both dict and list formats
            if isinstance(ff_data, dict):
                items = list(ff_data.values())[:max_samples//2]
            else:
                items = ff_data[:max_samples//2]
            questions.extend([{"type": "free-form", "id": i, **q} for i, q in enumerate(items)])
    
    if mc_path.exists():
        with open(mc_path) as f:
            mc_data = json.load(f)
            # Handle both dict and list formats
            if isinstance(mc_data, dict):
                items = list(mc_data.values())[:max_samples//2]
            else:
                items = mc_data[:max_samples//2]
            questions.extend([{"type": "multiple-choice", "id": i + len(questions), **q} for i, q in enumerate(items)])
    
    total_questions = len(questions)
    print(f"\n  📋 Loaded {total_questions} questions")
    print(f"  🚀 Starting evaluation of {model_id}...")
    print()
    
    # Process questions with progress bar
    for i, q in enumerate(questions):
        pct = (i + 1) / total_questions * 100
        bar_len = 30
        filled = int(bar_len * (i + 1) / total_questions)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # Print progress on same line
        print(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{total_questions}) ", end="", flush=True)
        
        try:
            # Format prompt
            if q["type"] == "multiple-choice":
                prompt = format_mc_prompt(q)
            else:
                prompt = q.get("prompt", q.get("question", ""))
            
            # Get response
            # OpenAI reasoning models (o1, o3, o4) require max_completion_tokens
            is_reasoning_model = any(x in model_id.lower() for x in ['/o1', '/o3', '/o4-'])
            
            if is_reasoning_model:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                )
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                )
            
            answer = response.choices[0].message.content
            
            results["responses"].append({
                "id": q.get("id", i),
                "type": q["type"],
                "prompt": prompt,
                "response": answer,
                "expected": q.get("answer", q.get("target", "")),
            })
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            results["errors"].append({
                "id": q.get("id", i),
                "error": str(e),
            })
            time.sleep(1)  # Back off on errors
    
    # Print final newline after progress bar
    print()
    
    # Save responses
    output_file = output_dir / "responses.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate preliminary score
    correct = sum(1 for r in results["responses"] if r.get("correct", False))
    total = len(results["responses"])
    score = (correct / total * 100) if total > 0 else 0
    
    print(f"\n  ✅ Completed: {len(results['responses'])} responses")
    print(f"  ❌ Errors: {len(results['errors'])}")
    print(f"  💾 Saved to: {output_file}")
    
    return results


def format_mc_prompt(question: Dict) -> str:
    """Format multiple-choice question."""
    prompt = question.get("prompt", question.get("question", ""))
    choices = question.get("choices", [])
    
    if choices:
        prompt += "\n\nChoices:\n"
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            prompt += f"{letter}. {choice}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D)."
    
    return prompt


def compute_scores(results_dir: Path) -> Dict[str, float]:
    """
    Compute MixEval scores from saved responses.
    
    This uses simple exact-match scoring for now.
    For full scoring, use MixEval's compute_metrics module.
    """
    scores = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        response_file = model_dir / "responses.json"
        if not response_file.exists():
            continue
        
        with open(response_file) as f:
            data = json.load(f)
        
        correct = 0
        total = 0
        
        for r in data.get("responses", []):
            response = r.get("response", "").strip()
            expected = r.get("expected", "")
            
            # Handle list of valid answers
            if isinstance(expected, list):
                expected_list = [str(e).strip().lower() for e in expected]
            else:
                expected_list = [str(expected).strip().lower()]
            
            if r["type"] == "multiple-choice":
                # Simple letter matching
                resp_upper = response.upper()
                
                # Extract just the letter
                if resp_upper and resp_upper[0] in "ABCD":
                    resp_letter = resp_upper[0]
                else:
                    resp_letter = resp_upper
                
                # Check if any expected answer matches
                is_correct = any(
                    resp_letter == e.upper()[0] if e and e[0].upper() in "ABCD" else resp_letter == e.upper()
                    for e in expected_list
                )
                
                if is_correct:
                    correct += 1
                total += 1
            else:
                # Free-form: check if response contains any expected answer
                resp_lower = response.lower()
                is_correct = any(exp in resp_lower for exp in expected_list)
                
                if is_correct:
                    correct += 1
                total += 1
        
        if total > 0:
            score = (correct / total) * 100
            model_id = data.get("model_id", model_dir.name)
            scores[model_id] = score
            logger.info(f"  {model_id}: {score:.1f}% ({correct}/{total})")
    
    return scores


def save_mixeval_scores(scores: Dict[str, float], benchmark: str = "mixeval"):
    """Save MixEval scores to data directory."""
    # Use different file for mixeval vs mixeval-hard
    if benchmark == "mixeval":
        output_file = DATA_PATH / "mixeval_scores.json"
    else:
        output_file = DATA_PATH / "mixeval_hard_scores.json"
    
    # Load existing scores if any
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    
    # Merge
    existing.update(scores)
    
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    
    logger.info(f"Saved {len(scores)} scores to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run MixEval via OpenRouter")
    parser.add_argument("--all", action="store_true", help="Run on all qualified models")
    parser.add_argument("--models", nargs="+", help="Specific OpenRouter model IDs")
    parser.add_argument("--benchmark", default="mixeval", choices=["mixeval", "mixeval-hard"])
    parser.add_argument("--version", default="2024-08-11")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--compute-only", action="store_true", help="Only compute scores from existing responses")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models")
    parser.add_argument("--samples", type=int, default=100, help="Questions per model (default: 100, full: 500)")
    parser.add_argument("--threads", type=int, default=3, 
                        help="Number of parallel threads for model evaluation (default: 3)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MIXEVAL EVALUATION VIA OPENROUTER")
    print("=" * 70)
    
    # Load qualified models (filtered by benchmark - skips models that already have scores)
    all_models = load_qualified_models(args.benchmark)
    print(f"Qualified models with OpenRouter IDs: {len(all_models)}")
    
    # Filter to requested models
    if args.models:
        models = [m for m in all_models if m.openrouter_id in args.models]
    elif args.all:
        models = all_models
    else:
        print("\nSpecify --all or --models <model_ids>")
        print("\nAvailable models:")
        for m in all_models[:10]:
            print(f"  {m.openrouter_id}")
        print(f"  ... and {len(all_models) - 10} more")
        return
    
    if args.max_models:
        models = models[:args.max_models]
    
    print(f"\nModels to evaluate: {len(models)}")
    
    if args.compute_only:
        # Just compute scores from existing responses
        results_dir = MIXEVAL_PATH / "mix_eval" / "data" / "model_responses"
        scores = compute_scores(results_dir)
        save_mixeval_scores(scores, args.benchmark)
        return
    
    # Check API keys
    if not args.dry_run:
        setup_openrouter_env()
    
    # Run evaluations
    print("\n" + "=" * 70)
    print(f"RUNNING EVALUATIONS (threads={args.threads})")
    print("=" * 70)
    print(f"\nTotal models to evaluate: {len(models)}")
    
    results = {}
    results_lock = threading.Lock()
    
    def evaluate_task(model):
        """Wrapper for thread pool execution."""
        result = run_mixeval_for_model(
            model,
            benchmark=args.benchmark,
            version=args.version,
            dry_run=args.dry_run,
            max_samples=args.samples,
        )
        return model, result
    
    # Use ThreadPoolExecutor for parallel model evaluation
    completed = 0
    total = len(models)
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(evaluate_task, model): model for model in models}
        
        for future in as_completed(futures):
            model = futures[future]
            try:
                model_result, result = future.result()
                if result:
                    with results_lock:
                        results[model_result.openrouter_id] = result
                completed += 1
                print(f"\n[{completed}/{total}] Completed: {model.name}")
            except Exception as e:
                completed += 1
                logger.error(f"[{completed}/{total}] Failed {model.name}: {e}")
    
    # Compute and save scores
    if not args.dry_run:
        print("\n" + "=" * 70)
        print("COMPUTING SCORES")
        print("=" * 70)
        
        results_dir = MIXEVAL_PATH / "mix_eval" / "data" / "model_responses"
        scores = compute_scores(results_dir)
        save_mixeval_scores(scores, args.benchmark)
    
    # Determine output file based on benchmark
    scores_file = "mixeval_scores.json" if args.benchmark == "mixeval" else "mixeval_hard_scores.json"
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"""
Results saved to:
  - Responses: external/MixEval/mix_eval/data/model_responses/
  - Scores: data/{scores_file}

To use in quality scoring:
  from llm_jury.optimization.intent_quality import IntentQualityScorer
  scorer = IntentQualityScorer(models, quality_target="mixeval_score")
""")


if __name__ == "__main__":
    main()

