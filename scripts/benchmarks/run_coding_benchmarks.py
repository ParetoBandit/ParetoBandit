#!/usr/bin/env python3
"""
Run HumanEval and MBPP coding benchmarks for all models in the cache.

This script:
1. Loads problems from the cloned repos (external/human-eval, external/google-research-mbpp)
2. Calls models via OpenRouter to generate code completions
3. Executes generated code against test suites
4. Calculates pass@1 scores and saves to data/

Usage:
    # Evaluate all models on both benchmarks
    python scripts/run_coding_benchmarks.py --all
    
    # Evaluate specific models
    python scripts/run_coding_benchmarks.py --models "gpt-4o,claude-3.5-sonnet"
    
    # Only HumanEval
    python scripts/run_coding_benchmarks.py --humaneval --all
    
    # Only MBPP
    python scripts/run_coding_benchmarks.py --mbpp --all
    
    # Dry run (show models without evaluating)
    python scripts/run_coding_benchmarks.py --dry-run

WARNING: This script executes model-generated code. While safety guards are
in place, it's recommended to run in a sandboxed environment (Docker, VM).
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"

sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Direct import to avoid package __init__ issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "coding_benchmarks_client", 
    PROJECT_ROOT / "llm_jury" / "etl" / "coding_benchmarks_client.py"
)
coding_benchmarks_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coding_benchmarks_client)

CodingBenchmarksClient = coding_benchmarks_client.CodingBenchmarksClient
CodingProblem = coding_benchmarks_client.CodingProblem
format_humaneval_prompt = coding_benchmarks_client.format_humaneval_prompt
format_mbpp_prompt = coding_benchmarks_client.format_mbpp_prompt
extract_code_from_response = coding_benchmarks_client.extract_code_from_response
calculate_pass_at_k = coding_benchmarks_client.calculate_pass_at_k

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output files
HUMANEVAL_SCORES_FILE = DATA_PATH / "humaneval_scores.json"
MBPP_SCORES_FILE = DATA_PATH / "mbpp_scores.json"
CODING_SCORES_FILE = DATA_PATH / "coding_benchmark_scores.json"

# Reasoning model patterns (require different API parameters)
REASONING_PATTERNS = ['/o1', '/o3', '/o4-']

# Rate limiting
REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60 / REQUESTS_PER_MINUTE


class OpenRouterClient:
    """OpenRouter API client with rate limiting."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
                    cls._instance._last_request_time = 0
        return cls._instance
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client
    
    def _is_reasoning_model(self, model_id: str) -> bool:
        """Check if model is a reasoning model (o1, o3, etc.)."""
        model_lower = model_id.lower()
        return any(pattern in model_lower for pattern in REASONING_PATTERNS)
    
    def _rate_limit(self):
        """Apply rate limiting."""
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < REQUEST_DELAY:
                time.sleep(REQUEST_DELAY - elapsed)
            self._last_request_time = time.time()
    
    def generate_completion(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0
    ) -> Optional[str]:
        """Generate a code completion from the model.
        
        Args:
            model_id: OpenRouter model ID
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
        
        Returns:
            Model response text or None on error
        """
        self._rate_limit()
        
        try:
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert Python programmer. Write clean, correct Python code. Return only the code, no explanations."
                },
                {"role": "user", "content": prompt}
            ]
            
            # Reasoning models use different parameters
            if self._is_reasoning_model(model_id):
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling {model_id}: {e}")
            return None


# Global client
_openrouter = OpenRouterClient()


def get_models_to_evaluate(
    cache_path: Path,
    model_filter: Optional[str] = None
) -> List[Dict]:
    """Get list of models to evaluate from cache.
    
    Args:
        cache_path: Path to models_cache.json
        model_filter: Optional comma-separated list of model names/slugs to include
    
    Returns:
        List of model dicts with openrouter_id
    """
    with open(cache_path) as f:
        cache = json.load(f)
    
    models = cache.get("models", cache)
    
    # Filter to models with OpenRouter IDs
    models = [m for m in models if m.get("openrouter_id")]
    
    # Apply name filter if provided
    if model_filter:
        filter_names = [n.strip().lower() for n in model_filter.split(",")]
        models = [
            m for m in models
            if any(
                fn in m.get("name", "").lower() or
                fn in m.get("slug", "").lower() or
                fn in m.get("openrouter_id", "").lower()
                for fn in filter_names
            )
        ]
    
    # Sort by price (cheapest first for efficiency)
    models.sort(key=lambda m: m.get("price_1m_blended", float('inf')))
    
    logger.info(f"Found {len(models)} models to evaluate")
    return models


def evaluate_humaneval(
    model_id: str,
    client: CodingBenchmarksClient,
    num_samples: int = 1,
    max_problems: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model on HumanEval.
    
    Args:
        model_id: OpenRouter model ID
        client: CodingBenchmarksClient instance
        num_samples: Number of samples per problem (for pass@k)
        max_problems: Limit number of problems (for testing)
    
    Returns:
        Dict with pass@1 score and detailed results
    """
    problems = client.load_humaneval()
    
    if max_problems:
        problems = dict(list(problems.items())[:max_problems])
    
    logger.info(f"Evaluating {model_id} on {len(problems)} HumanEval problems")
    
    results = []
    passed = 0
    
    for i, (task_id, problem) in enumerate(problems.items()):
        # Generate completion
        prompt = format_humaneval_prompt(problem)
        
        for sample_idx in range(num_samples):
            response = _openrouter.generate_completion(model_id, prompt)
            
            if response is None:
                results.append({
                    "task_id": task_id,
                    "passed": False,
                    "result": "API error",
                    "completion": None
                })
                continue
            
            # Extract code from response
            completion = extract_code_from_response(response, problem)
            
            # Check correctness
            result = client.check_correctness(problem, completion)
            results.append(result)
            
            if result["passed"]:
                passed += 1
        
        if (i + 1) % 10 == 0:
            current_rate = passed / ((i + 1) * num_samples) * 100
            logger.info(f"  Progress: {i + 1}/{len(problems)}, pass@1: {current_rate:.1f}%")
    
    # Calculate final score
    pass_at_1 = calculate_pass_at_k(results, k=1)
    
    return {
        "model_id": model_id,
        "benchmark": "humaneval",
        "pass_at_1": round(pass_at_1, 2),
        "num_problems": len(problems),
        "num_samples": num_samples,
        "total_passed": passed,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def evaluate_mbpp(
    model_id: str,
    client: CodingBenchmarksClient,
    num_samples: int = 1,
    max_problems: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model on MBPP.
    
    Args:
        model_id: OpenRouter model ID
        client: CodingBenchmarksClient instance
        num_samples: Number of samples per problem (for pass@k)
        max_problems: Limit number of problems (for testing)
    
    Returns:
        Dict with pass@1 score and detailed results
    """
    problems = client.load_mbpp()
    
    if max_problems:
        problems = dict(list(problems.items())[:max_problems])
    
    logger.info(f"Evaluating {model_id} on {len(problems)} MBPP problems")
    
    results = []
    passed = 0
    
    for i, (task_id, problem) in enumerate(problems.items()):
        # Generate completion
        prompt = format_mbpp_prompt(problem)
        
        for sample_idx in range(num_samples):
            response = _openrouter.generate_completion(model_id, prompt)
            
            if response is None:
                results.append({
                    "task_id": task_id,
                    "passed": False,
                    "result": "API error",
                    "completion": None
                })
                continue
            
            # Extract code from response
            completion = extract_code_from_response(response, problem)
            
            # Check correctness
            result = client.check_correctness(problem, completion)
            results.append(result)
            
            if result["passed"]:
                passed += 1
        
        if (i + 1) % 20 == 0:
            current_rate = passed / ((i + 1) * num_samples) * 100
            logger.info(f"  Progress: {i + 1}/{len(problems)}, pass@1: {current_rate:.1f}%")
    
    # Calculate final score
    pass_at_1 = calculate_pass_at_k(results, k=1)
    
    return {
        "model_id": model_id,
        "benchmark": "mbpp",
        "pass_at_1": round(pass_at_1, 2),
        "num_problems": len(problems),
        "num_samples": num_samples,
        "total_passed": passed,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def load_existing_scores(scores_file: Path) -> Dict[str, float]:
    """Load existing scores from file."""
    if scores_file.exists():
        with open(scores_file) as f:
            return json.load(f)
    return {}


def save_scores(scores: Dict[str, float], scores_file: Path):
    """Save scores to file."""
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "w") as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Saved scores to {scores_file}")


def update_models_cache(
    cache_path: Path,
    humaneval_scores: Dict[str, float],
    mbpp_scores: Dict[str, float]
):
    """Update models_cache.json with new scores."""
    with open(cache_path) as f:
        cache = json.load(f)
    
    models = cache.get("models", cache)
    updated = 0
    
    for model in models:
        openrouter_id = model.get("openrouter_id")
        if not openrouter_id:
            continue
        
        if openrouter_id in humaneval_scores:
            model["humaneval_score"] = humaneval_scores[openrouter_id]
            model["humaneval_source"] = "calculated"
            updated += 1
        
        if openrouter_id in mbpp_scores:
            model["mbpp_score"] = mbpp_scores[openrouter_id]
            model["mbpp_source"] = "calculated"
            updated += 1
    
    # Save updated cache
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    
    logger.info(f"Updated {updated} model scores in {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HumanEval and MBPP coding benchmarks"
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DATA_PATH / "models_cache.json",
        help="Path to models_cache.json"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models in cache"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names/slugs to evaluate"
    )
    parser.add_argument(
        "--humaneval",
        action="store_true",
        help="Run HumanEval benchmark"
    )
    parser.add_argument(
        "--mbpp",
        action="store_true",
        help="Run MBPP benchmark"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        help="Limit number of problems (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show models without evaluating"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip models with existing scores"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if scores exist"
    )
    
    args = parser.parse_args()
    
    # Default to both benchmarks if neither specified
    if not args.humaneval and not args.mbpp:
        args.humaneval = True
        args.mbpp = True
    
    # Require --all or --models
    if not args.all and not args.models:
        parser.print_help()
        print("\nError: Specify --all or --models")
        sys.exit(1)
    
    # Load models
    models = get_models_to_evaluate(
        args.cache_file,
        model_filter=args.models
    )
    
    if not models:
        logger.error("No models found to evaluate")
        sys.exit(1)
    
    # Load existing scores
    humaneval_scores = load_existing_scores(HUMANEVAL_SCORES_FILE)
    mbpp_scores = load_existing_scores(MBPP_SCORES_FILE)
    
    # Filter out already evaluated models if skip-existing
    if args.skip_existing and not args.force:
        if args.humaneval:
            models_to_eval_he = [
                m for m in models 
                if m["openrouter_id"] not in humaneval_scores
            ]
            logger.info(f"HumanEval: {len(models) - len(models_to_eval_he)} already evaluated, {len(models_to_eval_he)} remaining")
        else:
            models_to_eval_he = []
        
        if args.mbpp:
            models_to_eval_mbpp = [
                m for m in models 
                if m["openrouter_id"] not in mbpp_scores
            ]
            logger.info(f"MBPP: {len(models) - len(models_to_eval_mbpp)} already evaluated, {len(models_to_eval_mbpp)} remaining")
        else:
            models_to_eval_mbpp = []
    else:
        models_to_eval_he = models if args.humaneval else []
        models_to_eval_mbpp = models if args.mbpp else []
    
    # Dry run
    if args.dry_run:
        print("\n=== Models to evaluate ===")
        if args.humaneval:
            print(f"\nHumanEval ({len(models_to_eval_he)} models):")
            for m in models_to_eval_he[:20]:
                print(f"  - {m['name']} ({m['openrouter_id']})")
            if len(models_to_eval_he) > 20:
                print(f"  ... and {len(models_to_eval_he) - 20} more")
        
        if args.mbpp:
            print(f"\nMBPP ({len(models_to_eval_mbpp)} models):")
            for m in models_to_eval_mbpp[:20]:
                print(f"  - {m['name']} ({m['openrouter_id']})")
            if len(models_to_eval_mbpp) > 20:
                print(f"  ... and {len(models_to_eval_mbpp) - 20} more")
        
        return
    
    # Initialize benchmark client
    client = CodingBenchmarksClient(timeout=5.0)
    
    # Run evaluations
    if args.humaneval and models_to_eval_he:
        logger.info(f"\n=== Running HumanEval on {len(models_to_eval_he)} models ===")
        
        for model in models_to_eval_he:
            model_id = model["openrouter_id"]
            logger.info(f"\nEvaluating: {model['name']} ({model_id})")
            
            try:
                result = evaluate_humaneval(
                    model_id,
                    client,
                    max_problems=args.max_problems
                )
                
                humaneval_scores[model_id] = result["pass_at_1"]
                logger.info(f"✓ {model['name']}: HumanEval pass@1 = {result['pass_at_1']}%")
                
                # Save after each model
                save_scores(humaneval_scores, HUMANEVAL_SCORES_FILE)
                
            except Exception as e:
                logger.error(f"✗ Failed to evaluate {model['name']}: {e}")
    
    if args.mbpp and models_to_eval_mbpp:
        logger.info(f"\n=== Running MBPP on {len(models_to_eval_mbpp)} models ===")
        
        for model in models_to_eval_mbpp:
            model_id = model["openrouter_id"]
            logger.info(f"\nEvaluating: {model['name']} ({model_id})")
            
            try:
                result = evaluate_mbpp(
                    model_id,
                    client,
                    max_problems=args.max_problems
                )
                
                mbpp_scores[model_id] = result["pass_at_1"]
                logger.info(f"✓ {model['name']}: MBPP pass@1 = {result['pass_at_1']}%")
                
                # Save after each model
                save_scores(mbpp_scores, MBPP_SCORES_FILE)
                
            except Exception as e:
                logger.error(f"✗ Failed to evaluate {model['name']}: {e}")
    
    # Update models cache with new scores
    update_models_cache(args.cache_file, humaneval_scores, mbpp_scores)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    if args.humaneval:
        print(f"HumanEval: {len(humaneval_scores)} models scored")
        if humaneval_scores:
            top_he = sorted(humaneval_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 HumanEval:")
            for model_id, score in top_he:
                print(f"  {model_id}: {score}%")
    
    if args.mbpp:
        print(f"\nMBPP: {len(mbpp_scores)} models scored")
        if mbpp_scores:
            top_mbpp = sorted(mbpp_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 MBPP:")
            for model_id, score in top_mbpp:
                print(f"  {model_id}: {score}%")


if __name__ == "__main__":
    main()
