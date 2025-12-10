#!/usr/bin/env python3
"""
Unified MixEval Evaluation Script

Evaluates models on MixEval and MixEval-Hard benchmarks via OpenRouter.
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # research/kdd -> research -> project root
DATA_PATH = PROJECT_ROOT / "data"
MIXEVAL_PATH = PROJECT_ROOT / "external" / "MixEval"

sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Reasoning model patterns (require max_completion_tokens instead of max_tokens)
REASONING_PATTERNS = ['/o1', '/o3', '/o4-']


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str
    slug: str
    openrouter_id: str
    hallucination_rate: float
    
    @property
    def score_key(self) -> str:
        """Get the key used for storing scores."""
        return self.openrouter_id


def is_reasoning_model(model_id: str) -> bool:
    """Check if a model ID indicates a reasoning model."""
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in REASONING_PATTERNS)


class OpenRouterClient:
    """OpenRouter API client with caching."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._client = None
        return cls._instance
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv('OPENROUTER_API_KEY'),
                base_url="https://openrouter.ai/api/v1"
            )
        return self._client
    
    def call(self, model_id: str, prompt: str) -> Optional[str]:
        """Call a model via OpenRouter."""
        try:
            if is_reasoning_model(model_id):
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024,
                )
            else:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling {model_id}: {e}")
            return None


# Global client
_client = OpenRouterClient()


class DataManager:
    """Manages loading and caching of data files."""
    
    def __init__(self):
        self._models_cache: Optional[List[Dict]] = None
        self._scores_cache: Dict[str, Dict[str, float]] = {}
        self._questions_cache: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
    
    def get_models(self) -> List[Dict]:
        """Load and cache models from models_cache.json."""
        with self._lock:
            if self._models_cache is None:
                cache_path = DATA_PATH / "models_cache.json"
                with open(cache_path) as f:
                    data = json.load(f)
                self._models_cache = data.get("models", data)
            return self._models_cache
    
    def get_scores(self, benchmark: str) -> Dict[str, float]:
        """Load and cache scores for a benchmark."""
        with self._lock:
            if benchmark not in self._scores_cache:
                scores_file = "mixeval_scores.json" if benchmark == "mixeval" else "mixeval_hard_scores.json"
                scores_path = DATA_PATH / scores_file
                if scores_path.exists():
                    with open(scores_path) as f:
                        self._scores_cache[benchmark] = json.load(f)
                else:
                    self._scores_cache[benchmark] = {}
            return self._scores_cache[benchmark]
    
    def save_scores(self, scores: Dict[str, float], benchmark: str):
        """Save scores and update cache."""
        with self._lock:
            existing = self._scores_cache.get(benchmark, {}).copy()
            existing.update(scores)
            
            scores_file = "mixeval_scores.json" if benchmark == "mixeval" else "mixeval_hard_scores.json"
            output_path = DATA_PATH / scores_file
            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2)
            
            self._scores_cache[benchmark] = existing
            print(f"  Saved {len(scores)} scores to {scores_file}")
    
    def get_questions(self, benchmark: str, max_samples: int = 100) -> List[Dict]:
        """Load and cache questions for a benchmark."""
        cache_key = f"{benchmark}:{max_samples}"
        
        with self._lock:
            if cache_key not in self._questions_cache:
                self._questions_cache[cache_key] = self._load_questions(benchmark, max_samples)
            return self._questions_cache[cache_key]
    
    def _load_questions(self, benchmark: str, max_samples: int) -> List[Dict]:
        """Load questions from disk."""
        bench_folder = "mixeval" if benchmark == "mixeval" else "mixeval-hard"
        base_path = MIXEVAL_PATH / "mix_eval" / "data" / "mixeval-2024-08-11" / bench_folder
        
        questions = []
        
        # Load both question types
        for qtype, filename in [("free-form", "free-form.json"), ("multiple-choice", "multi-choice.json")]:
            file_path = base_path / filename
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                items = [data[str(i)] for i in range(len(data)) if str(i) in data] if isinstance(data, dict) else data
                for q in items:
                    q["type"] = qtype
                questions.extend(items)
        
        # Sample if needed
        if len(questions) > max_samples:
            random.seed(42)
            questions = random.sample(questions, max_samples)
        
        print(f"  Loaded {len(questions)} {benchmark} questions")
        return questions


# Global data manager
_data = DataManager()


def load_qualified_models(benchmarks: List[str]) -> List[ModelConfig]:
    """Load models that need evaluation for the specified benchmarks."""
    models_data = _data.get_models()
    
    # Load existing scores
    existing_scores = {bench: _data.get_scores(bench) for bench in benchmarks}
    
    qualified = []
    seen_ids = set()
    skipped_have_scores = 0
    skipped_no_api = 0
    
    for m in models_data:
        # Check hallucination rate
        halluc = m.get('hallucination_rate')
        if not halluc or float(halluc) <= 0:
            continue
        
        # Check math_500 benchmark
        math500 = m.get('math_500')
        if not math500 or float(math500) <= 0:
            continue
        
        openrouter_id = m.get('openrouter_id', '')
        if not openrouter_id:
            skipped_no_api += 1
            continue
        
        # Check if model needs any of the requested benchmarks
        needs_benchmark = False
        for bench in benchmarks:
            if openrouter_id not in existing_scores.get(bench, {}):
                needs_benchmark = True
                break
        
        if not needs_benchmark:
            skipped_have_scores += 1
            continue
        
        # Dedupe by OpenRouter ID
        if openrouter_id in seen_ids:
            continue
        seen_ids.add(openrouter_id)
        
        qualified.append(ModelConfig(
            name=m.get('name', ''),
            slug=m.get('slug', ''),
            openrouter_id=openrouter_id,
            hallucination_rate=float(halluc),
        ))
    
    if skipped_have_scores > 0:
        print(f"  (Skipped {skipped_have_scores} models that already have scores)")
    if skipped_no_api > 0:
        print(f"  (Skipped {skipped_no_api} models with no OpenRouter ID)")
    
    return qualified


def format_prompt(question: Dict) -> str:
    """Format a question as a prompt."""
    if question["type"] == "multiple-choice":
        prompt = question.get("prompt", question.get("question", ""))
        options = question.get("options", [])
        if options:
            prompt += "\n\nOptions:\n"
            for i, opt in enumerate(options):
                prompt += f"{chr(65+i)}. {opt}\n"
            prompt += "\nAnswer with just the letter (A, B, C, or D)."
        return prompt
    return question.get("prompt", question.get("question", ""))


def score_response(response: str, expected: Any, qtype: str) -> bool:
    """Score a single response against expected answer."""
    if not response:
        return False
    
    response_lower = response.strip().lower()
    
    if qtype == "multiple-choice":
        return isinstance(expected, str) and expected.upper() in response.upper()
    else:
        if isinstance(expected, list):
            return any(str(e).lower() in response_lower for e in expected)
        elif isinstance(expected, str):
            return expected.lower() in response_lower
        return False


def run_evaluation(model: ModelConfig, questions: List[Dict], benchmark: str, dry_run: bool = False) -> Optional[Dict]:
    """Run evaluation for a single model on a benchmark."""
    
    if dry_run:
        print(f"  [DRY RUN] Would evaluate {model.name} via {model.openrouter_id}")
        return {"score": 0, "dry_run": True}
    
    total = len(questions)
    correct = 0
    errors = 0
    
    for i, q in enumerate(questions):
        # Progress bar
        pct = (i + 1) / total * 100
        filled = int(30 * (i + 1) / total)
        bar = "█" * filled + "░" * (30 - filled)
        print(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{total}) ", end="", flush=True)
        
        try:
            prompt = format_prompt(q)
            answer = _client.call(model.openrouter_id, prompt)
            expected = q.get("answer", q.get("target", ""))
            
            if answer:
                if score_response(answer, expected, q["type"]):
                    correct += 1
            else:
                errors += 1
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            errors += 1
            logger.error(f"Error: {e}")
    
    print()  # New line after progress bar
    
    total_answered = total - errors
    score = (correct / total_answered * 100) if total_answered > 0 else 0
    
    print(f"  ✅ Score: {score:.1f}% ({correct}/{total_answered})")
    if errors:
        print(f"  ❌ Errors: {errors}")
    
    return {"score": score, "correct": correct, "total": total_answered, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Run MixEval evaluations via OpenRouter")
    parser.add_argument("--all", action="store_true", help="Run on all qualified models")
    parser.add_argument("--models", nargs="+", help="Specific model slugs to evaluate")
    parser.add_argument("--benchmark", default="both", choices=["mixeval", "mixeval-hard", "both"],
                        help="Which benchmark(s) to run")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models")
    parser.add_argument("--samples", type=int, default=100, help="Questions per model (default: 100)")
    parser.add_argument("--threads", type=int, default=3, help="Parallel threads (default: 3)")
    
    args = parser.parse_args()
    
    # Determine benchmarks to run
    benchmarks = ["mixeval", "mixeval-hard"] if args.benchmark == "both" else [args.benchmark]
    
    print("=" * 70)
    print("MIXEVAL EVALUATION VIA OPENROUTER")
    print("=" * 70)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Samples per benchmark: {args.samples}")
    print(f"Threads: {args.threads}")
    print()
    
    # Load qualified models
    all_models = load_qualified_models(benchmarks)
    print(f"\nQualified models: {len(all_models)}")
    
    # Filter to requested models
    if args.models:
        models = [m for m in all_models if m.slug in args.models]
    elif args.all:
        models = all_models
    else:
        print("\nSpecify --all or --models <slugs>")
        print("\nSample models:")
        for m in all_models[:10]:
            print(f"  {m.slug} -> {m.openrouter_id}")
        if len(all_models) > 10:
            print(f"  ... and {len(all_models) - 10} more")
        return
    
    if args.max_models:
        models = models[:args.max_models]
    
    print(f"\nModels to evaluate: {len(models)}")
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE]")
        for m in models:
            print(f"  {m.name} -> {m.openrouter_id}")
        return
    
    # Run evaluations for each benchmark
    for benchmark in benchmarks:
        print("\n" + "=" * 70)
        print(f"RUNNING {benchmark.upper()}")
        print("=" * 70)
        
        questions = _data.get_questions(benchmark, args.samples)
        
        results = {}
        results_lock = threading.Lock()
        
        def evaluate_model(model: ModelConfig):
            return model, run_evaluation(model, questions, benchmark)
        
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {executor.submit(evaluate_model, m): m for m in models}
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    m, result = future.result()
                    if result and result.get("score") is not None:
                        with results_lock:
                            results[m.score_key] = result["score"]
                except Exception as e:
                    logger.error(f"Failed {model.name}: {e}")
        
        if results:
            _data.save_scores(results, benchmark)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
