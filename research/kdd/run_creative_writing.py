#!/usr/bin/env python3
"""
Creative Writing Benchmark Evaluation Script

Evaluates models on the EQ-Bench Creative Writing Benchmark v3.
This benchmark measures creative writing quality using:
1. Rubric scoring (0-100 scale)
2. Elo ratings from pairwise comparisons

Benchmark: https://github.com/EQ-bench/creative-writing-bench
Metric: Creative Elo Score (normalized to leaderboard)

This script wraps the creative-writing-bench evaluation system and integrates
it with the LLM Jury model cache and scoring system.
"""

import os
import sys
import json
import argparse
import time
import subprocess
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
CREATIVE_BENCH_PATH = PROJECT_ROOT / "creative-writing-bench"
CREATIVE_RUNS_FILE = CREATIVE_BENCH_PATH / "creative_bench_runs.json"
ELO_RESULTS_FILE = CREATIVE_BENCH_PATH / "elo_results.json"

sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / ".env")

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


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


class DataManager:
    """Manages loading and caching of data files."""
    
    def __init__(self):
        self._models_cache: Optional[List[Dict]] = None
        self._scores_cache: Dict[str, float] = {}
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
    
    def get_scores(self) -> Dict[str, float]:
        """Load and cache Creative Writing scores."""
        with self._lock:
            if not self._scores_cache:
                scores_file = DATA_PATH / "creative_writing_scores.json"
                if scores_file.exists():
                    with open(scores_file) as f:
                        self._scores_cache = json.load(f)
                else:
                    self._scores_cache = {}
            return self._scores_cache
    
    def save_scores(self, scores: Dict[str, float]):
        """Save scores and update cache."""
        with self._lock:
            existing = self._scores_cache.copy()
            existing.update(scores)
            
            output_path = DATA_PATH / "creative_writing_scores.json"
            with open(output_path, "w") as f:
                json.dump(existing, f, indent=2)
            
            self._scores_cache = existing
            logger.info(f"Saved {len(scores)} scores to creative_writing_scores.json")


# Global data manager
_data = DataManager()


def load_qualified_models() -> List[ModelConfig]:
    """Load models that need evaluation."""
    models_data = _data.get_models()
    benchmarks_required = ['intelligence_index', 'coding_index', 'math_index']
    
    # Load existing scores
    existing_scores = _data.get_scores()
    
    qualified = []
    seen_ids = set()
    skipped_have_scores = 0
    skipped_no_api = 0
    
    for m in models_data:
        # Check hallucination rate
        halluc = m.get('hallucination_rate')
        if not halluc or float(halluc) <= 0:
            continue
        
        # Check required benchmarks
        if not all(m.get(b) and float(m.get(b, 0)) > 0 for b in benchmarks_required):
            continue
        
        openrouter_id = m.get('openrouter_id', '')
        if not openrouter_id:
            skipped_no_api += 1
            continue
        
        # Check if model already has a score
        if openrouter_id in existing_scores:
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
        logger.info(f"Skipped {skipped_have_scores} models that already have scores")
    if skipped_no_api > 0:
        logger.info(f"Skipped {skipped_no_api} models with no OpenRouter ID")
    
    return qualified


def setup_environment():
    """Setup environment variables for creative writing benchmark."""
    # Use OpenRouter for both test and judge models
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    # Set environment variables for the creative writing benchmark
    os.environ['TEST_API_KEY'] = openrouter_key
    os.environ['JUDGE_API_KEY'] = openrouter_key
    os.environ['TEST_API_URL'] = 'https://openrouter.ai/api/v1/chat/completions'
    os.environ['JUDGE_API_URL'] = 'https://openrouter.ai/api/v1/chat/completions'
    os.environ['REQUEST_TIMEOUT'] = '300'
    os.environ['MAX_RETRIES'] = '5'
    os.environ['RETRY_DELAY'] = '5'


def extract_elo_score(run_key: str) -> Optional[float]:
    """Extract the normalized Elo score from results."""
    try:
        # Load runs file
        if not CREATIVE_RUNS_FILE.exists():
            logger.error(f"Runs file not found: {CREATIVE_RUNS_FILE}")
            return None
        
        with open(CREATIVE_RUNS_FILE) as f:
            runs = json.load(f)
        
        run_data = runs.get(run_key)
        if not run_data:
            logger.error(f"Run key not found: {run_key}")
            return None
        
        # Extract normalized Elo score
        results = run_data.get("results", {}).get("benchmark_results", {})
        elo_norm = results.get("elo_normalized")
        
        if elo_norm is not None:
            return float(elo_norm)
        
        logger.warning(f"No normalized Elo score found for {run_key}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting Elo score: {e}")
        return None


def run_evaluation(
    model: ModelConfig,
    judge_model: str = "anthropic/claude-sonnet-4",
    iterations: int = 3,
    threads: int = 100,
    dry_run: bool = False
) -> Optional[Dict]:
    """Run Creative Writing evaluation for a single model."""
    
    if dry_run:
        print(f"  [DRY RUN] Would evaluate {model.name} via {model.openrouter_id}")
        return {"elo_normalized": 0, "dry_run": True}
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model.name}")
    print(f"Model ID: {model.openrouter_id}")
    print(f"Judge: {judge_model}")
    print(f"Iterations: {iterations}")
    print(f"{'='*70}")
    
    # Generate unique run ID
    run_id = f"{model.slug}_{int(time.time())}"
    
    try:
        # Build command
        cmd = [
            "python3",
            str(CREATIVE_BENCH_PATH / "creative_writing_bench.py"),
            "--test-model", model.openrouter_id,
            "--judge-model", judge_model,
            "--runs-file", str(CREATIVE_RUNS_FILE),
            "--creative-prompts-file", str(CREATIVE_BENCH_PATH / "data" / "creative_writing_prompts_v3.json"),
            "--run-id", run_id,
            "--threads", str(threads),
            "--verbosity", "INFO",
            "--iterations", str(iterations),
        ]
        
        print(f"\n▶️  Running Creative Writing Benchmark...")
        print(f"   This will take ~15-30 minutes per model")
        print(f"   Command: {' '.join(cmd)}")
        
        # Run the benchmark
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(CREATIVE_BENCH_PATH),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"Benchmark failed for {model.name}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None
        
        print(f"\n✅ Benchmark completed in {duration/60:.1f} minutes")
        
        # Extract the run key from output or construct it
        # The benchmark uses format: {run_id}_{test_model_sanitized}
        run_key = None
        for line in result.stdout.split('\n'):
            if "Run Key:" in line:
                run_key = line.split("Run Key:")[-1].strip()
                break
        
        if not run_key:
            # Fallback: construct run key
            model_sanitized = model.openrouter_id.replace('/', '__')
            run_key = f"{run_id}_{model_sanitized}"
        
        # Extract Elo score
        elo_score = extract_elo_score(run_key)
        
        if elo_score is None:
            logger.error(f"Failed to extract Elo score for {model.name}")
            return None
        
        print(f"\n🎨 Creative Writing Elo Score: {elo_score:.1f}")
        
        return {
            "elo_normalized": elo_score,
            "run_key": run_key,
            "duration_minutes": duration / 60
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timeout for {model.name} after 2 hours")
        return None
    except Exception as e:
        logger.error(f"Error evaluating {model.name}: {e}")
        return None


def unzip_canonical_results():
    """Unzip canonical benchmark results if not already done."""
    try:
        # Check if already unzipped
        if CREATIVE_RUNS_FILE.exists():
            logger.info("✓ Benchmark runs file exists")
            return True
        
        # Unzip creative_bench_runs.zip
        runs_zip = CREATIVE_BENCH_PATH / "creative_bench_runs.zip"
        if runs_zip.exists():
            print(f"📦 Unzipping canonical benchmark runs...")
            subprocess.run(
                ["unzip", "-o", str(runs_zip)],
                cwd=str(CREATIVE_BENCH_PATH),
                check=True,
                capture_output=True
            )
            print("✓ Unzipped creative_bench_runs.zip")
        
        # Unzip elo_results.zip
        elo_zip = CREATIVE_BENCH_PATH / "elo_results.zip"
        if elo_zip.exists():
            print(f"📦 Unzipping canonical Elo results...")
            subprocess.run(
                ["unzip", "-o", str(elo_zip)],
                cwd=str(CREATIVE_BENCH_PATH),
                check=True,
                capture_output=True
            )
            print("✓ Unzipped elo_results.zip")
        
        return True
        
    except Exception as e:
        logger.error(f"Error unzipping canonical results: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Creative Writing evaluation via OpenRouter")
    parser.add_argument("--all", action="store_true", help="Run on all qualified models")
    parser.add_argument("--models", nargs="+", help="Specific model slugs to evaluate")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4",
                        help="Judge model to use (default: anthropic/claude-sonnet-4)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be evaluated")
    parser.add_argument("--max-models", type=int, default=None, help="Limit number of models")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Iterations per prompt (default: 3)")
    parser.add_argument("--threads", type=int, default=100,
                        help="Parallel threads for benchmark (default: 100)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CREATIVE WRITING EVALUATION VIA OPENROUTER")
    print("=" * 70)
    print(f"Judge Model: {args.judge_model}")
    print(f"Iterations: {args.iterations}")
    print(f"Threads: {args.threads}")
    print()
    
    # Setup environment
    try:
        setup_environment()
        print("✓ Environment configured")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Unzip canonical results if needed
    print("\nSetting up benchmark...")
    if not unzip_canonical_results():
        print("⚠️  Warning: Could not unzip canonical results")
        print("   Scores may not be comparable to leaderboard")
    
    # Load qualified models
    all_models = load_qualified_models()
    print(f"\n✓ Qualified models: {len(all_models)}")
    
    # Filter to requested models
    if args.models:
        models = [m for m in all_models if m.slug in args.models]
        print(f"  Filtered to {len(models)} requested models")
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
    
    print(f"\n✓ Models to evaluate: {len(models)}")
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN MODE]")
        for m in models:
            print(f"  {m.name} -> {m.openrouter_id}")
        return
    
    # Warning about time and cost
    est_time = len(models) * 20  # ~20 minutes per model
    print(f"\n⏱️  Estimated time: {est_time} minutes ({est_time/60:.1f} hours)")
    print(f"💰 Estimated cost: ${len(models) * 10:.2f} (approx. $10 per model)")
    print("\n⚠️  This is a comprehensive evaluation that will take significant time.")
    print("   Each model generates 96 creative writing pieces (32 prompts x 3 iterations)")
    print("   and undergoes pairwise comparisons with judge model.")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Run evaluations
    results = {}
    results_lock = threading.Lock()
    
    if args.sequential:
        # Sequential evaluation
        for model in models:
            result = run_evaluation(
                model,
                judge_model=args.judge_model,
                iterations=args.iterations,
                threads=args.threads
            )
            
            if result and result.get("elo_normalized") is not None:
                with results_lock:
                    results[model.score_key] = result["elo_normalized"]
    else:
        # Parallel evaluation (not recommended due to cost, but supported)
        print("\n⚠️  Note: Parallel evaluation not recommended for this benchmark")
        print("   Running sequentially would be more cost-effective.")
        
        def evaluate_wrapper(model: ModelConfig):
            return model, run_evaluation(
                model,
                judge_model=args.judge_model,
                iterations=args.iterations,
                threads=args.threads
            )
        
        with ThreadPoolExecutor(max_workers=1) as executor:  # Force sequential
            futures = {executor.submit(evaluate_wrapper, m): m for m in models}
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    m, result = future.result()
                    if result and result.get("elo_normalized") is not None:
                        with results_lock:
                            results[m.score_key] = result["elo_normalized"]
                except Exception as e:
                    logger.error(f"Failed {model.name}: {e}")
    
    # Save results
    if results:
        _data.save_scores(results)
        
        # Print summary
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for model_id, score in sorted_results:
            print(f"  {model_id:<50} {score:>6.1f}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Scores saved to: {DATA_PATH / 'creative_writing_scores.json'}")
    print(f"📁 Full benchmark data: {CREATIVE_RUNS_FILE}")
    print(f"🎯 Elo analysis: {ELO_RESULTS_FILE}")


if __name__ == "__main__":
    main()

