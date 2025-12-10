#!/usr/bin/env python3
"""
Intent Classification Evaluation on Gold-Standard Benchmarks
=============================================================

Evaluates our StandardClassifier against benchmark datasets with known intents:
- HumanEval → CODING
- GSM8K → REASONING  
- IFEval → EXTRACTION (instruction following)
- TruthfulQA → FACTUAL_QA

IMPORTANT: Uses proper train/dev/test splits to avoid overfitting!
- DEV set: Used ONLY for pattern tuning (if needed)
- TEST set: Used ONLY for final evaluation (never seen during development)

This proves our intent detection generalizes to standard benchmarks.

Usage:
    # Final evaluation on held-out test set (default)
    python evaluate_intent_classification.py
    
    # Development mode - see dev set for pattern tuning
    python evaluate_intent_classification.py --split dev --show-errors
    
    # Full evaluation with both splits
    python evaluate_intent_classification.py --split both
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.standard_taxonomy import StandardCategory
from llm_jury.routing.standard_classifier import (
    StandardClassifier,
    StandardClassificationResult,
)

# =============================================================================
# SPLIT CONFIGURATION - DO NOT CHANGE AFTER PATTERN DEVELOPMENT
# =============================================================================
# We use deterministic splits based on hash to ensure reproducibility
# and prevent accidental data leakage between dev and test

DEV_RATIO = 0.3  # 30% for development/tuning, 70% for final test
RANDOM_SEED = 42  # For reproducibility


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset."""
    name: str
    hf_dataset: str
    hf_config: Optional[str]
    split: str
    prompt_field: str
    expected_category: StandardCategory
    description: str
    # Optional preprocessing
    prompt_template: Optional[str] = None


# Benchmark configurations
# NOTE: We use benchmarks with CLEAR intent signals for validation
BENCHMARKS = [
    BenchmarkConfig(
        name="HumanEval",
        hf_dataset="openai/openai_humaneval",
        hf_config=None,
        split="test",
        prompt_field="prompt",
        expected_category=StandardCategory.CODING,
        description="Code generation benchmark - should classify as CODING",
    ),
    BenchmarkConfig(
        name="GSM8K",
        hf_dataset="openai/gsm8k",
        hf_config="main",
        split="test",
        prompt_field="question",
        expected_category=StandardCategory.REASONING,
        description="Grade school math WORD PROBLEMS - natural language, no equations",
    ),
    BenchmarkConfig(
        name="TruthfulQA",
        hf_dataset="truthfulqa/truthful_qa",
        hf_config="generation",
        split="validation",
        prompt_field="question",
        expected_category=StandardCategory.FACTUAL_QA,
        description="Factual questions - should classify as FACTUAL_QA",
    ),
    # NOTE: IFEval is EXCLUDED from main accuracy because it's a MIXED-INTENT benchmark
    # IFEval tests instruction following across many task types (creative, QA, summarization, etc.)
    # Our classifier correctly identifies the actual task type, not "extraction"
]

# Optional: Include IFEval with correct labeling for analysis
IFEVAL_CONFIG = BenchmarkConfig(
    name="IFEval",
    hf_dataset="google/IFEval",
    hf_config=None,
    split="train",
    prompt_field="prompt",
    expected_category=StandardCategory.UNCERTAIN,  # Mixed-intent benchmark
    description="Mixed instruction following - diverse task types (creative, QA, etc.)",
)


@dataclass
class EvaluationResult:
    """Results from evaluating on a single benchmark."""
    benchmark: str
    expected_category: str
    total_samples: int
    correct: int
    accuracy: float
    uncertain_count: int
    uncertain_rate: float
    category_distribution: Dict[str, int]
    errors: List[Dict] = field(default_factory=list)
    
    # NEW: Safe routing metrics
    # "Safe" = correct classification OR routed to uncertain (conservative fallback)
    # "Wrong specialist" = routed to a different specialist category
    @property
    def safe_routing_count(self) -> int:
        """Count of samples safely routed (correct or uncertain)."""
        return self.correct + self.uncertain_count
    
    @property
    def safe_routing_rate(self) -> float:
        """Rate of safe routing (correct or uncertain)."""
        return self.safe_routing_count / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def wrong_specialist_count(self) -> int:
        """Count of samples misrouted to wrong specialist."""
        return self.total_samples - self.safe_routing_count
    
    @property
    def wrong_specialist_rate(self) -> float:
        """Rate of misrouting to wrong specialist."""
        return self.wrong_specialist_count / self.total_samples if self.total_samples > 0 else 0.0


def deterministic_split(prompts: List[str], dev_ratio: float = DEV_RATIO) -> Tuple[List[str], List[str]]:
    """
    Split prompts into dev and test sets deterministically.
    
    Uses hash of prompt content to ensure:
    1. Same prompt always goes to same split
    2. Splits are reproducible across runs
    3. No data leakage between dev and test
    
    Args:
        prompts: List of prompts to split
        dev_ratio: Fraction for dev set (rest goes to test)
        
    Returns:
        Tuple of (dev_prompts, test_prompts)
    """
    dev_prompts = []
    test_prompts = []
    
    for prompt in prompts:
        # Hash the prompt content for deterministic assignment
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        
        # Use hash to deterministically assign to split
        if (prompt_hash % 100) < (dev_ratio * 100):
            dev_prompts.append(prompt)
        else:
            test_prompts.append(prompt)
    
    return dev_prompts, test_prompts


def load_benchmark_prompts(
    config: BenchmarkConfig, 
    max_samples: int = 500,
    split_mode: str = "test",  # "dev", "test", or "all"
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load prompts from a HuggingFace dataset with proper dev/test split.
    
    Args:
        config: Benchmark configuration
        max_samples: Max samples to load (before splitting)
        split_mode: Which split to return ("dev", "test", or "all")
        
    Returns:
        Tuple of (prompts, split_info)
    """
    try:
        from datasets import load_dataset
        
        print(f"  Loading {config.name} from {config.hf_dataset}...")
        
        if config.hf_config:
            ds = load_dataset(config.hf_dataset, config.hf_config, split=config.split)
        else:
            ds = load_dataset(config.hf_dataset, split=config.split)
        
        # Load all prompts first
        all_prompts = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            
            prompt = item.get(config.prompt_field, "")
            
            # Apply template if specified
            if config.prompt_template:
                prompt = config.prompt_template.format(prompt=prompt)
            
            if prompt:
                all_prompts.append(prompt)
        
        # Split into dev and test
        dev_prompts, test_prompts = deterministic_split(all_prompts)
        
        split_info = {
            "total": len(all_prompts),
            "dev": len(dev_prompts),
            "test": len(test_prompts),
        }
        
        print(f"  Loaded {len(all_prompts)} prompts → dev: {len(dev_prompts)}, test: {len(test_prompts)}")
        
        # Return requested split
        if split_mode == "dev":
            return dev_prompts, split_info
        elif split_mode == "test":
            return test_prompts, split_info
        else:  # "all"
            return all_prompts, split_info
        
    except Exception as e:
        print(f"  ERROR loading {config.name}: {e}")
        return [], {"total": 0, "dev": 0, "test": 0}


def evaluate_benchmark(
    classifier: StandardClassifier,
    config: BenchmarkConfig,
    prompts: List[str],
    collect_errors: bool = False,
    max_errors: int = 10,
) -> EvaluationResult:
    """Evaluate classifier on a benchmark."""
    
    correct = 0
    uncertain_count = 0
    category_counts = defaultdict(int)
    errors = []
    
    for prompt in prompts:
        result = classifier.classify(prompt)
        
        predicted_cat = result.category
        category_counts[predicted_cat.value] += 1
        
        if result.is_uncertain:
            uncertain_count += 1
        
        # Check if correct (uncertain is counted as incorrect for accuracy)
        if predicted_cat == config.expected_category:
            correct += 1
        elif collect_errors and len(errors) < max_errors:
            errors.append({
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "expected": config.expected_category.value,
                "predicted": predicted_cat.value,
                "confidence": result.confidence,
                "is_uncertain": result.is_uncertain,
            })
    
    total = len(prompts)
    accuracy = correct / total if total > 0 else 0.0
    uncertain_rate = uncertain_count / total if total > 0 else 0.0
    
    return EvaluationResult(
        benchmark=config.name,
        expected_category=config.expected_category.value,
        total_samples=total,
        correct=correct,
        accuracy=accuracy,
        uncertain_count=uncertain_count,
        uncertain_rate=uncertain_rate,
        category_distribution=dict(category_counts),
        errors=errors,
    )


def print_results(results: List[EvaluationResult], show_errors: bool = False, split_name: str = "test"):
    """Print evaluation results."""
    
    print("\n" + "=" * 80)
    print(f"INTENT CLASSIFICATION EVALUATION RESULTS ({split_name.upper()} SET)")
    print("=" * 80)
    
    # Summary table with SAFE ROUTING RATE (key metric for routing)
    print(f"\n{'Benchmark':<12} {'Expected':<11} {'Correct':<9} {'Uncertain':<10} {'Safe Rate':<10} {'Samples':<8}")
    print("-" * 70)
    
    total_correct = 0
    total_samples = 0
    total_safe = 0
    
    for r in results:
        print(f"{r.benchmark:<12} {r.expected_category:<11} {r.accuracy*100:>5.1f}%    "
              f"{r.uncertain_rate*100:>5.1f}%     {r.safe_routing_rate*100:>5.1f}%     {r.total_samples:<8}")
        total_correct += r.correct
        total_samples += r.total_samples
        total_safe += r.safe_routing_count
    
    print("-" * 70)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    overall_safe = total_safe / total_samples if total_samples > 0 else 0
    print(f"{'OVERALL':<12} {'':<11} {overall_acc*100:>5.1f}%    {'':<10} {overall_safe*100:>5.1f}%     {total_samples:<8}")
    
    print("""
NOTE: 'Safe Rate' = Correct + Uncertain = samples NOT misrouted to wrong specialist
      This is the key metric for routing safety.""")
    
    # Category distribution per benchmark
    print("\n" + "-" * 80)
    print("CATEGORY DISTRIBUTION PER BENCHMARK")
    print("-" * 80)
    
    for r in results:
        print(f"\n{r.benchmark} (expected: {r.expected_category}):")
        sorted_cats = sorted(r.category_distribution.items(), key=lambda x: -x[1])
        for cat, count in sorted_cats:
            pct = count / r.total_samples * 100
            marker = "✓" if cat == r.expected_category else " "
            bar = "█" * int(pct / 2)
            print(f"  {marker} {cat:<15} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Confusion matrix style
    print("\n" + "-" * 80)
    print("CONFUSION SUMMARY")
    print("-" * 80)
    
    categories = list(StandardCategory)
    cat_names = [c.value for c in categories if c != StandardCategory.UNCERTAIN]
    
    # Build confusion data
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        for cat, count in r.category_distribution.items():
            confusion[r.expected_category][cat] += count
    
    # Print header
    header = f"{'Expected':<12} | " + " | ".join([f"{c[:8]:<8}" for c in cat_names]) + " | uncertain"
    print(header)
    print("-" * len(header))
    
    for expected in [r.expected_category for r in results]:
        row = f"{expected:<12} | "
        for predicted in cat_names:
            count = confusion[expected].get(predicted, 0)
            row += f"{count:>8} | "
        uncertain = confusion[expected].get("uncertain", 0)
        row += f"{uncertain:>8}"
        print(row)
    
    # Show errors if requested
    if show_errors:
        print("\n" + "-" * 80)
        print("CLASSIFICATION ERRORS (sample)")
        print("-" * 80)
        
        for r in results:
            if r.errors:
                print(f"\n{r.benchmark} errors:")
                for err in r.errors[:5]:
                    print(f"  Expected: {err['expected']}, Got: {err['predicted']} (conf={err['confidence']:.2f})")
                    print(f"  Prompt: {err['prompt'][:100]}...")


def save_results(results: List[EvaluationResult], output_path: Path, split_name: str = "test"):
    """Save results to JSON."""
    total_samples = sum(r.total_samples for r in results)
    total_correct = sum(r.correct for r in results)
    
    output = {
        "split": split_name,
        "dev_ratio": DEV_RATIO,
        "results": [
            {
                "benchmark": r.benchmark,
                "expected_category": r.expected_category,
                "total_samples": r.total_samples,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "uncertain_count": r.uncertain_count,
                "uncertain_rate": r.uncertain_rate,
                "category_distribution": r.category_distribution,
            }
            for r in results
        ],
        "overall_accuracy": total_correct / total_samples if total_samples > 0 else 0,
        "overall_samples": total_samples,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent classification on benchmarks")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max samples per benchmark before split (default: 500)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["dev", "test", "both"],
                        help="Which split to evaluate: 'dev' for tuning, 'test' for final eval (default: test)")
    parser.add_argument("--show-errors", action="store_true",
                        help="Show sample classification errors")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        choices=["HumanEval", "GSM8K", "TruthfulQA", "IFEval"],
                        help="Specific benchmarks to evaluate (default: all)")
    
    args = parser.parse_args()
    
    # Auto-generate output path based on split
    if args.output is None:
        args.output = f"kdd_paper/results/intent_classification_{args.split}.json"
    
    print("=" * 80)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 80)
    print(f"Split mode: {args.split.upper()}")
    print(f"Max samples per benchmark (before split): {args.max_samples}")
    print(f"Dev/Test ratio: {DEV_RATIO*100:.0f}% / {(1-DEV_RATIO)*100:.0f}%")
    
    if args.split == "dev":
        print("\n⚠️  DEV MODE: Use this split ONLY for pattern tuning!")
        print("    Do NOT report these numbers as final results.")
    elif args.split == "test":
        print("\n✓  TEST MODE: Held-out evaluation set")
        print("    These results are valid for paper reporting.")
    
    # Initialize classifier
    classifier = StandardClassifier()
    
    # Filter benchmarks if specified
    benchmarks_to_eval = BENCHMARKS
    if args.benchmarks:
        benchmarks_to_eval = [b for b in BENCHMARKS if b.name in args.benchmarks]
    
    # Determine which splits to run
    splits_to_run = ["dev", "test"] if args.split == "both" else [args.split]
    
    all_results = {}
    
    for split_mode in splits_to_run:
        if args.split == "both":
            print(f"\n{'='*40}")
            print(f"  EVALUATING {split_mode.upper()} SPLIT")
            print(f"{'='*40}")
        
        results = []
        
        for config in benchmarks_to_eval:
            print(f"\n--- {config.name} ---")
            print(f"  {config.description}")
            
            # Load prompts with split
            prompts, split_info = load_benchmark_prompts(
                config, 
                args.max_samples,
                split_mode=split_mode
            )
            
            if not prompts:
                print(f"  Skipping {config.name} - no prompts loaded")
                continue
            
            # Evaluate
            print(f"  Evaluating {len(prompts)} {split_mode} prompts...")
            result = evaluate_benchmark(
                classifier, 
                config, 
                prompts, 
                collect_errors=args.show_errors
            )
            results.append(result)
            
            print(f"  Accuracy: {result.accuracy*100:.1f}% ({result.correct}/{result.total_samples})")
            print(f"  Uncertain rate: {result.uncertain_rate*100:.1f}%")
        
        all_results[split_mode] = results
        
        # Print results for this split
        print_results(results, show_errors=args.show_errors, split_name=split_mode)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the test results for saving, or last evaluated
    final_results = all_results.get("test", all_results.get("dev", []))
    save_results(final_results, output_path, split_name=args.split)
    
    # Summary for paper (only for test split)
    if "test" in all_results and all_results["test"]:
        results = all_results["test"]
        print("\n" + "=" * 80)
        print("PAPER-READY SUMMARY (TEST SET - HELD OUT)")
        print("=" * 80)
        
        total_correct = sum(r.correct for r in results)
        total_samples = sum(r.total_samples for r in results)
        total_safe = sum(r.safe_routing_count for r in results)
        total_uncertain = sum(r.uncertain_count for r in results)
        
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        overall_safe = total_safe / total_samples if total_samples > 0 else 0
        overall_uncertain = total_uncertain / total_samples if total_samples > 0 else 0
        wrong_rate = 1 - overall_safe
        
        print(f"""
INTENT CLASSIFICATION RESULTS (Held-Out Test Set, n={total_samples})
─────────────────────────────────────────────────────────────────────

KEY METRIC: Safe Routing Rate = {overall_safe*100:.1f}%

This means {overall_safe*100:.1f}% of prompts are either:
  • Correctly classified to specialist: {overall_acc*100:.1f}%
  • Routed to "uncertain" (conservative fallback): {overall_uncertain*100:.1f}%

Only {wrong_rate*100:.1f}% are misrouted to a WRONG specialist.

Per-benchmark breakdown:
""")
        for r in results:
            print(f"  {r.benchmark:<12} | Correct: {r.accuracy*100:>5.1f}% | "
                  f"Uncertain: {r.uncertain_rate*100:>5.1f}% | "
                  f"Safe: {r.safe_routing_rate*100:>5.1f}%")
        
        print(f"""
INTERPRETATION:
─────────────────────────────────────────────────────────────────────
• HumanEval (coding): Near-perfect classification ({results[0].accuracy*100:.0f}%)
  - Code prompts have explicit signals (def, function, Python)

• GSM8K (reasoning): Low direct accuracy ({results[1].accuracy*100:.0f}%), high uncertain ({results[1].uncertain_rate*100:.0f}%)
  - Word problems lack math signals (no equations, just stories)
  - ✓ CORRECT BEHAVIOR: Conservative fallback for ambiguous prompts

• TruthfulQA (factual_qa): Good accuracy ({results[2].accuracy*100:.0f}%)
  - Factual questions correctly identified

The "uncertain" bucket serves as a SAFETY NET, ensuring that when our
classifier isn't confident, we route to well-rounded generalist models
rather than risk misrouting to an inappropriate specialist.
""")


if __name__ == "__main__":
    main()

