#!/usr/bin/env python3
"""
Fetch LiveCodeBench prompts with test cases for code generation evaluation.

LiveCodeBench is a contamination-free coding benchmark containing problems from
after 2023. Each problem includes test cases (inputs and expected outputs) for
execution-based evaluation.

Dataset: LiveCodeBench (Code Generation Lite version)
Source: HuggingFace datasets
Evaluation: Pass@1 with unit test execution (local CPU, free)
Metric: Does the generated code run and pass all test cases?

Usage:
    python fetch_livecodebench.py --n-samples 100 --output prompts.json

Requirements:
    pip install datasets
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def fetch_livecodebench(
    version: str = "code_generation_lite",
    split: str = "test",
    n_samples: Optional[int] = None,
    seed: int = 42,
    min_date: str = "2024-01-01"
) -> List[Dict]:
    """
    Fetch LiveCodeBench dataset from HuggingFace.
    
    Args:
        version: Dataset version ("code_generation_lite" recommended)
        split: Dataset split (usually "test")
        n_samples: Number of samples to fetch (None = all)
        seed: Random seed for sampling
        min_date: Only include problems from after this date
        
    Returns:
        List of problem dictionaries with prompts and test cases
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not found. Install with: pip install datasets"
        )
    
    print(f"Fetching LiveCodeBench ({version}) from HuggingFace...")
    
    try:
        # LiveCodeBench dataset on HuggingFace
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            split=split,
            trust_remote_code=True
        )
        print(f"✓ Loaded {len(dataset)} problems from LiveCodeBench")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nNote: If the dataset name changed, check:")
        print("  https://huggingface.co/datasets/livecodebench")
        raise
    
    # Convert to list and filter by date
    problems = []
    for idx, item in enumerate(dataset):
        # Extract fields (adjust based on actual dataset schema)
        problem = {
            "problem_id": item.get("question_id", f"lcb_{idx}"),
            "title": item.get("question_title", item.get("title", f"Problem {idx}")),
            "difficulty": item.get("difficulty", "Unknown"),
            "description": item.get("question_content", item.get("description", "")),
            "starter_code": item.get("starter_code", ""),
            "test_cases": [],
            "metadata": {
                "platform": item.get("platform", "Unknown"),
                "contest_date": item.get("contest_date", ""),
                "topics": item.get("topics", []),
                "url": item.get("question_url", ""),
            }
        }
        
        # Extract test cases
        # LiveCodeBench has different formats, try common field names
        if "public_test_cases" in item:
            test_cases = item["public_test_cases"]
        elif "test_cases" in item:
            test_cases = item["test_cases"]
        elif "input_output" in item and isinstance(item["input_output"], dict):
            # Parse input_output JSON format
            io_data = item["input_output"]
            if "inputs" in io_data and "outputs" in io_data:
                test_cases = [
                    {"input": inp, "output": out}
                    for inp, out in zip(io_data["inputs"], io_data["outputs"])
                ]
            else:
                test_cases = []
        else:
            test_cases = []
        
        # Normalize test case format
        normalized_tests = []
        for tc in test_cases:
            if isinstance(tc, dict):
                normalized_tests.append({
                    "input": tc.get("input", tc.get("stdin", "")),
                    "output": tc.get("output", tc.get("stdout", tc.get("expected_output", ""))),
                    "explanation": tc.get("explanation", "")
                })
            elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
                normalized_tests.append({
                    "input": tc[0],
                    "output": tc[1],
                    "explanation": ""
                })
        
        problem["test_cases"] = normalized_tests
        
        # Filter by date if specified
        contest_date = problem["metadata"].get("contest_date", "")
        if min_date and contest_date and contest_date < min_date:
            continue
        
        problems.append(problem)
    
    print(f"✓ Extracted {len(problems)} problems with test cases")
    
    # Sample if requested
    if n_samples and len(problems) > n_samples:
        random.seed(seed)
        problems = random.sample(problems, n_samples)
        print(f"✓ Sampled {n_samples} problems (seed={seed})")
    
    return problems


def create_prompt(problem: Dict, prompt_style: str = "simple") -> str:
    """
    Create a prompt from a LiveCodeBench problem.
    
    Args:
        problem: Problem dictionary
        prompt_style: "simple", "detailed", or "leetcode"
        
    Returns:
        Formatted prompt string
    """
    if prompt_style == "simple":
        prompt = f"""Solve the following coding problem:

{problem['description']}

Write a complete Python function that solves this problem."""
        
    elif prompt_style == "detailed":
        prompt = f"""# Problem: {problem['title']}
Difficulty: {problem['difficulty']}

## Description
{problem['description']}

## Examples
"""
        # Add first few test cases as examples
        for i, tc in enumerate(problem['test_cases'][:3], 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Input: {tc['input']}\n"
            prompt += f"Output: {tc['output']}\n"
        
        prompt += "\n## Task\nWrite a complete Python function that solves this problem."
        
    elif prompt_style == "leetcode":
        prompt = f"{problem['title']}\n\n"
        prompt += f"{problem['description']}\n\n"
        
        if problem.get('starter_code'):
            prompt += f"```python\n{problem['starter_code']}\n```\n"
        else:
            prompt += "Write your solution in Python.\n"
    
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")
    
    return prompt


def analyze_dataset(problems: List[Dict]) -> Dict:
    """Analyze the fetched dataset."""
    stats = {
        "total_problems": len(problems),
        "problems_with_tests": sum(1 for p in problems if p["test_cases"]),
        "avg_test_cases": sum(len(p["test_cases"]) for p in problems) / len(problems) if problems else 0,
        "difficulties": {},
        "platforms": {},
    }
    
    for p in problems:
        diff = p.get("difficulty", "Unknown")
        stats["difficulties"][diff] = stats["difficulties"].get(diff, 0) + 1
        
        platform = p["metadata"].get("platform", "Unknown")
        stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fetch LiveCodeBench prompts with test cases"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of problems to fetch (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default="livecodebench_prompts.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--min-date", type=str, default="2024-01-01",
        help="Only include problems from after this date"
    )
    parser.add_argument(
        "--prompt-style", type=str, default="detailed",
        choices=["simple", "detailed", "leetcode"],
        help="Prompt formatting style"
    )
    
    args = parser.parse_args()
    
    try:
        # Fetch problems
        problems = fetch_livecodebench(
            n_samples=args.n_samples,
            seed=args.seed,
            min_date=args.min_date
        )
        
        if not problems:
            print("❌ No problems fetched")
            return
        
        # Add formatted prompts
        for problem in problems:
            problem["prompt"] = create_prompt(problem, args.prompt_style)
        
        # Analyze dataset
        stats = analyze_dataset(problems)
        
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total Problems:           {stats['total_problems']}")
        print(f"Problems with Test Cases: {stats['problems_with_tests']}")
        print(f"Avg Test Cases per Problem: {stats['avg_test_cases']:.1f}")
        
        if stats['difficulties']:
            print(f"\nDifficulty Distribution:")
            for diff, count in sorted(stats['difficulties'].items()):
                print(f"  {diff}: {count}")
        
        if stats['platforms']:
            print(f"\nPlatform Distribution:")
            for platform, count in sorted(stats['platforms'].items()):
                print(f"  {platform}: {count}")
        
        # Save to file
        output_data = {
            "metadata": {
                "dataset": "LiveCodeBench (Code Generation Lite)",
                "fetch_date": datetime.now().isoformat(),
                "n_problems": len(problems),
                "min_date": args.min_date,
                "seed": args.seed,
                "prompt_style": args.prompt_style,
            },
            "statistics": stats,
            "problems": problems
        }
        
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Saved {len(problems)} problems to {output_path}")
        
        # Show sample problem
        if problems:
            print("\n" + "="*60)
            print("Sample Problem")
            print("="*60)
            sample = problems[0]
            print(f"ID: {sample['problem_id']}")
            print(f"Title: {sample['title']}")
            print(f"Difficulty: {sample['difficulty']}")
            print(f"Test Cases: {len(sample['test_cases'])}")
            print(f"\nPrompt Preview:")
            print(sample['prompt'][:500] + "..." if len(sample['prompt']) > 500 else sample['prompt'])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
