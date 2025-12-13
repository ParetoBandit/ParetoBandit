#!/usr/bin/env python3
"""
Example: Complete LiveCodeBench evaluation workflow.

This script demonstrates how to:
1. Fetch LiveCodeBench problems
2. Generate code with an LLM
3. Evaluate with execution-based testing
4. Compute Pass@1 metric

Usage:
    export OPENROUTER_API_KEY="your_key"
    python example_usage.py --model "anthropic/claude-3.5-sonnet"
"""

import json
import argparse
from pathlib import Path
import subprocess


def fetch_problems(n_samples: int = 10, output_file: str = "problems.json"):
    """Step 1: Fetch problems from HuggingFace."""
    print("="*60)
    print("Step 1: Fetching LiveCodeBench Problems")
    print("="*60)
    
    cmd = [
        "python", "fetch_livecodebench.py",
        "--n-samples", str(n_samples),
        "--output", output_file,
        "--prompt-style", "detailed"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    return True


def generate_responses(
    problems_file: str,
    model_id: str,
    output_file: str = "responses.json"
):
    """Step 2: Generate code responses using LLM."""
    print("\n" + "="*60)
    print("Step 2: Generating Code with LLM")
    print("="*60)
    
    try:
        import os
        from openai import OpenAI
    except ImportError:
        print("❌ OpenAI library not found. Install with: pip install openai")
        return False
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        return False
    
    # Load problems
    with open(problems_file) as f:
        data = json.load(f)
    problems = data.get("problems", [])
    
    print(f"Generating responses for {len(problems)} problems using {model_id}...")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    responses = {}
    for i, problem in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] {problem['problem_id']}...", end=" ", flush=True)
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert programmer. Write clean, correct Python code."},
                    {"role": "user", "content": problem['prompt']}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content
            
            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            responses[problem['problem_id']] = code
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            responses[problem['problem_id']] = f"# Error: {e}"
    
    # Save responses
    with open(output_file, "w") as f:
        json.dump({"responses": responses}, f, indent=2)
    
    print(f"✓ Saved {len(responses)} responses to {output_file}")
    return True


def evaluate_responses(
    problems_file: str,
    responses_file: str,
    output_file: str = "evaluation.json"
):
    """Step 3: Evaluate responses with execution."""
    print("\n" + "="*60)
    print("Step 3: Evaluating Code Execution")
    print("="*60)
    
    cmd = [
        "python", "evaluate_code.py",
        "--problems", problems_file,
        "--responses", responses_file,
        "--output", output_file,
        "--timeout", "5"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    return True


def display_results(results_file: str):
    """Step 4: Display results summary."""
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    
    with open(results_file) as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Pass@1:              {metrics['pass_at_1']*100:.1f}%")
    print(f"  Problems Passed:     {metrics['problems_passed']}/{metrics['problems_evaluated']}")
    print(f"  Total Tests Passed:  {metrics['total_tests_passed']}/{metrics['total_tests']}")
    
    # Show failed problems
    failed = [r for r in data["results"] if not r["passed"]]
    if failed:
        print(f"\n❌ Failed Problems ({len(failed)}):")
        for r in failed[:5]:  # Show first 5
            print(f"  - {r['problem_id']}: {r['reason']}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more")
    
    # Show passed problems
    passed = [r for r in data["results"] if r["passed"]]
    if passed:
        print(f"\n✓ Passed Problems ({len(passed)}):")
        for r in passed[:5]:  # Show first 5
            print(f"  - {r['problem_id']}: {r['tests_passed']}/{r['tests_total']} tests")
        if len(passed) > 5:
            print(f"  ... and {len(passed)-5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Complete LiveCodeBench evaluation example"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic/claude-3.5-sonnet",
        help="Model ID for OpenRouter"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of problems to evaluate"
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip fetching problems (use existing)"
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip generating responses (use existing)"
    )
    
    args = parser.parse_args()
    
    print("LiveCodeBench Evaluation Example")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Problems: {args.n_samples}")
    print("="*60)
    
    # File names
    problems_file = "example_problems.json"
    responses_file = "example_responses.json"
    results_file = "example_evaluation.json"
    
    try:
        # Step 1: Fetch problems
        if not args.skip_fetch:
            if not fetch_problems(args.n_samples, problems_file):
                return 1
        else:
            print(f"Skipping fetch, using {problems_file}")
        
        # Step 2: Generate responses
        if not args.skip_generate:
            if not generate_responses(problems_file, args.model, responses_file):
                return 1
        else:
            print(f"Skipping generation, using {responses_file}")
        
        # Step 3: Evaluate
        if not evaluate_responses(problems_file, responses_file, results_file):
            return 1
        
        # Step 4: Display results
        display_results(results_file)
        
        print("\n" + "="*60)
        print("✓ Complete! Check files:")
        print(f"  - Problems: {problems_file}")
        print(f"  - Responses: {responses_file}")
        print(f"  - Results: {results_file}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
