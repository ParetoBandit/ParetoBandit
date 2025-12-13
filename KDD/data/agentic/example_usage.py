#!/usr/bin/env python3
"""
Example: Complete GAIA evaluation workflow.

This script demonstrates how to:
1. Fetch GAIA problems
2. Generate answers with an LLM/agent
3. Evaluate with exact match
4. Compute accuracy

Usage:
    export HF_TOKEN="your_huggingface_token"
    export OPENROUTER_API_KEY="your_key"
    python example_usage.py --model "anthropic/claude-3.5-sonnet"
"""

import json
import argparse
from pathlib import Path
import subprocess
import os


def fetch_problems(split: str = "validation", level: Optional[int] = None, n_samples: Optional[int] = None, output_file: str = "problems.json"):
    """Step 1: Fetch GAIA problems."""
    print("="*60)
    print("Step 1: Fetching GAIA Problems")
    print("="*60)
    
    cmd = [
        "python", "fetch_gaia.py",
        "--split", split,
        "--output", output_file,
        "--prompt-style", "detailed"
    ]
    
    if level:
        cmd.extend(["--level", str(level)])
    
    if n_samples:
        cmd.extend(["--n-samples", str(n_samples)])
    
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
    """Step 2: Generate responses using LLM."""
    print("\n" + "="*60)
    print("Step 2: Generating Responses with LLM")
    print("="*60)
    
    try:
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
    print("Note: GAIA often requires tools. This example uses LLM-only (limited).\n")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    responses = {}
    for i, problem in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] {problem['task_id']}...", end=" ", flush=True)
        
        try:
            # Create a prompt
            prompt = problem['prompt']
            
            if problem.get('file_name'):
                prompt += f"\n\nNote: File '{problem['file_name']}' is referenced but not accessible in this example."
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Answer questions concisely with just the final answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            responses[problem['task_id']] = answer
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            responses[problem['task_id']] = f"Error: {e}"
    
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
    """Step 3: Evaluate responses."""
    print("\n" + "="*60)
    print("Step 3: Evaluating Responses")
    print("="*60)
    
    cmd = [
        "python", "evaluate_gaia.py",
        "--problems", problems_file,
        "--responses", responses_file,
        "--output", output_file
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
    
    overall = data["metrics"]["overall"]
    by_level = data["metrics"].get("by_level", {})
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:         {overall['accuracy']*100:.1f}%")
    print(f"  Correct:          {overall['correct']}/{overall['total']}")
    
    if by_level:
        print(f"\n📊 By Difficulty Level:")
        for level in sorted(by_level.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            stats = by_level[level]
            difficulty = {1: "Easy", 2: "Medium", 3: "Hard"}.get(int(level) if str(level).isdigit() else 0, "Unknown")
            print(f"  Level {level} ({difficulty}): {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Show sample results
    results = data.get("results", [])
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]
    
    if correct:
        print(f"\n✓ Sample Correct Answers (showing {min(3, len(correct))}):")
        for r in correct[:3]:
            print(f"  - {r['task_id']}: {r['extracted_answer']}")
    
    if incorrect:
        print(f"\n❌ Sample Incorrect Answers (showing {min(3, len(incorrect))}):")
        for r in incorrect[:3]:
            print(f"  - {r['task_id']}")
            print(f"    Expected: {r['expected_answer']}")
            print(f"    Got:      {r['extracted_answer']}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete GAIA evaluation example"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic/claude-3.5-sonnet",
        help="Model ID for OpenRouter"
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["validation", "test"],
        help="GAIA split to use"
    )
    parser.add_argument(
        "--level", type=int, default=None,
        choices=[1, 2, 3],
        help="Filter by difficulty level"
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
    
    print("GAIA Evaluation Example")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    if args.level:
        print(f"Level: {args.level}")
    print(f"Problems: {args.n_samples}")
    print("="*60)
    print("\n⚠️  Note: GAIA requires tools (web search, file reading, etc.)")
    print("This example uses LLM-only, so results will be limited.\n")
    
    # File names
    problems_file = "example_gaia_problems.json"
    responses_file = "example_gaia_responses.json"
    results_file = "example_gaia_evaluation.json"
    
    try:
        # Step 1: Fetch problems
        if not args.skip_fetch:
            if not fetch_problems(args.split, args.level, args.n_samples, problems_file):
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
        print("\n💡 For better results, use an agent with tools!")
        
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
    from typing import Optional
    exit(main())
