#!/usr/bin/env python3
"""
End-to-End Example: Natural Questions RAG Evaluation

This script demonstrates the complete workflow:
1. Fetch Natural Questions dataset
2. Generate model responses (using OpenRouter API)
3. Evaluate using exact match
4. Save results

Note: Requires OPENROUTER_API_KEY environment variable
"""

import os
import json
import time
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from fetch_natural_questions import (
    fetch_natural_questions,
    create_rag_prompt,
    save_questions
)
from evaluate_natural_questions import (
    NaturalQuestionsEvaluator,
    save_evaluation_results
)

# Optional: Import OpenAI client for API calls
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")


def generate_responses_with_llm(
    questions: List[Dict],
    model_id: str = "openai/gpt-4o-mini",
    max_tokens: int = 100,
    temperature: float = 0.0
) -> List[str]:
    """
    Generate responses using an LLM via OpenRouter.
    
    Args:
        questions: List of question dicts
        model_id: OpenRouter model ID
        max_tokens: Max tokens for response
        temperature: Sampling temperature
    
    Returns:
        List of model responses
    """
    if not HAS_OPENAI:
        raise ImportError("OpenAI library required. Install with: pip install openai")
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    responses = []
    print(f"\n🤖 Generating responses with {model_id}...")
    
    for i, q in enumerate(questions):
        try:
            # Create prompt
            prompt = create_rag_prompt(q["question"], style="standard")
            
            # Call API
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = completion.choices[0].message.content.strip()
            responses.append(response)
            
            print(f"  [{i+1}/{len(questions)}] ✓")
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] ✗ Error: {e}")
            responses.append("")  # Empty response on error
    
    print("✓ Response generation complete")
    return responses


def run_full_evaluation(
    n_samples: int = 50,
    model_id: str = "openai/gpt-4o-mini",
    output_dir: str = "results"
) -> None:
    """
    Run complete evaluation workflow.
    
    Args:
        n_samples: Number of questions to evaluate
        model_id: Model to evaluate
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*80)
    print("Natural Questions RAG Evaluation")
    print("="*80)
    
    # Step 1: Fetch questions
    print("\n📥 Step 1: Fetching Natural Questions dataset...")
    questions = fetch_natural_questions(
        split="validation",
        n_samples=n_samples,
        seed=42
    )
    
    questions_file = output_path / "questions.json"
    save_questions(questions, questions_file)
    
    # Step 2: Generate responses
    print(f"\n🤖 Step 2: Generating responses with {model_id}...")
    if HAS_OPENAI and os.getenv("OPENROUTER_API_KEY"):
        responses = generate_responses_with_llm(questions, model_id=model_id)
        
        # Save responses
        responses_file = output_path / "responses.json"
        with open(responses_file, 'w') as f:
            json.dump(responses, f, indent=2)
        print(f"💾 Saved responses to {responses_file}")
    else:
        print("⚠️  Skipping response generation (no API key or OpenAI library)")
        print("💡 Tip: Set OPENROUTER_API_KEY and install openai library")
        return
    
    # Step 3: Evaluate
    print("\n📊 Step 3: Evaluating responses...")
    evaluator = NaturalQuestionsEvaluator(case_sensitive=False)
    results, metrics = evaluator.evaluate(questions, responses)
    
    # Save results
    results_file = output_path / "evaluation_results.json"
    save_evaluation_results(results, metrics, results_file)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: {model_id}")
    print(f"Questions: {metrics['total_questions']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}")
    print("="*80)
    
    # Show examples
    print("\n📝 Sample Results (first 5):\n")
    for i, result in enumerate(results[:5], 1):
        status = "✓" if result.is_correct else "✗"
        print(f"{status} Q{i}: {result.question}")
        print(f"   Valid: {result.ground_truth}")
        print(f"   Model: {result.model_response[:100]}...")
        if result.matched_answer:
            print(f"   Matched: '{result.matched_answer}'")
        print()


def run_simple_demo():
    """
    Run a simple demo without API calls.
    Shows the evaluation process with mock data.
    """
    print("="*80)
    print("Natural Questions Evaluation - Simple Demo")
    print("="*80)
    
    # Fetch some questions
    print("\n📥 Fetching sample questions...")
    questions = fetch_natural_questions(
        split="validation",
        n_samples=5,
        seed=42
    )
    
    print(f"✓ Loaded {len(questions)} questions\n")
    
    # Show questions
    print("📋 Questions:")
    for i, q in enumerate(questions, 1):
        print(f"\n{i}. {q['question']}")
        print(f"   Valid Answers: {q['answers']}")
    
    # Show prompt example
    print("\n" + "="*80)
    print("📝 Example Prompt:")
    print("="*80)
    print(create_rag_prompt(questions[0]["question"], style="standard"))
    
    print("\n💡 To run full evaluation with a real model:")
    print("   1. Set OPENROUTER_API_KEY environment variable")
    print("   2. Install openai library: pip install openai")
    print("   3. Run: python example_usage.py --full")


def main():
    """Main entry point."""
    import sys
    
    if "--full" in sys.argv:
        # Full evaluation with API calls
        run_full_evaluation(
            n_samples=50,  # Adjust as needed
            model_id="openai/gpt-4o-mini",
            output_dir="results"
        )
    else:
        # Simple demo without API calls
        run_simple_demo()


if __name__ == "__main__":
    main()
