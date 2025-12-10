#!/usr/bin/env python3
"""
TruthfulQA Evaluation Script for LLM Jury
==========================================

Evaluates models on the TruthfulQA benchmark to get hallucination metrics
for models missing this data in our cache.

Usage:
    python run_truthfulqa_evaluation.py --estimate-cost
    python run_truthfulqa_evaluation.py --model "GPT-4 Turbo"
    python run_truthfulqa_evaluation.py --all-missing
    python run_truthfulqa_evaluation.py --all-cache
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""
    name: str                    # Our internal name
    provider: str                # API provider
    api_model_id: str            # API model identifier
    input_cost_per_1m: float     # Cost per 1M input tokens
    output_cost_per_1m: float    # Cost per 1M output tokens
    
# Models missing hallucination data that we want to evaluate
# NOTE: Only includes models verified to work with current API keys
MISSING_MODELS = [
    # Anthropic (working)
    ModelConfig("Claude 3 Opus", "anthropic", "claude-3-opus-20240229", 15.0, 75.0),
    ModelConfig("Claude 3.5 Haiku", "anthropic", "claude-3-5-haiku-20241022", 1.0, 5.0),
    ModelConfig("Claude 3 Haiku", "anthropic", "claude-3-haiku-20240307", 0.25, 1.25),
    
    # OpenAI (working - o1 models not available)
    ModelConfig("GPT-4 Turbo", "openai", "gpt-4-turbo", 10.0, 30.0),
    ModelConfig("GPT-4", "openai", "gpt-4", 30.0, 60.0),
    ModelConfig("GPT-4o", "openai", "gpt-4o", 2.5, 10.0),
    
    # Google (only 2.0 Flash working)
    ModelConfig("Gemini 2.0 Flash", "google", "gemini-2.0-flash-exp", 0.075, 0.30),
    
    # Meta (via Together AI - working)
    ModelConfig("Llama 3.1 405B", "together", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", 3.5, 3.5),
    ModelConfig("Llama 3.1 70B", "together", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 0.88, 0.88),
    
    # Mistral (working)
    ModelConfig("Mistral Large", "mistral", "mistral-large-latest", 2.0, 6.0),
    ModelConfig("Codestral", "mistral", "codestral-latest", 0.2, 0.6),
    ModelConfig("Mixtral 8x7B", "together", "mistralai/Mixtral-8x7B-Instruct-v0.1", 0.6, 0.6),
    
    # Cohere (working - updated model IDs as of Dec 2025)
    ModelConfig("Command R+", "cohere", "command-r-plus-08-2024", 2.5, 10.0),
    ModelConfig("Command R", "cohere", "command-r-08-2024", 0.15, 0.6),
    ModelConfig("Command A", "cohere", "command-a-03-2025", 2.5, 10.0),
]

# Models already in cache that we want to verify/update
CACHE_MODELS_TO_VERIFY = [
    ModelConfig("GPT-4", "openai", "gpt-4", 30.0, 60.0),
    ModelConfig("GPT-4o", "openai", "gpt-4o", 2.5, 10.0),
    ModelConfig("GPT-3.5 Turbo", "openai", "gpt-3.5-turbo", 0.5, 1.5),
    ModelConfig("Claude 3 Haiku", "anthropic", "claude-3-haiku-20240307", 0.25, 1.25),
]

# ============================================================================
# TRUTHFULQA DATASET
# ============================================================================

def load_truthfulqa_dataset() -> List[Dict]:
    """Load TruthfulQA dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
        return list(ds)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Install with: pip install datasets")
        return []

def format_mc_question(item: Dict) -> Tuple[str, int]:
    """Format a multiple choice question and return correct answer index."""
    question = item['question']
    choices = item['mc1_targets']['choices']
    labels = item['mc1_targets']['labels']
    correct_idx = labels.index(1)
    
    prompt = f"Question: {question}\n\nOptions:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer with just the letter (A, B, C, etc.) of the correct answer."
    
    return prompt, correct_idx

# ============================================================================
# API CLIENTS
# ============================================================================

def call_openai(model_id: str, prompt: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def call_anthropic(model_id: str, prompt: str) -> str:
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model=model_id,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def call_google(model_id: str, prompt: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = genai.GenerativeModel(model_id)
    response = model.generate_content(prompt)
    return response.text.strip()

def call_mistral(model_id: str, prompt: str) -> str:
    """Call Mistral API."""
    from mistralai import Mistral
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    response = client.chat.complete(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def call_cohere(model_id: str, prompt: str) -> str:
    """Call Cohere API."""
    import cohere
    client = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    response = client.chat(
        model=model_id,
        message=prompt,
    )
    return response.text.strip()

def call_together(model_id: str, prompt: str) -> str:
    """Call Together AI API."""
    from together import Together
    client = Together()
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def call_model(config: ModelConfig, prompt: str) -> str:
    """Call the appropriate API based on provider."""
    providers = {
        "openai": call_openai,
        "anthropic": call_anthropic,
        "google": call_google,
        "mistral": call_mistral,
        "cohere": call_cohere,
        "together": call_together,
    }
    
    if config.provider not in providers:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    return providers[config.provider](config.api_model_id, prompt)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(config: ModelConfig, dataset: List[Dict], 
                   max_questions: int = 817, 
                   save_interval: int = 50) -> Dict:
    """
    Evaluate a model on TruthfulQA.
    
    Returns:
        Dict with accuracy, correct count, total count, and per-question results
    """
    results = {
        "model": config.name,
        "provider": config.provider,
        "api_model_id": config.api_model_id,
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "questions": []
    }
    
    output_dir = Path("kdd_paper/truthfulqa_results")
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / f"{config.name.replace(' ', '_').replace('/', '_')}_checkpoint.json"
    
    # Load checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            results = json.load(f)
            start_idx = results["total"]
            print(f"Resuming from question {start_idx}")
    
    questions_to_eval = min(max_questions, len(dataset))
    
    for i in range(start_idx, questions_to_eval):
        item = dataset[i]
        prompt, correct_idx = format_mc_question(item)
        
        try:
            response = call_model(config, prompt)
            
            # Parse answer
            answer_letter = response.upper().strip()
            if len(answer_letter) > 0:
                answer_letter = answer_letter[0]
            
            predicted_idx = ord(answer_letter) - ord('A') if answer_letter.isalpha() else -1
            is_correct = predicted_idx == correct_idx
            
            if is_correct:
                results["correct"] += 1
            results["total"] += 1
            
            results["questions"].append({
                "idx": i,
                "question": item["question"],
                "predicted": answer_letter,
                "correct": chr(65 + correct_idx),
                "is_correct": is_correct
            })
            
            # Progress
            if (i + 1) % 10 == 0:
                acc = results["correct"] / results["total"] * 100
                print(f"  [{config.name}] {i+1}/{questions_to_eval} - Accuracy: {acc:.1f}%")
            
            # Save checkpoint
            if (i + 1) % save_interval == 0:
                results["accuracy"] = results["correct"] / results["total"] * 100
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Rate limiting
            time.sleep(0.5)  # Adjust based on API limits
            
        except Exception as e:
            print(f"  Error on question {i}: {e}")
            time.sleep(2)  # Back off on error
            continue
    
    # Final accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"] * 100
    
    # Save final results
    final_file = output_dir / f"{config.name.replace(' ', '_').replace('/', '_')}_results.json"
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return results

# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_costs(models: List[ModelConfig], num_questions: int = 817) -> Dict:
    """
    Estimate API costs for running TruthfulQA evaluation.
    
    Assumptions:
    - Average prompt length: ~150 tokens (question + options)
    - Average response length: ~5 tokens (just the letter)
    """
    AVG_INPUT_TOKENS = 150
    AVG_OUTPUT_TOKENS = 5
    
    total_input_tokens = num_questions * AVG_INPUT_TOKENS
    total_output_tokens = num_questions * AVG_OUTPUT_TOKENS
    
    estimates = []
    total_cost = 0
    
    for model in models:
        input_cost = (total_input_tokens / 1_000_000) * model.input_cost_per_1m
        output_cost = (total_output_tokens / 1_000_000) * model.output_cost_per_1m
        model_cost = input_cost + output_cost
        total_cost += model_cost
        
        estimates.append({
            "model": model.name,
            "provider": model.provider,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": model_cost
        })
    
    return {
        "num_questions": num_questions,
        "avg_input_tokens": AVG_INPUT_TOKENS,
        "avg_output_tokens": AVG_OUTPUT_TOKENS,
        "models": estimates,
        "grand_total": total_cost
    }

def print_cost_estimate(estimates: Dict):
    """Print formatted cost estimate."""
    print("\n" + "=" * 80)
    print("TRUTHFULQA EVALUATION COST ESTIMATE")
    print("=" * 80)
    print(f"Questions: {estimates['num_questions']}")
    print(f"Avg input tokens/question: {estimates['avg_input_tokens']}")
    print(f"Avg output tokens/question: {estimates['avg_output_tokens']}")
    print()
    print(f"{'Model':<30} {'Provider':<12} {'Input $':<10} {'Output $':<10} {'Total $':<10}")
    print("-" * 80)
    
    for m in estimates['models']:
        print(f"{m['model']:<30} {m['provider']:<12} ${m['input_cost']:<9.4f} ${m['output_cost']:<9.4f} ${m['total_cost']:<9.4f}")
    
    print("-" * 80)
    print(f"{'GRAND TOTAL':<54} ${estimates['grand_total']:.2f}")
    print("=" * 80)

# ============================================================================
# CACHE UPDATE
# ============================================================================

# Name mappings from TruthfulQA evaluation names to cache names
TRUTHFULQA_TO_CACHE_NAME = {
    "Llama 3.1 70B": "Llama 3.1 Instruct 70B",
    "Llama 3.1 405B": "Llama 3.1 Instruct 405B",
    "Llama 3.1 8B": "Llama 3.1 Instruct 8B",
    "Mixtral 8x7B": "Mixtral 8x7B Instruct",
    "Mixtral 8x22B": "Mixtral 8x22B Instruct",
}

def get_cache_name(model_name: str) -> str:
    """Get the cache name for a model, applying mappings if needed."""
    return TRUTHFULQA_TO_CACHE_NAME.get(model_name, model_name)

def update_cache_with_results(results: Dict):
    """Update models_cache.json with TruthfulQA results."""
    cache_path = Path("data/models_cache.json")
    
    with open(cache_path) as f:
        cache_data = json.load(f)
    
    # Handle both old format (list) and new format (dict with 'models' key)
    if isinstance(cache_data, dict) and 'models' in cache_data:
        models = cache_data['models']
    else:
        models = cache_data
    
    # Find and update the model using name mapping
    model_name = results["model"]
    cache_name = get_cache_name(model_name)
    
    found = False
    for m in models:
        if m.get("name") == cache_name:
            # TruthfulQA accuracy maps to "truthfulness" not directly to hallucination
            # Higher TruthfulQA = lower hallucination tendency
            # Rough conversion: hallucination_rate ≈ 100 - truthfulqa_accuracy
            m["truthfulqa_accuracy"] = results["accuracy"]
            m["truthfulqa_correct"] = results["correct"]
            m["truthfulqa_total"] = results["total"]
            
            # If no hallucination rate, estimate from TruthfulQA
            # Note: This is an approximation - Vectara and TruthfulQA measure different things
            if m.get("hallucination_rate") is None:
                # Conservative estimate: TruthfulQA errors / 2 as proxy
                m["hallucination_rate_estimated"] = (100 - results["accuracy"]) / 2
            
            print(f"Updated {cache_name} in cache (from {model_name}):")
            print(f"  TruthfulQA Accuracy: {results['accuracy']:.1f}%")
            found = True
            break
    
    if not found:
        print(f"WARNING: Model '{model_name}' (cache name: '{cache_name}') not found in cache")
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Evaluation for LLM Jury")
    parser.add_argument("--estimate-cost", action="store_true", 
                        help="Estimate costs without running evaluation")
    parser.add_argument("--model", type=str, 
                        help="Evaluate a specific model by name")
    parser.add_argument("--all-missing", action="store_true",
                        help="Evaluate all models missing hallucination data")
    parser.add_argument("--all-cache", action="store_true",
                        help="Evaluate models already in cache for verification")
    parser.add_argument("--max-questions", type=int, default=817,
                        help="Maximum questions to evaluate (default: 817)")
    parser.add_argument("--update-cache", action="store_true",
                        help="Update models_cache.json with results")
    
    args = parser.parse_args()
    
    # Cost estimation
    if args.estimate_cost:
        print("\n=== MISSING MODELS (Need hallucination data) ===")
        missing_estimates = estimate_costs(MISSING_MODELS, args.max_questions)
        print_cost_estimate(missing_estimates)
        
        print("\n=== CACHE MODELS (Verification) ===")
        cache_estimates = estimate_costs(CACHE_MODELS_TO_VERIFY, args.max_questions)
        print_cost_estimate(cache_estimates)
        
        combined_total = missing_estimates['grand_total'] + cache_estimates['grand_total']
        print(f"\n*** COMBINED TOTAL: ${combined_total:.2f} ***")
        return
    
    # Load dataset
    print("Loading TruthfulQA dataset...")
    dataset = load_truthfulqa_dataset()
    if not dataset:
        print("Failed to load dataset. Exiting.")
        return
    print(f"Loaded {len(dataset)} questions")
    
    # Determine which models to evaluate
    models_to_eval = []
    
    if args.model:
        # Find specific model
        all_models = MISSING_MODELS + CACHE_MODELS_TO_VERIFY
        for m in all_models:
            if m.name.lower() == args.model.lower():
                models_to_eval.append(m)
                break
        if not models_to_eval:
            print(f"Model '{args.model}' not found in configuration")
            return
    elif args.all_missing:
        models_to_eval = MISSING_MODELS
    elif args.all_cache:
        models_to_eval = CACHE_MODELS_TO_VERIFY
    else:
        print("Please specify --model, --all-missing, or --all-cache")
        parser.print_help()
        return
    
    # Run evaluations
    all_results = []
    for config in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config.name}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_model(config, dataset, args.max_questions)
            all_results.append(results)
            
            print(f"\n{config.name} Results:")
            print(f"  Accuracy: {results['accuracy']:.1f}%")
            print(f"  Correct: {results['correct']}/{results['total']}")
            
            if args.update_cache:
                update_cache_with_results(results)
                
        except Exception as e:
            print(f"Error evaluating {config.name}: {e}")
            continue
    
    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'Accuracy':<10} {'Correct':<10}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['model']:<30} {r['accuracy']:.1f}%{'':<5} {r['correct']}/{r['total']}")

if __name__ == "__main__":
    main()

