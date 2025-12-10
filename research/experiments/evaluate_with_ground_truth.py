#!/usr/bin/env python3
"""
Evaluate LLM Jury vs FrugalGPT using actual HEADLINES ground truth labels.
"""

import os
import sys
import json
import csv
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_jury import get_best_models_for_budget

PAPER_DIR = Path(__file__).parent
DATA_DIR = PAPER_DIR / "frugalgpt_data"
FIGURES_DIR = PAPER_DIR / "figures"

VALID_LABELS = {"up", "down", "neutral", "none"}

# FrugalGPT reported results
FRUGALGPT_RESULTS = {
    0.5: ("J1-Large", 67),
    1.0: ("GPT-J 6B", 73),
    1.5: ("GPT-3.5-Turbo", 76),
    7.5: ("Cascade (trained)", 83),
    30.0: ("GPT-4", 83),
}


def load_headlines_data():
    """Load HEADLINES train/test data with ground truth labels."""
    
    # Load training examples (for few-shot prompt)
    train_path = DATA_DIR / "HEADLINES_train.csv"
    train_examples = []
    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                question = row[0].strip()
                label = row[1].strip().lower()
                if label in VALID_LABELS:
                    train_examples.append((question, label))
    
    # Load test examples (for evaluation)
    test_path = DATA_DIR / "HEADLINES_test.csv"
    test_examples = []
    with open(test_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                question = row[0].strip()
                label = row[1].strip().lower()
                if label in VALID_LABELS:
                    test_examples.append((question, label))
    
    return train_examples, test_examples


def create_fewshot_prompt(question, train_examples, num_examples=5):
    """Create few-shot prompt matching FrugalGPT format."""
    
    # Select diverse few-shot examples (one per label if possible)
    examples_by_label = {label: [] for label in VALID_LABELS}
    for q, l in train_examples:
        examples_by_label[l].append((q, l))
    
    selected = []
    for label in VALID_LABELS:
        if examples_by_label[label]:
            selected.append(random.choice(examples_by_label[label]))
    
    # Add one more random example
    if len(selected) < num_examples:
        remaining = [e for e in train_examples if e not in selected]
        selected.extend(random.sample(remaining, min(num_examples - len(selected), len(remaining))))
    
    # Build prompt
    prompt = "Please determine the price direction (up, down, neutral, or none) in the following news headlines.\n\n"
    
    for q, a in selected[:num_examples]:
        prompt += f"{q} {a}\n\n"
    
    # Add test question
    prompt += f"{question}"
    
    return prompt


def get_openrouter_id(model_name):
    """Map model name to OpenRouter ID."""
    cache_path = PAPER_DIR.parent / "data" / "models_cache.json"
    
    try:
        with open(cache_path) as f:
            cache_data = json.load(f)
            models = cache_data.get('models', cache_data.get('data', []))
            
        for m in models:
            if m.get('name') == model_name:
                return m.get('openrouter_id') or m.get('id')
    except:
        pass
    
    # Fallback mappings
    name_lower = model_name.lower()
    if 'gpt-5.1' in name_lower:
        return 'openai/gpt-5.1'
    elif 'gpt-5 mini' in name_lower:
        return 'openai/gpt-5-mini'
    elif 'gpt-oss' in name_lower:
        return 'meta-llama/llama-4-maverick:free'
    
    return None


def call_api(model_id, prompt, api_key, timeout=60):
    """Call OpenRouter API."""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        return None


def extract_label(response):
    """Extract label from model response."""
    if not response:
        return None
    
    response = response.lower().strip()
    
    # Check for exact match first
    for label in VALID_LABELS:
        if response == label:
            return label
    
    # Check if label is in first word
    first_word = response.split()[0] if response else ""
    for label in VALID_LABELS:
        if label in first_word:
            return label
    
    # Check anywhere in response
    for label in VALID_LABELS:
        if label in response:
            return label
    
    return None


def evaluate_single(args):
    """Evaluate single test example."""
    model_id, prompt, ground_truth, api_key = args
    
    response = call_api(model_id, prompt, api_key)
    prediction = extract_label(response)
    
    return {
        'prediction': prediction,
        'ground_truth': ground_truth,
        'correct': prediction == ground_truth if prediction else False,
        'raw_response': response,
    }


def evaluate_model(model_name, model_id, test_examples, train_examples, api_key, 
                   num_eval=500, num_threads=20):
    """Evaluate model on test set."""
    
    print(f"\n  Evaluating {model_name} ({model_id}) on {num_eval} examples...")
    sys.stdout.flush()
    
    # Sample test examples
    eval_examples = random.sample(test_examples, min(num_eval, len(test_examples)))
    
    # Create prompts
    args_list = []
    for question, ground_truth in eval_examples:
        prompt = create_fewshot_prompt(question, train_examples)
        args_list.append((model_id, prompt, ground_truth, api_key))
    
    correct = 0
    total = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(evaluate_single, args): i for i, args in enumerate(args_list)}
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result['prediction']:
                total += 1
                if result['correct']:
                    correct += 1
            
            if completed % 50 == 0:
                acc = (correct / total * 100) if total > 0 else 0
                print(f"    Progress: {completed}/{num_eval}, Accuracy: {acc:.1f}%")
                sys.stdout.flush()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  ✓ {model_name}: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy


def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env")
        return
    
    # Set seed for reproducibility
    random.seed(42)
    
    print("=" * 70)
    print("HEADLINES Evaluation with Ground Truth Labels")
    print("=" * 70)
    
    # Load data
    print("\nLoading HEADLINES dataset...")
    train_examples, test_examples = load_headlines_data()
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")
    
    # Budget tiers
    budgets = [0.5, 1.0, 1.5, 7.5, 30.0]
    
    # Get LLM Jury selections
    print("\n" + "=" * 70)
    print("LLM Jury Model Selections (from library)")
    print("=" * 70)
    
    selections = {}
    for budget in budgets:
        results = get_best_models_for_budget(max_budget=budget, top_k=1, verbose=False)
        if results:
            model_name = results[0].model_name
            model_id = get_openrouter_id(model_name)
            selections[budget] = (model_name, model_id)
            print(f"  ${budget}/M -> {model_name}")
        else:
            selections[budget] = (None, None)
    
    # Evaluate unique models
    print("\n" + "=" * 70)
    print("Evaluating on HEADLINES Test Set (with ground truth)")
    print("=" * 70)
    
    unique_models = {}
    for budget, (name, model_id) in selections.items():
        if name and name not in unique_models:
            unique_models[name] = model_id
    
    accuracies = {}
    for model_name, model_id in unique_models.items():
        if model_id:
            acc = evaluate_model(
                model_name, model_id, 
                test_examples, train_examples, 
                api_key, num_eval=500
            )
            accuracies[model_name] = acc
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: FrugalGPT vs LLM Jury (HEADLINES Dataset)")
    print("=" * 80)
    print(f"\n{'Budget':<10} {'FrugalGPT':<20} {'Acc%':<8} {'LLM Jury':<20} {'Acc%':<8}")
    print("-" * 80)
    
    results_data = []
    for budget in budgets:
        frugal_model, frugal_acc = FRUGALGPT_RESULTS[budget]
        llm_model, _ = selections[budget]
        llm_acc = accuracies.get(llm_model, 0) if llm_model else 0
        
        print(f"${budget:<9} {frugal_model:<20} {frugal_acc:<8} {llm_model or 'N/A':<20} {llm_acc:.1f}")
        
        results_data.append({
            'budget': budget,
            'frugalgpt_model': frugal_model,
            'frugalgpt_acc': frugal_acc,
            'llm_jury_model': llm_model,
            'llm_jury_acc': round(llm_acc, 1),
        })
    
    print("-" * 80)
    
    # Save results
    output_path = FIGURES_DIR / "headlines_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

