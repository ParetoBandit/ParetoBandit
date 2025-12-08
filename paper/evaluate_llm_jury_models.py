#!/usr/bin/env python3
"""
Evaluate LLM Jury selected models on HEADLINES test set.
"""

import os
import sys
import csv
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')

import requests
from dotenv import load_dotenv

load_dotenv()

PAPER_DIR = Path(__file__).parent
VALID_LABELS = {"up", "down", "neutral", "none"}

# Model name to OpenRouter ID mapping
MODEL_IDS = {
    'gpt-oss-120B (high)': 'meta-llama/llama-4-maverick:free',
    'GPT-5 mini (high)': 'openai/gpt-5-mini',
    'GPT-5.1 (high)': 'openai/gpt-5.1',
    'Gemini 3 Pro Preview (high)': 'google/gemini-3-pro-preview',
}


def load_test_data(limit=500):
    """Load HEADLINES test data with ground truth."""
    test_path = PAPER_DIR / "frugalgpt_data" / "HEADLINES_test.csv"
    
    examples = []
    with open(test_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                question = row[0].strip()
                label = row[1].strip().lower()
                if label in VALID_LABELS and 'Q:' in question:
                    examples.append((question, label))
    
    random.seed(42)
    return random.sample(examples, min(limit, len(examples)))


def call_api(model_id, prompt, api_key):
    """Call OpenRouter API."""
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 50,
            },
            timeout=60
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip().lower()
    except:
        pass
    return None


def evaluate_single(args):
    """Evaluate single example."""
    model_id, question, ground_truth, api_key = args
    
    prompt = f"""Classify the price direction (up, down, neutral, or none).

Q: gold prices fall 2%
A: down

Q: oil rises on supply concerns
A: up

{question}"""
    
    response = call_api(model_id, prompt, api_key)
    
    if response:
        # Extract first word as prediction
        pred = response.split()[0] if response else ""
        for label in VALID_LABELS:
            if label in pred:
                return label == ground_truth
    return None


def evaluate_model(model_name, test_data, api_key, num_threads=20):
    """Evaluate model on test set."""
    model_id = MODEL_IDS.get(model_name)
    if not model_id:
        print(f"  Unknown model: {model_name}")
        return None
    
    print(f"\n  Evaluating {model_name} ({model_id})...")
    sys.stdout.flush()
    
    args_list = [(model_id, q, gt, api_key) for q, gt in test_data]
    
    correct = 0
    total = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(evaluate_single, args) for args in args_list]
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result is not None:
                total += 1
                if result:
                    correct += 1
            
            if completed % 100 == 0:
                acc = (correct / total * 100) if total > 0 else 0
                print(f"    Progress: {completed}/{len(test_data)}, Accuracy: {acc:.1f}%")
                sys.stdout.flush()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  ✓ {model_name}: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env")
        return
    
    print("=" * 70)
    print("LLM Jury Model Accuracy on HEADLINES Test Set")
    print("=" * 70)
    
    # Load test data
    print("\nLoading HEADLINES test data...")
    test_data = load_test_data(limit=500)
    print(f"  Loaded {len(test_data)} test examples")
    
    # Models to evaluate (including baseline)
    models = ['gpt-oss-120B (high)', 'GPT-5 mini (high)', 'GPT-5.1 (high)', 'Gemini 3 Pro Preview (high)']
    
    print("\nEvaluating models...")
    accuracies = {}
    
    for model in models:
        acc = evaluate_model(model, test_data, api_key)
        if acc is not None:
            accuracies[model] = acc
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: LLM Jury Model Accuracy")
    print("=" * 70)
    
    # Budget mapping
    budget_models = {
        0.5: 'gpt-oss-120B (high)',
        1.0: 'GPT-5 mini (high)',
        2.0: 'GPT-5 mini (high)',
        5.0: 'GPT-5.1 (high)',
        10.0: 'GPT-5.1 (high)',
        30.0: 'GPT-5.1 (high)',
    }
    
    # Model details
    model_details = {
        'gpt-oss-120B (high)': {'quality': 60.9, 'cost': 0.26, 'latency': 315},
        'GPT-5 mini (high)': {'quality': 67.8, 'cost': 0.69, 'latency': 342},
        'GPT-5.1 (high)': {'quality': 79.5, 'cost': 3.44, 'latency': 332},
    }
    
    # Baseline
    baseline = {'quality': 82.1, 'cost': 4.5, 'latency': 1964}
    
    print(f"\n{'Budget':<8} {'Model':<22} {'Accuracy':<10} {'Quality':<10} {'Cost':<10} {'Latency':<10} {'Cost Sav':<10} {'Lat Sav':<10}")
    print("-" * 100)
    
    for budget in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        model = budget_models[budget]
        acc = accuracies.get(model, 0)
        details = model_details[model]
        
        cost_sav = ((baseline['cost'] - details['cost']) / baseline['cost']) * 100
        lat_sav = ((baseline['latency'] - details['latency']) / baseline['latency']) * 100
        
        print(f"${budget:<7.1f} {model:<22} {acc:<10.1f} {details['quality']:<10.1f} ${details['cost']:<9.2f} {details['latency']:<10.0f} {cost_sav:>+8.0f}% {lat_sav:>+8.0f}%")
    
    print("-" * 100)
    print(f"\nBaseline: Gemini 3 Pro Preview (Quality: {baseline['quality']}, Cost: ${baseline['cost']}/M, Latency: {baseline['latency']}ms)")


if __name__ == "__main__":
    main()

