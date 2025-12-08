#!/usr/bin/env python3
"""
Compare FrugalGPT vs LLM Jury model selections at equivalent budget tiers.

Uses the LLM Jury library's optimization algorithm for model selection,
then evaluates actual accuracy on FrugalGPT's HEADLINES dataset.
"""

import os
import sys
import json
import sqlite3
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury import get_best_models_for_budget

# Paths
PAPER_DIR = Path(__file__).parent
DATA_DIR = PAPER_DIR / "frugalgpt_data"
FIGURES_DIR = PAPER_DIR / "figures"

# FrugalGPT benchmark results (from their paper Table 2)
FRUGALGPT_RESULTS = {
    0.5: ("J1-Large", 67),
    1.0: ("GPT-J 6B", 73),
    1.5: ("GPT-3.5-Turbo", 76),
    7.5: ("Cascade (trained)", 83),
    30.0: ("GPT-4", 83),
}

VALID_LABELS = {"up", "down", "neutral", "none"}


def load_headlines_queries(limit=500):
    """Load HEADLINES queries with GPT-4 ground truth."""
    db_path = DATA_DIR / "HEADLINES.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM unnamed")
    
    gpt4_predictions = {}
    all_queries = {}
    
    for key_str, value_blob in cursor.fetchall():
        try:
            key_data = eval(key_str)
            value_data = pickle.loads(value_blob)
            
            service_id = str(key_data.get('service_id', ''))
            query_text = key_data.get('query', '')
            
            if 'price direction' not in query_text:
                continue
            
            headline = query_text.split('Q:')[-1].split('A:')[0].strip()
            completion = value_data.get('completion', '').strip().lower()
            
            # Store GPT-4 predictions as ground truth
            if service_id == '60001':
                if completion in VALID_LABELS:
                    gpt4_predictions[headline] = completion
            
            all_queries[headline] = query_text
            
        except:
            continue
    
    conn.close()
    
    # Build queries with ground truth
    queries = []
    for headline, query in all_queries.items():
        if headline in gpt4_predictions:
            queries.append({
                'prompt': query,
                'headline': headline,
                'ground_truth': gpt4_predictions[headline],
            })
    
    return queries[:limit]


def get_model_openrouter_id(model_name):
    """Get OpenRouter ID for a model name from the cache."""
    cache_path = PAPER_DIR.parent / "data" / "models_cache.json"
    
    try:
        with open(cache_path) as f:
            cache_data = json.load(f)
            models_data = cache_data.get('models', cache_data.get('data', []))
    except:
        models_data = []
    
    for m in models_data:
        if m.get('name', '') == model_name or m.get('model_name', '') == model_name:
            return m.get('openrouter_id') or m.get('id')
    
    # Fallback mappings for common names
    name_lower = model_name.lower()
    if 'gpt-5.1' in name_lower:
        return 'openai/gpt-5.1'
    elif 'gpt-5 mini' in name_lower or 'gpt-5-mini' in name_lower:
        return 'openai/gpt-5-mini'
    elif 'gpt-oss' in name_lower or 'maverick' in name_lower:
        return 'meta-llama/llama-4-maverick:free'
    
    return None


def call_openrouter(model_id, prompt, api_key, timeout=60):
    """Call OpenRouter API."""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 20,
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None


def evaluate_single(args):
    """Evaluate single query."""
    model_id, query, api_key = args
    response = call_openrouter(model_id, query['prompt'], api_key)
    
    if response:
        resp_lower = response.lower().split()[0] if response else ""
        for label in VALID_LABELS:
            if label in resp_lower:
                resp_lower = label
                break
        return resp_lower == query['ground_truth']
    return None


def evaluate_model(model_name, model_id, queries, api_key, num_threads=20):
    """Evaluate model accuracy using parallel threads."""
    print(f"  Evaluating {model_name} ({model_id}) on {len(queries)} queries...")
    sys.stdout.flush()
    
    args_list = [(model_id, q, api_key) for q in queries]
    
    correct = 0
    total = 0
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(evaluate_single, args): i for i, args in enumerate(args_list)}
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            if result is not None:
                total += 1
                if result:
                    correct += 1
            
            if completed % 50 == 0:
                acc = (correct / total * 100) if total > 0 else 0
                print(f"    Progress: {completed}/{len(queries)}, Accuracy: {acc:.1f}%")
                sys.stdout.flush()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  Done: {accuracy:.1f}% ({correct}/{total})")
    return accuracy


def main():
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env")
        return
    
    print("=" * 70)
    print("FrugalGPT vs LLM Jury Comparison")
    print("=" * 70)
    
    # Load queries
    print("\nLoading HEADLINES queries...")
    queries = load_headlines_queries(limit=500)
    print(f"  Loaded {len(queries)} queries with GPT-4 ground truth")
    
    # Budget tiers to compare
    budgets = [0.5, 1.0, 1.5, 7.5, 30.0]
    
    # Get LLM Jury selections using the library
    print("\n" + "=" * 70)
    print("Getting LLM Jury Selections (from library)")
    print("=" * 70)
    
    llm_jury_selections = {}
    for budget in budgets:
        print(f"\nBudget ${budget}/M:")
        results = get_best_models_for_budget(max_budget=budget, top_k=1, verbose=False)
        
        if results:
            model_name = results[0].model_name
            model_id = get_model_openrouter_id(model_name)
            llm_jury_selections[budget] = (model_name, model_id)
            print(f"  -> {model_name} ({model_id})")
        else:
            llm_jury_selections[budget] = (None, None)
            print(f"  -> No model found")
    
    # Evaluate unique models
    print("\n" + "=" * 70)
    print("Evaluating LLM Jury Models on HEADLINES")
    print("=" * 70)
    
    unique_models = {}
    for budget, (model_name, model_id) in llm_jury_selections.items():
        if model_name and model_name not in unique_models:
            unique_models[model_name] = model_id
    
    model_accuracies = {}
    for model_name, model_id in unique_models.items():
        if model_id:
            acc = evaluate_model(model_name, model_id, queries, api_key)
            model_accuracies[model_name] = acc
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: FrugalGPT vs LLM Jury")
    print("=" * 80)
    print(f"\n{'Budget':<10} {'FrugalGPT Selection':<20} {'Acc':<6} {'LLM Jury Selection':<20} {'Acc':<6}")
    print("-" * 80)
    
    results_data = []
    for budget in budgets:
        frugal_model, frugal_acc = FRUGALGPT_RESULTS[budget]
        llm_model, _ = llm_jury_selections[budget]
        llm_acc = model_accuracies.get(llm_model, 0) if llm_model else 0
        
        print(f"${budget:<9} {frugal_model:<20} {frugal_acc:<6} {llm_model or 'N/A':<20} {llm_acc:.1f}")
        
        results_data.append({
            'budget': budget,
            'frugalgpt_model': frugal_model,
            'frugalgpt_acc': frugal_acc,
            'llm_jury_model': llm_model,
            'llm_jury_acc': llm_acc,
        })
    
    print("-" * 80)
    
    # Save results
    output_path = FIGURES_DIR / "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Print key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\nFrugalGPT advantages:")
    print("  - Higher accuracy via cascade routing (trains on labeled data)")
    print("  - Query-specific model selection")
    print("\nLLM Jury advantages:")
    print("  - Zero labeled data required")
    print("  - Zero training required")
    print("  - Single model call (no cascade overhead)")
    print("  - Interpretable: models selected via business targets")


if __name__ == "__main__":
    main()

