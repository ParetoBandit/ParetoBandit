#!/usr/bin/env python3
"""
OpenRouter-Based Categorization Validation

Validate categorization using multiple LLMs via OpenRouter.
Cost-effective and supports diverse model families.

Usage:
    python3 validate_with_openrouter.py --n-samples 100
"""

import sys
import json
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def categorize_prompt(prompt: str) -> str:
    """Keyword-based categorization (from analyze_dataset_composition.py)"""
    prompt_lower = prompt.lower()
    
    # Coding indicators
    coding_keywords = [
        'code', 'function', 'class', 'debug', 'python', 'javascript', 
        'java', 'c++', 'rust', 'programming', 'algorithm', 'implement',
        'script', 'def ', 'import ', 'const ', 'var ', '```', 'compile',
        'syntax', 'error', 'bug', 'api', 'library', 'framework'
    ]
    
    # Math/Logic indicators
    math_keywords = [
        'math', 'calculus', 'integral', 'derivative', 'equation', 'theorem',
        'proof', 'algebra', 'geometry', 'statistics', 'probability',
        'solve', 'calculate', 'formula', 'logic', 'reasoning', '\\frac',
        '\\int', 'trigonometry', 'matrix', 'vector'
    ]
    
    # Creative indicators
    creative_keywords = [
        'story', 'write', 'poem', 'poetry', 'creative', 'fiction',
        'narrative', 'character', 'plot', 'dialogue', 'essay',
        'article', 'blog', 'novel', 'screenplay', 'prose'
    ]
    
    # Knowledge indicators
    knowledge_keywords = [
        'what is', 'who is', 'when did', 'where is', 'why does',
        'explain', 'describe', 'define', 'tell me about', 'history',
        'science', 'biology', 'chemistry', 'physics', 'geography',
        'economics', 'politics', 'culture'
    ]
    
    # Count matches
    coding_score = sum(1 for kw in coding_keywords if kw in prompt_lower)
    math_score = sum(1 for kw in math_keywords if kw in prompt_lower)
    creative_score = sum(1 for kw in creative_keywords if kw in prompt_lower)
    knowledge_score = sum(1 for kw in knowledge_keywords if kw in prompt_lower)
    
    # Determine category (highest score wins)
    scores = {
        'Coding': coding_score,
        'Math/Logic': math_score,
        'Creative': creative_score,
        'Knowledge': knowledge_score,
        'Conversational': 0  # Default if no strong signal
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'Conversational'
    
    return max(scores.items(), key=lambda x: x[1])[0]


LLM_CATEGORIZATION_PROMPT = """You are a research assistant helping to categorize prompts for an academic study.

Please categorize the following prompt into EXACTLY ONE of these categories:

1. **Coding**: Programming tasks, debugging, code review, software development
2. **Math/Logic**: Mathematics, logical reasoning, proofs, calculations
3. **Creative**: Writing, storytelling, poetry, creative content
4. **Knowledge**: Factual questions, explanations, definitions, "what is" queries
5. **Conversational**: General chat, advice, open-ended discussion

PROMPT TO CATEGORIZE:
\"\"\"
{prompt}
\"\"\"

Respond with ONLY the category name (Coding, Math/Logic, Creative, Knowledge, or Conversational).
Do not include any explanation or additional text."""


def llm_categorize_openrouter(prompt: str, model: str) -> Tuple[str, float]:
    """
    Categorize using OpenRouter API.
    
    Returns: (category, cost_usd)
    """
    try:
        from openai import OpenAI
        
        # Use OpenRouter with OpenAI-compatible API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": LLM_CATEGORIZATION_PROMPT.format(prompt=prompt[:1000])}
            ],
            temperature=0.0,
            max_tokens=20,
            extra_headers={
                "HTTP-Referer": "https://github.com/banditgpt",  # Optional
                "X-Title": "BanditGPT Validation"  # Optional
            }
        )
        
        category = response.choices[0].message.content.strip()
        
        # Estimate cost (OpenRouter doesn't return cost in response)
        # Rough estimates based on typical pricing (per prompt, ~1000 tokens)
        cost_estimates = {
            'openai/gpt-4o-mini': 0.00015,  # $0.15 per 1M input tokens
            'anthropic/claude-3-haiku': 0.00025,
            'google/gemini-flash-1.5-8b': 0.000075,  # Updated model
            'google/gemini-2.0-flash-exp': 0.00000,  # Free during preview
            'meta-llama/llama-3.1-8b-instruct': 0.00006,
            'meta-llama/llama-3.3-70b-instruct': 0.00035,
        }
        estimated_cost = cost_estimates.get(model, 0.0002)
        
        # Normalize category variations
        category_map = {
            'coding': 'Coding',
            'math': 'Math/Logic',
            'math/logic': 'Math/Logic',
            'creative': 'Creative',
            'knowledge': 'Knowledge',
            'conversational': 'Conversational'
        }
        
        category_lower = category.lower()
        for key, value in category_map.items():
            if key in category_lower:
                return value, estimated_cost
        
        return category, estimated_cost
        
    except Exception as e:
        print(f"   Error with {model}: {e}")
        return None, 0.0


def compute_fleiss_kappa(annotations: List[List[str]]) -> float:
    """Compute Fleiss' kappa for inter-annotator agreement"""
    
    n_items = len(annotations)
    if n_items == 0:
        return None
    
    n_raters = len(annotations[0]) if annotations else 0
    if n_raters == 0:
        return None
    
    # Get all categories
    categories = sorted(set([r for item in annotations for r in item if r]))
    n_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Build matrix
    matrix = np.zeros((n_items, n_cats))
    for i, item_ratings in enumerate(annotations):
        for rating in item_ratings:
            if rating:
                matrix[i, cat_to_idx[rating]] += 1
    
    # Fleiss' kappa
    P_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    P_j = np.sum(matrix, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(P_j ** 2)
    
    if P_e_bar == 1:
        return 1.0
    
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def validate_with_openrouter(samples: List[Tuple[str, str]], models: List[str]) -> Dict:
    """
    Validate categorization using multiple LLMs via OpenRouter.
    
    Args:
        samples: List of (prompt, heuristic_category) tuples
        models: List of model IDs to use (e.g., 'openai/gpt-4o-mini')
    
    Returns:
        Validation results dictionary
    """
    
    print(f"\n🤖 Validating with {len(models)} LLM annotator(s) via OpenRouter...")
    print(f"   Sample size: {len(samples)}")
    print(f"   Models: {', '.join([m.split('/')[-1] for m in models])}")
    
    results = []
    annotations = []
    total_cost = 0.0
    
    for i, (prompt, heuristic_cat) in enumerate(samples):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(samples)} (cost: ${total_cost:.4f})")
        
        llm_cats = []
        
        # Query each model
        for model in models:
            cat, cost = llm_categorize_openrouter(prompt, model)
            if cat:
                llm_cats.append(cat)
                total_cost += cost
        
        # Majority vote
        if llm_cats:
            vote_counts = Counter(llm_cats)
            majority_cat = vote_counts.most_common(1)[0][0]
            agreement = vote_counts[majority_cat] / len(llm_cats)
        else:
            majority_cat = None
            agreement = 0.0
        
        results.append({
            'prompt': prompt[:200],  # Truncate for storage
            'heuristic': heuristic_cat,
            'llm_votes': llm_cats,
            'majority': majority_cat,
            'agreement': agreement
        })
        
        annotations.append(llm_cats)
    
    print(f"   ✅ Completed {len(results)} annotations")
    print(f"   💰 Total cost: ${total_cost:.4f}")
    
    # Compute metrics
    valid_results = [r for r in results if r['majority']]
    
    # Accuracy
    correct = sum(1 for r in valid_results if r['heuristic'] == r['majority'])
    accuracy = correct / len(valid_results) if valid_results else 0.0
    
    # Inter-annotator agreement
    fleiss_kappa = compute_fleiss_kappa(annotations)
    
    # Pairwise agreement
    pairwise_agreements = []
    for item in annotations:
        valid = [r for r in item if r]
        if len(valid) >= 2:
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    pairwise_agreements.append(1 if valid[i] == valid[j] else 0)
    
    pairwise_agreement = np.mean(pairwise_agreements) if pairwise_agreements else None
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid_results:
        confusion[r['heuristic']][r['majority']] += 1
    
    # Per-category precision/recall
    per_category_metrics = {}
    all_cats = sorted(set([r['heuristic'] for r in valid_results] + 
                          [r['majority'] for r in valid_results]))
    
    for cat in all_cats:
        # True positives
        tp = confusion[cat][cat]
        
        # False positives (predicted cat but actually other)
        fp = sum(confusion[other][cat] for other in all_cats if other != cat)
        
        # False negatives (actually cat but predicted other)
        fn = sum(confusion[cat][other] for other in all_cats if other != cat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_category_metrics[cat] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(valid_results),
        'fleiss_kappa': fleiss_kappa,
        'pairwise_agreement': pairwise_agreement,
        'confusion_matrix': dict(confusion),
        'per_category_metrics': per_category_metrics,
        'total_cost': total_cost,
        'results': results,
        'models_used': models
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate categorization using LLMs via OpenRouter"
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help="Number of samples to validate (default: 100)"
    )
    parser.add_argument(
        '--output', type=str, default='openrouter_validation_results.json',
        help="Output JSON file"
    )
    parser.add_argument(
        '--models', type=str, nargs='+',
        default=[
            'openai/gpt-4o-mini',
            'anthropic/claude-3-haiku',
            'google/gemini-flash-1.5-8b'
        ],
        help="OpenRouter model IDs to use"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY not set")
        print("   It should be in .env file")
        return
    
    print(f"✅ OpenRouter API key found")
    
    # Load data
    DEV_PROMPTS = PROJECT_ROOT / "data" / "dev_prompts_for_rejudge.jsonl"
    HOLDOUT_PROMPTS = PROJECT_ROOT / "data" / "holdout_prompts_for_rejudge.jsonl"
    
    print("\n📊 Loading prompts...")
    all_prompts = []
    for file_path in [DEV_PROMPTS, HOLDOUT_PROMPTS]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                if prompt:
                    all_prompts.append(prompt)
    
    print(f"   Loaded {len(all_prompts):,} total prompts")
    
    # Sample
    np.random.seed(args.seed)
    indices = np.random.choice(len(all_prompts), min(args.n_samples, len(all_prompts)), replace=False)
    samples = [(all_prompts[i], categorize_prompt(all_prompts[i])) for i in indices]
    
    # Validate
    results = validate_with_openrouter(samples, args.models)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\n✅ Heuristic Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"\n📊 Inter-Annotator Agreement:")
    print(f"   Fleiss' κ: {results['fleiss_kappa']:.3f}")
    print(f"   Pairwise agreement: {results['pairwise_agreement']:.1%}")
    print(f"\n💰 Total Cost: ${results['total_cost']:.4f}")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix (Heuristic → LLM Majority):")
    all_cats = sorted(set(list(results['confusion_matrix'].keys()) + 
                         [cat for row in results['confusion_matrix'].values() for cat in row.keys()]))
    
    print(f"\n{'Heuristic':<20}", end='')
    for cat in all_cats:
        print(f"{cat[:12]:>13}", end='')
    print()
    print("-" * (20 + 13 * len(all_cats)))
    
    for heur_cat in all_cats:
        print(f"{heur_cat:<20}", end='')
        for llm_cat in all_cats:
            count = results['confusion_matrix'].get(heur_cat, {}).get(llm_cat, 0)
            if count > 0:
                print(f"{count:>13}", end='')
            else:
                print(f"{'':>13}", end='')
        print()
    
    # Per-category metrics
    print(f"\n📊 Per-Category Metrics:")
    print(f"{'Category':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 68)
    
    for cat in all_cats:
        metrics = results['per_category_metrics'].get(cat, {})
        print(f"{cat:<20} {metrics.get('precision', 0):>11.1%} "
              f"{metrics.get('recall', 0):>11.1%} "
              f"{metrics.get('f1', 0):>11.1%} "
              f"{metrics.get('support', 0):>10}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print(f"INTERPRETATION FOR PAPER")
    print(f"{'='*60}")
    
    if results['accuracy'] >= 0.80:
        print(f"✅ EXCELLENT: Heuristic accuracy ≥80%")
        quality = "excellent"
    elif results['accuracy'] >= 0.70:
        print(f"⚠️  GOOD: Heuristic accuracy 70-80%")
        quality = "good"
    else:
        print(f"❌ MODERATE: Heuristic accuracy <70%")
        quality = "moderate"
    
    kappa = results['fleiss_kappa']
    if kappa >= 0.60:
        print(f"✅ SUBSTANTIAL AGREEMENT: Fleiss' κ ≥0.60")
        agreement = "substantial"
    elif kappa >= 0.40:
        print(f"⚠️  MODERATE AGREEMENT: Fleiss' κ 0.40-0.60")
        agreement = "moderate"
    else:
        print(f"❌ FAIR AGREEMENT: Fleiss' κ <0.40")
        agreement = "fair"
    
    # LaTeX snippet for paper
    print(f"\n📝 For your table notes:")
    print(f"\n\\item \\textbf{{Categorization Validation:}} Heuristic validated using")
    print(f"{len(args.models)} LLM annotators via OpenRouter ({', '.join([m.split('/')[-1] for m in args.models])};")
    print(f"Fleiss' κ={kappa:.2f}, accuracy={results['accuracy']:.0%}, n={results['total']}).")
    print(f"LLM validation has been shown to correlate highly with human judgments")
    print(f"for text classification tasks [cite Gilardi 2023].")
    
    print(f"\n💡 Quality Assessment:")
    print(f"   Heuristic quality: {quality}")
    print(f"   Inter-rater agreement: {agreement}")
    print(f"   Cost: ${results['total_cost']:.4f}")
    print(f"   Time: ~{len(samples) * 2 / 60:.1f} minutes")
    
    if results['accuracy'] >= 0.75 and kappa >= 0.55:
        print(f"\n✅ RECOMMENDATION: These results are strong enough for paper acceptance.")
        print(f"   Your categorization heuristic is validated and reliable.")
    else:
        print(f"\n⚠️  RECOMMENDATION: Results are moderate. Consider:")
        print(f"   1. Refining keyword heuristics based on confusion matrix")
        print(f"   2. Using LLM categories directly (if accuracy is low)")
        print(f"   3. Acknowledging as limitation in paper")


if __name__ == "__main__":
    main()
