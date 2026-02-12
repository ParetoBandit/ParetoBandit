#!/usr/bin/env python3
"""
LLM-Based Categorization Validation

When human annotators are unavailable, use multiple LLMs as pseudo-annotators.
This provides a reasonable validation of the heuristic categorization.

Research has shown that LLM annotations can achieve high agreement with human
annotations for classification tasks (Gilardi et al., 2023).

Usage:
    python3 validate_with_llm.py --n-samples 100 --output llm_validation_results.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import categorization function
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

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


def llm_categorize_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Categorize using OpenAI API (cost-effective with gpt-4o-mini)"""
    try:
        import openai
        from openai import OpenAI
        import os
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": LLM_CATEGORIZATION_PROMPT.format(prompt=prompt[:1000])}
            ],
            temperature=0.0,
            max_tokens=20
        )
        
        category = response.choices[0].message.content.strip()
        
        # Normalize variations
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
                return value
        
        return category
        
    except Exception as e:
        print(f"   Error with OpenAI: {e}")
        return None


def llm_categorize_anthropic(prompt: str, model: str = "claude-3-haiku-20240307") -> str:
    """Categorize using Anthropic API (cost-effective with Haiku)"""
    try:
        import anthropic
        import os
        
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model=model,
            max_tokens=20,
            temperature=0.0,
            messages=[
                {"role": "user", "content": LLM_CATEGORIZATION_PROMPT.format(prompt=prompt[:1000])}
            ]
        )
        
        category = message.content[0].text.strip()
        
        # Normalize
        category_map = {
            'coding': 'Coding',
            'math': 'Math/Logic',
            'creative': 'Creative',
            'knowledge': 'Knowledge',
            'conversational': 'Conversational'
        }
        
        category_lower = category.lower()
        for key, value in category_map.items():
            if key in category_lower:
                return value
        
        return category
        
    except Exception as e:
        print(f"   Error with Anthropic: {e}")
        return None


def compute_agreement(annotations: List[List[str]]) -> Dict:
    """Compute inter-annotator agreement metrics"""
    
    # Fleiss' kappa
    n_items = len(annotations)
    n_raters = len(annotations[0]) if annotations else 0
    
    if n_items == 0 or n_raters == 0:
        return {'fleiss_kappa': None, 'pairwise_agreement': None}
    
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
        kappa = 1.0
    else:
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    
    # Pairwise agreement
    agreements = []
    for item in annotations:
        valid = [r for r in item if r]
        if len(valid) >= 2:
            # Count agreements
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    agreements.append(1 if valid[i] == valid[j] else 0)
    
    pairwise_agreement = np.mean(agreements) if agreements else None
    
    return {
        'fleiss_kappa': kappa,
        'pairwise_agreement': pairwise_agreement,
        'n_items': n_items,
        'n_raters': n_raters
    }


def validate_with_llms(samples: List[Tuple[str, str]], use_apis: List[str]) -> Dict:
    """
    Validate categorization using multiple LLMs as annotators.
    
    Args:
        samples: List of (prompt, heuristic_category) tuples
        use_apis: List of APIs to use: ['openai', 'anthropic']
    
    Returns:
        Validation results dictionary
    """
    
    print(f"\n🤖 Validating with {len(use_apis)} LLM annotator(s)...")
    print(f"   Sample size: {len(samples)}")
    print(f"   Annotators: {', '.join(use_apis)}")
    
    results = []
    annotations = []
    
    for i, (prompt, heuristic_cat) in enumerate(samples):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(samples)}")
        
        llm_cats = []
        
        # OpenAI
        if 'openai' in use_apis:
            cat = llm_categorize_openai(prompt)
            if cat:
                llm_cats.append(cat)
        
        # Anthropic
        if 'anthropic' in use_apis:
            cat = llm_categorize_anthropic(prompt)
            if cat:
                llm_cats.append(cat)
        
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
    
    # Compute metrics
    valid_results = [r for r in results if r['majority']]
    
    # Accuracy
    correct = sum(1 for r in valid_results if r['heuristic'] == r['majority'])
    accuracy = correct / len(valid_results) if valid_results else 0.0
    
    # Inter-annotator agreement
    agreement_metrics = compute_agreement(annotations)
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for r in valid_results:
        confusion[r['heuristic']][r['majority']] += 1
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(valid_results),
        'agreement_metrics': agreement_metrics,
        'confusion_matrix': dict(confusion),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate categorization using LLMs as pseudo-annotators"
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help="Number of samples to validate (default: 100)"
    )
    parser.add_argument(
        '--output', type=str, default='llm_validation_results.json',
        help="Output JSON file"
    )
    parser.add_argument(
        '--annotators', type=str, nargs='+', 
        default=['openai', 'anthropic'],
        choices=['openai', 'anthropic'],
        help="Which LLM APIs to use"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Check API keys
    import os
    if 'openai' in args.annotators and not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return
    
    if 'anthropic' in args.annotators and not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("   Set it with: export ANTHROPIC_API_KEY='your-key'")
        return
    
    # Load data
    from pathlib import Path
    DEV_PROMPTS = PROJECT_ROOT / "data" / "dev_prompts_for_rejudge.jsonl"
    HOLDOUT_PROMPTS = PROJECT_ROOT / "data" / "holdout_prompts_for_rejudge.jsonl"
    
    print("📊 Loading prompts...")
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
    results = validate_with_llms(samples, args.annotators)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\n✅ Heuristic Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
    print(f"\n📊 Inter-Annotator Agreement:")
    print(f"   Fleiss' κ: {results['agreement_metrics']['fleiss_kappa']:.3f}")
    print(f"   Pairwise agreement: {results['agreement_metrics']['pairwise_agreement']:.1%}")
    
    print(f"\n📊 Confusion Matrix:")
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
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")
    
    if results['accuracy'] >= 0.80:
        print(f"✅ EXCELLENT: Heuristic accuracy ≥80%")
        print(f"   The keyword-based categorization performs well.")
    elif results['accuracy'] >= 0.70:
        print(f"⚠️  GOOD: Heuristic accuracy 70-80%")
        print(f"   Acceptable for research purposes. Consider noting as limitation.")
    else:
        print(f"❌ POOR: Heuristic accuracy <70%")
        print(f"   Consider refining keywords or using LLM categories directly.")
    
    kappa = results['agreement_metrics']['fleiss_kappa']
    if kappa >= 0.60:
        print(f"\n✅ SUBSTANTIAL AGREEMENT: Fleiss' κ ≥0.60")
        print(f"   Multiple annotators show consistent categorization.")
    elif kappa >= 0.40:
        print(f"\n⚠️  MODERATE AGREEMENT: Fleiss' κ 0.40-0.60")
        print(f"   Categories may have some ambiguity.")
    else:
        print(f"\n❌ LOW AGREEMENT: Fleiss' κ <0.40")
        print(f"   Categories may be too subjective or unclear.")
    
    print(f"\n💡 For your paper:")
    print(f"   \"Categorization validated using {len(args.annotators)} LLM annotators")
    print(f"   (Fleiss' κ={kappa:.2f}, accuracy={results['accuracy']:.0%}).\"")


if __name__ == "__main__":
    main()
