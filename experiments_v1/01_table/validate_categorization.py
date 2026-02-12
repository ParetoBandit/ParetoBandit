#!/usr/bin/env python3
"""
Categorization Validation Helper

This script helps validate the keyword-based categorization heuristic
by sampling prompts for human annotation and computing inter-rater reliability.

Usage:
    # Generate sample for annotation
    python3 validate_categorization.py --generate --n-samples 100 --output validation_samples.csv
    
    # After annotation, compute agreement
    python3 validate_categorization.py --compute --annotated validation_samples_annotated.csv

Expected CSV format for annotated file:
    prompt,predicted_category,annotator1,annotator2,annotator3
"""

import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import categorization function by loading it from the same directory
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def categorize_prompt(prompt: str) -> str:
    """
    Categorize a prompt into semantic categories.
    (Copied from analyze_dataset_composition.py for independence)
    """
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
    
    # Return category with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

# Data paths
DEV_PROMPTS = PROJECT_ROOT / "data" / "dev_prompts_for_rejudge.jsonl"
HOLDOUT_PROMPTS = PROJECT_ROOT / "data" / "holdout_prompts_for_rejudge.jsonl"


def load_prompts(file_path: Path) -> list:
    """Load prompts from JSONL file."""
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                if prompt:
                    prompts.append(prompt)
            except:
                continue
    return prompts


def generate_validation_sample(n_samples: int, output_file: Path, seed: int = 42):
    """Generate stratified sample for human annotation."""
    np.random.seed(seed)
    
    # Load all prompts
    dev_prompts = load_prompts(DEV_PROMPTS)
    holdout_prompts = load_prompts(HOLDOUT_PROMPTS)
    all_prompts = dev_prompts + holdout_prompts
    
    print(f"📊 Loaded {len(all_prompts):,} total prompts")
    
    # Categorize and group
    categorized = defaultdict(list)
    for prompt in all_prompts:
        category = categorize_prompt(prompt)
        categorized[category].append(prompt)
    
    print(f"\nCategory distribution:")
    for cat, prompts in sorted(categorized.items()):
        print(f"  {cat:20s}: {len(prompts):5,} prompts")
    
    # Stratified sampling - proportional to category size
    samples = []
    for category, prompts in categorized.items():
        # Sample proportionally
        n_cat = max(5, int(n_samples * len(prompts) / len(all_prompts)))
        n_cat = min(n_cat, len(prompts))
        
        indices = np.random.choice(len(prompts), n_cat, replace=False)
        cat_samples = [prompts[i] for i in indices]
        
        for prompt in cat_samples:
            samples.append({
                'prompt': prompt,
                'predicted_category': category,
                'annotator1': '',
                'annotator2': '',
                'annotator3': ''
            })
        
        print(f"  Sampled {len(cat_samples)} from {category}")
    
    # Shuffle
    np.random.shuffle(samples)
    samples = samples[:n_samples]
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'prompt', 'predicted_category', 'annotator1', 'annotator2', 'annotator3'
        ])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"\n✅ Generated {len(samples)} samples → {output_file}")
    print(f"\n📝 Instructions for annotators:")
    print(f"   1. Open {output_file} in Excel or Google Sheets")
    print(f"   2. Label each prompt in columns annotator1/2/3:")
    print(f"      - Coding: Programming, debugging, code review")
    print(f"      - Math/Logic: Mathematics, reasoning, proofs")
    print(f"      - Creative: Writing, storytelling, poetry")
    print(f"      - Knowledge: Factual questions, explanations")
    print(f"      - Conversational: Chat, advice, general queries")
    print(f"   3. Save as {output_file.stem}_annotated.csv")
    print(f"   4. Run: python3 validate_categorization.py --compute --annotated {output_file.stem}_annotated.csv")


def fleiss_kappa(ratings):
    """
    Compute Fleiss' kappa for inter-rater reliability.
    
    ratings: list of lists, where each sublist contains category ratings
             from different annotators for the same item
    """
    n_items = len(ratings)
    n_raters = len(ratings[0])
    
    # Get all categories
    categories = sorted(set([r for item in ratings for r in item]))
    n_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Build matrix: n_items x n_categories
    matrix = np.zeros((n_items, n_cats))
    for i, item_ratings in enumerate(ratings):
        for rating in item_ratings:
            matrix[i, cat_to_idx[rating]] += 1
    
    # Compute P_i (proportion of agreement for each item)
    P_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    
    # Compute P_j (proportion of ratings in each category)
    P_j = np.sum(matrix, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(P_j ** 2)
    
    # Fleiss' kappa
    if P_e_bar == 1:
        return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    
    return kappa


def compute_agreement(annotated_file: Path):
    """Compute inter-rater reliability and accuracy."""
    print(f"📊 Analyzing annotations from {annotated_file}")
    
    # Load annotated data
    with open(annotated_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    print(f"   Loaded {len(data)} annotated samples")
    
    # Filter out incomplete annotations
    complete = []
    for row in data:
        if row['annotator1'] and row['annotator2']:  # At least 2 annotators
            complete.append(row)
    
    print(f"   {len(complete)} samples have ≥2 annotations")
    
    # Compute inter-rater reliability
    print(f"\n{'='*60}")
    print(f"INTER-RATER RELIABILITY")
    print(f"{'='*60}")
    
    # Build ratings matrix
    ratings_2 = []  # For 2 annotators
    ratings_3 = []  # For 3 annotators
    
    for row in complete:
        ann1 = row['annotator1'].strip()
        ann2 = row['annotator2'].strip()
        ann3 = row.get('annotator3', '').strip()
        
        if ann1 and ann2:
            ratings_2.append([ann1, ann2])
            if ann3:
                ratings_3.append([ann1, ann2, ann3])
    
    # Fleiss' kappa
    if len(ratings_2) >= 5:
        kappa_2 = fleiss_kappa(ratings_2)
        print(f"\n📊 Fleiss' Kappa (2 annotators, n={len(ratings_2)}): {kappa_2:.3f}")
        
        if kappa_2 < 0:
            interpretation = "Poor agreement (worse than chance)"
        elif kappa_2 < 0.2:
            interpretation = "Slight agreement"
        elif kappa_2 < 0.4:
            interpretation = "Fair agreement"
        elif kappa_2 < 0.6:
            interpretation = "Moderate agreement"
        elif kappa_2 < 0.8:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        
        print(f"   Interpretation: {interpretation}")
    
    if len(ratings_3) >= 5:
        kappa_3 = fleiss_kappa(ratings_3)
        print(f"\n📊 Fleiss' Kappa (3 annotators, n={len(ratings_3)}): {kappa_3:.3f}")
    
    # Compute accuracy vs. heuristic predictions
    print(f"\n{'='*60}")
    print(f"HEURISTIC ACCURACY")
    print(f"{'='*60}")
    
    # Use majority vote as ground truth
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    
    for row in complete:
        ann1 = row['annotator1'].strip()
        ann2 = row['annotator2'].strip()
        ann3 = row.get('annotator3', '').strip()
        
        votes = [a for a in [ann1, ann2, ann3] if a]
        if not votes:
            continue
        
        # Majority vote
        vote_counts = Counter(votes)
        majority, count = vote_counts.most_common(1)[0]
        
        # Only use if clear majority (>50%)
        if count / len(votes) <= 0.5:
            continue
        
        predicted = row['predicted_category'].strip()
        
        confusion[predicted][majority] += 1
        if predicted == majority:
            correct += 1
    
    total = sum(sum(row.values()) for row in confusion.values())
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ Heuristic Accuracy: {accuracy:.1%} ({correct}/{total})")
    
    # Confusion matrix
    print(f"\n📊 Confusion Matrix:")
    print(f"(Rows = Predicted, Columns = Human Label)")
    
    all_cats = sorted(set(list(confusion.keys()) + 
                         [cat for row in confusion.values() for cat in row.keys()]))
    
    # Header
    print(f"\n{'Predicted':<20}", end='')
    for cat in all_cats:
        print(f"{cat[:12]:>13}", end='')
    print()
    print("-" * (20 + 13 * len(all_cats)))
    
    # Rows
    for pred_cat in all_cats:
        print(f"{pred_cat:<20}", end='')
        for true_cat in all_cats:
            count = confusion[pred_cat][true_cat]
            if count > 0:
                print(f"{count:>13}", end='')
            else:
                print(f"{'':>13}", end='')
        print()
    
    # Per-category accuracy
    print(f"\n📊 Per-Category Metrics:")
    print(f"{'Category':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 58)
    
    for cat in all_cats:
        # Precision: Of all predicted as cat, how many were actually cat?
        pred_as_cat = sum(confusion[cat].values())
        true_pos = confusion[cat][cat]
        precision = true_pos / pred_as_cat if pred_as_cat > 0 else 0
        
        # Recall: Of all true cat, how many were predicted as cat?
        true_cat = sum(confusion[pred][cat] for pred in all_cats)
        recall = true_pos / true_cat if true_cat > 0 else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{cat:<20} {precision:>11.1%} {recall:>11.1%} {f1:>11.1%}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if accuracy >= 0.8:
        print(f"✅ Heuristic performs well (≥80% accuracy)")
        print(f"   → Acceptable for research purposes")
    elif accuracy >= 0.7:
        print(f"⚠️  Heuristic is moderate (70-80% accuracy)")
        print(f"   → Consider refinements or note as limitation")
    else:
        print(f"❌ Heuristic is poor (<70% accuracy)")
        print(f"   → Major refinements needed or use manual labels")
    
    if len(ratings_2) > 0:
        kappa = fleiss_kappa(ratings_2)
        if kappa < 0.4:
            print(f"\n⚠️  Low inter-rater agreement (κ < 0.4)")
            print(f"   → Provide clearer annotation guidelines")
            print(f"   → Consider merging ambiguous categories")


def main():
    parser = argparse.ArgumentParser(
        description="Validate categorization heuristic with human annotations"
    )
    parser.add_argument(
        '--generate', action='store_true',
        help="Generate sample for annotation"
    )
    parser.add_argument(
        '--compute', action='store_true',
        help="Compute agreement from annotated file"
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        '--output', type=str, default='validation_samples.csv',
        help="Output CSV file for samples"
    )
    parser.add_argument(
        '--annotated', type=str,
        help="Path to annotated CSV file"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    if args.generate:
        output_path = Path(args.output)
        generate_validation_sample(args.n_samples, output_path, args.seed)
    
    elif args.compute:
        if not args.annotated:
            print("❌ Error: --annotated required for --compute")
            return
        annotated_path = Path(args.annotated)
        if not annotated_path.exists():
            print(f"❌ Error: {annotated_path} not found")
            return
        compute_agreement(annotated_path)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
