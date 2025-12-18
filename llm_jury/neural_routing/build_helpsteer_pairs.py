#!/usr/bin/env python3
"""
Build Pairwise Training Data from HelpSteer2

HelpSteer2 contains prompts with multiple human-rated responses.
This script creates proper (prompt, response_A, response_B, winner) pairs
for Bradley-Terry training.

Each response has human ratings for:
- correctness (1-5)
- helpfulness (1-5)
- coherence (1-5)
- complexity (1-5)
- verbosity (1-5)

We create pairs where one response is clearly better than another.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


def compute_quality_score(response: Dict, weights: Dict = None) -> float:
    """
    Compute composite quality score from human ratings.
    
    Args:
        response: Dict with rating fields
        weights: Optional custom weights (default: correctness=0.5, helpfulness=0.3, coherence=0.2)
    
    Returns:
        Quality score in [0, 1] range
    """
    if weights is None:
        weights = {
            'correctness': 0.5,
            'helpfulness': 0.3,
            'coherence': 0.2,
        }
    
    score = 0.0
    for field, weight in weights.items():
        # Ratings are 1-5, normalize to 0-1
        rating = response.get(field, 3)  # Default to middle
        normalized = (rating - 1) / 4.0
        score += weight * normalized
    
    return score


def create_pairs_from_helpsteer2(
    margin: float = 0.15,
    max_pairs_per_prompt: int = 3,
    save_path: Path = None,
) -> List[Dict]:
    """
    Create pairwise training data from HelpSteer2.
    
    Args:
        margin: Minimum quality difference to create a pair (avoids noisy comparisons)
        max_pairs_per_prompt: Maximum pairs to create per prompt
        save_path: Path to save the data (optional)
    
    Returns:
        List of pair dicts with keys: prompt, response_a, response_b, label, score_a, score_b
    """
    from datasets import load_dataset
    
    print("="*60)
    print("Building Pairwise Data from HelpSteer2")
    print("="*60)
    print(f"Margin: {margin}")
    print(f"Max pairs per prompt: {max_pairs_per_prompt}")
    print()
    
    # Load dataset
    print("Loading HelpSteer2 from HuggingFace...")
    ds = load_dataset('nvidia/HelpSteer2', split='train')
    print(f"✓ Loaded {len(ds):,} samples")
    
    # Group by prompt
    print("\nGrouping responses by prompt...")
    prompt_to_responses = defaultdict(list)
    for item in tqdm(ds, desc="Grouping"):
        prompt_to_responses[item['prompt']].append({
            'response': item['response'],
            'correctness': item['correctness'],
            'helpfulness': item['helpfulness'],
            'coherence': item['coherence'],
            'complexity': item['complexity'],
            'verbosity': item['verbosity'],
        })
    
    multi_response_prompts = {p: r for p, r in prompt_to_responses.items() if len(r) >= 2}
    print(f"✓ Found {len(multi_response_prompts):,} prompts with 2+ responses")
    
    # Create pairs
    print(f"\nCreating pairs (margin={margin})...")
    pairs = []
    skipped_no_margin = 0
    
    for prompt, responses in tqdm(multi_response_prompts.items(), desc="Mining pairs"):
        # Compute quality scores
        scored_responses = []
        for r in responses:
            score = compute_quality_score(r)
            scored_responses.append((r, score))
        
        # Sort by score (descending)
        scored_responses.sort(key=lambda x: -x[1])
        
        # Create pairs with margin
        pairs_for_prompt = 0
        for i in range(len(scored_responses)):
            if pairs_for_prompt >= max_pairs_per_prompt:
                break
            
            for j in range(i + 1, len(scored_responses)):
                r_better, score_better = scored_responses[i]
                r_worse, score_worse = scored_responses[j]
                
                # Only create pair if margin is sufficient
                if score_better - score_worse >= margin:
                    pairs.append({
                        'prompt': prompt,
                        'response_a': r_better['response'],  # Better response
                        'response_b': r_worse['response'],   # Worse response
                        'label': 1.0,  # A is better than B
                        'score_a': score_better,
                        'score_b': score_worse,
                        'correctness_a': r_better['correctness'],
                        'correctness_b': r_worse['correctness'],
                        'helpfulness_a': r_better['helpfulness'],
                        'helpfulness_b': r_worse['helpfulness'],
                    })
                    pairs_for_prompt += 1
                    
                    if pairs_for_prompt >= max_pairs_per_prompt:
                        break
                else:
                    skipped_no_margin += 1
    
    print(f"\n✓ Created {len(pairs):,} pairs")
    print(f"  Skipped (insufficient margin): {skipped_no_margin:,}")
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(pairs)
    
    # Also create reversed pairs for balance (50% of the time, swap A and B)
    balanced_pairs = []
    for p in pairs:
        if np.random.random() < 0.5:
            # Keep as is: A is better
            balanced_pairs.append(p)
        else:
            # Swap: B becomes A, label becomes 0
            balanced_pairs.append({
                'prompt': p['prompt'],
                'response_a': p['response_b'],
                'response_b': p['response_a'],
                'label': 0.0,  # Now B (original A) is better
                'score_a': p['score_b'],
                'score_b': p['score_a'],
                'correctness_a': p['correctness_b'],
                'correctness_b': p['correctness_a'],
                'helpfulness_a': p['helpfulness_b'],
                'helpfulness_b': p['helpfulness_a'],
            })
    
    pairs = balanced_pairs
    np.random.shuffle(pairs)
    
    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total pairs: {len(pairs):,}")
    print(f"Unique prompts: {len(set(p['prompt'] for p in pairs)):,}")
    
    score_diffs = [abs(p['score_a'] - p['score_b']) for p in pairs]
    print(f"Score difference: mean={np.mean(score_diffs):.3f}, std={np.std(score_diffs):.3f}")
    
    labels = [p['label'] for p in pairs]
    print(f"Label balance: {np.mean(labels):.1%} A-wins, {1-np.mean(labels):.1%} B-wins")
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(pairs, f, indent=2)
        print(f"\n✓ Saved to {save_path}")
    
    return pairs


def load_helpsteer_pairs(path: Path = None) -> List[Dict]:
    """Load pre-built HelpSteer pairs."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "helpsteer2_pairs.json"
    
    if not path.exists():
        print(f"Pairs file not found at {path}")
        print("Building pairs from HelpSteer2...")
        return create_pairs_from_helpsteer2(save_path=path)
    
    with open(path) as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs):,} pairs from {path}")
    return pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build pairwise data from HelpSteer2")
    parser.add_argument("--margin", type=float, default=0.15, 
                       help="Minimum quality difference for pairs")
    parser.add_argument("--max-pairs", type=int, default=3,
                       help="Maximum pairs per prompt")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data/helpsteer2_pairs.json)")
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "data" / "helpsteer2_pairs.json"
    
    pairs = create_pairs_from_helpsteer2(
        margin=args.margin,
        max_pairs_per_prompt=args.max_pairs,
        save_path=output_path,
    )
    
    # Show samples
    print("\n" + "="*60)
    print("Sample Pairs")
    print("="*60)
    for i, p in enumerate(pairs[:3]):
        print(f"\nPair {i+1}:")
        print(f"  Prompt: {p['prompt'][:80]}...")
        print(f"  Response A (score={p['score_a']:.2f}): {p['response_a'][:80]}...")
        print(f"  Response B (score={p['score_b']:.2f}): {p['response_b'][:80]}...")
        print(f"  Label: {'A wins' if p['label'] == 1.0 else 'B wins'}")
