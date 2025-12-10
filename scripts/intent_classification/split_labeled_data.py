"""
Split Labeled Data into Train/Val/Test Sets.

Creates stratified splits ensuring balanced class distribution.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


def stratified_split(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        samples: List of labeled samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test) sample lists
    """
    random.seed(seed)
    
    # Group by label
    by_label = defaultdict(list)
    for sample in samples:
        label = sample['label']
        by_label[label].append(sample)
    
    train, val, test = [], [], []
    
    # Split each class proportionally
    for label, label_samples in by_label.items():
        # Shuffle
        random.shuffle(label_samples)
        
        n = len(label_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train.extend(label_samples[:n_train])
        val.extend(label_samples[n_train:n_train + n_val])
        test.extend(label_samples[n_train + n_val:])
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Split labeled data")
    parser.add_argument(
        '--input',
        default='data/real_intent_labeled.json',
        help='Input labeled data'
    )
    parser.add_argument(
        '--output',
        default='data/real_intent_labeled_split.json',
        help='Output file with splits'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Train split ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test split ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Split Labeled Data")
    print("="*60)
    
    # Load
    print(f"\n📂 Loading from: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"  ✓ Loaded {len(samples)} labeled samples")
    
    # Split
    print(f"\n✂️  Splitting...")
    print(f"  Train: {args.train_ratio*100:.0f}%")
    print(f"  Val:   {args.val_ratio*100:.0f}%")
    print(f"  Test:  {args.test_ratio*100:.0f}%")
    
    train, val, test = stratified_split(
        samples,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    
    # Add split field
    for s in train:
        s['split'] = 'train'
    for s in val:
        s['split'] = 'val'
    for s in test:
        s['split'] = 'test'
    
    # Combine
    all_samples = train + val + test
    
    # Count by category and split
    from collections import Counter
    train_counts = Counter(s['label'] for s in train)
    val_counts = Counter(s['label'] for s in val)
    test_counts = Counter(s['label'] for s in test)
    
    # Print summary
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    
    print(f"\n{'Category':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    
    all_categories = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    
    for cat in all_categories:
        t = train_counts.get(cat, 0)
        v = val_counts.get(cat, 0)
        te = test_counts.get(cat, 0)
        total = t + v + te
        print(f"{cat:<20} {t:>8} {v:>8} {te:>8} {total:>8}")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {len(train):>8} {len(val):>8} {len(test):>8} {len(all_samples):>8}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_samples': len(all_samples),
                'splits': {
                    'train': len(train),
                    'val': len(val),
                    'test': len(test),
                },
                'by_category': {
                    'train': dict(train_counts),
                    'val': dict(val_counts),
                    'test': dict(test_counts),
                },
                'seed': args.seed,
            },
            'samples': all_samples,
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("NEXT STEP")
    print("="*60)
    print("\nTrain XGBoost classifier:")
    print(f"  python scripts/train_xgboost_intent.py --dataset {args.output}")
    
    print("\n✅ Splitting complete!")


if __name__ == '__main__':
    main()

