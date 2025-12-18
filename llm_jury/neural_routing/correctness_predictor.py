#!/usr/bin/env python3
"""
Correctness Predictor v2 - Fixed "short=bad" bias

Key changes from v1:
1. Cross-encoder (not bi-encoder with mean pooling)
2. Ordinal classification (0<1<2<3<4) instead of binary
3. Length-balanced batch sampling
4. Length-bucketed evaluation metrics
5. Lower LR with warmup + gradient clipping

Usage:
    python -m llm_jury.neural_routing.correctness_predictor --epochs 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CorrectnessConfig:
    """Configuration for Correctness Predictor."""
    # Cross-encoder backbone - use a model trained for query-response ranking
    # This model already knows how to compare query tokens to answer tokens
    backbone: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Training
    batch_size: int = 32
    learning_rate: float = 5e-5  # Lower for fine-tuning
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_epochs: int = 3
    max_length: int = 256
    
    # Ordinal classification: predict P(score > k) for k in {0,1,2,3}
    num_thresholds: int = 4  # 4 thresholds for 5 classes (0-4)
    
    # Length buckets for balanced sampling
    length_buckets: List[int] = field(default_factory=lambda: [3, 10, 30])  # -> [1-3, 4-10, 11-30, 31+]
    
    # Validation
    val_split: float = 0.1
    
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "correctness_predictor"
    )


def get_device() -> torch.device:
    """Get optimal device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Dataset with Length Bucketing
# =============================================================================

def get_length_bucket(word_count: int, buckets: List[int]) -> int:
    """Assign word count to a bucket. Returns bucket index."""
    for i, threshold in enumerate(buckets):
        if word_count <= threshold:
            return i
    return len(buckets)  # Last bucket (31+)


class HelpSteerDataset(Dataset):
    """
    Dataset for HelpSteer2 correctness prediction.
    
    Input: "[QUESTION] {prompt} [ANSWER] {response}"
    Target: correctness score 0-4 (for ordinal classification)
    """
    
    def __init__(self, df: pd.DataFrame, tokenizer, config: CorrectnessConfig):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        
        # Precompute length buckets for balanced sampling
        self.df['response_words'] = self.df['response'].apply(lambda x: len(str(x).split()))
        self.df['length_bucket'] = self.df['response_words'].apply(
            lambda x: get_length_bucket(x, config.length_buckets)
        )
        # Discretize correctness for sampling groups
        self.df['correctness_bin'] = self.df['correctness'].apply(
            lambda x: 0 if x <= 1 else (1 if x == 2 else 2)  # BAD, BORDERLINE, GOOD
        )
        # Group key for balanced sampling
        self.df['sample_group'] = self.df.apply(
            lambda r: f"{r['length_bucket']}_{r['correctness_bin']}", axis=1
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Cross-encoder format: concatenate question and answer
        text = f"[QUESTION] {row['prompt']} [ANSWER] {row['response']}"
        
        # Correctness target (0-4 integer)
        correctness = int(row['correctness'])
        
        # For ordinal classification: create binary targets for each threshold
        # ordinal_targets[k] = 1 if correctness > k, else 0
        ordinal_targets = torch.zeros(self.config.num_thresholds, dtype=torch.float32)
        for k in range(self.config.num_thresholds):
            if correctness > k:
                ordinal_targets[k] = 1.0
        
        enc = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'ordinal_targets': ordinal_targets,
            'correctness': torch.tensor(correctness, dtype=torch.long),
            'length_bucket': row['length_bucket'],
            'response_words': row['response_words'],
        }
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create a sampler that balances across (length_bucket, correctness_bin) groups.
        This ensures every batch has short/long × good/bad examples.
        """
        group_counts = Counter(self.df['sample_group'])
        
        # Weight each sample inversely to its group frequency
        weights = []
        for idx in range(len(self.df)):
            group = self.df.iloc[idx]['sample_group']
            weights.append(1.0 / group_counts[group])
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )


def load_helpsteer_data(config: CorrectnessConfig, seed: int = 42):
    """Load HelpSteer2 with grouped splitting (no data leakage)."""
    from datasets import load_dataset
    from sklearn.model_selection import GroupShuffleSplit
    
    print("Loading HelpSteer2...")
    dataset = load_dataset("nvidia/HelpSteer2", split="train")
    df = pd.DataFrame(dataset)
    print(f"  Raw: {len(df):,} samples")
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['prompt', 'response'])
    print(f"  After dedup: {len(df):,}")
    
    # Report correctness distribution
    print(f"\nCorrectness distribution:")
    for score in range(5):
        n = len(df[df['correctness'] == score])
        print(f"  {score}: {n:,} ({100*n/len(df):.1f}%)")
    
    # Group split by prompt (no leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df['prompt']))
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # Verify no overlap
    assert len(set(train_df['prompt']) & set(val_df['prompt'])) == 0, "Data leakage!"
    
    print(f"\nSplit (no prompt overlap):")
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    return train_df, val_df


# =============================================================================
# Cross-Encoder Model with Ordinal Classification
# =============================================================================

class CorrectnessPredictor(nn.Module):
    """
    Cross-encoder for correctness prediction using ordinal classification.
    
    Uses AutoModelForSequenceClassification with 4 outputs for ordinal thresholds:
    - logit[0]: P(correctness > 0)
    - logit[1]: P(correctness > 1)
    - logit[2]: P(correctness > 2)
    - logit[3]: P(correctness > 3)
    
    Final score = sum of sigmoid(logits), ranges 0-4.
    """
    
    def __init__(self, config: CorrectnessConfig = None):
        super().__init__()
        
        if config is None:
            config = CorrectnessConfig()
        self.config = config
        
        # Load cross-encoder - the pre-trained model has 1 output, we need num_thresholds
        # So we load the model first, then replace the classifier head
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            config.backbone,
            num_labels=config.num_thresholds,
            ignore_mismatched_sizes=True  # Allow replacing classifier head
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    
    def forward(self, input_ids, attention_mask):
        """
        Returns ordinal logits (batch_size, num_thresholds).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # (batch, num_thresholds)
    
    def predict_score(self, input_ids, attention_mask) -> torch.Tensor:
        """
        Predict correctness score 0-4 from ordinal logits.
        Score = sum of P(correctness > k) for k in 0..3
        """
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)  # (batch, 4)
        scores = probs.sum(dim=1)  # (batch,) in range [0, 4]
        return scores
    
    def predict(self, prompt: str, response: str) -> Dict[str, float]:
        """Predict for a single (prompt, response) pair."""
        device = next(self.parameters()).device
        
        text = f"[QUESTION] {prompt} [ANSWER] {response}"
        enc = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = self.forward(
                enc['input_ids'].to(device),
                enc['attention_mask'].to(device)
            )
            probs = torch.sigmoid(logits).squeeze()  # (4,)
            score = probs.sum().item()
        
        # Convert to quality assessment
        return {
            'score': score,  # 0-4 continuous
            'is_good': score >= 2.5,  # threshold for "good enough"
            'is_bad': score < 1.5,  # threshold for "definitely bad"
            'ordinal_probs': probs.tolist(),  # [P(>0), P(>1), P(>2), P(>3)]
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': {
                'backbone': self.config.backbone,
                'max_length': self.config.max_length,
                'num_thresholds': self.config.num_thresholds,
            },
            'state_dict': self.state_dict(),
        }, path)
        print(f"Saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'CorrectnessPredictor':
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = CorrectnessConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model


# =============================================================================
# Ordinal Loss Function
# =============================================================================

def ordinal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Ordinal classification loss (sum of binary cross-entropies).
    
    Args:
        logits: (batch, num_thresholds) - raw logits for each threshold
        targets: (batch, num_thresholds) - binary targets for each threshold
    
    Returns:
        Scalar loss
    """
    return F.binary_cross_entropy_with_logits(logits, targets)


# =============================================================================
# Length-Bucketed Evaluation
# =============================================================================

def evaluate_by_length(
    model: CorrectnessPredictor,
    val_loader: DataLoader,
    config: CorrectnessConfig,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance by answer length bucket.
    
    Returns metrics for each bucket: accuracy, MAE, correlation.
    """
    model.eval()
    
    # Collect predictions by bucket
    bucket_preds = {i: [] for i in range(len(config.length_buckets) + 1)}
    bucket_targets = {i: [] for i in range(len(config.length_buckets) + 1)}
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            correctness = batch['correctness']
            length_buckets = batch['length_bucket']
            
            # Get predicted scores
            scores = model.predict_score(input_ids, attention_mask).cpu()
            
            for i in range(len(scores)):
                bucket = length_buckets[i].item()
                pred = scores[i].item()
                target = correctness[i].item()
                
                bucket_preds[bucket].append(pred)
                bucket_targets[bucket].append(target)
                all_preds.append(pred)
                all_targets.append(target)
    
    # Compute metrics per bucket
    bucket_names = ['1-3 words', '4-10 words', '11-30 words', '31+ words']
    results = {}
    
    for bucket_idx, name in enumerate(bucket_names):
        preds = bucket_preds[bucket_idx]
        targets = bucket_targets[bucket_idx]
        
        if len(preds) == 0:
            results[name] = {'n': 0, 'mae': float('nan'), 'acc': float('nan')}
            continue
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        # MAE
        mae = np.mean(np.abs(preds - targets))
        
        # Accuracy (round prediction to nearest integer)
        acc = np.mean(np.round(preds) == targets)
        
        # Binary accuracy (is_good: score >= 2.5)
        binary_pred = (preds >= 2.5).astype(int)
        binary_target = (targets >= 3).astype(int)  # correctness 3 or 4 = good
        binary_acc = np.mean(binary_pred == binary_target)
        
        # Correlation
        if len(set(targets)) > 1:
            corr = np.corrcoef(preds, targets)[0, 1]
        else:
            corr = float('nan')
        
        results[name] = {
            'n': len(preds),
            'mae': mae,
            'acc': acc,
            'binary_acc': binary_acc,
            'corr': corr,
        }
    
    # Overall metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    results['OVERALL'] = {
        'n': len(all_preds),
        'mae': np.mean(np.abs(all_preds - all_targets)),
        'acc': np.mean(np.round(all_preds) == all_targets),
        'binary_acc': np.mean((all_preds >= 2.5) == (all_targets >= 3)),
        'corr': np.corrcoef(all_preds, all_targets)[0, 1] if len(set(all_targets)) > 1 else float('nan'),
    }
    
    return results


def print_evaluation_table(results: Dict[str, Dict[str, float]]):
    """Print evaluation results as a formatted table."""
    print(f"\n{'='*75}")
    print("EVALUATION BY ANSWER LENGTH")
    print(f"{'='*75}")
    print(f"{'Bucket':<15} {'N':>8} {'MAE':>8} {'Acc':>8} {'BinAcc':>8} {'Corr':>8}")
    print(f"{'-'*75}")
    
    for bucket, metrics in results.items():
        n = metrics['n']
        mae = f"{metrics['mae']:.3f}" if not np.isnan(metrics['mae']) else "N/A"
        acc = f"{metrics['acc']:.3f}" if not np.isnan(metrics['acc']) else "N/A"
        bin_acc = f"{metrics['binary_acc']:.3f}" if not np.isnan(metrics.get('binary_acc', float('nan'))) else "N/A"
        corr = f"{metrics['corr']:.3f}" if not np.isnan(metrics.get('corr', float('nan'))) else "N/A"
        
        marker = " ← SHORT" if bucket == '1-3 words' else ""
        print(f"{bucket:<15} {n:>8} {mae:>8} {acc:>8} {bin_acc:>8} {corr:>8}{marker}")
    
    print(f"{'='*75}")


# =============================================================================
# Training with Warmup + Gradient Clipping
# =============================================================================

def train_correctness_predictor(
    config: CorrectnessConfig = None,
    device: torch.device = None
) -> Tuple[CorrectnessPredictor, DataLoader]:
    """
    Train the correctness predictor with:
    - Ordinal classification loss
    - Length-balanced sampling
    - Warmup + gradient clipping
    - Length-bucketed evaluation
    """
    if config is None:
        config = CorrectnessConfig()
    if device is None:
        device = get_device()
    
    print(f"\n{'='*60}")
    print("Correctness Predictor Training (Cross-Encoder + Ordinal)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone}")
    print(f"Batch size: {config.batch_size}")
    print(f"LR: {config.learning_rate}")
    print(f"Warmup: {config.warmup_ratio*100:.0f}%")
    print(f"Epochs: {config.max_epochs}")
    print(f"{'='*60}\n")
    
    # Load data
    train_df, val_df = load_helpsteer_data(config)
    
    # Create tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    
    # Create datasets
    train_ds = HelpSteerDataset(train_df, tokenizer, config)
    val_ds = HelpSteerDataset(val_df, tokenizer, config)
    
    # Length-balanced sampler for training
    balanced_sampler = train_ds.get_balanced_sampler()
    
    # Report sampling distribution
    print("\nLength × Correctness distribution (training):")
    group_counts = Counter(train_ds.df['sample_group'])
    for group, count in sorted(group_counts.items()):
        bucket, corr_bin = group.split('_')
        bucket_name = ['1-3', '4-10', '11-30', '31+'][int(bucket)]
        corr_name = ['BAD', 'BORDERLINE', 'GOOD'][int(corr_bin)]
        print(f"  {bucket_name} × {corr_name}: {count:,}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        sampler=balanced_sampler,  # Balanced sampling!
        num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = CorrectnessPredictor(config)
    model.to(device)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    best_mae = float('inf')
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.max_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ordinal_targets = batch['ordinal_targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(input_ids, attention_mask)
            
            # Ordinal loss
            loss = ordinal_loss(logits, ordinal_targets)
            
            # Backward with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Evaluate by length bucket
        results = evaluate_by_length(model, val_loader, config, device)
        
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print_evaluation_table(results)
        
        # Save best model (by overall MAE)
        overall_mae = results['OVERALL']['mae']
        if overall_mae < best_mae:
            best_mae = overall_mae
            model.save(config.checkpoint_dir / "best_correctness_predictor.pt")
            print(f"  ✓ New best model (MAE={best_mae:.3f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val MAE: {best_mae:.3f}")
    print(f"{'='*60}")
    
    return model, val_loader


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    _defaults = CorrectnessConfig()
    
    parser = argparse.ArgumentParser(description="Train Correctness Predictor")
    parser.add_argument("--epochs", type=int, default=_defaults.max_epochs)
    parser.add_argument("--batch-size", type=int, default=_defaults.batch_size)
    parser.add_argument("--lr", type=float, default=_defaults.learning_rate)
    parser.add_argument("--backbone", type=str, default=_defaults.backbone)
    
    args = parser.parse_args()
    
    config = CorrectnessConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone=args.backbone,
    )
    
    model, val_loader = train_correctness_predictor(config)
    
    # Test with specific examples
    print("\n" + "="*60)
    print("Test Predictions")
    print("="*60 + "\n")
    
    test_cases = [
        ("What is 2+2?", "4"),  # Short correct
        ("What is 2+2?", "The answer is 4."),  # Medium correct
        ("What is 2+2?", "I think maybe fish?"),  # Short wrong
        ("Capital of France?", "Paris"),  # Short correct
        ("Capital of France?", "Lyon"),  # Short wrong
        ("Capital of France?", "The capital of France is Paris, a beautiful city on the Seine."),  # Long correct
        ("Write Python to sort a list", "sorted(lst)"),  # Short correct code
        ("Explain quantum computing", "I don't know."),  # Short wrong/unhelpful
        ("What is the speed of light?", "Approximately 299,792,458 meters per second."),  # Medium correct
    ]
    
    model.eval()
    for prompt, response in test_cases:
        r = model.predict(prompt, response)
        words = len(response.split())
        status = "✓ GOOD" if r['is_good'] else ("✗ BAD" if r['is_bad'] else "? BORDERLINE")
        print(f"Q: {prompt[:40]}")
        print(f"A: {response[:50]} ({words} words)")
        print(f"   Score={r['score']:.2f}/4 {status}")
        print()
