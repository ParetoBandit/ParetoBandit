"""
Data loading utilities for BanditGPT experiments.

Provides consistent data loading across all experiments.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Define paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "bandit_gpt" / "data"


def load_test_prompts(filename: str = "test_prompts.jsonl") -> List[Dict]:
    """
    Load test prompts from LMSYS Arena dataset.
    
    Args:
        filename: name of the test prompts file
    
    Returns:
        list of prompt dicts with keys: ['prompt', 'task_type', 'complexity']
    
    Example:
        >>> prompts = load_test_prompts()
        >>> print(prompts[0]['prompt'])
        'Write a Python function to sort a list'
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Test prompts not found at {filepath}")
    
    prompts = []
    with open(filepath) as f:
        for line in f:
            prompts.append(json.loads(line))
    
    print(f"✓ Loaded {len(prompts)} test prompts")
    return prompts


def load_train_prompts(filename: str = "train_prompts.jsonl") -> List[Dict]:
    """
    Load training prompts for procedural warmup.
    
    Args:
        filename: name of the training prompts file
    
    Returns:
        list of prompt dicts
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Train prompts not found at {filepath}")
    
    prompts = []
    with open(filepath) as f:
        for line in f:
            prompts.append(json.loads(line))
    
    print(f"✓ Loaded {len(prompts)} train prompts")
    return prompts


def load_oracle_rewards(model_id: str, prompts: List[Dict]) -> np.ndarray:
    """
    [PLACEHOLDER] Load oracle rewards for a specific model.
    
    Args:
        model_id: model identifier (e.g., "gpt-4")
        prompts: list of prompts
    
    Returns:
        oracle rewards (shape: [n_prompts])
    """
    # TODO: Implement actual oracle reward loading
    # For now, return dummy data
    n = len(prompts)
    return np.random.uniform(0.7, 1.0, size=n)


def load_model_registry() -> Dict[str, Dict]:
    """
    Load model registry from models.json.
    
    Returns:
        dict mapping model_id -> {bias, weights}
    """
    models_file = PROJECT_ROOT / "src" / "bandit_gpt" / "config" / "models.json"
    
    if not models_file.exists():
        raise FileNotFoundError(f"models.json not found at {models_file}")
    
    with open(models_file) as f:
        models = json.load(f)
    
    print(f"✓ Loaded {len(models)} models from registry")
    return models


def create_train_test_split(
    prompts: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split prompts into train and test sets.
    
    Args:
        prompts: all prompts
        train_ratio: fraction for training
        seed: random seed for reproducibility
    
    Returns:
        (train_prompts, test_prompts)
    """
    rng = np.random.RandomState(seed)
    n = len(prompts)
    indices = rng.permutation(n)
    
    n_train = int(n * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train = [prompts[i] for i in train_indices]
    test = [prompts[i] for i in test_indices]
    
    print(f"✓ Split: {len(train)} train, {len(test)} test")
    return train, test


def filter_prompts_by_complexity(
    prompts: List[Dict],
    min_complexity: Optional[float] = None,
    max_complexity: Optional[float] = None
) -> List[Dict]:
    """
    Filter prompts by complexity range.
    
    Args:
        prompts: list of prompts
        min_complexity: minimum complexity (None = no limit)
        max_complexity: maximum complexity (None = no limit)
    
    Returns:
        filtered prompts
    """
    filtered = []
    for prompt in prompts:
        complexity = prompt.get("complexity", 0.5)
        if min_complexity is not None and complexity < min_complexity:
            continue
        if max_complexity is not None and complexity > max_complexity:
            continue
        filtered.append(prompt)
    
    print(f"✓ Filtered: {len(filtered)}/{len(prompts)} prompts")
    return filtered


if __name__ == "__main__":
    # Test data loader
    try:
        prompts = load_test_prompts()
        print(f"Sample prompt: {prompts[0]['prompt'][:50]}...")
        
        train, test = create_train_test_split(prompts)
        print(f"Train/test split: {len(train)}/{len(test)}")
        
        models = load_model_registry()
        print(f"Models: {list(models.keys())[:5]}...")
        
        print("✓ Data loader working correctly!")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
        print("   (This is expected if running from standalone test)")
