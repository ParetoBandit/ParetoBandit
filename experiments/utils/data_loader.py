"""
Data loading utilities for BanditGPT experiments.

Provides consistent data loading across all experiments.
"""

import sys
import gzip
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Define paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.rewards import extract_reward  # noqa: E402  (after path setup)

from bandit_gpt.config import (
    PROJECT_ROOT as CONFIG_ROOT,
    BANDIT_DATA_DIR as DATA_DIR,
    DEV_DATA_PATH_3MODELS as CANONICAL_DEV_REWARDS,
    HOLDOUT_DATA_PATH_3MODELS as CANONICAL_HOLDOUT_REWARDS,
    DEFAULT_MODEL_REGISTRY_PATH
)


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
    
    return prompts


def load_oracle_rewards(filename: str = "test_rewards_hle_models.jsonl") -> Dict[str, Dict[str, float]]:
    """
    Load oracle rewards from JSONL file for offline replay evaluation.
    
    Returns nested dict: prompt_text → model_id → reward
    This enables O(1) lookup: oracle_rewards[prompt][model_id]
    
    Automatically detects and decompresses .gz files.
    
    Args:
        filename: JSONL file with model responses and rewards
    
    Returns:
        Dict mapping prompt → {model_id → raw_score}
    
    Example:
        >>> oracle = load_oracle_rewards()
        >>> reward = oracle["What is 2+2?"]["openai/gpt-4.1"]
        0.95
    """
    # Try offline_dataset subdirectory first (HLE-filtered data)
    base_filepath = DATA_DIR / "offline_dataset" / filename
    
    # Check for compressed version first (.gz)
    if not base_filepath.exists() and not filename.endswith('.gz'):
        gz_path = DATA_DIR / "offline_dataset" / f"{filename}.gz"
        if gz_path.exists():
            base_filepath = gz_path
    
    # Fallback to main data directory
    if not base_filepath.exists():
        base_filepath = DATA_DIR / filename
        if not base_filepath.exists() and not filename.endswith('.gz'):
            gz_path = DATA_DIR / f"{filename}.gz"
            if gz_path.exists():
                base_filepath = gz_path
    
    if not base_filepath.exists():
        raise FileNotFoundError(f"Oracle rewards not found at {base_filepath}")
    
    oracle_rewards: Dict[str, Dict[str, float]] = {}
    
    # Open with gzip if .gz extension, otherwise normal open
    open_fn = gzip.open if str(base_filepath).endswith('.gz') else open
    mode = 'rt' if str(base_filepath).endswith('.gz') else 'r'
    
    with open_fn(base_filepath, mode) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):  # Only include successful responses
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                reward = extract_reward(entry)
                
                if prompt not in oracle_rewards:
                    oracle_rewards[prompt] = {}
                oracle_rewards[prompt][model_id] = reward
    
    return oracle_rewards


def load_dev_rewards() -> Dict[str, Dict[str, float]]:
    """
    Load development set rewards from canonical split.
    
    This is a convenience function that loads dev_rewards_complete.jsonl.gz,
    which contains rewards for all models on development prompts.
    
    Canonical location: src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz
    
    Returns:
        Dict mapping prompt → {model_id → raw_score}
    
    Example:
        >>> dev_rewards = load_dev_rewards()
        >>> # Contains 1,121 prompts × 42 models
    """
    return load_oracle_rewards("offline_dataset/dev_rewards_complete.jsonl.gz")


def load_holdout_rewards() -> Dict[str, Dict[str, float]]:
    """
    Load holdout (test) set rewards from canonical split.
    
    This is a convenience function that loads holdout_rewards_complete.jsonl.gz,
    which contains rewards for all models on holdout prompts.
    
    Canonical location: src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz
    
    Returns:
        Dict mapping prompt → {model_id → raw_score}
    
    Example:
        >>> holdout_rewards = load_holdout_rewards()
        >>> # Contains 750 prompts × 42 models
    """
    return load_oracle_rewards("offline_dataset/holdout_rewards_complete.jsonl.gz")


def load_model_registry(path: Optional[str | Path] = None) -> Dict[str, Dict]:
    """
    Load model registry from models.json or a custom path.
    
    Returns:
        dict mapping model_id -> model config
    """
    if path:
        models_file = Path(path)
    else:
        models_file = DEFAULT_MODEL_REGISTRY_PATH
    
    if not models_file.exists():
        raise FileNotFoundError(f"models.json not found at {models_file}")
    
    with open(models_file) as f:
        data = json.load(f)
    
    # Handle nested format: {"models": [...]}
    if isinstance(data, dict) and "models" in data:
        models_list = data["models"]
    elif isinstance(data, list):
        models_list = data
    else:
        raise ValueError(f"Unexpected models.json format: {type(data)}")
    
    # Convert to dict keyed by model_id
    registry = {m["model_id"]: m for m in models_list}
    
    return registry


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
