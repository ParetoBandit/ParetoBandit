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
    OFFLINE_DATASET_DIR,
    TRAIN_DATA_PATH_ALL_MODELS as CANONICAL_TRAIN_REWARDS,
    VAL_DATA_PATH_ALL_MODELS as CANONICAL_VAL_REWARDS,
    DEV_DATA_PATH_ALL_MODELS as CANONICAL_DEV_REWARDS,
    HOLDOUT_DATA_PATH_ALL_MODELS as CANONICAL_HOLDOUT_REWARDS,
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


def load_oracle_rewards(filename: str) -> Dict[str, Dict[str, float]]:
    """
    Load oracle rewards from a JSONL rewards file for offline replay evaluation.

    Searches ``OFFLINE_DATASET_DIR`` first, then ``DATA_DIR``.  Automatically
    detects and decompresses ``.gz`` files.

    Use the convenience wrappers :func:`load_dev_rewards` and
    :func:`load_holdout_rewards` for the canonical splits:

    - **dev** (train + val, 2,854 prompts):
      ``dev_rewards_complete_all_models.jsonl.gz``
    - **holdout** (test, 1,500 prompts):
      ``holdout_rewards_complete_all_models.jsonl.gz``

    Args:
        filename: Basename of the JSONL (or ``.jsonl.gz``) rewards file.

    Returns:
        Dict mapping prompt text → {model_id → raw_score}.  Only rows
        where ``entry["ok"] is True`` are included.

    Example:
        >>> oracle = load_dev_rewards()
        >>> reward = oracle["What is 2+2?"]["openai/gpt-4.1"]
        0.95
    """
    # Try canonical rewards directory first
    base_filepath = OFFLINE_DATASET_DIR / filename

    # Check for compressed version first (.gz)
    if not base_filepath.exists() and not filename.endswith('.gz'):
        gz_path = OFFLINE_DATASET_DIR / f"{filename}.gz"
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


def load_train_rewards() -> Dict[str, Dict[str, float]]:
    """Load prior-train set rewards (1,028 prompts) from the canonical split.

    Returns:
        Dict mapping prompt -> {model_id -> raw_score}
    """
    return load_oracle_rewards(Path(CANONICAL_TRAIN_REWARDS).name)


def load_val_rewards() -> Dict[str, Dict[str, float]]:
    """Load online-learn / validation set rewards (1,543 prompts) from the canonical split.

    Returns:
        Dict mapping prompt -> {model_id -> raw_score}
    """
    return load_oracle_rewards(Path(CANONICAL_VAL_REWARDS).name)


def load_dev_rewards() -> Dict[str, Dict[str, float]]:
    """Load combined dev rewards (train + val, 2,854 prompts).

    Use this when split membership is irrelevant (e.g. embedding
    pre-computation).  Prefer :func:`load_train_rewards` or
    :func:`load_val_rewards` when the split matters.

    Returns:
        Dict mapping prompt -> {model_id -> raw_score}
    """
    return load_oracle_rewards(Path(CANONICAL_DEV_REWARDS).name)


def load_holdout_rewards() -> Dict[str, Dict[str, float]]:
    """Load holdout (test) set rewards (1,500 prompts) from the canonical split.

    Returns:
        Dict mapping prompt -> {model_id -> raw_score}
    """
    return load_oracle_rewards(Path(CANONICAL_HOLDOUT_REWARDS).name)


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
