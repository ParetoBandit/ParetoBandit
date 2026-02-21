"""
Atomic Data Loader for Evaluation Scripts

This module provides strict prompt-reward alignment to prevent catastrophic
indexing bugs where prompts and rewards are mismatched.

Key Principle:
    Never maintain separate lists (prompts, rewards_map) and hope they align.
    Instead, iterate over atomic EvaluationItem objects that bundle
    prompt + ground-truth rewards together.

Date: 2026-01-26
"""

import json
import gzip
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationItem:
    """Atomic unit: one prompt with all its model rewards."""
    sample_id: int
    prompt: str
    rewards: Dict[str, float]  # {model_name: reward_score}
    
    def get_reward(self, model_name: str, default: float = 0.0) -> float:
        """
        Safely retrieve reward for a model.
        
        Args:
            model_name: Model identifier
            default: Fallback if model has no reward for this prompt
            
        Returns:
            Ground-truth reward for this specific prompt-model pair
        """
        return self.rewards.get(model_name, default)


class AlignedEvaluator:
    """
    Enforces 1:1 alignment between prompts and rewards.
    
    Prevents the catastrophic bug where:
        - Router sees prompt[i]
        - Gets rewarded for rewards[j % len(rewards)]
        - i != j → noise, not signal
    
    Usage:
        >>> evaluator = AlignedEvaluator.from_jsonl_gz(data_path)
        >>> for item in evaluator:
        ...     prompt = item.prompt
        ...     reward = item.get_reward(selected_model)
        ...     router.update(context, selected_model, reward)
    """
    
    def __init__(self, data: List[EvaluationItem]):
        """
        Initialize with pre-aligned data.
        
        Args:
            data: List of EvaluationItem objects
        """
        self.data = data
        self._validate()
        
    def _validate(self):
        """Sanity checks on data quality."""
        if not self.data:
            raise ValueError("AlignedEvaluator: Empty dataset!")
        
        # Check for duplicate sample_ids (would indicate alignment bug)
        sample_ids = [item.sample_id for item in self.data]
        if len(sample_ids) != len(set(sample_ids)):
            logger.warning("⚠️  Duplicate sample_ids detected - check data loading!")
        
        # Check that all items have rewards
        empty_rewards = sum(1 for item in self.data if not item.rewards)
        if empty_rewards > 0:
            logger.warning(f"⚠️  {empty_rewards}/{len(self.data)} items have no rewards")
        
        logger.info(f"✅ Loaded {len(self.data)} aligned evaluation examples")
        
        # Log available models
        all_models = set()
        for item in self.data:
            all_models.update(item.rewards.keys())
        logger.info(f"📊 Available models in dataset: {sorted(all_models)}")
    
    @classmethod
    def from_jsonl_gz(cls, path: Path, required_models: List[str] = None) -> "AlignedEvaluator":
        """
        Load data from gzipped JSONL file with strict alignment.
        
        Expected format (per line):
            {
                "prompt": str,
                "model_id": str,
                "reward_logit": float (or "reward": float)
            }
            
        Note: If "sample_id" field exists, it will be used. Otherwise,
              prompts are grouped by their text content (hash-based).
        
        Args:
            path: Path to .jsonl.gz file
            required_models: Optional list of models that must be present
            
        Returns:
            AlignedEvaluator with strictly aligned data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required_models are missing from dataset
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Step 1: Build lookup table: prompt_key -> {prompt, {model: reward}}
        # Use prompt text hash as key if sample_id not available
        samples = {}
        
        with gzip.open(path, 'rt') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    continue
                
                prompt = d.get("prompt", "")
                if not prompt:
                    logger.warning(f"Skipping line {line_num}: missing prompt")
                    continue
                
                # Use sample_id if available, otherwise hash the prompt
                sample_id = d.get("sample_id")
                if sample_id is None:
                    # Create stable hash of prompt for grouping
                    prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    sample_key = prompt_hash
                else:
                    sample_key = str(sample_id)
                
                model_id = d.get("model_id")
                reward = d.get("reward_logit", d.get("reward", 0.0))
                
                # Initialize sample if first time seeing this prompt
                if sample_key not in samples:
                    samples[sample_key] = {
                        "prompt": prompt,
                        "rewards": {},
                        "sample_id": sample_id if sample_id is not None else len(samples)
                    }
                
                # Add this model's reward
                if model_id:
                    samples[sample_key]["rewards"][model_id] = float(reward)
        
        # Step 2: Convert to sorted list of EvaluationItems
        # Sort by sample_id to maintain consistent ordering
        data = []
        for sample_key in sorted(samples.keys(), key=lambda k: samples[k]["sample_id"]):
            item = EvaluationItem(
                sample_id=samples[sample_key]["sample_id"],
                prompt=samples[sample_key]["prompt"],
                rewards=samples[sample_key]["rewards"]
            )
            data.append(item)
        
        evaluator = cls(data)
        
        # Step 3: Validate required models are present
        if required_models:
            available_models = set()
            for item in data:
                available_models.update(item.rewards.keys())
            
            missing = set(required_models) - available_models
            if missing:
                raise ValueError(
                    f"Required models missing from dataset: {missing}\n"
                    f"Available: {sorted(available_models)}"
                )
            
            logger.info(f"✅ All required models present: {required_models}")
        
        return evaluator
    
    @classmethod
    def from_legacy_format(cls, prompts: List[str], rewards_map: Dict[str, List[float]]) -> "AlignedEvaluator":
        """
        Emergency converter for legacy buggy format.
        
        ⚠️  WARNING: This assumes prompts[i] corresponds to rewards_map[model][i]
                    which is what the original buggy code HOPED but didn't guarantee!
        
        Args:
            prompts: List of prompt strings
            rewards_map: Dict of {model: [reward1, reward2, ...]}
            
        Returns:
            AlignedEvaluator (with assumed alignment - may still be wrong!)
        """
        logger.warning("⚠️  Using legacy format converter - alignment not guaranteed!")
        
        # Get length of shortest reward list (to avoid index errors)
        min_length = min(len(rewards) for rewards in rewards_map.values())
        if min_length < len(prompts):
            logger.warning(
                f"⚠️  Reward lists shorter than prompt list! "
                f"Truncating from {len(prompts)} to {min_length} samples"
            )
        
        data = []
        for i in range(min(len(prompts), min_length)):
            rewards = {}
            for model, reward_list in rewards_map.items():
                if i < len(reward_list):
                    rewards[model] = reward_list[i]
            
            data.append(EvaluationItem(
                sample_id=i,
                prompt=prompts[i],
                rewards=rewards
            ))
        
        return cls(data)
    
    def __iter__(self) -> Iterator[EvaluationItem]:
        """Iterate over aligned evaluation items."""
        return iter(self.data)
    
    def __len__(self) -> int:
        """Number of evaluation samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> EvaluationItem:
        """Get specific evaluation item by index."""
        return self.data[idx]
    
    def filter_models(self, model_list: List[str]) -> "AlignedEvaluator":
        """
        Create a new evaluator with only specified models' rewards.
        
        Useful for experiments that only use a subset of models.
        
        Args:
            model_list: Models to keep
            
        Returns:
            New AlignedEvaluator with filtered rewards
        """
        filtered_data = []
        for item in self.data:
            filtered_rewards = {
                m: r for m, r in item.rewards.items() 
                if m in model_list
            }
            # Only keep items that have at least one required model
            if filtered_rewards:
                filtered_data.append(EvaluationItem(
                    sample_id=item.sample_id,
                    prompt=item.prompt,
                    rewards=filtered_rewards
                ))
        
        return AlignedEvaluator(filtered_data)
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dict with summary statistics
        """
        all_models = set()
        for item in self.data:
            all_models.update(item.rewards.keys())
        
        # Count how many samples each model appears in
        model_coverage = {m: 0 for m in all_models}
        for item in self.data:
            for model in item.rewards.keys():
                model_coverage[model] += 1
        
        return {
            "num_samples": len(self.data),
            "num_models": len(all_models),
            "models": sorted(all_models),
            "model_coverage": model_coverage,
            "avg_models_per_sample": sum(len(item.rewards) for item in self.data) / len(self.data)
        }

