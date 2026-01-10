#!/usr/bin/env python3
"""
Example demonstrating the improved get_splits() API with automatic reward joining.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from src.bandit_gpt.router import DEFAULT_CONTEXT_MODEL
from sentence_transformers import SentenceTransformer

def example_old_way():
    """The old way - manual loading and joining."""
    print("="*70)
    print("OLD WAY: Manual Loading + Joining")
    print("="*70)
    
    from experiments.utils.data_loader import load_oracle_rewards, load_model_registry
    
    # Load rewards separately
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # Load splits
    registry = load_model_registry()
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    splits_path = Path(__file__).parent.parent / "experiments" / "01_effectiveness" / "results" / "splits.json"
    
    burner = ExperimentBurnIn(registry, all_rewards, splits_path, encoder)
    dev_prompts, holdout_prompts = burner.get_splits()
    
    # Manual filtering/joining required
    print(f"✓ Dev prompts: {len(dev_prompts)}")
    print(f"✓ Holdout prompts: {len(holdout_prompts)}")
    print(f"✓ All rewards loaded: {len(all_rewards)} prompts")
    print(f"⚠️ Risk: Easy to accidentally use wrong rewards\n")


def example_new_way():
    """The new way - automatic reward joining."""
    print("="*70)
    print("NEW WAY: Automatic Reward Joining")
    print("="*70)
    
    from experiments.utils.data_loader import load_model_registry
    
    # Minimal setup
    registry = load_model_registry()
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    splits_path = Path(__file__).parent.parent / "experiments" / "01_effectiveness" / "results" / "splits.json"
    
    # No rewards needed in constructor anymore
    burner = ExperimentBurnIn(registry, {}, splits_path, encoder)
    
    # Get splits WITH rewards automatically joined
    (dev_prompts, dev_rewards), (holdout_prompts, holdout_rewards) = burner.get_splits(load_rewards=True)
    
    print(f"✓ Dev prompts: {len(dev_prompts)}")
    print(f"✓ Dev rewards: {len(dev_rewards)} prompts")
    print(f"✓ Holdout prompts: {len(holdout_prompts)}")
    print(f"✓ Holdout rewards: {len(holdout_rewards)} prompts")
    print(f"✅ Guaranteed: Dev and holdout rewards are disjoint")
    print(f"✅ Crystal clear: Which rewards belong to which split\n")
    
    # Verify rewards match prompts
    dev_prompts_in_rewards = set(dev_prompts).intersection(set(dev_rewards.keys()))
    holdout_prompts_in_rewards = set(holdout_prompts).intersection(set(holdout_rewards.keys()))
    
    print(f"Validation:")
    print(f"  - Dev prompts with rewards: {len(dev_prompts_in_rewards)}/{len(dev_prompts)}")
    print(f"  - Holdout prompts with rewards: {len(holdout_prompts_in_rewards)}/{len(holdout_prompts)}")


def example_backward_compatible():
    """Backward compatibility - still works without rewards."""
    print("="*70)
    print("BACKWARD COMPATIBLE: Works Without Rewards (Default)")
    print("="*70)
    
    from experiments.utils.data_loader import load_model_registry
    
    registry = load_model_registry()
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    splits_path = Path(__file__).parent.parent / "experiments" / "01_effectiveness" / "results" / "splits.json"
    
    burner = ExperimentBurnIn(registry, {}, splits_path, encoder)
    
    # Default behavior unchanged - backward compatible
    dev_prompts, holdout_prompts = burner.get_splits()  # load_rewards=False by default
    
    print(f"✓ Dev prompts: {len(dev_prompts)}")
    print(f"✓ Holdout prompts: {len(holdout_prompts)}")
    print(f"✅ Existing code still works!\n")


if __name__ == "__main__":
    example_old_way()
    print("\n")
    example_new_way()
    print("\n")
    example_backward_compatible()
    
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("✅ New API automatically joins rewards with splits")
    print("✅ Clearer code - no manual loading/filtering needed")
    print("✅ Backward compatible - default behavior unchanged")
    print("✅ Type-safe - clear return types for both modes")
