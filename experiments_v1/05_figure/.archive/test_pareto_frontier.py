#!/usr/bin/env python3
"""
Quick test of Pareto Frontier with small subset (50 prompts)
Tests that RouteLLM integration works before running full experiment.
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
from typing import Dict, List, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.config_legacy import (
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
)
from routellm.controller import Controller
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_model_costs() -> Dict:
    """Load model costs from models.json."""
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    
    costs = {}
    for model in data["models"]:
        model_id = model["openrouter_id"]
        input_cost = model.get("price_1m_input", 0.0)
        output_cost = model.get("price_1m_output", 0.0)
        cost_per_request = (100 * input_cost + 400 * output_cost) / 1_000_000
        
        costs[model_id] = {
            "name": model.get("display_name", model_id),
            "cost": cost_per_request
        }
    
    return costs


def load_small_dataset(n_prompts: int = 50) -> Tuple[List[Dict], List[Dict]]:
    """Load small subset for testing."""
    logger.info(f"\n📥 Loading SMALL TEST dataset ({n_prompts} prompts)...")
    
    def load_split(filepath, label, limit):
        prompt_rewards = defaultdict(lambda: {})
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    prompt = entry["prompt"]
                    model_id = entry["model_id"]
                    score = entry.get("raw_score", 0.0)
                    prompt_rewards[prompt][model_id] = score
        
        prompts_data = []
        for prompt, rewards in prompt_rewards.items():
            if len(rewards) == 2:
                prompts_data.append({"prompt": prompt, "rewards": rewards})
            if len(prompts_data) >= limit:
                break
        
        logger.info(f"  {label}: {len(prompts_data)} prompts")
        return prompts_data
    
    train_data = load_split(CANONICAL_DEV_DATA_PATH, "Train (Dev)", n_prompts)
    eval_data = load_split(CANONICAL_HOLDOUT_DATA_PATH, "Test (Holdout)", n_prompts)
    
    return train_data, eval_data


def test_routellm(eval_data: List[Dict], model_costs: Dict):
    """Test RouteLLM with a few thresholds."""
    logger.info("\n🧪 Testing RouteLLM Matrix Factorization Router...")
    
    weak_model = "mistralai/mixtral-8x7b-instruct"
    strong_model = "openai/gpt-4-turbo"
    
    # Initialize controller
    logger.info("   Initializing controller...")
    controller = Controller(
        routers=['mf'],
        strong_model=strong_model,
        weak_model=weak_model
    )
    logger.info("   ✓ Controller initialized")
    
    # Test 3 thresholds
    thresholds = [0.0, 0.5, 1.0]
    results = []
    
    for threshold in thresholds:
        logger.info(f"\n   Testing threshold {threshold:.1f}...")
        
        total_reward = 0.0
        total_cost = 0.0
        model_counts = defaultdict(int)
        
        for i, prompt_data in enumerate(eval_data):
            prompt = prompt_data["prompt"]
            rewards = prompt_data["rewards"]
            
            # Route with RouteLLM
            selected_model = controller.route(prompt, router='mf', threshold=threshold)
            model_counts[selected_model] += 1
            
            # Get reward and cost
            if selected_model in rewards:
                reward = rewards[selected_model]
                cost = model_costs[selected_model]["cost"]
                total_reward += reward
                total_cost += cost
            
            if (i + 1) % 10 == 0:
                logger.info(f"      Processed {i+1}/{len(eval_data)} prompts...")
        
        avg_reward = total_reward / len(eval_data)
        avg_cost = total_cost / len(eval_data)
        
        results.append((threshold, avg_reward, avg_cost))
        
        logger.info(f"   ✓ Threshold {threshold:.1f}:")
        logger.info(f"      Reward: {avg_reward:.4f}")
        logger.info(f"      Cost: ${avg_cost:.6f}")
        logger.info(f"      Model usage: {dict(model_counts)}")
    
    return results


def main():
    logger.info("="*70)
    logger.info("QUICK TEST: Pareto Frontier with RouteLLM")
    logger.info("="*70)
    
    # Load costs
    logger.info("\n📦 Loading model costs...")
    model_costs = load_model_costs()
    logger.info(f"   ✓ Loaded costs for {len(model_costs)} models")
    
    # Load small dataset
    train_data, eval_data = load_small_dataset(n_prompts=50)
    
    # Test RouteLLM
    results = test_routellm(eval_data, model_costs)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ TEST COMPLETE!")
    logger.info("="*70)
    logger.info("\n📊 Summary:")
    for threshold, reward, cost in results:
        logger.info(f"  Threshold {threshold:.1f}: Reward={reward:.4f}, Cost=${cost:.6f}")
    
    logger.info("\n✅ RouteLLM integration works!")
    logger.info("   Ready to run full experiment with 750 prompts.")


if __name__ == "__main__":
    main()

