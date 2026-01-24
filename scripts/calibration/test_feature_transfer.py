#!/usr/bin/env python3
"""
Feature-Only Transfer Experiment (Correcting Domain Mismatch)

This script tests the hypothesis that transferring only the COVARIANCE (A) 
while resetting the REWARD HISTORY (b) allows the router to benefit from 
learned representations without being misled by incorrect priors.

Mechanism:
1. Load warmup priors (A, b)
2. Scale A by gamma (feature importance transfer)
3. Reset b to zero (remove bias toward specific models)
4. Compare against Tabula Rasa and Standard Warmup
"""

import sys
import joblib
import json
import gzip
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH
)

def reset_rewards(priors: dict) -> dict:
    """
    Keep covariance (A) but reset rewards (b) to zero.
    This transfers 'what matters' (features) but not 'who wins' (bias).
    """
    new_priors = priors.copy()
    new_priors['b'] = {m: np.zeros_like(priors['b'][m]) for m in priors['models']}
    return new_priors

def run_experiment(
    calibration_data: List[dict],
    warmup_priors: dict,
    encoder: SentenceTransformer,
    pca_model,
    gamma: float,
    mode: str = "standard"  # 'standard', 'feature_only', 'tabula_rasa'
) -> Tuple[float, float, float]:
    
    # Configure priors based on mode
    if mode == "tabula_rasa":
        # Initialize from scratch
        router = SimpleLinUCBRouter(
            models=warmup_priors['models'],
            warmup_priors={
                'A': {m: np.eye(warmup_priors['context_dim']) for m in warmup_priors['models']},
                'b': {m: np.zeros(warmup_priors['context_dim']) for m in warmup_priors['models']},
                'context_dim': warmup_priors['context_dim']
            }
        )
    else:
        # Apply gamma scaling first
        priors_scaled = apply_gamma_scaling(warmup_priors, gamma)
        
        if mode == "feature_only":
            # Reset rewards (b) to zero
            priors_scaled = reset_rewards(priors_scaled)
            
        router = SimpleLinUCBRouter(
            models=warmup_priors['models'],
            warmup_priors=priors_scaled
        )

    total_reward = 0.0
    cumulative_regret = 0.0
    
    # Map for data lookup
    # Note: dev_rewards_2models.jsonl.gz uses "mistralai/mixtral-8x7b-instruct"
    # and "openai/gpt-4-turbo", which matches the router models exactly.
    
    for item in calibration_data:
        # Embed
        context = embed_prompt(item['prompt'], encoder, pca_model)
        
        # Select
        selected = router.select_model(context)
        
        # Get reward
        reward = item['rewards'].get(selected, 0.0)
        
        # Update
        router.update(context, selected, reward)
        
        # Metrics
        total_reward += reward
        
        # Oracle for regret
        available_rewards = list(item['rewards'].values())
        oracle_reward = max(available_rewards) if available_rewards else 0.0
        cumulative_regret += (oracle_reward - reward)
        
    avg_reward = total_reward / len(calibration_data)
    return avg_reward, cumulative_regret

def main():
    parser = argparse.ArgumentParser(description="Feature-Only Transfer Experiment")
    parser.add_argument("--output", type=str, default="experiments_v1/03_figure/feature_transfer_results")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 STARTING FEATURE-ONLY TRANSFER EXPERIMENT")
    print("="*60)
    
    # 1. Load Resources
    print("📥 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca_model = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # 2. Load Data (Use 2-model dataset)
    print(f"📊 Loading data from {CANONICAL_DEV_DATA_PATH}...")
    calibration_data = []
    with gzip.open(CANONICAL_DEV_DATA_PATH, 'rt') as f:
        prompts_seen = set()
        for line in f:
            entry = json.loads(line)
            if not entry.get('ok', True): continue
            
            prompt = entry['prompt']
            model = entry['model_id']
            score = entry['raw_score']
            
            # Group by prompt
            found = False
            for item in calibration_data:
                if item['prompt'] == prompt:
                    item['rewards'][model] = score
                    found = True
                    break
            
            if not found:
                calibration_data.append({
                    'prompt': prompt,
                    'rewards': {model: score}
                })
    
    # Filter complete entries
    calibration_data = [
        item for item in calibration_data 
        if len(item['rewards']) == 2
    ]
    print(f"✅ Loaded {len(calibration_data)} complete prompts")
    
    # 3. Run Experiments
    gammas = [1.0, 0.5, 0.1, 0.05, 0.01, 0.002]
    results = []
    
    # Baseline: Tabula Rasa
    print("\n🧪 Running Tabula Rasa Baseline...")
    tr_reward, tr_regret = run_experiment(
        calibration_data, warmup_priors, encoder, pca_model, 1.0, mode="tabula_rasa"
    )
    print(f"   Reward: {tr_reward:.4f} | Regret: {tr_regret:.2f}")
    
    results.append({
        "gamma": "TR",
        "mode": "Tabula Rasa",
        "reward": tr_reward,
        "regret": tr_regret
    })
    
    print("\n🧪 Running Gamma Sweeps...")
    print(f"{'Gamma':<8} | {'Standard Reward':<15} | {'Feat-Only Reward':<15} | {'Improvement':<12}")
    print("-" * 60)
    
    for gamma in gammas:
        # Standard Warmup
        std_reward, std_regret = run_experiment(
            calibration_data, warmup_priors, encoder, pca_model, gamma, mode="standard"
        )
        
        # Feature-Only Transfer
        ft_reward, ft_regret = run_experiment(
            calibration_data, warmup_priors, encoder, pca_model, gamma, mode="feature_only"
        )
        
        imp = ft_reward - std_reward
        print(f"{gamma:<8} | {std_reward:.4f}          | {ft_reward:.4f}          | {imp:+.4f}")
        
        results.append({
            "gamma": gamma,
            "mode": "Standard Warmup",
            "reward": std_reward,
            "regret": std_regret
        })
        results.append({
            "gamma": gamma,
            "mode": "Feature-Only",
            "reward": ft_reward,
            "regret": ft_regret
        })

    # 4. Save Results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Experiment Complete!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

