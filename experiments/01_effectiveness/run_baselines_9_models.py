#!/usr/bin/env python3
"""
run_baselines_9_models.py

Runs the effectiveness baseline experiment with the full 9-model set.
- Uses optimal hyperparameters: N_eff=100.0, Alpha=0.05
- Uses the specific 9-model warmup priors: data/priors_warmup.joblib
- Ensures RouteLLM uses the same anchors (Gemini 3 Pro + GPT-OSS).
"""

import sys
import json
import copy
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Reuse components from run_baselines.py
from run_baselines import (
    run_random_baseline,
    run_vanilla_linucb,
    run_routellm_baseline,
    test_router,
    perform_burn_in,
    prepare_data_split,
    generate_curriculum,
    analyze_by_difficulty
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from utils.data_loader import load_oracle_rewards, load_model_registry
from utils.metrics import calculate_cumulative_regret

def main():
    print("=" * 70)
    print("EXPERIMENT 01: 9-MODEL BASELINE COMPARISON")
    print("Models: Full 9-Model Portfolio")
    print("Params: N_eff=100.0, Alpha=0.05")
    print("=" * 70)
    
    # 1. Load Data
    print("\n📦 Loading and Merging Corpus...")
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards, **test_rewards}
    
    # 2. Split Data (Dev/Test)
    dev_prompts, test_prompts = prepare_data_split()
    print(f"  ✓ Dev Set: {len(dev_prompts)}")
    print(f"  ✓ Test Set: {len(test_prompts)}")
    
    # 3. Curriculum
    print("\n🎓 Generating Curriculum...")
    burn_in_list = generate_curriculum(dev_prompts, all_rewards)
    
    # 4. Registry (All 9 Models)
    full_registry = load_model_registry()
    
    # Filter to models that have rewards in at least 50% of test prompts (Standard logic)
    # This usually results in the 9 models we expect
    model_coverage = defaultdict(int)
    for prompt in test_prompts:
        prompt_rewards = all_rewards.get(prompt, {})
        for model_id in prompt_rewards:
            model_coverage[model_id] += 1
            
    min_coverage = len(test_prompts) * 0.5
    available_models = [
        m for m in full_registry.keys() 
        if model_coverage.get(m, 0) >= min_coverage
    ]
    registry = {k: v for k, v in full_registry.items() if k in available_models}
    
    print(f"  ✓ Registry has {len(available_models)} models with ≥50% coverage")
    print(f"    Models: {available_models}")
    
    # 5. Initialize Encoder
    print("\n🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # 6. Calculate Oracle Best (Test Set)
    oracle_best = []
    valid_test_prompts = []
    for p in test_prompts:
        rewards = all_rewards.get(p, {})
        valid_rewards = [rewards.get(m, 0.0) for m in available_models if m in rewards]
        if valid_rewards:
            oracle_best.append(max(valid_rewards))
            valid_test_prompts.append(p)
    oracle_best = np.array(oracle_best)
    test_prompts = valid_test_prompts
    print(f"  ✓ Oracle computed on {len(oracle_best)} prompts")
    
    # 7. Experiment Config
    n_seeds = 10
    best_n_eff = 100.0  # From tuning
    best_alpha = 0.05   # From tuning
    priors_file = Path(__file__).parent.parent.parent / "data" / "priors_warmup.joblib"
    
    results = {}
    method_raw_rewards = defaultdict(list)
    
    # 8. Burn-In Master Router
    print("\n" + "="*40)
    print("🔥 BURN-IN PHASE (BanditGPT)")
    print("="*40)
    
    master_router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors=str(priors_file),
        prior_n_effective=best_n_eff
    )
    master_router.bandit.alpha = best_alpha
    
    perform_burn_in(master_router, burn_in_list, all_rewards)
    
    # 9. Test Loop
    print("\n" + "="*40)
    print(f"🚀 TEST PHASE ({n_seeds} Seeds)")
    print("="*40)
    
    test_data = list(zip(test_prompts, oracle_best))
    rng = np.random.RandomState(42)
    rng.shuffle(test_data)
    shuffled_prompts = [x[0] for x in test_data]
    shuffled_oracle = np.array([x[1] for x in test_data])
    
    # RouteLLM Cache
    routellm_scores_cache = {}
    
    for seed in range(n_seeds):
        print(f"\nSEED {seed + 1}/{n_seeds}")
        
        # A. Random
        res_rand = run_random_baseline(shuffled_prompts, all_rewards, available_models, seed=seed)
        
        # B. LinUCB (Cold)
        res_lin = run_vanilla_linucb(shuffled_prompts, all_rewards, available_models, alpha=1.0, seed=seed)
        
        # C. RouteLLM
        res_route = run_routellm_baseline(
            shuffled_prompts, all_rewards, registry, available_models, 
            cached_scores=routellm_scores_cache, seed=seed
        )
        
        # D. HLE Baseline
        router_hle = BanditRouter.create(
            registry,
            context_encoder=encoder,
            priors="hle",
            prior_n_effective=10.0,
            alpha=best_alpha
        )
        res_hle = test_router(router_hle, shuffled_prompts, all_rewards, priors="hle", seed=seed)
        
        # E. BanditGPT (Warmup + Burned In)
        router_hot = copy.deepcopy(master_router)
        router_hot.bandit.rng = np.random.RandomState(seed)
        res_bandit = test_router(router_hot, shuffled_prompts, all_rewards, priors="warmup", seed=seed)
        
        # Collect
        batch = [res_rand, res_lin, res_route, res_hle, res_bandit]
        for res in batch:
            m = res["method"]
            cregret = calculate_cumulative_regret(res["rewards"], shuffled_oracle)
            
            if m not in results: results[m] = []
            results[m].append(cregret.tolist())
            method_raw_rewards[m].append(res["rewards"])
            
            print(f"  {m:25s}: Regret={cregret[-1]:7.1f}")
            
    # 10. Save & Analyze
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "effectiveness_results_9_models.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Results saved to {output_dir / 'effectiveness_results_9_models.json'}")
    
    avg_rewards = {m: np.mean(np.array(r), axis=0) for m, r in method_raw_rewards.items()}
    analyze_by_difficulty(avg_rewards, shuffled_prompts, shuffled_oracle, available_models, all_rewards)

if __name__ == "__main__":
    main()
