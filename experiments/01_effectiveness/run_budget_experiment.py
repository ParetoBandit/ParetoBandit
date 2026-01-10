#!/usr/bin/env python3
"""
Experiment: Hyperparameter Tuning with Budget Constraints (Gold Standard CV)

Strategy: "Robust Model Selection via 3-Fold Cross-Validation"
Objective: Find stable hyperparameters (N_eff, Alpha) and generate Heatmap data.

Protocol:
    1. Split: 60% Development (1200), 40% Hold-out (800).
    2. Tuning (3-Fold CV on Dev):
       - Split Dev into 3 folds.
       - For each fold:
            - Train on 2/3 (Curriculum Oversampling applied HERE).
            - Offline Eval on 1/3.
       - Average Regret across folds.
    3. Final Check:
       - Retrain on FULL Dev set (Curriculum applied).
       - Online Evaluation on Hold-out (Bandit continues learning).
"""

import sys
import numpy as np
import random
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from sentence_transformers import SentenceTransformer
from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def save_heatmap_data(results):
    """Save (N, Alpha, Regret) triples for plotting."""
    path = Path(__file__).parent / "results" / "heatmap_data.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   💾 Saved Heatmap Data to {path}")

def run_gold_standard_tuning():
    print("="*70)
    print("🏆 GOLD STANDARD TUNING (3-Fold CV + Hold-out)")
    print("="*70)
    
    # 1. Load Data
    print("📦 Loading corpus...")
    registry = load_model_registry()
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    full_corpus = {**train_rewards, **test_rewards}
    all_prompts = list(full_corpus.keys())
    
    # 2. Strict Split (60% Dev / 40% Hold-out)
    dev_pool, holdout_pool = train_test_split(all_prompts, test_size=0.4, random_state=42)
    
    print(f"  ✓ Corpus: {len(all_prompts)}")
    print(f"  ✂️  Dev Set (CV): {len(dev_pool)}")
    print(f"  🔒 Hold-out Set: {len(holdout_pool)}")
    
    # Save splits for reproducibility
    splits_path = Path(__file__).parent / "results" / "splits.json"
    splits_path.parent.mkdir(exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump({
            "dev_pool": dev_pool,
            "holdout_pool": holdout_pool
        }, f, indent=2)
    print(f"  💾 Saved Splits to {splits_path}")
    
    
    # 3. 3-Fold Cross Validation
    print("\n🔄 Starting 3-Fold CV Grid Search...")
    
    # ASSERTION: Ensure Strict Separation
    dev_set = set(dev_pool)
    holdout_set = set(holdout_pool)
    assert dev_set.isdisjoint(holdout_set), "CRITICAL: Data Leakage! Dev and Hold-out sets overlap."
    print(f"  ✅ Verified Disjoint Splits (Intersection: {len(dev_set.intersection(holdout_set))})")

    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # Initialize ExperimentBurnIn for curriculum generation
    burn_in_helper = ExperimentBurnIn(
        registry=registry,
        oracle_rewards=full_corpus,
        splits_path=splits_path,
        encoder=encoder
    )
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_n_eff = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    grid_alpha = [0.05, 0.1, 0.5, 1.0]
    
    heatmap_results = []
    best_config = None
    best_score = float('inf')
    
    # Pre-calculate folds to ensure consistency? No, kf.split is deterministic with state.
    
    print("-" * 80)
    print(f"{'N_eff':<8} | {'Alpha':<6} | {'Avg Regret':<12} | {'Std Dev':<10} | {'Status'}")
    print("-" * 80)
    
    for n in grid_n_eff:
        for alpha in grid_alpha:
            fold_regrets = []
            
            for fold_i, (train_idx, val_idx) in enumerate(kf.split(dev_pool)):
                fold_train = [dev_pool[i] for i in train_idx]
                fold_val = [dev_pool[i] for i in val_idx]
                
                # ASSERTION: Fold Separation
                train_set = set(fold_train)
                val_set = set(fold_val)
                assert train_set.isdisjoint(val_set), "CRITICAL: Data Leakage! Train and Val folds overlap."
                assert train_set.isdisjoint(holdout_set), "CRITICAL: Data Leakage! Train fold overlaps with Hold-out."
                assert val_set.isdisjoint(holdout_set), "CRITICAL: Data Leakage! Val fold overlaps with Hold-out."
                
                # A. Apply Curriculum to Fold Train (using centralized curriculum generation)
                curriculum = burn_in_helper.generate_curriculum(fold_train)
                
                # B. Burn-in (Prime)
                router = BanditRouter.create(
                    registry, 
                    context_encoder=encoder, 
                    priors="warmup",
                    prior_n_effective=n
                )
                router.bandit.alpha = alpha # Set targeted exploration immediately?
                # Usually burn-in uses a fixed policy. Let's assume the router uses 'alpha' for routing
                # but 'update' logic is policy-agnostic (LinUCB). 
                # Yes, we set it.
                
                for p in curriculum:
                    m, _ = router.route(p, profile="max_quality")
                    r = full_corpus[p].get(m, 0.0)
                    router.update(m, p, r)
                    
                # C. Offline Eval on Fold Val
                val_regret = 0.0
                for p in fold_val:
                    m, _ = router.route(p, profile="max_quality")
                    r = full_corpus[p].get(m, 0.0)
                    best = max(full_corpus[p].values()) if full_corpus[p] else 0.0
                    val_regret += (best - r)
                    # No Update for Offline Eval
                
                fold_regrets.append(val_regret / len(fold_val))
                
            mean_regret = np.mean(fold_regrets)
            std_regret = np.std(fold_regrets)
            
            heatmap_results.append({
                "n_eff": n, "alpha": alpha, 
                "mean_regret": mean_regret, "std_regret": std_regret
            })
            
            tag = ""
            if mean_regret < best_score:
                best_score = mean_regret
                best_config = {"n_eff": n, "alpha": alpha}
                tag = "🌟 New Best"
                
            print(f"{n:<8.1f} | {alpha:<6.1f} | {mean_regret:<12.4f} | {std_regret:<10.4f} | {tag}")
            
    save_heatmap_data(heatmap_results)
    print("-" * 80)
    print(f"🏆 Winner: N={best_config['n_eff']}, Alpha={best_config['alpha']} (Regret: {best_score:.4f})")
    
    # 4. Final Evaluation
    print("\n🔒 Final Evaluation on Hold-out...")
    
    # A. Train on FULL Dev Set with Curriculum (using centralized curriculum generation)
    full_curriculum = burn_in_helper.generate_curriculum(dev_pool)
    final_router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors="warmup",
        prior_n_effective=best_config['n_eff']
    )
    final_router.bandit.alpha = best_config['alpha']
    
    print(f"   🔥 Burning in on {len(full_curriculum)} samples (Full Dev Curriculum)...")
    for p in full_curriculum:
        m, _ = final_router.route(p, profile="max_quality")
        r = full_corpus[p].get(m, 0.0)
        final_router.update(m, p, r)
        
    # B. Online Eval on Hold-out
    print(f"   🚀 Evaluating Online on {len(holdout_pool)} Hold-out prompts...")
    holdout_regret = 0.0
    for p in holdout_pool:
        m, _ = final_router.route(p, profile="max_quality")
        r = full_corpus[p].get(m, 0.0)
        best = max(full_corpus[p].values()) if full_corpus[p] else 0.0
        holdout_regret += (best - r)
        
        # ONLINE UPDATE
        final_router.update(m, p, r)
        
    avg_holdout = holdout_regret / len(holdout_pool)
    print(f"🏆 Final Hold-out Regret: {holdout_regret:.2f} (Avg: {avg_holdout:.4f})")
    
    # Save Final Metric
    with open("experiments/01_effectiveness/results/gold_standard_metric.txt", "w") as f:
        f.write(f"Final Test Regret (Online): {avg_holdout:.4f}\n")
        f.write(f"Optimal N_eff: {best_config['n_eff']}\n")
        f.write(f"Optimal Alpha: {best_config['alpha']}\n")

if __name__ == "__main__":
    run_gold_standard_tuning()
