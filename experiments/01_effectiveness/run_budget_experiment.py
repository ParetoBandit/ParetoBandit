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
            - Online Eval on 1/3 (Interleaved Test-Then-Train).
       - Average Regret across folds.
    3. Final Check:
       - Retrain on FULL Dev set (Curriculum applied).
       - Online Evaluation on Hold-out (Bandit continues learning).

Expected Runtime:
    - First run: 30-45 minutes (includes 10-15 min data loading + grid search)
    - Subsequent runs: 15-30 minutes (uses cached rewards, ~5 sec loading)
    - Grid: 28 hyperparameter configs × 3 folds = 84 iterations

Progress:
    Script will print detailed progress for each phase. If silent for >1 minute,
    data loading is in progress (gzip decompression).
"""

import sys
import numpy as np
import random
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import time

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
    
    # Determinism controls for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # 1. Initialize Experiment Framework
    print("📦 Initializing Experiment Framework...")
    registry = load_model_registry()
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # Save splits path for centralized loading
    splits_path = Path(__file__).parent / "results" / "splits.json"
    
    # Initialize ExperimentBurnIn
    burn_in_helper = ExperimentBurnIn(
        registry=registry,
        splits_path=splits_path,
        encoder=encoder
    )
    
    # 2. Get Canonical Splits with Rewards (Centralized)
    print("📊 Loading Canonical KDD Splits...")
    # This replaces manual loading and manual train_test_split
    (dev_pool, dev_rewards), (holdout_pool, holdout_rewards) = burn_in_helper.get_splits(load_rewards=True)
    
    # Combine into full filtered corpus
    filtered_corpus = {**dev_rewards, **holdout_rewards}
    registry_models = set(registry.keys())
    
    print(f"  ✓ Dev Set (CV): {len(dev_pool)}")
    print(f"  🔒 Hold-out Set: {len(holdout_pool)}")
    print(f"  ✓ Filtered corpus: {len(filtered_corpus)} prompts × {len(registry_models)} models")
    
    
    # 3. 3-Fold Cross Validation
    print("\n🔄 Starting 3-Fold CV Grid Search...")
    
    # ASSERTION: Ensure Strict Separation
    dev_set = set(dev_pool)
    holdout_set = set(holdout_pool)
    assert dev_set.isdisjoint(holdout_set), "CRITICAL: Data Leakage! Dev and Hold-out sets overlap."
    print(f"  ✅ Verified Disjoint Splits (Intersection: {len(dev_set.intersection(holdout_set))})")

    # Baseline: Random Policy Regret
    print("\n📊 Computing Random Baseline...")
    random_regret = 0.0
    for p in dev_pool:
        if p in filtered_corpus:
            models = list(filtered_corpus[p].keys())
            if models:
                m = random.choice(models)
                r = filtered_corpus[p].get(m, 0.0)
                best = max(filtered_corpus[p].values())
                random_regret += (best - r)
    avg_random_regret = random_regret / len(dev_pool)
    print(f"  📉 Random Policy Regret: {avg_random_regret:.4f}")
    print(f"     (Target: Beat this baseline with learned policy)")

    
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_n_eff = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    grid_alpha = [0.05, 0.1, 0.5, 1.0]
    
    # Variance penalty for robust hyperparameter selection
    # With only 3 folds, we penalize high-variance configs to avoid overfitting to noise
    VARIANCE_PENALTY = 0.5  # 0.0 = ignore variance, 1.0 = heavily penalize
    
    heatmap_results = []
    best_config = None
    best_score = float('inf')
    best_mean = float('inf')
    best_std = float('inf')
    
    # Store per-fold results for debugging
    fold_regrets_all = {}
    
    # Pre-calculate folds to ensure consistency? No, kf.split is deterministic with state.
    
    print("-" * 80)
    print(f"{'N_eff':<8} | {'Alpha':<6} | {'Avg Regret':<12} | {'Std Dev':<10} | {'Time':<8} | {'Status'}")
    print("-" * 80)
    
    total_configs = len(grid_n_eff) * len(grid_alpha)
    config_idx = 0
    start_time = time.time()
    
    for n in grid_n_eff:
        for alpha in grid_alpha:
            config_idx += 1
            config_start = time.time()
            
            fold_regrets = []
            
            print(f"\n[{config_idx}/{total_configs}] Testing N_eff={n}, Alpha={alpha}...")
            
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
                # Use existing PCA artifact (don't recreate)
                pca_path = Path(__file__).parent.parent.parent / "artifacts" / "pca_23.joblib"
                router = BanditRouter.create(
                    registry, 
                    context_encoder=encoder, 
                    priors="warmup",
                    prior_n_effective=n,
                    alpha=alpha,  # Set exploration parameter during creation
                    pca_path=pca_path  # Use existing PCA data
                )
                
                for p in tqdm(curriculum, desc=f"  Fold {fold_i+1}/3 Burn-in", leave=False):
                    m, _ = router.route(p, profile="max_quality")
                    r = filtered_corpus[p].get(m, 0.0)
                    router.update(m, p, r)
                    
                # C. Online Eval on Fold Val (Interleaved Test-Then-Train)
                # CRITICAL FIX: Update bandit to credit exploration (aligns with hold-out protocol)
                val_regret = 0.0
                for p in tqdm(fold_val, desc=f"  Fold {fold_i+1}/3 Validation", leave=False):
                    m, _ = router.route(p, profile="max_quality")
                    r = filtered_corpus[p].get(m, 0.0)
                    best = max(filtered_corpus[p].values()) if filtered_corpus[p] else 0.0
                    val_regret += (best - r)
                    
                    # ONLINE UPDATE: Credit exploration for future decisions
                    router.update(m, p, r)
                
                fold_regrets.append(val_regret / len(fold_val))
                
            mean_regret = np.mean(fold_regrets)
            std_regret = np.std(fold_regrets)
            
            config_time = time.time() - config_start
            
            heatmap_results.append({
                "n_eff": n, "alpha": alpha, 
                "mean_regret": mean_regret, "std_regret": std_regret
            })
            
            # Store per-fold results for debugging
            fold_regrets_all[(n, alpha)] = fold_regrets
            
            # Variance-aware selection: penalize unstable configurations
            score = mean_regret + VARIANCE_PENALTY * std_regret
            
            tag = ""
            if score < best_score:
                best_score = score
                best_mean = mean_regret
                best_std = std_regret
                best_config = {"n_eff": n, "alpha": alpha}
                tag = "🌟 New Best"
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            avg_time_per_config = elapsed / config_idx
            remaining_configs = total_configs - config_idx
            eta_seconds = avg_time_per_config * remaining_configs
            eta_min = eta_seconds / 60
                
            print(f"{n:<8.1f} | {alpha:<6.2f} | {mean_regret:<12.4f} | {std_regret:<10.4f} | {config_time:<8.1f}s | {tag}")
            if remaining_configs > 0:
                print(f"         ETA: {eta_min:.1f} minutes ({remaining_configs} configs remaining)")
            
            
    save_heatmap_data(heatmap_results)
    print("-" * 80)
    print(f"🏆 Winner: N={best_config['n_eff']}, Alpha={best_config['alpha']}")
    print(f"   Score (mean + {VARIANCE_PENALTY}*std): {best_score:.4f}")
    print(f"   Raw Regret: {best_mean:.4f} ± {best_std:.4f}")
    
    improvement = (avg_random_regret - best_mean) / avg_random_regret if avg_random_regret > 0 else 0
    print(f"   🚀 Improvement over Random: {improvement:.1%}")
    
    # 4. Final Evaluation
    print("\n🔒 Final Evaluation on Hold-out...")
    
    # A. Train on FULL Dev Set with Curriculum (using centralized curriculum generation)
    full_curriculum = burn_in_helper.generate_curriculum(dev_pool)
    # Use existing PCA artifact (don't recreate)
    pca_path = Path(__file__).parent.parent.parent / "artifacts" / "pca_23.joblib"
    final_router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors="warmup",
        prior_n_effective=best_config['n_eff'],
        alpha=best_config['alpha'],  # Set exploration parameter during creation
        pca_path=pca_path  # Use existing PCA data
    )
    
    print(f"   🔥 Burning in on {len(full_curriculum)} samples (Full Dev Curriculum)...")
    for p in tqdm(full_curriculum, desc="   Final Burn-in", leave=False):
        m, _ = final_router.route(p, profile="max_quality")
        r = filtered_corpus[p].get(m, 0.0)
        final_router.update(m, p, r)
        
    # B. Online Eval on Hold-out
    print(f"   🚀 Evaluating Online on {len(holdout_pool)} Hold-out prompts...")
    holdout_regret = 0.0
    for p in tqdm(holdout_pool, desc="   Holdout Evaluation", leave=False):
        m, _ = final_router.route(p, profile="max_quality")
        r = filtered_corpus[p].get(m, 0.0)
        best = max(filtered_corpus[p].values()) if filtered_corpus[p] else 0.0
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
