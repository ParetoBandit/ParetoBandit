#!/usr/bin/env python3
"""
run_dual_tuning.py

Runs hyperparameter tuning (N_eff, Alpha) for two configurations:
1. Full 9-model portfolio (Default)
2. 2-model baseline (Gemini 3 Pro + GPT-OSS-120B)

Saves results to a consolidated JSON file.
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
from sentence_transformers import SentenceTransformer
from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def get_variance(prompt_rewards):
    if not prompt_rewards:
        return 0.0
    values = list(prompt_rewards.values())
    return np.var(values)

def create_curriculum(prompts, full_corpus, oversample_rate=3):
    """Create a balanced curriculum by oversampling Hard prompts."""
    hard_prompts = [p for p in prompts if get_variance(full_corpus[p]) > 0.05]
    easy_prompts = [p for p in prompts if get_variance(full_corpus[p]) <= 0.05]
    
    curriculum = []
    # 1. Add Hard Prompts (Boosted)
    curriculum.extend(hard_prompts * oversample_rate)
    
    # 2. Add Easy Prompts (Balanced)
    target_easy_size = len(hard_prompts) * oversample_rate
    if len(easy_prompts) > target_easy_size:
        curriculum.extend(random.sample(easy_prompts, target_easy_size))
    else:
        curriculum.extend(easy_prompts)
        
    random.shuffle(curriculum)
    return curriculum

def tune_configuration(config_name, registry, priors_path):
    print(f"\n{'='*70}")
    print(f"🔧 TUNING CONFIGURATION: {config_name}")
    print(f"   Models: {len(registry)}")
    print(f"   Priors: {priors_path}")
    print(f"{'='*70}")

    # 1. Load Data
    print("📦 Loading corpus...")
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    full_corpus = {**train_rewards, **test_rewards}
    all_prompts = list(full_corpus.keys())
    
    # 2. Strict Split
    dev_pool, holdout_pool = train_test_split(all_prompts, test_size=0.4, random_state=42)
    
    # 3. 3-Fold Grid Search
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_n_eff = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    grid_alpha = [0.05, 0.1, 0.5, 1.0]
    
    heatmap_results = []
    best_config = None
    best_score = float('inf')
    
    print("-" * 80)
    print(f"{'N_eff':<8} | {'Alpha':<6} | {'Avg Regret':<12} | {'Std Dev':<10} | {'Status'}")
    print("-" * 80)
    
    for n in grid_n_eff:
        for alpha in grid_alpha:
            fold_regrets = []
            
            for fold_i, (train_idx, val_idx) in enumerate(kf.split(dev_pool)):
                fold_train = [dev_pool[i] for i in train_idx]
                fold_val = [dev_pool[i] for i in val_idx]
                
                # Apply Curriculum to Fold Train
                curriculum = create_curriculum(fold_train, full_corpus)
                
                # Burn-in (Prime)
                # Note: We pass the absolute path string to 'priors' which works with our patched router
                # Also need to figure out 'prior_n_effective' logic.
                # In create(), if 'priors' is generic file path, it falls back to '10.0' default unless specified.
                # Here we strictly specify it.
                
                router = BanditRouter.create(
                    registry, 
                    context_encoder=encoder, 
                    priors=str(priors_path),
                    prior_n_effective=n
                )
                router.bandit.alpha = alpha
                
                for p in curriculum:
                    # Simulation: Route -> Get Reward -> Update
                    # We must simulate the router's choice
                    m, _ = router.route(p, profile="max_quality")
                    
                    # If model not in corpus for this prompt (rare but possible), skip or assume 0
                    if p not in full_corpus or m not in full_corpus[p]:
                        r = 0.0
                    else:
                        r = full_corpus[p][m]
                        
                    router.update(m, p, r)
                    
                # Offline Eval on Fold Val
                val_regret = 0.0
                eval_count = 0
                for p in fold_val:
                    if p not in full_corpus: continue
                    
                    m, _ = router.route(p, profile="max_quality")
                    
                    if m not in full_corpus[p]:
                         r = 0.0
                    else:
                         r = full_corpus[p][m]
                         
                    best = max(full_corpus[p].values()) if full_corpus[p] else 0.0
                    val_regret += (best - r)
                    eval_count += 1
                
                if eval_count > 0:
                    fold_regrets.append(val_regret / eval_count)
                else:
                    fold_regrets.append(0.0)
                
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
            
    print("-" * 80)
    print(f"🏆 Winner ({config_name}): N={best_config['n_eff']}, Alpha={best_config['alpha']} (Regret: {best_score:.4f})")
    
    # ---------------------------------------------------------
    # 4. Final Verification Lap (The "Gold Standard" Requirement)
    # ---------------------------------------------------------
    print(f"\n🔒 Final Verification on Hold-out ({len(holdout_pool)} samples)...")
    
    # A. Train on FULL Dev Set with Curriculum
    full_curriculum = create_curriculum(dev_pool, full_corpus)
    final_router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors=str(priors_path),
        prior_n_effective=best_config['n_eff']
    )
    final_router.bandit.alpha = best_config['alpha']
    
    print(f"   🔥 Burning in on {len(full_curriculum)} samples (Full Dev Curriculum)...")
    for p in full_curriculum:
        m, _ = final_router.route(p, profile="max_quality")
        if p not in full_corpus or m not in full_corpus[p]:
             r = 0.0
        else:
             r = full_corpus[p][m]
        final_router.update(m, p, r)
        
    # B. Online Eval on Hold-out
    print(f"   🚀 Evaluating Online on {len(holdout_pool)} Hold-out prompts...")
    holdout_regret = 0.0
    eval_count = 0
    
    for p in holdout_pool:
        if p not in full_corpus: continue
        
        m, _ = final_router.route(p, profile="max_quality")
        
        if m not in full_corpus[p]:
             r = 0.0
        else:
             r = full_corpus[p][m]
             
        best = max(full_corpus[p].values()) if full_corpus[p] else 0.0
        holdout_regret += (best - r)
        eval_count += 1
        
        # ONLINE UPDATE (Crucial for Bandit performance)
        final_router.update(m, p, r)
        
    avg_holdout = holdout_regret / eval_count if eval_count > 0 else 0.0
    print(f"🏆 Final Hold-out Regret: {avg_holdout:.4f}")

    return {
        "best_config": best_config,
        "best_cv_score": best_score,
        "holdout_regret": avg_holdout,
        "heatmap_data": heatmap_results
    }

def main():
    full_registry = load_model_registry()
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    results = {}
    
    # 1. 9-Model Configuration
    results["9_models"] = tune_configuration(
        "9_models", 
        full_registry, 
        data_dir / "priors_warmup.joblib"
    )
    
    # 2. 2-Model Configuration
    # Filter registry
    target_models = ["openai/gpt-oss-120b", "google/gemini-3-pro-preview"]
    subset_registry = {k: v for k, v in full_registry.items() if k in target_models}
    
    results["2_models"] = tune_configuration(
        "2_models", 
        subset_registry, 
        data_dir / "priors_warmup_2_models.joblib"
    )
    
    # Save Combined Results
    output_file = RESULTS_DIR / "dual_tuning_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Saved Combined Results to {output_file}")
    
    # Print Summary
    print("\n" + "="*40)
    print("🏁 FINAL SUMMARY")
    print("="*40)
    
    c9 = results["9_models"]["best_config"]
    print(f"9 Models: N_eff={c9['n_eff']}, Alpha={c9['alpha']} (Regret: {results['9_models']['best_cv_score']:.4f})")
    
    c2 = results["2_models"]["best_config"]
    print(f"2 Models: N_eff={c2['n_eff']}, Alpha={c2['alpha']} (Regret: {results['2_models']['best_cv_score']:.4f})")
    
if __name__ == "__main__":
    main()
