
import sys
import os
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import statsmodels.api as sm

# Add src to path
sys.path.append(str(Path.cwd() / "src"))
sys.path.append(str(Path.cwd() / "experiments"))

from bandit_gpt.router import BanditRouter
from utils.data_loader import load_oracle_rewards

def analyze_p_values():
    print("🚀 Starting Feature Significance Analysis (P-Values)...")
    
    # 1. Initialize Router with Pruned Features Active
    print("Initializing BanditRouter with 'use_pruned_features=True'...")
    # This automatically sets use_pruned_features=True in default config
    router = BanditRouter.create(priors="none")
    
    # Verify Dimension
    print(f"Bandit Dimension: {router.bandit.dim}")
    if router.bandit.dim != 13:
        print(f"❌ WARNING: Expected dim=13, got {router.bandit.dim}. Check configuration!")
        return

    # 2. Load Training Data
    print("Loading training data...")
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    prompts = list(train_rewards.keys())
    print(f"Loaded {len(prompts)} prompts")
    
    # 3. Extract Features for All Prompts
    print("Extracting 13 features (PCA-8 + Pruned Handcrafted)...")
    X = []
    
    # Define Feature Names explicitly based on pruning logic
    feature_names = [f"pca_{i}" for i in range(8)]
    feature_names.extend(["has_latex", "latex_density_log"])
    feature_names.extend(["anchor_math", "anchor_jokes"])
    feature_names.append("bias")
    
    print(f"Feature Names: {feature_names}")

    valid_prompts = []
    for prompt in tqdm(prompts):
        try:
            # router._get_context_vector is not directly exposed as public but used internally
            # We can use _build_routing_features but it returns (features, keys)
            # Actually, let's call the internal method since we imported the class
            x = router._get_context_vector(prompt)
            
            if len(x) != 13:
                 print(f"WARNING: Feature dim mismatch for prompt! Got {len(x)}")
                 continue
                 
            X.append(x)
            valid_prompts.append(prompt)
        except Exception as e:
            print(f"Error processing prompt: {e}")
            
    X = np.array(X)
    print(f"Feature Matrix Shape: {X.shape}")
    
    # 4. Analyze P-Values for Top Models
    # Identify top models
    model_wins = {}
    for prompt in valid_prompts:
        rewards = train_rewards[prompt]
        best_model = max(rewards, key=rewards.get)
        model_wins[best_model] = model_wins.get(best_model, 0) + 1
            
    top_models = sorted(model_wins.items(), key=lambda x: x[1], reverse=True)[:5]
    targets = [m[0] for m in top_models]
    
    print(f"\nAnalyzing P-Values for Top 5 Models: {targets}")
    
    for model_id in targets:
        print(f"\n=== Model: {model_id} ===")
        
        y = []
        X_model = []
        
        for i, prompt in enumerate(valid_prompts):
            rewards = train_rewards[prompt]
            if model_id in rewards:
                y.append(rewards[model_id])
                X_model.append(X[i])
                
        y = np.array(y)
        X_model = np.array(X_model)
        
        if len(y) < 50:
            print("Insufficient data.")
            continue
            
        try:
            # Statsmodels OLS
            # Note: Bias is already in X_model (last column), so we don't need sm.add_constant if we handle it correctly
            # But normally sm.OLS expects explicit constant. 
            # Our feature vector has bias at index -1 (1.0). 
            # Let's use it as is.
            
            # Create DataFrame for better labels
            df_X = pd.DataFrame(X_model, columns=feature_names)
            
            est = sm.OLS(y, df_X)
            est2 = est.fit()
            
            print(est2.summary().tables[1])
            
            # Highlight significant features (p < 0.05)
            p_values = est2.pvalues
            significant = p_values[p_values < 0.05]
            if not significant.empty:
                print("\nSignificant Features (p < 0.05):")
                print(significant.sort_values())
            else:
                print("\nNo significant features found (p < 0.05).")
                
        except Exception as e:
            print(f"OLS Error: {e}")

if __name__ == "__main__":
    analyze_p_values()
