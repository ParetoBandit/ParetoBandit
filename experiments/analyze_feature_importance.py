
import sys
import os
import json
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Add src to path
sys.path.append(str(Path.cwd() / "src"))
sys.path.append(str(Path.cwd() / "experiments"))

from bandit_gpt.router import BanditRouter, RouterConfig
from utils.data_loader import load_oracle_rewards

def analyze_features():
    print("🚀 Starting Feature Importance Analysis...")
    
    # 1. Initialize Router (to get feature extraction logic)
    print("Initializing BanditRouter for feature extraction...")
    # Use priors="none" to avoid overhead, we just need feature logic
    router = BanditRouter.create(priors="none")
    
    # 2. Load Training Data
    print("Loading training data...")
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    prompts = list(train_rewards.keys())
    print(f"Loaded {len(prompts)} prompts")
    
    # 3. Extract Features for All Prompts
    print("Extracting features (this may take a minute)...")
    X = []
    
    # We need feature names for interpretation
    # 32 dims (PCA) + 14 handcrafted + 5 anchors + 1 complexity + 1 bias
    feature_names = [f"pca_{i}" for i in range(32)]
    
    # Handcrafted (from router.py log)
    # 1-6 are continuous/original extracted
    # 7-14 are linearized pairs
    # Wait, router._get_context_vector calls _extract_handcrafted_features
    # Let's inspect the actual router logic to match names perfectly
    hc_names = [
        "is_code_heavy", "requires_json", "list_density", "instruction_density", 
        "flesch_kincaid", "toxicity_score", 
        "has_code_block", "code_block_count_log",
        "has_latex", "latex_density_log",
        "has_question", "question_count_log",
        "length_penalty_bin", "length_penalty_log"
    ]
    feature_names.extend(hc_names)
    
    anchor_names = ["anchor_coding", "anchor_math", "anchor_creative", "anchor_jokes", "anchor_reasoning"]
    feature_names.extend(anchor_names)
    
    feature_names.append("complexity_score")
    feature_names.append("bias")
    
    print(f"Feature Names ({len(feature_names)}): {feature_names}")

    # Extract X matrix
    valid_prompts = []
    for prompt in tqdm(prompts):
        try:
            x, _ = router._build_routing_features(prompt)
            if len(x) != len(feature_names):
                print(f"WARNING: Dimension mismatch! Got {len(x)}, expected {len(feature_names)}")
                # Adjust names if needed based on runtime check
                if len(x) > len(feature_names):
                    diff = len(x) - len(feature_names)
                    feature_names.extend([f"unknown_{i}" for i in range(diff)])
            X.append(x)
            valid_prompts.append(prompt)
        except Exception as e:
            print(f"Error processing prompt: {e}")
            
    X = np.array(X)
    print(f"Feature Matrix Shape: {X.shape}")
    
    # 4. Analyze Feature Importance for Top Models
    # Identify top models (most available data/wins)
    model_wins = {}
    model_data_count = {}
    
    for prompt in valid_prompts:
        rewards = train_rewards[prompt]
        best_model = max(rewards, key=rewards.get)
        model_wins[best_model] = model_wins.get(best_model, 0) + 1
        for m in rewards:
            model_data_count[m] = model_data_count.get(m, 0) + 1
            
    top_models = sorted(model_wins.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 Models by Wins: {top_models}")
    
    targets = [m[0] for m in top_models]
    
    results = {}
    
    for model_id in targets:
        print(f"\nAnalyzing model: {model_id}")
        
        # Build Y vector (rewards)
        # Filter X to only prompts where we have data for this model
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
            print(f"Skipping {model_id} - insufficient data ({len(y)} samples)")
            continue
            
        # Normalize features for fair coefficient comparison
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_model)
        
        # Train Lasso (L1) to enforce sparsity
        # alpha=0.01 is a reasonable starting point for sparsity
        lasso = Lasso(alpha=0.001, max_iter=10000) 
        lasso.fit(X_scaled, y)
        
        # Get coefficients
        coefs = pd.Series(lasso.coef_, index=feature_names)
        
        # Sort by absolute value
        important = coefs.abs().sort_values(ascending=False).head(10)
        print(f"Top 10 Features for {model_id}:")
        print(important)
        
        results[model_id] = coefs
        
    # Aggregate results: Average Absolute Coefficient across models
    print("\n\n=== GLOBAL FEATURE IMPORTANCE (Average Abs Coef) ===")
    df = pd.DataFrame(results)
    df_abs = df.abs()
    mean_imp = df_abs.mean(axis=1).sort_values(ascending=False)
    
    print(mean_imp)
    
    # List features with zero importance (candidates for removal)
    zero_imp = mean_imp[mean_imp < 1e-4]
    print(f"\n\n=== CANDIDATES FOR REMOVAL (Zero Importance) ===")
    print(zero_imp.index.tolist())

if __name__ == "__main__":
    analyze_features()
