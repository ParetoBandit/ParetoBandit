#!/usr/bin/env python3
"""
Feature Significance Analysis

Analyze which features are statistically significant predictors of reward
using linear regression on the dev training data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from sentence_transformers import SentenceTransformer
from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def get_feature_names(router):
    """Get human-readable feature names from the router."""
    feature_names = []
    
    # PCA components
    if router.pca:
        n_components = router.pca.n_components
        feature_names.extend([f"pca_{i}" for i in range(n_components)])
    else:
        feature_names.extend([f"emb_{i}" for i in range(384)])
    
    # Handcrafted features (14)
    handcrafted = [
        "is_code_heavy", "requires_json", "list_density", "instruction_density",
        "flesch_kincaid", "toxicity_score",
        "has_code_block", "code_block_count_log",
        "has_latex", "latex_density_log",
        "has_question", "question_count_log",
        "length_penalty_bin", "length_penalty_log"
    ]
    feature_names.extend(handcrafted)
    
    # Virtual anchors (5)
    anchor_names = ["anchor_coding", "anchor_math", "anchor_creative", "anchor_jokes", "anchor_reasoning"]
    feature_names.extend(anchor_names)
    
    # Hardness score (1)
    feature_names.append("hardness_score")
    
    # Bias term (1)
    feature_names.append("bias")
    
    return feature_names

def analyze_feature_significance():
    print("=" * 80)
    print("FEATURE SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    # 1. Load data
    print("\n📦 Loading data...")
    registry = load_model_registry()
    train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    full_corpus = {**train_rewards, **test_rewards}
    
    # 2. Load dev splits
    splits_path = Path(__file__).parent / "results" / "splits.json"
    if not splits_path.exists():
        print(f"❌ Error: {splits_path} not found. Run run_budget_experiment.py first.")
        return
    
    print(f"📂 Loading splits from {splits_path}")
    burn_in_helper = ExperimentBurnIn(
        registry=registry,
        oracle_rewards=full_corpus,
        splits_path=splits_path,
        encoder=None
    )
    
    dev_prompts, holdout_prompts = burn_in_helper.get_splits()
    print(f"  ✓ Dev prompts: {len(dev_prompts)}")
    print(f"  ✓ Holdout prompts: {len(holdout_prompts)}")
    
    # 3. Initialize router to extract features
    print("\n🔧 Initializing router for feature extraction...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors="warmup"
    )
    
    feature_names = get_feature_names(router)
    n_features = len(feature_names)
    print(f"  ✓ Feature dimension: {n_features}")
    
    # 4. Extract features for dev prompts
    print(f"\n🔍 Extracting features for {len(dev_prompts)} dev prompts...")
    features_matrix = []
    for prompt in dev_prompts:
        context_vec = router._get_context_vector(prompt)
        features_matrix.append(context_vec)
    
    X = np.array(features_matrix)  # Shape: (n_prompts, n_features)
    print(f"  ✓ Feature matrix shape: {X.shape}")
    
    # 5. Run regression for each model
    print("\n📊 Running regression analysis per model...")
    print("-" * 80)
    
    model_ids = list(registry.keys())
    
    all_results = []
    
    for model_id in model_ids:
        # Extract rewards for this model
        y = []
        for prompt in dev_prompts:
            reward = full_corpus.get(prompt, {}).get(model_id, 0.0)
            y.append(reward)
        
        y = np.array(y)
        
        # Skip if all zeros or very low variance
        if y.sum() == 0 or y.std() < 1e-6:
            print(f"⚠️  Skipping {model_id}: no/low variance in rewards")
            continue
        
        # Remove bias term (last feature) to avoid singularity
        X_no_bias = X[:, :-1]
        feature_names_no_bias = feature_names[:-1]
        
        # Standardize features (important for interpretability)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_no_bias)
        
        # Add constant term for intercept (via LinearRegression fit_intercept=True)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_scaled, y)
        
        # Calculate p-values using t-test
        n_samples = X_scaled.shape[0]
        n_predictors = X_scaled.shape[1]
        
        # Predictions and residuals
        y_pred = reg.predict(X_scaled)
        residuals = y - y_pred
        
        # Mean squared error of residuals
        mse = np.sum(residuals**2) / (n_samples - n_predictors - 1)
        
        # Variance-covariance matrix
        try:
            X_with_intercept = np.column_stack([np.ones(n_samples), X_scaled])
            var_covar_matrix = mse * np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
            
            # Standard errors (skip intercept for reporting)
            se_coef = np.sqrt(var_covar_matrix.diagonal()[1:])
            
            # t-statistic
            t_stat = reg.coef_ / se_coef
            
            # p-values (two-tailed test)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), n_samples - n_predictors - 1))
        except np.linalg.LinAlgError:
            print(f"⚠️  Skipping {model_id}: matrix inversion failed")
            continue
        
        # Store results
        for i, (feat_name, coef, pval) in enumerate(zip(feature_names_no_bias, reg.coef_, p_values)):
            all_results.append({
                'model': model_id,
                'feature': feat_name,
                'coefficient': coef,
                'p_value': pval,
                'significant': pval < 0.05
            })
        
        # Report significant features for this model
        sig_features = [(feature_names_no_bias[i], reg.coef_[i], p_values[i]) 
                       for i in range(len(feature_names_no_bias)) if p_values[i] < 0.05]
        
        print(f"\n{model_id}")
        print(f"  R² score: {reg.score(X_scaled, y):.4f}")
        print(f"  Significant features (p < 0.05): {len(sig_features)}/{len(feature_names_no_bias)}")
        
        if sig_features:
            # Sort by absolute coefficient value
            sig_features.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"  Top 5 significant features:")
            for feat_name, coef, pval in sig_features[:5]:
                print(f"    {feat_name:25s}: coef={coef:+.4f}, p={pval:.4e}")
    
    # 6. Aggregate analysis: Which features are significant across models?
    print("\n" + "=" * 80)
    print("AGGREGATE FEATURE SIGNIFICANCE")
    print("=" * 80)
    
    df = pd.DataFrame(all_results)
    
    # Count how many models each feature is significant for
    feature_significance = df[df['significant']].groupby('feature').size().sort_values(ascending=False)
    
    print(f"\nFeatures significant (p < 0.05) across multiple models:")
    print("-" * 80)
    print(f"{'Feature':<25s} | {'# Models':<10s} | {'Frequency'}")
    print("-" * 80)
    
    for feat, count in feature_significance.items():
        freq = count / len(model_ids)
        print(f"{feat:<25s} | {count:<10d} | {freq:.1%}")
    
    # 7. Save detailed results
    output_path = Path(__file__).parent / "results" / "feature_significance.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved detailed results to {output_path}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_feature_significance()
