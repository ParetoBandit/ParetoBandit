#!/usr/bin/env python3
"""
Feature Significance Analysis with Interaction Terms

Uses polynomial features (degree=2) to capture pairwise interactions
and Lasso regression for automatic feature selection.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score

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
    
    return feature_names

def analyze_with_interactions():
    print("=" * 80)
    print("FEATURE SIGNIFICANCE ANALYSIS WITH INTERACTIONS")
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
    
    # 3. Initialize router to extract features
    print("\n🔧 Initializing router for feature extraction...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors="warmup"
    )
    
    base_feature_names = get_feature_names(router)
    print(f"  ✓ Base features: {len(base_feature_names)}")
    
    # 4. Extract features for dev prompts
    print(f"\n🔍 Extracting features for {len(dev_prompts)} dev prompts...")
    features_matrix = []
    for prompt in dev_prompts:
        context_vec = router._get_context_vector(prompt)
        features_matrix.append(context_vec)
    
    X_base = np.array(features_matrix)[:, :-1]  # Remove bias term
    print(f"  ✓ Base feature matrix: {X_base.shape}")
    
    # 5. Generate interaction terms
    print("\n🔗 Generating interaction terms...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_interactions = poly.fit_transform(X_base)
    
    # Get feature names including interactions
    # base_feature_names already has 28 items (no bias), matching X_base columns
    print(f"  DEBUG: len(base_feature_names) = {len(base_feature_names)}")
    print(f"  DEBUG: X_base.shape[1] = {X_base.shape[1]}")
    print(f"  DEBUG: base_feature_names = {base_feature_names}")
    interaction_feature_names = poly.get_feature_names_out(base_feature_names)
    
    print(f"  ✓ With interactions: {X_interactions.shape[1]} features")
    print(f"  ✓ Added {X_interactions.shape[1] - X_base.shape[1]} interaction terms")
    
    # 6. Run Lasso regression for each model
    print("\n📊 Running Lasso regression per model...")
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
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_interactions)
        
        # Use cross-validated Lasso for automatic alpha selection
        print(f"\n{model_id}")
        print(f"  Running cross-validated Lasso...")
        
        lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=5000)
        lasso_cv.fit(X_scaled, y)
        
        print(f"  ✓ Optimal alpha: {lasso_cv.alpha_:.6f}")
        print(f"  ✓ R² score (CV): {lasso_cv.score(X_scaled, y):.4f}")
        
        # Count non-zero coefficients
        non_zero = np.sum(lasso_cv.coef_ != 0)
        print(f"  ✓ Selected features: {non_zero}/{len(interaction_feature_names)}")
        
        # Get significant features (non-zero coefficients)
        for i, (feat_name, coef) in enumerate(zip(interaction_feature_names, lasso_cv.coef_)):
            if coef != 0:
                all_results.append({
                    'model': model_id,
                    'feature': feat_name,
                    'coefficient': coef,
                    'abs_coefficient': abs(coef),
                    'is_interaction': ' ' in feat_name  # Interaction terms have space
                })
        
        # Report top features
        top_features = [(interaction_feature_names[i], lasso_cv.coef_[i]) 
                       for i in range(len(lasso_cv.coef_)) if lasso_cv.coef_[i] != 0]
        
        if top_features:
            # Sort by absolute coefficient value
            top_features.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n  Top 10 selected features:")
            for feat_name, coef in top_features[:10]:
                feat_type = "INTERACTION" if ' ' in feat_name else "MAIN      "
                print(f"    [{feat_type}] {feat_name[:50]:<50s}: {coef:+.5f}")
    
    # 7. Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)
    
    df = pd.DataFrame(all_results)
    
    # Separate main effects and interactions
    df['is_interaction'] = df['feature'].str.contains(' ')
    
    main_effects = df[~df['is_interaction']]
    interactions = df[df['is_interaction']]
    
    print(f"\n📊 Summary:")
    print(f"  Total selected features: {len(df)}")
    print(f"  Main effects: {len(main_effects)}")
    print(f"  Interaction terms: {len(interactions)}")
    
    # Most frequently selected features
    print("\n🔝 Most frequently selected features (across models):")
    print("-" * 80)
    
    feature_freq = df['feature'].value_counts().head(20)
    print(f"\n{'Feature':<60s} | {'Frequency'}")
    print("-" * 80)
    
    for feat, count in feature_freq.items():
        feat_type = "[INT]" if ' ' in feat else "[MAIN]"
        display_name = feat if len(feat) <= 50 else feat[:47] + "..."
        print(f"{feat_type} {display_name:<55s} | {count}")
    
    # Top interaction terms by coefficient magnitude
    print("\n🔗 Strongest interaction terms (by |coefficient|):")
    print("-" * 80)
    
    top_interactions = interactions.nlargest(15, 'abs_coefficient')
    print(f"\n{'Model':<30s} | {'Interaction':<40s} | {'Coefficient'}")
    print("-" * 100)
    
    for _, row in top_interactions.iterrows():
        interaction_name = row['feature'] if len(row['feature']) <= 40 else row['feature'][:37] + "..."
        print(f"{row['model']:<30s} | {interaction_name:<40s} | {row['coefficient']:+.5f}")
    
    # 8. Save results
    output_path = Path(__file__).parent / "results" / "feature_significance_interactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved detailed results to {output_path}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_with_interactions()
