#!/usr/bin/env python3
"""
Predict Accuracy Using NVIDIA Classifier Features + CRS Score

This script tests whether NVIDIA prompt complexity features combined with CRS scores
can predict model accuracy on reasoning tasks.

Features used:
- is_complex (boolean from NVIDIA)
- is_reasoning_heavy (boolean from NVIDIA)  
- complexity_level (categorical: trivial/simple/moderate/complex/expert)
- CRS score (Composite Reasoning Score from BLF)

Target:
- Model accuracy on ARC-Challenge dataset
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier


def load_arc_challenge_prompts(n_samples: int = 50, seed: int = 42) -> List[Dict]:
    """Load ARC-Challenge prompts for classification."""
    print("\n📚 Loading ARC-Challenge Dataset...")
    
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    # Sample problems
    np.random.seed(seed)
    indices = np.random.choice(len(arc_challenge), size=min(n_samples, len(arc_challenge)), replace=False)
    
    problems = []
    for idx in indices:
        item = arc_challenge[int(idx)]
        
        # Format as multiple choice question
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D)."
        
        problems.append({
            'problem_id': item['id'],
            'prompt': prompt,
            'question': item['question']
        })
    
    print(f"   Loaded {len(problems)} ARC-Challenge problems")
    return problems


def classify_prompts_with_nvidia(prompts: List[str]) -> pd.DataFrame:
    """Classify prompts using NVIDIA classifier and extract features."""
    print("\n🤖 Classifying prompts with NVIDIA complexity classifier...")
    
    classifier = NvidiaComplexityClassifier()
    results = classifier.classify_batch(prompts)
    
    # Extract features into dataframe
    data = []
    for result in results:
        data.append({
            'prompt': result.prompt,
            'is_complex': result.is_complex,
            'is_reasoning_heavy': result.is_reasoning_heavy,
            'complexity_level': result.complexity_level,
            'task_type': result.task_type_1,
            'prompt_complexity_score': result.prompt_complexity_score,
            'reasoning_score': result.reasoning,
            'creativity_scope': result.creativity_scope,
            'constraint_ct': result.constraint_ct,
            'domain_knowledge': result.domain_knowledge,
            'contextual_knowledge': result.contextual_knowledge,
        })
    
    df = pd.DataFrame(data)
    
    print(f"   Classified {len(df)} prompts")
    print(f"   Complex prompts: {df['is_complex'].sum()} ({df['is_complex'].mean()*100:.1f}%)")
    print(f"   Reasoning-heavy: {df['is_reasoning_heavy'].sum()} ({df['is_reasoning_heavy'].mean()*100:.1f}%)")
    print(f"   Complexity levels: {df['complexity_level'].value_counts().to_dict()}")
    
    return df


def load_accuracy_data() -> pd.DataFrame:
    """Load model accuracy data from ARC validation results."""
    results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
    
    print(f"\n📊 Loading accuracy data from {results_path.name}...")
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("   Please run arc_easy_vs_challenge_validation.py first")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    models = []
    for model in data['models']:
        models.append({
            'model_name': model['name'],
            'crs_score': model['crs_score'],
            'crs_rank': model['crs_rank'],
            'challenge_accuracy': model['challenge_accuracy'],
            'easy_accuracy': model['easy_accuracy'],
            'accuracy_gap': model['accuracy_gap'],
        })
    
    df = pd.DataFrame(models)
    print(f"   Loaded {len(df)} models")
    print(f"   CRS range: {df['crs_score'].min():.2f} to {df['crs_score'].max():.2f}")
    print(f"   Accuracy range: {df['challenge_accuracy'].min():.1f}% to {df['challenge_accuracy'].max():.1f}%")
    
    return df


def build_feature_matrix(nvidia_features: pd.DataFrame, accuracy_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix combining NVIDIA features with CRS scores.
    
    Since NVIDIA features are prompt-level but accuracy is model-level,
    we'll aggregate prompt features (e.g., average complexity).
    """
    print("\n🔧 Building feature matrix...")
    
    # Aggregate NVIDIA features (these are properties of the task set)
    prompt_aggregates = {
        'avg_complexity_score': nvidia_features['prompt_complexity_score'].mean(),
        'pct_complex': nvidia_features['is_complex'].mean(),
        'pct_reasoning_heavy': nvidia_features['is_reasoning_heavy'].mean(),
        'avg_reasoning_score': nvidia_features['reasoning_score'].mean(),
        'avg_domain_knowledge': nvidia_features['domain_knowledge'].mean(),
    }
    
    print(f"\n   Task Set Characteristics (ARC-Challenge):")
    print(f"   - Avg complexity: {prompt_aggregates['avg_complexity_score']:.3f}")
    print(f"   - % Complex: {prompt_aggregates['pct_complex']*100:.1f}%")
    print(f"   - % Reasoning-heavy: {prompt_aggregates['pct_reasoning_heavy']*100:.1f}%")
    
    # For this analysis, we'll use CRS score as the main predictor
    # and examine whether adding prompt complexity improves prediction
    
    # Create feature dataframe (one row per model)
    features_df = accuracy_data.copy()
    
    # Add prompt aggregate features (same for all models on same task)
    for key, value in prompt_aggregates.items():
        features_df[key] = value
    
    print(f"\n   Feature matrix: {features_df.shape[0]} models x {features_df.shape[1]} features")
    
    return features_df


def train_accuracy_predictor(df: pd.DataFrame):
    """Train models to predict accuracy from features."""
    print("\n" + "="*80)
    print("TRAINING ACCURACY PREDICTION MODELS")
    print("="*80)
    
    # Define feature sets
    feature_sets = {
        'CRS Only': ['crs_score'],
        'CRS + Task Complexity': ['crs_score', 'avg_complexity_score'],
        'CRS + All NVIDIA Features': [
            'crs_score', 
            'avg_complexity_score', 
            'pct_complex', 
            'pct_reasoning_heavy',
            'avg_reasoning_score',
            'avg_domain_knowledge'
        ],
    }
    
    target = 'challenge_accuracy'
    
    results = []
    
    for feature_set_name, features in feature_sets.items():
        print(f"\n{'─'*80}")
        print(f"Feature Set: {feature_set_name}")
        print(f"Features: {', '.join(features)}")
        print(f"{'─'*80}")
        
        X = df[features].values
        y = df[target].values
        
        # Correlations (for single feature sets)
        if len(features) == 1:
            spearman_r, spearman_p = spearmanr(X.flatten(), y)
            pearson_r, pearson_p = pearsonr(X.flatten(), y)
            print(f"\n📊 Correlation with {target}:")
            print(f"   Spearman ρ: {spearman_r:+.3f} (p={spearman_p:.4f})")
            print(f"   Pearson r:  {pearson_r:+.3f} (p={pearson_p:.4f})")
        
        # Try multiple model types
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=2),
        }
        
        print(f"\n🤖 Model Performance (5-fold CV):")
        print(f"   {'Model':<20} {'R² Score':<12} {'MAE':<10} {'RMSE':<10}")
        print(f"   {'-'*20} {'-'*12} {'-'*10} {'-'*10}")
        
        for model_name, model in models.items():
            # Cross-validation
            cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
            
            # R² score
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            r2_mean = r2_scores.mean()
            
            # MAE
            mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            mae_mean = mae_scores.mean()
            
            # RMSE
            mse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            rmse_mean = np.sqrt(mse_scores.mean())
            
            print(f"   {model_name:<20} {r2_mean:>6.3f} ± {r2_scores.std():.3f}  {mae_mean:>6.2f}%  {rmse_mean:>7.2f}%")
            
            # Train final model on all data for feature importance
            model.fit(X, y)
            
            results.append({
                'feature_set': feature_set_name,
                'model': model_name,
                'r2_score': r2_mean,
                'mae': mae_mean,
                'rmse': rmse_mean,
                'fitted_model': model,
                'features': features,
            })
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                print(f"\n   Feature Importances for {model_name}:")
                importances = model.feature_importances_
                for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
                    print(f"      {feat:<30} {imp:.3f}")
    
    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY: Best Models by Feature Set")
    print(f"{'='*80}")
    
    # Get best model for each feature set
    feature_set_results = {}
    for result in results:
        fs = result['feature_set']
        if fs not in feature_set_results or result['r2_score'] > feature_set_results[fs]['r2_score']:
            feature_set_results[fs] = result
    
    print(f"\n{'Feature Set':<30} {'Best Model':<20} {'R² Score':<12} {'MAE':<10}")
    print(f"{'-'*30} {'-'*20} {'-'*12} {'-'*10}")
    
    for fs_name in feature_sets.keys():
        if fs_name in feature_set_results:
            r = feature_set_results[fs_name]
            print(f"{fs_name:<30} {r['model']:<20} {r['r2_score']:>6.3f}       {r['mae']:>6.2f}%")
    
    return results


def main():
    print("="*80)
    print("PREDICT MODEL ACCURACY USING NVIDIA FEATURES + CRS SCORE")
    print("="*80)
    
    # Load ARC-Challenge prompts
    arc_problems = load_arc_challenge_prompts(n_samples=50, seed=42)
    prompts = [p['prompt'] for p in arc_problems]
    
    # Classify with NVIDIA
    nvidia_features = classify_prompts_with_nvidia(prompts)
    
    # Load accuracy data
    accuracy_data = load_accuracy_data()
    
    # Build feature matrix
    features_df = build_feature_matrix(nvidia_features, accuracy_data)
    
    # Train predictive models
    results = train_accuracy_predictor(features_df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Examined whether NVIDIA prompt features improve accuracy prediction")
    print("2. Compared CRS-only vs CRS+NVIDIA features")
    print("3. Evaluated multiple regression models (Linear, Ridge, RF, GBM)")
    
    # Save results
    output_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "nvidia_accuracy_prediction.json"
    
    results_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'task': 'ARC-Challenge',
        'n_models': len(accuracy_data),
        'n_prompts_analyzed': len(nvidia_features),
        'prompt_characteristics': {
            'avg_complexity': float(nvidia_features['prompt_complexity_score'].mean()),
            'pct_complex': float(nvidia_features['is_complex'].mean()),
            'pct_reasoning_heavy': float(nvidia_features['is_reasoning_heavy'].mean()),
        },
        'models': [
            {
                'feature_set': r['feature_set'],
                'model': r['model'],
                'r2_score': float(r['r2_score']),
                'mae': float(r['mae']),
                'rmse': float(r['rmse']),
                'features': r['features'],
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
