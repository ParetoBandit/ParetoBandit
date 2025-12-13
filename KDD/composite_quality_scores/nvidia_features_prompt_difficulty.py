#!/usr/bin/env python3
"""
NVIDIA Features as Predictors of Prompt Difficulty

This script examines whether NVIDIA complexity features can predict which prompts
are harder for models to answer correctly.

Approach:
- For each prompt, calculate: % of models that got it right
- Use NVIDIA features to predict this "prompt difficulty"
- Combine with model CRS to predict individual model-prompt success

This addresses the limitation that NVIDIA features are prompt-level while
CRS/accuracy are model-level.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier


def load_arc_responses() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load individual model-prompt responses from ARC validation.
    
    Returns:
        (DataFrame with columns: model_name, crs_score, problem_id, is_correct, prompt_text),
         List of unique problem_ids
    """
    results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
    
    print(f"\n📊 Loading individual model-prompt responses...")
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Flatten into individual responses
    responses = []
    problem_texts = {}
    
    for model in data['models']:
        for response in model['responses']:
            if response['difficulty'] == 'challenge':  # Focus on challenge problems
                responses.append({
                    'model_name': model['name'],
                    'crs_score': model['crs_score'],
                    'problem_id': response['problem_id'],
                    'is_correct': response['is_correct'],
                })
    
    df = pd.DataFrame(responses)
    
    print(f"   ✓ Loaded {len(df)} model-prompt pairs")
    print(f"   ✓ {len(df['model_name'].unique())} models")
    print(f"   ✓ {len(df['problem_id'].unique())} prompts")
    
    unique_problems = df['problem_id'].unique().tolist()
    
    return df, unique_problems


def load_arc_problem_texts(problem_ids: List[str]) -> Dict[str, str]:
    """Load the actual text for ARC problems."""
    print(f"\n📚 Loading ARC problem texts...")
    
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    problem_texts = {}
    for item in arc_challenge:
        problem_id = f"ARC-CHALLENGE/{item['id']}"
        
        if problem_id in problem_ids:
            # Format as multiple choice
            prompt = f"{item['question']}\n\nOptions:\n"
            for label, text in zip(item['choices']['label'], item['choices']['text']):
                prompt += f"{label}. {text}\n"
            
            problem_texts[problem_id] = prompt
    
    print(f"   ✓ Loaded {len(problem_texts)} problem texts")
    
    return problem_texts


def classify_prompts_with_nvidia(problem_texts: Dict[str, str]) -> pd.DataFrame:
    """Classify prompts using NVIDIA classifier."""
    print(f"\n🤖 Classifying prompts with NVIDIA complexity classifier...")
    
    classifier = NvidiaComplexityClassifier()
    
    problem_ids = list(problem_texts.keys())
    prompts = [problem_texts[pid] for pid in problem_ids]
    
    results = classifier.classify_batch(prompts)
    
    # Build dataframe
    data = []
    for problem_id, result in zip(problem_ids, results):
        data.append({
            'problem_id': problem_id,
            'is_complex': result.is_complex,
            'is_reasoning_heavy': result.is_reasoning_heavy,
            'complexity_level': result.complexity_level,
            'task_type': result.task_type_1,
            'prompt_complexity_score': result.prompt_complexity_score,
            'reasoning_score': result.reasoning,
            'creativity_scope': result.creativity_scope,
            'domain_knowledge': result.domain_knowledge,
        })
    
    df = pd.DataFrame(data)
    
    print(f"   ✓ Classified {len(df)} prompts")
    print(f"   - Complex: {df['is_complex'].sum()} ({df['is_complex'].mean()*100:.1f}%)")
    print(f"   - Reasoning-heavy: {df['is_reasoning_heavy'].sum()} ({df['is_reasoning_heavy'].mean()*100:.1f}%)")
    
    return df


def analyze_prompt_difficulty(responses_df: pd.DataFrame, nvidia_df: pd.DataFrame):
    """Analyze which prompts are hardest and whether NVIDIA features predict difficulty."""
    print(f"\n" + "="*80)
    print("NVIDIA FEATURES AS PREDICTORS OF PROMPT DIFFICULTY")
    print("="*80)
    
    # Calculate difficulty for each prompt (% of models that got it wrong)
    prompt_stats = responses_df.groupby('problem_id').agg({
        'is_correct': ['sum', 'count', 'mean']
    }).reset_index()
    
    prompt_stats.columns = ['problem_id', 'n_correct', 'n_total', 'pct_correct']
    prompt_stats['difficulty'] = 1 - prompt_stats['pct_correct']  # Higher = harder
    
    # Merge with NVIDIA features
    merged = prompt_stats.merge(nvidia_df, on='problem_id')
    
    print(f"\n📊 Prompt Difficulty Statistics:")
    print(f"   Easiest prompt: {merged['pct_correct'].max()*100:.0f}% models correct")
    print(f"   Hardest prompt: {merged['pct_correct'].min()*100:.0f}% models correct")
    print(f"   Average: {merged['pct_correct'].mean()*100:.1f}% models correct")
    
    # Correlation between NVIDIA features and difficulty
    print(f"\n📊 Correlation: NVIDIA Features vs Prompt Difficulty")
    print(f"   {'Feature':<30} {'Spearman ρ':<12} {'p-value':<12} {'Interpretation'}")
    print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*30}")
    
    features_to_test = [
        ('prompt_complexity_score', 'Complexity Score'),
        ('reasoning_score', 'Reasoning Score'),
        ('domain_knowledge', 'Domain Knowledge'),
        ('creativity_scope', 'Creativity'),
    ]
    
    correlations = {}
    for feat_col, feat_name in features_to_test:
        rho, p = spearmanr(merged[feat_col], merged['difficulty'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        interp = "Higher complexity → Harder" if rho > 0.3 else "Weak/No relationship"
        print(f"   {feat_name:<30} {rho:>+6.3f} {sig:<5}  {p:>6.4f}       {interp}")
        correlations[feat_col] = rho
    
    # Boolean features
    print(f"\n   Boolean Features:")
    for bool_feat in ['is_complex', 'is_reasoning_heavy']:
        complex_prompts = merged[merged[bool_feat] == True]
        simple_prompts = merged[merged[bool_feat] == False]
        
        if len(complex_prompts) > 0 and len(simple_prompts) > 0:
            avg_diff_complex = complex_prompts['difficulty'].mean()
            avg_diff_simple = simple_prompts['difficulty'].mean()
            
            print(f"   {bool_feat:<30} Complex: {avg_diff_complex:.3f}, Simple: {avg_diff_simple:.3f}, Δ: {avg_diff_complex - avg_diff_simple:+.3f}")
    
    return merged, correlations


def predict_individual_success(responses_df: pd.DataFrame, nvidia_df: pd.DataFrame):
    """
    Build a model to predict whether a specific model will answer a specific prompt correctly.
    
    Features: model CRS + prompt NVIDIA features
    Target: is_correct (binary)
    """
    print(f"\n" + "="*80)
    print("PREDICTING INDIVIDUAL MODEL-PROMPT SUCCESS")
    print("="*80)
    
    # Merge responses with NVIDIA features
    data = responses_df.merge(nvidia_df, on='problem_id')
    
    print(f"\n📊 Dataset: {len(data)} model-prompt pairs")
    print(f"   Success rate: {data['is_correct'].mean()*100:.1f}%")
    
    # Encode complexity_level
    le = LabelEncoder()
    data['complexity_level_encoded'] = le.fit_transform(data['complexity_level'])
    
    # Define feature sets
    feature_sets = {
        'CRS Only': ['crs_score'],
        'NVIDIA Only': ['prompt_complexity_score', 'reasoning_score', 'domain_knowledge', 
                        'is_complex', 'is_reasoning_heavy', 'complexity_level_encoded'],
        'CRS + NVIDIA': ['crs_score', 'prompt_complexity_score', 'reasoning_score', 
                        'domain_knowledge', 'is_complex', 'is_reasoning_heavy', 
                        'complexity_level_encoded'],
    }
    
    results = []
    
    for fs_name, features in feature_sets.items():
        print(f"\n{'─'*80}")
        print(f"Feature Set: {fs_name}")
        print(f"{'─'*80}")
        
        X = data[features].values
        y = data['is_correct'].values
        
        # Try classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3),
        }
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'ROC-AUC':<12}")
        print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
        
        for clf_name, clf in classifiers.items():
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
            auc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            
            acc_mean = acc_scores.mean()
            f1_mean = f1_scores.mean()
            auc_mean = auc_scores.mean()
            
            print(f"{clf_name:<25} {acc_mean:.3f} ± {acc_scores.std():.3f}  {f1_mean:.3f} ± {f1_scores.std():.3f}  {auc_mean:.3f} ± {auc_scores.std():.3f}")
            
            results.append({
                'feature_set': fs_name,
                'classifier': clf_name,
                'accuracy': acc_mean,
                'f1': f1_mean,
                'auc': auc_mean,
            })
            
            # Feature importance (for tree-based models trained on full data)
            if hasattr(clf, 'feature_importances_'):
                clf.fit(X, y)
                print(f"\n   Top Features for {clf_name}:")
                importances = clf.feature_importances_
                for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:5]:
                    print(f"      {feat:<30} {imp:.3f}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY: Best Performance by Feature Set")
    print(f"{'='*80}")
    
    best_by_fs = {}
    for result in results:
        fs = result['feature_set']
        if fs not in best_by_fs or result['auc'] > best_by_fs[fs]['auc']:
            best_by_fs[fs] = result
    
    print(f"\n{'Feature Set':<20} {'Classifier':<25} {'Accuracy':<12} {'AUC':<12}")
    print(f"{'-'*20} {'-'*25} {'-'*12} {'-'*12}")
    
    for fs_name in feature_sets.keys():
        if fs_name in best_by_fs:
            r = best_by_fs[fs_name]
            print(f"{fs_name:<20} {r['classifier']:<25} {r['accuracy']:.3f}       {r['auc']:.3f}")
    
    return results


def main():
    print("="*80)
    print("NVIDIA FEATURES FOR PREDICTING PROMPT DIFFICULTY & MODEL SUCCESS")
    print("="*80)
    
    # Load responses
    responses_df, problem_ids = load_arc_responses()
    
    # Load problem texts
    problem_texts = load_arc_problem_texts(problem_ids)
    
    # Classify with NVIDIA
    nvidia_df = classify_prompts_with_nvidia(problem_texts)
    
    # Analyze prompt difficulty
    prompt_analysis, correlations = analyze_prompt_difficulty(responses_df, nvidia_df)
    
    # Predict individual success
    prediction_results = predict_individual_success(responses_df, nvidia_df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n📊 Key Findings:")
    print(f"1. Examined {len(nvidia_df)} prompts with NVIDIA complexity features")
    print(f"2. Analyzed {len(responses_df)} model-prompt interactions")
    print(f"3. Tested whether NVIDIA features predict prompt difficulty")
    print(f"4. Built models combining CRS + NVIDIA features to predict success")
    
    # Save results
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    output_path = output_dir / "nvidia_prompt_difficulty_analysis.json"
    
    results_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_prompts': len(nvidia_df),
        'n_models': len(responses_df['model_name'].unique()),
        'n_interactions': len(responses_df),
        'nvidia_feature_correlations': {k: float(v) for k, v in correlations.items()},
        'prediction_results': prediction_results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
