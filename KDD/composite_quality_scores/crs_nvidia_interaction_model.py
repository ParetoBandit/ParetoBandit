#!/usr/bin/env python3
"""
Interaction Model: CRS × Task Complexity for Performance Prediction

This script implements a hierarchical regression model to predict individual
model-prompt success rates using:
- Model features: CRS score
- Prompt features: NVIDIA complexity metrics
- Interaction terms: CRS × complexity

Key hypothesis: High-CRS models should be more robust to complex prompts
(positive interaction coefficient β₃ > 0)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier


def load_arc_responses() -> Tuple[pd.DataFrame, List[str]]:
    """Load individual model-prompt responses from ARC validation."""
    results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
    
    print(f"\n📊 Loading model-prompt responses...")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    responses = []
    for model in data['models']:
        for response in model['responses']:
            if response['difficulty'] == 'challenge':
                responses.append({
                    'model_name': model['name'],
                    'crs_score': model['crs_score'],
                    'problem_id': response['problem_id'],
                    'is_correct': response['is_correct'],
                })
    
    df = pd.DataFrame(responses)
    unique_problems = df['problem_id'].unique().tolist()
    
    print(f"   ✓ {len(df)} model-prompt pairs")
    print(f"   ✓ {len(df['model_name'].unique())} models")
    print(f"   ✓ {len(unique_problems)} unique prompts")
    
    return df, unique_problems


def load_arc_problem_texts(problem_ids: List[str]) -> Dict[str, str]:
    """Load the actual text for ARC problems."""
    print(f"\n📚 Loading ARC problem texts...")
    
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    problem_texts = {}
    for item in arc_challenge:
        problem_id = f"ARC-CHALLENGE/{item['id']}"
        if problem_id in problem_ids:
            prompt = f"{item['question']}\n\nOptions:\n"
            for label, text in zip(item['choices']['label'], item['choices']['text']):
                prompt += f"{label}. {text}\n"
            problem_texts[problem_id] = prompt
    
    print(f"   ✓ {len(problem_texts)} problem texts")
    return problem_texts


def classify_prompts_with_nvidia(problem_texts: Dict[str, str]) -> pd.DataFrame:
    """Classify prompts using NVIDIA classifier."""
    print(f"\n🤖 Classifying prompts with NVIDIA...")
    
    classifier = NvidiaComplexityClassifier()
    
    problem_ids = list(problem_texts.keys())
    prompts = [problem_texts[pid] for pid in problem_ids]
    results = classifier.classify_batch(prompts)
    
    data = []
    for problem_id, result in zip(problem_ids, results):
        data.append({
            'problem_id': problem_id,
            'prompt_complexity_score': result.prompt_complexity_score,
            'reasoning_score': result.reasoning,
            'creativity_scope': result.creativity_scope,
            'domain_knowledge': result.domain_knowledge,
            'constraint_ct': result.constraint_ct,
            'is_complex': int(result.is_complex),
            'is_reasoning_heavy': int(result.is_reasoning_heavy),
        })
    
    df = pd.DataFrame(data)
    print(f"   ✓ Classified {len(df)} prompts")
    
    return df


def check_multicollinearity(df: pd.DataFrame, feature_cols: List[str]):
    """Check for multicollinearity using correlation matrix and VIF."""
    
    print(f"\n🔍 MULTICOLLINEARITY DIAGNOSTICS")
    print(f"{'='*80}")
    
    X = df[feature_cols].copy()
    
    # 1. Correlation matrix
    print(f"\n📊 Correlation Matrix:")
    corr_matrix = X.corr()
    
    print(f"\n   {'Feature':<30} " + "  ".join([f"{col[:8]:<8}" for col in feature_cols]))
    print(f"   {'-'*30} " + "  ".join(["-"*8 for _ in feature_cols]))
    
    for i, row_name in enumerate(feature_cols):
        row_str = f"   {row_name:<30}"
        for j, col_name in enumerate(feature_cols):
            val = corr_matrix.iloc[i, j]
            row_str += f" {val:>7.3f} "
        print(row_str)
    
    # Find high correlations (excluding diagonal)
    print(f"\n   ⚠️  High Correlations (|r| > 0.7):")
    high_corr_found = False
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:
                print(f"      {feature_cols[i]:<30} ↔ {feature_cols[j]:<30} r = {corr_matrix.iloc[i, j]:+.3f}")
                high_corr_found = True
    
    if not high_corr_found:
        print(f"      ✓ No high correlations detected")
    
    # 2. Variance Inflation Factor (VIF)
    print(f"\n📊 Variance Inflation Factor (VIF):")
    print(f"   (VIF > 5 indicates multicollinearity, VIF > 10 is severe)")
    print(f"\n   {'Feature':<30} {'VIF':<10} {'Status'}")
    print(f"   {'-'*30} {'-'*10} {'-'*20}")
    
    vif_data = []
    X_values = X.values
    
    for i, col in enumerate(feature_cols):
        try:
            vif = variance_inflation_factor(X_values, i)
            vif_data.append({'feature': col, 'vif': vif})
            
            if np.isnan(vif) or np.isinf(vif):
                status = "⚠️  Unable to compute"
                vif_display = "N/A"
            else:
                vif_display = f"{vif:>6.2f}"
                if vif > 10:
                    status = "❌ SEVERE multicollinearity"
                elif vif > 5:
                    status = "⚠️  Moderate multicollinearity"
                elif vif > 2.5:
                    status = "⚡ Mild multicollinearity"
                else:
                    status = "✓ OK"
            
            print(f"   {col:<30} {vif_display:<10} {status}")
        except:
            print(f"   {col:<30} {'N/A':<10} ⚠️  Unable to compute")
            vif_data.append({'feature': col, 'vif': np.nan})
    
    # 3. Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    problematic_features = [v['feature'] for v in vif_data if not np.isnan(v['vif']) and not np.isinf(v['vif']) and v['vif'] > 5]
    
    if problematic_features:
        print(f"   ⚠️  Consider removing or combining these features:")
        for feat in problematic_features:
            print(f"      - {feat}")
        print(f"\n   Alternatives:")
        print(f"   1. Use only ONE of the correlated features")
        print(f"   2. Create a composite feature (PCA or average)")
        print(f"   3. Use regularization (Ridge/Lasso)")
    else:
        print(f"   ✓ No severe multicollinearity detected")
        print(f"   → Safe to proceed with standard linear regression")
    
    return corr_matrix, vif_data


def select_uncorrelated_features(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    """Select features with low correlation."""
    
    print(f"\n🎯 AUTO-SELECTING UNCORRELATED FEATURES (threshold = {threshold}):")
    
    # Start with all features
    features = corr_matrix.columns.tolist()
    selected = []
    
    # Greedy selection: keep features with lowest average correlation
    avg_corr = corr_matrix.abs().mean().sort_values()
    
    for feat in avg_corr.index:
        if feat in selected:
            continue
        
        # Check correlation with already selected features
        is_uncorrelated = True
        for sel_feat in selected:
            if abs(corr_matrix.loc[feat, sel_feat]) > threshold:
                is_uncorrelated = False
                break
        
        if is_uncorrelated:
            selected.append(feat)
    
    print(f"   Selected features: {selected}")
    print(f"   Removed: {[f for f in features if f not in selected]}")
    
    return selected


def build_regression_models(data: pd.DataFrame):
    """Build and compare regression models using INDIVIDUAL model-prompt pairs."""
    
    print("\n" + "="*80)
    print("REGRESSION MODELS: Predicting Individual Success (Model-Prompt Level)")
    print("="*80)
    
    # Use individual model-prompt pairs (gives proper variance in both CRS and NVIDIA features)
    print(f"\n📊 Data Structure:")
    print(f"   {len(data)} model-prompt pairs")
    print(f"   {data['model_name'].nunique()} unique models")
    print(f"   {data['problem_id'].nunique()} unique prompts")
    print(f"   Success rate: {data['is_correct'].mean()*100:.1f}%")
    
    # Check multicollinearity on NVIDIA features
    nvidia_features = ['prompt_complexity_score', 'reasoning_score', 'domain_knowledge', 
                       'constraint_ct', 'is_complex', 'is_reasoning_heavy']
    
    corr_matrix, vif_data = check_multicollinearity(data, nvidia_features)
    
    # Auto-select uncorrelated features
    selected_nvidia = select_uncorrelated_features(corr_matrix, threshold=0.8)
    
    # Also compute model-level aggregations for visualization
    model_level = data.groupby('model_name').agg({
        'crs_score': 'first',
        'is_correct': 'mean',  # Average accuracy
    }).reset_index()
    model_level.columns = ['model_name', 'crs_score', 'accuracy']
    
    # Define feature sets (using uncorrelated NVIDIA features)
    feature_sets = {
        'CRS Only': ['crs_score'],
        'CRS + Complexity': ['crs_score', 'prompt_complexity_score'],
        'CRS + Reasoning': ['crs_score', 'reasoning_score'],
        'CRS + Domain': ['crs_score', 'domain_knowledge'],
        'CRS + Selected NVIDIA (Uncorrelated)': ['crs_score'] + selected_nvidia,
        'CRS + All NVIDIA (Full)': ['crs_score'] + nvidia_features,
        'CRS × Complexity (Interaction)': ['crs_score', 'prompt_complexity_score', 'crs_x_complexity'],
    }
    
    # Add interaction term
    data['crs_x_complexity'] = data['crs_score'] * data['prompt_complexity_score']
    
    # Target (binary: is_correct)
    y = data['is_correct'].values
    
    results = []
    model_predictions = {}  # For aggregated model-level predictions
    
    for fs_name, features in feature_sets.items():
        print(f"\n{'─'*80}")
        print(f"Model: {fs_name}")
        print(f"{'─'*80}")
        
        X = data[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Logistic Regression for binary classification
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Logistic regression for binary outcome
        clf = LogisticRegression(random_state=42, max_iter=1000)
        
        # Cross-validation
        from sklearn.metrics import roc_auc_score, make_scorer
        acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')
        
        acc_mean = acc_scores.mean()
        auc_mean = auc_scores.mean()
        
        print(f"\n   Accuracy: {acc_mean:.3f} ± {acc_scores.std():.3f}")
        print(f"   ROC-AUC:  {auc_mean:.3f} ± {auc_scores.std():.3f}")
        
        # Train on full data
        clf.fit(X_scaled, y)
        y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
        
        # Aggregate to model level for visualization
        data_with_pred = data.copy()
        data_with_pred['y_pred_proba'] = y_pred_proba
        
        model_agg = data_with_pred.groupby('model_name').agg({
            'crs_score': 'first',
            'is_correct': 'mean',
            'y_pred_proba': 'mean',
        }).reset_index()
        
        model_predictions[fs_name] = {
            'model_names': model_agg['model_name'].values,
            'y_true': model_agg['is_correct'].values * 100,
            'y_pred': model_agg['y_pred_proba'].values * 100,
            'features': features,
        }
        
        results.append({
            'feature_set': fs_name,
            'accuracy': acc_mean,
            'auc': auc_mean,
            'features': features,
        })
        
        # Show feature coefficients
        if hasattr(clf, 'coef_'):
            print(f"\n   Coefficients:")
            for feat, coef in zip(features, clf.coef_[0]):
                print(f"      {feat:<30} {coef:>8.3f}")
            if hasattr(clf, 'intercept_'):
                print(f"      {'Intercept':<30} {clf.intercept_[0]:>8.3f}")
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Performance by Feature Set")
    print(f"{'='*80}")
    
    print(f"\n{'Feature Set':<40} {'Accuracy':<12} {'ROC-AUC':<12}")
    print(f"{'-'*40} {'-'*12} {'-'*12}")
    
    for result in results:
        print(f"{result['feature_set']:<40} {result['accuracy']:>6.3f}       {result['auc']:>6.3f}")
    
    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    crs_only = [r for r in results if r['feature_set'] == 'CRS Only'][0]
    best_combined = max([r for r in results if r['feature_set'] != 'CRS Only'], key=lambda x: x['auc'])
    
    improvement = best_combined['auc'] - crs_only['auc']
    pct_improvement = (improvement / max(abs(crs_only['auc']), 0.01)) * 100
    
    print(f"\n   CRS Only:        Accuracy = {crs_only['accuracy']:.3f}, AUC = {crs_only['auc']:.3f}")
    print(f"   Best Combined:   Accuracy = {best_combined['accuracy']:.3f}, AUC = {best_combined['auc']:.3f}")
    print(f"   Feature Set:     {best_combined['feature_set']}")
    print(f"   Improvement:     ΔAUC = {improvement:+.3f} ({pct_improvement:+.1f}%)")
    
    if improvement > 0.05:
        print(f"\n   ✓ NVIDIA features provide MEANINGFUL improvement")
    elif improvement > 0:
        print(f"\n   ~ NVIDIA features provide MARGINAL improvement")
    else:
        print(f"\n   ✗ NVIDIA features do NOT improve prediction")
    
    return model_level, model_predictions, results


def create_comparison_plot(model_level: pd.DataFrame, predictions: Dict, output_dir: Path):
    """Create side-by-side comparison plots."""
    
    print(f"\n📊 Creating comparison visualization...")
    
    # Select best CRS-only and best combined model
    crs_only = predictions.get('CRS Only')
    
    # Find best combined model (by R²)
    combined_predictions = {k: v for k, v in predictions.items() if k != 'CRS Only'}
    if combined_predictions:
        best_combined_name = max(combined_predictions.keys(), 
                                 key=lambda k: r2_score(combined_predictions[k]['y_true'], 
                                                       combined_predictions[k]['y_pred']))
        best_combined = combined_predictions[best_combined_name]
    else:
        best_combined = None
        best_combined_name = None
    
    if not crs_only or not best_combined:
        print("   ⚠️  Missing predictions for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: CRS Only - Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(crs_only['y_true'], crs_only['y_pred'], s=100, alpha=0.6, c='steelblue')
    
    # Perfect prediction line
    min_val = min(crs_only['y_true'].min(), crs_only['y_pred'].min())
    max_val = max(crs_only['y_true'].max(), crs_only['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Perfect prediction')
    
    r2 = r2_score(crs_only['y_true'], crs_only['y_pred'])
    mae = mean_absolute_error(crs_only['y_true'], crs_only['y_pred'])
    
    ax1.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.2f}%", 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Actual Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('CRS Only', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined - Predicted vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(best_combined['y_true'], best_combined['y_pred'], s=100, alpha=0.6, c='forestgreen')
    
    min_val2 = min(best_combined['y_true'].min(), best_combined['y_pred'].min())
    max_val2 = max(best_combined['y_true'].max(), best_combined['y_pred'].max())
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', linewidth=2, alpha=0.7, label='Perfect prediction')
    
    r2_combined = r2_score(best_combined['y_true'], best_combined['y_pred'])
    mae_combined = mean_absolute_error(best_combined['y_true'], best_combined['y_pred'])
    
    ax2.text(0.05, 0.95, f"R² = {r2_combined:.3f}\nMAE = {mae_combined:.2f}%", 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_xlabel('Actual Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{best_combined_name}', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual plot - CRS Only
    ax3 = axes[1, 0]
    residuals_crs = crs_only['y_true'] - crs_only['y_pred']
    ax3.scatter(crs_only['y_pred'], residuals_crs, s=100, alpha=0.6, c='steelblue')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Predicted Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Residual (Actual - Predicted, %)', fontsize=11, fontweight='bold')
    ax3.set_title('Residuals: CRS Only', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual plot - Combined
    ax4 = axes[1, 1]
    residuals_combined = best_combined['y_true'] - best_combined['y_pred']
    ax4.scatter(best_combined['y_pred'], residuals_combined, s=100, alpha=0.6, c='forestgreen')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Predicted Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residual (Actual - Predicted, %)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Residuals: {best_combined_name}', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "crs_nvidia_regression_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {plot_path}")
    
    plt.close()


def main():
    print("="*80)
    print("INTERACTION MODEL: CRS × NVIDIA COMPLEXITY")
    print("="*80)
    
    # Load data
    responses_df, problem_ids = load_arc_responses()
    problem_texts = load_arc_problem_texts(problem_ids)
    nvidia_df = classify_prompts_with_nvidia(problem_texts)
    
    # Merge
    data = responses_df.merge(nvidia_df, on='problem_id')
    
    print(f"\n✓ Complete dataset: {len(data)} model-prompt pairs")
    
    # Build models
    model_level, predictions, results = build_regression_models(data)
    
    # Create visualization
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    create_comparison_plot(model_level, predictions, output_dir)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    # Save results
    output_path = output_dir / "crs_nvidia_interaction_results.json"
    
    results_summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_models': len(model_level),
        'n_prompts': len(problem_ids),
        'models': results,
        'model_predictions': {
            k: {
                'r2': float(r2_score(v['y_true'], v['y_pred'])),
                'mae': float(mean_absolute_error(v['y_true'], v['y_pred'])),
                'rmse': float(np.sqrt(mean_squared_error(v['y_true'], v['y_pred']))),
            }
            for k, v in predictions.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
