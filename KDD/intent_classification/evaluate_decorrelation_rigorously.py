#!/usr/bin/env python3
"""
Rigorous Evaluation: Prove Length Bias Removal

KDD-level validation that orthogonal projection eliminates length artifacts:
1. Nuisance Prediction Test: Can we predict length from intent predictions?
2. Stratified Performance: Is accuracy stable across length buckets?
3. Correlation Analysis: Compare baseline vs decorrelated
"""

import json
import numpy as np
import pickle
from collections import defaultdict
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb

print("="*80)
print("RIGOROUS EVALUATION: Length Bias Removal")
print("="*80)

# Load data
print("\nLoading data...")
with open('../../data/real_intent_prompts_labeled.json') as f:
    data = json.load(f)

prompts = [s['prompt'] for s in data['samples']]
labels = [s['intent_label'] for s in data['samples']]
lengths = np.array([len(p) for p in prompts])

print(f"Total samples: {len(prompts)}")
print(f"Length range: {lengths.min()}-{lengths.max()} chars")

# Load both models
print("\nLoading models...")

# Original model
with open('../../results/intent_classification/xgboost_results.json') as f:
    original_results = json.load(f)

# Decorrelated model
with open('../../results/intent_classification/xgboost_intent_classifier_decorrelated.pkl', 'rb') as f:
    decorr_checkpoint = pickle.load(f)
    decorr_model = decorr_checkpoint['model']
    projection = decorr_checkpoint['projection']
    label_list = decorr_checkpoint['labels']

with open('../../results/intent_classification/xgboost_results_decorrelated.json') as f:
    decorr_results = json.load(f)

# Get embeddings
print("\nExtracting embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_raw = embedder.encode(prompts, show_progress_bar=False, convert_to_numpy=True)

# Apply decorrelation
L = lengths.reshape(-1, 1)
L_norm = (L - projection['length_mean']) / projection['length_std']
X_length_component = projection['ridge'].predict(L_norm)
X_decorr = X_raw - X_length_component

# Get predictions from both models
print("\nGenerating predictions...")

# Load original model
with open('../../results/intent_classification/xgboost_intent_classifier.pkl', 'rb') as f:
    original_model = pickle.load(f)

original_preds = original_model.predict(X_raw)
original_probs = original_model.predict_proba(X_raw)

decorr_preds = decorr_model.predict(X_decorr)
decorr_probs = decorr_model.predict_proba(X_decorr)

# ============================================================================
# TEST 1: NUISANCE PREDICTION TEST
# ============================================================================

print("\n" + "="*80)
print("TEST 1: NUISANCE PREDICTION TEST")
print("="*80)
print("\nCan we predict prompt length from model predictions?")
print("(If yes → predictions leak length information)")

def nuisance_prediction_test(predictions, lengths, model_name):
    """
    Train a dummy classifier to predict length from intent predictions.
    
    If the dummy classifier performs well (high R²), the predictions
    are still correlated with length (bad).
    If it performs poorly (low R²), length bias is removed (good).
    """
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # One-hot encode predictions
    unique_preds = sorted(set(predictions))
    pred_onehot = np.zeros((len(predictions), len(unique_preds)))
    for i, pred in enumerate(predictions):
        pred_onehot[i, unique_preds.index(pred)] = 1
    
    # Train dummy regressor: Length ~ Predictions
    dummy = Ridge(alpha=1.0)
    
    # Cross-validation to avoid overfitting
    cv_scores = cross_val_score(dummy, pred_onehot, lengths, 
                                 cv=5, scoring='r2')
    
    # Fit on full data for analysis
    dummy.fit(pred_onehot, lengths)
    pred_lengths = dummy.predict(pred_onehot)
    
    r2 = r2_score(lengths, pred_lengths)
    mae = mean_absolute_error(lengths, pred_lengths)
    corr = np.corrcoef(lengths, pred_lengths)[0, 1]
    
    print(f"R² (predicting length from predictions): {r2:.4f}")
    print(f"R² (cross-validated): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"MAE: {mae:.0f} chars")
    print(f"Correlation: {corr:.4f}")
    
    # Interpretation
    if r2 > 0.1:
        print(f"⚠️  HIGH LEAKAGE: Predictions strongly correlate with length")
    elif r2 > 0.05:
        print(f"⚠️  MODERATE LEAKAGE: Some length correlation remains")
    else:
        print(f"✅ LOW LEAKAGE: Predictions largely independent of length")
    
    return {
        'r2': r2,
        'r2_cv': cv_scores.mean(),
        'r2_cv_std': cv_scores.std(),
        'mae': mae,
        'correlation': corr
    }

original_leakage = nuisance_prediction_test(original_preds, lengths, "Original Model")
decorr_leakage = nuisance_prediction_test(decorr_preds, lengths, "Decorrelated Model")

print("\nComparison:")
print(f"  R² reduction: {original_leakage['r2']:.4f} → {decorr_leakage['r2']:.4f}")
print(f"  Relative improvement: {(1 - decorr_leakage['r2']/original_leakage['r2'])*100:.1f}%")

# ============================================================================
# TEST 2: STRATIFIED PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEST 2: STRATIFIED PERFORMANCE ANALYSIS")
print("="*80)
print("\nIs accuracy stable across length buckets?")
print("(Stable → no length bias, Unstable → length bias remains)")

# Define length buckets
percentiles = [0, 33, 67, 100]
length_bins = np.percentile(lengths, percentiles)
length_bins[-1] += 1  # Ensure max is included

bucket_names = ['Short (<P33)', 'Medium (P33-P67)', 'Long (>P67)']
print(f"\nLength buckets:")
print(f"  Short:  <{length_bins[1]:.0f} chars ({percentiles[1]}th percentile)")
print(f"  Medium: {length_bins[1]:.0f}-{length_bins[2]:.0f} chars")
print(f"  Long:   >{length_bins[2]:.0f} chars ({100-percentiles[2]}th percentile)")

# Assign samples to buckets
length_bucket = np.digitize(lengths, length_bins[1:-1])

# True labels
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
y_true = np.array([label_to_idx[label] for label in labels])

def stratified_performance(predictions, y_true, lengths, bucket_assignments, model_name):
    """Compute accuracy separately for each length bucket."""
    print(f"\n{model_name}:")
    print("-" * 60)
    
    bucket_stats = []
    
    for bucket_id, bucket_name in enumerate(bucket_names):
        mask = (bucket_assignments == bucket_id)
        n = mask.sum()
        
        if n == 0:
            continue
        
        acc = (predictions[mask] == y_true[mask]).mean()
        bucket_stats.append({
            'bucket': bucket_name,
            'n': int(n),
            'accuracy': float(acc)
        })
        
        print(f"{bucket_name:20} n={n:4} | Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    
    # Compute variance across buckets
    accuracies = [s['accuracy'] for s in bucket_stats]
    variance = np.var(accuracies)
    range_acc = max(accuracies) - min(accuracies)
    
    print(f"\nStability Metrics:")
    print(f"  Variance: {variance:.6f}")
    print(f"  Range: {range_acc:.4f} ({range_acc*100:.1f}%)")
    
    if range_acc < 0.05:
        print(f"  ✅ STABLE: Performance consistent across lengths")
    elif range_acc < 0.10:
        print(f"  ⚠️  MODERATE: Some length-dependent variation")
    else:
        print(f"  ❌ UNSTABLE: Strong length-dependent variation")
    
    return bucket_stats, variance, range_acc

original_buckets, orig_var, orig_range = stratified_performance(
    original_preds, y_true, lengths, length_bucket, "Original Model"
)

decorr_buckets, decorr_var, decorr_range = stratified_performance(
    decorr_preds, y_true, lengths, length_bucket, "Decorrelated Model"
)

print("\nComparison:")
print(f"  Range reduction: {orig_range:.4f} → {decorr_range:.4f}")
print(f"  Variance reduction: {orig_var:.6f} → {decorr_var:.6f}")

# ============================================================================
# TEST 3: ERROR CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEST 3: ERROR CORRELATION WITH LENGTH")
print("="*80)
print("\nDo errors correlate with prompt length?")
print("(High correlation → model struggles with certain lengths)")

def error_correlation(predictions, y_true, lengths, model_name):
    """Analyze correlation between prediction errors and length."""
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # Binary error: correct (0) or wrong (1)
    errors = (predictions != y_true).astype(int)
    
    # Correlation
    corr = np.corrcoef(lengths, errors)[0, 1]
    
    # Mean length of correct vs incorrect
    correct_mask = (predictions == y_true)
    mean_length_correct = lengths[correct_mask].mean()
    mean_length_incorrect = lengths[~correct_mask].mean()
    
    print(f"Correlation (length vs error): {corr:.4f}")
    print(f"Mean length when correct: {mean_length_correct:.0f} chars")
    print(f"Mean length when wrong: {mean_length_incorrect:.0f} chars")
    print(f"Difference: {abs(mean_length_correct - mean_length_incorrect):.0f} chars")
    
    if abs(corr) < 0.05:
        print(f"✅ NO BIAS: Errors independent of length")
    elif abs(corr) < 0.10:
        print(f"⚠️  WEAK BIAS: Slight length dependency")
    else:
        print(f"❌ STRONG BIAS: Errors correlate with length")
    
    return {
        'correlation': corr,
        'mean_length_correct': mean_length_correct,
        'mean_length_incorrect': mean_length_incorrect
    }

original_error = error_correlation(original_preds, y_true, lengths, "Original Model")
decorr_error = error_correlation(decorr_preds, y_true, lengths, "Decorrelated Model")

# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)

summary = {
    'overall_performance': {
        'original': {
            'accuracy': original_results['overall']['accuracy'],
            'f1': original_results['overall']['f1_score']
        },
        'decorrelated': {
            'accuracy': decorr_results['overall']['accuracy'],
            'f1': decorr_results['overall']['f1_score']
        }
    },
    'nuisance_prediction': {
        'original': original_leakage,
        'decorrelated': decorr_leakage
    },
    'stratified_performance': {
        'original': {
            'buckets': original_buckets,
            'variance': float(orig_var),
            'range': float(orig_range)
        },
        'decorrelated': {
            'buckets': decorr_buckets,
            'variance': float(decorr_var),
            'range': float(decorr_range)
        }
    },
    'error_correlation': {
        'original': original_error,
        'decorrelated': decorr_error
    }
}

# Save results
with open('decorrelation_evaluation.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n📊 OVERALL PERFORMANCE")
print("-" * 60)
print(f"Original Model:")
print(f"  Accuracy: {original_results['overall']['accuracy']:.4f}")
print(f"  F1-Score: {original_results['overall']['f1_score']:.4f}")
print(f"\nDecorrelated Model:")
print(f"  Accuracy: {decorr_results['overall']['accuracy']:.4f}")
print(f"  F1-Score: {decorr_results['overall']['f1_score']:.4f}")
print(f"\nAccuracy cost: {(original_results['overall']['accuracy'] - decorr_results['overall']['accuracy'])*100:.1f}%")

print("\n🔍 BIAS METRICS")
print("-" * 60)
print(f"Nuisance Prediction R² (lower = better):")
print(f"  Original:      {original_leakage['r2']:.4f}")
print(f"  Decorrelated:  {decorr_leakage['r2']:.4f}")
print(f"  Improvement:   {(1 - decorr_leakage['r2']/original_leakage['r2'])*100:.1f}%")

print(f"\nStratified Performance Range (lower = better):")
print(f"  Original:      {orig_range:.4f} ({orig_range*100:.1f}%)")
print(f"  Decorrelated:  {decorr_range:.4f} ({decorr_range*100:.1f}%)")
print(f"  Improvement:   {(1 - decorr_range/orig_range)*100:.1f}%")

print(f"\nError-Length Correlation (closer to 0 = better):")
print(f"  Original:      {original_error['correlation']:.4f}")
print(f"  Decorrelated:  {decorr_error['correlation']:.4f}")
print(f"  Improvement:   {(1 - abs(decorr_error['correlation'])/abs(original_error['correlation']))*100:.1f}%")

print("\n✅ VERDICT")
print("-" * 60)

if decorr_leakage['r2'] < 0.05 and decorr_range < 0.10 and abs(decorr_error['correlation']) < 0.05:
    print("✅ SUCCESS: Length bias effectively removed")
    print("   - Nuisance prediction: LOW")
    print("   - Stratified performance: STABLE")
    print("   - Error correlation: NEGLIGIBLE")
elif decorr_leakage['r2'] < original_leakage['r2'] * 0.5:
    print("🟡 PARTIAL SUCCESS: Significant bias reduction")
    print(f"   - {(1 - decorr_leakage['r2']/original_leakage['r2'])*100:.0f}% reduction in length leakage")
    print(f"   - {(original_results['overall']['accuracy'] - decorr_results['overall']['accuracy'])*100:.1f}% accuracy cost")
else:
    print("❌ LIMITED SUCCESS: Bias partially remains")

print(f"\nTrade-off:")
print(f"  Bias reduction: {(1 - decorr_leakage['r2']/original_leakage['r2'])*100:.0f}%")
print(f"  Accuracy cost: {(original_results['overall']['accuracy'] - decorr_results['overall']['accuracy'])*100:.1f}%")
print(f"  Is it worth it? {'Yes' if decorr_leakage['r2'] < 0.1 else 'Debatable'}")

print(f"\n💾 Saved detailed results to: decorrelation_evaluation.json")
