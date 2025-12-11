#!/usr/bin/env python3
"""
Stratified Performance Analysis - KDD-Level Evaluation

Analyzes classifier performance across length buckets to detect length bias.

Key Question: Does accuracy remain stable across "Short", "Medium", and "Long" prompts?
- Stable → No length bias
- Unstable → Model relies on length correlations

This is a critical fairness/invariance test for KDD reviewers.
"""

import json
import numpy as np
import pickle
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from pathlib import Path

print("="*80)
print("STRATIFIED PERFORMANCE ANALYSIS")
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

# Define length buckets (tertiles)
percentiles = [0, 33, 67, 100]
length_bins_edges = np.percentile(lengths, percentiles)
length_bins_edges[-1] += 1

print(f"\nLength buckets (tertiles):")
print(f"  Short:  <{length_bins_edges[1]:.0f} chars")
print(f"  Medium: {length_bins_edges[1]:.0f}-{length_bins_edges[2]:.0f} chars")
print(f"  Long:   >{length_bins_edges[2]:.0f} chars")

# Assign samples to buckets
length_buckets = np.digitize(lengths, length_bins_edges[1:-1])
bucket_names = ['Short', 'Medium', 'Long']

# Get embeddings
print("\nExtracting embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(prompts, show_progress_bar=False, convert_to_numpy=True)

# Prepare labels
label_list = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(label_list)}
y = np.array([label_to_idx[label] for label in labels])

# ============================================================================
# ANALYZE TRAINING DATA DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("TRAINING DATA: LENGTH DISTRIBUTION BY INTENT")
print("="*80)

intent_length_dist = defaultdict(lambda: defaultdict(int))
for label, bucket in zip(labels, length_buckets):
    intent_length_dist[label][bucket] += 1

print(f"\n{'Intent':<20} | {'Short':<15} | {'Medium':<15} | {'Long':<15} | Total")
print("-" * 85)

for intent in sorted(label_list):
    total = sum(intent_length_dist[intent].values())
    short = intent_length_dist[intent][0]
    medium = intent_length_dist[intent][1]
    long = intent_length_dist[intent][2]
    
    print(f"{intent:<20} | {short:4} ({short/total*100:5.1f}%) | {medium:4} ({medium/total*100:5.1f}%) | {long:4} ({long/total*100:5.1f}%) | {total:4}")

print("\n⚠️  KEY OBSERVATION:")
print("If SUMMARIZATION is predominantly 'Long' and others are 'Short/Medium',")
print("the model can achieve high accuracy by just learning 'Long → SUMMARIZATION'.")

# ============================================================================
# CROSS-VALIDATION WITH STRATIFIED TRACKING
# ============================================================================

print("\n" + "="*80)
print("5-FOLD CV: TRACKING PERFORMANCE BY LENGTH BUCKET")
print("="*80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store predictions for each fold
cv_predictions = defaultdict(list)  # bucket_id -> [(y_true, y_pred), ...]

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold_idx}/{n_folds}...")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Track by length bucket
    val_buckets = length_buckets[val_idx]
    for bucket_id in range(3):
        mask = (val_buckets == bucket_id)
        for true_label, pred_label in zip(y_val[mask], y_pred[mask]):
            cv_predictions[bucket_id].append((true_label, pred_label))

# ============================================================================
# STRATIFIED PERFORMANCE TABLE
# ============================================================================

print("\n" + "="*80)
print("STRATIFIED PERFORMANCE: OVERALL")
print("="*80)

print(f"\n{'Length Bucket':<15} | {'N Samples':<12} | {'Accuracy':<12} | {'F1-Score':<12}")
print("-" * 60)

bucket_metrics = {}

for bucket_id, bucket_name in enumerate(bucket_names):
    if bucket_id not in cv_predictions:
        continue
    
    preds = cv_predictions[bucket_id]
    y_true_bucket = np.array([p[0] for p in preds])
    y_pred_bucket = np.array([p[1] for p in preds])
    
    acc = accuracy_score(y_true_bucket, y_pred_bucket)
    f1 = f1_score(y_true_bucket, y_pred_bucket, average='weighted')
    n = len(y_true_bucket)
    
    bucket_metrics[bucket_name] = {
        'n': n,
        'accuracy': float(acc),
        'f1_score': float(f1)
    }
    
    print(f"{bucket_name:<15} | {n:<12} | {acc:.4f} ({acc*100:5.1f}%) | {f1:.4f}")

# Overall
all_true = np.array([p[0] for bucket_preds in cv_predictions.values() for p in bucket_preds])
all_pred = np.array([p[1] for bucket_preds in cv_predictions.values() for p in bucket_preds])
overall_acc = accuracy_score(all_true, all_pred)
overall_f1 = f1_score(all_true, all_pred, average='weighted')

print("-" * 60)
print(f"{'OVERALL':<15} | {len(all_true):<12} | {overall_acc:.4f} ({overall_acc*100:5.1f}%) | {overall_f1:.4f}")

# Stability metrics
accuracies = [m['accuracy'] for m in bucket_metrics.values()]
variance = np.var(accuracies)
acc_range = max(accuracies) - min(accuracies)

print(f"\n📊 STABILITY METRICS:")
print(f"  Variance across buckets: {variance:.6f}")
print(f"  Accuracy range: {acc_range:.4f} ({acc_range*100:.1f}%)")

if acc_range < 0.05:
    print(f"  ✅ STABLE: Performance consistent across lengths")
elif acc_range < 0.10:
    print(f"  ⚠️  MODERATE: Some length-dependent variation")
else:
    print(f"  ❌ UNSTABLE: Strong length bias detected")

# ============================================================================
# STRATIFIED PERFORMANCE: PER-INTENT
# ============================================================================

print("\n" + "="*80)
print("STRATIFIED PERFORMANCE: PER-INTENT BREAKDOWN")
print("="*80)

per_intent_stratified = {}

for intent_idx, intent_name in enumerate(label_list):
    print(f"\n{intent_name.upper()}")
    print("-" * 60)
    
    intent_bucket_metrics = {}
    
    for bucket_id, bucket_name in enumerate(bucket_names):
        if bucket_id not in cv_predictions:
            continue
        
        preds = cv_predictions[bucket_id]
        
        # Filter for this intent
        intent_true = [(t, p) for t, p in preds if t == intent_idx]
        
        if len(intent_true) == 0:
            print(f"  {bucket_name:<15} | No samples in validation")
            continue
        
        y_true_intent = np.array([p[0] for p in intent_true])
        y_pred_intent = np.array([p[1] for p in intent_true])
        
        acc = accuracy_score(y_true_intent, y_pred_intent)
        n = len(y_true_intent)
        
        intent_bucket_metrics[bucket_name] = {
            'n': n,
            'accuracy': float(acc)
        }
        
        print(f"  {bucket_name:<15} | n={n:3} | Accuracy: {acc:.4f} ({acc*100:5.1f}%)")
    
    per_intent_stratified[intent_name] = intent_bucket_metrics
    
    # Intent-level stability
    if len(intent_bucket_metrics) > 1:
        intent_accs = [m['accuracy'] for m in intent_bucket_metrics.values()]
        intent_range = max(intent_accs) - min(intent_accs)
        print(f"  Range: {intent_range:.4f} ({intent_range*100:.1f}%)")

# ============================================================================
# DETECT PROBLEMATIC PATTERNS
# ============================================================================

print("\n" + "="*80)
print("BIAS DETECTION: PROBLEMATIC PATTERNS")
print("="*80)

print("\nLooking for: Intents that are predominantly ONE length in training")
print("              but perform poorly on OTHER lengths in validation")

problems_detected = []

for intent in sorted(label_list):
    intent_idx = label_to_idx[intent]
    
    # Training distribution
    train_short = intent_length_dist[intent][0]
    train_medium = intent_length_dist[intent][1]
    train_long = intent_length_dist[intent][2]
    train_total = train_short + train_medium + train_long
    
    train_dist = {
        'Short': train_short / train_total,
        'Medium': train_medium / train_total,
        'Long': train_long / train_total
    }
    
    # Validation performance
    val_perf = per_intent_stratified.get(intent, {})
    
    # Check: If intent is >80% one length in training, does it fail on other lengths?
    dominant_bucket = max(train_dist, key=train_dist.get)
    dominant_pct = train_dist[dominant_bucket]
    
    if dominant_pct > 0.8:
        print(f"\n⚠️  {intent.upper()}: {dominant_pct*100:.0f}% '{dominant_bucket}' in training")
        
        # Check performance on NON-dominant buckets
        other_buckets = [b for b in bucket_names if b != dominant_bucket]
        for other_bucket in other_buckets:
            if other_bucket in val_perf:
                other_acc = val_perf[other_bucket]['accuracy']
                other_n = val_perf[other_bucket]['n']
                
                if other_n > 0:
                    print(f"   Performance on '{other_bucket}': {other_acc:.1%} (n={other_n})")
                    
                    if other_acc < 0.7:
                        problems_detected.append({
                            'intent': intent,
                            'dominant_bucket': dominant_bucket,
                            'dominant_pct': dominant_pct,
                            'problem_bucket': other_bucket,
                            'problem_accuracy': other_acc,
                            'problem_n': other_n
                        })
                        print(f"   ❌ LOW ACCURACY: Model likely learned length shortcut")
            else:
                print(f"   '{other_bucket}': No validation samples")

if len(problems_detected) == 0:
    print("\n✅ No obvious length-based shortcuts detected in validation data")
else:
    print(f"\n❌ DETECTED {len(problems_detected)} LENGTH BIAS ISSUES")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'overall_stratified': bucket_metrics,
    'per_intent_stratified': per_intent_stratified,
    'stability': {
        'variance': float(variance),
        'range': float(acc_range)
    },
    'training_distribution': {
        intent: {
            'Short': int(intent_length_dist[intent][0]),
            'Medium': int(intent_length_dist[intent][1]),
            'Long': int(intent_length_dist[intent][2])
        }
        for intent in label_list
    },
    'problems_detected': problems_detected
}

output_path = Path('stratified_performance_analysis.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Saved detailed results to: {output_path}")

# ============================================================================
# RECOMMENDATIONS FOR KDD PAPER
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS FOR KDD PAPER")
print("="*80)

print("""
Include this stratified performance table in your paper:

Table X: Classifier Performance Stratified by Prompt Length

| Length Bucket | N Samples | Accuracy | F1-Score |
|---------------|-----------|----------|----------|""")

for bucket_name in bucket_names:
    if bucket_name in bucket_metrics:
        m = bucket_metrics[bucket_name]
        print(f"| {bucket_name:<13} | {m['n']:<9} | {m['accuracy']:.4f}   | {m['f1_score']:.4f}   |")

print(f"| Overall       | {len(all_true):<9} | {overall_acc:.4f}   | {overall_f1:.4f}   |")

print(f"""
Variance: {variance:.6f}
Range: ±{acc_range/2:.2%}

**Interpretation:**
- If variance < 0.0001 and range < 5%: "Performance is stable across length buckets,
  demonstrating that the classifier relies on semantic features rather than length."
  
- If variance > 0.001 or range > 10%: "Performance varies across length buckets,
  suggesting the model has learned some length-dependent shortcuts. This is expected
  given the training distribution where SUMMARIZATION intents are predominantly long."

**Critical for Reviewers:**
This table demonstrates awareness of potential length bias and provides empirical
evidence of model behavior across the length spectrum.
""")

print("="*80)
