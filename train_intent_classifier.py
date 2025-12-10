#!/usr/bin/env python3
"""Train intent classifier and save comprehensive results."""

import json
import numpy as np
import pickle
from collections import Counter
import subprocess
import sys
from pathlib import Path

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "sentence-transformers", "xgboost", "scikit-learn", 
                "matplotlib", "seaborn"], check=False)

from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Installing dependencies...")
print("\nImporting libraries...")

# Load data
print("="*80)
print("Loading labeled intent data...")
with open('data/real_intent_prompts_labeled.json') as f:
    data = json.load(f)

prompts = [s['prompt'] for s in data['samples']]
labels = [s['intent_label'] for s in data['samples']]

print(f"Total samples: {len(prompts)}")
label_counts = Counter(labels)
for label, count in sorted(label_counts.items()):
    print(f"  {label:<20} {count} ({count/len(labels)*100:.1f}%)")

# Extract embeddings
print("\n" + "="*80)
print("Extracting embeddings (this may take a minute)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X = embedder.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
print(f"Embedding shape: {X.shape}")

# Encode labels
unique_labels = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
y = np.array([label_to_idx[label] for label in labels])

# 5-Fold CV with detailed tracking
print("\n" + "="*80)
print("Training XGBoost with 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_labels),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"  Fold {fold}/5: Accuracy = {acc:.4f}")
    
    fold_results.append({
        'fold': fold,
        'accuracy': float(acc),
        'f1_score': float(f1),
        'n_train': len(train_idx),
        'n_val': len(val_idx)
    })
    
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

# Overall results
overall_acc = accuracy_score(all_y_true, all_y_pred)
overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')

print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)
print(f"Accuracy: {overall_acc:.4f}")
print(f"F1-Score: {overall_f1:.4f}")

# Per-class metrics
y_true_labels = [unique_labels[i] for i in all_y_true]
y_pred_labels = [unique_labels[i] for i in all_y_pred]

print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true_labels, y_pred_labels, digits=4))

# Confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Per-class results with detailed metrics
from sklearn.metrics import precision_recall_fscore_support
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_true_labels, y_pred_labels, labels=unique_labels
)

per_class_results = []
for i, label in enumerate(unique_labels):
    total = cm[i].sum()
    correct = cm[i, i]
    acc = correct / total if total > 0 else 0
    
    per_class_results.append({
        'intent': label,
        'samples': int(supports[i]),
        'accuracy': float(acc),
        'precision': float(precisions[i]),
        'recall': float(recalls[i]),
        'f1_score': float(f1s[i]),
        'correct': int(correct),
        'total': int(total)
    })
    
    print(f"{label:<20} {correct:>4}/{total:<4} = {acc:>6.1%}")

print("\n" + "="*80)
print("Plotting confusion matrix...")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels, ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=unique_labels, yticklabels=unique_labels,
            ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()

# Save confusion matrix
output_dir = Path('results/intent_classification')
output_dir.mkdir(parents=True, exist_ok=True)

cm_path = output_dir / 'confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Saved to: {cm_path}")

# Train final model
print("\n" + "="*80)
print("Training final model on full dataset...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(unique_labels),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
final_model.fit(X, y)

# Save model
model_path = output_dir / 'xgboost_intent_classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Saved model to: {model_path}")

# Save comprehensive results JSON
results_json = {
    'metadata': {
        'date': str(Path('data/real_intent_prompts_labeled.json').stat().st_mtime),
        'n_samples': len(prompts),
        'n_classes': len(unique_labels),
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dim': X.shape[1],
        'classifier': 'XGBoost',
        'cv_strategy': '5-fold stratified',
        'random_seed': 42
    },
    'overall': {
        'accuracy': float(overall_acc),
        'accuracy_std': float(np.std([f['accuracy'] for f in fold_results])),
        'f1_score': float(overall_f1),
        'f1_std': float(np.std([f['f1_score'] for f in fold_results]))
    },
    'fold_results': fold_results,
    'per_class': per_class_results,
    'confusion_matrix': {
        'labels': unique_labels,
        'counts': cm.tolist(),
        'normalized': cm_norm.tolist()
    },
    'label_distribution': {label: int(count) for label, count in label_counts.items()}
}

results_path = output_dir / 'xgboost_results.json'
with open(results_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Saved detailed results to: {results_path}")

print("\n✅ DONE! Check results/intent_classification/ for outputs.")
