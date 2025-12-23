"""Debug: Check risk classifier scores vs ground truth"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from final_release.kdd_paper.table_3.router_performance_comparison import (
    load_battle_dataset,
    load_model_registry,
    run_judging_pipeline,
)
from final_release.high_risk_prompt_classifier import HighRiskPromptClassifier
import numpy as np
from tqdm import tqdm

# Load data
df = load_battle_dataset(500)
df = run_judging_pipeline(df)

# Ground truth
is_high_risk = (df["weak_is_valid"] == 0).values

# Get risk scores
clf = HighRiskPromptClassifier(threshold=5.0)
risk_scores = []

print("\n[Scoring] Computing risk classifier scores...")
for q in tqdm(df["question"]):
    result = clf.classify(q)
    risk_scores.append(result.score)

risk_scores = np.array(risk_scores)

# Analyze
print(f"\nGround Truth:")
print(f"  Safe: {(~is_high_risk).sum()} ({100*(~is_high_risk).mean():.1f}%)")
print(f"  Risky: {is_high_risk.sum()} ({100*is_high_risk.mean():.1f}%)")

print(f"\nRisk Classifier Scores:")
print(f"  Safe queries  - mean: {risk_scores[~is_high_risk].mean():.2f}, std: {risk_scores[~is_high_risk].std():.2f}")
print(f"  Risky queries - mean: {risk_scores[is_high_risk].mean():.2f}, std: {risk_scores[is_high_risk].std():.2f}")

print(f"\nRisk Probability (after sigmoid):")
risk_probs = 1 / (1 + np.exp(-(risk_scores - 5.0)))
print(f"  Safe queries  - mean: {risk_probs[~is_high_risk].mean():.3f}")
print(f"  Risky queries - mean: {risk_probs[is_high_risk].mean():.3f}")

print(f"\nFinal Confidence (base=0.8, penalty=0.7*risk_prob):")
final_conf = 0.8 - 0.6 * risk_probs
print(f"  Safe queries  - mean: {final_conf[~is_high_risk].mean():.3f}")
print(f"  Risky queries - mean: {final_conf[is_high_risk].mean():.3f}")

print(f"\nDoes risk classifier = ground truth?")
clf_high = risk_scores >= 5.0
agreement = (clf_high == is_high_risk).mean()
print(f"  Agreement: {agreement:.1%}")
