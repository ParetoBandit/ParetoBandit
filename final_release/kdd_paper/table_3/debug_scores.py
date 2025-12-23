"""Debug script to check BanditGPT score distribution"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from final_release.kdd_paper.table_3.router_performance_comparison import (
    load_battle_dataset,
    load_model_registry,
    run_judging_pipeline,
    run_bandit_burnin,
    BanditGPTRouter,
)
import numpy as np
from tqdm import tqdm

# Load samples
df = load_battle_dataset(1000)
df = run_judging_pipeline(df)

model_registry = load_model_registry()
bandit_router = BanditGPTRouter(model_registry)

# BURN-IN: Train BanditGPT first
print("\n[Burn-in] Training BanditGPT...")
run_bandit_burnin(df, bandit_router, n_burnin=500)
print("✓ Burn-in complete")

# Score queries AFTER burn-in
scores = []
is_high_risk = (df["weak_is_valid"] == 0).values

print("\n[Scoring] Computing scores after burn-in...")
for q in tqdm(df["question"]):
    score = bandit_router.predict_proba(q)
    scores.append(score)

scores = np.array(scores)

# Analyze
print(f"\nTotal queries: {len(scores)}")
print(f"High-risk: {is_high_risk.sum()} ({100*is_high_risk.mean():.1f}%)")
print(f"Safe: {(~is_high_risk).sum()} ({100*(~is_high_risk).mean():.1f}%)")

print(f"\nBanditGPT Scores:")
print(f"  Safe queries - mean: {scores[~is_high_risk].mean():.3f}, std: {scores[~is_high_risk].std():.3f}")
print(f"  Risky queries - mean: {scores[is_high_risk].mean():.3f}, std: {scores[is_high_risk].std():.3f}")

print(f"\nAt threshold=0.5:")
to_weak = scores >= 0.5
print(f"  Traffic to weak: {to_weak.mean():.1%}")
print(f"  Leakage (risky sent to weak): {to_weak[is_high_risk].mean():.1%}")
