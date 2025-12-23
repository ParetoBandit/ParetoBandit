"""Check BanditGPT score distribution for unrestricted queries"""
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
from final_release.kdd_paper.table_3.router_evaluation import get_evaluator
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Load data
df = load_battle_dataset(1000)
df = run_judging_pipeline(df)

# Initialize and train
model_registry = load_model_registry()
bandit_router = BanditGPTRouter(model_registry)

print("\n[Burn-in] Training BanditGPT...")
run_bandit_burnin(df, bandit_router, n_burnin=500)

# Get shared evaluator (ensures consistency with Figure 9 and Table 3)
evaluator = get_evaluator(policy_threshold=5.0)

# Classify restricted queries using SHARED evaluator
print("\n[Policy] Classifying restricted queries...")
restricted = evaluator.classify_policy_restricted(
    df["question"].tolist(),
    desc="Policy classification"
)

# Score queries
print("\n[Scoring] Computing scores...")
scores = []
for q in tqdm(df["question"], desc="Scoring"):
    score = bandit_router.predict_proba(q)
    scores.append(score)

scores = np.array(scores)

# Analyze
print(f"\n{'='*60}")
print("SCORE DISTRIBUTION ANALYSIS")
print(f"{'='*60}")
print(f"\nTotal queries: {len(scores)}")
print(f"Restricted: {restricted.sum()} ({100*restricted.mean():.1f}%)")
print(f"Unrestricted: {(~restricted).sum()} ({100*(~restricted).mean():.1f}%)")

print(f"\n{'='*60}")
print("RESTRICTED QUERIES (Should be ~0.01)")
print(f"{'='*60}")
restricted_scores = scores[restricted]
print(f"  Mean: {restricted_scores.mean():.4f}")
print(f"  Std:  {restricted_scores.std():.4f}")
print(f"  Min:  {restricted_scores.min():.4f}")
print(f"  Max:  {restricted_scores.max():.4f}")

print(f"\n{'='*60}")
print("UNRESTRICTED QUERIES (Should span 0.3-0.7)")
print(f"{'='*60}")
unrestricted_scores = scores[~restricted]
print(f"  Mean: {unrestricted_scores.mean():.4f}")
print(f"  Std:  {unrestricted_scores.std():.4f}")
print(f"  Min:  {unrestricted_scores.min():.4f}")
print(f"  Max:  {unrestricted_scores.max():.4f}")

# Check spread
spread = unrestricted_scores.max() - unrestricted_scores.min()
print(f"\n  Spread: {spread:.4f}")
if spread < 0.2:
    print(f"  ⚠️  WARNING: Scores too clustered! Need more variation.")
    print(f"  → LinUCB dominated by policy, not learning query difficulty")
else:
    print(f"  ✓ Good spread - smooth cost/efficiency tradeoff possible")

# Distribution
print(f"\n{'='*60}")
print("SCORE HISTOGRAM (Unrestricted only)")
print(f"{'='*60}")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(unrestricted_scores, bins=bins)
for i in range(len(bins)-1):
    pct = 100 * hist[i] / len(unrestricted_scores)
    bar = '█' * int(pct / 2)
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {pct:5.1f}% {bar}")

print(f"\n{'='*60}")
print("SCORE HISTOGRAM (ALL queries - shows bimodal)")
print(f"{'='*60}")
hist_all, _ = np.histogram(scores, bins=bins)
for i in range(len(bins)-1):
    pct = 100 * hist_all[i] / len(scores)
    bar = '█' * int(pct / 2)
    count = hist_all[i]
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {pct:5.1f}% ({count:4d} queries) {bar}")

# Create visual histogram
print(f"\n{'='*60}")
print("Creating histogram plot...")
print(f"{'='*60}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram
bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
colors = ['#d62728' if bc < 0.15 else '#2ca02c' for bc in bin_centers]  # Red for restricted, green for unrestricted

ax.bar(bin_centers, hist_all, width=0.08, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Annotations
ax.axvline(0.01, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Policy Threshold (0.01)')
ax.text(0.01, max(hist_all) * 0.95, 'RESTRICTED\n(Policy Violation)', 
        ha='left', va='top', fontsize=10, fontweight='bold', color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
ax.text(0.4, max(hist_all) * 0.95, 'UNRESTRICTED\n(Safe to Route)', 
        ha='center', va='top', fontsize=10, fontweight='bold', color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8))

# Labels
ax.set_xlabel('BanditGPT Confidence Score (Probability of Using Weak Model)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Queries', fontsize=12, fontweight='bold')
ax.set_title('BanditGPT Score Distribution After Safety-Aware Training\n(Bimodal: Restricted vs Unrestricted)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.05, 1.05)

# Stats annotation
stats_text = f"Total Queries: {len(scores)}\n"
stats_text += f"Restricted: {restricted.sum()} ({100*restricted.mean():.1f}%)\n"
stats_text += f"Unrestricted: {(~restricted).sum()} ({100*(~restricted).mean():.1f}%)"
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
output_path = Path(__file__).parent / "banditgpt_score_distribution.png"
plt.savefig(output_path, dpi=150, facecolor='white')
print(f"✓ Histogram saved to {output_path}")
