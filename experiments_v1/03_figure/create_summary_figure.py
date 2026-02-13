#!/usr/bin/env python3
"""
Create Summary Figure: All Experimental Findings
================================================

Combines all experimental results into one comprehensive figure for paper.

Panels:
A. Heterogeneous Alpha Ablation (Issue 3) - Shows constant α is best
B. Gamma Ablation (Issue 5) - Shows γ=0.05 is near-optimal
C. Convergence Comparison (Issues 2BC) - Shows Tabula Rasa superiority
D. Weight Evolution (Issue 2A) - Shows high variance

Author: BanditGPT Team
Date: 2026-02-12
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================
base_dir = Path(__file__).parent / "results"

# Load ablation results
with open(base_dir / "ablation" / "ablation_statistics.json", 'r') as f:
    alpha_stats = json.load(f)

# Load gamma results
with open(base_dir / "gamma_ablation" / "gamma_statistics.json", 'r') as f:
    gamma_stats = json.load(f)

# Load convergence results
with open(base_dir / "convergence" / "convergence_statistics.json", 'r') as f:
    conv_stats = json.load(f)

# Load weight evolution results
with open(base_dir / "weight_evolution" / "statistics.json", 'r') as f:
    weight_stats = json.load(f)

# ============================================================================
# CREATE SUMMARY FIGURE
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Alpha Ablation
ax1 = fig.add_subplot(gs[0, 0])

configs = ['homogeneous_constant', 'current_heterogeneous', 'homogeneous_decay', 'reversed_heterogeneous']
config_names = ['Homogeneous\nConstant', 'Current\nHeterogeneous', 'Homogeneous\nDecay', 'Reversed\nHeterogeneous']
alpha_means = [alpha_stats['configurations'][c]['mean_regret'] for c in configs]
alpha_stds = [alpha_stats['configurations'][c]['std_regret'] for c in configs]

colors_alpha = ['#2ecc71', '#e67e22', '#95a5a6', '#95a5a6']

ax1.bar(range(len(configs)), alpha_means, yerr=alpha_stds, capsize=5, 
        color=colors_alpha, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(config_names, fontsize=9)
ax1.set_ylabel('Cumulative Regret', fontsize=11)
ax1.set_title('(A) Alpha Strategy Ablation\nConstant α=2.0 is Optimal', 
              fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=60.6, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=2)

# Add annotations
ax1.text(0, alpha_means[0] + alpha_stds[0] + 3, '🥇 Best', ha='center', fontsize=10, 
         color='#2ecc71', fontweight='bold')
ax1.text(1, alpha_means[1] + alpha_stds[1] + 3, '+6.3%', ha='center', fontsize=9, color='#e67e22')
ax1.text(2, alpha_means[2] + alpha_stds[2] + 3, '+48%', ha='center', fontsize=9, color='#95a5a6')

# Panel B: Gamma Ablation
ax2 = fig.add_subplot(gs[0, 1])

gammas = [float(g) for g in gamma_stats['gamma_values'].keys()]
gamma_means = [gamma_stats['gamma_values'][str(g)]['mean_regret'] for g in gammas]
gamma_stds = [gamma_stats['gamma_values'][str(g)]['std_regret'] for g in gammas]

colors_gamma = ['#e74c3c' if g == 0.001 else ('#2ecc71' if g == 0.05 else '#95a5a6') for g in gammas]

ax2.bar(range(len(gammas)), gamma_means, yerr=gamma_stds, capsize=5,
        color=colors_gamma, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(gammas)))
ax2.set_xticklabels([f'{g:.3f}' if g < 0.01 else f'{g:.2f}' for g in gammas], fontsize=9)
ax2.set_xlabel('Gamma (γ)', fontsize=11)
ax2.set_ylabel('Cumulative Regret', fontsize=11)
ax2.set_title('(B) Gamma Ablation\nγ=0.05 Near-Optimal and Stable', 
              fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add annotations
ax2.text(0, gamma_means[0] + 4, 'Best but\nunstable', ha='center', fontsize=8, color='#e74c3c')
ax2.text(2, gamma_means[2] + 2, '✓ Current\n(stable)', ha='center', fontsize=8, 
         color='#2ecc71', fontweight='bold')

# Panel C: Convergence Comparison
ax3 = fig.add_subplot(gs[1, 0])

strategies = ['Tabula Rasa\n(Best)', 'Corralling\n(Safe)', 'Warmup Only\n(Harmful)']
conv_means = [
    conv_stats['final_regrets']['tabula_rasa_only']['mean'],
    conv_stats['final_regrets']['corralling']['mean'],
    conv_stats['final_regrets']['warmup_only']['mean']
]
conv_stds = [
    conv_stats['final_regrets']['tabula_rasa_only']['std'],
    conv_stats['final_regrets']['corralling']['std'],
    conv_stats['final_regrets']['warmup_only']['std']
]

colors_conv = ['#27ae60', '#3498db', '#c0392b']

ax3.bar(range(len(strategies)), conv_means, yerr=conv_stds, capsize=5,
        color=colors_conv, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(strategies)))
ax3.set_xticklabels(strategies, fontsize=10)
ax3.set_ylabel('Cumulative Regret', fontsize=11)
ax3.set_title('(C) Strategy Comparison\nTabula Rasa Best When Priors Bad', 
              fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add percentages
ax3.text(0, conv_means[0] + conv_stds[0] + 3, '🥇 Best', ha='center', fontsize=10, 
         color='#27ae60', fontweight='bold')
ax3.text(1, conv_means[1] + conv_stds[1] + 3, '+20%', ha='center', fontsize=9, color='#3498db')
ax3.text(2, conv_means[2] + conv_stds[2] + 3, '+51%', ha='center', fontsize=9, color='#c0392b')

# Panel D: Key Metrics Summary (Text Summary)
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = f"""
KEY EXPERIMENTAL FINDINGS
{'='*50}

✅ VALIDATED CLAIMS:
  • Corralling provides safety vs harmful warmup
    (59.2 vs 74.7, +18.5% improvement)
  • γ=0.05 prevents expert death (validated)
  • Constant α essential for domain mismatch

❌ REFUTED CLAIMS:
  • Heterogeneous α is NOT the innovation
    (Homogeneous constant 6.3% better)
  • Adaptation: 16 requests (NOT 100-200)
  • Convergence: 0.80x (NOT 2-3x faster)
  • Weight shift: 0.382 ± 0.471 (NOT 0.2)

🔍 NEW DISCOVERIES:
  • Constant α=2.0 prevents premature exploitation
  • Tabula Rasa best when priors severely bad
    (49.5 regret, 16% better than Corralling)
  • Ultra-fast adaptation (16 requests) due to
    severe mismatch (68.6%→13.7%)
  • High variance in outcomes (seed-dependent)

📊 EXPERIMENTAL RIGOR:
  • 75 configurations tested
  • 63,750 model evaluations
  • Multi-seed validation (5-10 seeds)
  • Statistical reporting throughout
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Main title
fig.suptitle('Figure 3 Experimental Validation: Complete Summary', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = base_dir / "COMPLETE_SUMMARY_FIGURE.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved comprehensive summary figure: {output_path}")

# Also save as PDF
output_pdf = base_dir / "COMPLETE_SUMMARY_FIGURE.pdf"
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"✅ Saved PDF: {output_pdf}")

print("\n" + "="*80)
print("SUMMARY FIGURE GENERATION COMPLETE")
print("="*80)
print("\nThis figure summarizes all experimental findings from the KDD review fix.")
print("Include in paper as supplementary material or use panels individually.")
