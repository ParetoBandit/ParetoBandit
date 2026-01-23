#!/usr/bin/env python3
"""
Generate Figure 1 for KDD submission: Addressing Data Loss and Structural Uncertainty.

Panel A: Sigmoid vs. Hard Clipping (The "Normalization Cliff")
Panel B: Covariance Matrix Structure (Cold Start vs. Procedural Warmup)

Uses REAL data from BanditRouter to demonstrate the mathematical intuition.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.router import BanditRouter, sigmoid

def old_hard_clip(x, min_val=-0.15, max_val=0.25):
    """Old normalization: hard clipping creates information loss."""
    normalized = (x - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)

def new_sigmoid_norm(x, mu=-0.0037, sigma=0.095):
    """New normalization: sigmoid preserves gradient everywhere."""
    k = 1.0 / sigma
    z = k * (x - mu)
    return sigmoid(z)

def create_panel_a():
    """Panel A: Sigmoid vs Hard Clipping."""
    # Generate range of raw complexity projections
    x_raw = np.linspace(-0.5, 0.5, 500)
    
    # Old approach (hard clipping)
    y_clip = old_hard_clip(x_raw)
    
    # New approach (sigmoid)
    y_sigmoid = np.array([new_sigmoid_norm(x) for x in x_raw])
    
    # Create subplot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot old approach
    ax.plot(x_raw, y_clip, 'r--', linewidth=2.5, label='Hard Clipping (Old)', alpha=0.8)
    
    # Plot new approach
    ax.plot(x_raw, y_sigmoid, 'b-', linewidth=2.5, label='Sigmoid (New)', alpha=0.9)
    
    # Annotations
    # Information loss zones
    ax.axhspan(0.95, 1.0, alpha=0.15, color='red')
    ax.text(0.25, 0.975, 'Information Loss\n(Zero Gradient)', 
            fontsize=9, ha='center', va='center', color='darkred',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    # Sigmoid advantage
    ax.annotate('Retains Sensitivity\nto Outliers', 
                xy=(0.35, new_sigmoid_norm(0.35)), xytext=(0.4, 0.6),
                fontsize=9, color='darkblue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark calibration point
    ax.axvline(-0.0037, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(-0.0037, 0.05, 'μ (LMSYS)', fontsize=8, ha='center', color='green')
    
    # Labels and formatting
    ax.set_xlabel('Raw Complexity Projection', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Signal into Bandit', fontsize=11, fontweight='bold')
    ax.set_title('Panel A: Sigmoid vs. Hard Clipping\n("Normalization Cliff" Fix)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.05, 1.05)
    
    return fig, ax

def create_panel_b():
    """Panel B: Covariance Matrix Structure using REAL router data."""
    print("\n" + "="*70)
    print("PANEL B: EXTRACTING REAL COVARIANCE MATRICES")
    print("="*70)
    
    # Load models
    project_root = Path(__file__).parent.parent.parent.parent
    models_path = project_root / "banditgpt" / "models.json"
    
    with open(models_path) as f:
        registry = {m["openrouter_id"]: m for m in json.load(f)["models"]}
    
    print(f"\n✓ Loaded {len(registry)} models")
    
    # Create COLD START router (no warmup)
    print("\n--- Creating Cold Start Router (A = I) ---")
    router_cold = BanditRouter.create(
        model_registry=registry,
        priors="none",  # Cold start: identity initialization, no procedural warmup
        prior_n_effective=20.0
    )
    
    # Get a sample model's A_inv
    sample_model = list(router_cold.bandit.models)[0]
    A_inv_cold = router_cold.bandit.A_inv[sample_model]
    
    print(f"✓ Cold start A_inv shape: {A_inv_cold.shape}")
    print(f"  Diagonal dominance: {np.mean(np.diag(A_inv_cold)) / np.mean(np.abs(A_inv_cold)):.2f}x")
    
    # Create WARMUP router
    print("\n--- Creating Warmup Router (Procedural Warmup) ---")
    router_warm = BanditRouter.create(
        model_registry=registry,
        priors="hle",  # HLE mode: includes procedural warmup
        prior_n_effective=20.0
    )
    
    A_inv_warm = router_warm.bandit.A_inv[sample_model]
    
    print(f"✓ Warmup A_inv shape: {A_inv_warm.shape}")
    print(f"  Diagonal dominance: {np.mean(np.diag(A_inv_warm)) / np.mean(np.abs(A_inv_warm)):.2f}x")
    
    # Feature indices (from router.py structure)
    # [Embedding(32) | Handcrafted(15) | Anchors(5) | Complexity(1) | Bias(1)]
    EMB_DIM = 32
    HANDCRAFTED_DIM = 15
    ANCHOR_DIM = 5
    
    handcrafted_start = EMB_DIM
    anchor_start = EMB_DIM + HANDCRAFTED_DIM
    complexity_idx = EMB_DIM + HANDCRAFTED_DIM + ANCHOR_DIM
    
    # Specific indices
    IDX_HAS_LATEX = handcrafted_start + 9
    IDX_LATEX_LOG = handcrafted_start + 10
    IDX_MATH_ANCHOR = anchor_start + 1
    
    print(f"\n📍 Feature Indices:")
    print(f"  Math Anchor: {IDX_MATH_ANCHOR}")
    print(f"  Has LaTeX: {IDX_HAS_LATEX}")
    print(f"  LaTeX Log: {IDX_LATEX_LOG}")
    
    # Extract correlation
    cold_corr = A_inv_cold[IDX_MATH_ANCHOR, IDX_HAS_LATEX]
    warm_corr = A_inv_warm[IDX_MATH_ANCHOR, IDX_HAS_LATEX]
    
    print(f"\n🔗 Math ↔ LaTeX Correlation:")
    print(f"  Cold Start: {cold_corr:.6f}")
    print(f"  After Warmup: {warm_corr:.6f}")
    print(f"  Ratio: {abs(warm_corr / cold_corr):.2f}x stronger")
    
    # Create figure with two heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Focus on the relevant feature block (handcrafted + anchors)
    focus_start = handcrafted_start
    focus_end = complexity_idx
    
    A_cold_block = A_inv_cold[focus_start:focus_end, focus_start:focus_end]
    A_warm_block = A_inv_warm[focus_start:focus_end, focus_start:focus_end]
    
    # Normalize for visualization (use same scale)
    vmax = max(np.abs(A_cold_block).max(), np.abs(A_warm_block).max())
    
    # Cold start heatmap
    im1 = ax1.imshow(A_cold_block, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax1.set_title('Cold Start ($A = I$)\nSpherical Uncertainty', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Feature Index', fontsize=10)
    ax1.set_ylabel('Feature Index', fontsize=10)
    
    # Warmup heatmap
    im2 = ax2.imshow(A_warm_block, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax2.set_title('After Warmup ($A \\approx \\Sigma + xx^T$)\nStructured Uncertainty', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Feature Index', fontsize=10)
    ax2.set_ylabel('Feature Index', fontsize=10)
    
    # Highlight Math ↔ LaTeX correlation
    math_idx = IDX_MATH_ANCHOR - focus_start
    latex_idx = IDX_HAS_LATEX - focus_start
    
    # Draw rectangles
    rect1 = mpatches.Rectangle((latex_idx-0.5, math_idx-0.5), 1, 1, 
                                linewidth=2, edgecolor='lime', facecolor='none')
    rect2 = mpatches.Rectangle((latex_idx-0.5, math_idx-0.5), 1, 1, 
                                linewidth=2, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    
    # Add colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    cbar.set_label('Covariance Magnitude', fontsize=10)
    
    # Add text annotation
    ax2.text(latex_idx, math_idx + 2, 'Math ↔ LaTeX\nCorrelation', 
             fontsize=8, ha='center', color='lime', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.suptitle('Panel B: Covariance Matrix Structure\n("Loose Initialization" Fix)', 
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig, (ax1, ax2), router_cold, router_warm

def main():
    print("=" * 70)
    print("GENERATING KDD FIGURE 1: ADDRESSING DATA LOSS & STRUCTURAL UNCERTAINTY")
    print("=" * 70)
    
    # Create the full two-panel figure
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Panel A: Sigmoid vs Clipping
    print("\n--- Creating Panel A: Sigmoid vs Hard Clipping ---")
    ax_a = fig.add_subplot(gs[0, 0])
    
    x_raw = np.linspace(-0.5, 0.5, 500)
    y_clip = old_hard_clip(x_raw)
    y_sigmoid = np.array([new_sigmoid_norm(x) for x in x_raw])
    
    ax_a.plot(x_raw, y_clip, 'r--', linewidth=2.5, label='Hard Clipping (Old)', alpha=0.8)
    ax_a.plot(x_raw, y_sigmoid, 'b-', linewidth=2.5, label='Sigmoid (New)', alpha=0.9)
    
    ax_a.axhspan(0.95, 1.0, alpha=0.15, color='red')
    ax_a.text(0.25, 0.975, 'Information Loss\n(Zero Gradient)', 
              fontsize=9, ha='center', va='center', color='darkred',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax_a.annotate('Retains Sensitivity\nto Outliers', 
                  xy=(0.35, new_sigmoid_norm(0.35)), xytext=(0.4, 0.6),
                  fontsize=9, color='darkblue',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'),
                  arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    ax_a.axvline(-0.0037, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_a.text(-0.0037, 0.05, 'μ (LMSYS)', fontsize=8, ha='center', color='green')
    
    ax_a.set_xlabel('Raw Complexity Projection', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Normalized Signal into Bandit', fontsize=11, fontweight='bold')
    ax_a.set_title('Panel A: Sigmoid vs. Hard Clipping\n("Normalization Cliff" Fix)', 
                   fontsize=12, fontweight='bold', pad=10)
    ax_a.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    ax_a.set_xlim(-0.5, 0.5)
    ax_a.set_ylim(-0.05, 1.05)
    
    print("✓ Panel A complete")
    
    # Panel B: Covariance matrices
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis('off')  # Placeholder, we'll create subplots within
    
    # Create Panel B separately to extract real data
    fig_b, axes_b, router_cold, router_warm = create_panel_b()
    
    # Save individual panels for inspection
    output_dir = Path(__file__).parent
    
    fig_a, _ = create_panel_a()
    fig_a.savefig(output_dir / 'panel_a_sigmoid_vs_clipping.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved Panel A to: {output_dir / 'panel_a_sigmoid_vs_clipping.png'}")
    
    fig_b.savefig(output_dir / 'panel_b_covariance_structure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved Panel B to: {output_dir / 'panel_b_covariance_structure.png'}")
    
    print("\n" + "=" * 70)
    print("✅ FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  1. panel_a_sigmoid_vs_clipping.png")
    print(f"  2. panel_b_covariance_structure.png")
    print("\n📊 These figures demonstrate:")
    print("  • Panel A: How sigmoid normalization preserves gradient vs hard clipping")
    print("  • Panel B: How procedural warmup learns feature correlations (e.g., Math ↔ LaTeX)")
    print("\n🎯 Ready for KDD submission!")

if __name__ == "__main__":
    main()
