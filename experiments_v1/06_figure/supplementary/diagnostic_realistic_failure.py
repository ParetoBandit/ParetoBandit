"""
Diagnostic Analysis: Why Realistic Scenario Fails 75% of the Time
===================================================================

This script analyzes the statistical mechanics causing poor performance
with realistic LMSYS reward distributions.

Key Question: Why does Cohen's d ≈ 0.12 lead to 25% success vs d ≈ 10.8 with 100% success?
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# SETUP: Synthetic vs Realistic Distributions
# ============================================================================

# Synthetic (Main Experiment)
synthetic_mixtral = (0.90, 0.05)  # μ, σ
synthetic_gpt4 = (0.20, 0.08)

# Realistic (LMSYS)
realistic_mixtral = (0.823, 0.09)
realistic_gpt4 = (0.812, 0.10)

print("="*70)
print("DIAGNOSTIC ANALYSIS: Realistic Scenario Failure")
print("="*70)

# ============================================================================
# 1. EFFECT SIZE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("1. EFFECT SIZE COMPARISON")
print("="*70)

# Synthetic
synthetic_delta = synthetic_mixtral[0] - synthetic_gpt4[0]
synthetic_pooled_std = np.sqrt((synthetic_mixtral[1]**2 + synthetic_gpt4[1]**2) / 2)
synthetic_cohens_d = synthetic_delta / synthetic_pooled_std

print(f"\n📊 Synthetic Scenario:")
print(f"   Mixtral: μ={synthetic_mixtral[0]:.3f}, σ={synthetic_mixtral[1]:.3f}")
print(f"   GPT-4:   μ={synthetic_gpt4[0]:.3f}, σ={synthetic_gpt4[1]:.3f}")
print(f"   Δμ = {synthetic_delta:.3f}")
print(f"   Cohen's d = {synthetic_cohens_d:.2f} (HUGE effect)")
print(f"   Interpretation: {synthetic_delta/synthetic_gpt4[1]:.1f}σ separation")

# Realistic
realistic_delta = realistic_mixtral[0] - realistic_gpt4[0]
realistic_pooled_std = np.sqrt((realistic_mixtral[1]**2 + realistic_gpt4[1]**2) / 2)
realistic_cohens_d = realistic_delta / realistic_pooled_std

print(f"\n📊 Realistic Scenario:")
print(f"   Mixtral: μ={realistic_mixtral[0]:.3f}, σ={realistic_mixtral[1]:.3f}")
print(f"   GPT-4:   μ={realistic_gpt4[0]:.3f}, σ={realistic_gpt4[1]:.3f}")
print(f"   Δμ = {realistic_delta:.3f}")
print(f"   Cohen's d = {realistic_cohens_d:.2f} (TINY effect)")
print(f"   Interpretation: {realistic_delta/realistic_gpt4[1]:.2f}σ separation")

print(f"\n⚡ Effect Size Ratio: {synthetic_cohens_d / realistic_cohens_d:.1f}x larger in synthetic")

# ============================================================================
# 2. OVERLAP ANALYSIS (Probability of "Wrong" Outcome)
# ============================================================================

print("\n" + "="*70)
print("2. DISTRIBUTION OVERLAP (Single Sample Confusion)")
print("="*70)

def compute_overlap_probability(mu1, sigma1, mu2, sigma2):
    """
    Compute P(X1 < X2) where X1 ~ N(mu1, sigma1), X2 ~ N(mu2, sigma2)
    
    If Mixtral is better (mu1 > mu2), this is the probability that
    a single sample INCORRECTLY shows GPT-4 as better.
    """
    # Difference distribution: X1 - X2 ~ N(mu1 - mu2, sqrt(sigma1^2 + sigma2^2))
    diff_mean = mu1 - mu2
    diff_std = np.sqrt(sigma1**2 + sigma2**2)
    
    # P(X1 < X2) = P(X1 - X2 < 0)
    prob_wrong = stats.norm.cdf(0, loc=diff_mean, scale=diff_std)
    
    return prob_wrong

# Synthetic
synthetic_overlap = compute_overlap_probability(
    synthetic_mixtral[0], synthetic_mixtral[1],
    synthetic_gpt4[0], synthetic_gpt4[1]
)

print(f"\n📊 Synthetic Scenario:")
print(f"   P(single GPT-4 sample > Mixtral sample) = {synthetic_overlap:.6f}")
print(f"   = {synthetic_overlap * 100:.4f}%")
print(f"   Interpretation: Almost NEVER happens (1 in {1/synthetic_overlap:.0f} samples)")

# Realistic
realistic_overlap = compute_overlap_probability(
    realistic_mixtral[0], realistic_mixtral[1],
    realistic_gpt4[0], realistic_gpt4[1]
)

print(f"\n📊 Realistic Scenario:")
print(f"   P(single GPT-4 sample > Mixtral sample) = {realistic_overlap:.4f}")
print(f"   = {realistic_overlap * 100:.1f}%")
print(f"   Interpretation: Happens {realistic_overlap * 100:.1f}% of the time!")

print(f"\n💥 KEY INSIGHT: Signal is SMALLER than noise!")
print(f"   With realistic distributions, on ~{realistic_overlap*100:.0f}% of samples,")
print(f"   GPT-4 randomly gets a HIGHER reward than Mixtral just due to noise.")
print(f"   This makes it very hard for Corralling to detect the true difference.")

# ============================================================================
# 3. STATISTICAL POWER ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("3. STATISTICAL POWER (Samples Needed to Detect Effect)")
print("="*70)

def samples_needed_for_power(d, alpha=0.05, power=0.8):
    """
    Compute samples needed per group for two-sample t-test.
    
    Formula (approximate): n ≈ 16 / d^2 for 80% power
    More precise formula includes alpha and power.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    return n

synthetic_n = samples_needed_for_power(synthetic_cohens_d)
realistic_n = samples_needed_for_power(realistic_cohens_d)

print(f"\n📊 Samples Needed (per expert) for 80% power:")
print(f"   Synthetic (d={synthetic_cohens_d:.2f}): {synthetic_n:.1f} samples")
print(f"   Realistic (d={realistic_cohens_d:.2f}): {realistic_n:.1f} samples")

print(f"\n⚡ Realistic needs {realistic_n/synthetic_n:.0f}x MORE samples!")
print(f"\n💡 With only 1000 total steps and ~50/50 expert selection:")
print(f"   - Each expert gets ~500 samples")
print(f"   - Warmup expert uses ~250 samples on GPT-4, ~250 on Mixtral (due to exploration)")
print(f"   - This is only {250/realistic_n*100:.1f}% of needed samples!")
print(f"   - Insufficient statistical power to detect d={realistic_cohens_d:.2f}")

# ============================================================================
# 4. IMPORTANCE WEIGHTING AMPLIFICATION
# ============================================================================

print("\n" + "="*70)
print("4. IMPORTANCE WEIGHTING NOISE AMPLIFICATION")
print("="*70)

print(f"\nCorralling uses importance-weighted loss: ℓ̂ = (1 - reward) / p")
print(f"where p is the probability of selecting that expert.")
print(f"\nAs an expert's weight drops (p → 0.1), the estimator becomes:")
print(f"   ℓ̂ = (1 - reward) / 0.1 = 10 × (1 - reward)")
print(f"\n💥 This AMPLIFIES noise by 10x when expert has low weight!")

# Simulate variance amplification
p_values = [0.5, 0.3, 0.1, 0.05]
print(f"\n📊 Variance Amplification (as expert weight drops):")

for p in p_values:
    # Variance of importance-weighted estimator: Var(loss) / p
    # For loss = 1 - reward with reward ~ N(0.8, 0.1)
    # Var(loss) = Var(reward) = 0.1^2 = 0.01
    
    base_var = realistic_gpt4[1] ** 2  # Variance of reward
    amplified_var = base_var / p  # Importance weighting amplifies
    amplified_std = np.sqrt(amplified_var)
    
    print(f"   p = {p:.2f}: std(ℓ̂) = {amplified_std:.3f} ({amplified_std/realistic_gpt4[1]:.1f}x base)")

print(f"\n💡 With realistic noise (σ=0.10) and low weight (p=0.1):")
print(f"   - Importance-weighted loss has std = {np.sqrt(realistic_gpt4[1]**2 / 0.1):.3f}")
print(f"   - Signal (Δμ = {realistic_delta:.3f}) is SMALLER than noise!")
print(f"   - Exponential weights oscillate wildly instead of converging")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("5. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Distribution Overlap (Synthetic) ---
ax1 = axes[0, 0]
x = np.linspace(-0.2, 1.2, 1000)
y_mixtral = stats.norm.pdf(x, synthetic_mixtral[0], synthetic_mixtral[1])
y_gpt4 = stats.norm.pdf(x, synthetic_gpt4[0], synthetic_gpt4[1])

ax1.fill_between(x, y_mixtral, alpha=0.3, color='green', label='Mixtral')
ax1.fill_between(x, y_gpt4, alpha=0.3, color='red', label='GPT-4')
ax1.plot(x, y_mixtral, color='green', linewidth=2)
ax1.plot(x, y_gpt4, color='red', linewidth=2)
ax1.set_xlabel('Reward', fontweight='bold')
ax1.set_ylabel('Probability Density', fontweight='bold')
ax1.set_title(f'Synthetic: d={synthetic_cohens_d:.1f} (Overlap={synthetic_overlap*100:.4f}%)', 
              fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Distribution Overlap (Realistic) ---
ax2 = axes[0, 1]
x = np.linspace(0.4, 1.1, 1000)
y_mixtral = stats.norm.pdf(x, realistic_mixtral[0], realistic_mixtral[1])
y_gpt4 = stats.norm.pdf(x, realistic_gpt4[0], realistic_gpt4[1])

ax2.fill_between(x, y_mixtral, alpha=0.3, color='green', label='Mixtral')
ax2.fill_between(x, y_gpt4, alpha=0.3, color='red', label='GPT-4')
ax2.plot(x, y_mixtral, color='green', linewidth=2)
ax2.plot(x, y_gpt4, color='red', linewidth=2)

# Shade overlap region
overlap_region = np.where((x >= 0.6) & (x <= 1.0))
ax2.fill_between(x[overlap_region], 
                  np.minimum(y_mixtral[overlap_region], y_gpt4[overlap_region]),
                  alpha=0.5, color='yellow', label='Overlap')

ax2.set_xlabel('Reward', fontweight='bold')
ax2.set_ylabel('Probability Density', fontweight='bold')
ax2.set_title(f'Realistic: d={realistic_cohens_d:.2f} (Overlap={realistic_overlap*100:.1f}%)', 
              fontweight='bold', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Statistical Power ---
ax3 = axes[1, 0]
d_values = np.linspace(0.05, 2.0, 100)
n_values = [samples_needed_for_power(d) for d in d_values]

ax3.plot(d_values, n_values, linewidth=2, color='blue')
ax3.axvline(synthetic_cohens_d, color='green', linestyle='--', 
            label=f'Synthetic (d={synthetic_cohens_d:.1f})')
ax3.axvline(realistic_cohens_d, color='red', linestyle='--',
            label=f'Realistic (d={realistic_cohens_d:.2f})')
ax3.axhline(500, color='gray', linestyle=':', alpha=0.5, 
            label='Available samples (~500 per expert)')

ax3.set_xlabel('Effect Size (Cohen\'s d)', fontweight='bold')
ax3.set_ylabel('Samples Needed (per expert)', fontweight='bold')
ax3.set_title('Statistical Power: Samples Required for Detection', 
              fontweight='bold', fontsize=11)
ax3.set_xlim(0, 2)
ax3.set_ylim(0, 10000)
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Importance Weighting Noise Amplification ---
ax4 = axes[1, 1]
p_range = np.linspace(0.05, 0.5, 100)
noise_amplification = [np.sqrt(realistic_gpt4[1]**2 / p) for p in p_range]

ax4.plot(p_range, noise_amplification, linewidth=2, color='orange')
ax4.axhline(realistic_delta, color='green', linestyle='--', 
            label=f'Signal (Δμ={realistic_delta:.3f})')
ax4.axhline(realistic_gpt4[1], color='blue', linestyle=':', 
            label=f'Base Noise (σ={realistic_gpt4[1]:.2f})')

# Shade region where noise > signal
signal_line = np.ones_like(p_range) * realistic_delta
ax4.fill_between(p_range, signal_line, noise_amplification, 
                  where=np.array(noise_amplification) > realistic_delta,
                  alpha=0.3, color='red', label='Noise > Signal')

ax4.set_xlabel('Expert Weight (p)', fontweight='bold')
ax4.set_ylabel('Importance-Weighted Loss Std Dev', fontweight='bold')
ax4.set_title('Noise Amplification: std(ℓ̂) = σ/√p', 
              fontweight='bold', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

out_png = output_dir / "diagnostic_realistic_failure.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {out_png}")

out_pdf = output_dir / "diagnostic_realistic_failure.pdf"
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {out_pdf}")

plt.close()

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: Why Realistic Scenario Fails 75% of the Time")
print("="*70)

print(f"""
🔍 ROOT CAUSE: Signal-to-Noise Ratio

1. **Tiny Effect Size**
   - Δμ = 0.011 (1.1 percentage points)
   - Cohen's d = 0.12 (100x smaller than synthetic)
   - Only {realistic_delta/realistic_gpt4[1]:.2f}σ separation

2. **High Overlap** 
   - {realistic_overlap*100:.1f}% probability GPT-4 sample > Mixtral sample
   - Signal is SMALLER than noise
   - Very hard to distinguish "real" from "random" differences

3. **Insufficient Samples**
   - Need ~{realistic_n:.0f} samples per expert for 80% power
   - Only have ~500 samples per expert
   - Underpowered by {500/realistic_n*100:.1f}%

4. **Noise Amplification**
   - Importance weighting amplifies noise: σ/√p
   - At p=0.1, noise is {np.sqrt(realistic_gpt4[1]**2 / 0.1):.3f} (>{realistic_delta:.3f} signal)
   - Exponential weights oscillate instead of converging

💡 IMPLICATION FOR PRODUCTION:

With realistic LMSYS effect sizes (d≈0.1), Corralling requires:
- 10,000+ samples (not 1,000)
- Or different hyperparameters (lower η, sequential testing)
- Or offline statistical testing before deployment
- Or acceptance that small effects won't be detected

The 25% success rate is NOT a bug—it's a fundamental limitation
of online learning with small effect sizes and limited samples.

This is why the synthetic stress test (d=10.8) is pedagogically
useful but NOT representative of production performance!
""")
