# Figure 2 (Distribution Shift) Caption Correction

**Date**: January 25, 2026  
**Issue**: Bottom subplot description was incorrect

---

## Problem Identified

The original caption's description of the **(Bottom)** subplot was misleading. It stated that the bottom subplot shows "a higher prevalence of 'Easy Tasks' **in production**" - but the bottom subplot actually displays the **training/source data** decomposition, not the production/deployment data.

### Original Caption
```latex
\caption{Feature Distribution Shift. (Top) The deployment distribution (Red) 
is significantly left-shifted compared to the training data (Blue), with PSI=0.275. 
(Bottom) This shift is driven by a higher prevalence of ``Easy Tasks'' in production 
than anticipated.}
```

### What Each Subplot Actually Shows

**Top Subplot:**
- Blue curve: Source/Prior Data (training) - mean = 0.060
- Red curve: RouteLLM Data (deployment) - mean = -0.004
- Red IS left-shifted compared to Blue ✅ **Caption was CORRECT**
- Mean shift = -0.064 (negative = toward Easy cluster)

**Bottom Subplot:**
- Title: "Source/Prior Data: Easy vs Hard Task Distribution"
- Green curve: **Source** Easy tasks (PC1 < 0.0): 45.4%, centered at -0.105
- Purple curve: **Source** Hard tasks (PC1 > 0.2): 22.4%, centered at 0.365
- Purpose: Explains the **bimodal structure of the training data**
- **NOT showing production/deployment data** ❌ **Caption was INCORRECT**

---

## The Confusion

The caption conflated two different things:

1. **Top subplot interpretation**: The left-shift of RouteLLM (red) vs Source (blue) DOES indicate more easy tasks in production - this is correct!

2. **Bottom subplot content**: But the bottom subplot doesn't show production data at all. It shows how the **training data itself** has a bimodal structure with two clusters.

The bottom subplot's purpose is to explain **why the distribution shift matters** - because the training data has distinct Easy and Hard clusters, so when deployment shifts toward the Easy cluster, the priors learned from the mixed training distribution become miscalibrated.

---

## Solution

### Corrected Caption
```latex
\caption{Feature Distribution Shift. (Top) The deployment distribution (Red) 
is significantly left-shifted compared to the training data (Blue), with PSI=0.275, 
indicating more easy tasks in production. (Bottom) The training data exhibits a 
bimodal structure with two distinct task clusters: Easy tasks (45.4\%, PC1 $<$ 0.0) 
and Hard tasks (22.4\%, PC1 $>$ 0.2), explaining why the distribution shift impacts 
prior calibration.}
```

### Key Changes

1. **Top description**: Added "indicating more easy tasks in production" to clarify the interpretation of the left-shift
2. **Bottom description**: Now correctly states it shows "The training data exhibits a bimodal structure..."
3. **Connection**: Added "explaining why the distribution shift impacts prior calibration" to connect the two subplots

---

## Verification from Code

From `plot_distribution_shift.py` lines 393-450:

```python
# === Plot 2: Source Data - Easy vs Hard Clustering (Explains Bimodal Structure) ===
ax2 = axes[1]

# Plot SOURCE difficulty-based densities
if len(pc1_source_easy) > 50 and len(pc1_source_hard) > 50:
    kde_source_easy = gaussian_kde(pc1_source_easy, bw_method=0.1)
    kde_source_hard = gaussian_kde(pc1_source_hard, bw_method=0.1)
    
    ax2.plot(x, density_source_easy, label=f'Easy (PC1 < 0.0): {source_easy_pct:.1f}%', ...)
    ax2.plot(x, density_source_hard, label=f'Hard (PC1 > 0.2): {source_hard_pct:.1f}%', ...)

ax2.set_title(
    'Source/Prior Data: Easy vs Hard Task Distribution\n'
    f'Bimodal Structure Explained by Two Distinct Task Clusters',
    ...
)
```

The code explicitly plots **SOURCE** data and titles it "Source/Prior Data".

---

## Data Verification

**From the actual figure:**
- Source mean: 0.060
- RouteLLM mean: -0.004
- Mean shift: -0.064 (negative)

**From code interpretation (lines 554-558):**
```python
shift_dir = "Easy" if mean_shift < 0 else "Hard"
if mean_shift < 0:
    print(f"      → More easy prompts in RouteLLM vs Source")
```

Since mean_shift = -0.064 < 0, this confirms: **RouteLLM has more easy prompts than Source**.

---

## Impact

✅ **Caption now accurately describes what each subplot shows**  
✅ **Top subplot interpretation clarified (left-shift = more easy tasks)**  
✅ **Bottom subplot correctly identified as showing training data structure**  
✅ **Connection between the two subplots explained (why shift matters)**  
✅ **Maintains the narrative: distribution shift causes prior miscalibration**

---

## Files Modified

- `paper/sections/empirical_motivation.tex` - Updated Figure 2 caption

## Files Referenced

- `experiments_v1/01.5_figure/plot_distribution_shift.py` - Figure generation code
- `paper/figures/figure2_distribution_shift.png` - Actual figure

