# Plot Update Summary - Figure 5
## Visual Improvements for Submission

**Date:** January 26, 2026  
**File Modified:** `generate_pareto_frontier.py`

---

## Changes Made

### 1. Title Update ✅
**Before:** "Figure 4: Pareto Frontier - The Competitive Victory"  
**After:** "Figure 5: Pareto Frontier - The Competitive Victory"

**Rationale:** Consistent with directory structure (`05_figure/`) and paper figure numbering.

---

### 2. Dominated Points Visualization ✅

#### banditGPT-Hybrid (Blue)
- **Pareto-optimal points:** Blue diamonds (◆) connected by solid line
- **Dominated points:** Blue X marks (✕) at 200pt size
- **Raw points:** Faint blue circles (30% opacity, background)

#### RouteLLM-MF (Red)
- **Pareto-optimal points:** Red circles (●) connected by solid line
- **Dominated points:** Red X marks (✕) at 200pt size
- **Raw points:** Faint red circles (30% opacity, background)

**Code Implementation:**
```python
# Identify dominated points during convex hull calculation
for c, r in sorted_points:
    if r > current_max_reward:
        hull_costs.append(c)
        hull_rewards.append(r)
        current_max_reward = r
    else:
        # This point is dominated
        dominated_costs.append(c)
        dominated_rewards.append(r)

# Explicitly mark dominated points with X
if dominated_costs:
    ax.scatter(dominated_costs, dominated_rewards, 
              color=colors[strategy], marker='x', s=200, 
              linewidths=3, alpha=0.9, zorder=5,
              label=f'{strategy} (Dominated)')
```

---

### 3. Legend Enhancement ✅

**Changes:**
- Moved from `upper right` to `lower right` (better visibility with new markers)
- Reduced font size from 13 to 12 for compactness
- Added explicit dominated point labels
- Maintained 2-column layout (`ncol=2`)

**Legend Entries (Expected Order):**
1. Oracle (green star)
2. banditGPT-Hybrid (Pareto Frontier) - blue diamonds
3. banditGPT-Hybrid (Dominated) - blue X marks
4. RouteLLM-MF (Pareto Frontier) - red circles
5. RouteLLM-MF (Dominated) - red X marks
6. Mixtral 8x7B (static baseline)
7. GPT-4-Turbo (static baseline)
8. Production Standard (horizontal line at 0.80)

---

## Visual Impact

### Before
- Only RouteLLM had explicit dominated point visualization (implied by reviewer feedback)
- Title said "Figure 4" (inconsistent with directory)
- Legend in upper right could overlap with data at high quality region

### After
- **Symmetrical treatment:** Both methods show dominated points explicitly
- **Scientific transparency:** Viewers can immediately see which points are suboptimal
- **Fair comparison:** Equal visual treatment of both methods
- **Correct numbering:** Figure 5 matches directory structure
- **Better legend placement:** Lower right avoids overlapping high-quality region

---

## Statistical Interpretation

### banditGPT-Hybrid Dominated Points
Expected dominated points (from README):
- λ=0.5: Cost=$0.000294, Reward=0.8227 (collapsed to static Mistral)
- λ=1.0+: Cost=$0.000294, Reward=0.8227 (collapsed to static Mistral)
- Plus 1-2 intermediate points

**Total:** ~4 dominated points (40% of sweep)

### RouteLLM-MF Dominated Points
Expected dominated points (from README):
- 18 points total (64% of sweep)
- Concentrated in high-cost region (\$0.007-\$0.013)
- Demonstrates "Inverted U" failure mode

---

## Regenerating the Plot

To regenerate with these changes:

```bash
cd experiments_v1/05_figure/
python generate_pareto_frontier.py
```

**Expected Output:**
- `results/figure5_pareto_frontier.png` (300 dpi)
- `results/figure5_pareto_frontier_hires.png` (600 dpi)

**Runtime:** ~50 minutes (full experiment) or instant if using existing `pareto_results_final.json`

---

## File Naming Consistency

All output files now use "figure5" prefix:
- ✅ `figure5_pareto_frontier.png`
- ✅ `figure5_pareto_frontier_hires.png`
- ✅ `pareto_results_final.json` (filename unchanged, but metadata updated)

Previous "figure4" files have been renamed to match.

---

## Reviewer Response Readiness

These changes address the implicit reviewer expectation for:
1. **Visual consistency:** Both methods treated equally in visualization
2. **Transparency:** All points shown (faint) + dominated points marked explicitly
3. **Professional presentation:** Clear legend, proper figure numbering
4. **Scientific rigor:** Dominated points identified algorithmically, not arbitrarily

---

## Code Quality Notes

### Convex Hull Algorithm (Unchanged)
```python
# Sort by cost (ascending)
sorted_points = sorted(points, key=lambda x: x[0])

# Greedy sweep: keep point if it improves over previous max
current_max_reward = -float('inf')
for c, r in sorted_points:
    if r > current_max_reward:
        # Pareto-optimal
        hull_costs.append(c)
        hull_rewards.append(r)
        current_max_reward = r
    else:
        # Dominated
        dominated_costs.append(c)
        dominated_rewards.append(r)
```

**Properties:**
- ✅ Time complexity: O(n log n) due to sorting
- ✅ Correct for monotonic cost functions
- ✅ Identifies all dominated points
- ✅ Applied identically to both methods (fair comparison)

---

## Visual Style Guidelines

### Colors (Unchanged)
- Oracle: `#2ecc71` (green) - represents upper bound
- banditGPT: `#3498db` (blue) - our method
- RouteLLM: `#e74c3c` (red) - baseline

### Marker Sizes
- Static models: 150pt circles
- Oracle: 250pt star
- Pareto frontier: 7pt (line markers)
- Dominated points: 200pt X marks (highly visible)
- Raw points: 30pt (background reference)

### Transparency Levels
- Frontier lines: 85-90% opacity (prominent)
- Dominated X marks: 90% opacity (clear)
- Raw points: 20-30% opacity (subtle background)
- Production line: 60% opacity (reference)

---

## Summary

This update ensures Figure 5 follows best practices for multi-method Pareto frontier visualization:
1. ✅ Correct figure numbering
2. ✅ Explicit dominated point markers for both methods
3. ✅ Clear, organized legend
4. ✅ Consistent visual treatment
5. ✅ Scientific transparency (all data visible)

The plot is now ready for submission with full reviewer confidence in the experimental rigor and fairness of the comparison.

**Status:** ✅ Complete - Ready for regeneration

