# Phased Shift Update: Distribution Shift Detection

## Major Improvement

Changed from "immediate divergence" to "phased response to distribution shift" methodology.

## The Problem with Previous Approach

**Before**: Both experts start at 50%, then immediately diverge from t=0
- No clear "before/after" story
- Looks like arbitrary decommissioning
- Doesn't show the system detecting a problem

## The New Approach: Phased Stress Test

**Phase 1 (t=0-50)**: Neutral zone
- Both models perform equally well (μ=0.85)
- System maintains roughly 50/50 weights (with exploration noise)
- No strong signal to prefer one expert over another

**Phase 2 (t=50+)**: Alignment tax emerges
- Distribution shift: GPT-4 collapses (μ=0.2), Mixtral succeeds (μ=0.9)
- System detects the mismatch
- Rapid decommissioning: Warmup drops from ~25% → 0% in 14 steps

## Visual Improvements

### Before (Immediate Divergence)
- Red line: 50% → 0% starting at t=0
- Green line: 50% → 100% starting at t=0
- No clear "reaction" point
- Looks like a decay curve, not a response to an event

### After (Phased Response)
- **t=0-50**: Both lines hover around 40-60% (exploration noise, but relatively stable)
- **t=50**: Clear "knee" where blue vertical line marks the shift
- **t=50-64**: Dramatic drop (red) and rise (green) - visible reaction
- **t=64+**: Stable convergence after decommissioning

## Key Metrics

- **Shift point**: t=50
- **Decommissioning**: t=64
- **Reaction time**: 14 steps (vs instant in old version)
- **Learning rate**: η=0.3 (balances Phase 1 stability with Phase 2 responsiveness)
- **Loss gap**: +109.0 (287% more loss for failed expert)

## Scientific Value

This demonstrates three key properties:

1. **Non-arbitrary**: System doesn't decommission when both experts are equally good
2. **Adaptive**: Detects distribution shifts and reacts appropriately
3. **Decisive**: Once evidence accumulates, decommissioning is rapid (14 steps)

## Configuration Choices

### Shift Point (t=50)
- Early enough to show clear "before" phase
- Late enough to accumulate some pre-shift data
- 50 steps = ~16% of experiment, good balance

### Learning Rate (η=0.3)
- Low enough: Minimizes pre-shift drift from sampling noise
- High enough: Clear post-shift reaction (14 steps to decommission)
- Sweet spot between stability and responsiveness

### Phase 1 Rewards (Both μ=0.85)
- Slightly lower than Phase 2 Mixtral (0.9) to show it's a different distribution
- High enough to be "good" (simulates easy/general tasks)
- Same for both models (no signal difference)

### Phase 2 Rewards
- Mixtral: μ=0.9 (high, consistent with "good for constrained tasks")
- GPT-4: μ=0.2 (dramatic failure, represents "alignment tax")
- Large gap (0.7) ensures clear signal for decommissioning

## Narrative Update

### Old Narrative
"The algorithm rapidly decommissions a failing expert"
- True but uninspiring
- Doesn't show *when* or *why* decommissioning happens

### New Narrative
"The system detects a distribution shift and reacts decisively"
- Shows adaptive behavior
- Clear cause (shift) and effect (decommissioning)
- Demonstrates robustness to changing environments

## Implementation Changes

### Environment Class
```python
class PhasedEnvironment:
    def get_reward(self, model: str) -> float:
        if self.t < self.shift_step:
            # Phase 1: Both models good
            return np.clip(self.rng.normal(0.85, 0.05), 0.0, 1.0)
        else:
            # Phase 2: Quality inversion
            if model == "mistralai/mixtral-8x7b-instruct":
                return np.clip(self.rng.normal(0.90, 0.05), 0.0, 1.0)
            else:
                return np.clip(self.rng.normal(0.20, 0.08), 0.0, 1.0)
```

### Visualization Updates
- Added blue vertical line at shift point
- Added annotations for "Distribution Shift" and "Decisive Decommissioning"
- Added "Reaction time (Δt)" to decommissioning annotation
- Shaded Phase 2 region (red tint)
- Updated loss plot to show "Losses Diverge (After Shift)"

## Comparison to Real-World

This phased scenario simulates:

**Real scenario**: 
- Production system trained on general conversation data
- Traffic shifts to include more "strict constraint" prompts (e.g., output formatting requirements)
- GPT-4 over-explains and violates constraints (alignment tax)
- Mixtral follows instructions more literally and succeeds

**Our simulation**:
- Phase 1: General tasks (both models good)
- Phase 2: Constraint-heavy tasks (cheap model wins)
- System adapts to the new distribution

## Future Extensions

1. **Multiple shifts**: Show recovery if distribution shifts back
2. **Gradual shift**: Make transition smooth (t=40-60) instead of instant (t=50)
3. **Seasonal patterns**: Periodic shifts (e.g., business hours vs night)
4. **Three+ models**: More complex expert portfolios

## Files Updated

1. `generate_figure5_synthetic.py`:
   - New `PhasedEnvironment` class
   - Updated visualization with shift annotations
   - Added reaction_time metric
   
2. `figure5_corralling_kdd.tex`:
   - Updated caption to mention distribution shift
   - New phase descriptions (Exploration → Shift → Decommissioning)
   - Updated metrics (shift point, reaction time)

3. `README.md`:
   - New "Phased Methodology" section
   - Comparison table (phased vs immediate divergence)
   - Design rationale for shift point and learning rate

## Status

✅ Phased environment implemented  
✅ Visualization updated with shift annotations  
✅ Clear "knee" at t=50 visible  
✅ Reaction time: 14 steps (optimal for clarity)  
✅ Ready for paper integration  

