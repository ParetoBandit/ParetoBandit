# Progress Update: Figure 4 Pareto Frontier

## Current Status: ✅ Script Updated with Real RouteLLM

### What's Been Done

1. **✅ Removed all simulations**
   - Deleted fake threshold-based routing that was "cheating" by using actual rewards
   - Now only using REAL trained models

2. **✅ Integrated actual RouteLLM library**
   - Using `routellm.controller.Controller` with Matrix Factorization router ('mf')
   - Real routing decisions based on RouteLLM's trained model
   - No more simulations or fallbacks

3. **✅ Proper train/test split**
   - banditGPT trains on DEV set (1,121 prompts)
   - All methods evaluate on HOLDOUT set (750 prompts)
   - No train-on-test leakage

4. **✅ Optimizations**
   - Reuse RouteLLM controller across thresholds (faster)
   - Reduced thresholds from 20 to 10 (still good coverage)
   - Added progress logging

### Methods in the Script

1. **Oracle** - Upper bound (always picks best model)
2. **Static Baselines** - Always use Mixtral or GPT-4
3. **RouteLLM-MF** - Real RouteLLM Matrix Factorization router
4. **banditGPT Hybrid** - Our Corralling algorithm (η=1.0)

### Performance Note

**RouteLLM is slow** (~0.5s per prompt):
- 750 prompts × 10 thresholds = 7,500 routing calls
- Estimated time: **~60 minutes** for RouteLLM evaluation
- banditGPT is much faster (trains in ~2 minutes)

### Next Steps

1. **Run the full script** (will take ~60-70 minutes total)
   ```bash
   cd experiments_v1/04_figure
   python generate_pareto_frontier.py
   ```

2. **Results will show**:
   - How banditGPT compares to real RouteLLM (not simulated)
   - Pareto frontier on held-out test set
   - Fair comparison (all methods on same data)

### Files Updated

- `generate_pareto_frontier.py` - Main script with real RouteLLM
- `PROGRESS_UPDATE.md` - This file

### What Changed from Before

**Before**:
- ❌ Simulated RouteLLM by using actual rewards (cheating)
- ❌ Combined dev+holdout (train-on-test leakage for banditGPT)
- ❌ Warmup-Only was also simulated

**Now**:
- ✅ Real RouteLLM library (Matrix Factorization router)
- ✅ Proper train (dev) / test (holdout) split
- ✅ Only real trained models, no simulations

### Expected Output

```
experiments_v1/04_figure/results/
├── figure4_pareto_frontier.png       # Pareto plot
├── figure4_pareto_frontier_hires.png # High-res version
└── pareto_results.json               # Raw data
```

### Estimated Timeline

- Oracle + Static baselines: ~1 second
- RouteLLM (10 thresholds): ~60 minutes
- banditGPT (10 trials): ~20 minutes
- **Total: ~80 minutes**

### Why So Slow?

RouteLLM's Matrix Factorization router:
1. Embeds each prompt
2. Computes routing score
3. Makes decision

This takes ~0.5s per prompt, which adds up with 750 prompts × 10 thresholds.

### Alternative: Use Fewer Thresholds

If 80 minutes is too long, we can reduce to 5 thresholds:
```python
thresholds = np.linspace(0.0, 1.0, 5)  # ~30 minutes instead
```

This still gives a good Pareto curve.

