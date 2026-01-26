# Experiment Status: Figure 4 Pareto Frontier

## Current Run (Reproducible)
**Started**: 2026-01-25 13:01 PM
**PID**: 18176
**Log**: `pareto_final_run.log`

## Progress Tracking

### Phase 1: RouteLLM Augmentation ✅ IN PROGRESS
- **Strategy**: Smart augmentation - load existing 10 points, add 12 more
- **Target**: 22 total points (smooth curve)
- **Method**: Sequential processing with rate-limit handling
- **Status**: 2/12 thresholds complete (as of 13:07 PM)
- **Est. Time**: ~18 minutes

### Phase 2: banditGPT-Hybrid Sweep ⏳ PENDING
- **Sweep**: 10 λ values from 0.0 to 5.0
- **Trials**: 5 repetitions per λ (for statistical stability)
- **Total runs**: 50 bandit training sessions
- **Est. Time**: ~30-40 minutes

### Phase 3: Visualization ⏳ PENDING
- Convex hull filtering for both RouteLLM and banditGPT
- High-resolution PNG output (300 dpi + 600 dpi)
- KDD-compliant plot formatting

## Expected Output Files

### Data
- `results/pareto_results.json` - Final Pareto frontier data
- `results/intermediate_pareto_results.json` - Live progress snapshots

### Visualizations
- `results/figure4_pareto_frontier.png` (300 dpi)
- `results/figure4_pareto_frontier_hires.png` (600 dpi)

## Monitoring Commands

```bash
# Watch live progress
tail -f pareto_final_run.log

# Check process status
ps aux | grep generate_pareto_frontier

# View latest results
tail -50 pareto_final_run.log

# Check intermediate data
cat results/intermediate_pareto_results.json | jq '.strategies["RouteLLM-MF"] | length'
```

## Reproducibility Notes

### Random Seeds
- banditGPT trials use seeds: 42, 43, 44, 45, 46
- Ensures consistent results across runs

### Rate Limit Handling
- Sequential RouteLLM processing (no parallelism)
- Exponential backoff: 0.2s, 0.4s, 0.8s, 1.6s, 3.2s
- 0.3s pause between thresholds

### Data Integrity
- Zero leakage: Normalization uses train data only
- Sanitized priors: `priors_warmup_normalized.joblib`
- Prior strength: 10 effective samples (prevents "Arrogant Prior")

## Key Fixes Applied

### ✅ KDD Compliance
1. Convex hull filtering (monotonic frontiers)
2. Zero data leakage in normalization
3. Rate-limit safe API calls
4. Proper train/test split

### ✅ Prior Calibration
1. Auto-calibration in `CostAwareLinUCBRouter.__init__()`
2. Trace normalization to 10 effective samples
3. Pre-flight checks to verify predictions in [0, 1]

### ✅ Smart Augmentation
1. Loads existing RouteLLM results
2. Only runs missing thresholds
3. Merges old and new data
4. Saves intermediate progress

## Expected Results

### RouteLLM-MF
- **Points**: 22-25 (smooth curve from threshold sweep)
- **Cost range**: $0.000294 to $0.013000
- **Reward range**: ~0.81 to ~0.88

### banditGPT-Hybrid
- **Points**: 10 (one per λ value)
- **Cost range**: Similar to RouteLLM
- **Reward range**: Expected to dominate RouteLLM across all budgets

### Oracle
- **Single point**: Maximum quality achievable
- **Cost**: $0.001954
- **Reward**: 0.9533

## Next Steps

Once the run completes:
1. ✅ Verify convex hull filtering worked
2. ✅ Check that banditGPT dominates RouteLLM
3. ✅ Inspect high-res plot for KDD submission
4. 📝 Update KDD_COMPLIANCE_CHECKLIST.md
5. 📝 Document final results in SUCCESS_SUMMARY.md

