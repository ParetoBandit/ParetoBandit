# Experiment 01.5 Figure: Feature Distribution Shift Analysis

**Created**: 2026-01-24  
**Purpose**: Visualize covariate shift between Source and RouteLLM data using PC1 density plots

## What This Experiment Does

This experiment provides **visual proof of covariate shift** by:

1. **Loading two distributions**:
   - Source/Prior data (dev + holdout datasets)
   - RouteLLM deployment data (battle dataset)

2. **Projecting to 1D**:
   - Uses pre-trained PCA to project both distributions onto PC1
   - PC1 captures the most semantic variance (3-5%)

3. **Computing shift metrics**:
   - Population Stability Index (PSI): Industry standard for distribution monitoring
   - Mean shift: Direction and magnitude of semantic drift

4. **Creating visualizations**:
   - **Plot 1**: Source vs RouteLLM density overlay with PSI
   - **Plot 2**: Easy vs Hard task clustering in RouteLLM data

## Key Output

### Distribution Shift Plot

The main output shows:
- Blue curve = Source distribution (what we trained on)
- Red curve = RouteLLM distribution (what we deployed on)
- PSI value = Quantifies the shift magnitude

### Interpretation

- **PSI < 0.1**: No significant shift → Model is stable
- **0.1 ≤ PSI < 0.2**: Moderate shift → Monitor performance
- **PSI ≥ 0.2**: Significant shift → Consider retraining

## Quick Start

```bash
# Run the analysis:
python3 experiments_v1/01.5_figure/plot_distribution_shift.py

# Results saved to:
# experiments_v1/01.5_figure/results/distribution_shift_pc1.png
# experiments_v1/01.5_figure/results/distribution_shift_pc1_hires.png
```

## Prerequisites

1. **PCA model must exist**:
   ```bash
   # If not present, train it:
   python3 scripts/train_pca_from_routellm.py
   ```

2. **Data files must be accessible**:
   - `src/bandit_gpt/data/offline_dataset/dev_rewards_2models.jsonl.gz`
   - `src/bandit_gpt/data/offline_dataset/holdout_rewards_2models.jsonl.gz`
   - `src/bandit_gpt/data/offline_dataset/routellm_battles_rewards.jsonl`

## File Structure

```
experiments_v1/01.5_figure/
├── plot_distribution_shift.py    # Main analysis script
├── README.md                      # Detailed documentation
├── EXPERIMENT_SUMMARY.md          # This file
└── results/                       # Output directory
    ├── distribution_shift_pc1.png       # Main figure (300 DPI)
    └── distribution_shift_pc1_hires.png # High-res version (600 DPI)
```

## Scientific Value

This experiment provides evidence for:

1. **Covariate Shift Detection**: Proves deployment ≠ training distribution
2. **Domain Adaptation Justification**: Shows why warmup priors are needed
3. **Deployment Validation**: Quantifies how different the distributions are

## Expected Results

Based on the bimodal structure observed in other experiments:

- **Moderate shift** (PSI ≈ 0.12-0.18)
- **Direction**: RouteLLM slightly easier than Source (mean shift ≈ -0.05)
- **Implication**: Warmup priors are valuable but model is reasonably robust

## Paper Integration

This provides **Figure 1.2** for the paper:

**Caption suggestion**:
> "Feature distribution shift between training and deployment data. We project prompt embeddings onto the first principal component (PC1) and compare density distributions. The Population Stability Index (PSI = X.XX) indicates [no/moderate/significant] covariate shift, justifying our domain adaptation approach via warmup priors."

## Next Steps

After running this experiment:

1. **Include in paper**: Use figure in Section 3 (Problem Setup)
2. **Update calibration**: If PSI > 0.2, consider retraining
3. **Monitor over time**: Re-run periodically to detect concept drift

## Technical Details

- **Sample size**: 10K prompts per distribution (configurable)
- **PCA components**: 32 (retains ~35% variance)
- **Bandwidth**: 0.1 (Gaussian KDE)
- **Bins**: 20 (for PSI calculation)
- **Runtime**: ~5-10 minutes (depends on GPU availability)

## Troubleshooting

### Common Issues

1. **"PCA file not found"**
   → Run `python3 scripts/train_pca_from_routellm.py` first

2. **"No prompts loaded"**
   → Check data file paths in config_legacy.py

3. **Memory error**
   → Reduce `max_samples` or `batch_size` in script

4. **KDE fails**
   → Need at least 100 samples per distribution

## References

- **PSI**: Yurdakul (2018) - Statistical properties of population stability index
- **Covariate Shift**: Shimodaira (2000) - Improving predictive inference
- **PCA**: Johnson & Wichern (2007) - Applied multivariate statistical analysis

