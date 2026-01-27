# Experiment 02: Feature Distribution Shift Analysis (Statistical Analysis Only)

**Purpose**: Analyze and visualize covariate shift between Source/Prior data and RouteLLM deployment data using Population Stability Index (PSI) and 1D density plots of the first principal component.

**Important**: This is a **pure statistical analysis experiment** - it does NOT use BanditRouter or corralling. It provides motivational evidence for why adaptive routing is needed. For actual routing experiments that use corralling, see experiments like `07_figure/plot_ablation.py`.

## Overview

This experiment provides **visual proof of covariate shift** by comparing the semantic distribution of prompts between:
- **Source/Prior Data**: The dev/holdout datasets used for training warmup priors
- **RouteLLM Data**: The actual deployment data from user battles

## Key Question

**Has the "semantic center of gravity" shifted between training and deployment?**

If the RouteLLM data is shifted toward the "Easy" cluster compared to the Prior data, it proves that the deployment distribution differs from the training distribution—a classic case of **covariate shift**.

## The Analysis

### Population Stability Index (PSI)

PSI is a standard metric for detecting distribution shift in production ML systems:

```
PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
```

**Interpretation**:
- **PSI < 0.1**: No significant shift → Model is stable
- **0.1 ≤ PSI < 0.2**: Moderate shift → Monitor closely
- **PSI ≥ 0.2**: Significant shift → Consider retraining or domain adaptation

### 1D Density Plot on PC1

The first principal component (PC1) captures the most variance in the semantic embedding space. By projecting both distributions onto PC1 and plotting their densities:

1. **Overlap**: High overlap = similar distributions
2. **Separation**: Clear separation = covariate shift
3. **Direction**: Shift toward Easy or Hard cluster

## Files

- `plot_distribution_shift.py`: Main analysis script
- `results/distribution_shift_pc1.png`: Main visualization (300 DPI)
- `results/distribution_shift_pc1_hires.png`: High-resolution version (600 DPI)

## Usage

### Basic Usage

```bash
python3 experiments_v1/01.5_figure/plot_distribution_shift.py
```

### Prerequisites

1. **PCA Model**: Pre-trained PCA model must exist
   ```bash
   # If not available, train it:
   python3 scripts/train_pca_from_routellm.py
   ```

2. **Data Files**: 
   - Source data: `src/bandit_gpt/data/offline_dataset/dev_rewards_2models.jsonl.gz`
   - Source data: `src/bandit_gpt/data/offline_dataset/holdout_rewards_2models.jsonl.gz`
   - RouteLLM data: `src/bandit_gpt/data/offline_dataset/routellm_battles_rewards.jsonl`

### Configuration

By default, the script:
- Loads 10,000 prompts from Source (5K dev + 5K holdout)
- Loads 10,000 prompts from RouteLLM
- Uses 20 bins for PSI calculation
- Projects to PC1 using pre-trained PCA (32 components)

## Output

### Plot 1: Distribution Comparison

Shows overlaid density curves for:
- **Blue curve**: Source/Prior data distribution
- **Red curve**: RouteLLM data distribution
- **Dashed lines**: Mean values for each distribution
- **PSI value**: Quantifies the shift magnitude

### Plot 2: Difficulty Clustering

Shows RouteLLM data segmented by task difficulty:
- **Blue curve**: Easy tasks (Gap ≤ 0.3, Mixtral sufficient)
- **Red curve**: Hard tasks (Gap > 0.6, GPT-4 required)
- **Annotation**: Shows which direction the shift occurred

## Interpretation Guide

### Scenario 1: No Shift (PSI < 0.1)

```
Source Mean: 0.125
RouteLLM Mean: 0.118
Mean Shift: -0.007
```

**Conclusion**: Distributions are similar. No covariate shift. Model should perform as expected.

### Scenario 2: Shift Toward Easy (PSI > 0.1, Negative Mean Shift)

```
Source Mean: 0.150
RouteLLM Mean: 0.080
Mean Shift: -0.070
```

**Conclusion**: RouteLLM data has more easy prompts. Implications:
- Mixtral usage will be higher than training predicted
- Cost savings will exceed expectations
- GPT-4 usage will be lower

### Scenario 3: Shift Toward Hard (PSI > 0.1, Positive Mean Shift)

```
Source Mean: 0.100
RouteLLM Mean: 0.180
Mean Shift: +0.080
```

**Conclusion**: RouteLLM data has more hard prompts. Implications:
- GPT-4 usage will be higher than training predicted
- Cost will exceed expectations
- May need more aggressive exploration or calibration

## Statistical Foundation

### Why PC1?

The first principal component:
1. **Captures most variance**: Typically 3-5% of total variance (384D → 1D)
2. **Preserves semantic structure**: Tasks cluster by difficulty along PC1
3. **Robust to noise**: PCA filters out high-frequency noise
4. **Interpretable**: Easy tasks → negative values, Hard tasks → positive values

### Why PSI?

PSI is industry-standard for production ML:
1. **Model-agnostic**: Works with any distribution
2. **Interpretable**: Clear thresholds (0.1, 0.2)
3. **Actionable**: Directly informs retraining decisions
4. **Efficient**: Fast to compute, suitable for monitoring

### Statistical Significance

For N=10,000 samples per distribution:
- PSI > 0.02 is statistically significant (p < 0.05)
- Our threshold of 0.1 is conservative (high confidence)

## Connection to Paper

This experiment provides **Figure 1.2** evidence for:

1. **Section 3.1 (Covariate Shift)**:
   - "The deployment distribution differs from the training distribution"
   - Visual proof via density plot shift

2. **Section 3.2 (Domain Adaptation)**:
   - "We observe PSI = X.XXX, indicating [moderate/significant] shift"
   - Justifies the need for transfer learning or warmup priors

3. **Section 4.1 (Experimental Setup)**:
   - "We quantify distribution shift using PSI on PC1"
   - Shows due diligence in validating deployment assumptions

## Expected Results

Based on preliminary analysis, we expect:

1. **Moderate shift** (PSI ≈ 0.12-0.18)
   - RouteLLM data is slightly easier than Source data
   - Mean shift ≈ -0.05 to -0.10

2. **Bimodal structure preserved**
   - Both distributions show Easy/Hard clustering
   - Relative proportions may differ

3. **Actionable insight**
   - Warmup priors are valuable (distribution differs)
   - But not drastically (PSI < 0.3)
   - Current approach is appropriate

## Extensions

### 1. Multi-Dimensional PSI

Currently we compute PSI on PC1 only. Could extend to:
- PSI on PC1-5 jointly
- Weighted PSI by explained variance
- 2D density comparison (PC1 × PC2)

### 2. Temporal Shift

Track PSI over time:
- Early RouteLLM battles vs. recent battles
- Detect concept drift during deployment

### 3. Cluster-Level PSI

Compute PSI separately for:
- Easy prompts only
- Hard prompts only
- Show if shift affects one cluster more

### 4. Model-Specific Shift

Compare distributions for:
- GPT-4-turbo battles only
- Mixtral battles only
- Show if user selection bias exists

## References

1. **Population Stability Index**:
   - Yurdakul, B. (2018). "Statistical Properties of Population Stability Index"
   - Industry standard: PSI thresholds (0.1, 0.2)

2. **Covariate Shift**:
   - Shimodaira, H. (2000). "Improving predictive inference under covariate shift"
   - Foundation for domain adaptation

3. **PCA for Distribution Comparison**:
   - Hotelling's T² test (generalization to multivariate)
   - Johnson & Wichern (2007). "Applied Multivariate Statistical Analysis"

## Troubleshooting

### Issue: PCA file not found

```bash
# Train PCA model first:
python3 scripts/train_pca_from_routellm.py
```

### Issue: No prompts loaded

Check that data files exist:
```bash
ls -lh src/bandit_gpt/data/offline_dataset/
```

### Issue: KDE fails (too few samples)

Increase `max_samples` in the script:
```python
source_prompts = load_source_prompts(dev_file, holdout_file, max_samples=20000)
```

### Issue: Memory error

Reduce batch size:
```python
pc1_values = project_to_pc1(prompts, pca_file, batch_size=32)
```

## Contact

For questions about this experiment:
- See main project README
- Check `experiments_v1/README.md` for experiment guidelines

