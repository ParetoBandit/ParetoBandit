# Key Numbers for Paper: Distribution Shift Analysis

## Quick Reference Card

Copy-paste these exact numbers when writing about this experiment:

### Primary Metric
- **PSI = 0.275** (exceeds 0.25 threshold for "substantial shift")

### Distribution Statistics
- **Mean shift = -0.064** (deployment shifted toward easier tasks)
- **Source mean PC1 = 0.060** (training/warmup priors)
- **Target mean PC1 = -0.004** (deployment/RouteLLM)

### Task Clustering
- **Easy tasks**: 45.4% of source data, centered at PC1 = -0.105
- **Hard tasks**: 22.4% of source data, centered at PC1 = 0.365
- **Ambiguous tasks**: 32.2% (between thresholds)

### Model Performance Mismatch
| Model | Prior Expectation | Observed Reward | Delta |
|-------|-------------------|-----------------|-------|
| GPT-4-Turbo | 0.94 | 0.84 | **-10.6%** |
| Mixtral-8x7B | 0.45 | 0.81 | **+80.0%** |

### Hybrid Performance
- **1.26× near-optimal recovery** despite PSI = 0.275

### PSI Interpretation Thresholds
- PSI < 0.1: No significant shift
- 0.1 ≤ PSI < 0.2: Moderate shift
- 0.2 ≤ PSI < 0.25: Significant shift (monitoring required)
- **PSI ≥ 0.25: Substantial shift (adaptive correction required)** ← Our case

## How to Use These Numbers

### In Abstract
> "We identify substantial distribution shift (PSI = 0.275) between training and deployment, demonstrating that warmup priors trained on historical data may over-route to expensive models. Our hybrid approach achieves 1.26× near-optimal performance by automatically adapting to this mismatch."

### In Introduction
> "Real-world deployments exhibit significant covariate shift---we measure PSI = 0.275 in our setting---making fixed routing policies inherently brittle."

### In Results
> "Despite Mixtral showing 80% higher utility on deployment data compared to prior expectations (Table X), our hybrid system automatically corrects this miscalibration."

### In Discussion
> "The PSI = 0.275 we observe exceeds the 0.25 threshold for substantial shift, providing mathematical justification for adaptive routing mechanisms."

## Common Mistakes to Avoid

❌ Don't say: "PSI = 0.27" (too few decimals, looks imprecise)  
✅ Do say: "PSI = 0.275" (three decimals is standard)

❌ Don't say: "PSI is high" (vague, unscientific)  
✅ Do say: "PSI = 0.275 exceeds the 0.25 threshold" (specific, justified)

❌ Don't say: "Mixtral improves by 80%" (ambiguous)  
✅ Do say: "Mixtral's observed reward is 80% higher than prior expectations" (clear)

❌ Don't say: "Distribution shifts a lot" (informal)  
✅ Do say: "Substantial covariate shift (PSI = 0.275)" (formal, quantitative)

## Citation Support

When citing these numbers, reference:
1. **Figure 1.2**: The visualization showing bimodal structure and shift
2. **Table 1**: Domain mismatch showing reward deltas
3. **Equation 1**: PSI formula for reproducibility

## Connecting to Other Results

### Link to Corralling (Figure 5)
> "When PSI ≥ 0.25, the importance-weighted loss for the Warmup Expert increases sharply (Figure 5), triggering automatic down-weighting. This explains the meta-weight volatility and subsequent 1.26× recovery."

### Link to Cold-Start Analysis
> "Despite miscalibrated priors (80% error for Mixtral), warmup initialization provides T < 1000 advantage over pure bandit learning, demonstrating the value of hybrid approaches."

### Link to Ablation Studies
> "The 1.26× near-optimal recovery under PSI = 0.275 demonstrates robustness: hybrid outperforms both prior-only (fixed miscalibration) and bandit-only (slow cold-start) baselines."

## Narrative Arc

Use these numbers to tell this story:

1. **Setup**: "We analyze distribution shift between training and deployment"
2. **Discovery**: "We find PSI = 0.275, exceeding the 0.25 threshold for substantial shift"
3. **Evidence**: "This manifests as 80% reward discrepancy for Mixtral (Table 1)"
4. **Implication**: "Fixed priors would over-route to expensive GPT-4"
5. **Solution**: "Our hybrid approach automatically adapts, achieving 1.26× near-optimal"
6. **Impact**: "This robustness is critical for production where distributions evolve continuously"

## Data Provenance

All numbers come from:
- **Script**: `experiments_v1/01.5_figure/plot_distribution_shift.py`
- **Data sources**:
  - Source (P): `dev_rewards_2models.jsonl.gz` + `holdout_rewards_2models.jsonl.gz`
  - Target (Q): `routellm_battles_rewards.jsonl`
- **Embedding**: SentenceTransformer `all-MiniLM-L6-v2`
- **PCA model**: `src/artifacts/pca_model.joblib` (trained on RouteLLM data)
- **PSI bins**: B = 10 (standard)

## Reproducibility Statement

For paper appendix:

> "Distribution shift analysis (Section X) uses Population Stability Index with B = 10 bins on PC1-projected embeddings from SentenceTransformer (all-MiniLM-L6-v2). Source distribution: 80K prompts from dev/holdout datasets. Target distribution: 20K prompts from RouteLLM battles. Code and data available at [repository URL]."

