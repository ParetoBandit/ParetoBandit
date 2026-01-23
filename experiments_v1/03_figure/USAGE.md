# Figure 3: Optimal Gamma Analysis - Usage Guide

## Quick Start

```bash
cd experiments_v1/03_figure

# Run with default settings (uses canonical dev set)
python find_optimal_gamma.py --output results/

# View the generated figure
open results/optimal_gamma_analysis.pdf
```

## What This Script Does

The `find_optimal_gamma.py` script systematically evaluates different gamma (covariance inflation) values to determine the optimal balance between:

1. **Warmup Priors**: Knowledge from 80,000 pre-training samples
2. **Calibration Data**: Domain-specific samples (typically 100-1,500)

### Why Gamma Matters

When adapting a pre-trained router to a new domain:

- **γ = 1.0** (no scaling): Prior is too strong, calibration data has minimal influence
- **γ too small** (< 0.001): Prior is too weak, discards valuable warmup knowledge
- **γ optimal** (0.002-0.02): Balanced influence, efficient adaptation

## Command-Line Options

### Basic Usage

```bash
python find_optimal_gamma.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration-data` | `dev_rewards_complete.jsonl.gz` | Path to calibration data |
| `--warmup-priors` | `artifacts/priors_warmup.joblib` | Path to warmup priors |
| `--pca` | `artifacts/pca_23.joblib` | Path to PCA model |
| `--output` | `results` | Output directory |
| `--gamma-values` | `[1.0, 0.1, 0.05, ...]` | Gamma values to test |
| `--target-usage` | `None` | Target strong model usage % |
| `--verbose` | `False` | Print detailed progress |

## Examples

### Example 1: Default Analysis

Uses the canonical dev set (1,121 samples) and tests 8 gamma values:

```bash
python find_optimal_gamma.py --output results/
```

**Output:**
- Comprehensive 8-panel figure
- Recommended gamma value
- LaTeX caption and results section
- JSON with numerical results

### Example 2: Custom Gamma Range

Test specific gamma values:

```bash
python find_optimal_gamma.py \
  --gamma-values 1.0 0.05 0.02 0.01 0.005 0.002 \
  --output results/custom_range/
```

### Example 3: With Target Usage

If you know the oracle strong model usage (e.g., from a held-out validation set):

```bash
python find_optimal_gamma.py \
  --target-usage 25.0 \
  --output results/targeted/
```

This will recommend the gamma that gets closest to 25% strong model usage.

### Example 4: Custom Calibration Data

Use your own calibration dataset:

```bash
python find_optimal_gamma.py \
  --calibration-data /path/to/my_calibration.jsonl \
  --output results/custom_data/
```

**Required format:**

Option 1 (Aggregated):
```jsonl
{"prompt": "...", "rewards": {"model_a": 0.85, "model_b": 0.95}}
```

Option 2 (Oracle format - auto-converted):
```jsonl
{"prompt": "...", "model_id": "model_a", "raw_score": 0.85, "ok": true}
{"prompt": "...", "model_id": "model_b", "raw_score": 0.95, "ok": true}
```

### Example 5: Verbose Mode

See detailed progress for each gamma value:

```bash
python find_optimal_gamma.py --verbose --output results/
```

## Understanding the Output

### Console Output

```
================================================================================
FIGURE 3: OPTIMAL GAMMA CALIBRATION ANALYSIS
================================================================================

📥 Loading resources...
   ✅ Warmup priors: 80,000 samples
   ✅ PCA: 23 components
   ✅ Models: mistralai/mixtral-8x7b-instruct, openai/gpt-4-turbo

📊 Loading calibration data...
   ✅ Loaded 1,121 calibration samples

🔬 Testing 8 gamma values...
[Progress bar]

================================================================================
RESULTS: Gamma Factor Comparison
================================================================================

   Gamma     Eff. N  Calib/Prior    Strong%     Reward    Conv.Rate
--------------------------------------------------------------------------------
   1.000     80,000        0.014      46.7%     0.8109     0.002976
   0.100      8,000        0.140      44.2%     0.7856     0.003421
   0.050      4,000        0.280      41.8%     0.7623     0.004156
   0.020      1,600        0.701      35.3%     0.7124     0.007892
   0.010        800        1.401      22.2%     0.6717     0.012653
   0.005        400        2.803      15.8%     0.6234     0.018945
   0.002        160        7.006       8.3%     0.5891     0.025678
   0.001         80       14.013       5.1%     0.5723     0.029234
================================================================================

💡 RECOMMENDED: γ = 0.010
   Reason: provides balanced Calib/Prior influence
   Final strong usage: 22.2%
   Avg reward: 0.6717
   Calib/Prior ratio: 1.401
```

### Key Metrics Explained

1. **Gamma**: The covariance inflation factor
2. **Eff. N**: Effective sample size of the prior (= 80,000 × γ)
3. **Calib/Prior**: Ratio of calibration samples to effective prior samples
   - < 1.0: Prior dominates
   - ≈ 1.0: Balanced influence
   - > 1.0: Calibration dominates
4. **Strong%**: Final strong model usage percentage
5. **Reward**: Average reward achieved during calibration
6. **Conv.Rate**: Convergence rate (higher = faster adaptation)

### Generated Files

```
results/
├── optimal_gamma_analysis.png     # High-res figure (300 DPI)
├── optimal_gamma_analysis.pdf     # Vector format for papers
├── optimal_gamma_analysis.eps     # Alternative vector format
├── gamma_results.json             # Numerical results
├── figure_caption.tex             # LaTeX caption
└── gamma_results.tex              # LaTeX results section
```

### The Figure (8 Panels)

1. **Top-left (large)**: Policy adaptation curves for all gamma values
2. **Top-center**: Final usage vs gamma (shows prior weakening effect)
3. **Top-right**: Influence balance (Calib/Prior ratio)
4. **Middle-left**: Adaptation magnitude (change from baseline)
5. **Middle-center**: Prior strength (effective N)
6. **Middle-right**: Quality performance (average reward)
7. **Bottom-left**: Convergence rate (adaptation speed)
8. **Bottom-right**: Summary table with key metrics

## Interpreting Results

### Recommendation Criteria

The script recommends gamma based on multiple criteria:

1. **Target Matching** (if `--target-usage` provided): Closest to target
2. **Maximum Adaptation**: Largest change from baseline
3. **Balanced Influence**: Calib/Prior ratio closest to 1.0 ✓ (default)
4. **Fastest Convergence**: Highest convergence rate

### When to Use Different Gamma Values

| Your Situation | Recommended γ | Rationale |
|----------------|---------------|-----------|
| Small calibration set (50-150) | 0.001 - 0.002 | High influence per sample needed |
| Medium set (150-500) | 0.002 - 0.010 | Balanced approach |
| Large set (500-1500) | 0.005 - 0.020 | Lower influence per sample OK |
| Very large set (1500+) | 0.010 - 0.050 | Prior still valuable |
| Known oracle policy | Run with `--target-usage` | Match target directly |
| Unknown oracle | 0.010 (default) | Works for most domains |

### Red Flags

⚠️ **Warning signs that gamma might be wrong:**

1. **Calib/Prior < 0.1**: Prior too strong, minimal adaptation
2. **Calib/Prior > 10**: Prior too weak, might discard valuable knowledge
3. **Reward drops significantly**: Over-adaptation, losing quality
4. **Strong usage doesn't change**: Insufficient prior weakening

## Integration with Paper

### Including the Figure

In your LaTeX document:

```latex
\input{experiments_v1/03_figure/results/figure_caption.tex}
```

### Including Results Section

```latex
\input{experiments_v1/03_figure/results/gamma_results.tex}
```

### Citation Example

> We systematically evaluated gamma values from 0.001 to 1.0 to determine the optimal covariance inflation factor (Figure~\ref{fig:optimal_gamma}). Our analysis reveals that γ=0.010 provides the optimal balance, achieving a Calibration/Prior ratio of 1.401. This enables 1,121 calibration samples to effectively adapt 80,000 warmup priors, resulting in a 24.5 percentage point shift in routing strategy while maintaining 0.67 average reward.

## Next Steps After Finding Optimal Gamma

### 1. Calibrate Your Router

Use the recommended gamma with the calibration script:

```bash
cd ../../scripts/calibration/

python calibrate_router.py \
  --calibration-data ../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \
  --gamma 0.010 \
  --output my_calibrated_router.joblib
```

### 2. Evaluate on Holdout Set

```bash
cd ../../experiments_v1/calibration/

python evaluate_calibrated_router.py \
  --router-path ../../scripts/calibration/my_calibrated_router.joblib \
  --holdout-data ../../src/bandit_gpt/data/offline_dataset/holdout_rewards_complete.jsonl.gz
```

### 3. Deploy to Production

```python
from bandit_gpt.calibration import CalibratedRouter
from sentence_transformers import SentenceTransformer
import joblib

# Load resources
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pca_model = joblib.load("artifacts/pca_23.joblib")
router = CalibratedRouter.load("my_calibrated_router.joblib", encoder, pca_model)

# Use in production
selected_model = router.select_model(user_prompt)
response = call_llm(selected_model, user_prompt)

# Optional: Continue learning online
router.update(user_prompt, observed_reward)
```

## Troubleshooting

### Issue: "ImportError: cannot import name..."

**Solution:** Make sure you're running from the project root or the script directory:

```bash
cd /path/to/banditGPT/experiments_v1/03_figure
python find_optimal_gamma.py --output results/
```

### Issue: "FileNotFoundError: calibration data not found"

**Solution:** Check the data path or provide explicit path:

```bash
python find_optimal_gamma.py \
  --calibration-data ../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \
  --output results/
```

### Issue: "Invalid format! Expected..."

**Solution:** The script auto-detects two formats:

1. Aggregated: `{"prompt": "...", "rewards": {"model": 0.0}}`
2. Oracle: `{"prompt": "...", "model_id": "...", "raw_score": 0.0}`

Make sure your data matches one of these formats.

### Issue: Script runs but recommendation seems wrong

**Solution:** Try these diagnostics:

1. **Check your data size**: Very small (<50) or very large (>5000) calibration sets may need custom gamma ranges
2. **Verify data quality**: Ensure rewards are meaningful (not all 0s or 1s)
3. **Test with target usage**: If you know the oracle policy, use `--target-usage`
4. **Review the figure**: Look at all 8 panels to understand the tradeoffs

## Advanced Usage

### Custom Gamma Search

For fine-grained search around a specific value:

```bash
python find_optimal_gamma.py \
  --gamma-values 0.015 0.012 0.010 0.008 0.006 \
  --output results/fine_search/
```

### Comparing Multiple Datasets

```bash
# Dataset 1
python find_optimal_gamma.py \
  --calibration-data dataset1.jsonl \
  --output results/dataset1/

# Dataset 2
python find_optimal_gamma.py \
  --calibration-data dataset2.jsonl \
  --output results/dataset2/

# Compare the recommended gammas and Calib/Prior ratios
```

### Sensitivity Analysis

Test how sensitive the recommendation is to data size:

```bash
# Use first 200 samples
head -200 calibration.jsonl > calibration_200.jsonl
python find_optimal_gamma.py --calibration-data calibration_200.jsonl --output results/n200/

# Use first 500 samples
head -500 calibration.jsonl > calibration_500.jsonl
python find_optimal_gamma.py --calibration-data calibration_500.jsonl --output results/n500/

# Use all samples
python find_optimal_gamma.py --calibration-data calibration.jsonl --output results/full/
```

## Related Documentation

- **Figure 1**: Semantic task specialization (`experiments_v1/01_figure/`)
- **Figure 2**: Calibration convergence analysis (`experiments_v1/02_figure/`)
- **Calibration Guide**: Complete workflow (`experiments_v1/calibration/README.md`)
- **CLI Tools**: Command-line calibration (`scripts/calibration/README.md`)
- **Library API**: Python API (`src/bandit_gpt/calibration.py`)

## Support

For questions or issues:

1. Check this guide and the main README
2. Review the calibration documentation: `experiments_v1/calibration/README.md`
3. Check existing scripts: `scripts/calibration/find_gamma.py`
4. Open an issue on GitHub with:
   - Command you ran
   - Error message or unexpected output
   - Data format and size
   - Python version and installed packages

---

**Last Updated**: January 23, 2026  
**Script Version**: 1.0  
**Compatible with**: BanditGPT v1.0+

