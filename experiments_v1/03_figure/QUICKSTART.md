# Figure 3: Quick Start Guide

## 30-Second Start

```bash
cd experiments_v1/03_figure
python find_optimal_gamma.py --output results/
```

**Done!** Check `results/optimal_gamma_analysis.pdf`

## What You Get

- ✅ 8-panel publication-quality figure (PDF + PNG + EPS)
- ✅ Recommended gamma value
- ✅ LaTeX caption and results section
- ✅ JSON with all numerical results

## Common Commands

### Basic Usage
```bash
# Use defaults (canonical dev set, 8 gamma values)
python find_optimal_gamma.py --output results/
```

### Custom Gamma Range
```bash
# Test specific values
python find_optimal_gamma.py \
  --gamma-values 1.0 0.05 0.02 0.01 0.005 \
  --output results/
```

### With Target Usage
```bash
# If you know oracle usage (e.g., 25%)
python find_optimal_gamma.py \
  --target-usage 25.0 \
  --output results/
```

### Custom Data
```bash
# Use your own calibration data
python find_optimal_gamma.py \
  --calibration-data /path/to/my_data.jsonl \
  --output results/
```

### Verbose Mode
```bash
# See detailed progress
python find_optimal_gamma.py --verbose --output results/
```

## Understanding Output

### Console Output
```
💡 RECOMMENDED: γ = 0.010
   Reason: provides balanced Calib/Prior influence
   Final strong usage: 22.2%
   Avg reward: 0.6717
   Calib/Prior ratio: 1.401
```

### Key Metrics

| Metric | Meaning | Good Range |
|--------|---------|------------|
| **Gamma** | Covariance inflation factor | 0.001 - 0.05 |
| **Calib/Prior** | Balance of influence | 0.5 - 2.0 |
| **Strong %** | Strong model usage | Depends on domain |
| **Reward** | Average quality | Higher is better |

### Files Generated

```
results/
├── optimal_gamma_analysis.pdf    # Main figure (use this in paper)
├── optimal_gamma_analysis.png    # High-res raster
├── optimal_gamma_analysis.eps    # Alternative vector format
├── gamma_results.json            # All numerical results
├── figure_caption.tex            # LaTeX caption
└── gamma_results.tex             # LaTeX results section
```

## Next Steps

### For Papers
```latex
% In your LaTeX document
\input{experiments_v1/03_figure/results/figure_caption.tex}
\input{experiments_v1/03_figure/results/gamma_results.tex}
```

### For Production
```bash
# Use recommended gamma to calibrate router
cd ../../scripts/calibration
python calibrate_router.py \
  --gamma 0.010 \
  --output my_router.joblib
```

## Troubleshooting

### "ImportError: cannot import..."
```bash
# Make sure you're in the right directory
cd /path/to/banditGPT/experiments_v1/03_figure
python find_optimal_gamma.py --output results/
```

### "FileNotFoundError: calibration data..."
```bash
# Provide explicit path
python find_optimal_gamma.py \
  --calibration-data ../../src/bandit_gpt/data/offline_dataset/dev_rewards_complete.jsonl.gz \
  --output results/
```

### "Invalid format..."
Your data should be one of:
```jsonl
{"prompt": "...", "rewards": {"model_a": 0.85, "model_b": 0.95}}
```
or
```jsonl
{"prompt": "...", "model_id": "model_a", "raw_score": 0.85, "ok": true}
```

## More Help

- **Detailed usage**: See `USAGE.md`
- **Integration guide**: See `INTEGRATION.md`
- **Overview**: See `SUMMARY.md`
- **Auto-generated results**: See `README.md`

## One-Liner Examples

```bash
# Quick test with 2 values
python find_optimal_gamma.py --gamma-values 1.0 0.01 --output test/

# Fine-grained search
python find_optimal_gamma.py --gamma-values 0.015 0.012 0.010 0.008 --output fine/

# With target
python find_optimal_gamma.py --target-usage 30.0 --output targeted/

# Custom everything
python find_optimal_gamma.py \
  --calibration-data my_data.jsonl \
  --warmup-priors my_priors.joblib \
  --pca my_pca.joblib \
  --gamma-values 1.0 0.02 0.01 0.005 \
  --output my_results/
```

---

**Need more details?** See `USAGE.md` for comprehensive documentation.

