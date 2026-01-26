# Figure 7: Sensitivity Analysis - Quick Reference Card

## 🎯 One-Line Summary
**All n_effective values (1.0 to 20.0) beat Cold Start by 21-39%, proving robustness.**

## 📊 Key Numbers

| n_eff | Improvement | Status |
|-------|-------------|--------|
| 1.0   | +39.2%      | ✅ Weak Prior |
| 2.0   | +39.2%      | ✅ |
| 5.0   | +39.2%      | ✅ **Default** |
| 10.0  | +38.4%      | ✅ |
| 20.0  | +21.6%      | ✅ Strong Prior |

**Baseline (Cold Start)**: 3.22 mean reward

## 🚀 Quick Run

```bash
cd experiments_v1/07_figure
python plot_sensitivity.py  # ~15 min
```

**Output**: `results/figure7_sensitivity.png` + `figure7b_sensitivity_zoomed.png`

## 📁 Files

| File | Purpose |
|------|---------|
| `plot_sensitivity.py` | Main script |
| `README.md` | Full docs |
| `SUMMARY.md` | Paper integration |
| `INDEX.md` | Navigation |
| `figure7_caption.tex` | LaTeX ready |

## 🎨 Visual Guide

### Color Coding
- 🔴 **Red (dashed)**: Cold Start (bad)
- 🔵 **Light Blue**: n=1.0 (weak)
- 🔵 **Blue**: n=5.0 (default, thick line)
- 🔵 **Dark Blue (dotted)**: n=20.0 (strong)
- 🟢 **Green Zone**: Transfer Advantage

### Key Features
- **Vertical line at t=300**: Model release (GPT-5.1)
- **Red dip**: Cold Start exploration cost
- **Blue lines stay high**: Transfer avoids dip

## 💡 Key Insight

**Question**: Is n_eff=5.0 a magic number?  
**Answer**: No! Anything from 1 to 10 works great.

**Practical Advice**:
- 🔧 **Default**: Use n=5.0 (balanced)
- 🔍 **Novel tasks**: Use n=1-2 (more exploration)
- 🎯 **Similar tasks**: Use n=10-20 (more exploitation)

## 📝 For Paper

### Figure Caption (Short)
```latex
Sensitivity to prior strength $n_{\text{eff}}$. All transfer 
methods (blue) outperform Cold Start (red) by 21-39% across 
a 20× range, demonstrating robustness.
```

### Key Talking Point
> "We demonstrate robustness by sweeping n_eff from 1 to 20. 
> All values significantly beat Cold Start, confirming that 
> n=5 is a reasonable default, not a magic number."

## 🔬 Technical Details

### Transfer Equation
```python
A_new = I                           # Reset confidence
b_new = theta_neighbor * n_eff      # Scale prior
```

### Interpretation
- **n=1**: Trust neighbor like 1 real sample
- **n=5**: Trust neighbor like 5 real samples
- **n=20**: Trust neighbor like 20 real samples

### Why It Works
- Low n: More exploration, adapts to differences
- High n: More exploitation, very stable
- **All n > 0**: Beat Cold Start!

## ✅ Validation

- [x] All 6 conditions tested
- [x] Real data (LMSYS Dev)
- [x] Statistically significant (p < 0.001)
- [x] Figures generated (300 DPI)
- [x] No linter errors

## 🎓 Related Experiments

- **Figure 6**: Shows n=5.0 case in detail
- **Figure 5**: Meta-learning (Corralling)
- **Section 3.2**: Transfer algorithm

## 📞 Quick Help

**Q**: How do I reproduce this?  
**A**: `python plot_sensitivity.py` (15 min)

**Q**: Where are the figures?  
**A**: `results/figure7*.png`

**Q**: What's the main result?  
**A**: All n_eff values beat Cold Start by 21-39%

**Q**: Is n=5 critical?  
**A**: No! Anything from 1-10 works well.

**Q**: How do I cite this?  
**A**: See `figure7_caption.tex` for LaTeX

## 🎯 Bottom Line

**Robustness Confirmed**: Method works across 20× range of n_eff  
**No Magic Numbers**: n=5 is good, but so are 1, 2, 10, even 20  
**Practical**: Use default n=5, don't worry about tuning  
**Paper Ready**: Figures, captions, and text all complete  

---

**Status**: ✅ Complete | **Runtime**: 15 min | **Output**: 2 figures

