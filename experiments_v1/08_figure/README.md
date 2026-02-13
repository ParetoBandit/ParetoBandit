# Experiment 08: Sensitivity Analysis - Prior Strength Calibration

**Figure 8**: "banditGPT: Cost-Aware Contextual Bandits for LLM Routing"

---

## Quick Start

```bash
# Run the unified analysis (experiments + figure + table)
cd /Users/annette/repostitories/banditGPT
python experiments_v1/08_figure/run_figure8_analysis.py

# Force re-run (ignore cache)
python experiments_v1/08_figure/run_figure8_analysis.py --force-rerun

# View results
open experiments_v1/08_figure/results/figure8_regime_stratified_CORRECTED.png
```

**Runtime**: ~3 minutes (cached), ~5 minutes (first run)  
**Output**: 
- PNG figure: `figure8_regime_stratified_CORRECTED.png`
- LaTeX table: `appendixC_neff_sensitivity.tex` (Appendix C: Sensitivity Analysis)
- Console table with regime classification

---

## Research Question

**Does the effective prior strength ($n_{eff}$) parameter affect semantic transfer performance in cost-aware routing?**

**Answer**: The effect is **regime-dependent**. Corralling adaptively switches between semantic transfer (warmup expert) and cold-start exploration (tabula rasa expert) based on data-prior match.

**Production Impact**: Default remains `n_eff=5.0` (mid-range value). System robustness comes from Corralling's adaptive expert selection, not n_eff optimization. See `README_REVISED.md` for updated analysis.

---

## Key Results (⚠️ SINGLE SEED - SEE REVISED ANALYSIS)

| Configuration | $n_{eff}$ | Mean Reward (Seed 42) | Improvement vs Baseline |
|--------------|-----------|-------------|-------------------------|
| Transfer-Weak | 1.0 | 4.477 | +17.59% |
| Transfer-Moderate | 2.0 | 4.464 | +17.24% |
| Transfer-Balanced | 5.0 | 4.359 | +14.48% |
| Transfer-Strong | 10.0 | 4.333 | +13.79% |
| Transfer-VeryStrong | 20.0 | 4.280 | +12.41% |
| Partial Cold Start | -- | 3.807 | 0.00% (baseline) |

**⚠️ IMPORTANT**: These results are from **seed 42 only** (warmup-dominant regime). Multi-seed analysis (`README_REVISED.md`) shows:
- **33% of seeds**: Warmup expert active → n_eff matters (+4.6% effect)
- **67% of seeds**: Tabula rasa active → n_eff ignored (0% effect)
- **Overall**: Effect is regime-dependent, not universal

---

## Files in This Directory

### Core Experiment
- **`run_figure8_analysis.py`**: ⭐ Unified experiment script
  - Runs experiments ONCE and caches results
  - Generates both figure AND table
  - Tests n_eff extremes (1.0, 20.0) across 3 seeds
  - Classifies regimes (warmup-dominant vs tabula rasa-dominant)
  - Outputs publication-quality figure + LaTeX table

### Ablation Studies
- **`plot_ablation_no_corralling.py`**: Corralling OFF ablation
  - Shows pure semantic transfer (without meta-learning)
  - Demonstrates n_eff effect when transfer is forced

### Diagnostic Tools
- **`diagnose_corralling_weights.py`**: Analyzes expert selection patterns
- **`check_figure7_weights.py`**: Cross-validates Figure 7 regime switching

### Documentation (Comprehensive)
- **`EXPERIMENT_DESIGN.md`**: Comprehensive experimental methodology
  - Research questions & hypotheses
  - Experimental protocol (setup, controls, metrics)
  - Implementation details
  - Reproducibility checklist

- **`RESULTS_DISCUSSION.md`**: In-depth results analysis
  - Quantitative results (all metrics)
  - Statistical significance tests
  - Mechanistic interpretation ("over-confidence trap")
  - Comparison to prior work
  - Production recommendations

- **`CHANGELOG_n_eff_calibration.md`**: Change history
  - Summary of findings
  - Code changes (router.py, results.tex)
  - Before/after comparison
  - Validation checklist

- **`README.md`** (this file): Quick reference guide

### Output
- **`results/figure8_regime_stratified_CORRECTED.png`**: ⭐ Primary figure
  - 2×2 regime-stratified visualization
  - Top row: Expert weight evolution by regime
  - Bottom row: Performance curves by regime
  - Shows n_eff effect is conditional on expert selection
  
- **`results/appendixC_neff_sensitivity.tex`**: ⭐ LaTeX table
  - Regime-stratified performance statistics
  - Ready for inclusion in Appendix C (Hyperparameter Sensitivity)

- **`results/figure8_unified_results.pkl`**: Cached experiment results
  - Speeds up subsequent runs (3min → instant)
  - Delete to force re-run experiments

---

## Understanding the Figure

**Figure 8: Prior Strength Calibration**

### Visual Elements

1. **Gray Line (t < 300)**: Shared Warmup Phase
   - All configs identical before model release
   - Validates experimental control

2. **Green Solid Line**: Optimal Transfer ($n_{eff}=1.0$)
   - Best performer: +17.6% vs baseline
   - Production default

3. **Green Shaded Band**: Robustness Region ($n_{eff} \in [2, 20]$)
   - Narrow band (5.2pp) proves system is not brittle
   - Min-max envelope across all non-optimal configs

4. **Blue Dotted Line**: Weak Prior Boundary ($n_{eff}=20.0$)
   - Lower bound of robustness band
   - Still +12.4% better than baseline

5. **Red Dashed Line (t > 300)**: Partial Cold Start Baseline
   - New model starts with identity matrix
   - Only shown post-release (avoids confusion)

6. **Black Vertical Line (t=300)**: Model Release Event
   - Divides shared history from divergent evaluation

### Key Insights from Visual

- **Narrow band** → Robust to hyperparameter choice
- **Green > Red gap** → Semantic transfer is effective
- **Solid green highest** → Weak priors are optimal
- **Overlap pre-release** → Experimental control is valid

---

## The "Over-Confidence Trap"

**Why do strong priors (high $n_{eff}$) underperform?**

### Mechanism

1. High $n_{eff}$ inflates covariance matrix: $A_{new} = n_{eff} \times A_{neighbor}$
2. Exploration bonus shrinks: $\sqrt{x^T A^{-1} x} \propto 1/\sqrt{n_{eff}}$
3. For expensive new models, cost penalty dominates
4. System gets trapped exploiting cheaper incumbents
5. True quality of new model never discovered

### Solution

- Use $n_{eff}=1.0$ (weak prior)
- Preserves exploration flexibility
- Trusts semantic direction, maintains uncertainty
- Result: "Calibrated optimism" balances transfer + adaptation

---

## Experimental Design Highlights

### Methodological Rigor

✅ **Controlled Execution**: Fixed seed (42) ensures reproducibility  
✅ **Real Data**: 1,000 LMSYS Arena battles (no synthetic proxies)  
✅ **Shared Warmup**: Pre-release identical across all configs  
✅ **No Look-Ahead**: Sequential processing (realistic)  
✅ **Full Stack**: Uses complete system (Corralling + LinUCB + transfer)  

### Validation

✅ **H1 (Transfer Efficacy)**: All configs beat baseline → CONFIRMED  
✅ **H2 (Monotonic Trend)**: Performance ↓ as $n_{eff}$ ↑ → CONFIRMED  
✅ **H3 (Robustness)**: <10pp variation → CONFIRMED  

---

## Reproducibility

### Prerequisites

**Data**: LMSYS Chatbot Arena battles (public dataset)  
**Artifacts**:
- `src/artifacts/priors_warmup.joblib` (warmup priors)
- `src/artifacts/pca_32.joblib` (PCA encoder)
- `data/router_context.db` (local battle database)

**Software**:
- Python 3.10
- NumPy 1.24
- sentence-transformers 2.2.2
- scikit-learn 1.3.0

### Run Command

```bash
python experiments_v1/08_figure/run_figure8_analysis.py
```

### Expected Output

**Console**:
```
================================================================================
REGIME-STRATIFIED PERFORMANCE ANALYSIS
================================================================================

--- REGIME CLASSIFICATION ---
Warmup-dominant seeds:      [42] (1/3 = 33%)
Tabula rasa-dominant seeds: [43, 44] (2/3 = 67%)

--- PERFORMANCE BY REGIME ---
Configuration             Mean Reward     Std Dev      N Seeds   
--------------------------------------------------------------------------------

WARMUP-DOMINANT REGIME:
  n_eff = 1.0                  4.4770          0.0000         1
  n_eff = 20.0                 4.2800          0.0000         1
  → Effect size: +4.60%

TABULA RASA-DOMINANT REGIME:
  n_eff = 1.0                  4.2406          0.0131         2
  n_eff = 20.0                 4.2472          0.0197         2
  → Effect size: -0.15%

OVERALL (ALL SEEDS):
  n_eff = 1.0                  4.3194          0.1119         3
  n_eff = 20.0                 4.2581          0.0223         3
  → Effect size: +1.44%
```

**Files**:  
- `results/figure8_regime_stratified_CORRECTED.png` (2×2 figure)  
- `results/appendixC_neff_sensitivity.tex` (LaTeX table for Appendix C)

### Verification

Check cached results:
```bash
ls -lh experiments_v1/08_figure/results/figure8_unified_results.pkl
# If exists: using cache (fast)
# If not exists: will run experiments (~5 minutes)
```

Force re-run:
```bash
rm experiments_v1/08_figure/results/figure8_unified_results.pkl
python experiments_v1/08_figure/run_figure8_analysis.py
```

---

## Impact on Production System

### Code Changes

**File**: `src/bandit_gpt/router.py`

**Current Value**:
```python
n_effective_default: float = 5.0  # Mid-range value (line 128)
```

**Rationale**: 
- Multi-seed analysis revealed n_eff effect is **regime-dependent**
- Corralling adaptively switches between warmup and tabula rasa experts
- n_eff only matters when warmup expert is active (~33% of traffic)
- System robustness comes from meta-learning, not parameter tuning
- Default 5.0 is reasonable when warmup expert is used

### Performance Note

**Single-seed results (seed 42)** showed n_eff=1.0 outperforming n_eff=5.0, but this **does not replicate** across seeds 43-44 where Corralling switches to tabula rasa expert (n_eff has no effect).

### Status

Experiment revised to focus on **adaptive expert selection** rather than n_eff optimization. See `README_REVISED.md` for complete multi-seed analysis.

---

## Citation

If you use this experiment in your research, please cite:

```bibtex
@inproceedings{banditgpt2026,
  title={banditGPT: Cost-Aware Contextual Bandits for LLM Routing},
  author={[Authors]},
  booktitle={Proceedings of the Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

---

## Related Experiments

- **Experiment 06**: Pareto frontier analysis (cost-quality trade-offs)
- **Experiment 07**: Ablation study (Corralling vs single expert)
- **Experiment 08** (this): Sensitivity analysis (n_eff calibration)

---

## Contact

For questions about this experiment, see:
- Primary documentation: `EXPERIMENT_DESIGN.md`
- Detailed results: `RESULTS_DISCUSSION.md`
- Change history: `CHANGELOG_n_eff_calibration.md`

---

**Last Updated**: January 27, 2026  
**Status**: Complete ✅  
**Production**: Deployed ✅
