# Domain Alignment Experiment: Clean Folder Structure

## ✅ Cleaned and Ready for KDD Submission

This folder contains all essential files for the **Domain Alignment via Covariance Inflation** experiment, with redundant and exploratory files removed.

---

## 📁 Folder Structure

```
pareto_stability_analysis/
│
├── 📄 README.md                                    # Experiment overview
├── 📄 KDD_FINAL_FRAMEWORK.md                       # Complete paper framework
├── 📄 DOMAIN_ALIGNMENT_FRAMEWORK.md                # When and why to recalibrate
│
├── 📜 paper_section.tex                            # LaTeX methods section (with domain alignment)
├── 📜 paper_results.tex                            # LaTeX results section
│
├── 🐍 run_domain_adaptation_inertia_corrected.py   # Main experiment (covariance inflation)
├── 🐍 create_bimodal_heatmap.py                    # Visualization script
├── 🐍 create_merged_dataset.py                     # Data preparation script
├── 🐍 compute_static_oracle.py                     # Oracle baseline calculation
│
└── 📂 results/
    ├── 📊 domain_adaptation_gamma_scaling.png           # Main figure (4-panel)
    ├── 📊 bimodal_discovery_heatmap_hires.png           # Domain mismatch visualization
    │
    ├── 📄 domain_adaptation_gamma_scaling_results.json  # Numerical results
    ├── 📄 bimodal_heatmap_data.json                     # Heatmap data
    ├── 📄 static_oracle_results.json                    # Oracle baseline
    │
    ├── 💾 eval_rewards_mixtral_gpt4turbo.jsonl          # Evaluation dataset (747 prompts)
    └── 💾 eval_dataset_metadata.json                    # Dataset statistics
```

---

## 🎯 Key Files for Paper

### Figures (Publication-Ready)

1. **`results/domain_adaptation_gamma_scaling.png`**
   - 4-panel visualization showing covariance inflation results
   - Includes KDD contribution statement as suptitle
   - Shows: adaptation curves, final usage vs γ, quality preservation, calibration delta

2. **`results/bimodal_discovery_heatmap_hires.png`** (300 DPI)
   - Side-by-side comparison of source (80K synthetic) vs target (747 real)
   - Demonstrates "Oil and Water" bimodal pattern
   - Shows 0% moderate tasks in real data vs 19.9% in synthetic

### Data Files

1. **`results/eval_rewards_mixtral_gpt4turbo.jsonl`**
   - 747 real-world prompts with GPT-4o judged rewards
   - Format: `{"prompt_id": str, "model_id": str, "raw_score": float}`
   - Used for calibration (149) and holdout (598) evaluation

2. **`results/static_oracle_results.json`**
   - Optimal static routing thresholds for different λ values
   - Provides upper bound for static (non-contextual) routing

3. **`results/domain_adaptation_gamma_scaling_results.json`**
   - Complete numerical results for all γ values tested
   - Includes calibration metrics, holdout results, and Oracle comparison

### Documentation

1. **`KDD_FINAL_FRAMEWORK.md`**
   - Complete paper framework with correct mathematical notation
   - Research question, key discoveries, experimental setup
   - Results, visualizations, and reviewer responses

2. **`DOMAIN_ALIGNMENT_FRAMEWORK.md`**
   - When and why to apply covariance inflation
   - One-time vs continuous recalibration
   - Practical deployment workflow
   - FAQ for reviewers

3. **`paper_section.tex`**
   - LaTeX methods section with domain alignment subsection
   - Includes mathematical formulation of covariance inflation
   - References Table 1 (gamma scaling results)

---

## 🚀 Reproducing Results

### Run All Experiments

```bash
cd /Users/annette/repostitories/banditGPT/experiments_v1/pareto_stability_analysis

# 1. Create merged dataset (Mixtral + GPT-4-turbo)
python3 create_merged_dataset.py

# 2. Compute static oracle baseline
python3 compute_static_oracle.py

# 3. Run domain adaptation with covariance inflation
python3 run_domain_adaptation_inertia_corrected.py

# 4. Create bimodal heatmap visualization
python3 create_bimodal_heatmap.py
```

### Expected Runtime
- Dataset creation: ~10 seconds
- Oracle computation: ~5 seconds
- Domain adaptation: ~2 minutes (4 γ values × 747 prompts)
- Bimodal heatmap: ~3 minutes (embedding 5,747 prompts)

**Total**: ~6 minutes on a standard laptop

---

## 📊 Key Results

| γ | N_eff | Calib/Prior | Adaptation | Final GPT-4% | vs Oracle |
|---|-------|-------------|------------|--------------|-----------|
| 1.0 | 80,000 | 0.002 | 0% | 99.7% | +80.4% |
| 0.1 | 8,000 | 0.019 | -8% | 96.8% | +77.5% |
| 0.01 | 800 | 0.186 | -36% | 65.7% | +46.4% |
| **0.002** | **160** | **0.931** | **-56%** | **40.0%** | **+20.7%** |
| Oracle | — | — | — | 19.3% | — |

**Key Finding**: With γ=0.002, the router achieves **74% reduction in GPT-4 over-usage** (from +80.4% to +20.7%), demonstrating successful cross-domain transfer.

---

## 📝 What Was Removed

### Scripts (Superseded)
- ❌ `run_banditgpt.py` (not in final narrative)
- ❌ `run_domain_adaptation.py` (superseded by inertia_corrected version)
- ❌ `run_domain_adaptation_recalibrated.py` (only recalibrated b, not A)
- ❌ `analyze_difficulty.py` (exploratory)
- ❌ `analyze_full_distribution.py` (exploratory)

### Logs
- ❌ All `*.log` files

### Documentation (Redundant/Outdated)
- ❌ `ALGORITHM.md`, `ALGORITHM_SUMMARY.md` (too detailed)
- ❌ `COMPARISON_SUMMARY.md`, `FINDINGS_SUMMARY.md` (consolidated)
- ❌ `DATA_STATUS.md`, `IMPLEMENTATION_READY.md` (outdated)
- ❌ `IMPROVEMENTS_SUMMARY.md`, `IMPLEMENTATION_COMPLETE.md` (redundant)
- ❌ `KDD_PAPER_FRAMEWORK.md` (consolidated into KDD_FINAL_FRAMEWORK)
- ❌ `ORACLE_WARNING.md`, `ROUTELLM_BASELINE_STRATEGY.md` (incorporated)
- ❌ `DOMAIN_ADAPTATION_FRAMEWORK.md` (superseded by DOMAIN_ALIGNMENT_FRAMEWORK)

### Results (Old/Exploratory)
- ❌ `banditgpt_results.json`
- ❌ `domain_adaptation_curves.png` (old)
- ❌ `domain_adaptation_recalibrated_curves.png` (old)
- ❌ `domain_adaptation_recalibrated_results.json` (old)
- ❌ `domain_adaptation_results.json` (old)
- ❌ `difficulty_analysis.json`, `difficulty_analysis.png` (exploratory)
- ❌ `full_difficulty_analysis.json`, `full_difficulty_distribution.png` (exploratory)
- ❌ `bimodal_discovery_3d.png` (redundant, kept 2D high-res)
- ❌ `bimodal_discovery_heatmap.png` (lower res, kept high-res version)
- ❌ `COMPARISON_SUMMARY.md` (redundant)

### Misc
- ❌ `__pycache__/` (Python cache)
- ❌ `config.json` (unused)
- ❌ `paper/` (empty directory)

---

## ✅ Status

**READY FOR KDD SUBMISSION**

- ✅ Correct mathematical notation (γ ∈ (0,1])
- ✅ Clean folder structure (essential files only)
- ✅ Publication-ready figures (300 DPI)
- ✅ Complete LaTeX sections
- ✅ Reproducible experiments (~6 minutes)
- ✅ Comprehensive documentation
- ✅ Domain alignment narrative integrated

---

**Last Updated**: 2026-01-22  
**Experiment Status**: ✅ COMPLETE AND CLEAN  
**Paper Status**: 📝 LATEX UPDATED WITH DOMAIN ALIGNMENT

