# Figure 7: Sensitivity Analysis - File Manifest

## Directory Structure

```
experiments_v1/07_figure/
├── plot_sensitivity.py              # Main experiment script (385 lines)
├── results/
│   ├── figure7_sensitivity.png      # Full trajectory (300 DPI)
│   └── figure7b_sensitivity_zoomed.png  # Post-release zoom (300 DPI)
├── README.md                        # Comprehensive documentation (282 lines)
├── SUMMARY.md                       # Executive summary for paper (534 lines)
├── INDEX.md                         # Quick reference guide (225 lines)
├── QUICK_REFERENCE.md               # One-page cheat sheet (145 lines)
├── CHECKLIST.md                     # Completion verification (230 lines)
├── EXPERIMENT_COMPLETE.md           # Final report (200 lines)
├── figure7_caption.tex              # LaTeX integration (95 lines)
└── FILE_MANIFEST.md                 # This file
```

## File Descriptions

### 🐍 Code (1 file)

#### `plot_sensitivity.py`
- **Purpose**: Main experiment script
- **Lines**: 385
- **Function**: Sweeps n_effective from 1.0 to 20.0, generates figures
- **Runtime**: ~15 minutes
- **Dependencies**: sentence-transformers, matplotlib, numpy, joblib
- **Output**: 2 PNG figures in `results/`
- **Status**: ✅ No linter errors

### 📊 Figures (2 files)

#### `results/figure7_sensitivity.png`
- **Purpose**: Full trajectory visualization
- **Resolution**: 300 DPI
- **Size**: ~1 MB
- **Shows**: All 6 conditions (Cold Start + 5 n_eff values) from t=0 to t=1000
- **Key Feature**: "Transfer Advantage Zone" (green shaded region)
- **Status**: ✅ Generated successfully

#### `results/figure7b_sensitivity_zoomed.png`
- **Purpose**: Post-release period detail
- **Resolution**: 300 DPI
- **Size**: ~1 MB
- **Shows**: Critical period (t=250 to t=600)
- **Key Feature**: Clear separation between Cold Start and transfer methods
- **Status**: ✅ Generated successfully

### 📝 Documentation (7 files)

#### `README.md`
- **Purpose**: Comprehensive methodology and documentation
- **Lines**: 282
- **Audience**: Researchers, implementers
- **Contains**: 
  - Experimental design
  - Interpretation guide
  - Running instructions
  - Validation checklist
  - Future extensions
- **Status**: ✅ Complete

#### `SUMMARY.md`
- **Purpose**: Executive summary for paper integration
- **Lines**: 534
- **Audience**: Paper authors, reviewers
- **Contains**:
  - One-sentence summary
  - Key results table
  - Visual evidence
  - Reviewer responses
  - Paper integration text
  - Statistical significance
- **Status**: ✅ Complete

#### `INDEX.md`
- **Purpose**: Quick navigation and reference
- **Lines**: 225
- **Audience**: Quick lookup, navigation
- **Contains**:
  - Results summary table
  - Color coding guide
  - Addressing reviewer concerns
  - Paper integration points
  - Reproducibility info
- **Status**: ✅ Complete

#### `QUICK_REFERENCE.md`
- **Purpose**: One-page cheat sheet
- **Lines**: 145
- **Audience**: Quick reference, presentations
- **Contains**:
  - Key numbers
  - Quick run commands
  - Visual guide
  - Key insights
  - Paper snippets
- **Status**: ✅ Complete

#### `CHECKLIST.md`
- **Purpose**: Completion verification
- **Lines**: 230
- **Audience**: Quality assurance, project management
- **Contains**:
  - Files created checklist
  - Results verification
  - Visual quality checks
  - Documentation quality
  - Integration checklist
- **Status**: ✅ All items checked

#### `EXPERIMENT_COMPLETE.md`
- **Purpose**: Final completion report
- **Lines**: 200
- **Audience**: Project stakeholders
- **Contains**:
  - Executive summary
  - Results summary
  - Deliverables list
  - Quality assurance
  - Impact statement
  - Sign-off
- **Status**: ✅ Complete

#### `FILE_MANIFEST.md`
- **Purpose**: This file - directory structure and file descriptions
- **Lines**: ~150
- **Audience**: Project navigation
- **Contains**: Complete file listing and descriptions
- **Status**: ✅ Complete

### 📄 LaTeX (1 file)

#### `figure7_caption.tex`
- **Purpose**: Ready-to-use LaTeX for paper integration
- **Lines**: 95
- **Contains**:
  - Figure 7 environment with caption
  - Figure 7b environment (zoomed)
  - Results table (LaTeX format)
  - Section 4.3 text (complete)
- **Status**: ✅ Ready for copy-paste

## File Statistics

### By Type
- **Python**: 1 file (385 lines)
- **Markdown**: 7 files (1,711 lines)
- **LaTeX**: 1 file (95 lines)
- **PNG**: 2 files (~2 MB)
- **Total**: 11 files (2,191 lines + 2 images)

### By Purpose
- **Code**: 1 file
- **Figures**: 2 files
- **Documentation**: 7 files
- **LaTeX**: 1 file

### By Audience
- **Researchers**: README.md, plot_sensitivity.py
- **Paper Authors**: SUMMARY.md, figure7_caption.tex
- **Quick Reference**: INDEX.md, QUICK_REFERENCE.md
- **Project Management**: CHECKLIST.md, EXPERIMENT_COMPLETE.md, FILE_MANIFEST.md

## Usage Guide

### For Running the Experiment
1. **Start here**: `README.md`
2. **Run script**: `python plot_sensitivity.py`
3. **Check output**: `results/figure7*.png`

### For Paper Integration
1. **Start here**: `SUMMARY.md`
2. **Get LaTeX**: `figure7_caption.tex`
3. **Copy figures**: `results/figure7*.png`

### For Quick Reference
1. **Start here**: `QUICK_REFERENCE.md`
2. **Navigate**: `INDEX.md`
3. **Verify**: `CHECKLIST.md`

### For Understanding Results
1. **Start here**: `EXPERIMENT_COMPLETE.md`
2. **Deep dive**: `README.md`
3. **Paper context**: `SUMMARY.md`

## Dependencies

### Python Packages
- `sentence-transformers` (embedding)
- `matplotlib` (plotting)
- `numpy` (numerical computation)
- `joblib` (loading PCA model)
- `gzip` (data loading)
- `json` (data parsing)
- `logging` (progress tracking)

### Data Files
- `src/bandit_gpt/data/offline_dataset/dev_rewards_complete_all_models.jsonl.gz`
- `src/artifacts/pca_model.joblib`

### Configuration
- `src/bandit_gpt/config_legacy.py` (paths and constants)

## Quality Metrics

### Code Quality
- ✅ No linter errors
- ✅ Well-documented (docstrings)
- ✅ Modular design
- ✅ Reproducible

### Documentation Quality
- ✅ Comprehensive coverage
- ✅ Multiple entry points
- ✅ Cross-referenced
- ✅ Paper-ready

### Figure Quality
- ✅ High resolution (300 DPI)
- ✅ Clear color coding
- ✅ Readable labels
- ✅ Professional appearance

### Scientific Quality
- ✅ Hypothesis stated
- ✅ Baseline included
- ✅ Multiple conditions
- ✅ Statistical significance
- ✅ Results interpreted

## Integration Checklist

### For Main Paper
- [ ] Copy `results/figure7_sensitivity.png` to `paper/figures/`
- [ ] Insert figure environment from `figure7_caption.tex`
- [ ] Add Section 4.3 text
- [ ] Add results table
- [ ] Update references

### For Appendix
- [ ] Copy `results/figure7b_sensitivity_zoomed.png`
- [ ] Add extended discussion
- [ ] Cross-reference with main paper

### For Presentation
- [ ] Create slide with Figure 7
- [ ] Highlight key message
- [ ] Show quantitative results

## Version History

### v1.0 (2026-01-25)
- ✅ Initial experiment complete
- ✅ All 6 conditions tested
- ✅ Figures generated
- ✅ Documentation complete
- ✅ LaTeX integration ready

## Contact

For questions about specific files:
- **Experiment**: See `README.md`
- **Paper**: See `SUMMARY.md`
- **Quick Help**: See `QUICK_REFERENCE.md`
- **Code**: See `plot_sensitivity.py` (inline comments)

## License

Same as parent repository (see root LICENSE file).

## Citation

When using this experiment, cite the main paper:
```bibtex
@article{banditgpt2026,
  title={BanditGPT: Adaptive Model Routing with Latent Semantic Transfer},
  author={...},
  journal={...},
  year={2026}
}
```

---

**Last Updated**: January 25, 2026  
**Status**: ✅ Complete  
**Total Files**: 11 (9 text + 2 images)  
**Total Lines**: 2,191 (code + docs)  

