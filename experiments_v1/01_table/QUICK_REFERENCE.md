# Table 1: Quick Reference Card

## 📊 The Table (One Glance)

**81,871 prompts** across 5 semantic categories for routing evaluation

| Split | Size | Purpose | Source |
|-------|------|---------|--------|
| Warmup | 80,000 | PCA (384→**32**) + LinUCB priors (A∈ℝ³³ˣ³³, b∈ℝ³³) | LMSYS Arena |
| Dev | 1,121 | Online learning | KDD splits |
| Holdout | 750 | Final eval | KDD splits |

## 🎯 Semantic Breakdown

```
Coding (39%)         ████████████████████████████████████████
Conversational (38%) ████████████████████████████████████████
Creative (10%)       ██████████
Knowledge (10%)      ██████████  
Math/Logic (4%)      ████
```

## 🔢 Critical Numbers

- **PCA Components**: 32 (not 23!)
- **LinUCB Dimension**: 33 (32 + 1 bias)
  - **A matrix** (covariance): 33×33 = 1,089 parameters per model
  - **b vector** (beliefs): 33 parameters per model
- **Variance Retained**: ~90%
- **Size Reduction**: 92% (384→32)
- **Total Dataset**: 81,871 prompts

## 📁 File Locations

```
experiments_v1/01_table/
├── table_dataset_composition.tex  ← USE THIS IN PAPER
├── analyze_dataset_composition.py ← Regenerate if needed
├── README.md                      ← Full documentation
├── DATA_PROVENANCE.md             ← Detailed provenance
├── SUMMARY.md                     ← Executive summary
└── QUICK_REFERENCE.md            ← This file
```

## 🚀 Quick Actions

### Include in Paper
```latex
\input{experiments_v1/01_table/table_dataset_composition.tex}
```

### Regenerate Table
```bash
cd experiments_v1/01_table
python analyze_dataset_composition.py
```

### Verify PCA Dimensions
```bash
python -c "import joblib; p = joblib.load('src/artifacts/pca_32.joblib'); print(f'Components: {p.n_components_}')"
```

## 💡 Key Points for Paper

1. **Scale**: 81,871 prompts from public LMSYS Arena data
2. **Coverage**: 5 semantic categories (Coding, Conversational, Creative, Knowledge, Math/Logic)
3. **Quality**: Stratified splits, no data leakage, verified rewards
4. **Efficiency**: PCA reduces embeddings by 92% (384→32 dims)
5. **Reproducibility**: All scripts and data sources provided

## 📝 Citing Data Sources

```bibtex
@article{ong2024routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and others},
  journal={arXiv preprint arXiv:2406.18665},
  year={2024}
}

@inproceedings{zheng2023judging,
  title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author={Zheng, Lianmin and others},
  booktitle={NeurIPS},
  year={2023}
}
```

## ✅ Checklist

- [x] Created LaTeX table
- [x] Verified PCA dimensions (32 components)
- [x] Analyzed semantic categories
- [x] Documented data provenance
- [x] Provided regeneration scripts
- [x] No data leakage
- [x] Public data sources
- [x] Ready for KDD submission

## 🔍 Data Provenance (1 Sentence Each)

- **Warmup (80k)**: LMSYS Arena battles from HuggingFace `routellm/gpt4_judge_battles`, used for PCA (32 components) and LinUCB priors (covariance matrix **A** ∈ ℝ³³ˣ³³ and belief vector **b** ∈ ℝ³³ for each model).
- **Dev (1,121)**: Stratified KDD splits with mixtral-8x7b and gpt-4o, used for online learning and calibration.
- **Holdout (750)**: Held-out KDD splits with same models, used for final evaluation.

### What are LinUCB Priors?
LinUCB (Linear Upper Confidence Bound) maintains two matrices per model:
- **A (covariance)**: Captures feature correlations and uncertainty (33×33)
- **b (belief)**: Encodes reward expectations for different contexts (33×1)

These are initialized using 80k warmup prompts to "warm-start" the bandit with prior knowledge.

## 🎨 Table Format

- ✅ KDD compliant (booktabs)
- ✅ Descriptive caption
- ✅ Detailed notes
- ✅ Citations included
- ✅ Ready to compile

## 🔗 Related Files

- PCA artifact: `src/artifacts/pca_32.joblib`
- Warmup priors: `src/artifacts/priors_warmup.joblib`
- Dev prompts: `data/dev_prompts_for_rejudge.jsonl`
- Holdout prompts: `data/holdout_prompts_for_rejudge.jsonl`

---

**Status**: ✅ Complete  
**PCA**: 32 components (33 with bias) ✓  
**Last Updated**: 2026-01-24

