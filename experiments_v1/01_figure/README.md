# Figure 1: Alignment Tax Discovery

This folder generates Figure 1 for the paper, which visualizes the forensic discovery of the Alignment Tax in LMSYS holdout data.

## Files

### Generation Script
- `plot_lmsys_holdout_pca.py` - Main script to generate Figure 1
  - Projects 1,871 LMSYS prompts onto PCA space
  - Validates clusters against ground-truth reward gaps
  - Outputs visualization showing Alignment Tax discovery

### LaTeX Files
- `figure_1_caption.tex` - Figure caption for the paper
- `results_explanation.tex` - Detailed explanation for appendix/supplementary

### Results
- `results/figure1_lmsys_holdout_pca.png` - Main figure (300 DPI)
- `results/figure1_lmsys_holdout_pca_hires.png` - High-res version (600 DPI)

## Key Findings

The figure demonstrates:
- **Low PC1 (82.4%)**: Natural Language Zone where GPT-4-Turbo wins (+0.133)
- **High PC1 (17.6%)**: Strictness Zone where Mixtral wins (-0.682)
- **85% of High PC1**: Dominated by strict completion templates
- **Alignment Tax**: RLHF optimization makes frontier models fail at strict constraints

## Usage

```bash
python experiments_v1/01_figure/plot_lmsys_holdout_pca.py
```

Requires:
- Pre-trained PCA model from RouteLLM data
- LMSYS dev/holdout data with reward evaluations
- Sentence transformer model for embeddings

