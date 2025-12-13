# SummEdits Dataset Organization

**Date**: December 13, 2025  
**Action**: Consolidated and organized SummEdits benchmark files

## Summary

All SummEdits-related files have been consolidated into a single organized directory structure at:
```
KDD/data/sumedits/
```

## What Was Organized

### 1. Ground Truth Datasets (10 files)
**Source**: `factualNLG/data/summedits/`  
**Destination**: `KDD/data/sumedits/ground_truth/`

Files moved:
- summedits_news.json (3.0 MB)
- summedits_podcast.json (3.1 MB)
- summedits_billsum.json (8.6 MB)
- summedits_samsum.json (770 KB)
- summedits_sales_call.json (1.6 MB)
- summedits_sales_email.json (2.0 MB)
- summedits_shakespeare.json (4.6 MB)
- summedits_scitldr.json (762 KB)
- summedits_qmsumm.json (1.6 MB)
- summedits_ectsum.json (1.1 MB)

**Total**: ~27 MB of ground truth data

### 2. Prompt Templates (4 files)
**Source**: `factualNLG/prompts/summedits/`  
**Destination**: `KDD/data/sumedits/prompts/`

Files moved:
- standard_zs_prompt.txt
- step2_consistent.txt
- step2_inconsistent.txt
- edit_typing_gpt4.txt

### 3. Model Evaluation Scores (12 files)
**Source**: `data/summedits_*_scores.json`  
**Destination**: `KDD/data/sumedits/model_scores/`

Files moved:
- summedits_aggregate_scores.json
- summedits_aggregate_detailed.json
- Per-domain score files for all 10 domains (e.g., summedits_news_scores.json)

### 4. Evaluation Scripts (2 files)
**Source**: `research/kdd/`  
**Destination**: `KDD/data/sumedits/`

Files copied:
- run_summedits.py (31 KB) - Main evaluation script
- aggregate_summedits_scores.py (6.1 KB) - Score aggregation script

## Directory Structure

```
KDD/data/sumedits/
├── README.md                          # Documentation of the dataset and structure
├── run_summedits.py                   # Main evaluation script
├── aggregate_summedits_scores.py      # Score aggregation script
├── ground_truth/                      # Original SummEdits datasets with labels
│   ├── summedits_billsum.json
│   ├── summedits_ectsum.json
│   ├── summedits_news.json
│   ├── summedits_podcast.json
│   ├── summedits_qmsumm.json
│   ├── summedits_sales_call.json
│   ├── summedits_sales_email.json
│   ├── summedits_samsum.json
│   ├── summedits_scitldr.json
│   └── summedits_shakespeare.json
├── prompts/                           # Prompt templates for evaluation
│   ├── edit_typing_gpt4.txt
│   ├── standard_zs_prompt.txt
│   ├── step2_consistent.txt
│   └── step2_inconsistent.txt
└── model_scores/                      # Model evaluation results
    ├── summedits_aggregate_detailed.json
    ├── summedits_aggregate_scores.json
    ├── summedits_billsum_scores.json
    ├── summedits_ectsum_scores.json
    ├── summedits_news_scores.json
    ├── summedits_podcast_scores.json
    ├── summedits_qmsumm_scores.json
    ├── summedits_sales_call_scores.json
    ├── summedits_sales_email_scores.json
    ├── summedits_samsum_scores.json
    ├── summedits_scitldr_scores.json
    └── summedits_shakespeare_scores.json
```

## Benefits of This Organization

1. **Centralized Location**: All SummEdits materials are now in one place under `KDD/data/`
2. **Clear Categorization**: Files are organized by type (ground truth, prompts, scores, scripts)
3. **Easy Access**: Researchers can quickly find what they need
4. **Self-Contained**: Evaluation scripts included in the same directory for convenience
5. **Documentation**: README.md provides comprehensive information about the dataset and evaluation process
6. **Consistency**: Follows the same organizational pattern as other KDD data

## Original File Locations

The original files remain in their source locations:
- `factualNLG/data/summedits/` - Original ground truth data (preserved)
- `factualNLG/prompts/summedits/` - Original prompts (preserved)
- `data/summedits_*_scores.json` - Original score files (preserved)
- `research/kdd/run_summedits.py` - Original evaluation script (preserved)
- `research/kdd/aggregate_summedits_scores.py` - Original aggregation script (preserved)

Files were **copied** (not moved) to maintain backward compatibility with existing scripts and workflows.

## Related Scripts

Evaluation scripts that use these files:
- `research/kdd/run_summedits.py` - Main evaluation script
- `research/kdd/aggregate_summedits_scores.py` - Aggregation script
- `research/kdd/debug_gpt5_summedits.py` - Debugging utility
- `scripts/analysis/analyze_summedits.py` - Analysis script

## Next Steps

Consider updating scripts to reference the new centralized location:
- Update path constants in evaluation scripts
- Update documentation to point to new location
- Deprecate old locations once all references are updated
