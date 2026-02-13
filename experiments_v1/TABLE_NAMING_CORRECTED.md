# Table Naming Correction - Complete

**Date**: February 13, 2026  
**Status**: ✅ **COMPLETE**

---

## Problem

Table files didn't consistently use the `tableN_*` naming convention (where N is the table number), causing inconsistency with the figure naming scheme.

---

## Solution Applied

Updated all table files to use consistent `tableN_*` naming to match the figure naming convention (`figureN_*`).

---

## Changes Made

### 1. ✅ Updated 01_table (Table 1)

**Script**: `analyze_dataset_composition.py`

**Before**:
```python
output_file = Path(__file__).parent / "table_dataset_composition.tex"
```

**After**:
```python
output_file = Path(__file__).parent / "table1_dataset_composition.tex"
```

**File Renamed**:
- `table_dataset_composition.tex` → `table1_dataset_composition.tex` ✅

---

### 2. ✅ Updated 02_table (Table 2)

**Files Renamed** (6 files):
- `table_02_final.tex` → `table2_final.tex` ✅
- `table_02_final_corrected.tex` → `table2_final_corrected.tex` ✅
- `table_02_merged.tex` → `table2_merged.tex` ✅
- `table_02_merged_corrected.tex` → `table2_merged_corrected.tex` ✅
- `table_02_mismatch_robustness.tex` → `table2_mismatch_robustness.tex` ✅
- `table_02_performance_gap.tex` → `table2_performance_gap.tex` ✅

**Note**: The script `generate_table_from_results.py` uses `--output` argument, so no script changes needed. Just update documentation examples.

---

### 3. ✅ Updated Documentation

**README Files Updated**:
- `01_table/README.md` - Updated all references to use `table1_*`
- `02_table/README.md` - Updated all references to use `table2_*`

---

## Current Table Naming (All Correct)

### Main Paper Tables

| Folder | Paper Table | Primary File | Status |
|--------|-------------|--------------|--------|
| **01_table** | Table 1 | `table1_dataset_composition.tex` | ✅ **FIXED** (was `table_dataset_composition.tex`) |
| **02_table** | Table 2 | `table2_merged.tex` (recommended) | ✅ **FIXED** (was `table_02_merged.tex`) |

### Table 2 Variants (All Corrected)

| File | Purpose | Status |
|------|---------|--------|
| `table2_merged.tex` | **Recommended** - Complete "super table" | ✅ Renamed |
| `table2_final.tex` | Auto-generated from results | ✅ Renamed |
| `table2_final_corrected.tex` | Fixed terminology version | ✅ Renamed |
| `table2_merged_corrected.tex` | Corrected merged version | ✅ Renamed |
| `table2_mismatch_robustness.tex` | Domain mismatch focus | ✅ Renamed |
| `table2_performance_gap.tex` | Performance comparison focus | ✅ Renamed |

### Appendix Tables (Supplementary)

| Folder | Original Name | New Name | Appendix | Status |
|--------|---------------|----------|----------|--------|
| 08_figure | `table8_neff_sensitivity.tex` | `appendixC_neff_sensitivity.tex` | C (Sensitivity) | ✅ Renamed |
| 08_figure | `experiments_table.tex` | `appendixF_experimental_configs.tex` | F (Implementation) | ✅ Renamed |

---

## Naming Convention

### Consistency with Figures

```
Figures:  figureN_description.png
Tables:   tableN_description.tex

Examples:
  figure1_lmsys_holdout_pca.png     ←→  table1_dataset_composition.tex
  figure2_distribution_shift.png    ←→  table2_performance_gap.tex
  figure3_corralled_architecture.png
  ...
```

### Appendix Tables (If Needed)

Following the same convention as appendix figures:
```
Appendix D: appendixD_tablename.tex
Appendix E: appendixE_tablename.tex
```

---

## Verification

### All Tables Now Correctly Named ✅

```bash
$ ls 01_table/*.tex
01_table/table1_dataset_composition.tex                 ✅ Table 1

$ ls 02_table/*.tex
02_table/table2_final.tex                               ✅ Table 2
02_table/table2_final_corrected.tex                     ✅ Table 2 (variant)
02_table/table2_merged.tex                              ✅ Table 2 (recommended)
02_table/table2_merged_corrected.tex                    ✅ Table 2 (variant)
02_table/table2_mismatch_robustness.tex                 ✅ Table 2 (variant)
02_table/table2_performance_gap.tex                     ✅ Table 2 (variant)
```

---

## LaTeX Integration

### Including Tables in Paper

**Table 1**:
```latex
\input{experiments_v1/01_table/table1_dataset_composition.tex}
```

**Table 2** (recommended version):
```latex
\input{experiments_v1/02_table/table2_merged.tex}
```

**Table 2** (alternative - performance gap only):
```latex
\input{experiments_v1/02_table/table2_performance_gap.tex}
```

---

## Scripts Updated

| Script | Location | Change |
|--------|----------|--------|
| `analyze_dataset_composition.py` | `01_table/` | Updated output filename from `table_dataset_composition.tex` to `table1_dataset_composition.tex` |

**Note**: `02_table/generate_table_from_results.py` uses `--output` CLI argument, so no script changes needed.

---

## Usage Examples

### Generating Table 1

```bash
cd experiments_v1/01_table
python analyze_dataset_composition.py
# → Creates table1_dataset_composition.tex
```

### Generating Table 2

```bash
cd experiments_v1/02_table
python generate_table_from_results.py \
    --eta-01-results data/eta_0.1_holdout_multiseed/results_multiseed.json \
    --eta-10-results data/eta_1.0_holdout_multiseed/results_multiseed.json \
    --comparison data/statistical_comparison/comparison_results.json \
    --output table2_final.tex
# → Creates table2_final.tex (now with correct naming)
```

---

### 3. ✅ Updated 08_figure Supplementary Tables

**Script**: `run_figure8_analysis.py`

**Before**:
```python
latex_path = output_dir / "table8_neff_sensitivity.tex"
```

**After**:
```python
latex_path = output_dir / "appendixC_neff_sensitivity.tex"
```

**Files Renamed**:
- `results/table8_neff_sensitivity.tex` → `results/appendixC_neff_sensitivity.tex` ✅
- `experiments_table.tex` → `appendixF_experimental_configs.tex` ✅

**Rationale**: These are supplementary tables for Appendix C (Sensitivity Analysis) and Appendix F (Implementation Details), not main paper tables.

---

## Summary

✅ **Table 1**: Fixed naming (`table_dataset_composition.tex` → `table1_dataset_composition.tex`)  
✅ **Table 2**: Fixed naming (6 files: `table_02_*` → `table2_*`)  
✅ **Appendix Tables**: Fixed naming (2 files: `table8_*` → `appendixC_*`, `experiments_table.tex` → `appendixF_*`)  
✅ **Scripts Updated**: 2 scripts (`analyze_dataset_composition.py`, `run_figure8_analysis.py`)  
✅ **Documentation Updated**: 4 files (2 READMEs, 1 script consolidation summary, 1 figure README)  
✅ **Total Files Renamed**: 9 table files  

**Status**: All table naming is now **consistent with figure naming convention**.

---

## Quick Reference

### Table → Folder → File Mapping

```
Table 1 → 01_table → table1_dataset_composition.tex
Table 2 → 02_table → table2_merged.tex (recommended)
                  → table2_performance_gap.tex (alternative)
                  → table2_final.tex (auto-generated)
```

### Naming Pattern

```
Main paper tables:    tableN_description.tex
Appendix tables:      appendixX_description.tex (where X = section)
```

---

**Last Updated**: February 13, 2026  
**Status**: ✅ COMPLETE  
**Changes**: 
- Scripts updated: 2 (01_table, 08_figure)
- Files renamed: 9 (1 + 6 + 2)
- Documentation updated: 4 files  

---

## Related Documentation

- `FIGURE_NAMING_COMPLETE.md` - Figure naming corrections (figures 1-8)
- `06_figure/FIGURE_NAMING_UPDATED.md` - Appendix figure naming (appendixD/E)

**Overall**: Both figures AND tables now use consistent naming conventions throughout the repository.
