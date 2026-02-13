# Paper Integration Checklist - Experiment 08 Revisions

**Date**: February 13, 2026  
**Status**: All files created, ready for integration  
**Estimated Time**: 2-3 hours for manual integration  

---

## Files Created for Paper Integration

### **1. Main Figure (REQUIRED)**
- ✅ **`results/figure8_regime_stratified_CORRECTED.png`**
  - **Location in paper**: Figure 8 (replace existing figure)
  - **Caption file**: `figure8_caption_REVISED.tex`
  - **Action**: Copy figure to paper figures directory, include caption

### **2. Supplementary Figures (RECOMMENDED)**
- ✅ `results/figure8_ablation_no_corralling.png`
  - Supplementary Figure S1: "Corralling OFF ablation"
  - Shows pure semantic transfer mechanism (6.2% effect)
  
- ✅ `results/figure8_sensitivity_multiseed_revised.png`
  - Supplementary Figure S2: "Multi-seed validation"
  - Shows confidence intervals across seeds

### **3. Text Sections (REQUIRED)**
- ✅ **`results_section_REVISED.tex`**
  - Complete rewrite of Section 5.3 (Sensitivity Analysis)
  - Two-stage narrative: mechanism + production
  - Tables included inline
  - **Action**: Replace Section 5.3 in `paper/sections/results.tex`

- ✅ **`figure8_caption_REVISED.tex`**
  - Comprehensive 4-paragraph caption
  - Explains regime switching and adaptive behavior
  - **Action**: Use for Figure 8 caption

### **4. Abstract/Introduction Addendums (REQUIRED)**
- ✅ **`abstract_addendum.tex`**
  - Add after paragraph about semantic transfer in abstract
  - Summarizes regime-dependent findings
  - **Action**: Insert into `paper/main.tex` abstract

- ✅ **`contributions_addendum.tex`**
  - New contribution item about sensitivity analysis
  - **Action**: Insert as item 5 in `paper/sections/introduction.tex`

### **5. Limitations Section (REQUIRED)**
- ✅ **`limitations_addendum.tex`**
  - Discusses regime-dependent effects
  - Production monitoring recommendations
  - **Action**: Add to limitations section (Section 6 or Appendix)

### **6. Reviewer Response (REQUIRED FOR REVISION LETTER)**
- ✅ **`REVIEWER_RESPONSE.tex`**
  - Point-by-point response to all concerns
  - Summary of experiments conducted
  - **Action**: Include in revision letter to reviewers

---

## Integration Steps

### **Step 1: Update Main Figure** (15 min)

```bash
# Copy corrected figure to paper directory
cp experiments_v1/08_figure/results/figure8_regime_stratified_CORRECTED.png \
   paper/figures/figure8_regime_stratified.png

# Update figure reference in paper
# In paper/sections/results.tex, change:
#   \includegraphics[width=\textwidth]{figures/figure8_sensitivity_hybrid.png}
# To:
#   \includegraphics[width=\textwidth]{figures/figure8_regime_stratified.png}
```

**Files to edit**:
- `paper/sections/results.tex` - Update figure path

### **Step 2: Replace Section 5.3** (30 min)

**Action**: Replace entire Section 5.3 in `paper/sections/results.tex`

**Current location**: Search for `\subsection{Sensitivity Analysis` in results.tex

**Replacement**: Use content from `results_section_REVISED.tex`

**Verification**:
- [ ] Stage 1 (mechanism) included with Corralling OFF ablation
- [ ] Stage 2 (production) included with multi-seed results
- [ ] Regime-switching explanation included
- [ ] Root cause analysis (71.5% ties) included
- [ ] Tables formatted correctly
- [ ] All references (Figure~\ref{}, Table~\ref{}) updated

### **Step 3: Update Figure Caption** (10 min)

**Action**: Replace Figure 8 caption

**Current location**: Find `\caption{` for Figure 8 in results.tex

**Replacement**: Use content from `figure8_caption_REVISED.tex`

**Verification**:
- [ ] All 4 paragraphs included (top row, bottom row, key insight, experimental details)
- [ ] Label updated: `\label{fig:sensitivity_regime_stratified}`
- [ ] All figure references in text updated to match new label

### **Step 4: Update Abstract** (10 min)

**Action**: Add regime-dependent findings to abstract

**Current location**: `paper/main.tex`, after paragraph about semantic transfer

**Content to add**: `abstract_addendum.tex`

**Verification**:
- [ ] Flows naturally from existing semantic transfer paragraph
- [ ] Mentions 33%/67% split
- [ ] Explains adaptive behavior
- [ ] Keeps abstract under word limit (check conference requirements)

### **Step 5: Add Contribution Item** (10 min)

**Action**: Add new contribution about sensitivity analysis

**Current location**: `paper/sections/introduction.tex`, contributions list (after item 4, before Performance)

**Content to add**: `contributions_addendum.tex`

**Verification**:
- [ ] Numbered correctly (should be item 5)
- [ ] Follows same style as other contributions
- [ ] References figures/tables correctly

### **Step 6: Update Limitations** (15 min)

**Action**: Add regime-dependent discussion to limitations

**Current location**: Search for `\section{Limitations}` or `\subsection{Limitations}` in paper

**Content to add**: `limitations_addendum.tex`

**Verification**:
- [ ] Production monitoring recommendations included
- [ ] Generalizability discussion included
- [ ] Framed as feature, not limitation

### **Step 7: Add Supplementary Figures** (20 min)

**Action**: Create supplementary appendix with additional figures

**Location**: `paper/sections/appendix_X.tex` (create new if needed)

**Content structure**:
```latex
\section{Supplementary Analysis: Sensitivity Study}
\label{app:multiseed_sensitivity}

\subsection{Ablation: Semantic Transfer Without Corralling}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/figure8_ablation_no_corralling.png}
    \caption{Pure semantic transfer mechanism (Corralling disabled)...}
    \label{fig:ablation_no_corralling}
\end{figure}

\subsection{Multi-Seed Validation}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{figures/figure8_sensitivity_multiseed_revised.png}
    \caption{Performance across multiple seeds with confidence intervals...}
    \label{fig:multiseed_validation}
\end{figure}
```

**Copy figures**:
```bash
cp experiments_v1/08_figure/results/figure8_ablation_no_corralling.png \
   paper/figures/
cp experiments_v1/08_figure/results/figure8_sensitivity_multiseed_revised.png \
   paper/figures/
```

### **Step 8: Compile and Verify** (30 min)

**Actions**:
1. Compile LaTeX: `pdflatex paper/main.tex`
2. Resolve any reference errors
3. Check figure numbering (Figure 8 should be regime-stratified)
4. Verify table numbering
5. Check all cross-references resolve correctly
6. Review rendered PDF:
   - [ ] Figure 8 displays correctly (2×2 layout visible)
   - [ ] Caption is readable
   - [ ] Tables in Section 5.3 formatted correctly
   - [ ] References to supplementary figures work
   - [ ] Abstract fits on first page

---

## Verification Checklist

### **Content Verification**
- [ ] All mentions of "n_eff=1.0 optimal" removed or corrected
- [ ] Regime-dependent interpretation used throughout
- [ ] Corralling's adaptive behavior emphasized
- [ ] 33%/67% split mentioned consistently
- [ ] 71.5% ties root cause explained
- [ ] Multi-seed results reported (not just seed 42)
- [ ] Statistical significance (p>0.40) reported
- [ ] Production impact (~1.5%) stated correctly

### **Figure Verification**
- [ ] Figure 8: Regime-stratified visualization (2×2 layout)
- [ ] Supplementary Figure S1: Corralling OFF ablation
- [ ] Supplementary Figure S2: Multi-seed validation
- [ ] All figures have clear captions
- [ ] Figure quality acceptable for publication (300 dpi)

### **Cross-Reference Verification**
- [ ] `\ref{fig:sensitivity_regime_stratified}` works throughout text
- [ ] `\ref{sec:sensitivity_analysis}` points to Section 5.3
- [ ] `\ref{app:multiseed_sensitivity}` points to appendix
- [ ] `\ref{tab:ablation_no_corralling}` and `\ref{tab:multiseed_corralling}` work
- [ ] All supplementary figure references resolve

### **Style Consistency**
- [ ] n_eff formatted consistently (using `\neff` command)
- [ ] Corralling capitalized consistently
- [ ] "Warmup expert" vs "warmup expert" consistent
- [ ] Percentage formatting consistent (4.6% vs 4.6\%)
- [ ] Citation style matches rest of paper

---

## Common Issues and Fixes

### **Issue 1: Figure Too Large**
If Figure 8 (2×2 layout) is too large for page:

**Fix**:
```latex
\includegraphics[width=0.95\textwidth]{figures/figure8_regime_stratified.png}
% Or split into two figures (top row, bottom row)
```

### **Issue 2: Caption Too Long**
If 4-paragraph caption is too verbose:

**Fix**: Use shortened version:
```latex
\caption{\textbf{Regime-Dependent n-Effective Sensitivity.}
Corralling exhibits binary expert switching (top): 100\% warmup 
(left, 33\%) or 100\% tabula rasa (right, 67\%). Performance 
stratified by regime (bottom) reveals n-eff effects only when 
semantic transfer is used (+4.6\% warmup regime, 0\% tabula rasa). 
Overall impact: $\sim$1.5\%. Robustness comes from adaptive 
selection, not parameter insensitivity.}
```

### **Issue 3: Tables Don't Fit**
If tables in Section 5.3 overflow:

**Fix**: Use `\small` or `\footnotesize`:
```latex
\begin{table}[h]
\centering
\small  % Add this
\begin{tabular}{lcc}
...
```

### **Issue 4: References Undefined**
If you see "??" in PDF where references should be:

**Fix**: Run pdflatex twice:
```bash
pdflatex paper/main.tex
pdflatex paper/main.tex  # Second pass resolves references
```

### **Issue 5: Appendix Not Appearing**
If supplementary figures don't show up:

**Fix**: Ensure appendix is included in main.tex:
```latex
\input{sections/appendix_sensitivity}  % Add this line
```

---

## Timeline

| Task | Time | Cumulative |
|------|------|------------|
| Update main figure | 15 min | 15 min |
| Replace Section 5.3 | 30 min | 45 min |
| Update figure caption | 10 min | 55 min |
| Update abstract | 10 min | 65 min |
| Add contribution item | 10 min | 75 min |
| Update limitations | 15 min | 90 min |
| Add supplementary figures | 20 min | 110 min |
| Compile and verify | 30 min | 140 min |
| **TOTAL** | **2h 20min** | |

Add 30-60 min buffer for:
- Resolving LaTeX compilation errors
- Adjusting formatting/spacing
- Proofreading
- Checking consistency

**Total estimated time: 2.5-3 hours**

---

## Post-Integration Testing

After integration, verify:

1. **Compile successfully**: No LaTeX errors
2. **Figures render**: All figures appear correctly in PDF
3. **References work**: No "??" in cross-references
4. **Content accurate**: Claims match experimental data
5. **Style consistent**: Formatting matches rest of paper
6. **Page limits**: Paper doesn't exceed page limit

---

## Files for Revision Letter

When submitting revision:

1. **Cover letter**: Use `REVIEWER_RESPONSE.tex` as template
2. **Diff/changelog**: Mention all sections updated
3. **Supplementary materials**: Include all figures and data files

---

## Contact for Issues

If integration issues arise:

- **Experimental data**: All cached in `experiments_v1/08_figure/results/`
- **Documentation**: See `PAPER_REVISION_GUIDE.md`
- **Complete summary**: See `COMPLETE_FIX_SUMMARY.md`

---

**Status**: ✅ All files ready for integration  
**Next step**: Begin Step 1 (Update main figure)  
**Expected completion**: 2-3 hours from start

---

**Last Updated**: February 13, 2026  
**Prepared by**: Revision Team
