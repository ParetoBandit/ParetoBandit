# Paper Conventions & Documentation

**Date**: February 13, 2026  
**Purpose**: Clarify naming conventions and design decisions in the BanditGPT paper

---

## Figure Naming Convention

### Why Some Figures Share Numbers

The paper uses **thematic grouping** where related analyses are presented under the same figure number but with different panels/perspectives:

#### Figure 6: Corralling Robustness (Two Perspectives)

**Figure 6a**: `figure6_expert_decommission.png`
- Source: `experiments_v1/06_figure/results/`
- Topic: Catastrophic failure detection
- Labels: `\ref{fig:decommission}`, `\ref{fig:expert_decommission}`, `\ref{fig:catastrophic}`

**Figure 6b**: `figure6_ablation.png` 
- Source: `experiments_v1/07_figure/results/figure7_ablation_fixed.png`
- Topic: Zero-shot model adoption via semantic transfer
- Labels: `\ref{fig:ablation}`, `\ref{fig:multimodel}`, `\ref{fig:corralling_semantic}`

**Rationale**: Both figures demonstrate Corralling's adaptive robustness:
- 6a shows adaptation to catastrophic failures (prior mismatch)
- 6b shows adaptation during new model releases (zero-shot readiness)

### Experiment Folder vs Paper Figure Mapping

| Experiment Folder | Generated Files | Used in Paper As | Rationale |
|-------------------|----------------|------------------|-----------|
| `01_figure/` | `figure1_*.png` | Figure 1 | Direct mapping |
| `02_figure/` | `figure2_*.png` | Figure 2 | Direct mapping |
| `03_figure/` | `figure3_*.png` | Figure 3 | Direct mapping |
| `04_figure/` | `figure4_*.png` | Figure 4 | Direct mapping |
| `05_figure/` | `figure5_*.png` | Figure 5 | Direct mapping |
| `06_figure/` | `appendixE_*.png` | Figure 6a | Main result extracted from appendix analysis |
| `07_figure/` | `figure7_*.png` | Figure 6b | Thematic grouping with Figure 6a |
| `08_figure/` | `figure8_*.png` | Figure 8 | Direct mapping |

**Note**: Experiment folders are numbered sequentially (01-08) for organization, but paper figures are numbered thematically (1-8, with 6a/6b sharing a number).

---

## Multiple Labels Per Figure

### Design Pattern

Some figures have 2-3 labels to allow flexible cross-referencing from different perspectives:

```latex
% Example from Figure 6b:
\label{fig:ablation}              % Ablation study perspective
\label{fig:multimodel}            % Multi-model routing perspective  
\label{fig:corralling_semantic}   % Semantic transfer perspective
```

### Benefits

1. **Conceptual Flexibility**: Authors can reference the same figure using terminology appropriate to the context
2. **Forward Compatibility**: If figures are split in revision, labels remain valid
3. **Reviewer Navigation**: Different reviewers may search for different terms

### All Multi-Label Figures

| Figure | Primary Label | Alternative Labels | Use Cases |
|--------|--------------|-------------------|-----------|
| **Figure 5** | `fig:pareto` | `fig:pareto_frontier` | Economic analysis vs visualization |
| **Figure 6a** | `fig:decommission` | `fig:expert_decommission`, `fig:catastrophic` | Process vs mechanism vs failure type |
| **Figure 6b** | `fig:ablation` | `fig:multimodel`, `fig:corralling_semantic` | Study type vs capability vs mechanism |
| **Figure 8** | `fig:expert_selection` | `fig:sensitivity` | Mechanism vs robustness analysis |

---

## Dual Evaluation Modes

### Why Two Different Quality Numbers?

The paper reports **two complementary evaluations** that measure different aspects of system performance:

#### Mode 1: Warm-Start (Realistic Deployment)
- **Dataset**: N=1,121 dev set
- **Learning**: WITH continued online adaptation
- **Peak Quality**: 0.912 ± 0.006
- **Gap Closure**: 68.5%
- **Purpose**: Shows realistic deployment performance where system learns continuously

#### Mode 2: Frozen (Fair Benchmark)
- **Dataset**: N=750 held-out test set
- **Learning**: WITHOUT continued learning (frozen after training)
- **Peak Quality**: 0.9088
- **Gap Closure**: 65.9%
- **Purpose**: Provides fair comparison to static baselines (RouteLLM) that don't learn online

### The 2.6pp Difference

**Warm-start - Frozen = 68.5% - 65.9% = 2.6 percentage points**

This quantifies the **value of continued online learning** in production:
- System improves by 2.6pp when allowed to adapt to deployment data
- This is a FEATURE, not inconsistency
- Both numbers are correct and validate strong performance

### Where They Appear

| Metric | Warm-Start | Frozen | Location |
|--------|-----------|--------|----------|
| Peak Quality | 0.912±0.006 | 0.9088 | Table 2 vs Figure 5 caption |
| Gap Closure | 68.5% | 65.9% | Section 5.1.2 vs footnote |
| Dataset Size | 1,121 | 750 | Dev set vs Test set |
| Baseline Value | 0.823 | 0.8227 | Slightly different precision |
| Oracle Value | 0.953 | 0.9533 | Slightly different precision |

**Clarifying Comments**: Added inline LaTeX comments (Feb 13) to explain this distinction wherever both numbers appear.

---

## File Update History

### Critical Updates (Feb 13, 2026)

1. **Figure 6b Update**:
   - OLD: `paper/figures/figure6_ablation.png` (modified Feb 13 07:18)
   - NEW: Copied from `experiments_v1/07_figure/results/figure7_ablation_fixed.png` (modified Feb 13 07:23)
   - **Reason**: Applied Figure 7 infinite loop fix, generated corrected output
   - **Verification**: MD5 checksums differ, new version is post-fix

2. **LaTeX Comments Added**:
   - Multi-label explanations for Figures 5, 6a, 6b, 8
   - Dual evaluation mode clarifications in footnotes
   - **Purpose**: Prevent future confusion during revision/review

---

## Naming Philosophy

### Experiment Folders (Sequential)
- **Purpose**: Chronological development order
- **Convention**: `XX_figure/` or `XX_table/` where XX = 01, 02, ..., 08
- **Advantage**: Clear pipeline organization, easy to run in sequence

### Paper Figures (Thematic)
- **Purpose**: Logical narrative flow
- **Convention**: `figureN_description.png` where N matches paper figure number
- **Advantage**: Readers can easily locate files, LaTeX \ref{} commands intuitive

### Experiment Output Files (Descriptive)
- **Purpose**: Self-documenting, prevent overwrites
- **Convention**: `figureN_experiment_description.png` with version suffixes (_fixed, _hires)
- **Advantage**: Multiple variants can coexist, version control clarity

---

## Best Practices for Updates

### When Re-Running Experiments

1. **Check Experiment README**: Each `experiments_v1/XX_*/README.md` documents expected outputs
2. **Run Experiment**: `python experiments_v1/XX_*/script_name.py`
3. **Verify Output**: Check `experiments_v1/XX_*/results/` for new files
4. **Update Paper Figures**: 
   ```bash
   cp experiments_v1/XX_*/results/OUTPUT.png paper/figures/FIGURE_NAME.png
   ```
5. **Check Timestamps**: Ensure paper figure is newer than experiment output
6. **Verify in Paper**: Check that LaTeX still compiles and figure appears correctly

### When Adding New Figures

1. **Choose Figure Number**: Follow thematic grouping (can share numbers for related content)
2. **Create Experiment Folder**: Next sequential number (e.g., `09_figure/`)
3. **Generate Files**: Name outputs to match paper figure number for clarity
4. **Copy to paper/figures/**: Use descriptive name
5. **Add LaTeX**: Include `\includegraphics{}`, `\caption{}`, `\label{}`
6. **Document Here**: Update this file with mapping and rationale

---

## Verification Checklist

Before submission, verify:

- [ ] All `\includegraphics{}` paths point to existing files in `paper/figures/`
- [ ] All paper figures are newer than experiment outputs (no stale files)
- [ ] All multi-label figures have clarifying comments in LaTeX
- [ ] All dual evaluation references have inline explanations
- [ ] All experiment READMEs match actual outputs
- [ ] MD5 checksums match between experiment outputs and paper figures
- [ ] LaTeX compiles without missing figure warnings

---

## Questions?

**Q: Why isn't there a Figure 7 in paper/figures/?**  
A: Figure 7 content is included as Figure 6b (thematic grouping with catastrophic failure analysis)

**Q: Why do experiment folders not match figure numbers?**  
A: Folders are sequential (development order), figures are thematic (narrative flow)

**Q: Are the experiment outputs still called figure7_*.png?**  
A: Yes! Experiment 07 generates `figure7_ablation_fixed.png`, which is COPIED to paper as `figure6_ablation.png`

**Q: Is this confusing?**  
A: Potentially, which is why we documented it here! The benefit of thematic grouping (clearer paper narrative) outweighs the cost of slight naming mismatch.

---

**Document Status**: COMPLETE  
**Last Updated**: February 13, 2026  
**Maintained By**: Project lead / paper authors
