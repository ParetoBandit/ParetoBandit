# Figure Organization After Pivot

## New Figure Structure

### Figure 1: Deployment Workflow Comparison (NEW - Page 1 Teaser)
**Location**: `figures/figure1_deployment_comparison/`
**File**: `figure1_deployment_comparison.pdf`
**Purpose**: Aspirational teaser showing the "pain" (traditional routers) vs. "relief" (BanditGPT)
**Placement**: Introduction section, Page 1
**Caption**: 
> **Figure 1: Deployment Workflow Comparison.** Traditional LLM routers (e.g., FrugalGPT, RouteLLM) require days-to-weeks of offline calibration, including dataset collection (500-5000 examples), benchmarking, and retraining. **BanditGPT eliminates this bottleneck** through pure online learning, enabling immediate deployment without training data. When model capabilities change, traditional routers require manual recalibration, while BanditGPT adapts automatically through continuous feedback.

**Message**: "Zero calibration, immediate deployment, automatic adaptation"

---

### Figure 2: System Architecture (KEEP)
**Location**: `figures/architecture_diagram.pdf`
**Purpose**: Show the technical system design
**Placement**: Method Section 2.2
**Update Needed**: Remove mentions of "metadata-guided initialization providing quality heuristics"

---

### Figure 3: Negative Transfer (MOVED from old Figure 1)
**Location**: `figures/figure1_negative_transfer/figure1_negative_transfer_full.pdf`
**Purpose**: DEFENSE - Scientific evidence that offline calibration fails
**Placement**: Evaluation Section 4.2 (RQ1)
**Caption**: 
> **Figure 3: Offline Calibration Exhibits Consistent Negative Transfer (Out-of-Sample Evaluation).** **(A)** Mean cumulative regret curves with 95% confidence intervals (shaded regions) across 5 folds, evaluated on held-out prompts. Cold start (green, solid) consistently outperforms both warm-start strategies. **(B)** Per-fold performance changes relative to cold start. Each dot represents one fold of the cross-validation; all points falling above y=0 indicate performance degradation. All 10 data points (5 folds × 2 strategies) show degradation, demonstrating 100% directional consistency.

**Message**: "We tried priors. Here's proof they fail. We're data-driven, not lazy."

---

### Figure 4: Belief Recovery / Plasticity (KEEP - renumber from Figure 2)
**Location**: `figures/figure2_belief_recovery.png`
**Purpose**: Show adaptation to concept drift
**Placement**: Evaluation Section 4.3 (Plasticity)
**Note**: Keep as-is, just renumber

---

### Figure 5: Pareto Frontier (KEEP - renumber from Figure 4)
**Location**: `figures/figure4_pareto_frontier.pdf`
**Purpose**: Show cost-quality trade-offs
**Placement**: Evaluation Section 4.4 (Cost-Quality Efficiency)
**Note**: Keep as-is, just renumber

---

### Removed Figures:
- ❌ `distillation_diagram.pdf` - Removed (expert distillation section deleted)
- ❌ `figure3_specialist_landscape.pdf` - Removed (part of distillation narrative)

---

## Update Checklist

### main_CONCISE.tex
- [ ] Add Figure 1 (deployment comparison) in Introduction
- [ ] Renumber Figure 1 (negative transfer) → Figure 3
- [ ] Renumber Figure 2 (belief recovery) → Figure 4
- [ ] Renumber Figure 4 (pareto) → Figure 5
- [ ] Remove Figure references to distillation and specialist landscape

### evaluation.tex
- [ ] Update Figure 1 reference → Figure 3 in RQ1 section
- [ ] Update Figure 2 reference → Figure 4 in Plasticity section
- [ ] Update Figure 4 reference → Figure 5 in Cost-Quality section

### method.tex
- [ ] Remove Figure distillation reference
- [ ] Remove Figure specialist_landscape reference
- [ ] Update architecture diagram caption (remove metadata-guided mentions)

---

## Key Narrative Flow

**Page 1 (Introduction)**: Figure 1 sells the vision (fast deployment)
**Section 2 (Method)**: Architecture diagram shows how it works
**Section 4.2 (RQ1)**: Figure 3 defends why we don't use priors (scientific rigor)
**Section 4.3 (Plasticity)**: Figure 4 shows adaptation
**Section 4.4 (Efficiency)**: Figure 5 shows cost-quality results

---

## The Strategic Framing

✅ **Figure 1 = OFFENSE** (selling the benefit)
✅ **Figure 3 = DEFENSE** (proving we're not lazy)
✅ **Figures 4-5 = VALIDATION** (empirical results)

This structure ensures reviewers see:
1. **Why they should care** (Figure 1 - immediate deployment)
2. **Why our design is correct** (Figure 3 - offline calibration fails)
3. **That it actually works** (Figures 4-5 - empirical validation)

