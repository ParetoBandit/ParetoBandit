# Figure 8 Consistency Analysis: Executive Summary

**Date**: February 13, 2026  
**Reviewer**: Cross-Experiment Consistency Check  
**Status**: ⚠️ One critical contradiction found, resolution path identified

---

## TL;DR

**Question**: How does Figure 8 (revised) connect with previous experiments?

**Answer**: 
- ✅ **Generally consistent** with experiments 1-6
- ⚠️ **ONE CONTRADICTION** with Figure 7 regarding expert weight patterns
- ✅ **Resolution path identified** (alpha configuration difference)

---

## The Contradiction

### Figure 7 (Zero-Shot Model Adoption) Claims:
> "Expert weights remain stable (~75% Conservative, ~25% Adaptive) throughout episode"
  
(appears in 3 locations: lines 162, 166, 256 of results.tex)

### Figure 8 (Adaptive Expert Selection) Shows:
> "Corralling converges to binary selection: 100% warmup OR 100% tabula rasa by seed"

**These cannot both be true!**

---

## Root Cause: Different Alpha Configurations

### Figure 7 Configuration
```python
# HETEROGENEOUS experts
Expert 1 (Conservative): alpha decay 1.0 → 0.01
Expert 2 (Adaptive):     alpha constant 2.0
```

**Behavior**: Smooth hedging, stable blend (75/25)

### Figure 8 Configuration  
```python
# HOMOGENEOUS experts (recommended by Figure 3)
Expert 1 (Warmup):      alpha constant 2.0
Expert 2 (Tabula Rasa): alpha constant 2.0
```

**Behavior**: Decisive switching, binary regime selection (100/0 or 0/100)

---

## Why This Matters

### For Paper Credibility
If both claims appear without explanation, reviewers will notice the contradiction and question data quality/analysis rigor.

### For Scientific Understanding
This reveals an important finding: **Alpha configuration determines Corralling's decision style**
- Heterogeneous → Smooth, stable hedging
- Homogeneous → Decisive, adaptive switching

This is actually **interesting science** worth highlighting!

---

## Experiments 1-6: No Contradictions Found

### ✅ Figure 1: Alignment Tax
- Establishes problem (17.6% misalignment)
- Connects naturally to Figure 8's need for adaptive exploration

### ✅ Table 2: Performance Gap
- Reports variance from stochastic expert selection
- Consistent with Figure 8's regime-dependent behavior
- Both acknowledge seed-to-seed variation

### ✅ Figure 3: Architecture Validation
- Recommends homogeneous α=2.0 as optimal
- **Figure 8 follows this recommendation** ✅
- **Figure 7 uses different config for different regime** (needs clarification)

### ✅ Figure 5: Pareto Frontier
- Uses standard Corralling (η=1.0)
- No conflicts with Figure 8

### ✅ Figure 6: Catastrophic Failure Detection
- Demonstrates expert switching capability
- Actually **supports** Figure 8's regime switching narrative

---

## Secondary Issue: Figure 3 vs Figure 7 Config Choice

**Figure 3** proves: "Homogeneous constant α=2.0 is optimal" (60.6 regret)

**Figure 7** uses: Heterogeneous (Conservative decay, Adaptive constant)

**Question**: Why does Figure 7 deviate from Figure 3's recommendation?

**Possible Answer**: Different learning rate regimes may warrant different alpha strategies:
- **Standard regime** (η=1.0, Figure 3): Homogeneous for decisive adaptation
- **Cold-start regime** (η=0.1, Figure 7): Heterogeneous for stable short-term exploitation

**Action**: Add clarification explaining this design choice.

---

## Resolution Options

### Option A: Add Clarifying Text (5 minutes) ⭐ RECOMMENDED

**Update results.tex to acknowledge both patterns:**

```latex
\paragraph{Configuration-Dependent Expert Dynamics.}
Expert weight patterns depend critically on alpha configuration. With heterogeneous 
experts (Figure~\ref{fig:ablation}, Conservative: $\alpha$ decay 1.0$\to$0.01, 
Adaptive: $\alpha=2.0$ constant), weights stabilize at approximately 75\% Conservative, 
25\% Adaptive—appropriate for conservative learning ($\eta=0.1$) prioritizing 
short-term exploitation. With homogeneous constant-$\alpha$ configuration 
(Figure~\ref{fig:expert_selection}, both experts $\alpha=2.0$), Corralling exhibits 
regime-dependent binary selection (100\% warmup or 100\% tabula rasa by seed), 
enabling decisive adaptation. Configuration choice reflects deployment priorities: 
smooth hedging (heterogeneous) versus adaptive switching (homogeneous).
```

**Pros**: 
- No re-runs needed
- Turns contradiction into interesting scientific finding
- 5 minutes to implement

**Cons**: 
- Adds complexity to narrative

### Option B: Run Diagnostic First (30 minutes)

**Verify the 75/25 claim empirically:**

```bash
python experiments_v1/diagnose_figure7_weights.py
```

This will:
1. Run Figure 7 config for seeds 42-44
2. Report actual weights (pre/post release, by seed)
3. Confirm if 75/25 is accurate for heterogeneous config

**Pros**: 
- Data-driven resolution
- Confirms hypothesis before updating paper

**Cons**: 
- Requires 30 minutes runtime

### Option C: Align All Experiments to Homogeneous (1 hour)

**Re-run Figure 7 with homogeneous α=2.0 (matching Figure 3 recommendation).**

**Pros**: 
- Maximum consistency
- Simplifies narrative
- Validates Figure 3's optimality claim across all experiments

**Cons**: 
- Requires re-running experiments
- May change reported numbers
- If already submitted, creates version control issues

---

## Recommended Action Plan

### Phase 1: Diagnostic (30 min)
```bash
python experiments_v1/diagnose_figure7_weights.py
```

**This will answer**:
1. Is 75/25 accurate for heterogeneous config?
2. Do heterogeneous experts really show stable blending?
3. How different is it from homogeneous binary switching?

### Phase 2: Resolution (5 min)
Based on diagnostic results:

**If 75/25 is confirmed** → Add Option A clarifying text

**If 75/25 is wrong** → Update Figure 7 text to match actual measured weights

### Phase 3: Enhancement (Optional, 10 min)
Add a table/figure showing alpha configuration effects:

| Configuration | Behavior | Use Case |
|--------------|----------|----------|
| Heterogeneous | Smooth hedging (75/25) | Cold-start regime, stability priority |
| Homogeneous | Binary switching (100/0) | Standard regime, adaptation priority |

---

## Connection to Paper Narrative

### Current Figure 8 Story (CORRECT)
"Corralling adaptively chooses between semantic transfer and cold start based on data match. This choice is regime-dependent (varies by seed), demonstrating meta-learning robustness."

### Updated Figure 7 Story (NEEDS CLARIFICATION)
"With heterogeneous experts optimized for conservative learning (η=0.1), Corralling maintains stable hedging (~75/25) to exploit short-term benefits without aggressive adaptation. This contrasts with homogeneous configuration (Figure 8), which enables decisive regime switching for faster convergence."

### Unified Narrative
**The system adapts its adaptation strategy based on configuration:**
- Heterogeneous (Figure 7) → Stable exploitation, short-term focus
- Homogeneous (Figure 8) → Decisive adaptation, long-term focus

This is actually a **feature**, not a bug! It shows the system can be tuned for different deployment objectives.

---

## Bottom Line

### For the User
1. Run the diagnostic script (30 min) to confirm hypothesis
2. Add 2-3 sentences of clarifying text (5 min)
3. No major paper restructuring needed

### For Reviewers
The paper will demonstrate:
- Rigorous experimental methodology (caught and resolved contradiction)
- Deep understanding of system behavior (alpha config effects)
- Practical deployment guidance (when to use each config)

---

## Files Created for This Analysis

1. `experiments_v1/CROSS_EXPERIMENT_CONSISTENCY_REPORT.md` - Full detailed analysis
2. `experiments_v1/08_figure/CROSS_EXPERIMENT_ANALYSIS.md` - Initial findings
3. `experiments_v1/diagnose_figure7_weights.py` - Diagnostic script
4. `experiments_v1/FIGURE_8_CONSISTENCY_SUMMARY.md` - This document

**Next**: Run diagnostic or apply fix?
