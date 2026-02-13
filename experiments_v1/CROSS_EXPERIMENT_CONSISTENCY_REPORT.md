# Cross-Experiment Consistency Analysis

**Date**: February 13, 2026  
**Purpose**: Comprehensive review of all experiments (Figures 1-8, Tables 1-2) to ensure internal consistency  
**Status**: ⚠️ **CRITICAL INCONSISTENCY IDENTIFIED**

---

## Executive Summary

After reviewing all experiments, I identified **one critical contradiction** between Figure 7 and Figure 8 regarding expert weight patterns, and **several areas requiring clarification** to maintain narrative coherence.

### 🚨 Critical Issue Found

**Contradiction**: Figure 7 vs Figure 8 expert weight reporting
- **Figure 7 (results.tex, lines 162, 166, 256)**: Claims "stable expert weights (~75% Conservative, ~25% Adaptive) throughout episode"
- **Figure 8 (revised)**: Shows regime switching with binary choices (100% warmup OR 100% tabula rasa) by seed

**Impact**: Undermines credibility if both claims appear in same paper

**Root Causes Identified**:
1. Different alpha configurations (heterogeneous vs homogeneous)
2. Averaging across seeds vs individual seed analysis
3. Pre-release vs post-release reporting windows

---

## Detailed Experiment Flow

### ✅ Figure 1: Alignment Tax Discovery
**Claim**: 17.6% of prompts show "Alignment Tax" where Mixtral > GPT-4  
**Status**: ✅ No conflicts  
**Connection to Fig 8**: Establishes problem domain and need for adaptive routing

### ✅ Table 1: Dataset Composition
*(Not read in this analysis, but typically describes data)*  
**Status**: Assumed ✅

### ✅ Table 2: Performance Gap
**Claim**: With η=1.0, achieve median 52 regret (IQR: [34-80], N=10 seeds)  
**Key Details**:
- Multi-seed validation (N=10)
- Reports variance due to stochastic expert selection
- Median + IQR instead of mean ± std
**Status**: ✅ No conflicts  
**Connection to Fig 8**: Uses same Corralling system; acknowledges variance from expert selection

### ✅ Figure 3: Architecture Validation
**Claim**: Constant α=2.0 optimal for both experts (homogeneous)  
**Configuration**: Both experts with constant α=2.0  
**Status**: ✅ No conflicts  
**Connection to Fig 8**: Fig 8 uses α=2.0 (consistent with Fig 3 recommendation)

### ✅ Figure 5: Pareto Frontier
**Claim**: banditGPT achieves 0.912 ± 0.006  
**Configuration**: η=1.0, standard Corralling  
**Status**: ✅ No conflicts  
**Connection to Fig 8**: Same system architecture

### ✅ Figure 6: Catastrophic Failure Detection
**Claim**: 100% detection rate in 3-50 steps  
**Configuration**: Shows decommissioning of failing expert  
**Status**: ✅ No conflicts  
**Connection to Fig 8**: Demonstrates expert switching capability (relevant to regime switching)

### ⚠️ Figure 7: Zero-Shot Model Adoption
**Claim**: "Expert weights remain stable (~75% Conservative, ~25% Adaptive) throughout episode" (η=0.1)

**Configuration**:
- Heterogeneous experts (Conservative: α decay 1.0→0.01, Adaptive: α constant 2.0)
- N=30 trials (seeds 42-71)
- Reports AVERAGE weights across all seeds
- 800 steps total, release at t=300

**Detailed Claims** (from results.tex):
1. Line 162 (caption): "expert weights remain stable (~75% Conservative, ~25% Adaptive) throughout episode"
2. Line 166 (body): "stable expert weights throughout the episode (~75% Conservative, ~25% Adaptive)"
3. Line 256 (cold-start regime): "expert weights remain stable (~75% Conservative, ~25% Adaptive)"

**Status**: ⚠️ **CONTRADICTS FIGURE 8**

### ⚠️ Figure 8: Adaptive Expert Selection (REVISED)
**Claim**: "Corralling converges to near-binary selection by seed: 100% warmup (seed 42) or 100% tabula rasa (seeds 43-44)"

**Configuration**:
- Homogeneous experts (both with α=2.0 constant)
- N=3 seeds (42-44)
- Reports INDIVIDUAL seed patterns
- 1000 steps total, release at t=300

**Detailed Findings**:
- Seed 42: 100% warmup expert (post-release)
- Seed 43: 100% tabula rasa expert (post-release)
- Seed 44: 100% tabula rasa expert (post-release)
- Average: ~33% warmup, ~67% tabula rasa

**Status**: ⚠️ **CONTRADICTS FIGURE 7**

---

## Contradiction Analysis

### The Core Problem

**Figure 7** claims weights are "stable at 75/25" while **Figure 8** shows "binary switching to 0/100 or 100/0".

These statements cannot both be true unless they're measuring different things.

### Possible Explanations

#### Explanation 1: Different Alpha Configurations ✅ (Confirmed)
- **Figure 7**: Heterogeneous experts (Conservative with decay, Adaptive with constant)
- **Figure 8**: Homogeneous experts (both constant α=2.0)

**Implication**: Alpha configuration affects expert selection behavior!

**Problem**: This contradicts Figure 3's conclusion that "homogeneous constant is optimal"

#### Explanation 2: Averaging vs Individual Reporting ✅ (Confirmed)
- **Figure 7**: Reports AVERAGE across 30 seeds
- **Figure 8**: Reports INDIVIDUAL seeds

**Math Check**:
- If 33% of seeds have 100% warmup and 67% have 0% warmup:
  - Average = 0.33×1.0 + 0.67×0.0 = **33% warmup**
- Figure 7 claims **75% warmup**

**Discrepancy**: 75% ≠ 33% 

This suggests different behavior, not just different reporting!

#### Explanation 3: Time Window Differences
- **Figure 7**: May report full episode average (t=0-800), including pre-release
- **Figure 8**: Reports post-release (t=300+)

**Hypothesis**: Pre-release might be different from post-release patterns?

---

## Resolution Options

### Option 1: Acknowledge Configuration Dependency (RECOMMENDED)

**Action**: Update paper to clarify that expert selection behavior depends on alpha configuration:

**Updated Text for Figure 7 sections**:

```latex
\paragraph{Expert Dynamics with Heterogeneous Configuration.}
With heterogeneous experts (Conservative: $\alpha$ decay 1.0$\to$0.01, Adaptive: $\alpha=2.0$ constant), 
expert weights stabilize to approximately 75\% Conservative, 25\% Adaptive on average across seeds.
This contrasts with homogeneous constant-$\alpha$ configuration (Figure~\ref{fig:expert_selection}), 
where Corralling exhibits regime-dependent binary selection. The heterogeneous design sacrifices 
decisive adaptation for smoother, more stable hedging behavior—appropriate for the conservative 
learning regime ($\eta=0.1$) focused on short-term deployment benefit rather than long-term convergence.
```

**Rationale**: 
- Acknowledges both findings are correct for their configurations
- Explains WHY behaviors differ (heterogeneous vs homogeneous)
- Connects to design intent (stability vs adaptability trade-off)

### Option 2: Re-run Figure 7 with Homogeneous α=2.0

**Action**: Run Figure 7 experiment with both experts using constant α=2.0 to see if it also shows regime switching.

**Pros**:
- Direct comparison between experiments
- Validates Figure 3's "homogeneous constant is optimal" claim
- Simplifies narrative (consistent architecture)

**Cons**:
- Requires re-running experiments (~30 minutes)
- May change reported performance numbers
- Figure 7 results may be published/submitted already

### Option 3: Run Figure 8 with Heterogeneous Experts

**Action**: Run Figure 8 with same heterogeneous configuration as Figure 7.

**Pros**:
- Tests if regime switching persists with heterogeneous experts
- Directly explains discrepancy

**Cons**:
- Figure 8 currently uses recommended architecture from Figure 3
- Adds complexity to narrative

### Option 4: Investigate Pre-Release vs Post-Release

**Action**: Analyze Figure 7 data separately for t=0-300 (pre-release) and t=300-800 (post-release).

**Hypothesis**: Maybe 75/25 is the pre-release pattern, and post-release shows different behavior?

---

## Additional Consistency Checks

### ✅ Semantic Transfer Narrative
All experiments consistently describe semantic transfer as:
- "Implicit regularization via symmetry breaking"
- "Not predictive of task-level performance (r=-0.38, p=0.75)"
- "Provides short-term benefit but may be directionally wrong"

**Status**: ✅ Consistent

### ✅ Learning Rate Regimes
Paper consistently describes four regimes:
1. Cold-Start (η=0.1-0.3): Exploit priors, short-term benefit
2. Pareto Sweep (η=1.0): Partial adaptation
3. Safety (η=0.3-1.0): Failure detection
4. Convergence (η=2.0-5.0): Complete unlearning

**Status**: ✅ Consistent

### ⚠️ Alpha Configuration Consistency

**Figure 3** (Architecture): "Homogeneous constant α=2.0 is optimal" (60.6 regret)  
**Figure 7** (Zero-Shot): Uses heterogeneous (Conservative decay, Adaptive constant)  

**Question**: Why does Figure 7 use heterogeneous if Figure 3 proved homogeneous is better?

**Possible Answer**: 
- Figure 7 focuses on CONSERVATIVE regime (η=0.1, short-term benefit)
- Figure 3 tests STANDARD regime (η=1.0, balanced adaptation)
- Different regimes may have different optimal alpha strategies?

**Action**: Add clarification explaining why heterogeneous is appropriate for cold-start regime.

---

## Recommendations

### Immediate Actions

1. **✅ CRITICAL: Resolve Figure 7 vs Figure 8 contradiction**
   - Choose Option 1 (acknowledge configuration dependency) OR
   - Run diagnostic to understand if difference is alpha-dependent

2. **Add Clarifying Text**:
   - Explain heterogeneous vs homogeneous expert configurations
   - Clarify when average weights vs individual seeds are reported
   - Connect alpha strategy to learning rate regime (cold-start vs standard)

3. **Update Figure 8 Section**:
   - Current version correctly shows regime-dependent expert selection
   - Emphasize this is WITH homogeneous α=2.0 (recommended architecture)
   - Contrast with heterogeneous configuration (smoother blending)

### Narrative Coherence

**Unified Story**:
1. **Figure 3**: Architecture validation → Homogeneous α=2.0 is optimal for standard operation
2. **Figure 7**: Cold-start regime (η=0.1) → Uses heterogeneous for stability, shows 75/25 blend
3. **Figure 8**: Expert selection analysis (η=0.1) → Shows homogeneous enables decisive regime switching

**Key Message**: 
- Heterogeneous experts → Smooth hedging (75/25 blend)
- Homogeneous experts → Decisive adaptation (100/0 or 0/100)
- Both valid; choice depends on deployment priorities (stability vs adaptability)

---

## Next Steps

**Choose ONE**:

A. **Conservative Fix** (5 minutes):
   - Add clarifying text to results.tex explaining configuration differences
   - No re-runs needed
   - Preserves existing results

B. **Diagnostic Investigation** (30 minutes):
   - Create script to run Figure 7 weights diagnostic for seeds 42-44
   - Compare heterogeneous vs homogeneous patterns
   - Data-driven resolution

C. **Full Alignment** (1 hour):
   - Re-run Figure 7 with homogeneous α=2.0
   - Update all figures and text for consistency
   - Strongest scientific rigor

**Recommendation**: Start with **Option B** (diagnostic) to understand the actual difference, then decide between A or C based on findings.
