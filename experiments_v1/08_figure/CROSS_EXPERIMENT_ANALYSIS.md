# Cross-Experiment Analysis: Consistency Check

**Date**: February 13, 2026  
**Purpose**: Identify contradictions between Figure 8 (revised) and previous experiments

---

## Experiment Narrative Flow

### Figure 1: Problem Discovery
- **Finding**: Alignment Tax exists (17.6% of prompts where Mixtral > GPT-4)
- **Implication**: Need adaptive routing, not static allocation
- **Connection to Fig 8**: ✅ Establishes need for exploration/adaptation

### Table 2: Domain Mismatch Performance
- **Finding**: With η=1.0, achieve median 52 regret (1.3× vs optimal)
- **Expert weights**: Not explicitly reported in table
- **Connection to Fig 8**: ✅ Uses same Corralling system

### Figure 3: Architecture Validation
- **Finding**: Constant α=2.0 is optimal for both experts
- **Expert weights**: Not the focus (architecture diagram)
- **Connection to Fig 8**: ✅ Architectural consistency

### Figure 5: Pareto Frontier
- **Finding**: banditGPT achieves 0.912 ± 0.006
- **Configuration**: Uses Corralling with η=1.0
- **Connection to Fig 8**: ✅ Same system being tested

### Figure 6: Catastrophic Failure Detection
- **Finding**: 100% detection rate in 3-50 steps
- **Expert weights**: Shows decommissioning of warmup expert
- **Connection to Fig 8**: ✅ Shows expert switching capability

### Figure 7 (07_figure): Zero-Shot Model Adoption
- **Finding**: Semantic transfer provides 3.2% benefit
- **Configuration**: η=0.1 (CONSERVATIVE)
- **Expert weights**: "~75% Conservative, ~25% Adaptive"
- **Connection to Fig 8**: ⚠️ **POTENTIAL CONTRADICTION**

### Figure 8 (08_figure): Expert Selection
- **Finding**: Expert selection varies by seed (33% warmup, 67% tabula rasa)
- **Configuration**: η=0.1 (CONSERVATIVE)
- **Expert weights**: 100% warmup OR 100% tabula rasa (binary)
- **Connection to previous**: ⚠️ **CONTRADICTION WITH FIGURE 7**

---

## 🚨 CONTRADICTION IDENTIFIED

### Figure 7 Claims (results.tex, line 166):
> "stable expert weights throughout the episode (~75% Conservative, ~25% Adaptive)"

### Figure 8 Shows (our diagnostic):
```
Seed 42: Warmup 100%, Tabula Rasa 0%
Seed 43: Warmup 0%, Tabula Rasa 100%
Seed 44: Warmup 0%, Tabula Rasa 100%
```

**These cannot both be true!**

---

## Root Cause Investigation

### Hypothesis 1: Different Experimental Conditions

Let me check what's different between Figure 7 and Figure 8:

| Aspect | Figure 7 | Figure 8 |
|--------|----------|----------|
| Learning rate η | 0.1 | 0.1 |
| Models | Mixtral, GPT-4, GPT-5.1 | Mixtral, GPT-4, GPT-5.1 |
| Release step | 300 | 300 |
| Total steps | 800 | 1000 |
| Seeds | 30 trials (42-71) | 3 trials (42-44) |
| Focus | Reward performance | Expert weights |

**Potential difference**: Figure 7 uses seeds 42-71 (averaged), Figure 8 uses seeds 42-44 (individual)

### Hypothesis 2: Reporting vs Reality

**Figure 7 claim** (~75% warmup) might be:
1. **Averaged across 30 seeds** where some have 100% warmup, others 0%
2. **Pre-release average** (before t=300) not post-release
3. **Misreported** from a single seed

**Figure 8 reveals** the actual pattern:
- Individual seeds show binary choices (not 75/25 blend)
- Post-release, Corralling commits decisively

---

## Resolution Strategy

### Option 1: Check Figure 7 Data

Run Figure 7 experiment with weight tracking to see actual patterns.

### Option 2: Correct Figure 7 Text

If Figure 7 also shows regime switching, update the text to match:

**OLD** (Line 166):
> "stable expert weights throughout the episode (~75% Conservative, ~25% Adaptive)"

**CORRECTED**:
> "expert weights vary by seed: some seeds maintain warmup expert dominance (75%+), others switch to tabula rasa dominance (25%+)"

### Option 3: Acknowledge Regime-Dependence Throughout

Add consistent language across all experiments mentioning Corralling:
- "Corralling adaptively chooses experts based on data match"
- "Expert selection is regime-dependent (data ordering affects choice)"
- "Average weights reflect mixture of warmup-dominant and tabula-dominant seeds"

---

## Verification Needed

### Let's Check Figure 7 Actual Weights

I'll run diagnostic on Figure 7 experiment to see if it also shows regime switching.
