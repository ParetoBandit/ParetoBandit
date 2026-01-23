# Integration Guide: Figure 4 in Context

## The Four-Figure Story

Your paper tells a coherent story through four experiments. Each figure builds on the previous one, creating a compelling narrative arc.

### Figure 1: PCA Reward Gap
**Question:** Is there semantic structure in prompts that predicts model performance?

**Answer:** Yes! PCA analysis reveals that prompts cluster by semantic similarity, and these clusters correlate with reward gaps between models.

**Key Insight:** Linguistic structure is predictive of routing decisions.

**Sets up:** If semantic structure exists, can we leverage it?

---

### Figure 2: Convergence Analysis
**Question:** When does the policy converge—during calibration or holdout?

**Answer:** Convergence happens during calibration (1,121 samples), not holdout. The policy pivots from 20% to 85% strong model usage.

**Key Insight:** Calibration is where adaptation happens; holdout just evaluates the converged policy.

**Sets up:** If calibration is so powerful, why do we need warmup?

---

### Figure 3: Optimal Gamma
**Question:** How do we balance warmup priors with calibration data?

**Answer:** Gamma scaling (γ=0.002) allows 99.7% policy pivot while retaining semantic structure from warmup.

**Key Insight:** We can dramatically downweight priors and still benefit from them.

**Sets up:** But if we downweight so much, what value does warmup actually provide?

---

### Figure 4: Cold-Start Ablation (THIS EXPERIMENT)
**Question:** If calibration can pivot 99.7% of the policy, do we even need warmup?

**Answer:** YES! Warmup reduces Day 1 regret by 47.4% and accelerates convergence 3x.

**Key Insight:** Warmup provides a linguistic foundation that prevents catastrophic early errors. The value is in the learning trajectory, not just the final policy.

**Concludes:** The two-phase approach (warmup → calibration) is justified and necessary for practical deployment.

---

## How They Fit Together

### The Logical Flow

```
Figure 1: Semantic structure exists
    ↓
Figure 2: Calibration adapts the policy
    ↓
Figure 3: We can downweight priors aggressively
    ↓
Figure 4: But warmup still provides critical value
    ↓
Conclusion: Two-phase approach is optimal
```

### The Narrative Arc

1. **Discovery** (Figure 1): We discover semantic structure
2. **Adaptation** (Figure 2): We show how to adapt to new domains
3. **Optimization** (Figure 3): We find the optimal balance
4. **Validation** (Figure 4): We prove warmup is necessary

### The Reviewer's Journey

**Initial skepticism:** "Why not just train on domain data?"

**Figure 1:** "Oh, there's semantic structure that matters."

**Figure 2:** "Interesting, calibration does adapt the policy."

**Figure 3:** "Wait, you downweight priors by 99.7%? Then why bother?"

**Figure 4:** "Ah! The warmup provides a foundation that prevents early disasters. Now I get it."

**Conclusion:** "This is a well-designed system with clear practical value."

## Cross-References in the Paper

### In Figure 4 Description

"As shown in Figure 3, our optimal gamma (γ=0.002) allows the policy to pivot 99.7% away from the warmup prior. This raises a natural question: if calibration is so powerful, do we even need warmup?"

### In Figure 3 Description

"While gamma scaling allows dramatic policy adaptation (Figure 2), the cold-start ablation (Figure 4) reveals that even heavily downweighted priors provide substantial value during early calibration."

### In Figure 2 Description

"This convergence during calibration, combined with the semantic structure identified in Figure 1, enables our two-phase approach. However, as Figure 4 demonstrates, warmup remains critical for Day 1 performance."

### In Figure 1 Description

"This semantic structure, captured by PCA, provides the foundation for our warmup priors. Figures 2-4 show how these priors enable efficient domain adaptation while maintaining high Day 1 quality."

## Experimental Consistency

### Shared Components

All four experiments use:
- ✅ Same embedding model (all-MiniLM-L6-v2)
- ✅ Same PCA model (23 components)
- ✅ Same warmup priors (80k RouteLLM samples)
- ✅ Same calibration data (1,121 dev samples)
- ✅ Same models (Mixtral-8x7B, GPT-4o)

This consistency ensures:
- Fair comparisons across experiments
- Reproducible results
- Clear interpretation

### Progressive Refinement

Each experiment builds on the previous:

**Figure 1 → Figure 2:**
- Figure 1 shows semantic structure exists
- Figure 2 uses that structure (via PCA) for routing

**Figure 2 → Figure 3:**
- Figure 2 shows policy pivots dramatically
- Figure 3 quantifies how much we can downweight priors

**Figure 3 → Figure 4:**
- Figure 3 finds optimal gamma (0.002)
- Figure 4 uses that gamma to prove warmup value

## Paper Structure Suggestions

### Results Section Organization

```latex
\section{Experimental Results}

\subsection{Semantic Structure in Prompts}
% Figure 1: PCA Reward Gap
% Shows that semantic structure exists and is predictive

\subsection{Policy Adaptation During Calibration}
% Figure 2: Convergence Analysis
% Shows when and how the policy adapts

\subsection{Balancing Priors and Calibration}
% Figure 3: Optimal Gamma
% Shows how to find the right balance

\subsection{The Value of Warmup: A Cold-Start Ablation}
% Figure 4: Cold-Start Ablation
% Shows why warmup is necessary despite aggressive downweighting
```

### Alternative: Narrative-Driven Structure

```latex
\section{Experimental Results}

\subsection{Understanding Semantic Structure}
% Figure 1: Establishes foundation

\subsection{Domain Adaptation in Practice}
% Figures 2 & 3: Show how adaptation works
% \subsubsection{Convergence Dynamics}
% \subsubsection{Optimal Prior Weighting}

\subsection{Validating the Two-Phase Approach}
% Figure 4: Proves the approach is necessary
```

## Key Talking Points

### For the Introduction

"Our approach combines semantic priors from large-scale warmup data with domain-specific calibration. We validate this design through four experiments: (1) semantic structure analysis, (2) convergence dynamics, (3) prior weighting optimization, and (4) cold-start ablation."

### For the Methods

"We evaluate our approach through a series of experiments designed to answer key questions: Does semantic structure exist? (§4.1) When does adaptation occur? (§4.2) How should we balance priors and data? (§4.3) Is warmup necessary? (§4.4)"

### For the Results

"Figure 4 addresses a critical question: given that calibration can pivot 99.7% of the policy (Figure 3), is warmup even necessary? Our ablation study reveals that warmup reduces Day 1 regret by 47.4%, proving that semantic priors provide value beyond initialization."

### For the Discussion

"The progression from Figure 1 to Figure 4 tells a complete story: semantic structure exists (1), enables adaptation (2), can be balanced with domain data (3), and provides critical value for cold-start performance (4). Together, these results validate our two-phase approach."

## Supplementary Material Suggestions

### Supplementary Figure S1: Full Experimental Pipeline

Show a diagram connecting all four experiments:
- Input: RouteLLM data (80k samples)
- Process: PCA → Warmup → Gamma scaling → Calibration
- Outputs: Figures 1, 2, 3, 4

### Supplementary Table S1: Experimental Parameters

| Parameter | Figure 1 | Figure 2 | Figure 3 | Figure 4 |
|-----------|----------|----------|----------|----------|
| Embedding | MiniLM | MiniLM | MiniLM | MiniLM |
| PCA dims | 23 | 23 | 23 | 23 |
| Warmup samples | 80k | 80k | 80k | 80k |
| Calibration samples | N/A | 1,121 | 1,121 | 1,121 |
| Gamma | N/A | 0.002 | varied | 0.002 |
| Alpha | N/A | 1.0 | 1.0 | 1.0 |

### Supplementary Section S1: Reproducibility

Provide:
- Links to code for all four experiments
- Data availability statement
- Compute requirements
- Expected runtime for each experiment

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Treating Figures as Independent

**Wrong:** "We ran four experiments on our system."

**Right:** "We designed four experiments to progressively validate our approach, from establishing semantic structure (Figure 1) to proving the necessity of warmup (Figure 4)."

### ❌ Pitfall 2: Ignoring the Tension

**Wrong:** "Figure 3 shows we can downweight priors. Figure 4 shows warmup helps."

**Right:** "Figure 3 reveals a puzzle: if we downweight priors by 99.7%, why do they help? Figure 4 resolves this by showing that even heavily downweighted priors prevent catastrophic early errors."

### ❌ Pitfall 3: Weak Transitions

**Wrong:** "Next, we evaluate cold-start performance."

**Right:** "Given that gamma scaling allows 99.7% policy pivot (Figure 3), a natural question arises: do we even need warmup? Figure 4 addresses this through a controlled ablation study."

### ❌ Pitfall 4: Missing the Throughline

**Wrong:** Each figure is described independently without connections.

**Right:** Each figure references the others, building a cohesive narrative.

## Reviewer Response Templates

### Response to: "Why four experiments? Seems like overkill."

"Each experiment addresses a distinct question in our validation strategy:
- Figure 1 establishes that semantic structure exists
- Figure 2 shows when adaptation occurs
- Figure 3 optimizes the balance between priors and data
- Figure 4 validates the necessity of warmup

Together, they provide comprehensive evidence for our two-phase approach. Removing any one would leave a critical question unanswered."

### Response to: "Figure 4 contradicts Figure 3"

"Figure 3 and Figure 4 are complementary, not contradictory. Figure 3 shows that we CAN downweight priors dramatically (allowing 99.7% policy pivot). Figure 4 shows that we SHOULD still use priors (reducing Day 1 regret by 47.4%). The key insight is that even heavily downweighted priors provide critical semantic grounding."

### Response to: "The experiments use different metrics"

"Each experiment uses metrics appropriate to its research question:
- Figure 1: Reward gap (semantic structure)
- Figure 2: Policy evolution (convergence dynamics)
- Figure 3: Final policy vs. convergence rate (optimization)
- Figure 4: Cumulative regret (cold-start performance)

This diversity reflects the multifaceted nature of our evaluation, ensuring we validate all aspects of the system."

## Integration Checklist

Before submission, verify:

- [ ] All four figures use consistent visual style
- [ ] Each figure caption references related figures
- [ ] Text includes forward and backward references
- [ ] Shared parameters are documented in a table
- [ ] The narrative arc is clear in the introduction
- [ ] Each experiment's motivation is explicit
- [ ] Transitions between sections are smooth
- [ ] The conclusion synthesizes all four experiments
- [ ] Supplementary material connects the experiments
- [ ] Code/data availability covers all experiments

## Final Thoughts

Figure 4 is the **capstone** of your experimental validation. It:
- Resolves the tension created by Figure 3
- Validates the approach established in Figures 1-2
- Provides the strongest practical justification
- Addresses the most obvious reviewer question

When integrated properly with the other figures, it transforms your paper from "here's a system that works" to "here's a carefully designed system with principled justification and comprehensive validation."

**The four figures together tell a complete story. Make sure your paper narrative reflects that.**

