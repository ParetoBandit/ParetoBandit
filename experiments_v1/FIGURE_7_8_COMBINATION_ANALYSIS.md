# Should We Combine Figures 7 & 8? Trade-off Analysis

**Question**: Should we combine Figure 7 (Zero-Shot Adoption) and Figure 8 (Expert Selection) into a single figure focused on the heterogeneous vs. homogeneous trade-off?

---

## Current State

### Figure 7: Zero-Shot Readiness
**Research Question**: "Can we deploy new models without cold-start disruption?"

**Configuration**: Heterogeneous experts (Conservative decay, Adaptive constant)

**Focus**: Reward performance over time

**Key Finding**: "Semantic transfer provides 3.2% immediate benefit through implicit regularization"

**Panels**:
- Reward curves comparing Cold Start, Warmup Only, Realistic Baseline, Semantic Transfer
- Shows smooth adaptation with stable 75/25 expert weights

---

### Figure 8: Adaptive Expert Selection
**Research Question**: "When does semantic transfer actually help?"

**Configuration**: Homogeneous experts (both constant α=2.0)

**Focus**: Expert weight evolution and regime identification

**Key Finding**: "Corralling adaptively switches between warmup (33%) and tabula rasa (67%) based on data match"

**Panels**:
- Top: Expert weight evolution across 3 seeds
- Middle: Performance stratified by regime (warmup-dominant vs tabula-dominant)
- Bottom: Key findings summary

---

## Option A: Keep Separate ⭐ RECOMMENDED

### Pros
1. **Different research questions** - Each answers distinct scientific question
2. **Different audiences** - Practitioners (Fig 7) vs. Researchers (Fig 8)
3. **Clearer narrative** - Each figure has focused message
4. **Existing investment** - Figures already created and validated
5. **Paper structure** - Fit naturally in different subsections

### Cons
1. **Page limits** - Uses 2 figures instead of 1
2. **Potential confusion** - Readers might not notice config difference
3. **Redundancy** - Both show expert dynamics in some form

### Implementation
**Add cross-references and clarifying text:**

```latex
% In Figure 7 caption/text
...heterogeneous expert configuration (Conservative: $\alpha$ decay, Adaptive: constant) 
prioritizes stability for short-term deployment benefit. For analysis of alternative 
homogeneous configuration enabling decisive regime switching, see Figure~\ref{fig:expert_selection}.

% In Figure 8 caption/text
...homogeneous expert configuration (both $\alpha=2.0$ constant) enables decisive regime 
identification. This contrasts with heterogeneous configuration (Figure~\ref{fig:ablation}), 
which prioritizes smooth hedging for risk-averse deployments.
```

**Estimated effort**: 10 minutes (text updates only)

---

## Option B: Combine into Single Figure

### Design Option B1: Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│  Figure 7/8: Configuration Trade-offs in Expert Selection       │
├─────────────────────────┬───────────────────────────────────────┤
│ (A) Heterogeneous       │ (B) Homogeneous                       │
│     (Smooth Hedging)    │     (Decisive Switching)              │
├─────────────────────────┼───────────────────────────────────────┤
│ Reward over time        │ Reward over time                      │
│ (shows stability)       │ (shows variance by seed)              │
├─────────────────────────┼───────────────────────────────────────┤
│ Expert weights          │ Expert weights                        │
│ (75/25 stable)          │ (100/0 or 0/100 binary)              │
├─────────────────────────┼───────────────────────────────────────┤
│ Use case: Risk-averse   │ Use case: Fast adaptation             │
└─────────────────────────┴───────────────────────────────────────┘
```

**Caption**: 
> "Configuration Trade-offs: Heterogeneous vs. Homogeneous Expert Design. 
> (A) Heterogeneous configuration (Conservative: α decay 1.0→0.01, Adaptive: α=2.0) 
> exhibits smooth hedging (~75/25 weights), prioritizing stability. 
> (B) Homogeneous configuration (both α=2.0) enables decisive switching (100/0 or 0/100), 
> maximizing adaptation speed. Choice reflects deployment priorities: stability vs. adaptability."

### Pros
1. **Explicit trade-off** - Direct visual comparison
2. **Educational** - Shows design flexibility clearly
3. **Space efficient** - One figure instead of two
4. **Resolves "contradiction"** - Makes it obvious these are design choices

### Cons
1. **Complex figure** - Lots of information in one figure
2. **Dilutes both messages** - Neither story gets full attention
3. **Layout challenges** - Fitting 6+ panels clearly
4. **Different focuses** - Fig 7 is about reward, Fig 8 is about expert selection
5. **Significant rework** - Need to regenerate figure, update text

**Estimated effort**: 2-3 hours (figure redesign + text updates)

---

### Design Option B2: Stacked Configuration Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│  Figure 7/8: Expert Dynamics Across Configurations              │
├─────────────────────────────────────────────────────────────────┤
│ (A) Reward Performance: Heterogeneous vs Homogeneous            │
│     [Line plots comparing both configs over time]               │
├─────────────────────────────────────────────────────────────────┤
│ (B) Expert Weight Evolution                                     │
│     Left: Heterogeneous (smooth 75/25)                          │
│     Right: Homogeneous (binary 100/0 or 0/100)                  │
├─────────────────────────────────────────────────────────────────┤
│ (C) Key Findings                                                │
│     Table comparing use cases and trade-offs                    │
└─────────────────────────────────────────────────────────────────┘
```

**Estimated effort**: 3-4 hours (need to run heterogeneous for seeds 42-44 for fair comparison)

---

## Option C: Keep Separate + Add Appendix Figure

### Main Paper
- **Figure 7**: Zero-shot adoption (heterogeneous, smooth hedging)
- **Figure 8**: Expert selection (homogeneous, decisive switching)
- Both with clarifying cross-references

### Appendix
- **Figure A1**: Direct side-by-side comparison of configurations
- Shows same experiment with both configs
- Educational supplement for interested readers

### Pros
1. **Best of both worlds** - Focused main figures + explicit comparison
2. **Flexibility** - Detailed comparison available without cluttering main narrative
3. **Page limits** - Appendix doesn't count toward main limit

### Cons
1. **Most work** - Need to create appendix figure
2. **Reader effort** - Have to flip to appendix for comparison

**Estimated effort**: 2 hours (create appendix figure)

---

## Decision Factors

### If Page Limits are Tight → Option B (Combine)
KDD has strict 9-page limit. If you're at 8.5+ pages, combining saves space.

### If Narrative Clarity is Priority → Option A (Keep Separate)
Two distinct research questions deserve two distinct figures.

### If Reviewer Confusion is Concern → Option B or C
Explicit comparison prevents "why are these contradictory?" questions.

### If Time is Limited → Option A (Keep Separate)
Just add clarifying text (10 minutes) vs. regenerating figures (3+ hours).

---

## Recommendation by Scenario

### Scenario 1: Paper is Nearly Complete (95%+ done)
**Action**: Option A (Keep Separate)
**Reasoning**: Minimize disruption, quick fix with cross-references
**Time**: 10 minutes

### Scenario 2: Major Revision Phase
**Action**: Option C (Separate + Appendix)
**Reasoning**: Shows thoroughness, addresses reviewer concerns preemptively
**Time**: 2 hours

### Scenario 3: Severe Page Limit Pressure
**Action**: Option B (Combine)
**Reasoning**: Must save space, trade-off story is compelling
**Time**: 3 hours

### Scenario 4: Reviewers Explicitly Confused
**Action**: Option B (Combine)
**Reasoning**: Directly address confusion with visual comparison
**Time**: 3 hours

---

## My Recommendation: Keep Separate (Option A)

### Why?

1. **Different research questions**:
   - Fig 7: "Can we deploy smoothly?" → Practitioners care
   - Fig 8: "When does transfer help?" → Researchers care
   
2. **Current figures are good**:
   - Both tell clear, focused stories
   - Figure 8 already addresses main scientific question
   - Figure 7 provides practical deployment guidance

3. **Easy fix**:
   - Add 2-3 sentences explaining config choice
   - Cross-reference between figures
   - Maybe add small clarification box to each figure

4. **Combining might dilute**:
   - Figure 8's key insight is regime-dependent expert selection
   - Making it "just a config comparison" weakens that finding
   - Figure 7's value is showing semantic transfer works
   - Making it "one of two configs" weakens that value

### Proposed Text Updates

**For Figure 7 caption:**
```latex
\caption{[Existing caption]... This experiment uses heterogeneous expert configuration 
(Conservative: $\alpha$ decay 1.0$\to$0.01, Adaptive: $\alpha=2.0$ constant) to prioritize 
stable hedging behavior appropriate for conservative learning regime ($\eta=0.1$). For analysis 
of alternative homogeneous configuration enabling decisive regime switching, see 
Figure~\ref{fig:expert_selection}.}
```

**For Figure 8 text:**
```latex
\paragraph{Configuration Choice.}
This experiment uses homogeneous expert configuration (both $\alpha=2.0$ constant, recommended 
by Figure~\ref{fig:architecture}) to enable decisive regime identification. This design choice 
contrasts with heterogeneous configuration (Figure~\ref{fig:ablation}), which prioritizes smooth 
hedging ($\sim$75/25 weights) for risk-averse deployments. The homogeneous design enables clearer 
scientific analysis by creating sharp regime differentiation, revealing that Corralling's expert 
choice is data-dependent: 33\% of seeds select warmup expert, 67\% select tabula rasa expert.
```

**Add to methodology or discussion:**
```latex
\paragraph{Alpha Configuration Trade-offs.}
Our system supports two expert configurations with distinct behaviors. Heterogeneous configuration 
(Conservative: decaying $\alpha$, Adaptive: constant $\alpha$) exhibits smooth hedging ($\sim$75/25 
weight distribution), minimizing disruption during deployment—appropriate for risk-averse organizations, 
high exploration costs, or initial rollout phases. Homogeneous configuration (both constant $\alpha=2.0$) 
enables decisive regime switching (near-binary expert selection), maximizing adaptation speed—appropriate 
for high-traffic systems, low exploration costs, or when clear performance differences justify aggressive 
commitment. The choice reflects fundamental trade-offs between stability and adaptability in online learning.
```

---

## Bottom Line

**Keep them separate with clarifying text** unless:
- You have severe page limits (then combine)
- Reviewers explicitly confused about contradiction (then combine)
- You're in major revision phase (then add appendix comparison)

The scientific contributions are stronger with focused figures. The "trade-off" can be explained in text without visual comparison.

**Time investment**:
- Option A (Separate + text): 10 minutes ⭐
- Option B (Combine): 3 hours
- Option C (Separate + Appendix): 2 hours

What's your page limit situation and submission timeline?
