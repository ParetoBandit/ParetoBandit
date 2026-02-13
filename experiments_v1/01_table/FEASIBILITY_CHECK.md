# Feasibility Check: Table 1 Redesign Options

**Date**: February 13, 2026  
**Status**: Data availability assessed

---

## Data Availability Assessment

### **What We Found**

```bash
# Holdout data files:
data/holdout_prompts_for_rejudge.jsonl          # 750 prompts
data/holdout_rewards_gpt4turbo_rejudged.jsonl   # 1,195 entries (only gpt-4-turbo)

# Pareto results:
experiments_v1/05_figure/results/pareto_results_final.json  # Aggregate only
```

### **Key Discovery**: Limited per-prompt data

**Available**:
- ✅ 750 holdout prompts (raw text)
- ✅ GPT-4-Turbo rewards for each prompt (1,195 entries)
- ✅ Aggregate results from Pareto experiment (cost/reward by method)

**NOT Available** (for stratified analysis):
- ❌ Per-prompt rewards for Mixtral
- ❌ Per-prompt rewards for BanditGPT router selections
- ❌ Per-prompt rewards for RouteLLM selections

**Conclusion**: We **cannot** do full stratified performance analysis (Option 3) without regenerating per-prompt results.

---

## Revised Options

### **Option 1: Keep Current Table** ⚠️ **NOT RECOMMENDED**

**Status**: Already fixed (Tier 1 complete ✅)

**Pros**:
- No additional work

**Cons**:
- Categories with 49% accuracy still present
- "Why categorize?" question unanswered
- Disconnected from experiments

**Verdict**: Only if you're completely out of time

---

### **Option 2: Simplify to Pure Provenance** ✅ **RECOMMENDED**

**Status**: ✅ **FEASIBLE** (1 day implementation)

**What to do**:
1. Remove semantic categories entirely
2. Keep essential provenance:
   - Data sources (LMSYS Arena, RouteLLM)
   - Split sizes (80k/1,121/750)
   - Split purposes (PCA/warmup/dev/holdout)
   - Model details (mixtral, gpt-4-turbo, gpt-4o)
3. Streamline table design

**New Table Design**:
```latex
\begin{table}[t]
\centering
\caption{Dataset Description and Experimental Splits}
\label{tab:dataset}
\begin{tabular}{@{}llrl@{}}
\toprule
\textbf{Split} & \textbf{Source} & \textbf{Size} & \textbf{Purpose} \\
\midrule
PCA Training & RouteLLM Battles & 80,000 & Dimensionality reduction (384→32) \\
Warmup Priors & RouteLLM Battles & 80,000 & LinUCB initialization (A, b) \\
Development & LMSYS Arena & 1,121 & Online learning \& calibration \\
Holdout & LMSYS Arena & 750 & Final evaluation \\
\midrule
\textbf{Total} & & \textbf{81,871} & \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize
\textbf{Data Sources:} All prompts from LMSYS Chat Arena, a public dataset of real user-LLM interactions. 
\textbf{RouteLLM Battles}~\cite{ong2024routellm}: Pairwise comparisons (mixtral-8x7b vs gpt-4-turbo) 
from HuggingFace dataset \texttt{routellm/gpt4\_judge\_battles}. Used for PCA training and warmup prior 
generation (covariance matrix $\mathbf{A} \in \mathbb{R}^{33 \times 33}$, belief vector 
$\mathbf{b} \in \mathbb{R}^{33}$). 
\textbf{LMSYS Arena}: Stratified splits with mixtral-8x7b and gpt-4o evaluations. Model substitution 
(gpt-4-turbo$\rightarrow$gpt-4o) reflects current flagship tier availability. Zero data leakage verified 
(243 overlaps removed, 0.24\%). Holdout set provides 750 independent test samples for unbiased evaluation.
\textbf{Sample Size:} Evaluation set (1,871 total) exceeds prior work on LLM routing (RouteLLM: ~1,000).
\end{table}
```

**Pros**:
- ✅ Clean and focused
- ✅ No accuracy concerns
- ✅ Directly supports reproducibility
- ✅ Removes disconnected categories
- ✅ Quick to implement (1 day)

**Cons**:
- Loses "diversity" narrative (but was it needed?)

**Implementation**:
```bash
# Day 1 (4-6 hours):
1. Create simplified LaTeX table (2 hours)
2. Update analyze_dataset_composition.py to remove category analysis (1 hour)
3. Update README.md (1 hour)
4. Regenerate and test (1 hour)
```

---

### **Option 3: Stratified Performance Analysis** ⚠️ **NOT FEASIBLE WITHOUT MAJOR EFFORT**

**Status**: ⚠️ **REQUIRES REGENERATING EXPERIMENT DATA**

**What's missing**:
- Need per-prompt rewards for ALL methods (BanditGPT, RouteLLM, baselines)
- Current data only has aggregate results

**To make feasible, would need to**:
1. Re-run Pareto experiment with per-prompt logging (1 day)
2. Or extract per-prompt results from router's internal state (if saved)
3. Then implement stratified analysis (2 days)

**Total effort**: 3-4 days minimum

**Pros**:
- ✅ Strongest scientific contribution
- ✅ Justifies categorization
- ✅ Validates robustness

**Cons**:
- ❌ Requires significant additional work
- ❌ May need to re-run experiments
- ❌ Timeline extends by ~1 week

**Verdict**: Only pursue if:
- You have access to per-prompt router logs
- OR you're willing to re-run experiments with detailed logging
- OR you have 1 week extra time

---

### **Option 4: Minimal "Data Summary" in Appendix** ✅ **ULTRA-SAFE ALTERNATIVE**

**Status**: ✅ **EXTREMELY FEASIBLE** (2 hours)

**Radical simplification**: Move Table 1 to appendix, keep only essential info in main text

**Main text** (in methods section):
```latex
\paragraph{Dataset.} We evaluate on 1,871 prompts from LMSYS Chat Arena~\cite{zheng2023lmsys}, 
split into development (N=1,121) and holdout (N=750) sets. LinUCB warmup priors are initialized 
from 80,000 RouteLLM battles~\cite{ong2024routellm} (mixtral-8x7b vs gpt-4-turbo pairwise 
comparisons). All splits are disjoint (zero data leakage verified). See Appendix~\ref{app:data} 
for complete provenance.
```

**Appendix** (detailed table):
- Full provenance as in Option 2
- Split details
- Processing steps
- Quality assurance notes

**Pros**:
- ✅ Main text stays focused on methods/results
- ✅ Avoids "Table 1" pressure (usually dataset is shown in appendix anyway)
- ✅ No category controversy
- ✅ Extremely quick (2 hours)

**Cons**:
- Less prominent
- Readers need to flip to appendix

**Verdict**: ✅ **RECOMMENDED** if you want absolutely minimal risk and effort

---

## Decision Matrix

| Option | Feasibility | Effort | Scientific Value | Risk | Recommendation |
|--------|-------------|--------|------------------|------|----------------|
| **1. Keep current** | ✅ Done | 0 days | ⚠️ Low | ⚠️ High | ❌ **NOT RECOMMENDED** |
| **2. Simplify table** | ✅ High | 1 day | ✅ Good | ✅ Low | ✅ **RECOMMENDED** |
| **3. Stratified analysis** | ⚠️ Requires re-run | 3-4 days | 🎯 Excellent | ⚠️ Medium | ⚠️ Only if time permits |
| **4. Move to appendix** | ✅ Very high | 2 hours | ✅ Good | ✅ Very low | ✅ **ULTRA-SAFE** |

---

## Final Recommendation

### **Primary Choice**: Option 2 (Simplify) + Option 4 (Move detail to appendix)

**Hybrid approach**:
1. **Main text**: Brief dataset description in methods (3-4 sentences)
2. **Appendix**: Full provenance table (no categories)
3. **Remove**: Semantic categories entirely

**Implementation** (1 day):

**Morning (4 hours)**:
1. Write main text dataset description
2. Create appendix table (simplified from Option 2)
3. Update references throughout paper

**Afternoon (2 hours)**:
4. Remove category analysis from scripts
5. Update documentation
6. Test LaTeX compilation

**Result**:
- ✅ Clean, professional presentation
- ✅ No category accuracy concerns
- ✅ Complete provenance preserved
- ✅ Main text stays focused
- ✅ Low risk, quick implementation

---

### **Alternative**: Option 2 only (if table must be in main text)

**When to use**:
- Journal requires dataset table in main text
- Conference has space for it
- You prefer keeping it prominent

**Same implementation as above**, just keep table in main text instead of appendix.

---

### **NOT RECOMMENDED**: Option 3 (unless you have per-prompt data ready)

**Only pursue if**:
1. You find that per-prompt router logs already exist somewhere
2. You have 1 extra week and want strongest possible scientific contribution
3. A reviewer specifically requests stratified analysis

**Otherwise**: Too much effort for unclear payoff

---

## Next Steps

### **Recommended Path** (1 day total)

**Step 1** (2 hours): Design decision
```
- Choose: Option 2 (table in main) OR Option 2+4 (hybrid)
- Sketch new table layout
- Write main text description
```

**Step 2** (3 hours): Implementation
```
- Create new LaTeX table
- Update/simplify Python script
- Remove category analysis code
```

**Step 3** (2 hours): Integration
```
- Update README
- Update paper references
- Test compilation
```

**Step 4** (1 hour): Verification
```
- Check all references work
- Verify no broken citations
- Proofread
```

---

## Questions to Answer

### **Q1: Must Table 1 be in main text?**

**Check**:
- Conference guidelines (some require dataset table in main)
- Page limits (appendix might save space)
- Reviewer expectations (usually appendix is fine)

**Recommendation**: If unsure, use hybrid (brief main + detailed appendix)

### **Q2: Do we have per-prompt router logs anywhere?**

**Check**:
```bash
# Search for detailed logs
find experiments_v1 -name "*detailed*" -o -name "*per_prompt*" -o -name "*trace*"

# Check router output directories
ls experiments_v1/05_figure/results/
ls experiments_v1/02_table/data/
```

**If YES**: Option 3 becomes feasible  
**If NO**: Stick with Option 2 or 2+4

### **Q3: How much time do we have?**

**< 1 day**: Option 4 (move to appendix) - 2 hours  
**1 day**: Option 2 (simplify) or 2+4 (hybrid) - 6-8 hours  
**> 3 days**: Option 3 (stratified) - only if data exists  

---

## Implementation Template (Option 2+4)

### **Main Text** (methods section):

```latex
\subsection{Dataset}

We evaluate on N=1,871 prompts from LMSYS Chat Arena~\cite{zheng2023lmsys}, 
a public dataset of real user-LLM interactions. The dataset is split into 
a development set (N=1,121) for online learning and calibration, and a held-out 
test set (N=750) for final evaluation. 

LinUCB warmup priors are initialized from 80,000 RouteLLM battles~\cite{ong2024routellm}, 
representing pairwise comparisons between mixtral-8x7b-instruct and gpt-4-turbo. 
These priors encode user preferences learned from large-scale battle data, providing 
a strong starting point for the contextual bandit. PCA dimensionality reduction 
(384→32) is trained on the same 80,000 battles to create a compact semantic 
representation.

All data splits are strictly disjoint, with automated leakage detection removing 
243 overlapping prompts (0.24\%) from the warmup set. Development and holdout sets 
use mixtral-8x7b-instruct and gpt-4o evaluations, reflecting current model availability. 
Complete dataset description and provenance appear in Appendix~\ref{app:data}.
```

### **Appendix** (detailed table from Option 2):

```latex
\section{Dataset Details}
\label{app:data}

[Include full table from Option 2 above, plus any additional notes]
```

---

## Bottom Line

### **What to do**: Option 2 or 2+4

**Why**:
- ✅ Feasible with available data
- ✅ Quick to implement (1 day or less)
- ✅ Removes category controversy
- ✅ Preserves essential provenance
- ✅ Focuses on what matters

**What NOT to do**: Option 3 (stratified)

**Why**:
- ❌ Requires data we don't have
- ❌ Would need to re-run experiments  
- ❌ 3-4 days minimum
- ❌ Benefit unclear without testing first

**What to avoid**: Option 1 (keep current)

**Why**:
- ❌ Categories with 49% accuracy
- ❌ Disconnected from experiments
- ❌ Vulnerable to reviewer criticism

---

**Status**: Feasibility assessment complete  
**Recommendation**: Implement Option 2 (Simplify) or 2+4 (Hybrid)  
**Timeline**: 1 day  
**Next action**: Choose between main text table vs appendix table
