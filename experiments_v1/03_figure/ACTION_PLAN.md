# Action Plan: Fix Paper Based on Corrected Results

**Date:** February 13, 2026  
**Priority:** CRITICAL  
**Deadline:** Before any submission/publication

---

## Summary of Situation

The alpha ablation experiment had a critical bug that invalidated all results. After fixing and re-running:

- ❌ **Current design is suboptimal** (3rd of 4, not optimal)
- ✅ **Reversed design is best** (14% better than current)
- ❌ **"48% improvement" claim is invalid** (was artifact of bug)
- ✅ **Heterogeneity helps modestly** (2.3%, not dramatically)

---

## Decision Point: What Configuration to Use?

### Option A: Switch to Reversed (Recommended)

**Pros:**
- 14% better performance (43.4 vs 49.6 regret)
- Theoretically sound (informed priors + constant exploration)
- Honest - using the actual best configuration

**Cons:**
- Need to update router.py and re-run ALL experiments
- More work in short term
- Delays publication

**Recommendation:** ✅ Do this if you have time (2-3 days of work)

### Option B: Keep Current, Acknowledge Suboptimality

**Pros:**
- Less work - only update paper text
- Current config still "works" (3rd of 4, only 14% worse)
- Can frame as "conservative choice" or "future work"

**Cons:**
- Knowingly using suboptimal design
- Reviewers may question why not use best config
- Less impactful contribution

**Recommendation:** ⚠️ Only if publication deadline is imminent

---

## Action Plan: Option A (Switch to Reversed)

### Phase 1: Update Code (Day 1, ~4 hours)

#### 1.1 Update Router Configuration

**File:** `src/bandit_gpt/router.py`  
**Lines:** 2083-2112

```python
# CURRENT (SUBOPTIMAL):
expert_warmup = CostAwareLinUCBRouter(
    alpha_start=target_alpha / 2.0,  # Decaying
    alpha_end=0.01,                   # ← WRONG
)
expert_tabula_rasa = CostAwareTabulaRasaRouter(
    alpha_start=target_alpha,        # Constant
    alpha_end=target_alpha,           # ← WRONG
)

# NEW (OPTIMAL):
expert_warmup = CostAwareLinUCBRouter(
    alpha_start=target_alpha,        # Constant ✅
    alpha_end=target_alpha,          # Constant ✅
)
expert_tabula_rasa = CostAwareTabulaRasaRouter(
    alpha_start=target_alpha / 2.0,  # Decaying ✅
    alpha_end=0.01,                   # Decaying ✅
)
```

#### 1.2 Update Comments and Documentation

- Update code comments describing the strategy (lines 2072-2106)
- Update HETEROGENEOUS_EXPERTS_STRATEGY.md
- Update README.md configuration examples

#### 1.3 Add Validation Test

Create `tests/test_alpha_configuration.py`:
```python
def test_reversed_heterogeneous_config():
    """Ensure we're using the optimal reversed configuration."""
    router = BanditRouter.create(use_corralling=True, alpha=2.0)
    
    # Expert 1 (warmup) should have CONSTANT alpha
    assert router.corralling_router.experts[0].alpha_start == 2.0
    assert router.corralling_router.experts[0].alpha_end == 2.0
    
    # Expert 2 (tabula) should have DECAYING alpha
    assert router.corralling_router.experts[1].alpha_start == 1.0
    assert router.corralling_router.experts[1].alpha_end == 0.01
```

### Phase 2: Re-Run All Experiments (Day 1-2, ~8 hours compute)

#### 2.1 Core Ablation Experiments

- [x] experiment_3_heterogeneous_alpha_ablation.py (DONE)
- [ ] experiment_5_gamma_ablation.py (~2 hours)
- [ ] experiment_2a_weight_evolution.py (~1 hour)
- [ ] experiment_2bc_convergence_dynamics.py (~2 hours)

#### 2.2 Main Paper Experiments

Check which experiments use Corralling with heterogeneous experts:
- [ ] Figure 4: Corralling weight evolution
- [ ] Figure 7: Zero-shot readiness
- [ ] Figure 8: Sensitivity analysis
- [ ] Table 2: Performance comparison

**Run each and compare to existing results**

#### 2.3 Create Comparison Reports

For each re-run experiment:
```bash
cd experiments_v1/XX_figure
python experiment_YY.py > results_new.txt 2>&1
diff results_old.txt results_new.txt > diff.txt
```

Document:
- What changed?
- Why did it change?
- Does it affect paper claims?

### Phase 3: Update Paper (Day 2-3, ~8 hours)

#### 3.1 Abstract

**OLD:**
> "maintaining constant α preserves the system's ability to detect and adapt to distribution shifts, achieving 48% improvement over adaptive decay"

**NEW:**
> "maintaining appropriate exploration strategies per expert (constant for informed priors, decay for tabula rasa) provides incremental performance improvement (2-3%) while enabling robust adaptation to distribution shifts"

#### 3.2 Introduction

**Remove:**
- Any mention of "48% improvement"
- Claims about "constant α=2.0 is essential"

**Add:**
- "Expert-specific alpha strategies" as contribution
- "Role-based exploration tuning" concept

#### 3.3 Methodology (Section 3)

**File:** `paper/sections/methodology.tex`  
**Lines:** 67-69

**OLD:**
```latex
We use constant $\alpha=2.0$ for both experts throughout deployment.
[...] achieving 48\% improvement over adaptive decay in high-mismatch scenarios.
```

**NEW:**
```latex
We employ role-based alpha strategies: the Warmup Expert (informed priors)
uses constant $\alpha=2.0$ to maintain discovery potential, while the 
Tabula Rasa Expert (uninformed) decays $\alpha: 1.0 \to 0.01$ to balance 
initial exploration with eventual exploitation. This heterogeneous strategy 
provides 2.3\% improvement over homogeneous designs while maintaining 
adaptability to distribution shifts.
```

#### 3.4 Results (Section 4)

**Update Appendix D Table** (`paper/sections/appendix_d.tex` lines 173-177):

**NEW:**
```latex
\textbf{Reversed Heterogeneous} & \textbf{43.4 $\pm$ 12.4} & \textbf{--} \\
Homogeneous Constant ($\alpha=2.0$) & 45.2 $\pm$ 11.8 & +4.1\% \\
Current Heterogeneous & 49.6 $\pm$ 7.8 & +14.3\% \\
\midrule
Homogeneous Decay ($\alpha: 1.0 \to 0.01$) & 50.0 $\pm$ 17.1 & +15.2\% \\
```

**Add explanation:**
```latex
\paragraph{Role-Based Alpha Strategy.}
The optimal configuration assigns constant exploration to the informed expert
(warmup with priors) and decaying exploration to the uninformed expert 
(tabula rasa). This reverses our initial hypothesis but aligns with 
information-theoretic principles: experts with informative priors benefit 
from sustained exploration to detect distribution shifts, while blank-slate 
experts require initial exploration that converges to exploitation as they learn.
```

#### 3.5 Discussion

**Add new section:**
```latex
\subsection{Why Reversed Heterogeneity Works}

Our ablation studies revealed that the optimal alpha strategy depends on 
expert initialization:

\textbf{Warmup Expert (Informed Priors):} Constant $\alpha=2.0$ maintains 
the ability to detect when priors mismatch deployment data. Premature decay 
causes the expert to commit irreversibly to potentially incorrect beliefs.

\textbf{Tabula Rasa Expert (No Priors):} Decaying $\alpha: 1.0 \to 0.01$ 
provides initial exploration to build an internal model, then converges to 
exploitation as uncertainty reduces. Constant exploration would waste samples 
on prompts where the optimal model is already known.

This role-based strategy achieves 43.4 $\pm$ 12.4 cumulative regret, 
outperforming both homogeneous constant (45.2) and our initial reversed 
design (49.6) by 4-14\%.
```

#### 3.6 Conclusion

**Remove:**
- 48% improvement claim
- "Constant alpha is essential" claim

**Add:**
- Role-based exploration as contribution
- Modest but consistent heterogeneity benefit

### Phase 4: Update Supporting Materials (Day 3, ~2 hours)

#### 4.1 README.md

**Lines to update:**
- Line 25: Remove "Constant α=2.0 wins by 48%"
- Line 69: Update configuration table
- Line 182: Revise deployment recommendations

**Add:**
```markdown
### Optimal Alpha Configuration

Based on systematic ablation studies (N=5 seeds × 750 prompts):

| Expert | Priors? | Optimal α Strategy | Regret |
|--------|---------|-------------------|--------|
| **Warmup** | ✅ Yes | Constant 2.0 | 43.4 ± 12.4 |
| **Tabula Rasa** | ❌ No | Decay 1.0→0.01 | (same) |

**Why:** Informed experts need sustained exploration to detect drift. 
Uninformed experts need initial exploration that converges as they learn.
```

#### 4.2 HETEROGENEOUS_EXPERTS_STRATEGY.md

**Major rewrite:**
- Invert the expert descriptions
- Update diagrams/ASCII art
- Revise "Why This Works" section
- Update code examples

#### 4.3 experiments_v1/03_figure/README.md

**Complete rewrite:**
- Update all tables with new results
- Remove 48% improvement claims
- Add "Lessons Learned" section about the bug

---

## Action Plan: Option B (Keep Current, Acknowledge)

If publication deadline is imminent and you can't switch configs:

### Phase 1: Update Paper Only (Day 1, ~4 hours)

#### 1.1 Acknowledge Suboptimality

**Add to Discussion:**
```latex
\paragraph{Configuration Limitations.}
Systematic ablation revealed that reversing our expert alpha assignments 
(constant for warmup, decay for tabula rasa) achieves 14\% lower regret 
(43.4 vs 49.6). We retained our original configuration for this submission 
to maintain consistency with preliminary results, but recommend the reversed 
strategy for production deployments. The modest performance difference (6.2 regret 
units) suggests both configurations are viable, with future work exploring 
the theoretical underpinnings of role-based exploration strategies.
```

#### 1.2 Remove Invalid Claims

- Abstract: Remove 48% claim
- Introduction: Remove "constant α is essential"
- Methodology: Remove claim that current design is optimal
- Results: Update regret numbers
- Appendix D: Update table

#### 1.3 Downgrade Heterogeneity Claim

**Change:**
- "Heterogeneous strategy is core innovation" → "Heterogeneous strategy provides incremental benefit"
- "Significant improvement" → "Modest 2.3% improvement"

### Phase 2: Mark as Future Work (Day 1, ~1 hour)

**Add to Future Work:**
```latex
\subsection{Optimizing Expert-Specific Exploration}

Our ablation studies suggest that optimal alpha strategies may depend on 
expert initialization state. Future work should investigate:
\begin{itemize}
\item Theoretical framework for role-based exploration tuning
\item Adaptive meta-strategies that adjust alpha based on expert performance
\item Domain-specific alpha schedules learned from offline data
\end{itemize}
```

---

## Timeline Comparison

| Approach | Time Required | Paper Impact | Honesty |
|----------|--------------|--------------|---------|
| **Option A (Switch)** | 2-3 days | Strong contribution | ✅ High |
| **Option B (Acknowledge)** | ~8 hours | Moderate contribution | ⚠️ Medium |

---

## Recommendation

### If you have 2-3 days: Do Option A
- Results are stronger (43.4 vs 49.6 regret)
- Contribution is clearer (role-based exploration)
- Reviewers will appreciate using optimal config
- Paper will have more impact

### If deadline is <48 hours: Do Option B
- Acknowledges the finding honestly
- Retains most of paper structure
- Marks optimal config as future work
- Still publishable

---

## Checklist

### Before Submission

- [ ] All experiments re-run with fixed code
- [ ] All paper claims updated to match new results
- [ ] All figures regenerated with new data
- [ ] All tables updated with new numbers
- [ ] Code and paper are in sync
- [ ] README and documentation updated
- [ ] Supplementary materials updated

### Quality Checks

- [ ] No mention of "48% improvement" anywhere
- [ ] Alpha configuration described accurately
- [ ] Results tables match experiment outputs
- [ ] Code comments match actual behavior
- [ ] Tests validate correct configuration

### Final Validation

- [ ] Run all experiments start-to-finish
- [ ] Verify regret numbers match paper
- [ ] Check figures match experimental data
- [ ] Lint and format all code
- [ ] Spell check all documents

---

## Contact & Coordination

- **Technical lead:** Review code changes
- **Writing lead:** Review paper updates
- **PI:** Approve strategy (Option A vs B)
- **Team:** Review before submission

---

## Notes

- Keep old results for reviewer transparency if asked
- Document the bug fix in supplementary materials
- Add regression tests to prevent future bugs
- Consider blog post explaining the bug and fix

---

**IMPORTANT:** Do not submit or publicize until this action plan is complete!
