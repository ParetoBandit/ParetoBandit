# Quick Integration Checklist: Connecting All Experiments

**Purpose:** Actionable checklist for paper revisions  
**Date:** February 13, 2026

---

## ✅ Immediate Actions (High Priority)

### 1. Connect Figure 02 (Distribution Shift) to Regime Selection

**Where:** After Figure 02 discussion in Results section

**Add this paragraph:**
> "The substantial distribution shift (PSI=0.275, 95% CI: [0.243, 0.332]) quantifies the severity of domain mismatch that causes warmup priors to fail catastrophically in Table 2 (79 cumulative regret). This shift magnitude—training on 68.6% hard prompts but deploying on 13.7%—provides an actionable decision criterion for learning rate selection. For shifts this severe (PSI≥0.25), the convergence regime (η=5.0, Section X) enables complete prior unlearning within 300-1,121 steps, while moderate shifts (0.1≤PSI<0.2) warrant balanced adaptation (η=1.0, Section Y)."

**Impact:** Transforms Figure 02 from descriptive to prescriptive

---

### 2. Connect Alignment Tax (Fig 01) to Negative Intelligence Tax (Exp 05)

**Where:** Experiment 05 discussion (Pareto Frontier)

**Add this sentence:**
> "This 'Negative Intelligence Tax'—paying $43× more yields 1.3% worse quality—is a routing-level manifestation of the Alignment Tax discovered in Figure 1: static routers over-provision expensive models to High PC1 tasks where RLHF-optimized models systematically underperform cheaper alternatives due to format compliance failures."

**Impact:** Unifies task-level and routing-level economic findings

---

### 3. Clarify Semantic Categories (Table 01) vs Transfer Mechanism (Exp 07)

**Where:** Add footnote to Table 01

**Add footnote:**
> "Semantic categories serve data composition and diversity analysis purposes. The semantic transfer mechanism validation, which reveals that semantic similarity does not predict task-level performance correlation (r=-0.38, p=0.75), appears in Section X. Transfer success operates through implicit regularization (26× variance boost) rather than semantic accuracy."

**Impact:** Resolves apparent contradiction proactively

---

### 4. Add Unified Feature Space Statement

**Where:** Methodology section (Feature Engineering subsection)

**Add paragraph:**
> "All experiments operate in a unified semantic feature space to ensure consistency from problem discovery to solution validation. Prompts are embedded using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional), reduced to 32 dimensions via PCA trained on 80,000 prompts (35.14% cumulative variance). The primary semantic axis (PC1, 3.10% variance) discovered in Figure 1—which identifies the quality inversion at PC1=0.3—becomes the substrate for all adaptation experiments (4-7). This architectural choice ensures the router learns to exploit the exact quality inversion discovered during problem analysis."

**Impact:** Demonstrates methodological consistency and intentional design

---

### 5. Cross-Reference Conservative Learning (Table 02 ↔ Exp 07)

**Where:** Add footnote in Experiment 07 or Table 02 discussion

**Add footnote:**
> "The conservative learning rate (η=0.1) evaluated in Table 2 corresponds to the Cold-Start Regime (Section X), where it provides 3.2% short-term benefit over cold start but prevents long-term convergence (5.9% below optimal in 1,121-step evaluation). This regime prioritizes immediate deployment benefit over long-term quality maximization."

**Impact:** Clarifies why same η yields different interpretations

---

## 📊 Medium Priority: Add Decision Framework

### 6. PSI-Based Learning Rate Selection Algorithm

**Where:** Practical Deployment Recommendations section

**Add algorithm box:**

```
Algorithm: Learning Rate Selection via Distribution Shift

Input: Held-out deployment sample D (N=100-200)
Output: Recommended learning rate η

1. Compute PSI ← PopulationStabilityIndex(Training, D)
2. IF PSI < 0.1:
     RETURN η=0.1 (Cold-Start: exploit priors confidently)
3. ELIF 0.1 ≤ PSI < 0.2:
     RETURN η=1.0 (Pareto: balanced adaptation)
4. ELIF 0.2 ≤ PSI < 0.25:
     RETURN η∈[0.3,1.0] (Safety: prioritize detection)
5. ELSE (PSI ≥ 0.25):
     RETURN η=5.0 (Convergence: complete unlearning)

Example: Our data (PSI=0.275) → Use η=5.0 for optimal long-term performance
```

**Impact:** Provides actionable decision criterion for practitioners

---

## 🔗 Low Priority: Add Cross-References

### 7. Throughout Paper

Add these cross-references in relevant sections:

**Figure 01 discussion:**
> "(This quality inversion becomes the substrate for adaptation in Sections X-Y)"

**Figure 02 discussion:**
> "(This shift severity informs learning rate selection, see Algorithm Z)"

**Table 02 discussion:**
> "(This catastrophic baseline validates all safety claims in Experiments 4-7)"

**Experiment 04 discussion:**
> "(Complete unlearning validated at PSI≥0.25 severity, as quantified in Figure 2)"

**Experiment 05 discussion:**
> "(Partial adaptation trap arises from substantial shift, PSI=0.275, Figure 2)"

**Experiment 06 discussion:**
> "(Detection occurs 10× faster than unlearning, enabling safety even under severe mismatch)"

**Experiment 07 discussion:**
> "(Implicit regularization mechanism differs from semantic categorization in Table 1)"

---

## 📝 Suggested Abstract Update

**Current abstract mentions:**
- Three-regime framework ✅
- Implicit regularization ✅
- PSI quantification ❌ (MISSING)
- Economic impact ✅

**Add to abstract:**
> "Under substantial distribution shift (PSI=0.275), we demonstrate that learning rate selection should match shift severity: η=0.1 for rapid deployment (0-300 steps), η=1.0 for cost-quality balance (1,121 steps with partial adaptation), η=0.3-1.0 for catastrophic detection (3-50 steps), and η=5.0 for complete prior unlearning (300-1,121 steps)."

---

## 📊 Quick Reference Table for Paper

**Add this table to connect all experiments:**

| Experiment | Key Finding | Connects To | Integration Point |
|------------|-------------|-------------|-------------------|
| **Figure 01** | Quality inversion at PC1=0.3 | All experiments | Defines feature space |
| **Figure 02** | PSI=0.275 (substantial shift) | Table 02, Exp 04-07 | Explains warmup failure |
| **Table 01** | 81,871 prompts, semantic diversity | Exp 07 | Data organization ≠ prediction |
| **Table 02** | Baseline: 79 regret catastrophe | All regimes | Universal safety reference |
| **Figure 03** | α=2.0, γ=0.05 validated | All regimes | Architectural constants |
| **Exp 04** | η=5.0: complete unlearning | Figure 02 (PSI≥0.25) | Convergence regime |
| **Exp 05** | η=1.0: partial adaptation | Figure 01, 02 | Pareto regime, explains trap |
| **Exp 06** | η=0.3-1.0: 3-50 step detection | Exp 04 | Safety regime, timescale |
| **Exp 07** | η=0.1: 14% short-term benefit | Table 02, Table 01 | Cold-start regime |

---

## ✅ Validation Checklist

Before submission, verify:

- [ ] Figure 02 explicitly connected to learning rate selection
- [ ] Alignment Tax (Fig 01) linked to Negative Intelligence Tax (Exp 05)
- [ ] Semantic categories (Table 01) clarified vs transfer mechanism (Exp 07)
- [ ] Unified feature space mentioned in methodology
- [ ] Conservative learning (Table 02) cross-referenced with Cold-Start (Exp 07)
- [ ] PSI-based decision algorithm included
- [ ] All major experiments cross-referenced
- [ ] Abstract mentions PSI severity criterion
- [ ] No contradictions remain unaddressed

---

## 🎯 Key Messages to Emphasize

### Discovery → Solution Pipeline

```
Figure 01: Discover quality inversion (PC1=0.3, Alignment Tax)
          ↓
Figure 02: Quantify shift severity (PSI=0.275, substantial)
          ↓
Table 02: Establish baseline failure (79 regret, catastrophic)
          ↓
Figure 03: Validate architecture (α=2.0, γ=0.05, η=1.0)
          ↓
Exp 04-07: Provide regime-specific solutions (η selection)
```

### Economic Impact Chain

```
Figure 01: Task-level inefficiency (17.6% tasks, GPT-4 underperforms)
          ↓
Exp 05: Routing-level penalty ($43× cost, 1.3% worse quality)
          ↓
Solution: 27% cost reduction + 12.3% quality improvement
```

### Statistical Rigor Escalation

```
Figure 01: p < 10⁻¹⁴³ (overwhelming single-seed evidence)
          ↓
Table 02: N=10 seeds, variance quantification
          ↓
Figure 03: 5-10 seeds, design validation
          ↓
Exp 06: N=20 seeds, comprehensive ablation
```

---

## 💡 One-Sentence Summaries for Each Connection

**For quick reference during paper writing:**

1. **Fig 01 → All**: "PC1=0.3 boundary discovered in Fig 1 is the feature space for all adaptation."

2. **Fig 02 → Regimes**: "PSI=0.275 severity determines η selection: ≥0.25 requires convergence regime."

3. **Table 01 → Exp 07**: "Semantic categories organize data; transfer mechanism is implicit regularization."

4. **Table 02 → All**: "79 regret baseline validates all safety claims across regimes."

5. **Fig 03 → Regimes**: "α, γ are architectural constants; η is the operational variable controlling regimes."

6. **Fig 01 → Exp 05**: "Alignment Tax (task-level) causes Negative Intelligence Tax (routing-level)."

7. **Table 02 → Exp 07**: "η=0.1 in Table 2 is Cold-Start regime: short-term benefit, not long-term convergence."

8. **Fig 02 → Exp 05**: "PSI=0.275 explains partial adaptation trap: wrong direction + insufficient time."

9. **Exp 06 → Exp 04**: "3-50 step detection is 10× faster than 300-1,121 step unlearning: timescale separation."

10. **All → Unified**: "All experiments share 32-D PCA space trained on 80K prompts from Table 1."

---

**Status:** Ready for implementation  
**Estimated Time:** 2-3 hours for all high-priority additions  
**Impact:** Transforms disconnected experiments into unified narrative
