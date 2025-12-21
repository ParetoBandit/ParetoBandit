# KDD Paper Restructuring Guide: Democratization-First Framing

## Overview

This guide provides a roadmap for restructuring your KDD paper to emphasize democratization and accessibility while maintaining the technical rigor necessary for acceptance.

## Core Problem Identified

**Current framing:** "We built a better routing system that optimizes cost-quality trade-offs"  
**Revised framing:** "We built an accessibility tool that democratizes LLM access through intelligent routing"

The technical contributions remain identical, but the **narrative purpose** shifts from systems optimization to social impact.

---

## Restructuring Strategy

### Phase 1: Lead with Mission (Sections 1-2)

#### 1.1 Abstract (Replace existing)
**File:** `abstract_REVISED.tex`

**Key changes:**
- Lead with accessibility crisis (\$4-15/1k frontier costs create barriers)
- Position routing as the solution to over-provisioning
- Frame metrics in terms of **who benefits**: students, researchers, startups
- End with open-source release as democratization mechanism

**Integration:** Replace lines 84-88 in `main.tex` with content from `abstract_REVISED.tex`

#### 1.2 Introduction (Replace existing)
**File:** `introduction_REVISED.tex`

**Key changes:**
- **Opening (¶1):** Accessibility crisis → frontier costs price out users who need LLMs most
- **Problem (¶2):** Market fragmentation creates opportunity, but users lack navigation tools
- **Solution (¶3-4):** BanditGPT as accessibility tool with three core principles
- **Impact (¶5):** Concrete examples across user segments
- **Validation (¶6):** Technical proof builds trust for adoption
- **Release (¶7):** Open-source as democratization mechanism

**Integration:** Replace `introduction.tex` content entirely

#### 1.3 NEW SECTION: Democratization Through Use Cases
**File:** `use_cases.tex`

**Purpose:** Show concrete real-world examples BEFORE diving into technical method

**Structure:**
- Student projects (84% cost reduction enables AI education access)
- Independent researchers (68% reduction unlocks qualitative analysis at scale)
- Startups (converts AI from existential risk to viable feature)
- Enterprises (makes 10M query/year deployments economically feasible)
- Cross-cutting themes: volume enablement, quality preservation, creativity unlocking

**Integration:** Add `\input{use_cases}` between `introduction.tex` and `method.tex` in `main.tex`

**Page budget:** ~1 page (compress if needed; prioritize 2-3 detailed examples over 4 shallow ones)

---

### Phase 2: Reframe Technical Contributions (Sections 3-6)

#### 2.1 Method Section (Minor framing additions)
**File:** `method.tex` (keep existing, add framing sentences)

**Additions to make:**

**After Section 2.1 title:** Add paragraph:
```latex
The technical framework below exists to serve a single goal: making adaptive routing 
accessible to users who cannot afford extensive exploration. Each design choice—from 
shippable priors to decoupled scalability—addresses a specific accessibility barrier.
```

**After Section 2.7 (Distillation):** Add sentence:
```latex
This compression is critical for democratization: users download a 1MB prior and 
gain production intelligence immediately, without spending \$50–200 on cold-start exploration.
```

**After Section 2.9 (Scalability):** Add sentence:
```latex
Zero-overhead onboarding means users benefit from new, cheaper models the day they 
launch—without manual intervention or recalibration costs.
```

#### 2.2 Evaluation Section (Add interpretation layer)
**File:** `evaluation.tex` (keep existing results, add "What this means for users" paragraphs)

**After Table 1 (SOTA Comparison):** Add paragraph:
```latex
\paragraph{Accessibility Implications.} For a student processing 5,000 queries, 
Standard mode costs \$3.50 vs.\ \$21.90 for GPT-4o-only (84\% reduction). For an 
enterprise at 10M queries annually, this translates to \$7.0M vs.\ \$43.8M—the 
difference between viable deployment and budget-prohibitive experimentation.
```

**After Figure 4 (Pareto Frontier):** Add paragraph:
```latex
\paragraph{Tunable Trade-Offs for Diverse Users.} The smooth Pareto navigation 
enables distinct user profiles: Students operate in Standard mode (\$\lambda \gg 0$) 
to maximize query volume; Enterprises shift to Hybrid mode ($\lambda \approx 0$) 
for production workloads requiring 98\% reliability. This flexibility eliminates 
the "one size fits all" constraint of static systems.
```

---

### Phase 3: Elevate Democratization in Conclusion (Section 9)

#### 3.1 Conclusion (Replace existing)
**File:** `conclusion_REVISED.tex`

**Key changes:**
- **Opening:** Lead with democratization mission, not technical summary
- **Structure:**
  1. Democratization through technical innovation (flips the priority)
  2. Empirical validation for user trust (same metrics, user-focused framing)
  3. **Expanded "Broader Impact" → 5 paragraphs:**
     - Educational equity
     - Research democratization
     - Startup viability
     - Enterprise transformation
     - Environmental sustainability
  4. Ethical considerations (expanded from 2 to 3 risks)
  5. **New:** Open-source release details (what's included, who it serves)
  6. Limitations (same content, slightly reorganized)
  7. **New:** "A Call for Accessible AI Infrastructure" (aspirational closing)

**Integration:** Replace `conclusion.tex` content entirely

---

## Page Budget Management

**Current:** 8 pages (Introduction through Conclusions)  
**Additions:** +1 page (Use Cases section)  
**Required compression:** -1 page from existing content

### Suggested Compressions

1. **Related Work (¶0.75 → 0.5 pages):**
   - Remove "Static Model Selection" paragraph (tangential)
   - Compress Table 6 to 3 rows (keep only FrugalGPT, RouteLLM, Std. Bandits)

2. **Method Section (¶1.5 → 1.25 pages):**
   - Compress Section 2.4 (Regret Formulation) by removing second half of paragraph
   - Merge Section 2.6 (Why Contextual Bandits) into Section 2.1 as a single paragraph

3. **Evaluation (¶3 → 2.75 pages):**
   - Compress Section 4.1.1 (Benchmark Trap) to 3 sentences + Table 3 caption
   - Compress Section 4.4.1 (Confident Failure) to 2 paragraphs

**Net result:** 8 pages maintained with new Use Cases section included

---

## Integration Checklist

### Step 1: Update `main.tex`

```latex
% Replace abstract
\begin{abstract}
[Content from abstract_REVISED.tex]
\end{abstract}

% Document body
\begin{document}
\maketitle

\input{introduction}              % REVISED version
\input{use_cases}                 % NEW SECTION
\input{method}                    % Minor framing additions
\input{evaluation}                % Add interpretation paragraphs
\input{related_work}              % Compress to 0.5 pages
\input{conclusion}                % REVISED version

\bibliographystyle{ACM-Reference-Format}
\bibliography{references}
\input{appendix}
\end{document}
```

### Step 2: Test Compilation

```bash
cd kdd_paper/paper_submitted
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Step 3: Verify Page Count

```bash
# Check total pages
pdfinfo main.pdf | grep Pages

# Should show ≤8 pages for main content + references + appendix
```

---

## Narrative Arc Transformation

### Before (Current Structure)
1. **Introduction:** Technical problem (routing at scale)
2. **Method:** How we solve it (shippable priors, bandits)
3. **Evaluation:** We beat baselines (61% cost reduction)
4. **Conclusion:** This is good for systems; also helps accessibility [afterthought]

**Reader takeaway:** "Clever optimization technique for LLM routing"

### After (Revised Structure)
1. **Introduction:** Accessibility crisis → routing as solution → who benefits
2. **Use Cases:** Concrete examples of barriers removed
3. **Method:** Technical enablers of accessibility (framed as tools for democratization)
4. **Evaluation:** Proof that quality doesn't degrade + what cost savings mean for users
5. **Conclusion:** Democratization achieved through technical innovation → call to action

**Reader takeaway:** "Impactful accessibility tool with rigorous technical validation"

---

## Key Messaging Shifts

| Section | Old Frame | New Frame |
|---------|-----------|-----------|
| Abstract | "We optimize routing" | "We democratize access" |
| Intro ¶1 | "LLM ecosystem is fragmented" | "Frontier costs create barriers" |
| Method | "Here's how it works" | "Here's how we enable accessibility" |
| Results | "64.6% regret reduction" | "64.6% reduction → students afford 6× more queries" |
| Conclusion | "Good system + bonus: helps people" | "Helps people + here's the proof it works" |

---

## Addressing KDD Review Criteria

### Criterion 1: Technical Contribution
- **Maintained:** All proofs, algorithms, empirical results unchanged
- **Enhanced:** Stronger motivation (solving real-world accessibility problem vs. incremental optimization)

### Criterion 2: Experimental Rigor
- **Maintained:** Same datasets, baselines, metrics
- **Enhanced:** Added interpretation layer ("what this means for users")

### Criterion 3: Reproducibility
- **Enhanced:** Open-source release with pre-trained priors (lowers barrier to replication)

### Criterion 4: Impact (Applied DS Track)
- **Dramatically enhanced:** Concrete use cases, clear beneficiaries, democratization framing
- **Applied DS fit:** Shows real-world deployment viability, not just algorithmic novelty

---

## FAQ: Balancing Accessibility vs. Technical Depth

**Q: Will emphasizing democratization make the paper seem "less rigorous"?**  
A: No—the technical content is identical. You're changing the **narrative wrapper**, not the **scientific contribution**. KDD Applied DS explicitly values real-world impact alongside technical novelty.

**Q: Is the Use Cases section too "soft" for KDD?**  
A: No—concrete deployment scenarios demonstrate practical value, which is central to the Applied DS track. Every use case is backed by quantitative cost analysis grounded in your experimental results.

**Q: Will reviewers think I'm "overselling" the impact?**  
A: Not if you ground claims in evidence:
- "84% cost reduction" → directly from Table 7
- "Students afford 6× more queries" → arithmetic on \$3.50 vs \$21.90
- "Enables 10M query deployments" → enterprise scenario with real pricing

---

## Alternative: Minimal Restructuring (If Page Constraints Prohibit Use Cases Section)

If you cannot fit the full Use Cases section, implement this minimal version:

1. **Use revised abstract** (same length as original)
2. **Use revised introduction** (same length, reframed opening)
3. **Add 1 paragraph to Method** (democratization framing)
4. **Add 2 paragraphs to Evaluation** (accessibility implications)
5. **Use revised conclusion** (elevate Broader Impact from 1 → 3 paragraphs)

**Net page change:** +0.25 pages (achievable via minor compressions)  
**Narrative impact:** ~60% of full restructuring

---

## Next Steps

1. **Review revised files:**
   - `abstract_REVISED.tex`
   - `introduction_REVISED.tex`
   - `use_cases.tex`
   - `conclusion_REVISED.tex`

2. **Choose restructuring approach:**
   - Full restructuring (+1 page, requires compression)
   - Minimal restructuring (+0.25 pages, easier integration)

3. **Test compilation** to verify page count

4. **Iterate on use cases** if you have domain-specific examples that resonate more strongly

5. **Update figures** if needed (e.g., annotate Figure 4 with "Student Budget Threshold" line at \$0.70/1k)

---

## Optional Enhancements

### Title Alternatives

Current: "Beyond Fixed Chains: Scalable Predictive Routing with Shippable Priors"

Consider:
- "Democratizing LLM Access Through Adaptive Routing with Shippable Priors"
- "BanditGPT: An Open-Source Framework for Cost-Accessible LLM Deployment"
- "Making AI Affordable: Adaptive Model Routing for Cost-Constrained Users"

### Figure Annotations

**Figure 4 (Pareto Frontier):** Add annotations:
- Arrow at \$0.70/1k: "Student/Researcher Budget Threshold"
- Arrow at \$1.34/1k: "Startup Production Threshold"
- Shaded region above \$4.38/1k: "Frontier-Only (Prohibitive)"

**Figure 1 (Regret Curve):** Update caption:
```latex
\caption{\textbf{Cold-Start Elimination.} Shippable priors reduce day-one regret 
by 64.6\%, eliminating the \$50–200 exploration barrier that prices out 
resource-constrained users.}
```

### Keywords Update

Current: `LLM Routing, Contextual Bandits, Warm-Start Learning, Expert Distillation, Cost Optimization`

Consider adding: `Accessible AI, Democratization, Production ML Systems`

---

## Final Recommendation

**Implement the full restructuring.** The democratization message is powerful and aligns perfectly with the Applied Data Science track's emphasis on real-world impact. The technical rigor remains unchanged—you're simply giving reviewers and readers the **correct lens** through which to evaluate your contribution.

Your work isn't just "a better routing algorithm." It's **infrastructure for equitable AI access**. Make sure the paper tells that story from the first sentence.

