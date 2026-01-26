# LaTeX Package Summary: Distribution Shift Analysis

## What You Have

A complete, KDD-compliant LaTeX package for the distribution shift analysis, ready to integrate into your paper.

## Files Created

### 1. `figure_distribution_shift.tex` (Main LaTeX Content)
**Purpose**: Complete subsection with mathematical rigor and empirical analysis  
**Length**: ~750 words, ~1.5 pages  
**Contains**:
- Formal PSI definition with proper notation (P/Q for source/target)
- PSI interpretation with industry-standard thresholds
- Empirical findings with detailed statistics
- Domain mismatch table comparing prior expectations vs observed rewards
- Three-part implications section (miscalibration, unknown distribution, hybrid robustness)
- Comparison to related work (RouteLLM, FrugalGPT)
- Summary with key narrative points

**Key Features**:
- Uses your formal PSI equation: `PSI = sum((P_i - Q_i) * ln(P_i/Q_i))`
- Emphasizes PSI = 0.275 > 0.25 threshold
- Includes "negative transfer" narrative
- References 1.26× near-optimal recovery
- Connects to corralling meta-weight volatility

### 2. `PAPER_INTEGRATION.md` (Integration Guide)
**Purpose**: How to use the LaTeX in your paper  
**Contains**:
- Where to place the content (Section 3 vs Section 5 vs Appendix)
- Dependencies (figures, tables, citations, cross-references)
- Alternative versions (full, short paragraph, one-sentence caption)
- Integration steps with specific commands
- Key messages for different paper sections
- Common reviewer questions with prepared answers

### 3. `KEY_NUMBERS.md` (Quick Reference)
**Purpose**: All numbers to copy-paste when writing  
**Contains**:
- PSI = 0.275, mean shift = -0.064, task percentages (45.4%, 22.4%)
- Domain mismatch table (GPT-4: -10.6%, Mixtral: +80.0%)
- How to use numbers in abstract/intro/results/discussion
- Common mistakes to avoid
- Connecting to other results (corralling, cold-start, ablations)
- Complete narrative arc
- Data provenance for reproducibility

### 4. `CITATIONS.bib` (Bibliography Entries)
**Purpose**: All citations needed for the section  
**Contains**:
- Core references: Shimodaira (covariate shift), Yurdakul (PSI thresholds)
- Baseline comparisons: RouteLLM, FrugalGPT
- Supporting work: Lu et al. (concept drift), importance weighting
- Optional references: bandit algorithms, embeddings, production ML monitoring

### 5. `HYBRID_CONNECTION.md` (Conceptual Guide)
**Purpose**: Connects distribution shift to hybrid bandit solution  
**Contains**:
- Three-act structure: Problem → Unknown Distribution → Solution
- How hybrid addresses each challenge
- Performance under shift (robustness analysis)
- Timeline of adaptation (prior weight decay)
- Writing the narrative (opening hook → conclusion)
- Common pitfalls to avoid

### 6. `EXPERIMENT_SUMMARY.md` (Already Existed)
**Purpose**: Quick reference for the experiment itself  
**Updated**: Now references the formal LaTeX package

### 7. `README.md` (Already Existed)
**Purpose**: How to run the experiment and reproduce results

## How to Integrate into Your Paper

### Step 1: Copy Figure
```bash
cp experiments_v1/01.5_figure/results/distribution_shift_pc1.png paper/figures/
```

### Step 2: Add Content to Paper
Open your paper's main `.tex` file and insert the content from `figure_distribution_shift.tex` into the appropriate section (we recommend Section 3 or early Section 4).

### Step 3: Update References
Replace these placeholders with your actual section/equation numbers:
- `\ref{sec:hybrid_bandit}` → Your hybrid method section number
- `\ref{eq:hybrid_ucb}` → Your hybrid UCB equation number
- `\ref{fig:corralling_weights}` → Your corralling weights figure (from experiment 02)

### Step 4: Add Citations
Copy the relevant entries from `CITATIONS.bib` to your paper's `.bib` file. Minimum required:
- `shimodaira2000improving` (covariate shift)
- `yurdakul2018statistical` (PSI thresholds)
- `lu2018learning` (concept drift)
- `ong2024routellm` (baseline)
- `chen2024frugalgpt` (baseline)

### Step 5: Compile and Check
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## What Makes This KDD-Compliant

✅ **Formal mathematical notation**: PSI defined with proper equation environment  
✅ **Industry-standard thresholds**: Cites Yurdakul (2018) for PSI interpretation  
✅ **Empirical validation**: Table and figure support claims  
✅ **Related work comparison**: Contrasts with RouteLLM and FrugalGPT  
✅ **Reproducibility**: Data sources and methodology clearly stated  
✅ **Production relevance**: Emphasizes real-world deployment scenarios  
✅ **Quantitative results**: All claims backed by specific numbers (PSI = 0.275, 80% increase, etc.)  

## Quick Test: Does It Answer Reviewer Questions?

**Q1: "Why should I care about distribution shift?"**  
✅ Answered: PSI = 0.275 > 0.25 threshold proves it's substantial, not hypothetical

**Q2: "How do you know the shift is significant?"**  
✅ Answered: Industry-standard PSI metric with established thresholds (Yurdakul 2018)

**Q3: "What's the practical impact?"**  
✅ Answered: 80% reward discrepancy for Mixtral, over-routing to expensive model

**Q4: "Why not just retrain on new data?"**  
✅ Answered: Distribution evolves continuously; hybrid adapts automatically

**Q5: "Does your method actually handle this?"**  
✅ Answered: 1.26× near-optimal recovery despite PSI = 0.275

**Q6: "How does this compare to prior work?"**  
✅ Answered: RouteLLM/FrugalGPT don't analyze shift or provide adaptation

## Customization Options

### For Space-Constrained Venues

If you have strict page limits, consider:

1. **Move to Appendix**: Put full analysis in appendix, use short paragraph in main text
2. **Combine Figure+Table**: Create single two-panel figure with distribution + table
3. **Shorten Implications**: Keep only "Hybrid Bandits Provide Robust Adaptation" paragraph
4. **Remove Related Work**: If you have a separate related work section

### For Extended Version

If you have extra space (e.g., journal version), consider adding:

1. **Ablation**: Show performance under varying PSI levels
2. **Temporal Analysis**: PSI evolution over time
3. **Per-Model Analysis**: Distribution shift for each model type
4. **Alternative Metrics**: Jensen-Shannon divergence, KL divergence comparison

## Key Messages by Section

### Abstract (1 sentence)
> "We quantify substantial distribution shift (PSI = 0.275) between training and deployment, motivating our adaptive hybrid approach that achieves 1.26× near-optimal performance despite this mismatch."

### Introduction (1 paragraph)
> "Production LLM routing faces a fundamental challenge: deployment distributions often differ from training distributions. We measure PSI = 0.275 in our setting—exceeding the 0.25 threshold for substantial shift—indicating warmup priors trained on historical data will be miscalibrated. Our hybrid bandit framework addresses this by combining informative priors for cold-start with continuous adaptation for long-term robustness."

### Related Work (comparison)
> "Unlike RouteLLM [X] and FrugalGPT [Y], which assume deployment matches training, we explicitly quantify and adapt to distribution shift. Our PSI = 0.275 demonstrates this is not hypothetical: even with representative training data, substantial shift occurs in practice."

### Method (motivation)
> "Our hybrid formulation (Equation X) is motivated by empirical distribution shift (Section Y): warmup priors provide good initialization despite miscalibration, while bandit updates correct for deployment mismatch over time."

### Results (robustness)
> "Despite PSI = 0.275 indicating substantial shift—manifesting as 80% reward discrepancy for Mixtral (Table X)—our hybrid approach achieves 1.26× near-optimal performance, demonstrating robustness to domain mismatch."

### Discussion (impact)
> "The PSI = 0.275 we observe is representative of real-world deployment: evolving user behavior, seasonal effects, and new use cases cause continuous drift. Our adaptive framework provides robustness without requiring manual retraining triggers or distribution monitoring thresholds."

## Common Writing Patterns

### Introducing the Problem
```latex
Real-world ML systems must handle \emph{covariate shift}~\cite{shimodaira2000improving},
where deployment distributions $Q(x)$ differ from training distributions $P(x)$.
```

### Stating the Finding
```latex
We measure PSI = 0.275 (Section~\ref{sec:distribution_shift}), exceeding the 0.25 
threshold for substantial shift~\cite{yurdakul2018statistical}.
```

### Showing Impact
```latex
This manifests as an 80\% discrepancy in Mixtral's utility (Table~\ref{tab:domain_mismatch}), 
demonstrating that deployment prompts are substantially less complex than training data.
```

### Presenting Solution
```latex
Our hybrid approach achieves 1.26$\times$ near-optimal performance despite this mismatch
by automatically down-weighting miscalibrated priors through importance-weighted losses.
```

## Timeline for Paper Writing

**Week 1**: 
- Copy LaTeX content into paper
- Update figure paths and cross-references
- Add citations to bibliography

**Week 2**:
- Write transition paragraphs to/from this section
- Ensure consistency with other experiments
- Add forward references in introduction

**Week 3**:
- Polish language and notation
- Get co-author feedback
- Prepare rebuttal materials

**Week 4**:
- Final pass for consistency
- Check all numbers match KEY_NUMBERS.md
- Verify all citations compile

## Contact Points with Other Experiments

### Experiment 01 (PCA Analysis)
- Uses same PCA model and PC1 projection
- Validates that PC1 captures difficulty gradient
- Supports using PCA features for routing

### Experiment 02 (Corralling)
- PSI = 0.275 explains meta-weight volatility in Figure 5
- Down-weighting of Warmup Expert is justified
- 1.26× recovery connects to corralling performance

### Experiment 04 (Cold-Start)
- Miscalibrated priors still better than random initialization
- Demonstrates value of hybrid despite PSI = 0.275
- Explains why pure bandit has slow convergence

### Experiment 05 (Hybrid Comparison)
- PSI analysis motivates hybrid formulation
- Explains why prior-only baseline fails
- Validates importance-weighted adaptation

## Success Metrics

Your LaTeX package is successful if reviewers say:

✅ "The distribution shift analysis is rigorous and well-motivated"  
✅ "The PSI metric provides clear evidence for the problem"  
✅ "The connection to the hybrid solution is compelling"  
✅ "The empirical findings support the claims"  
✅ "The production relevance is clear"  

## Final Checklist

Before submitting your paper:

- [ ] Figure paths are correct
- [ ] All cross-references resolve (`\ref{...}` commands)
- [ ] All citations compile (no missing .bib entries)
- [ ] Numbers match KEY_NUMBERS.md exactly
- [ ] Table formatting matches your paper style
- [ ] Equation numbering is consistent
- [ ] Caption length is appropriate for venue
- [ ] Narrative connects to other sections smoothly
- [ ] PSI interpretation is clearly stated (≥ 0.25 = substantial)
- [ ] Negative transfer concept is explained
- [ ] 1.26× recovery is mentioned and contextualized

## Getting Help

If reviewers ask questions, consult:

- **Technical details**: `README.md` (how experiment was run)
- **Key numbers**: `KEY_NUMBERS.md` (exact values to cite)
- **Narrative**: `HYBRID_CONNECTION.md` (conceptual framing)
- **Rebuttals**: `PAPER_INTEGRATION.md` → "Common Reviewer Questions"

## You're Ready!

You now have everything needed to integrate distribution shift analysis into your KDD paper:
- ✅ Formal LaTeX with proper notation
- ✅ Empirical results with clear visualization
- ✅ Connection to hybrid solution
- ✅ Comparison to baselines
- ✅ Production relevance
- ✅ All citations and references

Go write that paper! 📝

