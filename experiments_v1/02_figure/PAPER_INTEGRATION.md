# Paper Integration Guide: Distribution Shift Analysis

## Overview

This experiment provides **Figure 1.2** for your KDD paper, demonstrating significant covariate shift between training (warmup prior) and deployment (production) distributions.

## Key Results

- **PSI = 0.2751** (significant shift, > 0.2 threshold)
- **Mean shift = -0.064** (deployment shifted toward easier tasks)
- **Bimodal structure** in training data explained by task difficulty clustering
- **Easy tasks**: 45.4% at PC1 = -0.105
- **Hard tasks**: 22.4% at PC1 = 0.365

## LaTeX Files

### Main File: `figure_distribution_shift.tex`

**Full subsection** (~1.5 pages) with:
- Complete distribution shift analysis
- PSI methodology and interpretation
- Detailed empirical findings
- Implications for routing (3 key points)
- Comparison to related work
- Summary

**Where to use**: Section 3 (Problem Setup) or Section 4 (Experimental Analysis)

**Dependencies**:
- **Figure**: `experiments_v1/01.5_figure/results/distribution_shift_pc1.png`
- **Table**: Domain mismatch comparison (included in LaTeX)
- **Equation**: PSI formula (Equation~\ref{eq:psi})
- **References**: 
  - `Figure~\ref{fig:distribution_shift}`: The distribution shift visualization
  - `Table~\ref{tab:domain_mismatch}`: Prior vs. observed reward comparison
  - `Figure~\ref{fig:corralling_weights}`: Meta-weight evolution (from experiment 02)
  - `Section~\ref{sec:hybrid_bandit}`: Your method section
  - `Equation~\ref{eq:hybrid_ucb}`: Your hybrid UCB formulation
- **Citations needed**:
  - `shimodaira2000improving`: Shimodaira, H. (2000). Improving predictive inference under covariate shift
  - `yurdakul2018statistical`: Yurdakul, B. (2018). Statistical properties of population stability index
  - `lu2018learning`: Lu, J., et al. (2018). Learning under concept drift
  - `ong2024routellm`: RouteLLM paper
  - `chen2024frugalgpt`: FrugalGPT paper

### Key Components in Main File

The full LaTeX includes:

1. **Formal PSI Definition** (Equation~\ref{eq:psi}):
   ```latex
   PSI = \sum_{i=1}^{B} \left( (\%P_i - \%Q_i) \cdot \ln\left(\frac{\%P_i}{\%Q_i}\right) \right)
   ```
   - Uses P (Warmup/Source) and Q (RouteLLM/Target) notation
   - Includes interpretation paragraph with thresholds
   - Emphasizes PSI ≥ 0.25 as "substantial shift"

2. **Domain Mismatch Table** (Table~\ref{tab:domain_mismatch}):
   - Shows GPT-4-Turbo: 0.94 → 0.84 (-10.6%)
   - Shows Mixtral: 0.45 → 0.81 (+80.0%)
   - Caption explains 80% increase indicates less complex target distribution

3. **Bimodal Structure Analysis**:
   - Easy tasks: 45.4% at PC1 = -0.105
   - Hard tasks: 22.4% at PC1 = 0.365
   - Explains shift toward easy cluster

4. **Hybrid Recovery Performance**:
   - States 1.26× near-optimal recovery despite PSI = 0.275
   - Connects to corralling meta-weight volatility
   - Explains automatic miscalibration detection

5. **Negative Transfer Narrative**:
   - "renders warmup bias toward flagship models a source of negative transfer"
   - Fixed priors would over-route to expensive GPT-4-Turbo
   - Cheaper Mixtral suffices for most production queries

### Alternative: Short Version

If you need a shorter version (e.g., for space constraints), use this condensed paragraph:

```latex
\paragraph{Distribution Shift Between Training and Deployment.}
We analyze covariate shift between warmup prior and deployment distributions 
by projecting embeddings onto PC1 and computing Population Stability Index 
(PSI). Figure~\ref{fig:distribution_shift} shows significant shift 
(PSI = 0.275 > 0.2 threshold), with deployment data left-shifted toward 
easier tasks (mean shift = $-0.064$). The training data exhibits bimodal 
structure arising from easy (45.4\%, PC1 = $-0.105$) and hard (22.4\%, 
PC1 = $0.365$) task clusters. This shift demonstrates: (1) warmup priors 
may be miscalibrated, (2) production distributions are often unknown 
\emph{a priori}, and (3) adaptive hybrid bandits that combine priors with 
online learning (Section~\ref{sec:hybrid_bandit}) provide robust performance 
under distribution shift.
```

## Figure Caption Options

### Detailed Caption (Current)

The current caption in `figure_distribution_shift.tex` is comprehensive and explains both panels clearly.

### Shorter Caption

If you need a more concise caption:

```latex
\caption{\textbf{Distribution shift between training and deployment.} 
Significant covariate shift (PSI = 0.275) with training data showing 
bimodal structure (easy and hard task clusters) while deployment data 
is left-shifted toward easier tasks.}
```

### One-Sentence Caption

For extremely tight space:

```latex
\caption{Significant distribution shift (PSI = 0.275) between training 
(blue) and deployment (red) distributions demonstrates the need for 
adaptive routing strategies.}
```

## Integration Steps

1. **Copy figure to paper directory**:
   ```bash
   cp experiments_v1/01.5_figure/results/distribution_shift_pc1.png paper/figures/
   ```

2. **Add LaTeX content**:
   - Copy relevant sections from `figure_distribution_shift.tex`
   - Adjust section numbers and references as needed
   - Update figure path if necessary

3. **Add citations to bibliography**:
   - Add the required citations (see Dependencies above)
   - Ensure citation keys match your .bib file

4. **Cross-reference**:
   - Update references to `Section~\ref{sec:hybrid_bandit}` to match your method section
   - Update references to `Equation~\ref{eq:hybrid_ucb}` if you have this equation

## Key Messages for Paper

### Problem Motivation

> "Production distributions often differ from training distributions (PSI = 0.275), 
> making fixed routing policies suboptimal. Our hybrid approach adapts continuously."

### Method Justification

> "We combine warmup priors with bandit learning because: (1) priors provide 
> good cold-start performance despite miscalibration, and (2) bandit updates 
> correct for distribution shift over time."

### Results Context

> "The 15% cumulative regret reduction (Table X) is achieved despite significant 
> covariate shift, demonstrating robustness of our hybrid approach."

## Related Sections to Update

If you include this distribution shift analysis, consider adding:

1. **Introduction**: Brief mention that distribution shift is a key challenge
   ```latex
   In practice, deployment distributions often differ from training distributions,
   requiring adaptive routing strategies that can correct for covariate shift.
   ```

2. **Problem Setup**: Reference the shift when motivating hybrid approach
   ```latex
   We observe significant distribution shift (Section~\ref{sec:distribution_shift}),
   motivating our hybrid bandit formulation that combines priors with adaptation.
   ```

3. **Experiments**: Use shift to explain why adaptation is necessary
   ```latex
   Despite PSI = 0.275 indicating significant shift, our hybrid approach achieves
   X% lower regret than prior-only baselines by adapting to deployment distribution.
   ```

4. **Discussion**: Highlight robustness to distribution shift
   ```latex
   Our results demonstrate robustness under covariate shift, a critical property
   for production deployment where user distributions evolve over time.
   ```

## Figure Placement Suggestions

### Option 1: Early in Paper (Section 3)
- Motivates the problem early
- Sets up why hybrid approach is needed
- Good for papers emphasizing problem complexity

### Option 2: In Results (Section 5)
- Shows empirical finding
- Explains why your method works
- Good for papers emphasizing experimental insights

### Option 3: In Appendix
- If main paper is space-constrained
- Full analysis available but not in main flow
- Reference from main text: "See Appendix X for distribution shift analysis"

## Statistics to Highlight

For the paper, emphasize:

1. **PSI = 0.275**: Industry-standard threshold is 0.2, so this is significant
2. **Mean shift = -0.064**: About 1/3 of a standard deviation
3. **Bimodal structure**: Shows task heterogeneity, validates using semantic features
4. **45.4% easy vs 22.4% hard**: Quantifies task distribution in training data

## Common Reviewer Questions

Be prepared to address:

**Q: Why is PSI the right metric?**
> PSI is industry-standard for production ML monitoring (credit scoring, fraud detection). 
> It's interpretable, well-calibrated, and has established thresholds.

**Q: How does shift affect your results?**
> Our hybrid approach is robust to shift (achieves X% regret reduction despite PSI = 0.275), 
> while baselines suffer more from miscalibration.

**Q: Could you just retrain on deployment data?**
> Deployment distribution evolves continuously. Our online learning naturally adapts, 
> while periodic retraining requires deciding when/how often to retrain.

**Q: Why not just use more training data?**
> Even with 80K training samples, PSI = 0.275 indicates fundamental distributional 
> difference. More data doesn't eliminate covariate shift.

## Word Count

- **Full section**: ~750 words
- **Short paragraph**: ~150 words
- **Caption**: 50-80 words (depending on version)

Choose based on your paper's space constraints and emphasis.

