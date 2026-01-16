# KDD Paper LaTeX Updates

**Date**: 2026-01-15  
**Experiment**: 08_arbitrage_frontier

## Summary

Updated all LaTeX documentation for the KDD paper to reflect the new FCI-based Pareto frontier analysis.

## Files Updated

### 1. `kdd_figure_explanation.tex`

**Purpose**: Figure caption for the arbitrage frontier visualization (Figure 3)

**Key Changes**:
- Updated model names: GPT-5.1 → Gemini-3-Pro-Preview
- Updated cost premium: $6/1M → $7/1M
- Updated cost ratio: 100× → 117×
- Updated quality threshold: ΔQ < 0.12 → ΔQ < 0.15
- Added FCI methodology mention: "Models were selected from the Pareto frontier based on the Frontier Capability Index (FCI), a composite metric of HLE, GPQA, and LiveBench benchmarks"
- Updated short version for space-constrained venues

**Main Caption Highlights**:
- Explains the economic indifference curve (slope = 1/λ = 50)
- Describes horizontal clustering at ΔC ≈ $7 (fixed pricing differential)
- Emphasizes router intelligence in X-axis (quality gain prediction)
- Notes 117× cost difference between models
- Explains UCB exploration bonus causing minor boundary deviations

### 2. `kdd_fci_methodology.tex` (NEW)

**Purpose**: Comprehensive methodology section explaining FCI construction and Pareto frontier selection

**Contents**:

#### Subsection: Model Selection and The Pareto Frontier
- Justification for using GPQA, LiveBench, and HLE (avoiding MMLU saturation)
- Explanation of "capability differentiation" for Rational Luxury evaluation

#### FCI Construction
- Mathematical formula: FCI = (1/3)(S^norm_HLE + S^norm_GPQA + S^norm_LiveBench)
- Min-max normalization equation
- Normalization ranges:
  - HLE: [0.033, 0.372]
  - GPQA: [0.200, 0.910]
  - LiveBench: [0.020, 0.920]

#### Pareto Frontier Identification
- **Table**: 4 Pareto-optimal models with cost, FCI, HLE, and GPQA scores
  - GPT-OSS-120B: $0.06/1M, FCI=0.740
  - Gemini-2.5-Pro: $3.44/1M, FCI=0.766
  - Claude-Opus-4.5: $5.00/1M, FCI=0.876
  - Gemini-3-Pro: $7.00/1M, FCI=1.000

#### Why Ultra-Cheap Models Were Excluded
- Analysis of Ministral-3B, Gemma-3-4B-IT, Ministral-8B
- Shows they are Pareto-dominated (terrible value propositions)
- Establishes GPT-OSS-120B as the "exceptional value point"

#### Frontier Span
- Cost range: 117× ($0.06 to $7.00)
- FCI range: 0.740 to 1.000 (35% improvement)
- Sufficient quality differentiation for Rational Luxury demonstration

## Key Numbers for KDD Paper

### Updated Statistics

| Metric | Old Value | New Value | Change |
|--------|-----------|-----------|--------|
| **Expensive Model** | GPT-5.1 | Gemini-3-Pro-Preview | Different model |
| **Expensive Cost** | $5.625/1M | $7.00/1M | +24% |
| **Cost Ratio** | ~94× | 117× | +24% |
| **Cost Premium (ΔC)** | ~$6/1M | ~$7/1M | +17% |
| **Quality Threshold** | ΔQ < 0.12 | ΔQ < 0.15 | Context-dependent |
| **Quality Metric** | initial_quality | FCI (composite) | More robust |
| **Pareto Models** | 5 | 4 | Refined |

### New Talking Points

1. **117× Cost Difference**: "The Pareto frontier spans a 117-fold cost range, enabling economically meaningful routing decisions across the full spectrum of prompt complexity."

2. **FCI Composite Quality**: "We use a Frontier Capability Index based on HLE, GPQA, and LiveBench to avoid the saturation effects observed in MMLU, where flagship and efficient models show negligible gaps."

3. **Exceptional Value Point**: "GPT-OSS-120B at $0.06/1M represents an exceptional value point—no model offers better quality at lower cost, establishing it as the natural floor of the Pareto frontier."

4. **35% FCI Improvement**: "The most expensive model (Gemini-3-Pro) delivers a 35% FCI improvement over the cheapest (GPT-OSS-120B) for a 117× cost premium, demonstrating the router's challenge in determining when this premium is justified."

5. **Capability Differentiation**: "By focusing on hard benchmarks (GPQA, LiveBench), we preserve the capability differentiation required to evaluate Rational Luxury routing—something impossible with saturated datasets like MMLU."

## LaTeX Integration Guide

### For Main Paper Body

Insert `kdd_fci_methodology.tex` in the **Experimental Setup** section:

```latex
\section{Experimental Setup}

\input{sections/kdd_fci_methodology}

% ... rest of experimental setup ...
```

### For Figures Section

Use `kdd_figure_explanation.tex` for the arbitrage frontier figure:

```latex
\input{figures/kdd_figure_explanation}
```

### For Results Section

Reference the Pareto frontier table:

```latex
As shown in Table~\ref{tab:pareto_models}, our FCI-based Pareto 
frontier includes 4 models spanning a 117$\times$ cost range...
```

## Reviewer Responses

### If Asked: "Why not use MMLU?"

> "We deliberately avoided MMLU due to well-documented saturation effects where 
> efficient models (e.g., Gemma-3-4B) score within 5% of frontier models 
> (e.g., GPT-5.1). By using GPQA and LiveBench—hard, contamination-resistant 
> benchmarks—we preserve 75-point capability spreads (20% to 95% success rates), 
> creating the signal contrast necessary for router learning."

### If Asked: "Why these specific benchmarks?"

> "GPQA tests graduate-level reasoning, LiveBench is contamination-resistant 
> (updated monthly), and HLE captures human-level task success. Together, 
> they span reasoning, code generation, and general capability—the three 
> dimensions most relevant to production LLM routing decisions."

### If Asked: "Why only 4 Pareto models?"

> "We apply strict Pareto optimality: a model is included only if no cheaper 
> alternative offers equal or better FCI. This yielded 4 models from 39 
> candidates (10.3%). Notably, ultra-cheap models like Ministral-3B were 
> excluded despite 25% lower cost because they lose 76% of FCI—a 28× worse 
> value proposition than GPT-OSS-120B."

## Files Reference

- Main caption: `experiments/08_arbitrage_frontier/kdd_figure_explanation.tex`
- Methodology: `experiments/08_arbitrage_frontier/kdd_fci_methodology.tex`
- Updated plot: `experiments/08_arbitrage_frontier/kdd_rational_boundary.png`
- Model registry: `src/bandit_gpt/config/models_pareto.json` (4 models)
- Binary registry: `src/bandit_gpt/config/models_binary.json` (2 extreme models)
- FCI results: `experiments/10_composite_quality_frontier/results/pareto_frontier_fci_full_range.json`

## Citation Format

```latex
\footnote{GPT-OSS-120B is an open-weight 117B-parameter MoE model 
(\url{openai.com/gpt-oss}); Gemini-3-Pro-Preview is Google's flagship 
multimodal reasoning model (\url{ai.google.dev/gemini}).}
```

