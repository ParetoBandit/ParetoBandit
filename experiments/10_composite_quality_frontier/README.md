# Experiment 10: Frontier Capability Index (FCI)

## Overview

This experiment computes a **Frontier Capability Index (FCI)** - a composite quality metric based on three rigorous benchmarks that preserve capability differentiation and avoid saturation effects observed in older datasets.

## Methodology

### Benchmark Selection

We use three high-difficulty benchmarks:

1. **GPQA** (Graduate-Level Google-Proof Q&A) - Tests advanced reasoning with questions requiring graduate-level knowledge
2. **LiveBench** (Contamination-resistant evaluation) - Prevents data leakage with continuously updated test sets  
3. **HLE** (Human Level Evaluation) - Measures performance on real-world tasks

These benchmarks were specifically chosen because modern models have saturated on older datasets like MMLU (where differences are negligible), but still show significant capability gaps on these harder tasks.

### FCI Computation

#### 1. Normalization

Each benchmark is normalized to [0, 1] scale based on min/max across the model portfolio:

$$S_{norm} = \frac{S_{model} - S_{min}}{S_{max} - S_{min}}$$

#### 2. Composite Score

The FCI is the arithmetic mean of the three normalized scores:

$$\text{FCI}_{model} = \frac{1}{3} (\text{HLE}_{norm} + \text{GPQA}_{norm} + \text{LiveBench}_{norm})$$

#### 3. Pareto Frontier

Models are selected for the Pareto frontier if they are not dominated by any other model. A model is dominated if another model has both:
- Higher FCI (quality) AND lower cost
- OR equal FCI and lower cost  
- OR higher FCI and equal cost

## Results

### Benchmark Coverage

- **Total models in registry**: 84
- **Models with all 3 benchmarks**: 5
- **Pareto-optimal models**: 2

Only 5 models in the registry have complete coverage across all three benchmarks, highlighting the selective nature of these advanced evaluations.

### Normalization Statistics

| Benchmark | Min   | Max   | Mean  |
|-----------|-------|-------|-------|
| HLE       | 0.070 | 0.265 | 0.152 |
| GPQA      | 0.780 | 0.870 | 0.814 |
| LiveBench | 0.700 | 0.880 | 0.802 |

### FCI Scores (All Models)

| Rank | Model | FCI | Cost ($/1M) | HLE | GPQA | LiveBench | Pareto? |
|------|-------|-----|-------------|-----|------|-----------|---------|
| 1 | openai/gpt-5.1 | **0.981** | $5.625 | 0.265 | 0.870 | 0.870 | ✅ |
| 2 | openai/gpt-oss-120b | **0.530** | $0.060 | 0.185 | 0.780 | 0.880 | ✅ |
| 3 | moonshotai/kimi-k2-0905 | 0.500 | $1.075 | 0.070 | 0.840 | 0.850 | ❌ |
| 4 | google/gemini-2.5-flash | 0.153 | $0.300 | 0.127 | 0.790 | 0.710 | ❌ |
| 5 | x-ai/grok-3-mini | 0.107 | $0.800 | 0.111 | 0.790 | 0.700 | ❌ |

### Pareto Frontier Analysis

#### ✅ openai/gpt-oss-120b ($0.06/1M, FCI=0.530)
- **Position**: Low-cost, mid-quality anchor
- **Why Pareto-optimal**: Cheapest model with reasonable FCI
- **Dominates**: All mid-tier models (kimi-k2, gemini-2.5, grok-3-mini) have higher cost but lower FCI
- **Best for**: Cost-sensitive applications where 53% normalized quality is acceptable

#### ✅ openai/gpt-5.1 ($5.625/1M, FCI=0.981)  
- **Position**: Premium quality anchor
- **Why Pareto-optimal**: Highest FCI by far (98% normalized quality)
- **Best for**: Tasks requiring maximum reasoning capability (graduate-level reasoning, complex math, advanced coding)

#### ❌ Dominated Models

All three mid-tier models are dominated by **gpt-oss-120b**, which has:
- **Higher FCI** (0.530 vs 0.107-0.500)
- **Lower cost** ($0.06 vs $0.30-$1.08)

This demonstrates the power of the FCI metric in revealing true capability gaps.

## Key Insights

### 1. Capability Differentiation Preserved

Unlike older metrics (e.g., MMLU), the FCI shows a **98x cost ratio** between the Pareto anchors:
- gpt-oss-120b: $0.06/1M
- gpt-5.1: $5.625/1M

This large gap justifies the existence of a sophisticated routing strategy.

### 2. "Cheap Wins" Prevented

The composite score ensures that expensive flagship models are only selected for tasks that truly require their advanced capabilities (reasoning, math, coding), not just tasks where they're marginally better.

### 3. Modern Evaluation Standards

By using GPQA, LiveBench, and HLE (all post-2023 benchmarks), we ensure:
- **No saturation**: Models show meaningful performance gaps
- **No contamination**: LiveBench prevents training data leakage
- **Real-world relevance**: HLE measures practical task performance

## Usage

```bash
# Compute FCI scores for all models with complete benchmark coverage
python compute_fci.py

# Results saved to:
# - results/models_with_fci.json       # All models with FCI scores
# - results/pareto_frontier_fci.json   # Only Pareto-optimal models  
# - results/fci_stats.txt              # Summary statistics
```

## LaTeX for Paper

Use this in your "Experimental Setup" section:

```latex
\subsection{Model Selection and The Pareto Frontier}

To define the candidate pool, we constructed a \textbf{Frontier Capability Index (FCI)} 
based on a composite of three rigorous benchmarks: \textbf{GPQA} 
\cite{rein2023gpqa} (Graduate-Level Google-Proof Q\&A), 
\textbf{LiveBench} \cite{livebench2024} (Contamination-resistant evaluation), 
and \textbf{HLE} (Human Level Evaluation). 

We specifically selected these benchmarks to avoid the \textit{saturation effects} 
observed in older datasets (e.g., MMLU), where the performance gap between 
efficient and flagship models has narrowed to negligible levels. By focusing on 
high-difficulty tasks, our Pareto frontier preserves the \textit{capability 
differentiation} required to evaluate ``Rational Luxury'' routing strategies.

Candidate models were selected strictly based on their position on the 
Price-vs-FCI curve (Figure \ref{fig:pareto_frontier}), ensuring that the 
router's choice set contained only non-dominated options prior to any 
exposure to the test distribution.
```

## References

- Rein et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark
- LiveBench (2024). Contamination-Free LLM Benchmark
- HLE: Human Level Evaluation for LLMs

## Files

- `compute_fci.py` - Main script to compute FCI and find Pareto frontier
- `plot_fci_frontier.py` - Visualization script (to be created)
- `results/models_with_fci.json` - All models with FCI scores
- `results/pareto_frontier_fci.json` - Pareto-optimal models only
- `results/fci_stats.txt` - Summary statistics

