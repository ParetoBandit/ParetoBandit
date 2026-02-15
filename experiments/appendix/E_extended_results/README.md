# Appendix E: Extended Experimental Results

## Overview
Additional experimental results, extended analyses, and supplementary evaluations that complement the main paper findings.

## Contents

### E.1: Catastrophic Failure Detection
**Files**: 
- `E1_catastrophic_failure.tex` (source: `06_figure/figure5_corralling_kdd.tex`)
- `E1_catastrophic_failure_extended.tex` (source: `06_figure/figure6_corralling_kdd.tex`)

**Content**:
- Three-phase catastrophic failure scenario
- Corralling as safety mechanism for fast automatic failover
- Detection of large, sudden quality drops (d > 1.0)

**Scenario Design**:
- **Phase 1 (t=0-100)**: Both models healthy (Mixtral & GPT-4: μ=0.80, σ=0.08)
- **Phase 2 (t=100-200)**: GPT-4 catastrophic failure (μ drops to 0.20)
- **Phase 3 (t=200-300)**: GPT-4 recovery (μ returns to 0.80)

**Key Insights**:
- Use Corralling for safety-critical failure detection (d > 1.0)
- NOT for subtle quality optimization (d < 0.2) - offline A/B testing better
- Fast weight rebalancing enables automatic failover
- System maintains performance during model degradation

**Supplementary Materials**:
- Additional catastrophic failure experiments in `06_figure/supplementary/`
- Learning rate ablation under catastrophic failure
- Realistic failure scenario testing
- Multi-seed validation

---

### E.2: Three-Model Routing Results
**Source**: `04_figure/results_3models/`

**Content**:
- Extended evaluation with 3 models (Mixtral, GPT-4-Turbo, GPT-4o)
- Demonstrates multi-model routing capabilities
- Semantic transfer initialization for GPT-4o

**Key Results**:
- GPT-4o emerges as dominant choice (70.8% usage)
- System correctly learns model preferences from data
- Semantic transfer enables zero-shot adoption

---

### E.3: Alternative Cost Profiles
**Content**:
- Extended cost-quality trade-off analysis
- Multiple cost budget scenarios
- Pareto frontier variations

**Profiles Evaluated**:
- Max Quality: High-cost, high-quality routing
- Arbitrage: Balanced cost-quality
- Best Value: Cost-minimizing strategies

---

### E.4: Distribution Shift Robustness
**Source**: `02_figure/` and `02_table/` extended analyses

**Content**:
- Mismatch robustness analysis
- Performance under distribution shift
- Covariate shift between training and deployment

**Key Metrics**:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test results
- Bootstrap confidence intervals
- Task difficulty clustering on reward gaps

---

## Figures

### Catastrophic Failure Figures
- Three-phase failure scenario visualization
- Weight evolution during failure and recovery
- Performance comparison: with/without Corralling

### Multi-Model Results
- Usage distribution across 3 models
- Cumulative regret comparison
- Learning curves for each model

### Cost Profile Analysis
- Extended Pareto frontier plots
- Cost vs. quality scatter plots
- Budget constraint analysis

---

## Key Takeaways

### When to Use Corralling
✅ **DO USE** for:
- Catastrophic failure detection (effect size d > 1.0)
- Fast automatic failover requirements
- Safety-critical applications
- Sudden quality drops or model crashes

❌ **DON'T USE** for:
- Subtle quality optimization (d < 0.2)
- Stable model performance
- Offline A/B testing scenarios
- When deployment latency is not critical

### Multi-Model Routing
- System scales beyond 2 models effectively
- Semantic transfer enables rapid new model adoption
- Corralling meta-algorithm handles heterogeneous expert ensembles
- No performance degradation with increased model count

### Economic Validation
- Intelligent routing delivers GPT-4 quality at 50% cost
- Pareto frontier demonstrates clear value proposition
- Multiple cost profiles accommodate different business needs

---

## Related Sections
- **Main Paper Figure 5**: Pareto frontier core results
- **Main Paper Figure 4**: Multi-model routing with semantic transfer
- **Appendix D**: Extended analysis and operating regimes
- **Appendix F**: Implementation details for production deployment

---

## Files
```
E_extended_results/
├── README.md                              (this file)
├── E1_catastrophic_failure.tex           (failure detection)
├── E1_catastrophic_failure_extended.tex  (extended analysis)
├── E2_three_model_routing.tex            (to be created)
├── E3_cost_profiles.tex                  (to be created)
├── E4_distribution_shift.tex             (to be created)
└── figures/
    ├── (catastrophic failure figures)
    ├── (multi-model routing figures)
    ├── (cost profile figures)
    └── (distribution shift figures)
```
