# Appendix E: Extended Experimental Results

## Overview
Catastrophic failure detection experiment (Figure 6), demonstrating Corralling's safety benefit on a 5-model portfolio with production router components.

## Contents

### E.1: Catastrophic Failure Detection
**File**: `E1_catastrophic_failure.tex`  
**Source**: K=5 production router experiment (`../E_catastrophic_failure_experiment/generate_figure6_5model.py`)

**Content**:
- 5-model portfolio (Mixtral, GPT-4-Turbo, GPT-3.5, Haiku, GPT-4o)
- Three-phase scenario: healthy -> catastrophic failure -> recovery
- Production router with semantic transfer for all 5 models
- Comparison: banditGPT vs EMA Tracker vs Static vs Oracle

**Key Results** (K=5, N=20 Seeds):
- Detection rate: 95% (19/20 seeds)
- Reaction time: Median 34 steps
- Gap vs EMA narrows with portfolio size: -0.286 (K=2) -> -0.090 (K=5)

**Key Insight**: Use Corralling for safety-critical failure detection (effect size d > 1.0), NOT for subtle quality optimization (d < 0.2).

---

## Related Sections
- **Appendix D.1**: Corralling ablation validates hyperparameter choices
- **Main Paper Figure 3**: Corralling insurance mechanism — Figure 6 extends this to catastrophic scenario
- **Main Paper Figure 4**: Pareto frontier — static benchmarks; Figure 6 tests dynamic challenge

---

## Files
```
E_extended_results/
├── README.md                         (this file)
└── E1_catastrophic_failure.tex       (included in APPENDIX_MASTER)
```

The experiment code lives in `../E_catastrophic_failure_experiment/`.

---

**Last Updated**: February 15, 2026
