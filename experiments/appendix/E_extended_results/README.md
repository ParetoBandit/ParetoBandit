# Appendix E: Extended Experimental Results

## Overview
Catastrophic failure detection experiment (Figure 6), demonstrating Corralling's safety benefit on a 5-model portfolio with production router components.

## Contents

### E.1: Catastrophic Failure Detection
**Canonical source**: `../E_catastrophic_failure_experiment/figure6_corralling_kdd.tex`

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

## Removed Content

| Item | Reason |
|------|--------|
| ~~E2: Three-Model Routing~~ | Experiment removed from scope |
| ~~E3: Alternative Cost Profiles~~ | Never created; Figure 4 Pareto sweep covers cost-quality tradeoffs |
| ~~E4: Distribution Shift Robustness~~ | Never created; B.2 covers cross-domain transfer |

---

## Related Sections
- **Appendix D.2**: Learning rate ablation under catastrophic failure (justifies $\eta=0.3$)
- **Main Paper Figure 3**: Corralling insurance mechanism — Figure 6 extends this to catastrophic scenario
- **Main Paper Figure 4**: Pareto frontier — static benchmarks; Figure 6 tests dynamic challenge

---

## Files
```
E_extended_results/
├── README.md                              (this file)
├── E1_catastrophic_failure.tex            (included in APPENDIX_MASTER)
└── E1_catastrophic_failure_extended.tex   (optional extended analysis)
```

The canonical experiment code and figures live in `../E_catastrophic_failure_experiment/`.

---

**Last Updated**: February 15, 2026
