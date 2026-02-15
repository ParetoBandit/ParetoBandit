# Appendix G: Limitations and Future Work

## Overview
System limitations, assumptions, and future research directions. Required for any serious venue submission.

## Contents

### G.1: Limitations
**File**: `G1_limitations.tex`  
**Source**: `03_figure/latex_section_6_limitations.tex`

**Content**:
- Data requirements (labeled warmup data, embedding stability)
- Computational overhead (real-time embeddings, linear algebra per decision)
- Model assumptions (reward stationarity or slow drift)
- Deployment constraints (continuous feedback loop needed)

### G.1 Addendum
**File**: `G1_limitations_addendum.tex`  
**Source**: `08_figure/limitations_addendum.tex`

**Content**:
- Additional limitations discovered during extended experiments
- Merge into G1_limitations.tex for camera-ready

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~G1: Practical Deployment Recommendations~~ | Duplicates Figure 3 strategy guide and F3; practitioner guidance belongs in GitHub README |
| ~~G3: Broader Impact~~ | Never created; only required if venue mandates it (e.g., NeurIPS) |
| ~~G4: Corralling vs Offline Optimization~~ | Never created; not required by Figures 1-4, 6 |

---

## Related Sections
- **Appendix C**: Hyperparameter sensitivity addresses the "brittle parameters" limitation
- **Appendix D**: Ablation studies address the "specific configuration" limitation
- **Figure 6**: Catastrophic failure addresses the "stationarity assumption" limitation

---

## Files
```
G_additional_discussion/
├── README.md                          (this file)
├── G1_limitations.tex                 (core limitations)
└── G1_limitations_addendum.tex        (additional limitations — merge for camera-ready)
```

---

**Last Updated**: February 15, 2026
