# Appendix F: Limitations and Future Work

## Overview
System limitations, assumptions, and future research directions. Required for any serious venue submission.

## Contents

### F.1: Limitations
**File**: `F1_limitations.tex`  
**Source**: `03_figure/latex_section_6_limitations.tex`

**Content**:
- Data requirements (labeled warmup data, embedding stability)
- Computational overhead (real-time embeddings, linear algebra per decision)
- Model assumptions (reward stationarity or slow drift)
- Deployment constraints (continuous feedback loop needed)

### F.1 Addendum
**File**: `F1_limitations_addendum.tex`  
**Source**: Originally from extended experiments analysis

**Content**:
- Additional limitations discovered during extended experiments
- Merge into F1_limitations.tex for camera-ready

---

## Removed Content

| Item | Reason |
|------|--------|
| ~~F2: Practical Deployment Recommendations~~ | Duplicates Figure 3 strategy guide; practitioner guidance belongs in GitHub README |
| ~~F3: Broader Impact~~ | Never created; only required if venue mandates it (e.g., NeurIPS) |
| ~~F4: Corralling vs Offline Optimization~~ | Never created; not required by Figures 1-4, 6 |

---

## Related Sections
- **Appendix A.3**: Prior transfer analysis addresses the "brittle parameters" limitation (n_eff robustness)
- **Appendix C**: Ablation studies address the "specific configuration" limitation
- **Figure 6**: Catastrophic failure addresses the "stationarity assumption" limitation

---

## Files
```
F_limitations_and_future_work/
├── README.md                          (this file)
├── F1_limitations.tex                 (core limitations)
└── F1_limitations_addendum.tex        (additional limitations — merge for camera-ready)
```

---

**Last Updated**: February 15, 2026
