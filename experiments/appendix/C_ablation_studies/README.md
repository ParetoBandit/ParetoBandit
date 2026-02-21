# [Consolidated] Former Appendix C: Ablation Studies

**Status**: Consolidated — no longer a standalone appendix section.

## What happened

The Corralling ablation content (45-experiment grid search over η and γ) was
folded into **Appendix A.2** (Safety Guarantee via γ-Mixing), which is its
natural theoretical home.  The key content that was moved:

| Content | New location |
|---------|-------------|
| 45-experiment grid search over η and γ | A.2, ablation table |
| Sublinear regret validation (β=0.669) | A.2, empirical validation |
| Exploration floor necessity (14× variance reduction) | A.2, γ-mixing analysis |
| Learning rate regime framework (η=0.3/1.0/5.0) | A.2, three-regime discussion |
| α/γ parameter sweep figures | Archived in this directory |

## Why

- The ablation table directly validates the γ-mixing safety guarantee proved in
  A.2, making it the natural co-location.
- Keeping the ablation as a separate appendix section added navigation overhead
  without adding content value.
- Removing C as a standalone section tightens the appendix from 6 to 5 sections (A-E).

## Files

The original `C1_corralling_ablation.tex` and figures are retained in this
directory and git history for reference, but are no longer `\input` from
`APPENDIX_MASTER.tex`.

```
C_ablation_studies/
├── README.md                          (this file)
├── C1_corralling_ablation.tex         (retained, not compiled)
└── figures/
    ├── figure6_learning_rate_ablation.pdf  (retained, not referenced)
    ├── figure_alpha_ablation.png           (retained, not referenced)
    └── figure_gamma_ablation.png           (retained, not referenced)
```

---

**Last Updated**: February 15, 2026
