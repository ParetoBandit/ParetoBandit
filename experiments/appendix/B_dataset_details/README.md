# Appendix B: Dataset Details

## Overview
Validation methodology and cross-domain transfer analysis supporting Figure 1
(routing signal validation) and Table 2 (data provenance).

## Contents

### B.1: Validation Methodology
**File**: `B1_validation_methodology.tex`

**Content**:
- Spearman rank correlation design (PC1 vs reward gap, N=750)
- Null baseline: 100 random orthonormal projections (QR-decomposed)
- Data independence argument (RouteLLM vs LMSYS provenance)
- Statistical tests: Spearman rho, Mann-Whitney U
- Design rationale: why Spearman instead of clustering
- Stratification sensitivity: existence vs magnitude of signal

**Supports**: Figure 1's core claim (p < 0.0001, 2.6x null median)

### B.2: Cross-Domain Transfer and Feature Pipeline
**File**: `B2_cross_domain_transfer.tex`

**Content**:
- Data provenance table (RouteLLM 80K vs LMSYS 750)
- Why same model pair enables cross-domain transfer
- Complete feature pipeline: 384D -> 32 PCA + 1 bias = 33D
- PCA variance analysis (35.14% at 32 components)
- Why 35% variance is sufficient for routing
- Limitations: two-model topology, near-duplicates, stratification, encoder dependency

**Supports**: Figure 1's cross-domain generalization claim, Table 2 provenance

---

## Related Sections
- **Main Paper Figure 1**: PCA validation (Spearman correlation)
- **Main Paper Table 2**: Dataset composition and splits
- **Appendix A**: Regret bounds reference d=33 feature dimension
- **Appendix D**: Implementation details (production configuration)

---

## Files
```
B_dataset_details/
├── README.md                          (this file)
├── B1_validation_methodology.tex      (Spearman-based validation)
├── B2_cross_domain_transfer.tex       (transfer analysis + feature pipeline)
├── B3_reward_signal.tex               (reward signal and judge model provenance)
└── figures/
```
