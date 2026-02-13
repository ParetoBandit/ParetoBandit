# Appendix A: Mathematical Foundations

## Overview
This section contains mathematical proofs, theoretical bounds, and formal derivations supporting the main paper.

## Contents

### A.1: Spectral Separation and Error Bounding
**File**: `A1_spectral_separation_proof.tex`  
**Source**: `03_appendix/spectral_separation_proof.tex`

**Content**:
- Mathematical proof of error bounding via spectral separation
- Derivation of margin of decision
- Routing regret bounds
- Meta-algorithm expected behavior
- Decision margin hypothesis

**Key Results**:
- Routing error bounded by spectral gap: $P_{\text{error}} \le \exp\left(-\frac{\Delta_{\text{gap}}^2}{2\sigma^2}\right)$
- Regret bound: $R(T) \le \mathcal{O}\left( \frac{\ln(N)}{\eta} + \sqrt{T \ln(T)} \right)$
- Spectral separation enables binary classification instead of 384D regression

---

## Related Sections
- **Main Paper Figure 1**: Uses spectral separation visualization
- **Appendix C**: Empirical validation of theoretical bounds
- **Appendix D**: Ablation studies confirming theoretical predictions

---

## Files
```
A_mathematical_foundations/
├── README.md                           (this file)
├── A1_spectral_separation_proof.tex    (main proof document)
└── figures/                            (future: proof diagrams)
```
