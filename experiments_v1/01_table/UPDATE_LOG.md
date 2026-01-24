# Table 1: Update Log

## ✅ Latest Update: Added LinUCB Priors Explanation

**Date**: 2026-01-24  
**Reason**: Clarify what "LinUCB priors (33 dims with bias)" means for KDD reviewers

### Changes Made

#### 1. LaTeX Table (`table_dataset_composition.tex`)

**Added two new table notes:**

1. **Warmup Set description** - Now includes mathematical notation:
   ```latex
   LinUCB warmup priors (covariance matrix $\mathbf{A} \in \mathbb{R}^{33 \times 33}$ 
   and belief vector $\mathbf{b} \in \mathbb{R}^{33}$)
   ```

2. **New LinUCB Priors note** - Explains what A and b are:
   ```latex
   \textbf{LinUCB Priors:} Warmup data initializes the contextual bandit with 
   covariance matrix $\mathbf{A}$ (capturing feature correlations) and belief 
   vector $\mathbf{b}$ (encoding reward expectations) for each model. Context 
   dimension: 33 (32 PCA components + 1 bias term).
   ```

#### 2. Documentation Updates

**Updated files:**
- ✅ `README.md` - Added A and b matrix descriptions
- ✅ `SUMMARY.md` - Expanded warmup priors explanation
- ✅ `QUICK_REFERENCE.md` - Added A and b dimensions
- ✅ `analyze_dataset_composition.py` - Source script updated

**New file:**
- ✅ `LINUCB_EXPLAINER.md` - Comprehensive 5.4KB explainer document

### What Reviewers Now See

**Before:**
> "LinUCB warmup priors (33 dims with bias)"

**After:**
> "LinUCB warmup priors (covariance matrix **A** ∈ ℝ³³ˣ³³ and belief vector **b** ∈ ℝ³³)"
>
> Plus explanation: "Warmup data initializes the contextual bandit with covariance 
> matrix A (capturing feature correlations) and belief vector b (encoding reward 
> expectations) for each model."

### Why This Matters

1. **Clarity**: Reviewers unfamiliar with contextual bandits now understand what the priors are
2. **Mathematical rigor**: Shows exact matrix dimensions (33×33 and 33×1)
3. **Intuition**: Explains what A and b represent (correlations vs. expectations)
4. **Completeness**: Shows how 32 PCA + 1 bias = 33 dimensions

### LinUCB Priors Quick Reference

For reviewers asking "What are these?":

| Component | Dimension | Purpose | Per Model Size |
|-----------|-----------|---------|----------------|
| **A** (covariance) | 33×33 | Feature correlations & uncertainty | 8.7 KB |
| **b** (beliefs) | 33×1 | Reward expectations | 264 bytes |
| **Total** | - | Warmup initialization | ~9 KB |

**For 2 models**: ~18 KB total (very lightweight!)

### Technical Details

**A matrix** (covariance):
- Tracks which prompt features predict good/bad outcomes
- Updated with: `A ← A + context × context^T`
- Initialized from 80k warmup prompts

**b vector** (beliefs):
- Encodes which prompt features lead to high rewards
- Updated with: `b ← b + reward × context`
- Initialized from 80k warmup prompts

**Context vector** (33 dimensions):
```
[PCA₁, PCA₂, ..., PCA₃₂, 1.0]
 └─────────────────────┘  └─┘
   32 semantic features   bias
```

### Files in Directory

```
experiments_v1/01_table/
├── table_dataset_composition.tex  ← LaTeX table (UPDATED)
├── analyze_dataset_composition.py ← Generation script (UPDATED)
├── README.md                      ← Documentation (UPDATED)
├── DATA_PROVENANCE.md             ← Data sources
├── SUMMARY.md                     ← Executive summary (UPDATED)
├── QUICK_REFERENCE.md            ← Quick ref (UPDATED)
├── LINUCB_EXPLAINER.md           ← NEW: Deep dive on LinUCB
└── UPDATE_LOG.md                 ← This file
```

### How to Use in Paper

The table now has all the information reviewers need. If they ask:

**Q: "What are LinUCB priors?"**  
**A:** Point to table note: "Warmup data initializes the contextual bandit with covariance matrix A (capturing feature correlations) and belief vector b (encoding reward expectations)..."

**Q: "What are the dimensions?"**  
**A:** "A ∈ ℝ³³ˣ³³ and b ∈ ℝ³³, where 33 = 32 PCA components + 1 bias term"

**Q: "How are they initialized?"**  
**A:** "From 80,000 LMSYS Arena battles (warmup set)"

For more details, point them to: `LINUCB_EXPLAINER.md`

### Verification

To verify the table compiles correctly:

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{amsmath}

\begin{document}
\input{experiments_v1/01_table/table_dataset_composition.tex}
\end{document}
```

Should compile without errors (requires `booktabs` and `amsmath` packages).

---

## Change History

| Date | Change | Reason |
|------|--------|--------|
| 2026-01-24 | Initial table creation | KDD submission requirement |
| 2026-01-24 | Updated PCA: 23→32 components | User correction |
| 2026-01-24 | Added LinUCB explanation | Clarify for reviewers |

---

**Status**: ✅ Complete  
**Ready for**: KDD submission  
**Last verified**: 2026-01-24

