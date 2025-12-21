# In-Sample vs. Out-of-Sample: Labeling Guide for Paper

## Critical Distinction

The paper must **clearly distinguish** between:

1. **In-Sample Calibration** (training data) - Shows the system works mechanically
2. **Out-of-Sample Generalization** (held-out data) - Shows whether it helps in deployment

---

## Figure 1: Out-of-Sample (The Primary Finding)

### Source
**ONLY:** `generate_figure1.py` (5-fold cross-validation)

### Label Requirements

**Figure Caption:**
```latex
\caption{\textbf{Offline Calibration Exhibits Consistent Negative Transfer 
(Out-of-Sample Evaluation).} \textbf{(A)} Mean cumulative regret curves with 
95\% confidence intervals across 5 folds, evaluated on held-out prompts. 
Cold start (green, solid) consistently outperforms both warm-start strategies. 
\textbf{(B)} Per-fold performance changes relative to cold start. All 10 data 
points show degradation on held-out data, demonstrating 100\% directional 
consistency.}
\label{fig:negative_transfer}
```

**Key Phrases:**
- ✅ "evaluated on held-out prompts"
- ✅ "out-of-sample evaluation"
- ✅ "5-fold cross-validation"
- ✅ "generalization performance"

**In-Text Reference:**
```latex
To evaluate \emph{out-of-sample generalization}, we conducted rigorous 5-fold 
cross-validation (Figure~\ref{fig:negative_transfer}). Each fold trained priors 
on 398 prompts and evaluated on 99 completely held-out prompts. Critically, 
warm-start strategies exhibited consistent negative transfer on unseen data: 
Shared covariance increased regret by +32.0\% (p=0.080), with 100\% directional 
consistency across folds.
```

---

## Optional: In-Sample Calibration (Mechanical Validation)

**IF** you want to show that the system works mechanically (can learn from data), you can include an in-sample plot, but it must be **clearly labeled** and **positioned as secondary**.

### Source
`/experiments/run_rq1.py` (evaluates on training prompts)

### Label Requirements

**Figure Caption:**
```latex
\caption{\textbf{In-Sample Calibration Efficiency (Not Generalization).} 
Cumulative regret when evaluated on the \emph{same prompts} used for prior 
training. This demonstrates that the learning mechanism functions correctly 
(priors can encode training data), but does \emph{not} indicate generalization 
to new prompts. For out-of-sample performance, see Figure~\ref{fig:negative_transfer}.}
\label{fig:in_sample_calibration}
```

**Key Phrases:**
- ✅ "in-sample calibration"
- ✅ "same prompts used for training"
- ✅ "not generalization"
- ✅ "mechanistic validation"

**In-Text Reference:**
```latex
To verify that the prior-training mechanism functions correctly, we evaluated 
in-sample calibration efficiency (Figure~\ref{fig:in_sample_calibration}). 
When evaluated on the \emph{same prompts} used for training, warm-start reduces 
regret, confirming the learning mechanism works. However, this in-sample efficiency 
does \emph{not} translate to out-of-sample generalization: on held-out prompts, 
warm-start exhibits negative transfer (Figure~\ref{fig:negative_transfer}). 
This divergence validates our metadata-guided cold-start architecture—offline 
calibration achieves memorization but fails generalization on <1K prompts.
```

---

## Recommended Paper Structure

### Option A: Focus Only on Out-of-Sample (RECOMMENDED)

**Structure:**
```
RQ1: Can Offline Calibration Outperform Metadata-Guided Cold Start?
├── Methods: 5-fold cross-validation setup
├── Results: Figure 1 (out-of-sample, negative transfer)
└── Discussion: Why cold start wins
```

**Advantages:**
- ✅ Cleaner narrative
- ✅ No risk of confusion
- ✅ Focuses on the important finding

**Disadvantages:**
- 🟡 Doesn't show that priors "can work" in principle

---

### Option B: Show Both In-Sample and Out-of-Sample

**Structure:**
```
RQ1: Investigating Offline Calibration
├── Methods: Two evaluation modes
│   ├── In-Sample: Calibration efficiency (same prompts)
│   └── Out-of-Sample: Generalization (5-fold CV)
├── Results:
│   ├── Figure S1 (Appendix): In-sample calibration works
│   └── Figure 1 (Main): Out-of-sample shows negative transfer
└── Discussion: Calibration ≠ Generalization
```

**Advantages:**
- ✅ Shows full picture
- ✅ Demonstrates system works mechanically
- ✅ Highlights calibration-generalization gap

**Disadvantages:**
- 🟡 More complex narrative
- 🟡 Risk of confusion if not labeled carefully

---

## LaTeX Example: Clear Labeling

### Main Text (Out-of-Sample)

```latex
\subsection{RQ1: Out-of-Sample Evaluation of Offline Calibration}

\textbf{Research Question:} Can priors trained on calibration data improve 
routing performance on \emph{new, unseen prompts}?

\textbf{Experimental Design:} We conducted 5-fold cross-validation to ensure 
rigorous out-of-sample evaluation. Each fold:
\begin{itemize}
\item \textbf{Training:} 398 prompts used to train priors (all models graded)
\item \textbf{Testing:} 99 \emph{completely held-out} prompts never seen during training
\item \textbf{Evaluation:} 2,000 routing decisions on test prompts
\end{itemize}

\textbf{Key Distinction:} This evaluates \emph{generalization}, not memorization. 
The test prompts were withheld from prior training, simulating real deployment 
where the router encounters new user requests.

\textbf{Finding: Consistent Negative Transfer}

Figure~\ref{fig:negative_transfer} shows that warm-start strategies consistently 
\emph{harm} performance on held-out prompts:
\begin{itemize}
\item \textbf{Shared Covariance:} +32.0\% regret increase (p=0.080, 100\% fold consistency)
\item \textbf{Disjoint Priors:} +27.4\% regret increase (p=0.107, 100\% fold consistency)
\end{itemize}

While p-values narrowly miss conventional thresholds, the 100\% directional 
consistency (10/10 fold-strategy pairs worse) provides strong evidence for 
negative transfer on unseen data.

\textbf{Implication:} Metadata-guided cold start outperforms offline calibration 
for out-of-sample routing, validating our zero-benchmark architecture.
```

### Appendix (Optional: In-Sample)

```latex
\subsection{Appendix: In-Sample Calibration Efficiency}

For completeness, we verify that the prior-training mechanism functions correctly 
by evaluating \emph{in-sample calibration}—routing performance on the \emph{same 
prompts} used for prior training.

\textbf{Setup:} Train priors on archetype grid (497 prompts), then evaluate 
routing on those same prompts.

\textbf{Result:} Warm-start reduces regret by 64\% relative to cold start 
(Figure~\ref{fig:in_sample_calibration}), confirming the learning mechanism 
encodes training data correctly.

\textbf{Critical Note:} This in-sample efficiency does \textbf{not} indicate 
generalization. When evaluated on held-out prompts (out-of-sample), warm-start 
exhibits \emph{negative transfer} (+32\%, Figure~\ref{fig:negative_transfer}). 
This calibration-generalization gap explains why we default to metadata-guided 
cold start: memorizing training prompts does not help with new user requests.
```

---

## Table: Side-by-Side Comparison

If you want to show both results clearly:

```latex
\begin{table}[h]
\centering
\caption{In-Sample vs. Out-of-Sample Performance of Warm-Start Strategies}
\label{tab:in_vs_out_sample}
\begin{tabular}{lcc}
\toprule
\textbf{Evaluation Mode} & \textbf{Shared Covariance} & \textbf{Interpretation} \\
\midrule
\textbf{In-Sample} & & \\
\quad (Same prompts as training) & \textbf{-64\% regret} ✓ & Memorization works \\
\midrule
\textbf{Out-of-Sample (5-fold CV)} & & \\
\quad (Held-out prompts) & \textbf{+32\% regret} ✗ & Generalization fails \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\small
Note: In-sample evaluation measures whether priors encode training data correctly. 
Out-of-sample evaluation measures whether this encoding generalizes to new prompts. 
Only out-of-sample performance is relevant for deployment. The divergence validates 
metadata-guided cold start as the superior approach for <1K calibration datasets.
\end{table}
```

---

## Terminology Guide

### ✅ USE for Out-of-Sample (Figure 1)

- "held-out prompts"
- "out-of-sample generalization"
- "5-fold cross-validation"
- "never seen during training"
- "deployment simulation"
- "generalization performance"

### ✅ USE for In-Sample (Optional Appendix Figure)

- "in-sample calibration"
- "same prompts used for training"
- "mechanistic validation"
- "calibration efficiency"
- "training set performance"
- "not generalization"

### ❌ NEVER USE (Ambiguous)

- "evaluated on archetype grid" (Which one? Train or test?)
- "tested on calibration set" (In-sample or out?)
- "Day-1 performance" (With what data?)
- Just "calibration" without qualifier

---

## Checklist for Final Paper

Before submission, verify:

### Figure 1 (Out-of-Sample)
- [ ] Caption includes "out-of-sample" or "held-out"
- [ ] Caption mentions "5-fold cross-validation"
- [ ] Source is `generate_figure1.py`
- [ ] All cited numbers match `figure1_statistics_enhanced.json`

### Optional Appendix Figure (In-Sample)
- [ ] Caption includes "in-sample" prominently
- [ ] Caption says "same prompts used for training"
- [ ] Caption says "not generalization"
- [ ] Caption references Figure 1 for out-of-sample results
- [ ] Source is clearly labeled as `/experiments/run_rq1.py`

### Text
- [ ] Methods section clearly defines in-sample vs. out-of-sample
- [ ] Results emphasize out-of-sample as the primary finding
- [ ] Discussion explains why only out-of-sample matters for deployment
- [ ] No ambiguous "calibration" without qualifier

---

## Visual Distinction

If showing both figures, use clear visual differences:

### Figure 1 (Out-of-Sample) - Main Paper
- **Border:** None or thin
- **Label:** "A" and "B" panels
- **Color:** Green (cold start), Red (shared), Blue (disjoint)
- **Size:** Full column width

### Figure S1 (In-Sample) - Appendix
- **Border:** Dashed red border (warning!)
- **Label:** Watermark: "IN-SAMPLE ONLY"
- **Color:** Grayscale (de-emphasize)
- **Size:** Half column width
- **Caption starts:** ⚠️ "In-Sample Calibration (Not Generalization)"

---

## Key Message

**For reviewers and readers:**

> "We distinguish between in-sample calibration (can the system learn?) and 
> out-of-sample generalization (does it help in deployment?). While warm-start 
> achieves in-sample efficiency, it exhibits consistent negative transfer 
> out-of-sample (+32%, 100% fold consistency). This validates metadata-guided 
> cold start as the scientifically correct default for <1K calibration datasets."

---

## Recommendation

**Best Practice:** Use Option A (out-of-sample only) unless you specifically need to address a reviewer concern about "but does the learning mechanism work at all?"

**Rationale:**
1. Cleaner narrative
2. Focuses on what matters (deployment)
3. No risk of misinterpretation

**If you include in-sample:** Always position it as secondary (appendix), clearly labeled, and explicitly contrasted with out-of-sample.

---

## Bottom Line

✅ **Figure 1 (Main):** Out-of-sample, from `generate_figure1.py`, negative transfer  
🟡 **Figure S1 (Optional Appendix):** In-sample, from `run_rq1.py`, clearly labeled  
📝 **Text:** Always specify "in-sample" or "out-of-sample" explicitly  

**The golden rule:** Never say "calibration" or "evaluated on X" without clarifying whether it's in-sample or out-of-sample.

