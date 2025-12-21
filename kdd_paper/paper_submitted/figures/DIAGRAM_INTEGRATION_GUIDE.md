# Diagram Integration Guide

## ✅ All 3 Diagrams Successfully Created!

**Location:** `/Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted/figures/`

| Diagram | File | Size | Status |
|---------|------|------|--------|
| **Architecture** | `architecture_diagram.pdf` | 113K | ✅ Ready |
| **Distillation** | `distillation_diagram.pdf` | 156K | ✅ Ready |
| **Decision Tree** | `decision_tree_diagram.pdf` | 133K | ✅ Ready |

---

## 📍 Where to Add Each Diagram in Your Paper

### **Diagram 1: Architecture Diagram**

**What it shows:** System overview - query flow, bandit with priors, model pool, feedback loop

**Recommended location:** Section 2 (Method) - Right after Section 2.1 "System Architecture"

**LaTeX code to add:**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/architecture_diagram.pdf}
    \caption{\textbf{BanditGPT Architecture.} The system accepts user queries, extracts contextual features, uses shippable priors to warm-start a LinUCB bandit, selects from an 81-model pool via UCB, and updates online via reward feedback. Offline distillation (left) amortizes expensive teacher supervision across all users. Online operation (right) requires zero user-provided calibration.}
    \label{fig:architecture}
\end{figure}
```

**Add after this text in `method.tex` (around line 20):**
> "Figure~\ref{fig:architecture} illustrates the complete system architecture."

---

### **Diagram 2: Distillation Diagram**

**What it shows:** Three-phase process for creating shippable priors: teacher supervision → covariance learning → compression

**Recommended location:** Section 2.3 "Shippable Priors via Offline Distillation"

**LaTeX code to add:**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/distillation_diagram.pdf}
    \caption{\textbf{Shippable Prior Distillation.} We amortize expensive offline distillation across users through a three-phase process: (Phase 1) Teacher supervision on calibration datasets generates quality labels; (Phase 2) Contextual bandit training learns per-model covariance matrices $\mathbf{A}_m$; (Phase 3) Low-rank compression reduces the prior to <1MB for distribution. Users download pre-trained priors, eliminating cold-start regret without providing calibration data.}
    \label{fig:distillation}
\end{figure}
```

**Add after this text in `method.tex` (around line 50, after explaining priors):**
> "Figure~\ref{fig:distillation} visualizes the three-phase distillation process."

---

### **Diagram 3: Decision Tree Diagram**

**What it shows:** Routing logic flow - Standard vs Hybrid mode decision paths, UCB calculation, model selection

**Recommended location:** Section 2.7 "Bandit-Guided Cascade" or Section 2.4 "Tunable Objective"

**LaTeX code to add:**

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/decision_tree_diagram.pdf}
    \caption{\textbf{Routing Decision Flow.} Users control system behavior via operating mode and sensitivity parameters ($\lambda_{cost}$, $\lambda_{latency}$, $\tau_{verify}$). Standard mode provides single-shot routing optimized for cost (left path). Hybrid mode performs uncertainty-aware routing, triggering selective verification for low-confidence predictions (right path). Online feedback continuously updates bandit parameters regardless of mode.}
    \label{fig:routing_logic}
\end{figure}
```

**Add after this text in `method.tex` (around line 150, after explaining hybrid mode):**
> "Figure~\ref{fig:routing_logic} shows the complete routing decision flow."

---

## 🔧 Quick Integration Instructions

### **Step 1: Add Figure References**

In your `method.tex`, add these three lines where appropriate:

```latex
% After Section 2.1
Figure~\ref{fig:architecture} illustrates the complete system architecture.

% After Section 2.3
Figure~\ref{fig:distillation} visualizes the three-phase distillation process.

% After Section 2.7
Figure~\ref{fig:routing_logic} shows the complete routing decision flow.
```

### **Step 2: Insert Figure Code**

Copy the three `\begin{figure}...\end{figure}` blocks from above into your `method.tex` at the recommended locations.

### **Step 3: Recompile Paper**

```bash
cd /Users/annette/repostitories/llm_jury/kdd_paper/paper_submitted
./compile.sh  # or use your compilation command
```

---

## 📊 Diagram Details

### **Architecture Diagram (`architecture_diagram.pdf`)**

**Components:**
- User Query → Sentence Encoder (384-dim)
- LinUCB Bandit (central component)
- Shippable Priors (left side, <1MB)
- Model Pool (right side, 81 models with prices)
- UCB Selection layer
- Feedback loop (reward signal)

**Annotations:**
- Offline: Distill priors, compress to <1MB
- Online: Zero calibration, autonomous learning, O(1) model addition

**Visual Style:**
- Blue blocks: Processing steps
- Green blocks: Data
- Orange blocks: Models
- Purple blocks: Priors
- Dashed red arrows: Feedback loop

---

### **Distillation Diagram (`distillation_diagram.pdf`)**

**Three Phases:**

1. **Phase 1: Offline Distillation** (left)
   - Calibration dataset (2k prompts)
   - Teacher model (GPT-4o)
   - Quality labels

2. **Phase 2: Covariance Learning** (middle)
   - Embed prompts (384-dim)
   - Train bandit
   - Learn covariance matrices $\mathbf{A}_m$

3. **Phase 3: Compression** (right)
   - Low-rank approximation (SVD/PCA)
   - Shippable prior (<1MB for 81 models)
   - API metadata (cost, latency)

**Bottom:** Deployment ready (download 1MB, zero calibration)

**Timeline Annotation:**
- Left brace: One-time offline cost (author-borne)
- Right brace: Amortized across all users

---

### **Decision Tree Diagram (`decision_tree_diagram.pdf`)**

**Flow:**

1. **Start:** User query $q$
2. **Feature Extraction:** Extract $\mathbf{x} = \phi(q)$ (384-dim)
3. **Mode Decision:** Standard or Hybrid?

**Standard Mode Path (Left):**
- Compute UCB for all models
- Calculate $\text{UCB}_m = \hat{Q} + \beta\sigma - \lambda_c C_m$
- Select $m^* = \argmax \text{UCB}_m$
- Route to single model

**Hybrid Mode Path (Right):**
- Compute UCB + uncertainty
- Decision: $\max \text{UCB}_m > \tau_{verify}$?
  - Yes: Route to best model (high confidence)
  - No: Cascade (best → verify)

**Common:**
- Execute on selected model(s)
- Compute reward: $r = Q - \lambda_c C$
- Update bandit (feedback loop)
- Return response

**Annotations:**
- Standard Mode box: Cost Leader, minimize cost, O(1) latency
- Hybrid Mode box: High Assurance, verify uncertainty, selective cascade
- User Constraints: $\lambda_{cost}$, $\lambda_{latency}$, $\tau_{verify}$

---

## 🎨 Visual Design Choices

### **Color Scheme:**
- **Blue:** Core processing/algorithms
- **Green:** Data inputs/outputs
- **Orange:** LLM models
- **Purple:** Shippable priors/compressed data
- **Yellow:** Decision points
- **Red (dashed):** Feedback/update loops

### **Shapes:**
- **Rectangles:** Processing blocks
- **Rounded rectangles:** Data blocks
- **Diamonds:** Decision points
- **Ellipses:** Start/end points
- **Dashed boxes:** Grouping/phases

### **Fonts:**
- Main text: Normal weight
- Section titles: Bold
- Small details: \small or \scriptsize
- Mathematical: Math mode

---

## 📐 Page Budget Impact

Adding 3 figures will add approximately **0.75-1.0 pages** to your paper.

**Current:** ~8 pages main content  
**With 3 diagrams:** ~8.75-9 pages main content

### **Options if Over Page Limit:**

1. **Move 1 diagram to appendix** (likely Decision Tree)
2. **Reduce figure size** (use `width=0.9\columnwidth` instead of `\columnwidth`)
3. **Compress another section** slightly
4. **Two-column subfigures** (combine 2 diagrams in one figure)

**My recommendation:** Add all 3 and see the page count. If over, move Decision Tree to appendix since it's less critical than Architecture and Distillation.

---

## ✅ Verification Checklist

After integrating:

- [ ] All 3 PDFs exist in `figures/` directory
- [ ] `\includegraphics` paths are correct
- [ ] `\label{fig:...}` tags are unique
- [ ] `Figure~\ref{...}` references in text
- [ ] Captions are descriptive and self-contained
- [ ] Paper compiles without errors
- [ ] Figures appear in correct sections
- [ ] Page count is still ≤8 pages (or acceptably close)

---

## 🚀 Quick Start: Add All 3 Figures Now

Copy this complete block into your `method.tex`:

```latex
% ============================================================
% SECTION 2.1: Architecture Overview (ADD AFTER LINE ~20)
% ============================================================

Figure~\ref{fig:architecture} illustrates the complete system architecture.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/architecture_diagram.pdf}
    \caption{\textbf{BanditGPT Architecture.} The system accepts user queries, extracts contextual features, uses shippable priors to warm-start a LinUCB bandit, selects from an 81-model pool via UCB, and updates online via reward feedback. Offline distillation (left) amortizes expensive teacher supervision across all users. Online operation (right) requires zero user-provided calibration.}
    \label{fig:architecture}
\end{figure}

% ============================================================
% SECTION 2.3: Shippable Priors (ADD AFTER LINE ~50)
% ============================================================

Figure~\ref{fig:distillation} visualizes the three-phase distillation process that creates shippable priors from expensive offline teacher supervision.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/distillation_diagram.pdf}
    \caption{\textbf{Shippable Prior Distillation.} We amortize expensive offline distillation across users through a three-phase process: (Phase 1) Teacher supervision on calibration datasets generates quality labels; (Phase 2) Contextual bandit training learns per-model covariance matrices $\mathbf{A}_m$; (Phase 3) Low-rank compression reduces the prior to <1MB for distribution. Users download pre-trained priors, eliminating cold-start regret without providing calibration data.}
    \label{fig:distillation}
\end{figure}

% ============================================================
% SECTION 2.7: Hybrid Architecture (ADD AFTER LINE ~150)
% ============================================================

Figure~\ref{fig:routing_logic} shows the complete routing decision flow, illustrating how user-specified parameters control the trade-off between cost optimization (Standard mode) and reliability assurance (Hybrid mode).

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/decision_tree_diagram.pdf}
    \caption{\textbf{Routing Decision Flow.} Users control system behavior via operating mode and sensitivity parameters ($\lambda_{cost}$, $\lambda_{latency}$, $\tau_{verify}$). Standard mode provides single-shot routing optimized for cost (left path). Hybrid mode performs uncertainty-aware routing, triggering selective verification for low-confidence predictions (right path). Online feedback continuously updates bandit parameters regardless of mode.}
    \label{fig:routing_logic}
\end{figure}
```

Then recompile your paper!

---

## 🎉 Success!

All 3 conceptual diagrams are now created and ready for integration:

1. ✅ **Architecture Diagram** - System overview with offline/online components
2. ✅ **Distillation Diagram** - Three-phase prior creation process
3. ✅ **Decision Tree Diagram** - Routing logic flow with Standard/Hybrid modes

**Total size:** 402K (113K + 156K + 133K)  
**Format:** High-quality vector PDF  
**Style:** Professional TikZ with consistent color scheme

**Your paper is now complete with all requested diagrams!** 🚀

