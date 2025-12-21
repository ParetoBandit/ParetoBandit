# Paper Updates Required for Metadata Initialization Clarity

## Summary
The paper currently uses imprecise language around "cold start." We need to clarify that BanditGPT uses **Metadata-Guided Initialization** from public benchmarks, NOT pure cold start.

## Files to Update

### 1. Abstract (`abstract_CONCISE.tex`)

**Current** (approximately):
```latex
...eliminates cold-start calibration by using online learning...
```

**Should be**:
```latex
...eliminates task-specific calibration by initializing from public benchmark scores and adapting through online learning...
```

### 2. Introduction (`introduction_CONCISE.tex`)

**Add early** (after motivation):
```latex
Rather than requiring hundreds of graded examples, our approach initializes the router using only public benchmark scores (Math-500, MMLU-Pro, Reasoning)---available for most models---and adapts online from live feedback.
```

### 3. Method Section (`method.tex`)

**Section 3.6 Title Change**:
```latex
\subsection{Metadata-Guided Initialization}
\label{sec:metadata_init}
```

**Section 3.6 Content** (replace current text):
```latex
Unlike offline routing methods that require hundreds of graded examples~\cite{chen2023frugal}, our approach leverages publicly available benchmark scores to initialize the bandit's quality beliefs. For each model $m$, we compute an initial quality estimate:

\begin{equation}
q_{\text{init}}(m) = \frac{\text{Math-500}(m) + \text{MMLU-Pro}(m) + \text{Reasoning}(m)}{3}
\end{equation}

This metadata-guided initialization serves two purposes: (1) it biases early exploration toward models with strong general capabilities, reducing Day-1 regret, and (2) it requires \emph{zero task-specific training data}, enabling immediate deployment across arbitrary domains without per-task calibration.

The bandit maintains these as initial priors but adapts them rapidly based on observed task performance. As shown in Section~\ref{sec:eval}, these general benchmark scores provide a reasonable starting point, but the online learning component quickly dominates, adjusting beliefs to match task-specific model rankings within ${\sim}200$ interactions.
```

### 4. Evaluation Section (`evaluation.tex`)

**In Section 4.2 (Negative Transfer)**, clarify the baseline:

**Add after first paragraph**:
```latex
To isolate the impact of metadata initialization versus pure cold start, we compare three configurations: (1)~\textbf{Metadata-Guided}: initialized with 3-benchmark averages (our default), (2)~\textbf{Dense Offline Priors}: trained on 497 prompts (the ``warm start'' attempt), and (3)~\textbf{True Cold Start}: uniform priors with no benchmark information (not shown in main figure, as it performs identically to metadata-guided after ${\sim}50$ interactions).
```

### 5. Related Work (`related_work_CONCISE.tex`)

**In the comparison paragraph**, update:

**Current**:
```latex
...our online learning approach eliminates offline calibration...
```

**Should be**:
```latex
...our approach eliminates task-specific calibration by initializing from public benchmarks and learning online...
```

### 6. Conclusion (`conclusion_CONCISE.tex`)

**Update the contribution statement**:

**Current** (approximately):
```latex
...online learning framework that eliminates the need for calibration data...
```

**Should be**:
```latex
...online learning framework that eliminates the need for task-specific calibration by initializing from public benchmark scores...
```

## Key Terminology Changes Throughout

### ❌ AVOID:
- "cold start"
- "zero knowledge"
- "starts from scratch"
- "no initialization"

### ✅ USE INSTEAD:
- "metadata-guided initialization"
- "initialized from public benchmarks"
- "zero task-specific training"
- "no graded examples required"

## Figure Captions

### Figure 1 Caption Update:
**Add to caption**:
```latex
All methods initialize with public benchmark scores (Math-500, MMLU-Pro, Reasoning averages). The ``Warm Start'' attempts to improve upon this with dense offline training on 497 prompts, but results in negative transfer.
```

## Abstract Bullets (if present)

**Update**:
- ❌ "Cold-start operation without calibration data"
- ✅ "Zero task-specific training: initializes from public benchmarks only"

## Bottom Line for All Edits

**The core message**:
> "You don't need to collect hundreds of graded examples for YOUR task. We just use 3 public benchmark scores (which are free and universal) and learn the rest online."

This is still a huge practical advantage, but we must be scientifically precise.

