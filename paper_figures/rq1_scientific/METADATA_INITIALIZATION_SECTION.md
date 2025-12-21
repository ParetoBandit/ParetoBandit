# Metadata-Guided Initialization: Technical Description for Paper

## For Method Section

```latex
\subsection{Metadata-Guided Cold-Start Initialization}
\label{sec:metadata_init}

Unlike existing routing systems that require hundreds to thousands of labeled 
examples for offline calibration~\cite{frugalgpt,routellm}, BanditGPT enables 
\textbf{zero-benchmark deployment} through metadata-guided initialization.

\subsubsection{Design Philosophy}

We distinguish between two types of routing information:

\begin{enumerate}
\item \textbf{Hard Constraints:} Cost limits, context window requirements, 
capability flags (e.g., "supports function calling"). These are \emph{known 
perfectly} from public API specifications and must be respected from Day 0.

\item \textbf{Quality Preferences:} Which model produces the best response for 
a given prompt. This is \emph{task-specific}, \emph{subjective}, and 
\emph{context-dependent}—it cannot be reliably transferred from offline calibration 
on different prompts (see RQ1).
\end{enumerate}

Our initialization strategy: \textbf{use metadata for constraints, learn quality 
from experience}.

\subsubsection{Initialization Protocol}

For each model $m \in \mathcal{M}$, we initialize DisjointLinUCB parameters as:

\begin{align}
A_m &= \lambda \cdot I_{d \times d} \label{eq:metadata_A} \\
b_m &= \lambda \cdot \phi(\text{metadata}_m) \label{eq:metadata_b}
\end{align}

Where:
\begin{itemize}
\item $\lambda$ is the \emph{prior strength} (typically 1.0--2.0)
\item $I$ is the identity matrix (isotropic uncertainty)
\item $\phi(\cdot)$ is the same sentence-BERT encoder used for prompts
\item $\text{metadata}_m$ is a structured text representation:
\end{itemize}

\begin{lstlisting}[language=Python]
metadata = f"""
Model: {model.name}
Provider: {model.provider}
Cost: ${model.price_per_1m_tokens:.2f} per 1M tokens
Context: {model.max_context_tokens:,} tokens
Capabilities: {", ".join(model.capabilities)}
Benchmarks: MMLU={model.mmlu_score:.1f}%, Math={model.math_score:.1f}%
"""
\end{lstlisting}

This metadata string is embedded into $\mathbb{R}^d$ using the prompt encoder, 
producing $\phi(\text{metadata}_m) \in \mathbb{R}^d$.

\subsubsection{Why This Works}

\textbf{Hard Constraints via Pre-Selection:} Before computing UCB scores, we 
filter the model pool based on metadata:
\begin{itemize}
\item If \texttt{max\_cost} specified, exclude models exceeding it
\item If \texttt{min\_context} needed, exclude models with insufficient windows
\item If \texttt{required\_capabilities}, exclude incompatible models
\end{itemize}

This ensures constraints are \emph{perfectly respected} from the first routing 
decision.

\textbf{Quality Heuristics via $b_m$:} The metadata embedding provides a "soft 
prior" on quality. For example:
\begin{itemize}
\item If metadata mentions "MMLU=90\%", the initial $\theta_m = A_m^{-1} b_m$ 
will have higher expected reward
\item This creates a \emph{slight bias} toward high-benchmark models on Day 1
\item However, the bias is weak—online experience quickly dominates
\end{itemize}

\textbf{Exploration via Isotropic $A_m$:} Critically, we keep $A_m = \lambda I$ 
rather than learning it from calibration data. This ensures the exploration bonus 
$\alpha \sqrt{x^T A_m^{-1} x}$ is \emph{uniformly high} on Day 1, encouraging 
the system to explore all models on relevant prompts.

Our RQ1 experiments demonstrate why this matters: learning $A_m$ from sparse 
offline data (<1K prompts) causes \textbf{negative transfer}. By keeping uncertainty 
high, metadata initialization avoids the "false confidence" trap.

\subsubsection{Comparison to Offline Calibration}

\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{Property} & \textbf{Metadata Init} & \textbf{Offline Calibration} \\
\midrule
Hard constraints (Day 0) & Perfect & Perfect* \\
Quality estimate (Day 0) & Rough (benchmarks) & Learned (5--500 samples) \\
Exploration bonus (Day 0) & High (encourages discovery) & Low (false confidence) \\
Data required & 0 labeled prompts & 500--5K prompts \\
Training time & None & Hours to days \\
Generalization risk & None & Negative transfer (RQ1) \\
Adaptation to new models & Instant & Retrain required \\
\bottomrule
\multicolumn{3}{l}{\footnotesize *Assuming constraints included in calibration data} \\
\end{tabular}
\caption{Metadata initialization vs. offline calibration. Metadata init provides 
immediate constraint satisfaction with zero calibration cost, while avoiding the 
negative transfer observed in RQ1.}
\label{tab:init_comparison}
\end{table}

\subsubsection{Implementation Details}

\textbf{Embedding Model:} We use \texttt{sentence-transformers/all-MiniLM-L6-v2} 
($d=384$) for both prompts and metadata, ensuring semantic alignment. For efficiency, 
metadata embeddings are precomputed and cached.

\textbf{Prior Strength:} We set $\lambda=1.0$ by default, treating metadata as 
equivalent to "one perfect observation." This provides light guidance without 
overwhelming online learning.

\textbf{Metadata Sources:} Cost and context limits are from official API 
documentation. Benchmark scores are from model cards or public leaderboards 
(LMSys Chatbot Arena, HuggingFace Open LLM Leaderboard). Missing benchmarks 
default to provider-average.

\textbf{Advanced Feature:} For enterprise users with >10K labeled interactions, 
our library supports loading custom-trained priors via \texttt{load\_priors(path)}. 
This is an \emph{optional} power-user feature—the default remains metadata-guided 
cold start.
```

---

## For Experimental Setup (RQ1)

```latex
\subsubsection{RQ1 Baseline: Metadata-Guided Cold Start}

Our primary baseline is the system's default configuration:

\textbf{Policy:} DisjointLinUCBPolicy with $\alpha=0.5$ (exploration parameter)

\textbf{Initialization:}
\begin{itemize}
\item $A_m = 1.0 \cdot I_{32 \times 32}$ for all models $m$
\item $b_m = \phi(\text{metadata}_m)$ where metadata includes cost, context, and 
MMLU/Math scores
\item Prior strength: $\lambda = 1.0$ (equivalent to one observation)
\end{itemize}

\textbf{Online Learning:} After each routing decision, the selected model $m$ 
receives a reward $r \in [0, 1]$ (from LLM-as-judge), and we update:
\begin{align}
A_m &\leftarrow A_m + x x^T \\
b_m &\leftarrow b_m + r \cdot x
\end{align}

\textbf{Rationale:} This configuration represents "Day 1 deployment" with no 
calibration data—it tests whether the system can learn effective routing purely 
from online interaction with metadata guidance.

\textbf{Warm-Start Alternatives:} We compare against two warm-start strategies:
\begin{enumerate}
\item \textbf{Shared Covariance:} $A_{shared}$ trained on 398 calibration prompts 
(all models), then applied to all models. Tests "universal difficulty" hypothesis.
\item \textbf{Disjoint Priors:} Individual $A_m$ trained per model on calibration 
data. Tests model-specific prior transfer.
\end{enumerate}

The key question: can learned priors outperform metadata-guided cold start on 
held-out prompts?
```

---

## For Results (After RQ1)

```latex
\subsubsection{RQ1 Conclusion: Metadata Initialization Validated}

Figure~\ref{fig:negative_transfer} provides empirical validation for metadata-guided 
cold start as the default initialization strategy. Key takeaways:

\begin{enumerate}

\item \textbf{Zero-Benchmark Deployment is Viable:} Cold start (no calibration) 
outperforms both warm-start strategies, proving that effective routing does not 
require expensive offline calibration on <1K prompt datasets.

\item \textbf{Metadata Provides Sufficient Guidance:} The metadata-initialized 
$b_m$ vectors provide enough signal to avoid catastrophic early mistakes (e.g., 
selecting expensive models for simple prompts), while high $A_m$ uncertainty 
encourages beneficial exploration.

\item \textbf{Online Learning is Superior:} Rather than attempting to "transfer 
brains" from offline data (which causes negative transfer), learning quality 
preferences from real usage avoids distribution shift and adapts to deployment-specific 
patterns.

\item \textbf{Practical Implication:} Practitioners can deploy BanditGPT immediately 
with metadata alone, achieving effective routing from Day 1 while learning improves 
performance continuously.

\end{enumerate}

These findings inform our default library configuration: \texttt{priors=None} 
(metadata-guided cold start), with custom prior loading available as an advanced 
feature for users with >10K calibration prompts.
```

---

## For Discussion

```latex
\subsection{Design Philosophy: Separating Constraints from Preferences}

A key insight from our work is the distinction between \emph{transferable metadata} 
and \emph{non-transferable preferences}:

\textbf{What Transfers:} Hard constraints (cost, context limits) and rough quality 
heuristics (benchmark scores) are universal—they apply equally to all users and 
use cases. These can be initialized from metadata.

\textbf{What Doesn't Transfer:} Fine-grained quality preferences are deployment-specific. 
A legal firm may prefer accuracy over speed; a content generator may prefer creativity. 
An educator may value explanation detail; a chatbot may value brevity. These 
preferences cannot be reliably transferred from offline calibration on different 
users' prompts.

Our RQ1 findings empirically demonstrate this: attempting to transfer learned 
preferences via $A_m$ matrices causes negative transfer (+32\% regret). By 
contrast, metadata initialization respects the separation: constraints are enforced, 
rough heuristics provide guidance, but fine-grained preferences are learned from 
each deployment's specific usage patterns.

This design philosophy—\textbf{use metadata for constraints, learn quality from 
experience}—enables zero-benchmark deployment while avoiding the brittleness of 
offline-calibrated systems.
```

---

## For Related Work (Contrast)

```latex
\paragraph{Warm-Start Bandit Methods.}

Prior work on contextual bandits emphasizes warm-start strategies to reduce 
cold-start regret~\cite{bietti2021contextual,jang2021bootstrapping}. However, 
these methods assume (1) large offline datasets ($>10$K samples) and (2) i.i.d. 
deployment distribution matching calibration data.

In LLM routing, these assumptions often fail:
\begin{itemize}
\item \textbf{Data Scarcity:} Typical deployments cannot afford to label thousands 
of prompts before launch (our RQ1 shows <1K is insufficient)
\item \textbf{Distribution Shift:} Each deployment has unique use cases, user 
preferences, and prompt patterns that differ from calibration data
\item \textbf{Model Evolution:} LLM APIs change frequently (pricing, capabilities, 
deprecations), causing prior staleness
\end{itemize}

Our contribution is demonstrating that \emph{metadata-guided cold start outperforms 
warm start} when these assumptions are violated. By separating transferable 
constraints (metadata) from non-transferable preferences (learned online), 
BanditGPT achieves effective routing without calibration costs or negative 
transfer risks.
```

---

## Figure Caption for Metadata Diagram

If you want to add a visual diagram showing metadata initialization:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{metadata_initialization_diagram.pdf}
\caption{\textbf{Metadata-Guided Cold-Start Initialization.} Each model $m$ is 
initialized with (1) isotropic uncertainty $A_m = \lambda I$ to encourage exploration, 
and (2) metadata embedding $b_m$ to encode known specifications (cost, context, 
benchmarks). Hard constraints (red) are enforced via pre-selection. Quality 
preferences (blue) are learned online, avoiding negative transfer from sparse 
offline data (RQ1).}
\label{fig:metadata_init}
\end{figure}
```

---

## Key Terminology

Throughout the paper, use consistent terminology:

✅ **Use:**
- "Metadata-guided initialization"
- "Zero-benchmark deployment"
- "Cold-start with metadata"
- "Online learning from live interactions"

❌ **Avoid:**
- "Shippable Priors"
- "Pre-trained covariance"
- "Expert distillation" (as a recommended approach)
- "Warm-start advantage"

---

## Bottom Line for Methods Section

The metadata initialization section should communicate:

1. ✅ **How it works:** Encode metadata → Initialize $b_m$, keep $A_m = I$
2. ✅ **Why it works:** Separates constraints (transferable) from preferences (learned)
3. ✅ **What's different:** Avoids negative transfer from sparse offline calibration
4. ✅ **When to use it:** Default for all deployments (vs. custom priors for >10K data)

This positions metadata initialization as the **scientifically validated default**, 
not a "basic mode" that should be upgraded.

