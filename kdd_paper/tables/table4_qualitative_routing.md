# Table 4: Qualitative Routing Analysis

Selected examples showing the router's decision logic.
**Scores denote estimated quality probability [0-1].**

| Task Category | Prompt Snippet | GPT-4o | Nova-Lite | Routing Logic |
|---------------|----------------|--------|-----------|---------------|
| SQL Generation | *"Generate a SQL query...."* | 0.00 | **1.00** | ✓ Specialist Win: Nova excels at rigid syntax |
| Creative Writing | *"You are acting as a recruiter for  Elite..."* | 0.00 | **0.90** | ✓ Specialist Win: Template prompts favor focused models |
| Math / Reasoning | *"What is the definite integral of sin(x)/..."* | **1.00** | 0.00 | ⚠ Fallback: Complex logic requires large-model reasoning |
| Domain Expert | *"You are an expert in regulating heat net..."* | **1.00** | 0.00 | ⚠ Fallback: Niche knowledge needs teacher model |
| General Q&A | *"why do birds chirp?..."* | 0.75 | 0.78 | 💰 Efficiency: Equal quality → 98% cheaper |

## Summary Statistics (Real Data)

| Category | Clusters | Interpretation |
|----------|----------|----------------|
| ✓ Specialist Wins | 28 | Nova-Lite > GPT-4o |
| ⚠ Teacher Fallback | 35 | GPT-4o essential |
| 💰 Equal Quality | 226 | 97.6% cost savings |

---

## LaTeX (Camera-Ready)

```latex
\begin{table*}[h]
\centering
\small
\caption{\textbf{Qualitative Routing Analysis.} Selected examples demonstrating the router's decision logic on the RQ1 dataset. Scores denote estimated quality probability [0--1]. The router successfully distinguishes between rigid syntactic tasks (Specialist Wins) and complex reasoning tasks (Teacher Fallbacks).}
\label{tab:qualitative_examples}
\begin{tabular}{lp{4.5cm}ccp{5.5cm}}
\toprule
\textbf{Task Category} & \textbf{Prompt Snippet} & \textbf{GPT-4o} & \textbf{Nova-Lite} & \textbf{Routing Logic} \\
\midrule
\textbf{SQL Generation} & \textit{"Generate a SQL query..."} & 0.00 & \textbf{1.00} & \textbf{Specialist Win:} Nova excels at rigid syntax and structured output generation. \\
\textbf{Creative Writing} & \textit{"You are acting as a recruiter..."} & 0.00 & \textbf{0.90} & \textbf{Specialist Win:} Template-based prompts favor focused models over generalists. \\
\midrule
\textbf{Math / Reasoning} & \textit{"What is the definite integral..."} & \textbf{1.00} & 0.00 & \textbf{Fallback:} Complex symbolic logic requires large-model reasoning capabilities. \\
\textbf{Domain Expert} & \textit{"You are an expert in regulating..."} & \textbf{1.00} & 0.00 & \textbf{Fallback:} Niche knowledge retrieval requires the teacher's world-model. \\
\midrule
\textbf{General Q\&A} & \textit{"why do birds chirp?..."} & 0.75 & 0.78 & \textbf{Efficiency:} Equal quality ($\Delta < 0.05$) allows for 98\% cost reduction. \\
\bottomrule
\end{tabular}
\end{table*}
```