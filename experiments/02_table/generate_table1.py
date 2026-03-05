#!/usr/bin/env python3
"""
Generate Table 1: Dataset Description and Experimental Splits

Creates a LaTeX table documenting:
- Data sources (LMSYS Arena, offline battle corpus)
- Split sizes and purposes
- Why these datasets, how they're used, production impact, data integrity
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config import (
    DATA_DIR,
    LMSYS_BATTLES_PATH,
)

# Data paths
DEV_PROMPTS = DATA_DIR / "dev_prompts_for_rejudge.jsonl"
HOLDOUT_PROMPTS = DATA_DIR / "holdout_prompts_for_rejudge.jsonl"
WARMUP_PROMPTS = LMSYS_BATTLES_PATH


def count_prompts(file_path: Path) -> int:
    """Count prompts in a JSONL file."""
    if not file_path.exists():
        return 0

    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def generate_table(warmup_count: int, dev_count: int, holdout_count: int) -> str:
    """Generate LaTeX table with dataset documentation narrative."""
    total_count = warmup_count + dev_count + holdout_count

    return r"""
\begin{table}[t]
\centering
\caption{Dataset Description and Experimental Splits}
\label{tab:dataset}
\small
\begin{tabular}{@{}llrl@{}}
\toprule
\textbf{Split} & \textbf{Source} & \textbf{Size} & \textbf{Purpose} \\
\midrule
PCA Training    & LMSYS Arena (disjoint) & ${\sim}$46K & PCA projection (1024$\rightarrow$15) \\
Prior Training  & LMSYS Arena (dev)      & 1,028 & LinUCB warmup priors ($\mathbf{A}$, $\mathbf{b}$) \\
Online Learning & LMSYS Arena (dev)      & """ + f"{dev_count:,}" + r""" & Online bandit learning \& calibration \\
Holdout         & LMSYS Arena            & """ + f"{holdout_count:,}" + r""" & Final bandit evaluation \\
\bottomrule
\end{tabular}

\vspace{0.8em}
\footnotesize

\textbf{Unified data source.}
All splits draw from LMSYS Chatbot Arena~\cite{zheng2023judging}, the most widely-cited platform for LLM evaluation.
This same-distribution design mirrors production deployments where the practitioner trains the PCA and warmup priors on representative workload prompts.
PCA training uses ${\sim}$46K LMSYS arena prompts that are strictly disjoint from the evaluation data.
The prior-training split (1{,}028~prompts) and online-learning split (1{,}543~prompts) are drawn from the 2{,}571 development prompts with full $K{=}10$ coverage via stratified sampling, ensuring prompt-level disjointness.

\textbf{How they are used.}
PCA compresses \texttt{BAAI/bge-m3} embeddings (1024D) into a 15-dimensional context representation.
The prior-training split generates LinUCB prior matrices ($\mathbf{A} \in \mathbb{R}^{16 \times 16}$, $\mathbf{b} \in \mathbb{R}^{16}$) that encode per-model reward surfaces across the prompt embedding space.
Together, the PCA and priors constitute a \emph{warm start}: a pre-trained belief about prompt--model affinity that can be refined online.
The development set is used for online bandit learning and hyperparameter calibration ($\gamma$-scaling, $\alpha$-scheduling).
The holdout set provides final evaluation under standard bandit protocol~\cite{lattimore2020bandit}: the router learns and acts simultaneously, with cumulative reward measured across the full interaction sequence including the early learning curve.

\textbf{Production impact.}
The PCA model and warmup priors ship with the banditGPT library as compact artifacts ($<$1\,MB combined).
This mitigates the cold-start problem common to online learning routers: without any prior, a contextual bandit requires hundreds of observations before it can outperform random routing.
When the match between warmup and deployment traffic is poor, the Corralling meta-learner (Section~\ref{sec:corralling}) adapts by shifting weight toward online-learned policies.
As the router observes user-specific traffic, online updates adapt the priors to the deployment distribution at $O(d^2)$ per observation via rank-one Sherman--Morrison updates.

\textbf{Data integrity.}
All three experimental splits (prior-training, online-learning, holdout) are verified to be prompt-disjoint by automated leakage checks at split creation time and again at experiment runtime.
Dev and holdout sets were created via stratified sampling across category, complexity, and difficulty to ensure representative coverage.

\end{table}
"""


def main():
    warmup_count = count_prompts(WARMUP_PROMPTS)
    dev_count = count_prompts(DEV_PROMPTS)
    holdout_count = count_prompts(HOLDOUT_PROMPTS)

    latex = generate_table(warmup_count, dev_count, holdout_count)
    output_file = Path(__file__).parent / "table1_dataset.tex"
    with open(output_file, 'w') as f:
        f.write(latex)


if __name__ == "__main__":
    main()
