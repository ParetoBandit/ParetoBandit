#!/usr/bin/env python3
"""
Generate Table 1: Dataset Description and Experimental Splits

Creates a LaTeX table documenting:
- Data sources (LMSYS Arena, RouteLLM)
- Split sizes and purposes
- Why these datasets, how they're used, production impact, data integrity
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config_legacy import (
    DATA_DIR,
    ROUTELLM_BATTLES_REWARDS_PATH
)

# Data paths
DEV_PROMPTS = DATA_DIR / "dev_prompts_for_rejudge.jsonl"
HOLDOUT_PROMPTS = DATA_DIR / "holdout_prompts_for_rejudge.jsonl"
WARMUP_PROMPTS = ROUTELLM_BATTLES_REWARDS_PATH


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
Warmup          & RouteLLM Battles & """ + f"{warmup_count:,}" + r""" & PCA training (384$\rightarrow$32) + LinUCB priors ($\mathbf{A}$, $\mathbf{b}$) \\
Development     & LMSYS Arena      & """ + f"{dev_count:,}" + r""" & Online learning \& calibration \\
Holdout         & LMSYS Arena      & """ + f"{holdout_count:,}" + r""" & Final bandit evaluation \\
\midrule
\textbf{Total} & & \textbf{""" + f"{total_count:,}" + r"""} & \\
\bottomrule
\end{tabular}

\vspace{0.8em}
\footnotesize

\textbf{Why these datasets.}
The warmup split uses RouteLLM's publicly available battle corpus (\texttt{routellm/gpt4\_judge\_battles} on HuggingFace)~\cite{ong2024routellm}, a curated collection of 80K pairwise human preferences between mixtral-8x7b-instruct and gpt-4-turbo.
We adopt this dataset for two reasons: (1)~it is the same data used by RouteLLM and other open-source routers, so our warmup priors are grounded in an established benchmark and results are directly comparable; and (2)~it provides a large, publicly reproducible source of preference signal for offline prior generation.
The evaluation splits (dev and holdout) draw from the broader LMSYS Chatbot Arena prompt pool~\cite{zheng2023lmsys}, the most widely-cited platform for LLM evaluation.
The same model pair is used throughout, so any performance differences across splits are attributable to distributional changes in prompt characteristics---not model capability differences.

\textbf{How they are used.}
The warmup data serves a dual role.
First, it trains a PCA projection (384$\rightarrow$32 dimensions) that compresses sentence embeddings into a compact context representation.
Second, it generates LinUCB prior matrices ($\mathbf{A} \in \mathbb{R}^{33 \times 33}$, $\mathbf{b} \in \mathbb{R}^{33}$) that encode 80K observations of which prompt characteristics predict model-specific quality.
Together, the PCA and priors constitute a \emph{warm start}: a pre-trained belief about prompt--model affinity that can be refined online.
The development set is used for online bandit learning and hyperparameter calibration ($\gamma$-scaling, $\alpha$-scheduling).
The holdout set provides final evaluation under standard bandit protocol~\cite{lattimore2020bandit}: the router learns and acts simultaneously, with cumulative reward measured across the full interaction sequence including the early learning curve.

\textbf{Production impact.}
The PCA model and warmup priors ship with the banditGPT library as compact artifacts ($<$1\,MB combined).
This mitigates the cold-start problem common to online learning routers: without any prior, a contextual bandit requires hundreds of observations before it can outperform random routing.
The warmup artifacts provide an informed starting point---no data collection phase, no API calls to external services, and no reliance on proprietary training corpora.
The degree of benefit depends on how well the RouteLLM battle distribution matches a given deployment setting; when the match is poor, the Corralling meta-learner (Section~\ref{sec:corralling}) adapts by shifting weight toward online-learned policies.
Cross-domain generalization of the PCA is validated in Figure~\ref{fig:lmsys_holdout_structure} (Spearman $\rho = 0.370$, $p < 0.0001$, 2.6$\times$ vs.\ random projections), confirming that the routing signal learned from RouteLLM battles transfers to unseen prompt populations.
As the router observes user-specific traffic, online updates adapt the priors to the deployment distribution at $O(d^2)$ per observation via rank-one Sherman--Morrison updates.

\textbf{Data integrity.}
The warmup and evaluation datasets are \emph{independent by provenance}---different data sources, sampling periods, and prompt populations.
Automated deduplication removed 243 incidental overlaps (0.24\%), arising from both datasets sampling the broader LMSYS user base.
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
