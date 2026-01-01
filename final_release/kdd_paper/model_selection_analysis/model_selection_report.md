# Model Selection Analysis for KDD Paper

## Objective
To strictly filter the pool of LLMs used in our bandit routing experiments. We aim to retain only those models that provide competitive value in terms of Cost, Quality (Success Rate per Prompt Cluster), or Latency. Models that are **universally dominated**—meaning they are worse in Cost, Latency, and Quality across *all* 100 prompt clusters—are candidates for removal.

## Methodology

### 1. Pareto Dominance Analysis
We analyzed the entire backup pool of 43 models (`models.json.bak`). For each model $M$, we checked if there exists *any* other model $M'$ in the pool such that for *every* prompt cluster $C_i$:
*   $Cost(M') \le Cost(M)$
*   $Latency(M') \le Latency(M)$
*   $Quality(M', C_i) \ge Quality(M, C_i)$
*   With strictly better performance in at least one metric.

Models satisfying this condition are "Always Dominated" and theoretically provide no utility to an optimal router.

### 2. Throughput / Reliability Check
Since some dominated models might serve as high-throughput "Reliability/Overflow" backups (handling 429 Rate Limits), we performed a secondary specific output throughput check. If a dominated model offers exceptionally high throughput compared to the baseline, it may be retained.

## Results

**Total Models Analyzed:** 43

### Identified Dominated Models
The following 7 models were found to be dominated in EVERY prompt cluster:
1.  `google/gemini-2.5-pro-preview-06-05`
2.  `meta-llama/llama-3.1-405b-instruct`
3.  `amazon/nova-lite-v1`
4.  `google/gemini-2.5-flash-preview-09-2025`
5.  `google/gemini-3-pro-preview`
6.  `google/gemini-2.5-flash-lite`
7.  `Nova Pro`

### Throughput Analysis
We compared the throughput of these dominated models against the average of non-dominated models to verify if they offer value as overflow arms.

**Baseline Statistics (Non-Dominated Models):**
*   **Average Throughput:** 224.40 tokens/s
*   **Max Throughput:** 3,646.83 tokens/s

| Model Name | Throughput (tok/s) | vs Baseline Avg | Verdict |
| :--- | :--- | :--- | :--- |
| **google/gemini-2.5-flash-lite** | 338.94 | +51.0% | **Delete** (10x slower than Max) |
| **google/gemini-2.5-flash-preview** | 249.70 | +11.3% | **Delete** (Marginal improvement) |
| **amazon/nova-lite-v1** | 156.10 | -30.4% | **Delete** (Below Average) |
| **google/gemini-2.5-pro-preview** | 138.15 | -38.4% | **Delete** (Below Average) |
| **google/gemini-3-pro-preview** | 109.36 | -51.3% | **Delete** (Below Average) |
| **Nova Pro** | 70.50 | -68.6% | **Delete** (Poor Performance) |
| **meta-llama/llama-3.1-405b** | 24.57 | -89.0% | **Delete** (Extremely Slow) |

## Conclusion and Action
All 7 dominated models fail to provide either Pareto-optimal utility or significant throughput advantages. They should be **removed** from the final `models.json` used for the KDD paper experiments, as they add noise without contributing to the optimal frontier or system reliability.
