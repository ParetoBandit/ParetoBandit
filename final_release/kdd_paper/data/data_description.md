# Data Methodology and Description

This section describes the datasets used to develop, warm-start, and evaluate the Bandit Router. Our data strategy focuses on three pillars: a comprehensive model registry, high-quality benchmark priors, and a leakage-free evaluation set.

## 1. Model Registry (`models.json`)

The core of the router is a curated registry of **80 Large Language Models (LLMs)**. For each model, we maintain the following metadata:

*   **Quality (HLE Score)**: Obtained from the **Artificial Analysis API**. The Humanity's Last Exam (HLE) benchmark provides a robust measure of model reasoning and knowledge. Our final registry achieves 100% coverage for HLE scores across all 80 models.
*   **Cost**: Input and output costs per million tokens, sourced from the **OpenRouter API**.
*   **Latency**: Real-world performance metrics including Time to First Token (TTFT) and Output Tokens Per Second (OTPS). While we utilize **Artificial Analysis** for baseline comparisons, our production latency data is obtained via a custom sampling script (`scripts/fetch_openrouter_latency.py`) that performs the following:
    *   **Sampling Strategy**: 100 independent API calls per model to OpenRouter with a 0.1-second delay between samples to capture variance.
    *   **TTFT Measurement**: Measured using the streaming API, recording the delta between the initial request start and the arrival of the first token.
    *   **OTPS Calculation**: Calculated by counting tokens received during the streaming session and dividing by the total generation time (excluding TTFT).

## 2. HLE Priors Dataset

To "warm-start" the bandit and avoid the "cold start" problem, we calculated a covariance matrix and sum vector from a large-scale prompt dataset.

*   **Source**: A subset of **33,000 prompts** from the LMSYS Chatbot Arena dataset.
*   **Deduplication**: We performed rigorous deduplication, resulting in **26,223 unique prompts**. This ensures that the "expert intuition" built into the router is not biased by repeated common queries (e.g., "Hello", "Hi").
*   **Embeddings**: All prompts were embedded into a 384-dimensional space using the `sentence-transformers/all-MiniLM-L6-v2` model.
*   **Leakage Prevention**: We explicitly excluded all prompts present in our evaluation set from this training set to ensure the integrity of our experiments.

## 3. Evaluation Dataset

The performance of the router was validated using a dedicated evaluation set with ground-truth rewards.

*   **Prompts**: A set of **496 unique prompts**, split into a training set (397) and a testing set (99) for cross-validation.
*   **Ground Truth Rewards**: For each prompt, we utilized a matrix of rewards representing the "true" quality of each model's response. These rewards were derived from historical LLM-as-a-judge evaluations.
*   **5-Fold Cross-Validation**: Our experiments utilize 5-fold cross-validation across this dataset to provide statistically robust measures of regret reduction.

## 4. Data Acquisition Summary

| Data Type | Source | Purpose |
| :--- | :--- | :--- |
| Model Benchmarks | Artificial Analysis API | Quality Priors (HLE) |
| Model Pricing | OpenRouter API | Cost Optimization |
| Model Latency | OpenRouter / Artificial Analysis | Latency Optimization |
| Prior Prompts | LMSYS Chatbot Arena | Warm-start Covariance |
| Evaluation Prompts | Internal Dataset | Performance Validation |

By combining real-world performance metrics with large-scale prompt embeddings, the Bandit Router achieves a **16.84% ± 4.76% regret reduction** out-of-the-box on unseen data.
