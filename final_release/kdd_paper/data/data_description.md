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

## 2. HLE Priors Methodology

The "HLE Priors" are not a raw dataset, but rather a set of statistical initializations ($A$ and $b$ matrices) that combine model benchmarks with prompt embeddings. This allows the router to start with "expert intuition" rather than a cold start.

*   **Model Benchmarks (The "Labels")**: We use the HLE scores from **Artificial Analysis** as the ground-truth average performance for each model.
*   **Prompt Dataset (The "Context")**: We use a deduplicated subset of **26,223 prompts** from the LMSYS Chatbot Arena to map the semantic space of user queries.
*   **Synthesis**: We utilize **Ridge Initialization** to "warm-start" the bandit. We mathematically simulate a scenario where each model has already processed all 26,223 prompts, receiving a reward equal to its HLE score for each. This populates the covariance matrix ($A$) and sum vector ($b$) for every model in the registry.
*   **Leakage Prevention**: All 496 prompts used in our evaluation were explicitly removed from the 26,223-prompt training set to ensure zero data leakage.

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
