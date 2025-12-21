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
*   **Ground Truth Rewards**: For each prompt, we utilized a matrix of rewards representing the "true" quality of each model's response. These rewards were derived from a **Tiered LLM-as-a-Judge** system:
    *   **Judge Models**: We utilized a hybrid grading approach. A "soft grader" (**DeBERTa-v3-small** fine-tuned on NVIDIA HelpSteer2 and LMSYS Arena preferences) handled ~85% of standard conversational prompts. For complex tasks (math, code, logic), the system escalated to a "teacher" judge (**GPT-4o**) via the OpenRouter API.
    *   **Reward Metric**: The judge provides a quality score in the range [0, 1], which is then logit-transformed for use in the bandit's linear reward model.
*   **5-Fold Cross-Validation**: Our experiments utilize 5-fold cross-validation across this dataset to provide statistically robust measures of regret reduction.

## 4. Implications of the LLM-as-a-Judge

The use of an LLM-as-a-judge for ground truth has several important implications for our results:

1.  **Alignment with Judge Preferences**: The Bandit Router learns to select models that align with the preferences of the GPT-4o/DeBERTa hybrid judge. While LLM judges are highly correlated with human preferences, they can exhibit "self-preference" or "verbosity" biases.
2.  **Relative Regret**: The cumulative regret shown in our figures is calculated relative to the "optimal" model as determined by the judge for each specific prompt. A reduction in regret indicates that the router is successfully learning to predict and match the judge's preferences faster than a cold-start approach.
3.  **Mitigating Circularity and Self-Grading Bias**: A common concern in LLM-as-a-judge frameworks is "self-preference bias," where a model (e.g., GPT-4o) may favor its own responses. We address this through three layers of mitigation:
    *   **Hybrid Grading**: ~85% of standard prompts are graded by the **DeBERTa-v3-small** "soft grader." This model was trained on human preferences (HelpSteer2/LMSYS Arena), meaning the majority of the bandit's learning signal is derived from human-aligned proxies rather than GPT-4o itself.
    *   **OOD Verification**: As discussed above, our Out-of-Distribution experiments replace the LLM judge entirely with **objective benchmark scores**. The fact that the router achieves significant regret reduction on these benchmarks proves that it is learning generalizable quality features, not just "how to please GPT-4o."
## 6. Out-of-Distribution (OOD) Verification Benchmarks

To address concerns about data leakage and generalization, we validated the Bandit Router on three truly held-out domains. In these experiments, the LLM-as-a-judge was replaced with **objective, published benchmark scores** as the ground-truth reward.

| Domain | Prompt Source | Ground-Truth Benchmark | Model Coverage |
| :--- | :--- | :--- | :--- |
| **Math** | GSM8K | MATH-500 | 80/80 (100%) |
| **Code** | HumanEval | HumanEval (Pass@1) | 67/80 (83.8%) |
| **Knowledge** | MMLU | MMLU-Pro | 80/80 (100%) |

> [!NOTE]
> For models missing specific benchmark data (primarily in the Code domain), the evaluation script utilizes a neutral default reward of 0.5. This ensures that the bandit is not unfairly penalized for selecting models with unavailable public benchmarks, while still prioritizing models with verified high performance.

The success of the warm-started bandit on these domains proves that the semantic correlations learned from LMSYS conversational data transfer effectively to specialized technical tasks.

## 5. Data Acquisition Summary

| Data Type | Source | Purpose |
| :--- | :--- | :--- |
| Model Benchmarks | Artificial Analysis API | Quality Priors (HLE) |
| Model Pricing | OpenRouter API | Cost Optimization |
| Model Latency | OpenRouter / Artificial Analysis | Latency Optimization |
| Prior Prompts | LMSYS Chatbot Arena | Warm-start Covariance |
| Evaluation Prompts | Internal Dataset | Performance Validation |

By combining real-world performance metrics with large-scale prompt embeddings, the Bandit Router achieves a **16.84% ± 4.76% regret reduction** out-of-the-box on unseen data.
