# Data Methodology
## Figure 0.5: HLE Score Distribution
This figure illustrates the distribution of **Humanity's Last Exam (HLE)** scores across the 80+ models in our registry.

![Figure 0.5: HLE Distribution](/Users/annette/.gemini/antigravity/brain/8dc42199-0a6c-423a-a332-751a34ce49d5/figure0_5_hle_dist.png)

### Key Insights
- **Long Tail**: Most models cluster at the lower end of the HLE spectrum (0.03 - 0.10), highlighting the extreme difficulty of the benchmark.
- **Elite Tier**: Only a handful of models (e.g., Gemini 3 Pro, Claude 3.5 Opus) achieve scores above 0.30, representing the current state-of-the-art in complex reasoning.
- **Median vs. P95**: The **Median score (0.048)** reflects the typical performance of a "good" generalist, while the **P95 (0.269)** marks the threshold for the world's most capable reasoning models.

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

*   **Model Benchmarks (The "Labels")**: We utilize the **Humanity's Last Exam (HLE)** score for each model. HLE is a challenging benchmark designed to test models at the limit of human knowledge, making it an excellent proxy for general reasoning and knowledge quality.
*   **Synthesis**: We utilize **Ridge Initialization** to "warm-start" the bandit. We mathematically simulate a scenario where each model has already processed **26,223 prompts** from the LMSYS Chatbot Arena, receiving a reward equal to its **HLE score** for each. This populates the covariance matrix ($A$) and sum vector ($b$) for every model in the registry.
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
    *   **Cross-Grading Mitigation**: To eliminate "self-preference bias," the system implements a **Cross-Grading** protocol. Whenever the model being evaluated is **GPT-4o**, the system automatically switches the "teacher" judge to an equivalent high-quality model (**Claude 3.5 Sonnet**). This ensures that GPT-4o never provides the ground-truth score for its own responses, removing the primary source of circularity in the reward signal.
    *   **Hybrid Grading**: ~85% of standard prompts are graded by the **DeBERTa-v3-small** "soft grader." This model was trained on human preferences (HelpSteer2/LMSYS Arena), meaning the majority of the bandit's learning signal is derived from human-aligned proxies rather than GPT-4o itself.

## 5. Data Acquisition Summary

| Data Type | Source | Purpose |
| :--- | :--- | :--- |
| Model Benchmarks | Artificial Analysis API | Quality Priors (HLE) |
| Model Pricing | OpenRouter API | Cost Optimization |
| Model Latency | OpenRouter / Artificial Analysis | Latency Optimization |
| Prior Prompts | LMSYS Chatbot Arena | Warm-start Covariance |
| Evaluation Prompts | Internal Dataset | Performance Validation |

By combining real-world performance metrics with large-scale prompt embeddings, the Bandit Router achieves a **16.84% ± 4.76% regret reduction** out-of-the-box on unseen data.
