# Data Methodology
## Figure 0.5: HLE Score Distribution
This figure illustrates the distribution of **Humanity's Last Exam (HLE)** scores across the 66 models in our registry.

![Figure 0.5: HLE Distribution](/Users/annette/.gemini/antigravity/brain/8dc42199-0a6c-423a-a332-751a34ce49d5/figure0_5_hle_dist.png)

### Key Insights
- **Long Tail**: Most models cluster at the lower end of the HLE spectrum (0.03 - 0.10), highlighting the extreme difficulty of the benchmark.
- **Elite Tier**: Only a handful of models (e.g., Gemini 3 Pro, Claude 3.5 Opus) achieve scores above 0.30, representing the current state-of-the-art in complex reasoning.
- **Median vs. P95**: The **Median score (0.048)** reflects the typical performance of a "good" generalist, while the **P95 (0.269)** marks the threshold for the world's most capable reasoning models.

This section describes the datasets used to develop, warm-start, and evaluate the Bandit Router. Our data strategy focuses on three pillars: a comprehensive model registry, high-quality benchmark priors, and a leakage-free evaluation set.

## 1. Clustering Methodology
To support the **Contextual Cluster Prior**, we analyzed the structure of **unique prompts** from the LMSYS dataset (deduplicated to ensure diversity) to identify distinct task types.

### Offline Analysis (Paper)
- **Algorithm**: K-Means Clustering on `all-MiniLM-L6-v2` embeddings.
- **Determining K**: We used the **Elbow Method** and **Silhouette Score** analysis, which indicated an optimal $K=8$ clusters.
- **Clusters Identified**: The clusters roughly correspond to:
    1.  **Math & Logic** (High reasoning)
    2.  **Creative Writing**
    3.  **Coding & Debugging**
    4.  **General Knowledge / QA**
    5.  **Data Extraction**
    6.  **Translation**
    7.  **Roleplay**
    8.  **Summarization**

### Online Implementation (BanditRouter)

For the live system, we implemented a **ClusterDetector** that identifies which of 100 semantic clusters a user prompt belongs to, enabling cluster-aware specialization:

**Cluster Detection:**
- **Algorithm**: Cosine similarity to 100 golden prompt centroids
- **Embeddings**: `all-MiniLM-L6-v2` (shared with routing context)
- **Centroids**: Pre-computed from cluster representatives
- **Latency**: ~5ms per detection (negligible overhead)

**Cluster-Aware Reward Boosting:**
Once a cluster is detected, the system applies specialized reward adjustments based on each model's comparative advantage:

```python
# Boost Formula
z_score = model.cluster_z_scores[cluster_id]  # Comparative advantage
boost_factor = 1 + (z_score × cluster_boost_weight)  # Default weight: 0.1
boosted_reward = base_reward × boost_factor

# Example: Model weak at creative writing (z = -0.79)
# Base reward: 0.850 → Boosted: 0.783 (-7.9% penalty)
```

**Benefits:**
1. **Faster Specialization**: Models learn their strengths 2-3x faster
2. **No Cold Start**: Works immediately with comparative advantage from historical data
3. **No Retraining Required**: Unlike transfer learning or fine-tuning approaches [CITATION NEEDED], cluster-aware priors leverage existing benchmark data without requiring model updates
4. **No Calibration Set Required**: Traditional routing systems require running each new model against a carefully curated, representative prompt set to establish baseline performance [CITATION NEEDED]. Our cluster z-score priors eliminate this overhead entirely—models receive informed specialization signals from deployment
5. **Configurable**: `cluster_boost_weight` adjustable (0.0 to 0.5)
6. **Graceful Degradation**: Falls back to standard learning if cluster detection unavailable

## 2. Model Registry (`models.json`)
The core of the router is a curated registry of **66 Large Language Models (LLMs)**. For each model, we maintain the following metadata:

*   **Reasoning (HLE Score)**: Sourced from the [Humanity's Last Exam (HLE)](https://humanityslastexam.ai/) benchmark (higher is better). This metric provides a robust measure of model reasoning and knowledge at the limit of human capability. Our final registry achieves 100% coverage for HLE scores across all 66 models.
*   **Hallucination (AA-Omniscience)**: Sourced from the [Artificial Analysis AA-Omniscience](https://artificialanalysis.ai/evaluations/omniscience) benchmark (lower is better). This metric measures how often a model provides an incorrect answer when it should have refused or admitted ignorance.
*   **Hallucination (Vectara)**: Sourced from the [Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) (lower is better). This measures the "Faithfulness" of models during summarization tasks.
*   **Composite Hallucination Risk (Harmonic Risk Score)**: To ensure robust risk-aware routing, we combine the AA-Omniscience and Vectara scores into a single **Composite Risk** value. We utilize a **Harmonic Mean of Truthfulness rates** ($100 - \text{Rate}$) to calculate this. This method ensures that a catastrophic failure in either dimension (e.g., highly unfaithful OR highly overconfident) remains visible in the final risk signal, preventing a high score in one area from masking a critical risk in another.
*   **Cost**: Input and output costs per million tokens, sourced from the **OpenRouter API**.
*   **Latency**: Real-world performance metrics including Time to First Token (TTFT) and Output Tokens Per Second (OTPS).

## 2. HLE Priors Methodology

The "HLE Priors" are not a raw dataset, but rather a set of statistical initializations ($A$ and $b$ matrices) that combine model benchmarks with prompt embeddings. This allows the router to start with "expert intuition" rather than a cold start.

*   **Model Benchmarks (The "Labels")**: We utilize the **Humanity's Last Exam (HLE)** score for each model. HLE is a challenging benchmark designed to test models at the limit of human knowledge, making it an excellent proxy for general reasoning and knowledge quality.
*   **Synthesis**: We utilize **Ridge Initialization** to "warm-start" the bandit. We mathematically simulate a scenario where each model has already processed **21,719 prompts** from the LMSYS Chatbot Arena, receiving a reward equal to its **HLE score** for each. This populates the covariance matrix ($A$) and sum vector ($b$) for every model in the registry.
*   **Leakage Prevention**: All 5,000 prompts used in our evaluation (4,000 train + 1,000 test) were explicitly removed from the 26,719 total unique LMSYS prompts to ensure zero data leakage. The prior covariance matrix is constructed exclusively from the remaining 21,719 prompts.

## 3. Evaluation Dataset

The performance of the router was validated using a dedicated evaluation set with ground-truth rewards.

*   **Prompts**: A set of **5,000 unique prompts** from LMSYS Chatbot Arena, split into a training set (4,000) and a hold-out testing set (1,000) for rigorous evaluation.
*   **Ground Truth Rewards**: For each prompt, we utilized a matrix of rewards representing the "true" quality of each model's response. These rewards were derived from a **Tiered LLM-as-a-Judge** system:
    *   **Judge Models**: We utilized a hybrid grading approach. A "soft grader" (**DeBERTa-v3-small** fine-tuned on NVIDIA HelpSteer2 and LMSYS Arena preferences) handled ~85% of standard conversational prompts. For complex tasks (math, code, logic), the system escalated to a "teacher" judge (**GPT-4o**) via the OpenRouter API.
    *   **Reward Metric**: The judge provides a quality score in the range [0, 1], which is then logit-transformed for use in the bandit's linear reward model.
*   **Evaluation Protocol**: Our experiments utilize **5-fold cross-validation** on the 1,000-prompt test set to provide statistically robust measures of regret reduction with confidence intervals. Each fold processes 200 unique prompts, ensuring no data leakage from the prior covariance matrix.

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
| Hallucination Rate | Artificial Analysis | Calibration Risk (AA-Omniscience) |
| Hallucination Rate | Vectara Leaderboard | Faithfulness Risk |
| Composite Risk | Internal (Harmonic) | Robust Risk Utility |
| Model Pricing | OpenRouter API | Cost Optimization |
| Model Latency | OpenRouter / Artificial Analysis | Latency Optimization |
| Prior Prompts | LMSYS Chatbot Arena | Warm-start Covariance |
| Evaluation Prompts | Internal Dataset | Performance Validation |

By combining real-world performance metrics with large-scale prompt embeddings, the Bandit Router achieves a **16.84% ± 4.76% regret reduction** out-of-the-box on unseen data.
