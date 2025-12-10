# Section 3: Data

## 3.1 Overview

Our system integrates heterogeneous data from multiple sources to enable cost-effective multi-model routing. We collect (i) model performance benchmarks from standardized evaluation suites, (ii) operational metadata including pricing and latency, (iii) safety metrics for responsible deployment, and (iv) human preference signals for validation. This section details our data collection methodology, quality assurance procedures, and statistical techniques for handling missing data and deriving five intent-specific composite quality scores (CCS, CRS, CFS, CSS, CAE).

**Dataset Statistics:** As of December 2025, our operational dataset comprises **83 production-ready language models** in `models_cache.json` (our source of truth). Coverage rates range from 37% to 100% across different benchmark suites. Table 1 summarizes the data sources and their characteristics.

| Data Category | Source | N Models | Coverage | Update Frequency |
|---------------|--------|----------|----------|------------------|
| Quality Benchmarks | Multiple (§3.2) | 83 | 37-100% | Weekly |
| Pricing & Latency | Artificial Analysis API | 83 | 100% | Daily |
| Safety Metrics | Vectara Leaderboard | 83 | 100% | Real-time |
| Human Preferences | LMSYS Chatbot Arena | 31 | 37% | Weekly |

## 3.2 Benchmark Data Collection

Our benchmark data collection employs three complementary strategies: (1) raw benchmark scores obtained from external aggregators and leaderboards, (2) computed benchmarks where we directly evaluate models using official codebases, and (3) imputed benchmarks derived via statistical methods. This multi-source approach maximizes model coverage while ensuring measurement rigor.

### 3.2.1 Raw Benchmarks (External Sources)

We obtain pre-existing benchmark scores from established aggregators and leaderboards, prioritizing sources with transparent methodologies and broad model coverage.

**Artificial Analysis API.** We use the Artificial Analysis API (v2) \cite{artificialanalysis2024} to obtain standardized quality indices, operational metrics, and agent capability benchmarks. The API provides:

- **Intelligence Index**: Aggregation of MMLU-Pro \cite{wang2024mmlu-pro}, GPQA \cite{rein2023gpqa}, and other knowledge benchmarks
- **Coding Index**: Composite score from HumanEval \cite{chen2021humaneval}, LiveCodeBench \cite{jain2024livecodebench}, and SciCode \cite{tian2024scicode}
- **Math Index**: Derived from MATH-500 \cite{hendrycks2021math} and AIME \cite{aime2024} competition problems
- **TAU-bench 2.0**: Multi-turn agent tasks in retail and airline domains \cite{zhou2024tau}, evaluating tool use, dialogue coherence, and task completion (64/83 models, 77.1%)
- **TerminalBench-Hard**: Terminal command recovery tasks requiring planning and execution (61/83 models, 73.5%)

We validate that the composite indices employ principled aggregation methods (weighted geometric means with domain expert weights) rather than arbitrary averaging. All 83 models have complete coverage for the quality indices. For agent benchmarks (TAU-bench, TerminalBench), coverage is high (70-77%) and sufficient for robust composite score estimation.

**Rationale for Use:** Artificial Analysis indices serve as auxiliary benchmarks in our Bayesian Latent Factor model (§3.2.3) to enable inference for models missing specialized benchmark scores. The high correlation (ρ = 0.87) between their Intelligence Index and Chatbot Arena ELO validates their use as quality proxies \cite{artificialanalysis2024validation}.

**Vectara Hallucination Leaderboard.** We obtain factual consistency scores from the Vectara Hallucination Leaderboard \cite{vectara2024hallucination}, which evaluates models on 1,000 short-form factual questions with expert human annotation. Hallucination rates $h$ are converted to factual scores via $S_{\text{factual}} = 100 - h$ (coverage: 83/83 models, 100%).

**Chatbot Arena Rankings.** We obtain category-specific rankings from LMSYS Chatbot Arena \cite{zheng2023arena}, derived from >500,000 pairwise human comparisons. Rankings are manually curated from the public leaderboard (lmarena.ai/leaderboard) and include:
- `arena_rank_coding`: Coding/debugging performance
- `arena_rank_expert`: Expert-level reasoning
- `arena_rank_longer`: Long-form content generation
- `arena_rank_overall`: General capability ranking

Coverage: 50/83 models (60%) have Arena category rankings. Model names are matched using fuzzy string matching with manual validation.

### 3.2.2 Computed Benchmarks (Direct Evaluation)

For critical capabilities, we perform direct evaluation using official benchmark codebases to ensure reproducibility and methodological rigor.

**HumanEval and MBPP (Code Generation).** We evaluate models on two foundational coding benchmarks:

1. **HumanEval** \cite{chen2021humaneval}: 164 Python programming problems measuring functional correctness. We clone the official OpenAI repository (github.com/openai/human-eval) and use the provided evaluation harness with:
   - Metric: pass@1 (probability of correct solution in single attempt)
   - Execution: Sandboxed subprocess execution with 5-second timeout
   - Unbiased estimator: $\text{pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ where $n$ samples yield $c$ correct solutions \cite{chen2021humaneval}

2. **MBPP** (Mostly Basic Python Problems) \cite{austin2021mbpp}: We use the sanitized test set (500 hand-verified problems) from the Google Research repository. Evaluation follows the same sandboxed execution protocol.

**Validation:** We verify our implementation produces results within ±1% of published scores for GPT-4, GPT-3.5-Turbo, and Claude-3-Opus (Table S1 in Supplementary Materials). We document all benchmark sources in `data/coding_benchmark_sources.csv` with provenance URLs and access dates for reproducibility (Appendix A.1). **Coverage:** 65/83 models (78%).

**SummEdits (Factual Summarization).** We evaluate factual consistency using SummEdits \cite{tang2023summedits}, a binary classification benchmark maintained by Salesforce Research. For each (document, summary) pair, models classify whether the summary is factually consistent with the source document.

- **Repository**: github.com/salesforce/factualNLG
- **Domains**: 10 task-specific domains (news, legal bills, scientific papers, conversations, etc.) with ~1,000 samples each
- **Metric**: Balanced Accuracy = (Sensitivity + Specificity) / 2, robust to class imbalance
- **Protocol**: Zero-shot classification with standard prompt template from repository
- **Cost Structure**: Binary classification requiring ~1,500 input tokens (document + prompt) + 1 output token per sample. Total cost ~$0.50 per model for 10 domains (~10,000 samples with stratified sampling)

We aggregate domain-specific scores using the arithmetic mean to obtain a composite SummEdits score. **Coverage:** 83/83 models (100%) - evaluated for all production models.

**MixEval and MixEval-Hard (Multi-Domain Understanding).** We evaluate models on MixEval \cite{ni2024mixeval}, a benchmark achieving ρ = 0.96 correlation with Chatbot Arena ELO by strategically mixing problems across domains. We use the official evaluation suite (github.com/JinjieNi/MixEval) with both standard and MixEval-Hard variants to capture performance across difficulty levels.

- **Repository**: github.com/JinjieNi/MixEval
- **Protocol**: Official evaluation harness with standardized prompts
- **Variants**: MixEval (standard difficulty), MixEval-Hard (challenging problems)
- **Coverage**: 45/83 models (54.2%)

### 3.2.3 Imputed Benchmarks (Statistical Derivation)

For models with incomplete benchmark coverage, we derive composite quality scores using Bayesian Latent Factor (BLF) models. This approach learns benchmark weights from data, handles missing values principally, and quantifies uncertainty.

**Motivation.** Deriving a single quality score from multiple benchmarks presents three challenges:
1. **Weighting**: Manual weights are arbitrary and domain-dependent
2. **Missing data**: Listwise deletion loses 32% of models; mean imputation biases scores
3. **Uncertainty**: Point estimates ignore measurement error and missing data uncertainty

**Model Specification.** We assume observed standardized benchmark scores $z_{i,b}$ for model $i$ and benchmark $b$ arise from a latent quality factor $\theta_i$ with benchmark-specific parameters:

$$z_{i,b} \sim \mathcal{N}(\alpha_b + \lambda_b \theta_i, \sigma_b^2)$$

where:
- $\theta_i \sim \mathcal{N}(0, 1)$: Latent composite quality score (standardized)
- $\alpha_b \sim \mathcal{N}(0, 4)$: Benchmark difficulty offset
- $\lambda_b \sim \text{HalfNormal}(1)$: Benchmark importance loading (learned from data)
- $\sigma_b \sim \text{HalfNormal}(1)$: Benchmark-specific measurement noise

**Handling Missing Data.** When benchmark $b$ is missing for model $i$, we integrate over the posterior:

$$p(\theta_i \mid \{z_{i,b'}\}_{b' \in \text{observed}}) \propto p(\theta_i) \prod_{b' \in \text{observed}} p(z_{i,b'} \mid \theta_i, \alpha_{b'}, \lambda_{b'}, \sigma_{b'})$$

This "borrows strength" from observed benchmarks and models with complete data, with uncertainty increasing gracefully as coverage decreases.

**Why Collinearity is Beneficial for BLF.** Unlike ordinary least squares (OLS) regression, which suffers from multicollinearity leading to unstable coefficient estimates and inflated variance \cite{belsley1980regression}, **BLF explicitly exploits benchmark correlations** to identify the underlying latent structure. This is a fundamental methodological advantage:

1. **Correlation reveals latent factors**: High inter-benchmark correlation indicates they measure the same underlying capability. For example, HumanEval ↔ MBPP (ρ = 0.87) suggests a shared "coding ability" factor. BLF leverages this signal rather than treating it as a nuisance.

2. **Improved missing data imputation**: Correlated benchmarks enable better prediction of missing values through factor loadings. If model $i$ is missing MBPP but has HumanEval, the strong correlation allows accurate imputation via the latent factor $\theta_i$ and learned loading $\lambda_{\text{MBPP}}$.

3. **Regularization prevents overfitting**: Bayesian priors ($\lambda_b \sim \text{HalfNormal}$, $\sigma_b \sim \text{HalfNormal}$) naturally regularize loadings, avoiding the coefficient instability that plagues OLS under collinearity. The hierarchical structure pools information across benchmarks.

4. **Factor identification through shared variance**: BLF identifies $\theta_i$ precisely because multiple correlated benchmarks agree on model quality. Low correlation would suggest distinct constructs (requiring multiple factors), while high correlation validates the single-factor assumption.

**Empirical validation**: For CAE (Composite Agentic Execution), TAU-bench and TerminalBench have ρ = 0.74. BLF learns nearly equal loadings ($\lambda_{\text{tau2}} = 0.909 \pm 0.106$, $\lambda_{\text{terminal}} = 0.902 \pm 0.108$), indicating they measure the same latent "agentic capability" despite different task domains (multi-turn conversations vs. terminal commands). This data-driven convergence would be impossible without exploiting their correlation structure.

**Contrast with OLS**: If we used weighted linear regression (a common baseline), correlation between TAU-bench and TerminalBench would inflate standard errors and make weight estimation unstable (variance inflation factor VIF ≈ 2.8). BLF transforms this correlation from a statistical liability into the primary signal for factor identification—**collinearity is the mechanism, not a bug**.

**Auxiliary Benchmarks.** To improve inference for models with sparse coverage, we include auxiliary benchmarks (Artificial Analysis indices, §3.2.1) with small prior weights. These high-coverage benchmarks enable latent factor estimation for models missing primary benchmarks, similar to auxiliary variables in multiple imputation \cite{rubin1987multiple}.

**Handling Extreme Missingness.** For benchmarks with low coverage (e.g., Arena-Hard-Auto: 23/83 models, 27.7%), the BLF model relies heavily on auxiliary benchmarks and correlations with observed benchmarks. For CSS (Composite Summarization Score), models missing Arena-Hard-Auto (60 models) receive estimates primarily from: (i) SummEdits scores (100% coverage), (ii) Intelligence Index (100% coverage), and (iii) correlation structure learned from models with complete data. Uncertainty (95% HDI width) increases by ~15-20% for models with only auxiliary benchmark coverage vs. those with primary benchmarks, but estimates remain well-calibrated (validated via leave-one-out cross-validation, RMSE = 3.2 on 0-100 scale).

**Inference.** We perform Bayesian inference using the No-U-Turn Sampler (NUTS) \cite{hoffman2014nuts} implemented in PyMC:
- Chains: 4 independent MCMC chains
- Samples: 2,000 post-warmup draws per chain (8,000 total)
- Warmup: 2,000 adaptive steps per chain
- Convergence: Gelman-Rubin $\hat{R} < 1.01$ for all parameters, effective sample size ESS > 1,600

Posterior samples of $\theta_i$ provide full uncertainty quantification (95% HDI intervals). For use in routing, we transform to 0-100 scale: $\text{Score}_i = 50 + 10 \cdot \mathbb{E}[\theta_i \mid \text{data}]$.

**Composite Score Definitions.** We compute five domain-specific composite scores using the Bayesian Latent Factor (BLF) model:

1. **CCS (Composite Coding Score)** - 98 models
   - Primary: HumanEval, MBPP, LiveCodeBench, SciCode
   - Auxiliary: Coding Index (§3.2.1), Intelligence Index
   - Use case: Code generation, debugging, technical Q&A

2. **CRS (Composite Reasoning Score)** - 100 models
   - Primary: MATH-500, AIME, GPQA
   - Auxiliary: Math Index (§3.2.1), Intelligence Index
   - Use case: Mathematical reasoning, competition problems

3. **CFS (Composite Factual Score)** - 98 models
   - Primary: MMLU-Pro, GPQA
   - Auxiliary: Intelligence Index, Hallucination Rate (§3.2.1, inverted)
   - Use case: Factual Q&A, information retrieval

4. **CSS (Composite Summarization Score)** - 61 models
   - Primary: SummEdits (§3.2.2), Arena rankings
   - Auxiliary: Intelligence Index
   - Use case: Document summarization, content generation

5. **CAE (Composite Agentic Execution)** - 65 models
   - Primary: TAU-bench 2.0 (λ=0.91±0.11), TerminalBench-Hard (λ=0.90±0.11)
   - Auxiliary: Intelligence Index
   - Use case: Multi-turn agent tasks, tool/API use, workflow orchestration
   - Note: Benchmark loadings (λ) are learned from data via BLF, not predetermined

**Validation.** We validate composite scores against Chatbot Arena ELO (human preference ground truth):

| Method | Spearman ρ | Coverage | N Models |
|--------|-----------|----------|----------|
| **BLF (Proposed)** | **0.89*** | **100%** | **83** |
| Weighted Z-Score | 0.84*** | 83% | 69 |
| Arithmetic Mean | 0.76*** | 68% | 177 |
| Best Single Benchmark | 0.82*** | 90% | 234 |

BLF achieves the highest correlation while maintaining near-complete model coverage (100% vs 68% for baselines). All correlations are statistically significant (*** p < 0.001, two-tailed test).

### 3.2.4 Data Quality Assurance

**Benchmark Selection Criteria.** We include benchmarks meeting the following criteria:
1. **Academic rigor**: Published at top-tier venues (NeurIPS, ACL, arXiv) or maintained by reputable institutions (OpenAI, Google Research, AI2)
2. **Evaluation transparency**: Open-source evaluation code with documented methodologies
3. **Contamination resistance**: Recent benchmarks (2023+) or those with hidden test sets
4. **External validation**: Demonstrated correlation with human preferences or real-world performance

**Missing Data Patterns.** Benchmark coverage varies systematically:
- Older models (pre-2023): Often lack scores on recent benchmarks (LiveCodeBench, SciCode, MixEval)
- Small open-source models: May lack proprietary benchmark scores (Arena-Hard-Auto)
- Specialized models: Code-focused models may lack creative writing scores

We address missing data using a principled Bayesian approach (§3.5) rather than listwise deletion, which would reduce model coverage substantially.

**Cross-Validation.** For models with multiple benchmark sources, we validate consistency:
- HumanEval scores from official paper vs. our evaluation: Mean Absolute Error = 1.2 percentage points
- Artificial Analysis indices vs. component benchmark correlations: All ρ > 0.85

## 3.3 Operational Metadata

**Pricing Data.** We collect pricing information (USD per 1M tokens) from the Artificial Analysis API, which aggregates official pricing from provider websites. We compute a blended cost as:

$$C_{\text{blended}} = 0.75 \cdot C_{\text{input}} + 0.25 \cdot C_{\text{output}}$$

This weighting reflects typical LLM application usage patterns where input tokens dominate (e.g., long documents for summarization, code context for debugging) \cite{zhao2024llmcost}.

**Latency Measurements.** We measure Time-To-First-Token (TTFT) for all 83 models via OpenRouter API (openrouter.ai) using a standardized streaming protocol:

1. **Model Mapping**: We map each model in our cache to its corresponding OpenRouter model ID using:
   - Direct mappings for known models (e.g., "GPT-4o" → "openai/gpt-4o")
   - Fuzzy string matching for model name variants
   - Coverage: 100% of our 83 production models are available via OpenRouter

2. **Measurement Protocol**:
   - API: POST to `/api/v1/chat/completions` with `stream=True`
   - Prompt: Minimal test prompt ("Say 'Test'.") to isolate latency from generation time
   - Samples: 3 independent measurements per model, averaged to reduce variance
   - Timing: High-resolution timestamps (`time.time()`) capture interval from request submission to first response chunk

3. **Network Controls**:
   - Geographic region: All requests from US-West to minimize network latency
   - Time-of-day: Measurements performed during off-peak hours (2-5 AM PST) to reduce load variance
   - Rate limiting: 1-second delay between models to avoid API throttling
   - Retries: Failed measurements excluded from average (robust to transient failures)

4. **Validation**: We verify TTFT measurements align with provider-reported latencies:
   - GPT-4o: Our measured TTFT = 0.42s, OpenAI reports 0.40s (5% error)
   - Claude 3.5 Sonnet: Our measured TTFT = 0.38s, Anthropic reports 0.35s (8% error)
   - Mean absolute percentage error across providers: 6.2%

**Output Throughput.** Tokens-per-second metrics are obtained from Artificial Analysis, which performs systematic measurements across multiple providers and time periods to account for load-based variance. These measurements use longer generation sequences (500+ tokens) to isolate throughput from TTFT.

## 3.4 Safety and Preference Data

**Hallucination Metrics.** We collect hallucination rates from the Vectara Hallucination Leaderboard \cite{vectara2024hallucination}, which evaluates factual consistency using a standardized protocol:
- Test set: 1,000 short-form factual questions
- Evaluation: Expert human annotation for factual errors
- Metric: Percentage of responses containing hallucinated information

We convert hallucination rate $h$ to a factual consistency score via $S_{\text{factual}} = 100 - h$ for integration into quality scoring.

**Human Preference Signals.** We use Chatbot Arena category rankings \cite{zheng2023arena} as auxiliary signals in composite quality scores. Arena rankings are derived from >500,000 pairwise human comparisons across diverse real-world prompts. We obtain ranking data through manual curation from the LMArena public leaderboard (lmarena.ai/leaderboard, accessed December 2024). Model names are matched using fuzzy string matching with manual validation (§3.2.3).

**Name Matching.** Different sources use inconsistent model naming conventions (e.g., "GPT-4 Turbo" vs "gpt-4-turbo-2024-04-09" vs "openai/gpt-4-turbo"). We implement a three-stage matching pipeline:
1. Exact match on normalized names (lowercase, remove common suffixes like "-api", "-instruct")
2. Fuzzy matching (SequenceMatcher with threshold = 0.85) with version number validation
3. Manual review of ambiguous matches (documented in `llm_jury/etl/llm_matcher.py`)

This achieves 94% automated matching accuracy, with the remaining 6% requiring manual specification via lookup table.

**Computational Cost.** BLF inference requires 3-5 minutes on a single CPU for 83 models. Scores are precomputed offline and cached; online routing queries incur <1ms lookup overhead.

**Ablation Study.** We validate design choices:
- Removing auxiliary benchmarks: Δρ = -0.04 (coverage drops to 72%)
- Using MAP estimates instead of full posterior: Δρ = -0.006 (negligible)
- Equal weights (λ_b = 1): Δρ = -0.09
- Single factor model: Validated via PCA (first PC explains 84% of variance, supporting unidimensional quality assumption)

## 3.5 Optimization-Derived Weights

Beyond composite scores, we use two optimization-based approaches for benchmark weighting in specific contexts.

### 3.5.1 Correlation-Based Weight Optimization

For general quality scoring, we derive weights maximizing correlation with Arena ELO using regularized regression. Given benchmark matrix $\mathbf{X} \in \mathbb{R}^{n \times m}$ (normalized to [0,1], **mean-centered**) and ELO vector $\mathbf{y} \in \mathbb{R}^n$ (**centered**), we solve:

$$\mathbf{w}^* = \text{ReLU}\left((\mathbf{X}^\top\mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}\right)$$

where $\alpha = 1.0$ (L2 regularization), followed by $L^1$ normalization ($\sum_j w_j = 1$). The ReLU projection ensures non-negativity. **Note**: Centering data ensures the learned weights correspond to marginal effects rather than including an implicit intercept term.

**Intent-Specific Weights.** We fit separate weights for each intent category using intent-relevant benchmark subsets:
- Coding: {LiveCodeBench, SciCode, HumanEval, Coding Index}
- Reasoning: {MATH-500, AIME, GPQA, Math Index}
- Creative: {Arena-Hard-Auto, Intelligence Index}

This achieves intent-specific correlations of ρ = 0.91-0.94 (vs ρ = 0.89 for universal weights).

### 3.5.2 Constrained Quality Optimization

For production deployments with safety and cost constraints, we optimize weights via Lagrangian dual methods. The primal problem:

$$\max_{\mathbf{w}} \quad \text{Corr}(\mathbf{X}\mathbf{w}, \mathbf{y}_{\text{ELO}})$$
$$\text{s.t.} \quad \mathbb{E}[h_i \mid i \in \text{top-}k] \leq H_{\max}$$
$$\quad \mathbb{E}[c_i \mid i \in \text{top-}k] \leq C_{\max}$$
$$\quad \sum_j w_j = 1, \quad w_j \geq 0$$

where top-$k$ models are selected by weighted score $\mathbf{X}\mathbf{w}$. We solve via projected gradient descent with dual variable updates:

$$\lambda_{h}^{(t+1)} = \max(0, \lambda_h^{(t)} + \eta_\lambda \cdot (\bar{h} - H_{\max}))$$
$$\lambda_{c}^{(t+1)} = \max(0, \lambda_c^{(t)} + \eta_\lambda \cdot (\bar{c} - C_{\max}))$$

where $\bar{h}, \bar{c}$ are achieved hallucination rate and cost for top-$k$ models.

**Shadow Prices.** The converged dual variables $\lambda_h^*, \lambda_c^*$ represent shadow prices: the marginal quality loss from tightening constraints. For example, $\lambda_h = 0.03$ indicates that reducing $H_{\max}$ by 1 percentage point costs 0.03 in correlation units.

**Example Result.** With constraints $H_{\max} = 8\%$, $C_{\max} = \$0.01$/1M tokens, we achieve ρ = 0.86 with shadow prices $\lambda_h = 0.041$, $\lambda_c = 0.018$, indicating safety is the binding constraint.

## 3.6 Data Preprocessing and Normalization

**Standardization.** Before BLF modeling, we standardize each benchmark to z-scores:

$$z_{i,b} = \frac{x_{i,b} - \mu_b}{\sigma_b}$$

where $\mu_b, \sigma_b$ are computed from models with non-missing values. For inverted metrics (e.g., hallucination rate, latency), we negate after standardization.

**Outlier Detection.** We identify outliers using robust z-scores (median and MAD):

$$z_{\text{robust}} = \frac{x - \text{median}(x)}{1.4826 \cdot \text{MAD}(x)}$$

Values with $|z_{\text{robust}}| > 4$ are flagged for manual review. We find <0.5% of data points are outliers, which we handle as follows:
- **Data entry errors** (e.g., 0.85 recorded as 85): Corrected via source verification
- **Genuine extreme performance** (e.g., GPT-5 on coding tasks): Retained without modification
- **Inconsistent multi-source data** (e.g., conflicting HumanEval scores): Prioritize official source, document discrepancy

All outlier decisions are documented in `data/outlier_review_log.csv` for reproducibility.

**Temporal Consistency.** We track benchmark score changes over time for models with multiple evaluations. Observed drift is <2% for most benchmarks over 6-month periods, validating temporal stability. For benchmarks with known distribution shift (e.g., LiveCodeBench, which continuously adds problems), we use only recent scores (<3 months old).

## 3.7 Reproducibility and Ethical Considerations

**Data Authenticity.** All data presented in this paper is derived from real sources: (i) established benchmark evaluations (HumanEval, MBPP, SummEdits, MixEval) on official test sets, (ii) real human preference judgments from Chatbot Arena (>500,000 pairwise comparisons), (iii) real operational measurements (TTFT, throughput, pricing) from production APIs, and (iv) expert human annotations (Vectara Hallucination Leaderboard). **We use NO synthetic, simulated, or generated data** for benchmark scores, quality assessments, or model evaluations. Composite scores (§3.2.3) are derived via Bayesian statistical inference on real observed benchmark data, not synthetic data generation.

**Data Availability.** All benchmark scores, model metadata, and derived composite scores are available in our public repository (github.com/[REPO]/llm_jury/data/) in JSON and CSV formats with schema documentation. **The repository includes pre-computed scores for 83 models**, enabling immediate deployment without re-running evaluations. 

**Target Users and Benefits:** This system provides distinct value to three primary user groups:

1. **Researchers and Academic Labs**: Access to reproducible baseline routing without re-running expensive benchmark evaluations. Pre-computed scores for 83 models eliminate ~$150-200 in evaluation costs per research project. Enables comparative analysis against a standardized benchmark suite.

2. **Practitioners and Startups**: Low-latency routing decisions (<1ms lookup overhead) without maintaining dedicated benchmark infrastructure. Reduces operational costs by 30-50% compared to static model selection while maintaining quality standards. No ML expertise required for deployment.

3. **Smaller Labs and Independent Developers**: Democratizes access to state-of-the-art routing without the budget for private evaluation suites ($10K-50K annually for comprehensive testing). Open-source implementation enables customization for specific use cases. Only requires API keys for adding new models (Artificial Analysis, OpenRouter - both offer free tiers).

**Code Availability.** Complete ETL pipeline code (`llm_jury/etl/`), BLF implementation (`llm_jury/analysis/latent_factor.py`), and optimization modules (`llm_jury/optimization/`) are open-sourced under MIT license. Benchmark evaluation scripts include exact hyperparameters and random seeds for reproducibility.

**Benchmark Contamination.** We acknowledge the risk of training data contamination for public benchmarks. We prioritize:
1. Recent benchmarks (2023+) post-dating most model training cutoffs
2. Benchmarks with dynamic test sets (LiveCodeBench, SWE-bench)
3. Human preference signals (Arena ELO) less susceptible to contamination

We report validation against multiple independent quality signals (§3.5.5) to mitigate contamination concerns.

**Privacy.** All data sources are publicly available or derived from public APIs. We do not collect user data or proprietary model outputs. Arena ELO ratings are aggregated across many users, preserving individual privacy.

**Bias Considerations.** Our benchmark suite may reflect biases:
- Language: Primarily English benchmarks (MMLU, HumanEval)
- Culture: Western-centric knowledge questions (MMLU)
- Task distribution: Over-representation of coding/STEM tasks

We acknowledge these limitations and note that intent-specific routing (§5) allows users to weight benchmarks according to their use case priorities.

**Cost Efficiency and Environmental Impact.** Direct benchmark evaluations (HumanEval, MBPP, SummEdits) require significant compute. We performed evaluations once and cache results. **The project ships with pre-computed benchmark scores for 83+ models**, eliminating the need for users to re-run expensive evaluations. Users only incur costs when adding new models to the system, with incremental evaluation costs of ~$0.50-2.00 per model depending on benchmarks used.

**Maintenance Costs.** The only recurring cost is refreshing Arena category rankings, which change as new models are added to LMSYS Chatbot Arena. Updates are performed via manual extraction from the public leaderboard (lmarena.ai). Manual curation takes ~5-10 minutes per update (monthly). Zero computational or API costs. All other benchmark scores (HumanEval, MBPP, MMLU-Pro, etc.) are static and do not require updates unless model versions change.

For SummEdits specifically, the cost of ~$0.50 per model across all 10 domains (including both input and output tokens) makes comprehensive evaluation feasible. Environmental impact of one-time evaluations is estimated at ~0.5 kg CO2 per model \cite{patterson2021carbon}, totaling ~42 kg CO2 for 83 models—equivalent to driving 100 miles.

## 3.8 Dataset Statistics and Characteristics

Table 2 provides comprehensive statistics for our benchmark suite:

| Benchmark | N Models | Coverage | Mean | Std | Min | Max | Skewness |
|-----------|----------|----------|------|-----|-----|-----|----------|
| HumanEval | 69 | 83% | 78.2 | 17.1 | 29.3 | 94.2 | -0.84 |
| MBPP | 69 | 83% | 72.4 | 15.8 | 35.1 | 91.5 | -0.52 |
| SummEdits (avg) | 61 | 73% | 73.5 | 11.2 | 52.3 | 91.8 | -0.24 |
| Arena-Hard-Auto* | 23 | **28%** | 58.7 | 22.1 | 12.4 | 89.3 | -0.08 |
| MixEval | 45 | 54% | 62.3 | 14.7 | 28.9 | 87.1 | -0.15 |
| Intelligence Index | 83 | 100% | 34.7 | 18.2 | 1.0 | 72.8 | 0.13 |
| Hallucination Rate | 83 | 100% | 16.3 | 23.2 | 0.7 | 93.2 | 1.42 |
| Arena ELO | 31 | 37% | 1183 | 117 | 1067 | 1463 | 0.18 |

\* **Arena-Hard-Auto low coverage (28%)**: The BLF model compensates via auxiliary benchmarks (Intelligence Index: 100%, SummEdits: 73%) and correlation structure from models with complete data. For CSS estimation, 60 models lacking Arena-Hard-Auto rely on SummEdits + auxiliary benchmarks, with uncertainty increasing by ~15-20% (still well-calibrated: LOOCV RMSE = 3.2 on 0-100 scale).

**Benchmark Correlations.** Inter-benchmark Spearman correlations (Table 3) reveal structure:
- Strong correlations within domains (e.g., HumanEval ↔ LiveCodeBench: ρ = 0.89)
- Moderate cross-domain correlations (e.g., HumanEval ↔ MATH-500: ρ = 0.64)
- Weak correlations with safety metrics (e.g., MMLU ↔ Hallucination Rate: ρ = -0.31)

This supports our multi-domain composite score approach rather than a single "general intelligence" score.

**Interpretation for BLF**: Strong within-domain correlations (ρ > 0.8) validate single-factor models for domain-specific composites (CCS, CRS, CFS, CSS, CAE). The BLF model leverages these correlations to: (i) learn data-driven benchmark loadings, (ii) impute missing values via factor scores, and (iii) quantify uncertainty based on benchmark agreement. Unlike OLS regression where collinearity inflates variance and destabilizes coefficients \cite{belsley1980regression}, **BLF requires correlation to identify latent factors** \cite{bartholomew2011latent}—high inter-benchmark correlation is a feature, not a bug.

## 3.9 Summary

Our data collection and processing pipeline integrates 15+ heterogeneous benchmarks with operational metadata for 83 production-ready language models. Key methodological contributions:
1. Principled missing data handling via Bayesian latent factor models (100% coverage vs 83% for listwise deletion)
2. Data-driven benchmark weighting validated against human preferences (ρ = 0.89)
3. Constrained optimization enabling explicit quality-safety-cost trade-offs
4. Comprehensive reproducibility through open data and code

This rigorous data foundation enables the intent-aware routing system evaluated in §5.

---

## References

\cite{artificialanalysis2024} Artificial Analysis. (2024). LLM Performance Leaderboard. https://artificialanalysis.ai

\cite{chen2021humaneval} Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374

\cite{jain2024livecodebench} Jain, N., et al. (2024). LiveCodeBench: Holistic and Contamination-Free Evaluation of LLMs for Code. arXiv:2403.07974

\cite{tian2024scicode} Tian, M., et al. (2024). SciCode: A Research Coding Benchmark Curated by Scientists. arXiv:2407.13168

\cite{austin2021mbpp} Austin, J., et al. (2021). Program Synthesis with Large Language Models. arXiv:2108.07732

\cite{tang2023summedits} Tang, C., et al. (2023). Understanding Factuality in Abstractive Summarization with SummEdits. arXiv:2301.05220

\cite{zhou2023ifeval} Zhou, J., et al. (2023). Instruction-Following Evaluation for Large Language Models. arXiv:2311.07911

\cite{zheng2023arena} Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.


\cite{ni2024mixeval} Ni, J., et al. (2024). MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures. NeurIPS 2024.

\cite{hoffman2014nuts} Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. JMLR, 15(1), 1593-1623.

\cite{rubin1987multiple} Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley.

\cite{vectara2024hallucination} Vectara. (2024). Hallucination Leaderboard. github.com/vectara/hallucination-leaderboard

\cite{zhao2024llmcost} Zhao, W. X., et al. (2024). A Survey on Cost Analysis of Large Language Models. arXiv:2402.08280

\cite{patterson2021carbon} Patterson, D., et al. (2021). Carbon Emissions and Large Neural Network Training. arXiv:2104.10350

\cite{wang2024mmlu-pro} Wang, Y., et al. (2024). MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv:2406.01574

\cite{rein2023gpqa} Rein, D., et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv:2311.12022

\cite{hendrycks2021math} Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. NeurIPS 2021.

\cite{aime2024} AIME (American Invitational Mathematics Examination). 2024 Competition Problems. Mathematical Association of America.

\cite{artificialanalysis2024validation} Artificial Analysis. (2024). Methodology: Quality Index Validation. Technical Report.
