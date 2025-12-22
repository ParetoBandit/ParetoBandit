# Table 3: Comparative Analysis of Operational Requirements
*BanditGPT eliminates the "Blocking Manual Labor" of offline calibration, enabling $O(1)$ adaptation to market changes.*

| Feature | **BanditGPT** (Ours) | RouteLLM | FrugalGPT |
| :--- | :--- | :--- | :--- |
| **Initialization (Day 0)** | **Zero-Shot**: Constraints-based (HLE Priors) | **Blocking**: Requires labeled dataset ($N>500$) | **Blocking**: Requires scoring function calibration |
| **Adaptation (Day 100)** | **Autonomous**: Explores new models via UCB | **Manual**: Re-benchmark & re-train classifier | **Manual**: Re-design cascade chain |
| **Feedback Signal** | **Reward Agnostic**: Implicit, Binary, or Scalar | **Supervised**: Needs Ground Truth Labels | **Proxy**: Needs Reliability/Cost Score |
| **Inference Cost** | **Constant ($O(1)$)**: Single-pass prediction | **Constant ($O(1)$)**: Single-pass prediction | **Linear ($O(K)$)**: Sequential API calls |
| **Model Space** | **Unbounded**: Scales to 80+ endpoints | **Limited**: typically binary (Strong/Weak) | **Limited**: Cascades degrade >3-5 steps |

## Analysis of Operational Efficiency
We contrast BanditGPT with state-of-the-art baselines across three dimensions of operational complexity:

### 1. Elimination of the "Cold Start" Barrier
Current approaches like RouteLLM and FrugalGPT impose a **Blocking Requirement**: deployment cannot begin until a domain-specific dataset is curated and labeled. This creates a circular dependency where users need a router to gather data, but need data to build a router. BanditGPT breaks this cycle by utilizing **Metadata Priors (HLE)** to enable non-blocking, zero-shot deployment, allowing the system to optimize immediately based on public benchmarks before adapting to live traffic.

### 2. $O(1)$ Adaptation to Market Velocity
The LLM market is highly non-stationary; new models (e.g., DeepSeek-V3) render static routers obsolete within weeks.
*   **Static Classifiers (RouteLLM)**: Incur $O(N)$ maintenance debt, requiring expensive re-labeling and re-training cycles for every registry update.
*   **BanditGPT**: Achieves $O(1)$ operational latency. A new model is simply registered as a new "arm" with an exploration budget. The Contextual Bandit mechanism automatically verifies its utility against existing baselines without human intervention.

### 3. Decoupling Reward from Supervision
Traditional routers rely on "Golden Labels" (Ground Truth), which are expensive and slow to acquire. BanditGPT leverages Contextual Banditry to learn from **partial feedback**. This allows the engine to optimize for heterogeneous, readily available signals—such as code compilation status, latency constraints, or user retention—making it a truly autonomous infrastructure component rather than a static artifact.
