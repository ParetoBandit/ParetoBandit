# The Bandit Router: Democratizing Intelligent LLM Orchestration

The **Bandit Router** (BanditGPT) represents a paradigm shift in how Large Language Models (LLMs) are selected and deployed. While existing solutions often require deep technical expertise, extensive training data, or rigid cascading rules, BanditGPT provides a plug-and-play, self-learning engine that balances quality, cost, and latency out-of-the-box.

## 1. Core Philosophy: No Up-Front Calibration Required

The primary goal of BanditGPT is to **democratize AI access** by removing the "Blocking Manual Labor" required to start using intelligent routing. While existing solutions like RouteLLM and FrugalGPT require significant up-front investment, BanditGPT is designed for **Non-Blocking Deployment**.

*   **No Up-Front Calibration**: Unlike RouteLLM, which requires a "Calibration Dataset" (500–2,000 labeled examples) before it can route a single query, BanditGPT can be deployed immediately with zero user-provided data.
*   **Metadata-Driven Start**: On "Day 0," the router uses public metadata (e.g., "Max Cost $0.50", "Claimed Quality 90%") and "Expert Intuition" from benchmarks like **Humanity's Last Exam (HLE)** to filter models.
*   **Autonomous Evolution**: The system learns in the background on live traffic. The "learning" happens as users interact with the tool, rather than blocking the tool's deployment.

## 2. Operational Superiority: The Bandit Advantage

BanditGPT is operationally superior to static classifiers because it converts a high-maintenance pipeline into an autonomous, self-correcting system.

### The "Day 0" Advantage: Zero-Calibration Deployment
The biggest friction point for users is getting started.
- **RouteLLM / FrugalGPT (The Blocking Requirement)**: You must collect a dataset, run it through all models, grade them, and train a classifier. This is a massive barrier: *"I can't use this tool until I manually label 1,000 prompts."*
- **BanditGPT (The Non-Blocking Start)**: You deploy immediately. The system works from Minute 1, refining its intelligence via online learning without requiring a pre-labeled dataset.

### The "Day 100" Advantage: Zero-Maintenance Evolution
The second biggest pain point is keeping the system alive as the market changes.
- **RouteLLM (Frozen Intelligence)**: The router is a static classifier. If a new model (e.g., DeepSeek-V3) is released, the router doesn't know it exists. To add it, you must re-run your calibration benchmarks—an $O(N)$ maintenance cost.
- **BanditGPT (Autonomous Evolution)**: You register the new model string and its price in 30 seconds. The bandit automatically allocates a small "Exploration" budget to the new model. If it performs well, it naturally climbs the leaderboard.

### The "Feedback" Friction: Cheaper than Labels
While the bandit requires feedback to learn, this feedback is sourced from **external signals** rather than a mandatory internal classifier. This makes the system significantly more user-friendly:
- **Programmatic Feedback**: For code tasks, the reward can be "Did the code compile?" (Automated, free).
- **Implicit Feedback**: For chat, the reward can be "Did the user copy the answer?" (No manual labeling).
- **Optional Oracle Checks**: Users *can* choose to use a "Teacher" model (e.g., GPT-4o) to grade a small percentage of traffic, but this is an optional optimization, not a Day 0 requirement.

## 3. Comparative Analysis: Feature Breakdown

| Feature | **BanditGPT** | RouteLLM / FrugalGPT |
| :--- | :--- | :--- |
| **Prerequisite** | **None**: Starts with public metadata (Price, Context) | **Heavy**: Needs 500+ labeled examples before launch |
| **New Models** | **Instant**: Register API string; learns via exploration | **Slow**: Must re-benchmark & re-train ($O(N)$ cost) |
| **Latency** | **Constant**: Single-shot routing decisions ($O(1)$) | **Variable**: FrugalGPT waits for models to fail ($O(K)$) |
| **Maintenance** | **Autonomous**: System self-corrects based on feedback | **Manual**: User must update chains/rules manually |
| **Learning** | **Online (Real-time)** | Offline (Trained/Static) |
| **Goal** | **Democratization** | Cost Reduction |

### Key Differentiators:
*   **vs. RouteLLM**: RouteLLM is powerful but often limited to routing between two models (e.g., GPT-4 and Mixtral). BanditGPT naturally scales to **80+ models**, treating each as an "arm" in a multi-armed bandit problem.
*   **vs. FrugalLLM**: FrugalLLM uses cascades (try cheap, then expensive). BanditGPT is more flexible; it might pick a mid-tier model that is "just right" for the specific prompt's embedding, rather than always starting at the bottom.
*   **vs. LiteLLM**: LiteLLM is the essential "plumbing" of the LLM world. BanditGPT adds the "brain" that decides *which* pipe to use.
*   **vs. Semantic Router**: Aurelio AI's Semantic Router is excellent for intent (e.g., "Is this a sales query?"). BanditGPT uses similar embedding technology but applies it to **Quality Prediction**, asking "Which model will give the best answer for this specific vector?"

## 3. User-Adjustable Parameters

BanditGPT empowers users with intuitive controls to align the router with their specific business or creative goals:

*   **Optimization Profiles**:
    *   `quality_first`: Prioritizes the best possible answer, regardless of cost.
    *   `cost_saver`: Aggressively targets the cheapest models that meet a minimum quality floor.
    *   `low_latency`: Minimizes Time to First Token (TTFT) for interactive applications.
    *   `balanced`: The default "sweet spot" for most users.
*   **Hard Constraints**: Users can set `max_cost` (per 1k tokens) or `max_latency` to ensure the router never exceeds their budget or performance requirements.
*   **Forgetting Factor** ($f$): Controls the rate at which the bandit "forgets" past observations (including the prior). A value of 1.0 retains all history, while lower values (e.g., 0.95) allow for faster adaptation to changing user needs.
    -   *Impact*: Lowering $f$ significantly accelerates the discovery of niche specialists (see [Figure 4](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/figure_4/figure_4_description.md) for a detailed speed comparison).
### Table 1: Latency Overhead
We measure the computational cost of the router to ensure it is suitable for production use.

| Component | Mean Latency (ms) | P95 Latency (ms) |
| :--- | :--- | :--- |
| **Total Overhead** | **21.07 ms** | **46.89 ms** |

- **Insight**: The overhead is <4% of a typical LLM request, making the router's intelligence effectively "free" in terms of user experience.
- **Details**: See [Table 1 Description](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/table_1/table_1_description.md).

| Component | Mean Latency (ms) | P95 Latency (ms) | Description |
| :--- | :--- | :--- | :--- |
| **Embedding** | 11.55 ms | 26.68 ms | Prompt vectorization via `all-MiniLM-L6-v2`. |
| **Filtering** | 0.07 ms | 0.18 ms | Constraint checking (cost, latency, quality). |
| **Scoring** | 9.44 ms | 22.27 ms | Contextual Bandit UCB calculation (80 models). |
| **Total** | **21.07 ms** | **46.89 ms** | **Total overhead per request.** |

> [!NOTE]
> For a detailed breakdown and practical implications, see [Table 1](file:///Users/annette/repostitories/llm_jury/final_release/kdd_paper/table_1/table_1_description.md).

To balance the competing objectives of **Quality**, **Cost**, and **Latency**, BanditGPT utilizes a **Pareto-Chebyshev Scalarization** approach. This ensures that the router finds a mathematically sound "compromise" that respects user-defined priorities.

### The Scalarization Function
For each model $m$, we calculate a combined utility score $U_m$ using the following formula:

$$U_m = \max_{i \in \{Q, C, L\}} \left( w_i \cdot |f_i(m) - z_i^*| \right)$$

Where:
*   $f_i(m)$ is the predicted value for objective $i$ (Quality, Cost, or Latency).
*   $z_i^*$ is the "ideal" value for that objective (e.g., maximum quality, zero cost, zero latency).
*   $w_i$ is the user-defined weight for that objective.

### Why Chebyshev?
Unlike simple weighted sums, which can fail to find solutions on non-convex Pareto fronts, Chebyshev scalarization is guaranteed to find any Pareto-optimal point. This is critical for LLM routing, where the trade-off between a "cheap but slow" model and a "fast but expensive" model may not be linear.

### Shadow Pricing for "Free" Models
To prevent models with zero cost (e.g., local models or free tier APIs) from having an infinite advantage, we implement a **Shadow Price** mechanism. This assigns a nominal, non-zero cost to free models during the optimization phase, ensuring they are selected based on their quality and latency merits rather than just their price tag.

## 4. Impact on Society

By making intelligent routing accessible, BanditGPT ensures that the benefits of the "LLM Revolution" are not restricted to those with deep pockets or PhDs in Machine Learning. It allows a diverse range of creators—from independent journalists to small-scale developers—to leverage the world's best AI models efficiently, fostering a more inclusive and creative digital society.
