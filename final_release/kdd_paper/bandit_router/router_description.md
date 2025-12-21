# The Bandit Router: Democratizing Intelligent LLM Orchestration

The **Bandit Router** (BanditGPT) represents a paradigm shift in how Large Language Models (LLMs) are selected and deployed. While existing solutions often require deep technical expertise, extensive training data, or rigid cascading rules, BanditGPT provides a plug-and-play, self-learning engine that balances quality, cost, and latency out-of-the-box.

## 1. Core Philosophy: Democratization through Online Learning

The primary goal of BanditGPT is to **democratize AI access**. By lowering the barrier to entry for intelligent routing, we enable diverse creative inputs from individuals and organizations who may not have the resources to train custom classifiers or manage complex model cascades.

*   **No Training Required**: Unlike RouteLLM, which relies on offline-trained classifiers, BanditGPT uses **Online Contextual Bandits (LinUCB)**. It learns which models perform best for specific prompt types in real-time.
*   **Zero-Knowledge Onboarding**: A user can add a brand-new model to the registry (even one with zero public benchmarks) by simply adding its API ID to a JSON file. The router will naturally "explore" the model and learn its strengths without any manual intervention.
*   **Expert Intuition (Warm-Start)**: While it can start from scratch, BanditGPT can be "warm-started" using public benchmarks like **Humanity's Last Exam (HLE)**. This gives the router "expert intuition" on day one, which it then refines based on actual usage.

## 2. Comparative Analysis: Standing on the Shoulders of Giants

BanditGPT builds upon the innovations of the open-source community while addressing key usability gaps:

| Feature | **BanditGPT** | RouteLLM | FrugalLLM | LiteLLM | Semantic Router |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning Mechanism** | **Online (Real-time)** | Offline (Trained) | Heuristic (Cascade) | Static (Fallback) | Semantic (Vector) |
| **Optimization** | **Multi-Objective** | Binary (Strong/Weak) | Cost-Accuracy | Load Balancing | Intent-based |
| **User Effort** | **Minimal (JSON)** | High (Training) | Moderate (Rules) | Minimal (Config) | Moderate (Utterances) |
| **New Model Support** | **Instant** | Requires Retraining | Requires Calibration | Instant | Instant |
| **Goal** | **Democratization** | Cost Reduction | Cost Efficiency | Unified API | Speed/Tooling |

### Key Differentiators:
*   **vs. RouteLLM**: RouteLLM is powerful but often limited to routing between two models (e.g., GPT-4 and Mixtral). BanditGPT naturally scales to **80+ models**, treating each as an "arm" in a multi-armed bandit problem.
*   **vs. FrugalLLM**: FrugalLLM uses cascades (try cheap, then expensive). BanditGPT is more flexible; it might pick a mid-tier model that is "just right" for the specific prompt's embedding, rather than always starting at the bottom.
*   **vs. LiteLLM**: LiteLLM is the essential "plumbing" of the LLM world. BanditGPT uses LiteLLM-compatible IDs but adds the "brain" that decides *which* pipe to use.
*   **vs. Semantic Router**: Aurelio AI's Semantic Router is excellent for intent (e.g., "Is this a sales query?"). BanditGPT uses similar embedding technology but applies it to **Quality Prediction**, asking "Which model will give the best answer for this specific vector?"

## 3. User-Adjustable Parameters

BanditGPT empowers users with intuitive controls to align the router with their specific business or creative goals:

*   **Optimization Profiles**:
    *   `quality_first`: Prioritizes the best possible answer, regardless of cost.
    *   `cost_saver`: Aggressively targets the cheapest models that meet a minimum quality floor.
    *   `low_latency`: Minimizes Time to First Token (TTFT) for interactive applications.
    *   `balanced`: The default "sweet spot" for most users.
*   **Hard Constraints**: Users can set `max_cost` (per 1k tokens) or `max_latency` to ensure the router never exceeds their budget or performance requirements.
## 4. Mathematical Foundation: Pareto-Chebyshev Optimization

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
