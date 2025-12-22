# Table 3: Comparative Operational Friction

## Overview
A core argument of BanditGPT is that it minimizes the "Blocking Manual Labor" required to deploy and maintain intelligent routing. This table contrasts the operational requirements of BanditGPT against existing state-of-the-art solutions like **RouteLLM** and **FrugalGPT**.

| Metric | **BanditGPT** (Our Work) | RouteLLM | FrugalGPT |
| :--- | :--- | :--- | :--- |
| **Day 0 Prerequisite** | **Zero**: Uses public metadata (Price, HLE) | **High**: Needs 500-2,000 labeled examples | **High**: Needs calibration dataset for cascades |
| **Maintenance ($O(N)$)** | **Instant**: Register new model string (30s) | **Slow**: Re-benchmark & Re-train classifier | **Manual**: Update cascade rules per model |
| **Feedback Signal** | **Flexible**: Implicit (clicks), programmatic (code test) | **Rigid**: Mandatory labeled ground truth | **Implicit**: Success/Failure signals |
| **Routing Decision** | **One-Shot**: Contextual Bandit ($O(1)$) | **One-Shot**: Classifier ($O(1)$) | **Sequential**: Try cheap -> expensive ($O(K)$) |
| **Scalability** | **High**: Naturally handles 80+ "arms" | **Low**: Typically restricted to binary routing | **Medium**: Cascades become complex with >5 models |

## Key Takeaways

### 1. The "Day 0" Advantage: Non-Blocking Deployment
RouteLLM and FrugalGPT represent "Blocking" technologies. A developer cannot use them until they have already solved the very problem they are trying to automate (labeling data). BanditGPT is "Non-Blocking"—it starts with expert intuition (the HLE Prior) and refines itself over time, allowing for immediate production deployment.

### 2. The "Day 100" Advantage: Market Agility
The LLM market moves at breakneck speed. When a new model is released (e.g., DeepSeek-V3), BanditGPT can incorporate it in seconds. The system automatically allocates a small "exploration" budget to the new model to verify its performance in the wild. In contrast, static classifiers require expensive re-labeling and re-training cycles every time the registry changes.

### 3. Feedback Efficiency
By leveraging contextual banditry, we decouple the "Reward" signal from the "Training" dataset. BanditGPT can learn from any scalar reward signal, including those that are effectively free to collect (e.g., code compilation, user retention, or implicit satisfaction), making it a truly "autonomous" orchestration engine.
