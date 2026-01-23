# Resolution Gap Analysis: HLE "Blur" vs CSR "Precision"

## Overview
This experiment provides the "Smoking Gun" proof for why HLE-based priors fail in production routing. By projecting prior beliefs into the task-cluster space, we visualize the fundamental difference in **Information Density** and **Alignment Accuracy**.

## The Hypotheses
1.  **HLE "Blur" Hypothesis**: Generic benchmarks act as a low-pass filter, creating a "blurry" prior alignment that lacks the resolution to distinguish specialist strengths.
2.  **CSR "Precision" Hypothesis**: Cluster-specific success rates provide a high-resolution map that accurately captures the jagged peaks of model capability.
3.  **Misinformation Gap**: HLE doesn't just provide less information—it often provides *wrong* information, actively discouraging the use of experts on their own specialist tasks.

## Technical Methodology
1.  **Specialist Model**: Analyzing `DeepSeek-V3`, a model with high task utility but moderate generic benchmark scores.
2.  **Task Space**: Computing centroids for 100 canonical task clusters in a 32-dimensional PCA-reduced SBERT feature space.
3.  **Alignment Metric**: Calculating the cosine similarity between the prior belief vectors ($b$) and the task centroids.
4.  **Distribution Analysis**: Comparing the variance ($\sigma$) of alignment scores to quantify resolution.

## Final Results (DeepSeek-V3 Case Study)

The script generates a professional 2-panel visualization (`resolution_gap_analysis.png`) for the KDD paper:

### Panel A: Information Density (Mechanistic Proof)
*   **Metric**: Standard deviation ($\sigma$) of prior alignment across clusters.
*   **Result**: CSR shows **$2.2\times$ higher variance** than HLE ($\sigma_{CSR} \approx 0.25$ vs $\sigma_{HLE} \approx 0.11$).
*   **Significance**: This proves that CSR priors are "opinionated" and task-specific, whereas HLE is a broad, generic average.

### Panel B: Misinformation Gap (The "Expert Suppression")
*   **Focus**: Evaluating alignment on the model's top 8 specialist clusters (e.g., Backend Dev, Distributed Systems).
*   **Observation**: In several key domains where the model excels (CSR $> 0.5$), the HLE prior shows **negative** alignment.
*   **Conclusion**: HLE actively misleads the router, forcing it to "unlearn" bad priors before reaching optimal performance.

## Files
*   [`resolution_gap_analysis.py`](file:///Users/annette/repostitories/llm_jury/banditgpt/experiments/priors_and_cummulative_regret/resolution_gap_analysis.py): Analysis and plotting script.
*   [`resolution_gap_analysis.png`](file:///Users/annette/repostitories/llm_jury/banditgpt/experiments/priors_and_cummulative_regret/resolution_gap_analysis.png): The professional KDD-ready visualization.

## How to Reproduce
```bash
cd banditgpt/experiments/priors_and_cummulative_regret
python resolution_gap_analysis.py
```
