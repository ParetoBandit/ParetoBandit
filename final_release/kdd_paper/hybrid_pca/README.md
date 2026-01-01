# Hybrid Dimensionality Reduction Strategy

To optimize the sample efficiency of the Disjoint LinUCB policy, we employ a Hybrid Dimensionality Reduction strategy. The regret of LinUCB is bounded by $\tilde{O}(d\sqrt{T})$. Our raw feature space ($d=397$) poses a risk of slow convergence.

We apply Principal Component Analysis (PCA) solely to the dense semantic embeddings, projecting $\mathbb{R}^{384} \rightarrow \mathbb{R}^{32}$, while retaining the 13 explicit complexity features (e.g., logical density, code ratios) in their raw form. This reduces the state space to $d=45$, accelerating convergence by approximately $8\times$ while preserving the high-signal complexity indicators required for separating model capabilities.

## Contents
*   `train_pca.py`: Script to train the PCA model on offline prompts and regenerate the bandit priors for the reduced feature space.
