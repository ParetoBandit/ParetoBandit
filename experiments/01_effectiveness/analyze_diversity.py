import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
PROJECT_ROOT = Path("/Users/annette/repostitories/banditGPT")
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "experiments"))

from src.bandit_gpt.router import BanditRouter
from experiments.utils.data_loader import load_oracle_rewards

def analyze_diversity():
    # Load test prompts
    test_oracle = load_oracle_rewards("test_rewards_hle_models.jsonl")
    prompts = list(test_oracle.keys())
    print(f"Analyzing {len(prompts)} test prompts...")

    # Initialize router just to get the encoder and the feature builder
    router = BanditRouter.create(priors="none")
    
    # Encode prompts using the actual router logic
    print("Building context vectors (this may take a moment)...")
    embeddings = np.array([router._get_context_vector(p) for p in prompts])
    
    # 1. Variance across components
    variances = np.var(embeddings, axis=0)
    total_variance = np.sum(variances)
    explained_var_ratio = variances / total_variance
    
    print("\n--- Embedding Stats ---")
    print(f"Shape: {embeddings.shape}")
    print(f"Total Variance: {total_variance:.4f}")
    print(f"Variance in Top 3 Components: {np.sum(explained_var_ratio[:3])*100:.1f}%")
    print(f"Variance in Last 10 Components: {np.sum(explained_var_ratio[-10:])*100:.1f}%")
    
    # 2. Pairwise Similarity
    print("\nCalculating pairwise similarity...")
    # Sample subset for speed if large, but 976 is fine for a matrix
    sim_matrix = cosine_similarity(embeddings)
    # Mask diagonal
    np.fill_diagonal(sim_matrix, np.nan)
    avg_sim = np.nanmean(sim_matrix)
    std_sim = np.nanstd(sim_matrix)
    
    print(f"Average Pairwise Cosine Similarity: {avg_sim:.4f} (±{std_sim:.4f})")
    
    if avg_sim > 0.8:
        print("\nConclusion: HIGH HOMOGENEITY.")
        print("The prompts are very similar in the 32-d space. This explains why Vanilla LinUCB (1-D bias) performs so well—there isn't enough contextual 'spread' for the 32-d model to find distinct niches yet.")
    else:
        print("\nConclusion: HIGH DIVERSITY.")
        print("Prompts are well-distributed. Underperformance may be due to 'over-plasticity' or noise in specific dimensions.")

if __name__ == "__main__":
    analyze_diversity()
