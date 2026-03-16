#!/usr/bin/env python3
"""
Train PCA for BanditRouter dimension reduction.

Compresses SentenceTransformer embeddings to 32 dimensions via PCA.

Note:
- The default encoder is `BAAI/bge-m3` (1024D). The shipped PCA reduces this to
  32D, and ParetoBandit appends a bias term to form 33D context vectors.
"""
import joblib
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Configuration
from pareto_bandit.config import DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH

MODEL_NAME = DEFAULT_SENTENCE_TRANSFORMER
N_COMPONENTS = 32
N_SAMPLES = 1000  # More samples for robust PCA fitting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_prompts(n=1000):
    """Generate diverse archetypal prompts to define latent space."""
    archetypes = {
        "coding": [
            "Write a Python function to", "Debug this React component", 
            "Explain threading vs multiprocessing",
            "Optimize this SQL query", "Implement Dijkstra's algorithm"
        ],
        "math": [
            "Solve for x in", "Calculate the derivative of",
            "Probability of rolling", "Proof that sqrt(2) is irrational",
            "Compute eigenvalues"
        ],
        "creative": [
            "Write a poem about", "Draft a short story",
            "Create a marketing tagline", "Write dialogue between",
            "Describe a sunset on Mars"
        ],
        "reasoning": [
            "Analyze logical fallacies in", "Pros and cons of",
            "Summarize main arguments", "Economic impact of",
            "Deduce the motive"
        ],
        "general": [
            "How do I", "What is the best way to",
            "Explain the concept of", "Compare and contrast",
            "Recommend a good"
        ]
    }
    
    import random
    prompts = []
    categories = list(archetypes.keys())
    
    for i in range(n):
        category = categories[i % len(categories)]
        base = random.choice(archetypes[category])
        # Add variation to prevent duplicates
        prompts.append(f"{base} example {random.randint(1000, 9999)}")
    
    return prompts

def main():
    logger.info(f"Loading encoder: {MODEL_NAME}")
    encoder = SentenceTransformer(MODEL_NAME)
    
    logger.info(f"Generating {N_SAMPLES} synthetic prompts...")
    prompts = generate_synthetic_prompts(N_SAMPLES)
    
    logger.info("Encoding prompts...")
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
    
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    logger.info(f"Fitting PCA to {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(embeddings)
    
    # Validation
    explained_var = np.sum(pca.explained_variance_ratio_)
    logger.info(f"Explained variance: {explained_var:.2%}")
    
    if explained_var < 0.60:
        logger.warning(f"⚠️  PCA captures only {explained_var:.1%} variance")
    else:
        logger.info(f"✓ PCA captures {explained_var:.1%} variance")
    
    # Save to package artifact location (matches FeatureService default)
    output_path = Path(DEFAULT_PCA_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving to {output_path}")
    joblib.dump(pca, output_path)
    
    logger.info("✓ Done!")
    logger.info("Context dimension at runtime: 32 (PCA) + 1 (bias) = 33")

if __name__ == "__main__":
    main()
