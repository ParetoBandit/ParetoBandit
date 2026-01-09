#!/usr/bin/env python3
"""
Create PCA with 23 components from dev split prompts
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

# Load data using experiment.py
print("Loading data...")
registry = load_model_registry()
train_rewards = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
test_rewards = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
full_corpus = {**train_rewards, **test_rewards}

# Get dev splits
splits_path = Path("experiments/01_effectiveness/results/splits.json")
burn_in_helper = ExperimentBurnIn(
    registry=registry,
    oracle_rewards=full_corpus,
    splits_path=splits_path,
    encoder=None
)

dev_prompts, _ = burn_in_helper.get_splits()
print(f"Loaded {len(dev_prompts)} dev prompts from splits")

# Encode prompts
print("Encoding prompts...")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(dev_prompts, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")

# Fit PCA
print("Fitting PCA with 23 components...")
pca = PCA(n_components=23, random_state=42)
pca.fit(embeddings)

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4%}")
print(f"Components shape: {pca.components_.shape}")

# Save
output_path = Path("artifacts/pca_23.joblib")
joblib.dump(pca, output_path)
print(f"✅ Saved to {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
