
import numpy as np
import joblib
from pathlib import Path
from src.bandit_gpt.router import BanditRouter
from src.bandit_gpt.features import FeatureExtractor

def debug():
    print("Debugging Dimensions...")
    
    # Check PCA
    pca_path = Path("data/pca_32.joblib")
    if pca_path.exists():
        pca = joblib.load(pca_path)
        print(f"PCA Components: {pca.n_components}")
    else:
        print("PCA file not found!")

    # Check Features
    fe = FeatureExtractor()
    feat_vec = fe.extract_features("test prompt")
    print(f"Feature Vector Size: {len(feat_vec)}")
    
    # Check Router Vector
    # We need a configured router
    try:
        router = BanditRouter.create(priors="none")
        vec = router._get_context_vector("test prompt")
        print(f"Router Context Vector Size: {len(vec)}")
        print(f"Router Bandit Dim: {router.bandit.dim}")
    except Exception as e:
        print(f"Router check failed: {e}")

if __name__ == "__main__":
    debug()
