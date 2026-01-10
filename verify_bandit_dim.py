
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

try:
    from bandit_gpt.router import BanditRouter
    print("Successfully imported BanditRouter")
except ImportError as e:
    print(f"Failed to import BanditRouter: {e}")
    sys.exit(1)

def verify_router():
    print("Creating BanditRouter...")
    try:
        # Create minimal registry
        registry = {
            "test/model": {
                "input_cost_per_m": 1.0, 
                "time_to_first_token_seconds": 0.5,
                "hle": 0.5
            }
        }
        
        router = BanditRouter.create(model_registry=registry, priors="none")
        
        print(f"PCA Loaded: {router.pca is not None}")
        if router.pca:
            print(f"PCA Components: {router.pca.n_components}")
            
        print(f"Bandit Dimension: {router.bandit.dim}")
        
        # Check if 384 or 32
        if router.bandit.dim > 100:
            print("❌ USING FULL 384 EMBEDDINGS (Total dim > 100)")
        else:
            print("✅ USING PCA FEATURES (Total dim < 100)")
            
    except Exception as e:
        print(f"Error creating router: {e}")
        import traceback
        traceback.print_exc()

    # Verify Joblib Artifact
    print("\nVerifying data/priors_warmup.joblib...")
    try:
        import joblib
        priors = joblib.load("data/priors_warmup.joblib")
        
        # Handle DisjointLinUCB structure (A is a dict of matrices)
        if isinstance(priors["A"], dict):
             print("Detected DisjointLinUCB structure (A is dict). Checking first arm...")
             first_A = next(iter(priors["A"].values()))
             first_b = next(iter(priors["b"].values()))
             A_shape = first_A.shape
             b_shape = first_b.shape
        else:
             # Assume shared structure? (Unlikely for this codebase)
             A_shape = priors["A"].shape
             b_shape = priors["b"].shape
             
        print(f"Priors A Shape: {A_shape}")
        print(f"Priors b Shape: {b_shape}")
        
        if A_shape == (11, 11) and b_shape == (11,):
            print("✅ PRIORS DIMENSION MATCH (11)")
        else:
            print(f"❌ PRIORS DIMENSION MISMATCH (Expected 11, Got {A_shape[0]})")
            
    except Exception as e:
        print(f"Failed to load priors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_router()
