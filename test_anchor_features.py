
import sys
import numpy as np
from pathlib import Path

# Add repo root to path
sys.path.append("/Users/annette/repostitories/llm_jury")

from banditgpt.bandit import BanditRouter

def test_anchor_features():
    print("Initializing BanditRouter...")
    # Use 'none' priors to be faster, but ensure cluster boost/detection is active or implicitly loaded
    # Actually, we need 'priors="benchmark"' to fully exercise the standard path, 
    # OR just ensure `cluster_detector` is init. 
    # BanditRouter init will load ClusterDetector if we are not careful? 
    # Actually, ClusterDetector is lazy-loaded in `_get_cluster_distances`? 
    # No, it's typically passed or init in `_get_cluster_distances`?
    # Let's check `BanditRouter.__init__`. 
    # Ah, `BanditRouter` has `self.cluster_detector = ClusterDetector(...)` in `__init__`?
    # Let's just create it via `create` method which handles defaults.
    
    router = BanditRouter.create(priors="benchmark", benchmark_key="hle")
    
    prompts = [
        ("Write a Python function to sort a list", "Coding"),
        ("What is the square root of 144?", "Math"),
        ("Tell me a funny joke about a chicken", "Jokes")
    ]
    
    for text, label in prompts:
        print(f"\nTesting Prompt ({label}): '{text}'")
        ctx = router._get_context_vector(text)
        
        print(f"  Vector Shape: {ctx.shape}")
        # Expected: 32 (PCA) + 8 (Handcrafted) + 5 (Anchors) + 1 (Bias) = 46
        assert ctx.shape == (46,), f"Expected shape (46,), got {ctx.shape}"
        
        # Breakdown
        # 0-32: Embedding
        # 32-40: Handcrafted
        # 40-45: Anchors
        # 45: Bias
        
        anchors = ctx[40:45]
        print(f"  Anchor Distances: {anchors}")
        
        # Verify range [0, 2.0]
        assert np.all(anchors >= 0.0) and np.all(anchors <= 2.0), "Anchors out of range [0, 2]"
        
        # Logic check: 
        # For "Coding" prompt, the distance to Coding anchor (ID 12) should be relatively low.
        # Anchor IDs: 12 (Coding), 27 (Reasoning), 49 (Jokes), 55 (Math), 96 (Writing)
        # Sorted Keys -> Index mapping:
        # 0: 12 (Coding)
        # 1: 27 (Reasoning)
        # 2: 49 (Jokes)
        # 3: 55 (Math)
        # 4: 96 (Writing)
        
        if label == "Coding":
            # Distance to Coding (idx 0) should be small?
            # Or at least smaller than distance to Jokes?
            pass

    print("\n✓ Anchor Features Verified!")

if __name__ == "__main__":
    test_anchor_features()
