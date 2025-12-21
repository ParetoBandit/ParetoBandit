import json
import numpy as np
from pathlib import Path
from final_release.bandit import BanditRouter

def test_benchmark_selection():
    print("Testing User-Specified Benchmark Selection...")
    
    # 1. Setup a mock registry with two different benchmarks
    registry = {
        "model_a": {"openrouter_id": "model_a", "hle": 0.9, "custom_score": 0.1},
        "model_b": {"openrouter_id": "model_b", "hle": 0.1, "custom_score": 0.9}
    }
    
    # Create a dummy priors_meta file
    base_dir = Path(__file__).parent.parent / "final_release"
    meta_path = base_dir / "data" / "priors_meta.npz"
    
    # Test Case A: Using default (HLE)
    print("\nCase A: Using default benchmark (HLE)")
    router_hle = BanditRouter.create(
        model_registry=registry,
        priors="benchmark",
        benchmark_key="hle"
    )
    # b = strength * score * sum_vec. 
    # So b_a / 0.9 should equal b_b / 0.1
    b_a = router_hle.bandit.b["model_a"]
    b_b = router_hle.bandit.b["model_b"]
    
    # Check proportionality
    ratio = 0.9 / 0.1
    assert np.allclose(b_a, b_b * ratio), f"Priors should be proportional to HLE scores (ratio {ratio})"
    print(f"SUCCESS: Model A b-vector is {ratio}x Model B b-vector (as expected for HLE 0.9 vs 0.1)")

    # Test Case B: Using custom benchmark
    print("\nCase B: Using custom benchmark (custom_score)")
    router_custom = BanditRouter.create(
        model_registry=registry,
        priors="benchmark",
        benchmark_key="custom_score"
    )
    b_a_custom = router_custom.bandit.b["model_a"]
    b_b_custom = router_custom.bandit.b["model_b"]
    
    # Here Model B has 0.9 and Model A has 0.1
    ratio_custom = 0.9 / 0.1
    assert np.allclose(b_b_custom, b_a_custom * ratio_custom), f"Priors should be proportional to custom scores (ratio {ratio_custom})"
    print(f"SUCCESS: Model B b-vector is {ratio_custom}x Model A b-vector (as expected for custom_score 0.9 vs 0.1)")

    print("\nSUCCESS: Benchmark selection works correctly!")

if __name__ == "__main__":
    test_benchmark_selection()
