import numpy as np
from final_release.bandit import BanditRouter

def test_add_model():
    # 1. Initialize Router with empty registry
    print("Initializing Router...")
    router = BanditRouter.create(model_registry={}, exploration="balanced", benchmark_key="hle")
    
    # 2. Add a "Math Specialist" model (Cheap & Tagged)
    print("Adding 'new-math-model'...")
    router.add_model("new-math-model", {
        "input_cost_per_m": 0.1, # $0.0001/1k (Cheap)
        "hle": 0.05,             # Low base score
        "tags": ["math", "reasoning"],
        "description": "A specialized math model."
    })
    
    # 3. Verify Initialization
    # Check if 'b' vector has been boosted
    # Base score 0.05 * Efficiency(~9.2) * Cluster(1.5) * Prior(20) ~ 13.8
    bias_weight = router.bandit.b["new-math-model"][-1]
    print(f"Bias Weight for 'new-math-model': {bias_weight:.4f}")
    
    if bias_weight > 10.0:
        print("SUCCESS: Model initialized with boosted prior!")
    else:
        print("FAILURE: Prior boost not applied correctly.")
        
    # 4. Test Routing
    # A math prompt should select this model (since it's the only one, but also high prior)
    print("Routing 'Solve 2x+5=15'...")
    selected, _ = router.route("Solve 2x+5=15")
    print(f"Selected: {selected}")
    
    assert selected == "new-math-model"
    print("SUCCESS: Router selected the new model.")

if __name__ == "__main__":
    test_add_model()
