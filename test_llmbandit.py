"""Quick test of LLMBandit router implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_release.baselines import LLMBanditRouter
from final_release.kdd_paper.table_3.router_performance_comparison import load_model_registry

def test_llmbandit():
    print("Testing LLM Bandit Router...")
    
    # Load registry
    registry = load_model_registry()
    print(f"✓ Loaded {len(registry)} models")
    
    # Initialize router
    router = LLMBanditRouter(registry=registry, lambda_pref=0.5)
    print("✓ Initialized LLMBanditRouter")
    
    # Test routing
    query = "What is the capital of France?"
    selected = router.route(query)
    print(f"✓ Routed query to: {selected}")
    
    # Test update
    router.update(selected, 0.95)
    print("✓ Updated Beta distributions")
    
    # Route again (should show learning)
    selected2 = router.route(query)
    print(f"✓ Second routing: {selected2}")
    
    # Test predict_proba
    prob = router.predict_proba(query)
    print(f"✓ Predict_proba: {prob:.4f}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_llmbandit()
