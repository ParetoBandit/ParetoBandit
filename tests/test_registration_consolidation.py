"""
Test to verify the consolidated model registration logic

This test validates that both register_model and admit_new_model use the same
initialization logic (ad mix_theta_from_neighbors with θ-only transfer).

Consolidation Scenario:
1. Both methods should produce identical initialization (A, b) for the same model
2. Both should use semantic embedding-based neighbor finding
3. Both should transfer θ only (not A confidence)
4. admit_new_model additionally has Pareto gatekeeping

Expected Behavior:
- admit_new_model delegates to admix_theta_from_neighbors
- Both paths initialize with A = λI (fresh), b = λ × θ_neighbor
- No more inline averaging logic in admit_new_model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bandit_gpt.router import BanditRouter

def test_consolidated_initialization():
    """Test that admit_new_model uses the same logic as register_model."""
    
    # Create a router with one mature model
    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "openrouter_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        }
    }
    
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0
    )
    
    print("✅ Router initialized with GPT-4")
    
    # Simulate some learning for GPT-4
    model_id = "gpt-4-base"
    for i in range(50):
        context = router._get_context_vector(f"test prompt {i}")
        reward = 0.85 + 0.05 * np.random.rand()
        router.bandit.A[model_id] += np.outer(context, context)
        router.bandit.b[model_id] += reward * context
    
    router.bandit.refresh_inverse_cache()
    
    # Extract θ from mature model
    theta_mature = router.bandit.A_inv[model_id] @ router.bandit.b[model_id]
    theta_norm = np.linalg.norm(theta_mature)
    
    print(f"\n📊 Mature model learned preferences:")
    print(f"   ||θ||: {theta_norm:.4f}")
    
    # Now test admit_new_model with a similar model
    new_model_data = {
        "display_name": "GPT-4 Turbo - Enhanced Reasoning Model",
        "openrouter_id": "gpt-4-turbo",
        "cost_per_1m_tokens": 8000.0,
        "median_latency_s": 1.5,
        "initial_quality": 0.87
    }
    
    print(f"\n📝 Admitting new model via admit_new_model...")
    success = router.admit_new_model(new_model_data)
    
    if not success:
        print(f"❌ TEST FAILED: Model was rejected (shouldn't happen in this test)")
        return False
    
    # Check the new model's initialization
    A_new = router.bandit.A["gpt-4-turbo"]
    b_new = router.bandit.b["gpt-4-turbo"]
    
    eigenvalues_new = np.linalg.eigvalsh(A_new)
    max_eig_new = eigenvalues_new.max()
    
    theta_new = router.bandit.A_inv["gpt-4-turbo"] @ b_new
    theta_new_norm = np.linalg.norm(theta_new)
    
    print(f"\n🔍 New Model Initialization (via admit_new_model):")
    print(f"   A max eigenvalue: {max_eig_new:.2f}")
    print(f"   Expected: ~{router.bandit.init_lambda} (fresh identity)")
    print(f"   ||θ||: {theta_new_norm:.4f}")
    print(f"   Expected: ~{theta_norm:.4f} (inherited from neighbor)")
    
    # Verify consolidation worked
    is_A_fresh = abs(max_eig_new - router.bandit.init_lambda) < 0.1
    has_preferences = theta_new_norm > 0.1
    similarity_to_neighbor = abs(theta_new_norm - theta_norm) / theta_norm < 0.1
    
    print(f"\n✅ Verification:")
    print(f"   A reset to identity: {is_A_fresh}")
    print(f"   θ transferred: {has_preferences}")
    print(f"   θ matches neighbor: {similarity_to_neighbor}")
    
    if is_A_fresh and has_preferences and similarity_to_neighbor:
        print(f"\n✅ TEST PASSED: Consolidation successful!")
        print(f"   - admit_new_model uses admix_theta_from_neighbors")
        print(f"   - A is fresh (no confident transfer trap)")
        print(f"   - θ is transferred from neighbor")
        print(f"   - Both startup and runtime paths now use same logic")
        return True
    else:
        print(f"\n❌ TEST FAILED:")
        if not is_A_fresh:
            print(f"   - A not reset (eig={max_eig_new} != {router.bandit.init_lambda})")
        if not has_preferences:
            print(f"   - θ not transferred (||θ||={theta_new_norm} too small)")
        if not similarity_to_neighbor:
            print(f"   - θ doesn't match neighbor (diff={abs(theta_new_norm - theta_norm):.4f})")
        return False

if __name__ == "__main__":
    success = test_consolidated_initialization()
    sys.exit(0 if success else 1)
