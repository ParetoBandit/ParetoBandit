"""
Test to verify the Confident Transfer Trap fix

This test validates that bootstrapping transfers only θ (preferences) from neighbors,
not A (confidence), preventing the "fossilization" of new models.

Bug Scenario (Pre-Fix):
1. "GPT-4" has 1M samples → A has large eigenvalues (high confidence)
2. Register "GPT-4-Turbo" (similar name)
3. Bug: Bootstrap transfers 80% of A → new model thinks it has 800k samples
4. Result: Tiny confidence intervals → no exploration → fossilized behavior

Expected Behavior (Post-Fix):
1. "GPT-4" has 1M samples → A has large eigenvalues
2. Register "GPT-4-Turbo"
3. Extract θ_neighbor = A_inv @ b_neighbor
4. Initialize with: A_new = λI (fresh), b_new = λ * θ_neighbor (preferences)
5. Result: Same preferences, maximum uncertainty → healthy exploration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bandit_gpt.router import BanditRouter

def test_theta_only_transfer():
    """Test that bootstrapping transfers θ (preferences) but resets A (confidence)."""
    
    # Create a minimal registry with one mature model
    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "openrouter_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        }
    }
    
    # Initialize router
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0
    )
    
    print("✅ Router initialized with 1 model (GPT-4)")
    
    # Simulate mature model with many samples
    # Add artificial samples to GPT-4 to make it "mature"
    model_id = "gpt-4-base"
    n_samples = 10000
    
    print(f"\n📊 Simulating {n_samples} samples for {model_id}...")
    
    # Artificially age the A matrix AND b vector (simulate many samples)
    # This creates high confidence (large eigenvalues) and learned preferences (non-zero θ)
    for i in range(100):
        context = router._get_context_vector(f"test prompt {i}")
        # Simulate update with positive reward: A += x @ x.T, b += reward * x
        reward = 0.8 + 0.1 * np.random.rand()  # Simulated reward ~0.8-0.9
        router.bandit.A[model_id] += np.outer(context, context)
        router.bandit.b[model_id] += reward * context
    
    router.bandit.refresh_inverse_cache()
    
    # Check A matrix eigenvalues (should be large after aging)
    eigenvalues_before = np.linalg.eigvalsh(router.bandit.A[model_id])
    max_eigenvalue_before = eigenvalues_before.max()
    
    # Check learned θ (should be non-zero)
    theta_before = router.bandit.A_inv[model_id] @ router.bandit.b[model_id]
    theta_norm_before = np.linalg.norm(theta_before)
    
    print(f"   Mature model A matrix max eigenvalue: {max_eigenvalue_before:.2f}")
    print(f"   (Should be >> init_lambda={router.bandit.init_lambda})")
    print(f"   Mature model ||θ||: {theta_norm_before:.4f}")
    print(f"   (Should be > 0, indicating learned preferences)")
    
    # Now register a similar model (should bootstrap from GPT-4)
    new_model = {
        "display_name": "GPT-4 Turbo - Enhanced Reasoning Model",  # Very similar
        "openrouter_id": "gpt-4-turbo",
        "cost_per_1m_tokens": 8000.0,
        "median_latency_s": 1.5,
        "initial_quality": 0.87
    }
    
    print(f"\n📝 Registering similar model: {new_model['openrouter_id']}")
    print(f"   Should bootstrap from: {model_id}")
    
    # Test the bootstrapping directly
    A_new, b_new = router.admix_theta_from_neighbors(
        model_id="gpt-4-turbo",
        registry={**router.registry, "gpt-4-turbo": new_model},
        bandit=router.bandit,
        encoder=router.encoder
    )
    
    # Check A matrix eigenvalues for the NEW model
    eigenvalues_new = np.linalg.eigvalsh(A_new)
    max_eigenvalue_new = eigenvalues_new.max()
    
    print(f"\n🔍 New Model Initialization:")
    print(f"   A_new max eigenvalue: {max_eigenvalue_new:.2f}")
    print(f"   Expected: ~init_lambda={router.bandit.init_lambda} (fresh identity)")
    
    # Check if b_new is non-zero (transferred preferences)
    b_new_norm = np.linalg.norm(b_new)
    print(f"   ||b_new||: {b_new_norm:.4f}")
    print(f"   Expected: > 0 (inherited preferences)")
    
    # Verify the fix
    is_A_fresh = abs(max_eigenvalue_new - router.bandit.init_lambda) < 0.1
    has_preferences = b_new_norm > 0.01
    
    print(f"\n✅ Verification:")
    print(f"   A reset to identity: {is_A_fresh}")
    print(f"   θ transferred: {has_preferences}")
    
    if is_A_fresh and has_preferences:
        print(f"\n✅ TEST PASSED: Confident transfer trap is fixed!")
        print(f"   - A was reset to identity (maximum uncertainty)")
        print(f"   - θ ws transferred from neighbor (preferences)")
        print(f"   - New model will explore healthily instead of being fossilized")
        return True
    else:
        print(f"\n❌ TEST FAILED:")
        if not is_A_fresh:
            print(f"   - A was NOT reset (still {max_eigenvalue_new:.2f} >> {router.bandit.init_lambda})")
        if not has_preferences:
            print(f"   - θ was NOT transferred (b_new is near-zero)")
        return False

if __name__ == "__main__":
    success = test_theta_only_transfer()
    sys.exit(0 if success else 1)
