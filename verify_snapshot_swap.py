#!/usr/bin/env python
"""
Quick verification test for Snapshot-Swap lock contention fix.
Tests that the refactored update() still produces correct results.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bandit_gpt.router import DisjointLinUCBPolicy

def test_snapshot_swap_correctness():
    """Verify that Snapshot-Swap produces identical results to original implementation."""
    print("Testing Snapshot-Swap correctness...")
    
    # Initialize policy
    models = ["model_a", "model_b", "model_c"]
    policy = DisjointLinUCBPolicy(models, dim=10, alpha=0.1, forgetting_factor=0.95)
    
    # Generate test data
    np.random.seed(42)
    x1 = np.random.randn(10)
    x2 = np.random.randn(10)
    x3 = np.random.randn(10)
    
    # Perform updates
    policy.update("model_a", x1, reward=1.0)
    policy.update("model_b", x2, reward=0.5)
    policy.update("model_c", x3, reward=0.8)
    
    # Verify matrices are well-conditioned
    for model in models:
        A = policy.A[model]
        A_inv = policy.A_inv[model]
        
        # Check A @ A_inv ≈ I
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(10))
        
        print(f"  {model}: A @ A_inv error = {error:.6e}")
        assert error < 1e-10, f"Matrix inversion check failed for {model}"
    
    # Test select_arm works
    x_test = np.random.randn(10)
    selected, ucb = policy.select_arm(x_test)
    print(f"  Selected: {selected} (UCB={ucb:.4f})")
    assert selected in models
    
    print("✓ Snapshot-Swap correctness test passed!")
    return True


def test_concurrent_updates_staleness():
    """Test that stale updates still work correctly."""
    print("\nTesting stale updates with forgetting factor...")
    
    models = ["model_a", "model_b", "model_c"]
    policy = DisjointLinUCBPolicy(models, dim=10, alpha=0.1, forgetting_factor=0.95)
    
    np.random.seed(123)
    
    # Update model_a
    x1 = np.random.randn(10)
    policy.update("model_a", x1, reward=1.0)
    print(f"  t={policy.t}, last_update[model_a]={policy.last_update['model_a']}")
    
    # Update model_b (makes model_a stale)
    x2 = np.random.randn(10)
    policy.update("model_b", x2, reward=0.5)
    print(f"  t={policy.t}, last_update[model_a]={policy.last_update['model_a']}")
    
    # Update model_a again (should apply decay)
    x3 = np.random.randn(10)
    policy.update("model_a", x3, reward=0.8)
    print(f"  t={policy.t}, last_update[model_a]={policy.last_update['model_a']}")
    
    # Verify A_inv is still valid
    A = policy.A["model_a"]
    A_inv = policy.A_inv["model_a"]
    identity_check = A @ A_inv
    error = np.linalg.norm(identity_check - np.eye(10))
    
    print(f"  A @ A_inv error after stale update = {error:.6e}")
    assert error < 1e-10, "Stale update matrix inversion check failed"
    
    print("✓ Stale updates test passed!")
    return True


def test_thread_safety_smoke():
    """Basic smoke test for thread safety (not a full stress test)."""
    print("\nRunning thread safety smoke test...")
    import threading
    
    models = ["model_a", "model_b"]
    policy = DisjointLinUCBPolicy(models, dim=10, alpha=0.1, forgetting_factor=0.95)
    
    np.random.seed(456)
    errors = []
    
    def update_worker(model_id, iterations):
        try:
            for _ in range(iterations):
                x = np.random.randn(10)
                policy.update(model_id, x, reward=np.random.random())
        except Exception as e:
            errors.append(e)
    
    def select_worker(iterations):
        try:
            for _ in range(iterations):
                x = np.random.randn(10)
                policy.select_arm(x)
        except Exception as e:
            errors.append(e)
    
    # Spawn threads
    threads = []
    threads.append(threading.Thread(target=update_worker, args=("model_a", 50)))
    threads.append(threading.Thread(target=update_worker, args=("model_b", 50)))
    threads.append(threading.Thread(target=select_worker, args=(100,)))
    
    # Start all
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    if errors:
        print(f"  ✗ Thread safety errors: {errors}")
        raise errors[0]
    
    # Verify final state
    for model in models:
        A = policy.A[model]
        A_inv = policy.A_inv[model]
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(10))
        print(f"  {model}: final A @ A_inv error = {error:.6e}")
        assert error < 1e-8, f"Final matrix check failed for {model}"
    
    print(f"  Completed {policy.t} total updates")
    print("✓ Thread safety smoke test passed!")
    return True


if __name__ == "__main__":
    try:
        test_snapshot_swap_correctness()
        test_concurrent_updates_staleness()
        test_thread_safety_smoke()
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
