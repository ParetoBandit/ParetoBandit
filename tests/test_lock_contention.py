"""
Unit tests for Snapshot-Swap lock contention fix.

Tests that the refactored update() method:
1. Produces mathematically identical results
2. Maintains thread safety
3. Allows concurrent routing during updates
"""
import pytest
import numpy as np
import threading
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.router import DisjointLinUCBPolicy


class TestSnapshotSwapCorrectness:
    """Test mathematical correctness of Snapshot-Swap implementation."""
    
    def test_matrix_inversion_accuracy(self):
        """Verify A @ A_inv ≈ I after updates."""
        policy = DisjointLinUCBPolicy(
            ["model_a", "model_b"], 
            dim=20, 
            alpha=0.1, 
            update_lambda=0.0
        )
        
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(20)
            policy.update("model_a", x, reward=np.random.random())
        
        # Check A @ A_inv = I
        A = policy.A["model_a"]
        A_inv = policy.A_inv["model_a"]
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(20))
        
        assert error < 1e-10, f"Matrix inversion error too large: {error}"
    
    def test_stale_update_with_decay(self):
        """Test that stale updates correctly apply decay to local copies."""
        policy = DisjointLinUCBPolicy(
            ["model_a", "model_b"],
            dim=10,
            alpha=0.1,
            update_lambda=0.0
        )
        
        np.random.seed(123)
        
        # Update model_a at t=0
        x1 = np.random.randn(10)
        policy.update("model_a", x1, reward=1.0)
        assert policy.last_update["model_a"] == 0
        
        # Update model_b at t=1 (makes model_a stale)
        x2 = np.random.randn(10)
        policy.update("model_b", x2, reward=0.5)
        assert policy.last_update["model_a"] == 0  # Still at t=0
        
        # Update model_a again at t=2 (should apply decay for dt=2)
        x3 = np.random.randn(10)
        policy.update("model_a", x3, reward=0.8)
        assert policy.last_update["model_a"] == 2
        
        # Verify matrix is still valid
        A = policy.A["model_a"]
        A_inv = policy.A_inv["model_a"]
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(10))
        
        assert error < 1e-10, f"Stale update produced invalid matrix: {error}"
    
    def test_weighted_update(self):
        """Test that importance weighting works correctly."""
        policy = DisjointLinUCBPolicy(["model_a"], dim=5, alpha=0.1)
        
        np.random.seed(456)
        x = np.random.randn(5)
        
        # Store initial state
        A_before = policy.A["model_a"].copy()
        b_before = policy.b["model_a"].copy()
        
        # Apply weighted update
        weight = 0.5
        reward = 1.0
        policy.update("model_a", x, reward=reward, weight=weight)
        
        # Verify weighted contribution
        A_after = policy.A["model_a"]
        b_after = policy.b["model_a"]
        
        expected_A = A_before + weight * np.outer(x, x)
        expected_b = b_before + weight * reward * x
        
        assert np.allclose(A_after, expected_A), "A matrix update incorrect"
        assert np.allclose(b_after, expected_b), "b vector update incorrect"


class TestThreadSafety:
    """Test thread safety of Snapshot-Swap implementation."""
    
    def test_concurrent_updates_no_exceptions(self):
        """Verify no race conditions during concurrent updates."""
        policy = DisjointLinUCBPolicy(
            ["model_a", "model_b"],
            dim=10,
            alpha=0.1,
            update_lambda=0.0
        )
        
        errors = []
        
        def update_worker(model_id, iterations):
            np.random.seed(threading.current_thread().ident % 10000)
            try:
                for _ in range(iterations):
                    x = np.random.randn(10)
                    policy.update(model_id, x, reward=np.random.random())
            except Exception as e:
                errors.append(e)
        
        threads = []
        for model_id in ["model_a", "model_b"]:
            t = threading.Thread(target=update_worker, args=(model_id, 100))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety violated: {errors}"
        
        # Verify final matrices are valid
        for model in ["model_a", "model_b"]:
            A = policy.A[model]
            A_inv = policy.A_inv[model]
            identity_check = A @ A_inv
            error = np.linalg.norm(identity_check - np.eye(10))
            assert error < 1e-8, f"Final matrix invalid for {model}: {error}"
    
    def test_concurrent_select_during_update(self):
        """Test that select_arm can proceed during heavy updates."""
        policy = DisjointLinUCBPolicy(
            ["model_a", "model_b"],
            dim=50,  # Larger dimension for slower updates
            alpha=0.1,
            update_lambda=0.0
        )
        
        # Pre-populate with some data
        np.random.seed(789)
        for _ in range(10):
            x = np.random.randn(50)
            policy.update("model_a", x, reward=np.random.random())
            policy.update("model_b", x, reward=np.random.random())
        
        errors = []
        select_times = []
        
        def heavy_update_worker():
            """Simulate heavy updates (like stale model with full inversion)."""
            np.random.seed(42)
            try:
                for _ in range(20):
                    x = np.random.randn(50)
                    # Make model_a stale by updating model_b first
                    policy.update("model_b", x, reward=0.1)
                    # This update of model_a will trigger decay
                    policy.update("model_a", x, reward=0.9)
            except Exception as e:
                errors.append(("update", e))
        
        def select_worker():
            """Try to select arms while updates are happening."""
            np.random.seed(123)
            try:
                for _ in range(100):
                    x = np.random.randn(50)
                    start = time.perf_counter()
                    policy.select_arm(x)
                    elapsed = time.perf_counter() - start
                    select_times.append(elapsed)
            except Exception as e:
                errors.append(("select", e))
        
        # Start concurrent workers
        update_thread = threading.Thread(target=heavy_update_worker)
        select_thread = threading.Thread(target=select_worker)
        
        update_thread.start()
        select_thread.start()
        
        update_thread.join()
        select_thread.join()
        
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        
        # Verify select operations completed (not blocked)
        assert len(select_times) == 100, "Select operations were blocked"
        
        # Most select operations should be fast (not waiting for 50ms inversions)
        median_select_time = np.median(select_times)
        assert median_select_time < 0.01, (
            f"Select operations too slow (median={median_select_time*1000:.1f}ms), "
            f"likely blocked by lock contention"
        )
    
    def test_no_data_corruption(self):
        """Verify concurrent updates don't corrupt matrix state."""
        policy = DisjointLinUCBPolicy(
            ["model_a"],
            dim=10,
            alpha=0.1,
            update_lambda=0.0
        )
        
        def update_worker(iterations, seed_offset):
            np.random.seed(42 + seed_offset)
            for _ in range(iterations):
                x = np.random.randn(10)
                policy.update("model_a", x, reward=np.random.random())
        
        # Run multiple threads updating the same model
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_worker, args=(50, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify matrix integrity
        A = policy.A["model_a"]
        A_inv = policy.A_inv["model_a"]
        
        # Check symmetry (A should be symmetric for our updates)
        assert np.allclose(A, A.T), "A matrix is not symmetric"
        
        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0), f"A matrix not positive definite: {eigenvalues}"
        
        # Check A @ A_inv = I
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(10))
        assert error < 1e-8, f"Matrix corrupted: {error}"


class TestRegressionPrevention:
    """Ensure Snapshot-Swap doesn't break existing functionality."""
    
    def test_select_arm_still_works(self):
        """Basic smoke test for select_arm."""
        policy = DisjointLinUCBPolicy(["model_a", "model_b"], dim=10, alpha=0.1)
        
        np.random.seed(42)
        for _ in range(5):
            x = np.random.randn(10)
            policy.update("model_a", x, reward=1.0)
            policy.update("model_b", x, reward=0.5)
        
        x_test = np.random.randn(10)
        selected, ucb = policy.select_arm(x_test)
        
        assert selected in ["model_a", "model_b"]
        assert isinstance(ucb, float)
        assert not np.isnan(ucb)
    
    def test_add_arm_dynamic(self):
        """Test that adding arms still works."""
        policy = DisjointLinUCBPolicy(["model_a"], dim=5, alpha=0.1)
        
        # Add new arm
        policy.add_arm("model_b")
        
        assert "model_b" in policy.models
        assert "model_b" in policy.A
        assert "model_b" in policy.A_inv
        
        # Update new arm
        x = np.random.randn(5)
        policy.update("model_b", x, reward=0.8)
        
        # Verify it works
        A_inv = policy.A_inv["model_b"]
        A = policy.A["model_b"]
        identity_check = A @ A_inv
        error = np.linalg.norm(identity_check - np.eye(5))
        
        assert error < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
