import json
import pytest
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List
import copy

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pareto_bandit.router import DisjointLinUCBPolicy, safe_inv

class TestSnapshotSwapCorrectness:
    """
    Verification tests for snapshot-swap correctness in bandit updates.
    """

    def test_numerical_stability_check(self):
        """Verify that _check_numerical_stability detects ill-conditioned inverse."""
        from pareto_bandit.router import RouterConfig
        dim = 20
        policy = DisjointLinUCBPolicy(
            ["model_a"],
            dim=dim,
            alpha=0.1,
            init_lambda=1.0,
        )
        config = RouterConfig()
        threshold = 1000 * dim

        # Well-conditioned: trace(A_inv) = 20 << threshold = 20000
        trace_ok = np.trace(policy.A_inv["model_a"])
        assert trace_ok < threshold

        # Blow up A_inv past the stability_threshold (default 1e6 in
        # RouterConfig) to trigger the reset.  _check_numerical_stability
        # adds init_lambda * I to A, then recomputes A_inv = safe_inv(A).
        big_scale = config.stability_threshold / dim + 1  # trace > threshold
        policy.A_inv["model_a"] = np.eye(dim) * big_scale
        assert np.trace(policy.A_inv["model_a"]) > config.stability_threshold
        policy._check_numerical_stability("model_a", config)
        trace_after = np.trace(policy.A_inv["model_a"])
        assert trace_after < config.stability_threshold, (
            f"Stability check should have reset A_inv: trace={trace_after:.1f}"
        )

    def test_safe_inv_handles_singular_matrix(self):
        """Test that safe_inv uses pseudoinverse/shrinkage for unstable matrices."""
        # Singular matrix (all zeros)
        A_singular = np.zeros((20, 20))
        A_inv = safe_inv(A_singular)
        
        # Inverse of zero matrix should be identity or zeros depending on implementation
        # usually safe_inv adds small epsilon to diagonal
        assert not np.isnan(A_inv).any()
        
        # Test near-singular
        A_near_singular = np.eye(20)
        A_near_singular[0,0] = 1e-15 
        A_inv = safe_inv(A_near_singular)
        
        identity_check = A_near_singular @ A_inv
        error = np.linalg.norm(identity_check - np.eye(20))
        
        assert error < 1e-10, f"Matrix inversion error too large: {error}"
    
    def test_stale_update_with_decay(self):
        """Test that stale updates correctly apply decay to local copies."""
        policy = DisjointLinUCBPolicy(
            ["model_a", "model_b"],
            dim=10,
            alpha=0.1,
            forgetting_factor=0.9
        )
        
        np.random.seed(123)
        
        # Update model_a at t=0
        x1 = np.random.randn(10)
        policy.update("model_a", x1, reward=1.0)
        
        # Move time forward by 10 steps
        policy.t = 10
        
        # Select arm - decay should be applied to model_a but not model_b (which has t_last=0 but b=0)
        best, ucb = policy.select_arm(x1)
        
        # Move forward more
        policy.t = 20
        best2, ucb2 = policy.select_arm(x1)
        
        # UCB should change due to variance inflation from decay
        assert ucb2 != ucb, "Decay should impact UCB scores over time"

    def test_concurrent_lock_release(self):
        """Verify that per-model locks prevent race conditions during updates."""
        policy = DisjointLinUCBPolicy(["model_a"], dim=10)
        
        import threading
        
        def slow_update():
            # Acquire lock and sleep
            with policy.model_locks["model_a"]:
                import time
                time.sleep(0.1)
                x = np.ones(10)
                policy.A["model_a"] += np.outer(x, x)
                policy.b["model_a"] += x
                policy.A_inv["model_a"] = safe_inv(policy.A["model_a"])
        
        t1 = threading.Thread(target=slow_update)
        t1.start()
        
        # Try to update from main thread - should block until t1 finishes
        start_time = 123 # Dummy
        import time
        t_start = time.time()
        policy.update("model_a", np.zeros(10), 0.0)
        t_end = time.time()
        
        # Should have waited at least 0.1s
        assert t_end - t_start >= 0.1
        t1.join()
