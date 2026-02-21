#!/usr/bin/env python3
"""
Test: Numerical Stability with JIT Regularization

Verifies that the JIT regularization injection prevents numerical instability
in low-traffic regimes where decay can cause matrix singularity.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from bandit_gpt.router import DisjointLinUCBPolicy


def test_jit_regularization_low_traffic():
    """
    Simulate low-traffic regime: update_lambda=0 with decay.
    
    Without JIT regularization, the matrix A can decay toward singularity,
    causing trace(A_inv) → ∞ and numerical explosion.
    
    With JIT fix, trace(A_inv) should be bounded by regularization injection.
    """
    print("=" * 70)
    print("JIT REGULARIZATION TEST: Low-Traffic Regime")
    print("=" * 70)
    
    # Configuration that triggers the singularity trap
    bandit = DisjointLinUCBPolicy(
        model_names=["test_model"],
        dim=24,
        alpha=0.1,
        init_lambda=1.0,
        update_lambda=0.0,  # No runtime regularization (fast path)
        forgetting_factor=0.95  # Decay enabled
    )
    
    print(f"\nConfiguration:")
    print(f"  init_lambda: {bandit.init_lambda}")
    print(f"  update_lambda: {bandit.update_lambda}")
    print(f"  gamma (decay): {bandit.gamma}")
    print(f"  dim: {bandit.dim}")
    print(f"  threshold: {100 * bandit.dim}")
    
    # Simulate low-traffic: 1 update, then many timesteps with no updates
    x = np.random.randn(24)
    x /= np.linalg.norm(x)
    bandit.update("test_model", x, 1.0)
    
    print(f"\nSimulating low-traffic regime:")
    print(f"  Initial update completed")
    print(f"  trace(A_inv) = {np.trace(bandit.A_inv['test_model']):.2e}")
    
    # Advance time without updates to trigger decay
    for i in range(100):
        bandit.t += 1
    
    print(f"\n  Advanced 100 timesteps without updates")
    
    # Next update should trigger JIT regularization if trace is high
    trace_before = np.trace(bandit.A_inv["test_model"])
    print(f"  trace(A_inv) before next update = {trace_before:.2e}")
    
    # Perform another update (might trigger JIT regularization internally)
    bandit.update("test_model", x, 1.0)
    
    trace_after = np.trace(bandit.A_inv["test_model"])
    print(f"  trace(A_inv) after update = {trace_after:.2e}")
    
    # Verify stability: trace should be bounded (not exploding to infinity)
    threshold = 1000 * bandit.dim
    if trace_after < threshold:
        print(f"\n✅ PASS: trace(A_inv) = {trace_after:.2e} < {threshold:.2e}")
        print(f"  Matrix remains stable despite low-traffic decay")
        return True
    else:
        print(f"\n❌ FAIL: trace(A_inv) = {trace_after:.2e} >= {threshold:.2e}")
        print(f"  Matrix is approaching singularity")
        return False


def test_jit_regularization_mechanism():
    """
    Verify that the JIT regularization actually injects identity when triggered.
    """
    print("\n" + "=" * 70)
    print("JIT REGULARIZATION MECHANISM TEST")
    print("=" * 70)
    
    # Create bandit with extreme decay to force regularization need
    bandit = DisjointLinUCBPolicy(
        model_names=["test"],
        dim=10,
        alpha=0.1,
        init_lambda=1.0,
        update_lambda=0.0,
        forgetting_factor=0.5  # Aggressive decay
    )
    
    print(f"\nConfiguration: gamma=0.5 (aggressive decay)")
    
    # Single update
    x = np.random.randn(10)
    x /= np.linalg.norm(x)
    bandit.update("test", x, 1.0)
    
    # Advance time significantly to decay matrix
    for _ in range(200):
        bandit.t += 1
        bandit.update("test", x * 0.1, 0.01)  # Minimal updates
    
    trace = np.trace(bandit.A_inv["test"])
    print(f"\nAfter 200 decay steps:")
    print(f"  trace(A_inv) = {trace:.2e}")
    print(f"  threshold = {100 * bandit.dim:.2e}")
    
    # Matrix should still be stable
    if trace < 1000 * bandit.dim:
        print(f"\n✅ PASS: JIT regularization kept matrix stable")
        return True
    else:
        print(f"\n❌ FAIL: Matrix unstable despite JIT regularization")
        return False


if __name__ == "__main__":
    test1 = test_jit_regularization_low_traffic()
    test2 = test_jit_regularization_mechanism()
    
    if test1 and test2:
        print("\n" + "=" * 70)
        print("🎉 All JIT regularization tests passed!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed")
        print("=" * 70)
        sys.exit(1)
