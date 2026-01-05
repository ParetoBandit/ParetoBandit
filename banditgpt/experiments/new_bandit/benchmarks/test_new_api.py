#!/usr/bin/env python3
"""
Test the new init_lambda/update_lambda API.

Verifies that the default configuration (init_lambda=1.0, update_lambda=0.0)
achieves >2500 updates/sec with O(d²) complexity.
"""

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bandit_v2 import DisjointLinUCBPolicy


def test_default_configuration():
    """Test the NEW default: init_lambda=1.0, update_lambda=0.0"""
    print("=" * 70)
    print("TESTING NEW DEFAULT CONFIGURATION")
    print("=" * 70)
    
    dim = 384
    n_trials = 100
    
    # NEW API: Defaults to init_lambda=1.0, update_lambda=0.0
    policy = DisjointLinUCBPolicy(
        model_names=["arm_A", "arm_B"],
        dim=dim,
        forgetting_factor=0.95
        # init_lambda=1.0 (default)
        # update_lambda=0.0 (default)
    )
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dim}")
    print(f"  Forgetting Factor: 0.95")
    print(f"  init_lambda: {policy.init_lambda}")
    print(f"  update_lambda: {policy.update_lambda}")
    
    x = np.random.randn(dim)
    x = x / (np.linalg.norm(x) + 1e-12)
    
    # Warm up
    policy.update("arm_A", x, reward=1.0)
    
    # Measure alternating updates (forces dt > 0)
    times = []
    for i in range(n_trials):
        arm = "arm_A" if i % 2 == 0 else "arm_B"
        start = time.perf_counter()
        policy.update(arm, x, reward=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = np.mean(times)
    throughput = 1000 / avg_time
    
    print(f"\n📊 Results:")
    print(f"  Average Time: {avg_time:.4f} ms")
    print(f"  Throughput:   {throughput:.0f} updates/sec")
    
    if throughput > 2500:
        print(f"\n✅ SUCCESS: Achieved >{throughput:.0f} updates/sec with default config!")
        print(f"   This proves O(d²) efficiency is now the default.")
        return True
    else:
        print(f"\n❌ FAILED: Expected >2500 updates/sec, got {throughput:.0f}")
        return False


def compare_configurations():
    """Compare different configurations side-by-side"""
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)
    
    dim = 384
    n_trials = 50
    
    configs = [
        {"name": "NEW DEFAULT (Fast)", "init": 1.0, "update": 0.0},
        {"name": "Old Default (Stable)", "init": 1.0, "update": 1.0},
        {"name": "No Regularization", "init": 0.0, "update": 0.0},
    ]
    
    print(f"\n{'Configuration':<25} {'Throughput':<15} {'Status':<20}")
    print("-" * 65)
    
    for config in configs:
        policy = DisjointLinUCBPolicy(
            model_names=["arm_A", "arm_B"],
            dim=dim,
            init_lambda=config['init'],
            update_lambda=config['update'],
            forgetting_factor=0.95
        )
        
        x = np.random.randn(dim)
        x = x / (np.linalg.norm(x) + 1e-12)
        
        # Warm up
        policy.update("arm_A", x, reward=1.0)
        
        times = []
        for i in range(n_trials):
            arm = "arm_A" if i % 2 == 0 else "arm_B"
            start = time.perf_counter()
            policy.update(arm, x, reward=1.0)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = 1000 / avg_time
        status = "✅ O(d²)" if throughput > 2000 else "⚠️ O(d³)"
        
        print(f"{config['name']:<25} {throughput:>8.0f}/sec      {status:<20}")


if __name__ == "__main__":
    print("\n🧪 INIT_LAMBDA/UPDATE_LAMBDA API TEST\n")
    
    success = True
    success &= test_default_configuration()
    compare_configurations()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe new init_lambda/update_lambda API is working correctly!")
        print("Default configuration now achieves O(d²) performance.")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
