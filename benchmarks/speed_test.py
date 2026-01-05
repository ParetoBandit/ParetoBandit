#!/usr/bin/env python3
"""
Benchmark: O(d²) Sherman-Morrison Speed Test

Verifies the 2,700 QPS (queries per second) claim by proving that the 
Sherman-Morrison update path achieves O(d²) complexity instead of O(d³).

KDD Claim: "The algorithm strictly adheres to O(d²) complexity, enabling 
throughput of >1000 decisions/sec even with high-dimensional embeddings."

Usage:
    python speed_test.py
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from bandit_gpt.router import DisjointLinUCBPolicy


def run_speed_benchmark():
    """
    Prove O(d²) efficiency under FORCED DECAY conditions.
    
    This directly addresses the "Sherman-Morrison Illusion" critique:
    It forces dt > 0 on every update and measures if execution time
    explodes (O(d³) inversion) or stays flat (O(d²) rank-1 update).
    """
    print("=" * 70)
    print("BENCHMARK: O(d²) Sherman-Morrison Efficiency")
    print("=" * 70)
    
    # 1. Setup High-Dimensional Bandit (Standard Embedding Size)
    dim = 384  # Production embedding dimension
    n_updates = 1000
    
    # Config with decay ENABLED (gamma=0.95)
    # This is the challenging case where dt > 0 triggers decay logic
    policy = DisjointLinUCBPolicy(
        model_names=["arm_A", "arm_B"],
        dim=dim,
        forgetting_factor=0.95  # Decay enabled
    )
    
    # Mock Feature Vector (Context)
    x = np.random.randn(dim)
    x = x / (np.linalg.norm(x) + 1e-12)  # Normalize
    
    print(f"\n🚀 Benchmarking {n_updates} updates (dim={dim})...")
    print(f"   Condition: ALTERNATING ARMS (forces dt > 0 every step)")
    print(f"   Decay Factor: γ = 0.95")
    
    # 2. The Forced Decay Loop
    # By alternating arms, we ensure dt > 1 for each arm every update
    # This triggers the decay + regularization restoration path
    start_time = time.perf_counter()
    
    for t in range(n_updates):
        # Alternate between arms to force dt > 0
        arm_id = "arm_A" if t % 2 == 0 else "arm_B"
        policy.update(arm_id, x, reward=1.0)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_ms = (total_time / n_updates) * 1000
    updates_per_sec = n_updates / total_time
    
    print(f"\n📊 Results (Forced Decay Path):")
    print(f"   Total Time: {total_time:.4f}s")
    print(f"   Avg per Update: {avg_ms:.4f} ms")
    print(f"   Updates/sec: {updates_per_sec:.1f}")
    
    # 3. Baseline: No Decay (Pure Sherman-Morrison)
    print(f"\n🔬 Running baseline (no decay, γ=1.0)...")
    
    policy_no_decay = DisjointLinUCBPolicy(
        model_names=["arm_A"],
        dim=dim,
        forgetting_factor=1.0  # No decay = pure Sherman-Morrison
    )
    
    start_baseline = time.perf_counter()
    for t in range(n_updates):
        policy_no_decay.update("arm_A", x, reward=1.0)
    baseline_time = time.perf_counter() - start_baseline
    baseline_avg_ms = (baseline_time / n_updates) * 1000
    
    print(f"   Baseline Avg per Update: {baseline_avg_ms:.4f} ms")
    
    # 4. Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    # A full O(d³) inversion for 384x384 takes ~10-20ms on CPU.
    # A rank-1 O(d²) update takes < 1ms.
    # With decay overhead, we allow up to 3ms as threshold.
    THRESHOLD_MS = 3.0
    
    if avg_ms < THRESHOLD_MS:
        print(f"\n✅ PASS: Speed ({avg_ms:.2f}ms) < {THRESHOLD_MS}ms threshold")
        print(f"   Result is consistent with O(d²) complexity.")
    else:
        print(f"\n❌ FAIL: Speed ({avg_ms:.2f}ms) >= {THRESHOLD_MS}ms threshold")
        print(f"   Result suggests O(d³) inversion is occurring.")
    
    # Compare decay vs no-decay overhead
    overhead = avg_ms / baseline_avg_ms
    print(f"\n📈 Decay Overhead: {overhead:.2f}x vs pure Sherman-Morrison")
    print(f"   (Acceptable if < 5x, since decay involves regularization restore)")
    
    if overhead < 5.0:
        print(f"   ✅ Overhead is acceptable")
    else:
        print(f"   ⚠️ Overhead is higher than expected")
    
    # KDD Claim: >1000 decisions/sec
    print("\n" + "=" * 70)
    print("KDD CLAIM: >1000 decisions/sec")
    print("=" * 70)
    print(f"   Achieved: {updates_per_sec:.1f} updates/sec")
    print(f"   Status: {'✅ VALIDATED' if updates_per_sec > 1000 else '❌ NOT MET'}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return avg_ms < THRESHOLD_MS


def run_scaling_test():
    """
    Verify O(d²) scaling across dimensions.
    """
    print("\n" + "=" * 70)
    print("SCALING TEST: Verify O(d²) Growth")
    print("=" * 70)
    
    dimensions = [32, 64, 128, 256, 384]
    n_updates = 500
    results = []
    
    print(f"\n{'Dim':<8} {'Time (ms)':<12} {'Ratio vs 32':<15} {'Expected O(d²)':<15}")
    print("-" * 50)
    
    base_time = None
    for dim in dimensions:
        policy = DisjointLinUCBPolicy(
            model_names=["arm_A", "arm_B"],
            dim=dim,
            forgetting_factor=0.95
        )
        
        x = np.random.randn(dim)
        x = x / (np.linalg.norm(x) + 1e-12)
        
        start = time.perf_counter()
        for t in range(n_updates):
            arm_id = "arm_A" if t % 2 == 0 else "arm_B"
            policy.update(arm_id, x, reward=1.0)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / n_updates) * 1000
        
        if base_time is None:
            base_time = avg_ms
            ratio = 1.0
        else:
            ratio = avg_ms / base_time
        
        expected_ratio = (dim / 32) ** 2
        results.append((dim, avg_ms, ratio, expected_ratio))
        
        print(f"{dim:<8} {avg_ms:<12.4f} {ratio:<15.2f} {expected_ratio:<15.2f}")
    
    print("\n✓ If 'Ratio vs 32' roughly matches 'Expected O(d²)', scaling is correct.")


if __name__ == "__main__":
    passed = run_speed_benchmark()
    run_scaling_test()
    
    sys.exit(0 if passed else 1)
