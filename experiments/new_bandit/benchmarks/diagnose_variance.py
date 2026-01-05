#!/usr/bin/env python3
"""
Diagnose the source of variance in Sherman-Morrison benchmarks.

Tests different hypotheses:
1. Policy creation overhead
2. Random data generation
3. NumPy operation variance
4. Memory allocation effects
"""

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from router import DisjointLinUCBPolicy


def test_hypothesis_1_policy_creation():
    """
    Hypothesis: Variance comes from policy creation overhead.
    
    Test: Reuse same policy vs create new policy each time.
    """
    print("=" * 70)
    print("HYPOTHESIS 1: Policy Creation Overhead")
    print("=" * 70)
    
    dim = 384
    n_trials = 100
    
    # Test A: Create policy once, reuse
    print("\nTest A: Reusing same policy instance")
    policy = DisjointLinUCBPolicy(
        model_names=["arm_A", "arm_B"],
        dim=dim,
        init_lambda=1.0,
        update_lambda=0.0,
        forgetting_factor=0.95
    )
    
    x = np.random.randn(dim)
    x = x / (np.linalg.norm(x) + 1e-12)
    
    # Warm up
    policy.update("arm_A", x, reward=1.0)
    
    times_reuse = []
    for i in range(n_trials):
        arm = "arm_A" if i % 2 == 0 else "arm_B"
        start = time.perf_counter()
        policy.update(arm, x, reward=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times_reuse.append(elapsed)
    
    print(f"  Mean: {np.mean(times_reuse):.4f} ms")
    print(f"  Std:  {np.std(times_reuse):.4f} ms")
    print(f"  CV:   {(np.std(times_reuse)/np.mean(times_reuse))*100:.1f}%")
    
    # Test B: Create new policy each time (like benchmark)
    print("\nTest B: Creating new policy each iteration")
    times_new = []
    for i in range(n_trials):
        policy_new = DisjointLinUCBPolicy(
            model_names=["arm_A", "arm_B"],
            dim=dim,
            init_lambda=1.0,
            update_lambda=0.0,
            forgetting_factor=0.95
        )
        
        x_new = np.random.randn(dim)
        x_new = x_new / (np.linalg.norm(x_new) + 1e-12)
        policy_new.update("arm_A", x_new, reward=1.0)
        
        arm = "arm_A" if i % 2 == 0 else "arm_B"
        start = time.perf_counter()
        policy_new.update(arm, x_new, reward=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times_new.append(elapsed)
    
    print(f"  Mean: {np.mean(times_new):.4f} ms")
    print(f"  Std:  {np.std(times_new):.4f} ms")
    print(f"  CV:   {(np.std(times_new)/np.mean(times_new))*100:.1f}%")
    
    print(f"\n🔍 Analysis:")
    if np.std(times_new) > np.std(times_reuse) * 2:
        print(f"  ✓ Policy creation adds significant variance!")
        print(f"    Variance increase: {(np.std(times_new)/np.std(times_reuse)):.1f}x")
    else:
        print(f"  ✗ Policy creation is not the main source")
    
    return times_reuse, times_new


def test_hypothesis_2_random_data():
    """
    Hypothesis: Variance comes from random data generation.
    
    Test: Fixed data vs random data.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Random Data Generation")
    print("=" * 70)
    
    dim = 384
    n_trials = 100
    
    policy = DisjointLinUCBPolicy(
        model_names=["arm_A", "arm_B"],
        dim=dim,
        init_lambda=1.0,
        update_lambda=0.0,
        forgetting_factor=0.95
    )
    
    # Test A: Fixed context vector
    print("\nTest A: Fixed context vector")
    x_fixed = np.random.randn(dim)
    x_fixed = x_fixed / (np.linalg.norm(x_fixed) + 1e-12)
    policy.update("arm_A", x_fixed, reward=1.0)
    
    times_fixed = []
    for i in range(n_trials):
        arm = "arm_A" if i % 2 == 0 else "arm_B"
        start = time.perf_counter()
        policy.update(arm, x_fixed, reward=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times_fixed.append(elapsed)
    
    print(f"  Mean: {np.mean(times_fixed):.4f} ms")
    print(f"  Std:  {np.std(times_fixed):.4f} ms")
    print(f"  CV:   {(np.std(times_fixed)/np.mean(times_fixed))*100:.1f}%")
    
    # Test B: Random context each time
    print("\nTest B: Random context each iteration")
    policy2 = DisjointLinUCBPolicy(
        model_names=["arm_A", "arm_B"],
        dim=dim,
        init_lambda=1.0,
        update_lambda=0.0,
        forgetting_factor=0.95
    )
    x_init = np.random.randn(dim)
    x_init = x_init / (np.linalg.norm(x_init) + 1e-12)
    policy2.update("arm_A", x_init, reward=1.0)
    
    times_random = []
    for i in range(n_trials):
        x_rand = np.random.randn(dim)
        x_rand = x_rand / (np.linalg.norm(x_rand) + 1e-12)
        
        arm = "arm_A" if i % 2 == 0 else "arm_B"
        start = time.perf_counter()
        policy2.update(arm, x_rand, reward=1.0)
        elapsed = (time.perf_counter() - start) * 1000
        times_random.append(elapsed)
    
    print(f"  Mean: {np.mean(times_random):.4f} ms")
    print(f"  Std:  {np.std(times_random):.4f} ms")
    print(f"  CV:   {(np.std(times_random)/np.mean(times_random))*100:.1f}%")
    
    print(f"\n🔍 Analysis:")
    if np.std(times_random) > np.std(times_fixed) * 1.5:
        print(f"  ✓ Random data adds variance!")
        print(f"    Variance increase: {(np.std(times_random)/np.std(times_fixed)):.1f}x")
    else:
        print(f"  ✗ Random data is not the main source")
    
    return times_fixed, times_random


def test_hypothesis_3_staleness_effect():
    """
    Hypothesis: Variance comes from staleness dt affecting numerical stability.
    
    Test: Different staleness levels with same policy.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Staleness-Dependent Numerical Effects")
    print("=" * 70)
    
    dim = 384
    n_trials = 50
    
    staleness_levels = [0, 1, 10, 100]
    
    print(f"\n{'Staleness':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'CV (%)':<12}")
    print("-" * 50)
    
    results = {}
    for dt in staleness_levels:
        times = []
        for _ in range(n_trials):
            policy = DisjointLinUCBPolicy(
                model_names=["arm_A"],
                dim=dim,
                init_lambda=1.0,
                update_lambda=0.0,
                forgetting_factor=0.95
            )
            
            x = np.random.randn(dim)
            x = x / (np.linalg.norm(x) + 1e-12)
            policy.update("arm_A", x, reward=1.0)
            
            if dt > 0:
                policy.last_update["arm_A"] = policy.t - dt
            
            start = time.perf_counter()
            policy.update("arm_A", x, reward=1.0)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = (std_time / mean_time) * 100
        
        results[dt] = (mean_time, std_time, cv)
        print(f"dt={dt:<9} {mean_time:<12.4f} {std_time:<12.4f} {cv:<12.1f}")
    
    print(f"\n🔍 Analysis:")
    cv_values = [cv for _, _, cv in results.values()]
    if max(cv_values) > min(cv_values) * 2:
        print(f"  ✓ Staleness affects variance significantly!")
        print(f"    CV range: {min(cv_values):.1f}% to {max(cv_values):.1f}%")
    else:
        print(f"  ✗ Staleness has minimal effect on variance")
    
    return results


if __name__ == "__main__":
    print("\n🔬 VARIANCE DIAGNOSTIC SUITE\n")
    
    # Run all tests
    reuse_times, new_times = test_hypothesis_1_policy_creation()
    fixed_times, random_times = test_hypothesis_2_random_data()
    staleness_results = test_hypothesis_3_staleness_effect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nCoefficient of Variation (CV) Comparison:")
    print(f"  Reused policy + fixed data:    {(np.std(fixed_times)/np.mean(fixed_times))*100:.1f}%")
    print(f"  Reused policy + random data:   {(np.std(random_times)/np.mean(random_times))*100:.1f}%")
    print(f"  New policy each time:          {(np.std(new_times)/np.mean(new_times))*100:.1f}%")
    
    print("\n🎯 Primary Sources of Variance:")
    
    # Determine primary source
    if np.std(new_times) > np.std(reuse_times) * 2:
        print("  1. ⚠️  Policy creation overhead (MAJOR)")
    if np.std(random_times) > np.std(fixed_times) * 1.5:
        print("  2. ⚠️  Random data generation (MODERATE)")
    
    print("\n💡 Recommendation:")
    print("  To reduce variance in benchmarks:")
    print("  - Increase number of trials (e.g., 100 → 500)")
    print("  - Add warmup period before timing")
    print("  - Use median instead of mean for robustness")
    print("  - Consider running GC before each trial")
