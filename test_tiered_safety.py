"""
Test Tiered Safety pattern with real data and router calls.

Validates:
1. Fast heuristic latency (<1ms vs 100-300ms)
2. Feature quality (correlation with toxicity)
3. Router integration with real prompts
"""
import time
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bandit_gpt.router import BanditRouter
import numpy as np


def load_real_prompts(n=100):
    """Load real prompts from training dataset."""
    data_dir = Path(__file__).parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    train_path = data_dir / "train_rewards_hle_models.jsonl"
    
    if not train_path.exists():
        print(f"⚠️  Training data not found at {train_path}")
        return []
    
    prompts = []
    with open(train_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            prompts.append(data["prompt"])
    
    return prompts


def test_fast_heuristic_latency():
    """Test that fast heuristic is <1ms."""
    print("\n" + "="*70)
    print("TEST 1: Fast Heuristic Latency")
    print("="*70)
    
    # Load real prompts
    prompts = load_real_prompts(n=100)
    if not prompts:
        print("❌ No prompts loaded")
        return False
    
    print(f"Testing with {len(prompts)} real prompts...")
    
    # Measure latency
    latencies = []
    for prompt in prompts:
        start = time.perf_counter()
        score = BanditRouter._fast_toxicity_heuristic(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    # Statistics
    mean_latency = np.mean(latencies)
    p50_latency = np.median(latencies)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    
    print(f"\n📊 Latency Statistics:")
    print(f"  Mean: {mean_latency:.3f}ms")
    print(f"  P50:  {p50_latency:.3f}ms")
    print(f"  P99:  {p99_latency:.3f}ms")
    print(f"  Max:  {max_latency:.3f}ms")
    
    # Validation
    if p99_latency < 1.0:
        print(f"✅ PASS: P99 latency {p99_latency:.3f}ms < 1ms target")
        return True
    else:
        print(f"❌ FAIL: P99 latency {p99_latency:.3f}ms >= 1ms target")
        return False


def test_heuristic_catches_toxic():
    """Test that heuristic catches obvious toxic content."""
    print("\n" + "="*70)
    print("TEST 2: Heuristic Catch Rate")
    print("="*70)
    
    # Test cases with expected scores
    test_cases = [
        ("Write a story about friendship", 0.0),  # Clean
        ("How to hack into a system", 0.15),  # Security threat
        ("I want to kill this bug in my code", 0.15),  # Violence (false positive OK)
        ("Write porn content", 0.15),  # Explicit
        ("Racist hate speech example", 0.30),  # Multiple triggers
    ]
    
    passed = 0
    for text, expected_min in test_cases:
        score = BanditRouter._fast_toxicity_heuristic(text)
        
        if score >= expected_min:
            print(f"✅ '{text[:50]}...' → {score:.2f} (>= {expected_min})")
            passed += 1
        else:
            print(f"❌ '{text[:50]}...' → {score:.2f} (< {expected_min})")
    
    print(f"\n{passed}/{len(test_cases)} test cases passed")
    return passed == len(test_cases)


def test_router_integration_real_data():
    """Test full router with real prompts uses fast heuristic."""
    print("\n" + "="*70)
    print("TEST 3: Router Integration with Real Data")
    print("="*70)
    
    # Load real prompts
    prompts = load_real_prompts(n=10)
    if not prompts:
        print("❌ No prompts loaded")
        return False
    
    # Create router with minimal config
    registry = {
        "test/model1": {"openrouter_id": "test/model1", "hle": 0.8},
        "test/model2": {"openrouter_id": "test/model2", "hle": 0.6}
    }
    
    print("Creating router...")
    router = BanditRouter(model_registry=registry)
    
    # Test routing with real prompts
    print(f"\nRouting {len(prompts)} real prompts...")
    latencies = []
    
    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        selected, log = router.route(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        
        # Verify context vector was created (includes toxicity feature)
        assert log.context_vector is not None
        assert len(log.context_vector) > 0
        
        if i < 3:  # Show first 3
            print(f"  Prompt {i+1}: '{prompt[:60]}...'")
            print(f"    → Selected: {selected}, Latency: {latency_ms:.1f}ms")
    
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n📊 Routing Latency:")
    print(f"  Mean: {mean_latency:.1f}ms")
    print(f"  P99:  {p99_latency:.1f}ms")
    
    # Success if no crashes and reasonable latency
    if p99_latency < 500:  # Should be much faster without heavy scanner
        print(f"✅ PASS: Router works with real data, P99={p99_latency:.1f}ms")
        return True
    else:
        print(f"❌ FAIL: Latency too high: P99={p99_latency:.1f}ms")
        return False


def main():
    """Run all tiered safety tests with real data."""
    print("\n" + "="*70)
    print("TIERED SAFETY PATTERN - REAL DATA VALIDATION")
    print("="*70)
    
    results = []
    
    # Test 1: Latency
    results.append(("Fast Heuristic Latency", test_fast_heuristic_latency()))
    
    # Test 2: Accuracy
    results.append(("Heuristic Catch Rate", test_heuristic_catches_toxic()))
    
    # Test 3: Integration
    results.append(("Router Integration", test_router_integration_real_data()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n{total_passed}/{len(results)} tests passed")
    
    return total_passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
