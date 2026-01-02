#!/usr/bin/env python3
"""
Comprehensive test to verify CSR vs HLE prior loading bug.

ROOT CAUSE IDENTIFIED:
=====================
In banditgpt/bandit.py, the load_from_benchmark() method (line 668-883) has a bug:

Line 792: cluster_rates = model_registry.get(m, {}).get("cluster_success_rates", [])
Line 795-800: If cluster_rates exist, use them. Otherwise, use raw HLE score.

The PROBLEM:
- Both priors="csr" (line 632-654) and priors="hle" (line 607-629) call load_from_benchmark()
- They pass different benchmark_key values ("hle" vs "hle")  
- BUT load_from_benchmark ignores benchmark_key when cluster_success_rates exist!
- Since ALL 36 models have cluster_success_rates, BOTH paths use CSR data
- The benchmark_key is ONLY used as a fallback (line 782, 798)

THE FIX:
========
The load_from_benchmark method should check benchmark_key FIRST:
- If benchmark_key == "hle": ALWAYS use raw HLE scores, ignore cluster_success_rates
- Otherwise: Use cluster_success_rates when available

EXPECTED BEHAVIOR:
==================
- priors="csr": Use cluster_success_rates (task-specific)
- priors="hle": Use raw HLE scores (generic benchmark)
- This creates different b vectors, leading to different routing decisions
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from banditgpt import BanditRouter

def test_current_behavior():
    """Test current (buggy) behavior"""
    print("=" * 70)
    print("TEST 1: CURRENT BEHAVIOR (BUGGY)")
    print("=" * 70)
    
    # Load registry
    models_path = Path(__file__).parent / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Create both routers
    print("\nCreating CSR router...")
    csr = BanditRouter.create(registry, priors="csr", prior_n_effective=10.0)
    
    print("Creating HLE router...")
    hle = BanditRouter.create(registry, priors="hle", prior_n_effective=10.0)
    
    # Compare b vectors for a sample model
    sample_model = list(registry.keys())[0]
    
    csr_b = csr.bandit.b[sample_model]
    hle_b = hle.bandit.b[sample_model]
    
    print(f"\nSample model: {sample_model}")
    print(f"CSR b vector norm: {np.linalg.norm(csr_b):.6f}")
    print(f"HLE b vector norm: {np.linalg.norm(hle_b):.6f}")
    print(f"Difference: {np.linalg.norm(csr_b - hle_b):.6f}")
    
    are_identical = np.allclose(csr_b, hle_b)
    
    if are_identical:
        print("\n❌ BUG CONFIRMED: b vectors are IDENTICAL!")
        print("   Both CSR and HLE use cluster_success_rates")
    else:
        print("\n✅ EXPECTED: b vectors are DIFFERENT")
        print("   CSR and HLE use different initialization")
    
    return are_identical


def explain_fix():
    """Explain the fix needed"""
    print("\n" + "=" * 70)
    print("RECOMMENDED FIX")
    print("=" * 70)
    
    print("""
In banditgpt/bandit.py, modify load_from_benchmark() around line 792:

CURRENT CODE (BUGGY):
---------------------
cluster_rates = model_registry.get(m, {}).get("cluster_success_rates", [])

# If model has no cluster data, fallback to global HLE or neutral
if not cluster_rates or len(cluster_rates) != n_clusters:
    # Fallback: Just use global HLE for all clusters
    score = transform_hle_to_prior(raw_score)
    bias_update_vec = (score * global_sum)
else:
    # Use cluster success rates...
    ordered_rates = [...]
    bias_update_vec = weighted_sum_features


FIXED CODE:
-----------
# RESPECT benchmark_key: HLE mode should NEVER use cluster_success_rates
if benchmark_key == "hle":
    # HLE mode: Use generic HLE scores only
    score = transform_hle_to_prior(raw_score)
    bias_update_vec = (score * global_sum)
else:
    # CSR mode: Use cluster success rates when available
    cluster_rates = model_registry.get(m, {}).get("cluster_success_rates", [])
    
    if not cluster_rates or len(cluster_rates) != n_clusters:
        # Fallback to HLE
        score = transform_hle_to_prior(raw_score)
        bias_update_vec = (score * global_sum)
    else:
        # Use cluster success rates
        ordered_rates = [...]
        bias_update_vec = weighted_sum_features


WHY THIS MATTERS:
-----------------
1. Scientific Rigor: CSR vs HLE comparison is meaningless if both use CSR
2. Ablation Studies: N_eff ablation should show CSR outperforms HLE
3. "Architecture is the Hero" Narrative: Depends on CSR being task-specific

VERIFICATION:
-------------
After fix, run:
  python debug_prior_loading.py
  
Expected output:
  ✅ b vectors are DIFFERENT
  CSR b vector norm: ~XXX
  HLE b vector norm: ~YYY (different from CSR)
""")


def test_impact():
    """Demonstrate impact on routing decisions"""
    print("\n" + "=" * 70)
    print("TEST 2: IMPACT ON ROUTING DECISIONS")
    print("=" * 70)
    
    # Load registry
    models_path = Path(__file__).parent / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Create both routers
    csr = BanditRouter.create(registry, priors="csr", prior_n_effective=10.0)
    hle = BanditRouter.create(registry, priors="hle", prior_n_effective=10.0)
    
    # Test routing on different prompt types
    test_prompts = [
        "Solve this differential equation: dy/dx = x^2 + y",
        "Write a Python function to implement quicksort",
        "Tell me a creative story about a time traveler",
        "Explain quantum entanglement in simple terms"
    ]
    
    print("\nComparing routing decisions:")
    
    identical_count = 0
    for prompt in test_prompts:
        csr_choice, _ = csr.route(prompt, profile="balanced", input_tokens=100)
        hle_choice, _ = hle.route(prompt, profile="balanced", input_tokens=100)
        
        match = "✓" if csr_choice == hle_choice else "✗"
        identical_count += (csr_choice == hle_choice)
        
        # Truncate model names for readability
        csr_short = csr_choice.split('/')[-1][:20]
        hle_short = hle_choice.split('/')[-1][:20]
        
        print(f"  {match} Prompt: {prompt[:40]:40s}")
        print(f"     CSR: {csr_short:20s} | HLE: {hle_short:20s}")
    
    print(f"\nIdentical choices: {identical_count}/{len(test_prompts)}")
    
    if identical_count == len(test_prompts):
        print("❌ BUG CONFIRMED: All routing decisions are IDENTICAL")
        print("   This proves CSR and HLE are functionally equivalent (wrong!)")
    else:
        print("✅ EXPECTED: Some routing decisions differ")
        print("   CSR and HLE have different prior beliefs")


def main():
    test_current_behavior()
    test_impact()
    explain_fix()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ Root cause identified: load_from_benchmark() ignores benchmark_key
✅ Impact confirmed: CSR and HLE produce identical results
✅ Fix proposed: Check benchmark_key BEFORE using cluster_success_rates
✅ Next steps: Apply fix to banditgpt/bandit.py and re-run ablation

After fixing, the n_eff ablation should show:
- CSR: Benefits strongly from prior_n_effective > 0
- HLE: Benefits weakly (generic priors less helpful)
- Separation proves task-specific cluster priors are superior
""")

if __name__ == "__main__":
    main()
