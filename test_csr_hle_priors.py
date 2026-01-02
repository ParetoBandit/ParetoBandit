#!/usr/bin/env python3
"""
Clean test script to validate CSR vs HLE prior loading fix.

This script verifies that:
1. CSR routers use cluster-specific success rates (task-specific priors)
2. HLE routers use generic HLE benchmark scores (generic priors)
3. The b vectors are meaningfully different between the two approaches
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from banditgpt import BanditRouter

def main():
    print("=" * 70)
    print("CSR vs HLE PRIOR LOADING VALIDATION")
    print("=" * 70)
    
    # Load models
    models_path = Path(__file__).parent / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
   # Create routers
    print("\n[1/3] Creating routers with N_eff=10...")
    csr_router = BanditRouter.create(registry, priors="csr", prior_n_effective=10.0)
    hle_router = BanditRouter.create(registry, priors="hle", prior_n_effective=10.0)
    
    # Verify benchmark_key values
    print(f"\n[2/3] Verifying configuration...")
    print(f"  CSR benchmark_key: '{csr_router.benchmark_key}' (should be 'csr')")
    print(f"  HLE benchmark_key: '{hle_router.benchmark_key}' (should be 'hle')")
    
    assert csr_router.benchmark_key == "csr", "CSR router should have benchmark_key='csr'"
    assert hle_router.benchmark_key == "hle", "HLE router should have benchmark_key='hle'"
    print("  ✓ Benchmark keys are correct")
    
    # Compare b vectors
    print(f"\n[3/3] Comparing prior beliefs (b vectors)...")
    
    sample_models = list(registry.keys())[:3]  # Test first 3 models
    
    differences_found = 0
    for model_id in sample_models:
        csr_b_norm = np.linalg.norm(csr_router.bandit.b[model_id])
        hle_b_norm = np.linalg.norm(hle_router.bandit.b[model_id])
        diff = abs(csr_b_norm - hle_b_norm)
        
        model_short = model_id.split('/')[-1][:25]
        print(f"  {model_short:25s}: CSR={csr_b_norm:8.4f}, HLE={hle_b_norm:8.4f}, Δ={diff:8.4f}")
        
        if diff > 0.001:  # Meaningful difference threshold
            differences_found += 1
    
    print(f"\n  Meaningful differences: {differences_found}/{len(sample_models)}")
    
    # Final validation
    print("\n" + "=" * 70)
    if differences_found > 0:
        print("✅ SUCCESS: CSR and HLE priors are DIFFERENT")
        print("   Task-specific cluster success rates vs generic HLE scores")
        print("   The n_eff ablation should now show meaningful separation!")
    else:
        print("❌ FAILURE: CSR and HLE priors are still IDENTICAL")
        print("   The bug is not fully fixed")
    print("=" * 70)

if __name__ == "__main__":
    main()
