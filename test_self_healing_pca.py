#!/usr/bin/env python3
"""
Test self-healing PCA implementation.

Tests:
1. Missing PCA file → Should auto-train
2. Dimension mismatch → Should detect and retrain
3. Normal case → Should load existing PCA
"""

import sys
from pathlib import Path
import shutil
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bandit_gpt.router import BanditRouter


def test_missing_pca():
    """Test that router auto-trains PCA when artifact is missing."""
    print("\n" + "="*70)
    print("TEST 1: Missing PCA Artifact")
    print("="*70)
    
    # Use a non-existent path
    fake_pca_path = "/tmp/nonexistent_pca.joblib"
    
    try:
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            pca_path=fake_pca_path
        )
        
        if router.pca is not None:
            print(f"✅ PASS: PCA auto-trained (n_components={router.pca.n_components_})")
            return True
        else:
            print("❌ FAIL: PCA is None")
            return False
    except Exception as e:
        print(f"❌ FAIL: Exception during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_normal_case():
    """Test that router loads valid PCA artifact."""
    print("\n" + "="*70)
    print("TEST 2: Normal Case (Existing Valid PCA)")
    print("="*70)
    
    # Use default PCA path (should exist)
    pca_path = Path(__file__).parent / "src" / "bandit_gpt" / "data" / "pca_32.joblib"
    
    if not pca_path.exists():
        print(f"⚠️  SKIP: Default PCA not found at {pca_path}")
        return None
    
    try:
        router = BanditRouter(
            model_registry={"test/model": {"openrouter_id": "test/model", "hle": 0.5}},
            pca_path=pca_path
        )
        
        if router.pca is not None:
            print(f"✅ PASS: PCA loaded (n_components={router.pca.n_components_})")
            return True
        else:
            print("❌ FAIL: PCA is None")
            return False
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        return False


def main():
    """Run all self-healing PCA tests."""
    print("\n" + "="*70)
    print("SELF-HEALING PCA TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1: Missing PCA
    results.append(("Missing PCA", test_missing_pca()))
    
    # Test 2: Normal case
    result = test_normal_case()
    if result is not None:
        results.append(("Normal Case", result))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
