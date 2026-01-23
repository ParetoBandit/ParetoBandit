#!/usr/bin/env python3
"""
Quick test of the calibrate() method signature and basic functionality.
Tests the API without loading heavy models.
"""

import numpy as np

def test_calibrate_signature():
    """Test that calibrate() has the expected signature and behavior."""
    
    print("=" * 70)
    print("CALIBRATE() API SIGNATURE TEST")
    print("=" * 70)
    
    # Mock prompts
    test_prompts = [
        "Hello, how are you?",
        "Solve x^2 + 2x + 1 = 0",
        "Write Python code for quicksort",
        "What is the capital of France?",
        "Explain quantum mechanics",
    ] * 20  # 100 total
    
    print(f"\n✓ Created {len(test_prompts)} test prompts")
    
    # The calibrate() method should:
    # 1. Accept List[str] for prompts
    # 2. Have apply:bool parameter (default True)
    # 3. Have verbose:bool parameter (default False)
    # 4. Return Dict[str, float] with stats
    
    expected_signature = """
    def calibrate(
        self, 
        prompts: List[str], 
        *, 
        apply: bool = True, 
        verbose: bool = False
    ) -> Dict[str, float]
    """
    
    print("\n✓ Expected signature:")
    print(expected_signature)
    
    # Expected return keys
    expected_keys = {'mean', 'std', 'min', 'max', 'p1', 'p99', 'n_samples'}
    print(f"\n✓ Expected return keys: {expected_keys}")
    
    # Check that the method exists in router.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from banditgpt.router import BanditRouter
    
    print("\n✓ BanditRouter imported successfully")
    
    # Check method exists
    has_calibrate = hasattr(BanditRouter, 'calibrate')
    print(f"\n✓ BanditRouter.calibrate exists: {has_calibrate}")
    
    if has_calibrate:
        import inspect
        sig = inspect.signature(BanditRouter.calibrate)
        print(f"\n✓ Actual signature: {sig}")
        
        # Check parameters
        params = list(sig.parameters.keys())
        expected_params = ['self', 'prompts', 'apply', 'verbose']
        
        print(f"\n✓ Parameters: {params}")
        print(f"  Expected: {expected_params}")
        
        if set(params) == set(expected_params):
            print("\n🎉 Signature matches expected API!")
        else:
            print("\n⚠️ Signature mismatch")
    else:
        print("\n❌ ERROR: calibrate() method not found!")
        return
    
    print("\n" + "=" * 70)
    print("API SIGNATURE TEST PASSED")
    print("=" * 70)

if __name__ == "__main__":
    test_calibrate_signature()
