#!/usr/bin/env python3
"""
Sanitize Priors Script
----------------------
Fixes "Scale Explosion" in cached priors by rescaling the b-vectors
so that initial predictions (theta) fall within the normalized [0, 1] range.

Why this is needed:
Scaling A and b by the same gamma (as done in router.py) changes CONFIDENCE
but preserves the PREDICTION (theta = A^-1 b). To fix the prediction scale,
we must change the ratio of b to A.
"""

import sys
from pathlib import Path
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Paths
INPUT_PATH = project_root / "src/artifacts/priors_warmup.joblib"
OUTPUT_PATH = project_root / "src/artifacts/priors_warmup_normalized.joblib"

def sanitize_priors():
    print(f"📥 Loading priors from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"❌ Error: Input file not found at {INPUT_PATH}")
        return

    data = joblib.load(INPUT_PATH)
    
    # Preserve ALL original metadata
    new_data = data.copy()
    new_data["A"] = {}
    new_data["b"] = {}
    
    # Target initial quality (0.8 = strong prior belief in high quality)
    TARGET_BIAS = 0.8
    
    print("\n🔧 SANITIZING MODELS:")
    print(f"{'Model':<40} | {'Old Pred':<10} | {'New Pred':<10} | {'Scale Factor'}")
    print("-" * 85)

    for model_id in data["A"].keys():
        A = data["A"][model_id]
        b = data["b"][model_id]
        
        # 1. Calculate current broken prediction
        # theta = A^-1 * b
        try:
            A_inv = np.linalg.inv(A)
            theta_old = A_inv @ b
            pred_old = theta_old[-1] # Bias term is usually the last element
        except np.linalg.LinAlgError:
            print(f"⚠️  {model_id}: Matrix singular, skipping...")
            continue
            
        # 2. Calculate Correction Factor
        # We want pred_new = TARGET_BIAS
        # Since pred is linear in b, scale_factor = Target / Old
        if abs(pred_old) > 1e-6:
            scale = TARGET_BIAS / pred_old
        else:
            scale = 1.0
            
        # 3. Apply Correction to b ONLY
        # Keeping A constant preserves our "confidence" (sample size)
        # Changing b changes our "preference" (utility estimate)
        b_new = b * scale
        
        # Verify
        theta_new = A_inv @ b_new
        pred_new = theta_new[-1]
        
        # Store
        new_data["A"][model_id] = A
        new_data["b"][model_id] = b_new
        
        print(f"{model_id[:40]:<40} | {pred_old:10.4f} | {pred_new:10.4f} | {scale:.2e}")

    # Save
    print("-" * 85)
    print(f"💾 Saving sanitized priors to: {OUTPUT_PATH}")
    
    # Create directory if it doesn't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(new_data, OUTPUT_PATH)
    print("✅ Done! Update your experiment script to point to this new file.")

if __name__ == "__main__":
    sanitize_priors()
