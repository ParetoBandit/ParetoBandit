
import joblib
import numpy as np

def compare_priors(file1, file2):
    print(f"Comparing {file1} and {file2}...")
    try:
        data1 = joblib.load(file1)
        data2 = joblib.load(file2)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Check top-level keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        print(f"Keys mismatch: {keys1} vs {keys2}")
    else:
        print(f"Keys match: {keys1}")

    # Check model list in A and b
    models1 = set(data1['A'].keys())
    models2 = set(data2['A'].keys())
    
    if models1 != models2:
        print(f"Model ID mismatch in A: {models1 ^ models2}")
    else:
        print(f"Models match ({len(models1)} models)")

    # Compare matrices
    all_match = True
    for model_id in models1:
        a1 = data1['A'][model_id]
        a2 = data2['A'][model_id]
        b1 = data1['b'][model_id]
        b2 = data2['b'][model_id]
        
        a_match = np.allclose(a1, a2)
        b_match = np.allclose(b1, b2)
        
        if not a_match:
            print(f"❌ Matrix A mismatch for {model_id}")
            all_match = False
        if not b_match:
            print(f"❌ Vector b mismatch for {model_id}")
            all_match = False
            
    if all_match:
        print("✅ All matrices and vectors are IDENTICAL (np.allclose).")
    else:
        print("Summary of differences:")
        for model_id in models1:
            diff_a = np.sum(np.abs(data1['A'][model_id] - data2['A'][model_id]))
            diff_b = np.sum(np.abs(data1['b'][model_id] - data2['b'][model_id]))
            if diff_a > 0 or diff_b > 0:
                print(f"  {model_id}: A_diff={diff_a:.4e}, b_diff={diff_b:.4e}")

if __name__ == "__main__":
    import sys
    f1 = "data/priors_warmup.joblib"
    f2 = "data/priors_warmup_9_models.joblib"
    compare_priors(f1, f2)
