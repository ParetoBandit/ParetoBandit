import json
import numpy as np
from pathlib import Path

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    models_path = base_dir / "final_release/models.json"
    priors_meta_path = base_dir / "final_release/data/priors_meta_large.npz"
    output_path = base_dir / "banditgpt/data/priors/expert_priors.npz"
    
    print(f"Loading models from {models_path}...")
    with open(models_path) as f:
        models_data = json.load(f)["models"]
    
    model_registry = {m["openrouter_id"]: m for m in models_data}
    model_names = sorted(model_registry.keys())
    
    print(f"Loading priors meta from {priors_meta_path}...")
    meta = np.load(priors_meta_path)
    cov_matrix = meta["cov_matrix"]
    sum_vec = meta["sum_vec"]
    dim = sum_vec.shape[0]
    
    print(f"Generating composite priors for {len(model_names)} models...")
    A_stack = []
    b_stack = []
    
    benchmark_key = "hle"
    
    for m_id in model_names:
        m_data = model_registry[m_id]
        score = float(m_data.get(benchmark_key, 0.0))
        
        # Ridge Initialization logic (strength=1.0 for the base file)
        # A = I + Cov
        # b = score * Sum
        A = np.eye(dim) + cov_matrix
        b = score * sum_vec
        
        A_stack.append(A)
        b_stack.append(b)
    
    A_stack = np.stack(A_stack)
    b_stack = np.stack(b_stack)
    
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        model_names=np.array(model_names, dtype=object),
        dim=dim,
        alpha=0.5,
        A_stack=A_stack.astype(np.float16),
        b_stack=b_stack.astype(np.float16),
        generated_at=np.array("composite_aa_quality_index", dtype=object)
    )
    print("Done.")

if __name__ == "__main__":
    main()
