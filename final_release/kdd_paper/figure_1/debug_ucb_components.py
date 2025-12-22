import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
try:
    from .bandit import BanditRouter
except (ImportError, ValueError):
    try:
        from final_release.bandit import BanditRouter
    except (ImportError, ValueError):
        from bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    root_dir = base_dir.parent.parent
    data_dir = root_dir / "data"
    
    # Load Models
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load Priors
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    
    # Initialize Router with HLE Priors
    print("Initializing Router with HLE Priors...")
    router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5, # Testing with 0.5
        prior_strength=20.0,
        priors_meta_path=priors_meta_path
    )
    
    # Dummy Context (Math)
    prompt = "Solve the integral of x^2."
    print(f"\nPrompt: '{prompt}'")
    # Use the local encoder instance
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    x = encoder.encode(prompt)
    x_bias = np.append(x, 1.0)
    
    print("\nInspecting UCB Components for Key Models:")
    print(f"{'Model':<40} | {'Mean (Prior)':<12} | {'Std (Uncertainty)':<12} | {'Alpha*Std':<12} | {'UCB':<12}")
    print("-" * 100)
    
    for m_id in ["meta-llama/llama-3.2-1b-instruct", "deepseek/deepseek-r1-0528-qwen3-8b", "google/gemini-2.0-flash-001", "openai/gpt-oss-20b"]:
        if m_id not in router.bandit.models:
            continue
            
        A_inv = router.bandit.A_inv[m_id]
        # Calculate theta on the fly as select_arm does
        theta = A_inv @ router.bandit.b[m_id]
        
        # Mean
        mean = float(theta.dot(x_bias))
        
        # Std
        var = float(x_bias.dot(A_inv).dot(x_bias))
        std = np.sqrt(var)
        
        # UCB
        alpha = 0.5
        confidence = alpha * std
        ucb = mean + confidence
        
        print(f"{m_id:<40} | {mean:.4f}       | {std:.4f}       | {confidence:.4f}       | {ucb:.4f}")
        
    print("-" * 100)
    print("Analysis:")
    print("If Mean >> Alpha*Std, the Prior dominates -> Alpha doesn't matter much.")
    print("If Mean ~ Alpha*Std, Exploration matters.")

if __name__ == "__main__":
    main()
