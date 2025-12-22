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
    
    # Dummy Data
    prompts = ["Test prompt " + str(i) for i in range(10)]
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(prompts, normalize_embeddings=True)
    
    # Run Trace
    for alpha in [0.1, 2.0]:
        print(f"\n{'='*40}")
        print(f"TRACING ALPHA = {alpha}")
        print(f"{'='*40}")
        
        router = BanditRouter(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=alpha,
            embedding_dim=embeddings.shape[1]
        )
        
        # Step 0: Initial State
        print("\nStep 0 (Before any updates):")
        x = embeddings[0]
        x_bias = np.append(x, 1.0)
        
        # Inspect UCBs manually
        print(f"{'Model':<40} | {'Mean':<8} | {'Std':<8} | {'UCB':<8}")
        print("-" * 70)
        for m in router.bandit.models[:5]: # Show first 5
            A_inv = router.bandit.A_inv[m]
            theta = A_inv @ router.bandit.b[m]
            mean = float(theta.dot(x_bias))
            var = float(x_bias.dot(A_inv).dot(x_bias))
            std = np.sqrt(var)
            ucb = mean + alpha * std
            print(f"{m:<40} | {mean:.4f}   | {std:.4f}   | {ucb:.4f}")
            
        # Simulate a selection and update
        chosen, _ = router.bandit.select_arm(x_bias)
        print(f"\nSelected: {chosen}")
        
        # Fake Reward (1.0 if Llama, 0.0 otherwise)
        reward = 1.0 if "llama" in chosen else 0.0
        print(f"Observed Reward: {reward}")
        router.bandit.update(chosen, x_bias, reward)
        
        # Step 1: After 1 update
        print("\nStep 1 (After 1 update):")
        x = embeddings[1]
        x_bias = np.append(x, 1.0)
        
        print(f"{'Model':<40} | {'Mean':<8} | {'Std':<8} | {'UCB':<8}")
        print("-" * 70)
        for m in router.bandit.models[:5]:
            A_inv = router.bandit.A_inv[m]
            theta = A_inv @ router.bandit.b[m]
            mean = float(theta.dot(x_bias))
            var = float(x_bias.dot(A_inv).dot(x_bias))
            std = np.sqrt(var)
            ucb = mean + alpha * std
            print(f"{m:<40} | {mean:.4f}   | {std:.4f}   | {ucb:.4f}")
            
        chosen, _ = router.bandit.select_arm(x_bias)
        print(f"\nSelected: {chosen}")

if __name__ == "__main__":
    main()
