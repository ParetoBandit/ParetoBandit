import json
import numpy as np
from pathlib import Path
import sys

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

try:
    from bandit import BanditRouter
except ImportError:
    from final_release.bandit import BanditRouter

def compare_forgetting():
    root_dir = Path(__file__).parent
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    
    target_model_id = "deepseek/deepseek-r1"
    teacher_pet_id = "google/gemini-3-pro-preview"
    target_reward = 5.8
    teacher_pet_reward = 4.2
    
    meta = np.load(priors_meta_path)
    sum_vec = meta["sum_vec"]
    context = sum_vec / np.linalg.norm(sum_vec)
    
    factors = [1.0, 0.95, 0.9]
    results = {}

    for f in factors:
        router = BanditRouter.load_from_benchmark(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=1.0,
            prior_strength=500.0, # Strong prior to show the effect
            priors_meta_path=priors_meta_path,
            forgetting_factor=f
        )
        
        # Track probability over time
        history = []
        for i in range(50): # Fewer steps to show speed
            router.update(target_model_id, context, target_reward)
            router.update(teacher_pet_id, context, teacher_pet_reward)
            probs = router.get_probabilities(context, [target_model_id, teacher_pet_id])
            history.append(probs[target_model_id])
        
        results[f] = history

    print("Discovery Speed (Probability of DeepSeek R1 being the best):")
    print(f"{'Step':<6} | {'f=1.0':<10} | {'f=0.95':<10} | {'f=0.9':<10}")
    print("-" * 45)
    for i in [0, 5, 10, 20, 49]:
        print(f"{i:<6} | {results[1.0][i]:<10.4f} | {results[0.95][i]:<10.4f} | {results[0.9][i]:<10.4f}")

if __name__ == "__main__":
    compare_forgetting()
