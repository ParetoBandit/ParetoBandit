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

def simulate_niche_discovery():
    root_dir = Path(__file__).parent
    
    # 1. Load Models
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # 2. Initialize Router with HLE Priors (The "Teacher")
    # We use a lower prior_strength so it can actually learn within a reasonable number of steps
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    prior_strength = 100.0
    router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=1.0,
        prior_strength=prior_strength, 
        priors_meta_path=priors_meta_path
    )
    
    # 3. Define the "Niche"
    # We'll pick a model that has low HLE priors but we'll simulate it being the BEST in a specific niche.
    # Let's pick 'deepseek/deepseek-r1' as our "Hidden Specialist".
    # In HLE it has 0.093, while Gemini 3 Pro has 0.372.
    # Niche: TimescaleDB IoT Specialist
    target_model_id = "deepseek/deepseek-r1"
    teacher_pet_id = "google/gemini-3-pro-preview"
    
    # Simulated rewards for this specific high-technical niche
    target_reward = 5.8  # DeepSeek R1 excels at complex technical reasoning
    teacher_pet_reward = 4.2  # Gemini is good but less specialized here
    other_reward = 2.0
    
    # 3. Simulation
    print(f"Simulating 150 requests in the 'TimescaleDB IoT' niche...")
    
    # Use the average context from the benchmark to ensure priors are aligned
    meta = np.load(priors_meta_path)
    sum_vec = meta["sum_vec"]
    context = sum_vec / np.linalg.norm(sum_vec)
    
    router.bandit.alpha = 2.0
    
    for i in range(150):
        router.bandit.update(target_model_id, context, target_reward)
        router.bandit.update(teacher_pet_id, context, teacher_pet_reward)
        if i % 5 == 0:
            for m_id in router.bandit.models:
                if m_id not in [target_model_id, teacher_pet_id]:
                    router.bandit.update(m_id, context, other_reward)

    # 4. Get Native Probabilities from the Router
    # This proves the data is "real" and coming directly from the router's Bayesian state.
    models_to_compare = [target_model_id, teacher_pet_id]
    
    # Get Priors (before simulation)
    router_initial = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=1.0,
        prior_strength=prior_strength, 
        priors_meta_path=priors_meta_path
    )
    prior_probs = router_initial.get_probabilities(context, models_to_compare)

    # Get Posteriors (after simulation)
    post_probs = router.get_probabilities(context, models_to_compare)
    
    print(f"Prior Probs: {prior_probs}")
    print(f"Post Probs: {post_probs}")
         
    # 6. Save Results
    results = {
        "target_model": {
            "id": target_model_id,
            "name": "DeepSeek R1",
            "prior": float(prior_probs[target_model_id]),
            "posterior": float(post_probs[target_model_id])
        },
        "teacher_pet": {
            "id": teacher_pet_id,
            "name": "Gemini 3 Pro Preview (high)",
            "prior": float(prior_probs[teacher_pet_id]),
            "posterior": float(post_probs[teacher_pet_id])
        },
        "niche": "TimescaleDB IoT Specialist"
    }
    
    with open(root_dir / "discovery_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Simulation complete. Results saved to discovery_results.json")

if __name__ == "__main__":
    simulate_niche_discovery()
