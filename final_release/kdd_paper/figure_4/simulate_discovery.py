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
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # 3. Initialize Router with Defaults (HLE)
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        # Use system defaults (alpha=1.0, prior_strength=40.0)
        priors_meta_path=priors_meta_path
    )
    
    # 3. Define the "Niche"
    # Let's pick 'deepseek/deepseek-r1' as our "Specialist".
    # In HLE it has 0.093, while Gemini 3 Pro has 0.372.
    target_model = "deepseek/deepseek-r1"
    teacher_pet = "google/gemini-3-pro-preview"
    
    print(f"Target Model: {target_model}")
    print(f"Teacher's Pet: {teacher_pet}")
    
    # Capture Priors
    priors = {}
    for m_id in [target_model, teacher_pet]:
        theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
        priors[m_id] = np.linalg.norm(theta)
    
    # 4. Run Simulation in the Niche
    # We simulate 100 requests where the target_model always gets reward 1.0 and others get 0.2
    n_requests = 150
    context = np.random.randn(384) # Fixed context for simplicity in this niche
    context /= np.linalg.norm(context)
    
    print(f"Simulating {n_requests} requests in the '{target_model}' niche...")
    
    for i in range(n_requests):
        # In this niche, the target_model is the clear winner
        # We teach the bandit about all models in this specific context
        context_bias = np.append(context, 1.0)
        for m_id in router.bandit.models:
            reward = 1.0 if m_id == target_model else 0.2
            # Add some noise
            reward += np.random.normal(0, 0.05)
            router.bandit.update(m_id, context_bias, reward)
            
    # 5. Capture Posteriors
    posteriors = {}
    for m_id in [target_model, teacher_pet]:
        theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
        posteriors[m_id] = np.linalg.norm(theta)
        
    # 6. Save Results
    results = {
        "target_model": {
            "id": target_model,
            "name": registry[target_model]["name"],
            "prior": priors[target_model],
            "posterior": posteriors[target_model]
        },
        "teacher_pet": {
            "id": teacher_pet,
            "name": registry[teacher_pet]["name"],
            "prior": priors[teacher_pet],
            "posterior": posteriors[teacher_pet]
        }
    }
    
    with open(root_dir / "discovery_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Simulation complete. Results saved to discovery_results.json")

if __name__ == "__main__":
    simulate_niche_discovery()
