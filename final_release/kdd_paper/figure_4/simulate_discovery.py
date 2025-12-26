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
    
    # 3. Initialize Router with Library Defaults
    # Defaults: prior_strength=40.0, exploration='safe' (α=0.1), forgetting_factor=0.95
    router = BanditRouter.create(
        model_registry=registry
        # All other parameters use library defaults
    )
    
    # 3. Define the "Niche"
    # Let's pick 'deepseek/deepseek-r1-distill-llama-70b' as our "Specialist".
    # In HLE it has low score, while Gemini 3 Pro has high score.
    target_model = "deepseek/deepseek-r1-distill-llama-70b"
    teacher_pet = "google/gemini-3-pro-preview"
    
    print(f"Target Model: {target_model}")
    print(f"Teacher's Pet: {teacher_pet}")
    
    # Capture Priors
    priors = {}
    for m_id in [target_model, teacher_pet]:
        theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
        priors[m_id] = np.linalg.norm(theta)
    
    # 4. Load Real Test Prompts (No Data Leakage)
    # Use test_prompts.jsonl to ensure no overlap with prior training
    test_prompts = []
    prompts_path = root_dir / "data" / "test_prompts.jsonl"
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                test_prompts.append(json.loads(line))
    
    # Sample 500 prompts for the simulation
    np.random.seed(42)
    n_requests = 500
    if len(test_prompts) >= n_requests:
        selected_prompts = np.random.choice(test_prompts, n_requests, replace=False)
    else:
        selected_prompts = test_prompts
        n_requests = len(selected_prompts)
    
    print(f"Simulating {n_requests} requests in the '{target_model}' niche using real test prompts...")
    
    for i, prompt_data in enumerate(selected_prompts):
        prompt_text = prompt_data["prompt"]
        # Get context vector from the router's encoder
        context_vec = router._get_context_vector(prompt_text)
        
        # In this niche, the target_model is the clear specialist
        # Simulate rewards where DeepSeek R1 excels
        for m_id in router.bandit.models:
            if m_id == target_model:
                reward = np.random.uniform(0.9, 1.0)  # High performance
            else:
                reward = np.random.uniform(0.1, 0.3)  # Lower performance
            
            router.bandit.update(m_id, context_vec, reward)
            
    # 5. Capture Posteriors
    posteriors = {}
    for m_id in [target_model, teacher_pet]:
        theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
        posteriors[m_id] = np.linalg.norm(theta)
        
    # 6. Save Results
    results = {
        "target_model": {
            "id": target_model,
            "name": registry[target_model]["display_name"],
            "prior": priors[target_model],
            "posterior": posteriors[target_model]
        },
        "teacher_pet": {
            "id": teacher_pet,
            "name": registry[teacher_pet]["display_name"],
            "prior": priors[teacher_pet],
            "posterior": posteriors[teacher_pet]
        }
    }
    
    output_path = root_dir / "kdd_paper" / "figure_4" / "discovery_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Simulation complete. Results saved to discovery_results.json")

if __name__ == "__main__":
    simulate_niche_discovery()
