
import json
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from final_release.bandit import BanditRouter

def run_gamma_sweep():
    print("Sweeping Gamma for Alpha=1.0 Adaptation...")
    
    base_dir = Path(__file__).parent
    root_dir = base_dir / "final_release"
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    with open(root_dir / "data/adaptation_sim_data.json") as f:
        sim_data = json.load(f)
        
    selected_models = [
        'google/gemini-3-pro-preview',
        'anthropic/claude-3.7-sonnet:thinking'
    ]
    
    gammas = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 0.995]
    
    print(f"{'Gamma':<6} | {'Eff N':<6} | {'Recovery (Last 50 Avg)':<22}")
    print("-" * 40)
    
    for gamma in gammas:
        router = BanditRouter.create(
            model_registry=registry,
            priors="benchmark",
            prior_strength=20.0,
            exploration="balanced", # alpha=1.0
            forgetting_factor=gamma 
        )
        
        n_phase1 = 50
        n_phase2 = 450
        n_total = n_phase1 + n_phase2
        
        rewards = []
        
        for t in range(n_total):
            if t < n_phase1:
                item = sim_data['coding'][t % len(sim_data['coding'])]
            else:
                item = sim_data['specialized'][(t - n_phase1) % len(sim_data['specialized'])]
            
            x = np.array(item['embedding'])
            x_bias = np.append(x, 1.0)
            
            chosen, _ = router.bandit.select_arm(x_bias, candidates=selected_models)
            
            logit = item['rewards'].get(chosen, -5.0)
            reward = 1 / (1 + np.exp(-logit))
            
            router.bandit.update(chosen, x_bias, reward)
            rewards.append(reward)
            
        final_recovery = np.mean(rewards[-50:])
        eff_n = 1 / (1 - gamma)
        print(f"{gamma:<6.3f} | {int(eff_n):<6} | {final_recovery:<22.4f}")

if __name__ == "__main__":
    run_gamma_sweep()
