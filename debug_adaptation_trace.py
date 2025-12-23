
import json
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from final_release.bandit import BanditRouter

def debug_adaptation():
    print("DEBUGGING ADAPTATION UCBs...")
    
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
    
    router = BanditRouter.create(
        model_registry=registry,
        priors="benchmark",
        exploration="balanced", # alpha=1.0
        forgetting_factor=0.96
    )
    
    n_phase1 = 50
    n_phase2 = 100 # Look at first 100 steps of phase 2
    n_total = n_phase1 + n_phase2
    
    print(f"{'Step':<4} | {'Model':<20} | {'Mean':<6} | {'Uncert':<6} | {'UCB':<6} | {'Choice'}")
    print("-" * 80)
    
    for t in range(n_total):
        if t < n_phase1:
            item = sim_data['coding'][t % len(sim_data['coding'])]
            phase = "P1"
        else:
            item = sim_data['specialized'][(t - n_phase1) % len(sim_data['specialized'])]
            phase = "P2"
        
        x = np.array(item['embedding'])
        x_bias = np.append(x, 1.0)
        
        # Manually inspect UCBs for the two key models
        ucbs = {}
        means = {}
        uncerts = {}
        
        for m in selected_models:
            # Replicate DisjointLinUCBPolicy logic
            theta = router.bandit.A_inv[m] @ router.bandit.b[m]
            mean = float(theta.dot(x_bias))
            var = float(x_bias.dot(router.bandit.A_inv[m]).dot(x_bias))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + router.bandit.alpha * std
            
            ucbs[m] = ucb
            means[m] = mean
            uncerts[m] = std
            
        chosen, _ = router.bandit.select_arm(x_bias, candidates=selected_models)
        
        logit = item['rewards'].get(chosen, -5.0)
        reward = 1 / (1 + np.exp(-logit))
        
        router.bandit.update(chosen, x_bias, reward)
        
        # Print logs for transition period
        if t >= n_phase1 - 5 and t < n_phase1 + 20:
            gem = 'google/gemini-3-pro-preview'
            cla = 'anthropic/claude-3.7-sonnet:thinking'
            
            print(f"{t:<4} | {gem[:20]:<20} | {means[gem]:<6.2f} | {uncerts[gem]:<6.2f} | {ucbs[gem]:<6.2f} | {'*' if chosen==gem else ''}")
            print(f"{'':<4} | {cla[:20]:<20} | {means[cla]:<6.2f} | {uncerts[cla]:<6.2f} | {ucbs[cla]:<6.2f} | {'*' if chosen==cla else ''}")
            print("-" * 80)

if __name__ == "__main__":
    debug_adaptation()
