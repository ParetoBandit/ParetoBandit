import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bandit import BanditRouter

def run_real_adaptation_simulation():
    """
    Simulates a domain shift from 'General Coding' to a 'Specialized Task' 
    where the HLE-favored model (Gemini) fails and a specialist (Claude) excels.
    """
    print("Running Real-Data Adaptation Simulation (Coding -> Specialist)...")
    
    # 1. Setup
    base_dir = Path(__file__).parent
    with open(base_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load pre-embedded real data (Coding and Specialist domains)
    with open(base_dir / "data/adaptation_sim_data.json") as f:
        sim_data = json.load(f)
    
    # Selected models for the plot (Focusing on the main adaptation story)
    selected_models = [
        'google/gemini-3-pro-preview',
        'anthropic/claude-3.7-sonnet:thinking'
    ]
    
    # 2. Initialize Router with HLE Priors
    # SESSION-LEVEL ADAPTATION: Very low prior_strength and aggressive forgetting
    router = BanditRouter.create(
        model_registry=registry,
        priors="benchmark",
        prior_strength=2.0, # Low trust in priors for specialized tasks
        exploration="aggressive", # High alpha for faster discovery
        forgetting_factor=0.75 # Aggressive forgetting
    )
    # router.bandit.alpha is set by exploration="aggressive" (~3.0)
    
    # 3. Simulation Loop
    n_phase1 = 50
    n_phase2 = 450 # Increased to show full Claude recovery
    n_total = n_phase1 + n_phase2
    
    # Metrics
    selection_counts = {m: 0 for m in selected_models}
    selection_rates = {m: [] for m in selected_models}
    per_step_rewards = []
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    print(f"Phase 1: General Coding ({n_phase1} steps)")
    for t in range(n_total):
        if t < n_phase1:
            # Domain: Coding (Gemini is strong)
            item = sim_data['coding'][t % len(sim_data['coding'])]
        else:
            if t == n_phase1:
                print(f"Phase 2: Specialized Task ({n_phase2} steps) - DISTRIBUTION SHIFT!")
            # Domain: Specialized (Claude is strong, Gemini is weak)
            item = sim_data['specialized'][(t - n_phase1) % len(sim_data['specialized'])]
        
        x = np.array(item['embedding'])
        
        # Select model
        chosen, _ = router.bandit.select_arm(x, candidates=selected_models)
        
        # Get real reward from data
        logit = item['rewards'].get(chosen, -5.0)
        reward = sigmoid(logit)
        
        # Update Bandit
        router.bandit.update(chosen, x, reward)
        
        # Track metrics
        if chosen in selection_counts:
            selection_counts[chosen] += 1
        
        for m in selected_models:
            selection_rates[m].append(selection_counts[m] / (t + 1))
            
        per_step_rewards.append(reward)
        
    # 4. Plotting
    print("Generating Figure 2...")
    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    
    steps = range(1, n_total + 1)
    
    # Left: Selection Rates
    colors = {
        'google/gemini-3-pro-preview': '#1F77B4', # Blue
        'anthropic/claude-3.7-sonnet:thinking': '#2CA02C', # Green
        'openai/gpt-4o': '#D62728', # Red
        'meta-llama/llama-3-70b-instruct': '#9467BD', # Purple
        'amazon/nova-lite-v1': '#FF7F0E' # Orange
    }
    
    for m in selected_models:
        label = m.split('/')[-1].replace(':thinking', '')
        ax1.plot(steps, selection_rates[m], color=colors[m], lw=2, label=label)
    
    ax1.axvline(x=n_phase1, color='black', ls='--', alpha=0.5)
    ax1.text(n_phase1/2, 1.15, "Coding", ha='center', fontweight='bold')
    ax1.text(n_phase1 + n_phase2/2, 1.15, "Specialist", ha='center', fontweight='bold')
    
    ax1.set_xlabel("Requests")
    ax1.set_ylabel("Selection Rate")
    ax1.set_title("Model Selection: Adaptation", fontweight="bold")
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_ylim(0, 1.3)
    
    # Right: Rolling Reward (Dip and Recover)
    window = 30
    rolling_reward = [np.mean(per_step_rewards[max(0, i-window):i+1]) for i in range(n_total)]
    
    ax2.plot(steps, rolling_reward, color="#333333", lw=2, label="Bandit Router")
    ax2.axvline(x=n_phase1, color='black', ls='--', alpha=0.5)
    
    # Reference lines for optimal rewards
    coding_opt = sigmoid(3.68)
    specialized_opt = sigmoid(9.21) # Claude's reward in Cluster 13
    ax2.hlines(y=coding_opt, xmin=0, xmax=n_phase1, color='#2CA02C', ls=':', alpha=0.6, label="Opt (Coding)")
    ax2.hlines(y=specialized_opt, xmin=n_phase1, xmax=n_total, color='#2CA02C', ls=':', alpha=0.6, label="Opt (Specialist)")
    
    ax2.set_xlabel("Requests")
    ax2.set_ylabel(f"Rolling Reward (w={window})")
    ax2.set_title("Reward: Dip and Recover", fontweight="bold")
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = base_dir / "figure2_adaptation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    
if __name__ == "__main__":
    run_real_adaptation_simulation()
