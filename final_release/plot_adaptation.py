import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bandit import BanditRouter

def run_adaptation_simulation():
    print("Running Adaptation Simulation (Dip and Recover)...")
    
    # 1. Setup
    base_dir = Path(__file__).parent
    with open(base_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Models for simulation
    default_model = "openai/gpt-4o"
    specialist_model = "amazon/nova-lite-v1"
    others = [
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3-70b-instruct"
    ]
    model_subset = [default_model, specialist_model] + others
    
    # 2. Initialize Router with HLE Priors
    # We'll use a high prior strength to show the "initial bias"
    router = BanditRouter.create(
        model_registry=registry,
        priors="benchmark",
        prior_strength=50.0,
        exploration="balanced"
    )
    
    # Force specialist to start COLD (high uncertainty) to simulate a "hidden expert"
    # In reality, Nova-Lite has an HLE score, but we want to show discovery.
    dim = router.bandit.dim
    router.bandit.A[specialist_model] = np.eye(dim) * 1.0
    router.bandit.b[specialist_model] = np.zeros(dim)
    router.bandit.A_inv[specialist_model] = np.linalg.inv(router.bandit.A[specialist_model])
    
    # 3. Ground Truth Rewards for "Niche Task"
    ground_truth = {
        specialist_model: 0.95,   # The hidden expert
        default_model: 0.55,      # The generalist (struggles here)
        "anthropic/claude-3.5-sonnet": 0.45,
        "google/gemini-2.0-flash-001": 0.40,
        "meta-llama/llama-3-70b-instruct": 0.42,
    }
    
    # 4. Simulation Loop
    n_steps = 200
    niche_direction = np.random.randn(dim)
    niche_direction /= np.linalg.norm(niche_direction)
    
    specialist_selections = 0
    default_selections = 0
    
    specialist_rate = []
    default_rate = []
    avg_rewards = []
    total_reward = 0.0
    
    for t in range(n_steps):
        # Context with noise
        x = niche_direction + np.random.randn(dim) * 0.1
        x /= np.linalg.norm(x)
        
        # Select (using the subset)
        chosen, _ = router.bandit.select_arm(x, candidates=model_subset)
        
        # Reward
        base_r = ground_truth.get(chosen, 0.4)
        reward = np.clip(base_r + np.random.randn() * 0.05, 0, 1)
        
        # Update
        router.bandit.update(chosen, x, reward)
        
        # Track
        if chosen == specialist_model: specialist_selections += 1
        if chosen == default_model: default_selections += 1
        
        total_reward += reward
        specialist_rate.append(specialist_selections / (t + 1))
        default_rate.append(default_selections / (t + 1))
        avg_rewards.append(total_reward / (t + 1))
        
    # 5. Plotting
    print("Generating Plot...")
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    
    steps = range(1, n_steps + 1)
    
    # Left: Selection Rates
    ax1.plot(steps, specialist_rate, color="#2CA02C", lw=2, label="Specialist (Nova-Lite)")
    ax1.plot(steps, default_rate, color="#D62728", lw=2, label="Default (GPT-4o)")
    ax1.set_xlabel("Requests")
    ax1.set_ylabel("Selection Rate")
    ax1.set_title("Model Selection: Discovery", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    
    # Right: Reward (Dip and Recover)
    ax2.plot(steps, avg_rewards, color="#1F77B4", lw=2, label="Adaptive Agent")
    ax2.axhline(y=0.95, color="#2CA02C", ls="--", alpha=0.5, label="Optimal (0.95)")
    ax2.axhline(y=0.55, color="#D62728", ls="--", alpha=0.5, label="Default Only (0.55)")
    ax2.set_xlabel("Requests")
    ax2.set_ylabel("Average Reward")
    ax2.set_title("Reward: Dip and Recover", fontweight="bold")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.3, 1.0)
    
    plt.tight_layout()
    output_path = base_dir / "figure2_adaptation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    
if __name__ == "__main__":
    run_adaptation_simulation()
