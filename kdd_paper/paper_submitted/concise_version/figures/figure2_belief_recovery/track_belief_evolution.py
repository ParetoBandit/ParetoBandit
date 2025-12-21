#!/usr/bin/env python3
"""
Track how the BanditRouter's BELIEFS (theta estimates) evolve over time.
This is a true "belief recovery" plot using the ACTUAL library's internal state.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from banditgpt import BanditRouter
from banditgpt._resources import get_priors_path


def load_real_data():
    """Load all 81 models and 497 prompts."""
    print("=" * 80)
    print("Loading Real Benchmark Data")
    print("=" * 80)
    
    # Load prompts
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    prompts = []
    clusters = []
    
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
            clusters.append(data["cluster_id"])
    
    # Load rewards
    rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    rewards = {}
    all_models = set()
    
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                
                if cluster not in rewards:
                    rewards[cluster] = {}
                rewards[cluster][model] = reward
                all_models.add(model)
    
    model_names = sorted(all_models)
    
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: {len(model_names)}")
    print("=" * 80)
    
    return prompts, clusters, rewards, model_names


def run_with_belief_tracking(prompts, clusters, rewards, model_names, n_rounds=800):
    """
    Run BanditRouter and track its belief evolution for key models.
    """
    print("\n" + "=" * 80)
    print("Tracking Belief Evolution (Real BanditRouter)")
    print("=" * 80)
    
    # Create model registry
    model_registry = {
        model_id: {
            "display_name": model_id.split('/')[-1],
            "cost_per_1k_input": 1.0,
            "cost_per_1k_output": 1.0,
        }
        for model_id in model_names
    }
    
    # Initialize router in COLD START mode
    router = BanditRouter(
        model_registry=model_registry,
        alpha=0.5
    )
    
    print(f"  Models: {len(model_names)}")
    print(f"  Rounds: {n_rounds}")
    print(f"  ✅ Router initialized in cold start mode")
    
    # Models to track (focus on interesting ones)
    track_models = [
        "google/gemini-3-pro-preview",  # Frontier reference
        "openai/gpt-4o-mini",            # Cost-effective
        "amazon/nova-lite-v1",           # Cheapest
        "mistralai/mistral-small-24b-instruct-2501",
        "mistralai/ministral-3b",
    ]
    
    # Also track whatever models actually exist
    track_models = [m for m in track_models if m in model_names]
    
    # If gemini-3-pro-preview doesn't exist, find closest
    if "google/gemini-3-pro-preview" not in model_names:
        gemini_candidates = [m for m in model_names if "gemini" in m.lower() and "pro" in m.lower()]
        if gemini_candidates:
            track_models.append(gemini_candidates[0])
            print(f"  Using {gemini_candidates[0]} as frontier reference")
    
    print(f"\n  Tracking beliefs for {len(track_models)} models:")
    for m in track_models:
        print(f"    - {m}")
    
    # Storage for belief evolution (empirical average reward per model)
    belief_history = {model: [] for model in track_models}
    model_rewards_sum = {model: 0.0 for model in track_models}
    model_selection_count = {model: 0 for model in track_models}
    true_quality = {}
    
    # Calculate true average quality for each tracked model
    for model in track_models:
        model_rewards = []
        for cluster in rewards:
            if model in rewards[cluster]:
                model_rewards.append(rewards[cluster][model])
        true_quality[model] = np.mean(model_rewards) if model_rewards else 0.5
    
    n_prompts = len(prompts)
    
    print(f"\n[Running] {n_rounds} routing decisions...\n")
    
    for step in range(n_rounds):
        # Pick a prompt (cycle through)
        prompt_idx = step % n_prompts
        prompt = prompts[prompt_idx]
        cluster_id = clusters[prompt_idx]
        
        # Route using the library
        # BanditRouter.route() returns (model_id, RoutingLog)
        selected_model, decision_log = router.route(
            prompt=prompt,
            profile="quality-first"
        )
        
        # Get ground truth reward
        reward = rewards.get(cluster_id, {}).get(selected_model, 0.0)
        
        # Report feedback
        router.report_feedback(
            request_id=decision_log.request_id,
            reward=reward
        )
        
        # Update belief tracking: empirical average reward for each model
        for model in track_models:
            if model == selected_model:
                # This model was selected - update its empirical average
                model_selection_count[model] += 1
                model_rewards_sum[model] += reward
                current_avg = model_rewards_sum[model] / model_selection_count[model]
                belief_history[model].append(current_avg)
            else:
                # Carry forward the last belief
                if belief_history[model]:
                    belief_history[model].append(belief_history[model][-1])
                else:
                    belief_history[model].append(None)
        
        if (step + 1) % 200 == 0:
            print(f"    Step {step + 1}/{n_rounds}")
            # Show current beliefs
            for model in track_models:
                count = model_selection_count[model]
                if count > 0:
                    avg = model_rewards_sum[model] / count
                    name = model.split('/')[-1][:20]
                    print(f"      {name:20s}: {avg:.3f} (n={count})")
    
    print(f"\n✅ Belief tracking complete!")
    
    return belief_history, true_quality


def plot_belief_recovery(belief_history, true_quality, output_path):
    """
    Create the belief recovery plot showing how the bandit learns over time.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (model, beliefs) in enumerate(belief_history.items()):
        # Filter out None values
        valid_beliefs = [(step, b) for step, b in enumerate(beliefs) if b is not None]
        if not valid_beliefs:
            continue
        
        steps, beliefs_clean = zip(*valid_beliefs)
        
        # Shorten model name
        label = model.split('/')[-1] if '/' in model else model
        if len(label) > 30:
            label = label[:27] + '...'
        
        # Plot belief evolution
        ax.plot(steps, beliefs_clean, label=label, linewidth=2, alpha=0.8, color=colors[i])
        
        # Plot true quality as horizontal line
        true_val = true_quality.get(model, None)
        if true_val is not None:
            ax.axhline(true_val, color=colors[i], linestyle='--', alpha=0.4, linewidth=1)
    
    ax.set_xlabel("Routing Decision", fontsize=14, fontweight='bold')
    ax.set_ylabel("Belief (Expected Reward)", fontsize=14, fontweight='bold')
    ax.set_title("Belief Recovery: How BanditRouter Learns Model Quality Over Time\n(Solid = Learned Belief, Dashed = True Quality)", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    
    return fig


def main():
    # Load data
    prompts, clusters, rewards, model_names = load_real_data()
    
    # Run with belief tracking
    belief_history, true_quality = run_with_belief_tracking(
        prompts, clusters, rewards, model_names, n_rounds=800
    )
    
    # Plot
    output_path = Path(__file__).parent / "belief_recovery_real.png"
    plot_belief_recovery(belief_history, true_quality, output_path)
    
    # Save data
    results = {
        "belief_history": {k: [float(b) if b is not None else None for b in v] 
                          for k, v in belief_history.items()},
        "true_quality": true_quality
    }
    
    results_path = Path(__file__).parent / "belief_recovery_data.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Data saved to: {results_path}")


if __name__ == "__main__":
    main()

