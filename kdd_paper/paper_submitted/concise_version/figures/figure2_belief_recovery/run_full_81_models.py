#!/usr/bin/env python3
"""
Use the ACTUAL BanditGPT library (BanditRouter) to route across all 81 models.
Show which models it naturally selects in COLD START mode.
"""
import json
import numpy as np
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Import the ACTUAL library
from banditgpt import BanditRouter
from banditgpt._resources import get_priors_path


def load_real_data():
    """Load all 81 models and 497 prompts."""
    print("=" * 80)
    print("Loading Real Benchmark Data (All 81 Models)")
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


def run_library_routing(prompts, clusters, rewards, model_names, n_rounds=800):
    """
    Use the ACTUAL BanditRouter from the library.
    Cold start mode (no priors loaded).
    """
    print("\n" + "=" * 80)
    print("Running BanditGPT Library (BanditRouter)")
    print("=" * 80)
    
    print(f"\n[Configuration]")
    print(f"  Router: BanditRouter (from library)")
    print(f"  Mode: COLD START (no priors)")
    print(f"  Models: {len(model_names)}")
    print(f"  Rounds: {n_rounds}")
    
    # Create minimal model registry
    model_registry = {
        model_id: {
            "display_name": model_id.split('/')[-1],
            "cost_per_1k_input": 1.0,  # Default cost
            "cost_per_1k_output": 1.0,
        }
        for model_id in model_names
    }
    
    # Initialize the ACTUAL library router in COLD START mode
    router = BanditRouter(
        model_registry=model_registry,
        alpha=0.5  # Exploration parameter
    )
    
    print(f"  ✅ Router initialized in cold start mode (no priors)")
    
    # Track routing decisions
    selections = []
    round_rewards = []
    
    n_prompts = len(prompts)
    
    print(f"\n[Simulation] Running {n_rounds} routing decisions...")
    
    for t in range(n_rounds):
        idx = t % n_prompts
        prompt = prompts[idx]
        cluster = clusters[idx]
        
        # Use library's route method
        selected_model, log = router.route(prompt, profile="quality-first")
        
        # Get real reward from benchmark
        true_reward = rewards.get(cluster, {}).get(selected_model, 0.5)
        
        # Update router with feedback
        router.report_feedback(
            request_id=log.request_id,
            reward=true_reward
        )
        
        selections.append(selected_model)
        round_rewards.append(true_reward)
        
        if (t + 1) % 100 == 0:
            recent_reward = np.mean(round_rewards[-100:])
            recent_counts = Counter(selections[-100:])
            top_3 = recent_counts.most_common(3)
            print(f"    Step {t+1}: reward={recent_reward:.3f}, "
                  f"top 3={[(m.split('/')[-1], c) for m, c in top_3]}")
    
    return selections, round_rewards


def analyze_selections(selections, model_names):
    """Analyze which models were selected."""
    print("\n" + "=" * 80)
    print("ROUTING ANALYSIS")
    print("=" * 80)
    
    counts = Counter(selections)
    total = len(selections)
    
    print(f"\nTotal routing decisions: {total}")
    print(f"Unique models selected: {len(counts)}")
    print(f"Models never selected: {len(model_names) - len(counts)}")
    
    # Top models
    print("\n" + "-" * 80)
    print("TOP 20 MODELS SELECTED:")
    print("-" * 80)
    
    for i, (model, count) in enumerate(counts.most_common(20), 1):
        pct = count / total * 100
        model_short = model.split('/')[-1]
        print(f"{i:2d}. {model_short:40s} {count:4d} ({pct:5.1f}%)")
    
    # Bottom (never or rarely selected)
    never_selected = [m for m in model_names if counts.get(m, 0) == 0]
    
    if never_selected:
        print(f"\n{len(never_selected)} models NEVER selected:")
        for m in never_selected[:10]:  # Show first 10
            print(f"  - {m}")
        if len(never_selected) > 10:
            print(f"  ... and {len(never_selected) - 10} more")
    
    return counts


def save_results(selections, rewards, counts, output_dir):
    """Save detailed results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    summary = {
        'total_rounds': len(selections),
        'unique_models_selected': len(counts),
        'mean_reward': float(np.mean(rewards)),
        'top_20_models': [
            {
                'model': model,
                'count': count,
                'percentage': count / len(selections) * 100
            }
            for model, count in counts.most_common(20)
        ]
    }
    
    summary_path = output_dir / "full_81_models_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Saved summary to {summary_path}")
    
    # Save full log
    log_path = output_dir / "full_81_models_selections.json"
    with open(log_path, 'w') as f:
        json.dump({
            'selections': selections,
            'rewards': [float(r) for r in rewards]
        }, f, indent=2)
    
    print(f"✅ Saved full log to {log_path}")


def main():
    # Load data
    prompts, clusters, rewards, model_names = load_real_data()
    
    # Run library routing
    selections, round_rewards = run_library_routing(
        prompts, clusters, rewards, model_names, n_rounds=800
    )
    
    # Analyze
    counts = analyze_selections(selections, model_names)
    
    # Save
    save_results(selections, round_rewards, counts, Path(__file__).parent)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
