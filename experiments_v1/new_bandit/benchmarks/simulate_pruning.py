#!/usr/bin/env python3
"""
Two-Strike Pruning Simulation with Real Data

Uses real training prompts for burn-in and test prompts for evaluation
to demonstrate which models get pruned vs protected.

Output:
- Models that survive (pass theory OR reality check)
- Models that get pruned (fail both strikes)
- Unicorn saves (fail theory but pass reality check)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from router import BanditRouter, RouterConfig
    BANDIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import BanditRouter: {e}")
    BANDIT_AVAILABLE = False
    sys.exit(1)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rewards_with_prompts(path: Path) -> tuple:
    """
    Load rewards file and extract unique prompts.
    Returns: (rewards_dict, prompts_list)
    - rewards_dict: {prompt_hash: {model_id: reward}}
    - prompts_list: list of unique prompts
    """
    rewards = defaultdict(dict)
    prompts = {}  # {hash: prompt} to deduplicate
    
    if not path.exists():
        print(f"⚠️  Rewards file not found: {path}")
        return rewards, []
    
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                prompt = row.get("prompt", "")
                model = row.get("model_id", row.get("model", ""))
                reward = row.get("raw_score", row.get("reward", 0.0))
                if prompt and model:
                    prompt_hash = hash(prompt[:200])
                    rewards[prompt_hash][model] = reward
                    prompts[prompt_hash] = prompt  # Store unique prompts
            except:
                pass
    
    return rewards, list(prompts.values())


def load_prompts(path: Path, limit: int = None) -> list:
    """Load prompts from JSONL file."""
    prompts = []
    if not path.exists():
        print(f"⚠️  Prompts file not found: {path}")
        return prompts
    
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                row = json.loads(line)
                prompt = row.get("prompt", "")
                if prompt:
                    prompts.append(prompt)
            except:
                pass
    return prompts


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation():
    """Run the Two-Strike pruning simulation with real data."""
    
    print("\n" + "="*70)
    print("TWO-STRIKE PRUNING SIMULATION (Real Data)")
    print("="*70)
    
    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "offline_dataset"
    
    # Load data - prompts come directly from rewards files (ensures all prompts have rewards)
    print("\n📂 Loading data...")
    train_rewards, train_prompts = load_rewards_with_prompts(data_dir / "train_rewards_1k.jsonl")
    test_rewards, test_prompts = load_rewards_with_prompts(data_dir / "test_rewards_pareto_dedup.jsonl")
    
    print(f"   Train: {len(train_prompts)} unique prompts with {sum(len(v) for v in train_rewards.values())} model-reward pairs")
    print(f"   Test: {len(test_prompts)} unique prompts with {sum(len(v) for v in test_rewards.values())} model-reward pairs")
    
    # Create router with HLE priors
    print("\n🔧 Initializing BanditRouter...")
    
    # Find models.json (go up from benchmarks -> new_bandit -> experiments -> banditgpt)
    models_path = Path(__file__).parent.parent.parent.parent / "models.json"
    if not models_path.exists():
        print(f"⚠️  models.json not found at {models_path}")
        sys.exit(1)
    
    # Load models as dict
    with open(models_path) as f:
        models_data = json.load(f)
    model_registry = {m["openrouter_id"]: m for m in models_data.get("models", [])[:30]}  # Top 30 models
    
    router = BanditRouter.create(
        model_registry=model_registry,
        priors="hle",  # Use HLE priors
        context_encoder=None  # Will initialize encoder
    )
    
    print(f"   Models registered: {len(router.registry)}")
    print(f"   Context dimension: {router.bandit.dim}")
    
    # Get models that have rewards in our dataset
    models_with_rewards = set()
    for prompt_hash, model_rewards in train_rewards.items():
        models_with_rewards.update(model_rewards.keys())
    for prompt_hash, model_rewards in test_rewards.items():
        models_with_rewards.update(model_rewards.keys())
    
    # Filter to models in router
    active_models = [m for m in router.bandit.models if m in models_with_rewards]
    print(f"   Models with reward data: {len(active_models)}")
    
    if len(active_models) < 5:
        print("⚠️  Not enough models with reward data for meaningful simulation")
        print("   Falling back to simulated rewards based on HLE scores")
        active_models = list(router.bandit.models)[:20]
    
    # ==========================================================================
    # PHASE 1: BURN-IN (Training prompts)
    # ==========================================================================
    print("\n🔥 PHASE 1: Burn-in (Training Prompts)")
    print(f"   Running {len(train_prompts)} training prompts...")
    
    model_selections = defaultdict(list)
    
    for i, prompt in enumerate(train_prompts):
        # Route the prompt
        model, log = router.route(prompt, profile="best_value")
        
        # Get reward (from real data or synthetic)
        prompt_hash = hash(prompt[:200])
        if prompt_hash in train_rewards and model in train_rewards[prompt_hash]:
            reward = train_rewards[prompt_hash][model]
        else:
            # Fallback: use HLE-based synthetic reward
            hle = router.registry.get(model, {}).get("hle", 0.5)
            reward = np.random.normal(0.5 + hle * 0.3, 0.15)
            reward = np.clip(reward, 0, 1)
        
        # Update bandit
        router.process_feedback(log.request_id, reward)
        model_selections[model].append(reward)
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(train_prompts)} prompts...")
    
    # Print burn-in stats
    print("\n   Burn-in Statistics:")
    for model in sorted(model_selections.keys(), key=lambda m: len(model_selections[m]), reverse=True)[:10]:
        rewards = model_selections[model]
        print(f"   {model[:35]:35} | Selections: {len(rewards):4} | Mean: {np.mean(rewards):.3f}")
    
    # ==========================================================================
    # PHASE 2: PRUNING CHECK
    # ==========================================================================
    print("\n✂️  PHASE 2: Two-Strike Pruning Check")
    
    # Check sample counts (Min-Sample Probation)
    print("\n   Sample Counts (Min-Sample Probation):")
    min_samples = RouterConfig.pruning_min_samples
    eligible_for_pruning = []
    protected_by_burnin = []
    
    for model in router.bandit.models:
        count = len(model_selections[model])
        if count >= min_samples:
            eligible_for_pruning.append(model)
        else:
            protected_by_burnin.append(model)
    
    print(f"   Threshold: {min_samples} samples")
    print(f"   Eligible for pruning: {len(eligible_for_pruning)}")
    print(f"   Protected (burn-in): {len(protected_by_burnin)}")
    
    # Run pruning - now returns dict with 'pruned' and 'unicorn_saves'
    print("\n   Running prune_arms()...")
    prune_result = router.prune_arms(confidence_alpha=2.0, niche_protection_threshold=0.80)
    pruned = prune_result["pruned"]
    unicorn_saves = prune_result["unicorn_saves"]
    
    # ==========================================================================
    # PHASE 3: TEST EVALUATION (Using test prompts for final metrics)
    # ==========================================================================
    print("\n" + "="*70)
    print("🧪 PHASE 3: Test Evaluation (Held-Out Data)")
    print("="*70)
    print(f"   Running {len(test_prompts)} test prompts...")
    
    test_selections = defaultdict(list)
    
    for i, prompt in enumerate(test_prompts):
        # Route the prompt (bandit selects model)
        model, log = router.route(prompt, profile="best_value")
        
        # Get reward from test data
        prompt_hash = hash(prompt[:200])
        if prompt_hash in test_rewards and model in test_rewards[prompt_hash]:
            reward = test_rewards[prompt_hash][model]
        else:
            # Fallback: use HLE-based synthetic reward
            hle = router.registry.get(model, {}).get("hle", 0.5)
            reward = np.random.normal(0.5 + hle * 0.3, 0.15)
            reward = np.clip(reward, 0, 1)
        
        # Update bandit (continued learning)
        router.process_feedback(log.request_id, reward)
        test_selections[model].append(reward)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(test_prompts)} test prompts...")
    
    print("\n   Test Phase Statistics:")
    for model in sorted(test_selections.keys(), key=lambda m: len(test_selections[m]), reverse=True)[:10]:
        rewards = test_selections[model]
        print(f"   {model[:35]:35} | Selections: {len(rewards):4} | Mean: {np.mean(rewards):.3f}")
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    print(f"\n🗑️  PRUNED ({len(pruned)} models):")
    for model in pruned:
        rewards = model_selections[model]
        mean_r = np.mean(rewards) if rewards else 0
        print(f"   ❌ {model[:40]:40} | Samples: {len(rewards):3} | Mean: {mean_r:.3f}")
    
    print(f"\n✅ SURVIVING ({len(router.bandit.models)} models):")
    surviving = sorted(
        router.bandit.models, 
        key=lambda m: len(model_selections[m]), 
        reverse=True
    )[:15]
    for model in surviving:
        rewards = model_selections[model]
        mean_r = np.mean(rewards) if rewards else 0
        print(f"   ✓ {model[:40]:40} | Samples: {len(rewards):3} | Mean: {mean_r:.3f}")
    
    print(f"\n🛡️  PROTECTED BY BURN-IN ({len(protected_by_burnin)} models):")
    for model in protected_by_burnin[:10]:
        rewards = model_selections[model]
        mean_r = np.mean(rewards) if rewards else 0
        print(f"   🔒 {model[:40]:40} | Samples: {len(rewards):3} | Mean: {mean_r:.3f}")
    if len(protected_by_burnin) > 10:
        print(f"   ... and {len(protected_by_burnin) - 10} more")
    
    # ==========================================================================
    # UNICORN GUARDRAIL RESULTS (FROM ACTUAL ROUTER CODE)
    # ==========================================================================
    print("\n" + "="*70)
    print("🦄 UNICORN GUARDRAIL RESULTS (Exact Definition from Router)")
    print("="*70)
    
    print(f"\n   Arms that failed Strike 1 (UCB domination): {prune_result['arms_evaluated']}")
    print(f"   Global mean reward: {prune_result['global_mean']:.3f}")
    
    if unicorn_saves:
        print(f"\n   🦄 UNICORN SAVES ({len(unicorn_saves)} models):")
        print("   (Failed theoretical domination but passed empirical reality check)")
        print()
        for save in unicorn_saves:
            threshold_val = save['global_mean'] * save['threshold']
            print(f"   🦄 {save['model'][:40]:40}")
            print(f"      Samples: {save['samples']} | Arm Mean: {save['arm_mean']:.3f} | Threshold: {threshold_val:.3f}")
            print(f"      Reason: {save['arm_mean']:.3f} >= {save['global_mean']:.3f} × {save['threshold']:.0%}")
            print()
    else:
        print("\n   No unicorn saves detected (no models failed Strike 1 with enough samples)")
    
    # ==========================================================================
    # SURPRISING FINDINGS
    # ==========================================================================
    print("\n" + "="*70)
    print("🔍 SURPRISING FINDINGS")
    print("="*70)
    
    # Find models with high selection but low reward (should be pruned)
    suspicious = []
    for model in router.bandit.models:
        rewards = model_selections[model]
        if len(rewards) >= 20:
            mean_r = np.mean(rewards)
            if mean_r < 0.4:
                suspicious.append((model, len(rewards), mean_r))
    
    if suspicious:
        print("\n⚠️  High Selection + Low Reward (potential pruning candidates):")
        for model, count, mean_r in sorted(suspicious, key=lambda x: x[1], reverse=True):
            print(f"   {model[:40]:40} | Selections: {count:3} | Mean: {mean_r:.3f}")
    
    print("\n" + "="*70)
    print("✅ Simulation Complete")
    print("="*70 + "\n")
    
    results = {
        "pruned": pruned,
        "surviving": list(router.bandit.models),
        "protected_by_burnin": protected_by_burnin,
        "model_selections": dict(model_selections),  # Training phase
        "test_selections": dict(test_selections),     # Test phase (for plots)
        "unicorn_saves": unicorn_saves,
        "prune_result": prune_result,
        "min_samples": min_samples
    }
    
    # Generate visualization
    plot_results(results)
    
    return results


def plot_results(results: dict):
    """Generate visualization of Two-Strike pruning results using TEST data."""
    
    # Use TEST selections for visualization (post burn-in performance)
    test_selections = results.get("test_selections", results["model_selections"])
    train_selections = results["model_selections"]  # For burn-in comparison
    min_samples = results["min_samples"]
    unicorn_saves = results["unicorn_saves"]
    prune_result = results["prune_result"]
    
    # Get models with TEST data, sorted by selection count
    models_with_data = [(m, len(r), np.mean(r) if r else 0) 
                        for m, r in test_selections.items() if len(r) > 0]
    models_with_data.sort(key=lambda x: x[1], reverse=True)
    
    if not models_with_data:
        print("No test data to plot")
        return
    
    # Setup figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ==========================================================================
    # Panel A: Sample Counts with Min-Sample Threshold
    # ==========================================================================
    ax1 = axes[0]
    
    top_n = min(15, len(models_with_data))
    models = [m[0].split('/')[-1][:18] for m in models_with_data[:top_n]]
    counts = [m[1] for m in models_with_data[:top_n]]
    rewards = [m[2] for m in models_with_data[:top_n]]
    
    colors = ['#27ae60' if c >= min_samples else '#e74c3c' for c in counts]
    
    y_pos = np.arange(len(models))
    bars = ax1.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add threshold line
    ax1.axvline(x=min_samples, color='#f39c12', linestyle='--', linewidth=2.5,
                label=f'Min-Sample Threshold = {min_samples}')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Sample Count (Requests Served)', fontsize=11, fontweight='bold')
    ax1.set_title(f'A. Min-Sample Probation\n{sum(c >= min_samples for c in counts)} eligible, {sum(c < min_samples for c in counts)} protected', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel B: Mean Reward by Model
    # ==========================================================================
    ax2 = axes[1]
    
    bar_colors = []
    unicorn_models = [u['model'] for u in unicorn_saves] if unicorn_saves else []
    
    for m, _, r in models_with_data[:top_n]:
        if m in unicorn_models:
            bar_colors.append('#f39c12')  # Unicorn
        elif m in results.get("pruned", []):
            bar_colors.append('#e74c3c')  # Pruned
        else:
            bar_colors.append('#3498db')  # Normal
    
    ax2.barh(y_pos, rewards, color=bar_colors, edgecolor='white', linewidth=1.5)
    
    # Add global mean line
    global_mean = prune_result.get("global_mean", 0.5)
    if global_mean > 0:
        ax2.axvline(x=global_mean * 0.8, color='#f39c12', linestyle='--', linewidth=2,
                    label=f'Unicorn Threshold (80% × {global_mean:.2f})')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('Mean Reward', fontsize=11, fontweight='bold')
    ax2.set_title('B. Empirical Performance\n(Unicorn Guardrail Check)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.1)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add annotation for unicorns
    if unicorn_saves:
        for i, (m, _, r) in enumerate(models_with_data[:top_n]):
            if m in unicorn_models:
                ax2.annotate('🦄', xy=(r + 0.02, i), fontsize=14)
    
    fig.suptitle('Two-Strike Pruning Simulation Results (Real Data)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "pruning_simulation_results.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Saved visualization to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    results = run_simulation()

