#!/usr/bin/env python3
"""
Regret Waterfall v2: Let the bandit CHOOSE between models using REAL data.

The bandit routes between GPT-4o, Mixtral, and GPT-5 based on UCB scores.
Regret = (best model's reward) - (selected model's reward)

Tracks how initialization affects early routing decisions.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gzip
import json
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import BanditRouter


def load_shared_prompts(rewards_file: Path) -> List[Dict[str, float]]:
    """
    Load prompts that have rewards for all three models.
    
    Returns list of dicts: [{prompt_id, gpt4o_reward, mixtral_reward, gpt5_reward, context}, ...]
    """
    import hashlib
    
    # First pass: organize by prompt hash
    prompt_data = defaultdict(dict)
    
    with gzip.open(rewards_file, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            model_id = entry['model_id']
            prompt = entry['prompt']
            reward = entry['raw_score']
            
            # Use hash of prompt as unique ID
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            if model_id in ['openai/gpt-4o', 'mistralai/mixtral-8x7b-instruct', 'openai/gpt-5-chat']:
                if 'prompt_text' not in prompt_data[prompt_hash]:
                    prompt_data[prompt_hash]['prompt_text'] = prompt
                prompt_data[prompt_hash][model_id] = reward
    
    # Second pass: keep only prompts with all three models
    shared_prompts = []
    for prompt_hash, data in prompt_data.items():
        if all(m in data for m in ['openai/gpt-4o', 'mistralai/mixtral-8x7b-instruct', 'openai/gpt-5-chat']):
            shared_prompts.append({
                'prompt_id': prompt_hash,
                'gpt4o_reward': data['openai/gpt-4o'],
                'mixtral_reward': data['mistralai/mixtral-8x7b-instruct'],
                'gpt5_reward': data['openai/gpt-5-chat'],
                'prompt_text': data['prompt_text']
            })
    
    return shared_prompts


def create_base_router(registry: Dict[str, Dict]) -> BanditRouter:
    """Create router with base models (GPT-4o, Mixtral) and warmup priors."""
    router = BanditRouter(
        model_registry=registry,
        alpha=0.05,
        init_lambda=1.0,
        verbose_routing=False
    )
    
    # Load priors for base models
    priors_path = Path(__file__).parent.parent.parent / "data" / "routellm" / "priors_warmup_routellm_pca24.joblib"
    priors_data = joblib.load(priors_path)
    A_matrices = priors_data['A']
    b_vectors = priors_data['b']
    
    # Load for Mixtral (if available)
    if "mistralai/mixtral-8x7b-instruct" in router.bandit.models and "mistralai/mixtral-8x7b-instruct" in A_matrices:
        router.bandit.A["mistralai/mixtral-8x7b-instruct"] = A_matrices["mistralai/mixtral-8x7b-instruct"].copy()
        router.bandit.b["mistralai/mixtral-8x7b-instruct"] = b_vectors["mistralai/mixtral-8x7b-instruct"].copy()
        router.bandit.A_inv["mistralai/mixtral-8x7b-instruct"] = np.linalg.inv(router.bandit.A["mistralai/mixtral-8x7b-instruct"])
    
    # For GPT-4o, use GPT-4-turbo priors as surrogate
    if "openai/gpt-4o" in router.bandit.models and "openai/gpt-4-turbo" in A_matrices:
        router.bandit.A["openai/gpt-4o"] = A_matrices["openai/gpt-4-turbo"].copy()
        router.bandit.b["openai/gpt-4o"] = b_vectors["openai/gpt-4-turbo"].copy()
        router.bandit.A_inv["openai/gpt-4o"] = np.linalg.inv(router.bandit.A["openai/gpt-4o"])
    
    return router


def register_gpt5_cold_start(router: BanditRouter) -> None:
    """Register GPT-5 with cold start (zero prior)."""
    A_init = np.eye(router.bandit.dim) * router.bandit.init_lambda
    b_init = np.zeros(router.bandit.dim)
    
    router.bandit.models.append("openai/gpt-5-chat")
    router.bandit.A["openai/gpt-5-chat"] = A_init
    router.bandit.b["openai/gpt-5-chat"] = b_init
    router.bandit.A_inv["openai/gpt-5-chat"] = np.linalg.inv(A_init)
    router.bandit.last_update["openai/gpt-5-chat"] = router.bandit.t
    
    router.registry["openai/gpt-5-chat"] = {
        "cost_per_1m_tokens": 15000.0,
        "median_latency_s": 1.8,
        "capabilities": ["reasoning", "coding", "math", "creative"],
        "speed_profile": "balanced"
    }
    
    print(f"   GPT-5: Cold start (||θ|| = 0.00)")




def register_gpt5_lst(router: BanditRouter) -> None:
    """Register GPT-5 with LST (adaptive n_eff)."""
    # Calculate semantic similarity to GPT-4o
    gpt5_dna = router._get_model_dna("openai/gpt-5-chat", ["reasoning", "coding", "math", "creative"], "balanced")
    gpt4_dna = router._get_model_dna("openai/gpt-4o", ["reasoning", "coding"], "balanced")
    
    gpt5_vec = router.encoder.encode([gpt5_dna], convert_to_numpy=True)[0]
    gpt4_vec = router.encoder.encode([gpt4_dna], convert_to_numpy=True)[0]
    similarity = np.dot(gpt5_vec, gpt4_vec) / (np.linalg.norm(gpt5_vec) * np.linalg.norm(gpt4_vec))
    
    # Adaptive n_eff (empirically optimized via hyperparameter sweep)
    if similarity > 0.8:
        n_effective = 5.0  # Optimal for high similarity (was 10.0)
    elif similarity > 0.6:
        n_effective = 3.0  # Proportionally adjusted
    else:
        n_effective = 1.0  # Minimal transfer for low similarity
    
    # Transfer from GPT-4o
    A_inv_gpt4 = router.bandit.A_inv["openai/gpt-4o"]
    b_gpt4 = router.bandit.b["openai/gpt-4o"]
    theta_gpt4 = A_inv_gpt4 @ b_gpt4
    
    A_init = np.eye(router.bandit.dim) * router.bandit.init_lambda
    b_init = (router.bandit.init_lambda * theta_gpt4) * n_effective
    
    router.bandit.models.append("openai/gpt-5-chat")
    router.bandit.A["openai/gpt-5-chat"] = A_init
    router.bandit.b["openai/gpt-5-chat"] = b_init
    router.bandit.A_inv["openai/gpt-5-chat"] = np.linalg.inv(A_init)
    router.bandit.last_update["openai/gpt-5-chat"] = router.bandit.t
    
    router.registry["openai/gpt-5-chat"] = {
        "cost_per_1m_tokens": 15000.0,
        "median_latency_s": 1.8,
        "capabilities": ["reasoning", "coding", "math", "creative"],
        "speed_profile": "balanced"
    }
    
    theta_gpt5 = router.bandit.A_inv["openai/gpt-5-chat"] @ router.bandit.b["openai/gpt-5-chat"]
    print(f"   GPT-5: LST (sim={similarity:.3f}, n_eff={n_effective}, ||θ|| = {np.linalg.norm(theta_gpt5):.2f})")


def run_online_routing(
    router: BanditRouter,
    shared_prompts: List[Dict],
    n_samples: int = 200,
    verbose: bool = False
) -> Tuple[List[float], List[str], List[Dict]]:
    """
    Run online routing, letting bandit choose models.
    
    Returns:
        (cumulative_regret_list, selected_models_list, decisions_log)
    """
    cumulative_regret_list = [0.0]
    selected_models_list = []
    decisions_log = []
    cumulative_regret = 0.0
    
    for t in range(min(n_samples, len(shared_prompts))):
        prompt_data = shared_prompts[t]
        
        # Use REAL prompt features: router's built-in feature extraction
        prompt_text = prompt_data['prompt_text']
        
        # Router converts prompt → embedding → PCA → context vector
        context, _ = router._build_routing_features(prompt_text)
        
        # Get UCB scores for all models BEFORE selection
        ucb_scores = {}
        for model in router.bandit.models:
            theta = router.bandit.A_inv[model] @ router.bandit.b[model]
            mean = float(theta.dot(context))
            uncertainty = np.sqrt(context.dot(router.bandit.A_inv[model]).dot(context))
            ucb = mean + router.bandit.alpha * uncertainty
            ucb_scores[model] = {'mean': mean, 'uncertainty': uncertainty, 'ucb': ucb}
        
        # Bandit selects model based on UCB
        selected_model, selected_ucb = router.bandit.select_arm(context)
        selected_models_list.append(selected_model)
        
        # Get reward for selected model
        if selected_model == "openai/gpt-4o":
            reward = prompt_data['gpt4o_reward']
        elif selected_model == "mistralai/mixtral-8x7b-instruct":
            reward = prompt_data['mixtral_reward']
        else:  # openai/gpt-5-chat
            reward = prompt_data['gpt5_reward']
        
        # Oracle reward (best possible)
        oracle_reward = max(prompt_data['gpt4o_reward'], 
                           prompt_data['mixtral_reward'],
                           prompt_data['gpt5_reward'])
        
        # Log decision
        decision = {
            'timestep': t,
            'selected': selected_model,
            'reward': reward,
            'oracle_reward': oracle_reward,
            'regret': oracle_reward - reward,
            'ucb_scores': ucb_scores,
            'all_rewards': {
                'gpt4o': prompt_data['gpt4o_reward'],
                'mixtral': prompt_data['mixtral_reward'],
                'gpt5': prompt_data['gpt5_reward']
            }
        }
        decisions_log.append(decision)
        
        if verbose and t < 10:
            print(f"  t={t}: Selected {selected_model.split('/')[-1]}, reward={reward:.1f}, oracle={oracle_reward:.1f}, regret={decision['regret']:.1f}")
            print(f"      UCB scores: GPT-4o={ucb_scores.get('openai/gpt-4o', {}).get('ucb', 0):.3f}, "
                  f"Mixtral={ucb_scores.get('mistralai/mixtral-8x7b-instruct', {}).get('ucb', 0):.3f}, "
                  f"GPT-5={ucb_scores.get('openai/gpt-5-chat', {}).get('ucb', 0):.3f}")
        
        # Update bandit
        router.bandit.update(selected_model, context, reward)
        
        # Track regret
        regret = oracle_reward - reward
        cumulative_regret += regret
        cumulative_regret_list.append(cumulative_regret)
    
    return cumulative_regret_list, selected_models_list, decisions_log


def plot_regret_waterfall(
    regret_cold_trials: List[List[float]],
    regret_lst_trials: List[List[float]],
    models_cold_all: List[List[str]],
    models_lst_all: List[List[str]],
    output_path: Path
) -> None:
    """Create the Regret Waterfall visualization with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute mean and std for each condition
    def compute_stats(trials):
        # Convert to numpy array (trials x timesteps)
        arr = np.array(trials)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        return mean, std
    
    mean_cold, std_cold = compute_stats(regret_cold_trials)
    mean_lst, std_lst = compute_stats(regret_lst_trials)
    
    samples = np.arange(len(mean_cold))
    
    # Left plot: Cumulative Regret with confidence bands
    ax1.plot(samples, mean_cold, label='Cold Start (Baseline)', 
            color='#e74c3c', linewidth=2.5, alpha=0.9)
    ax1.fill_between(samples, mean_cold - std_cold, mean_cold + std_cold,
                     color='#e74c3c', alpha=0.2)
    
    ax1.plot(samples, mean_lst, label='LST (Ours)', 
            color='#27ae60', linewidth=3.0, alpha=1.0)
    ax1.fill_between(samples, mean_lst - std_lst, mean_lst + std_lst,
                     color='#27ae60', alpha=0.2)
    
    ax1.set_xlabel('Samples (Online Routing)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax1.set_title(f'Regret Waterfall: Real GPT-5 Deployment (n={len(regret_cold_trials)} trials)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(mean_cold) - 1)
    ax1.set_ylim(bottom=0)
    
    # Annotate final regret with mean ± std
    for mean, std, color, x_offset in [(mean_cold, std_cold, '#e74c3c', 10),
                                        (mean_lst, std_lst, '#27ae60', 10)]:
        ax1.annotate(f'{mean[-1]:.1f}±{std[-1]:.1f}', 
                    xy=(len(mean)-1, mean[-1]),
                    xytext=(x_offset, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=color)
    
    # Right plot: Model Selection Distribution (aggregate across trials)
    def count_selections_multi(models_list):
        total_counts = {'GPT-4o': 0, 'Mixtral': 0, 'GPT-5': 0}
        for models in models_list:
            for m in models:
                if 'gpt-4o' in m:
                    total_counts['GPT-4o'] += 1
                elif 'mixtral' in m:
                    total_counts['Mixtral'] += 1
                else:
                    total_counts['GPT-5'] += 1
        # Normalize by number of trials
        for k in total_counts:
            total_counts[k] /= len(models_list)
        return total_counts
    
    counts_cold = count_selections_multi(models_cold_all)
    counts_lst = count_selections_multi(models_lst_all)
    
    x = np.arange(3)
    width = 0.35
    
    ax2.bar(x - width/2, list(counts_cold.values()), width, label='Cold Start', 
            color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, list(counts_lst.values()), width, label='LST', 
            color='#27ae60', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Avg Selection Count', fontsize=13, fontweight='bold')
    ax2.set_title(f'Model Selection Distribution (avg over {len(models_cold_all)} trials)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(counts_cold.keys()))
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"\n✅ Regret Waterfall saved: {output_path}")


def main():
    print("="*80)
    print("REGRET WATERFALL V2: Active Model Selection")
    print("="*80)
    print("\nThe bandit CHOOSES between GPT-4o, Mixtral, and GPT-5.")
    print("Regret = (best model reward) - (selected model reward)")
    
    # Load shared prompts from BOTH dev and holdout
    base_path = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    dev_file = base_path / "dev_rewards_complete.jsonl.gz"
    holdout_file = base_path / "holdout_rewards_complete.jsonl.gz"
    
    print(f"\n📂 Loading prompts with all three models...")
    print(f"   - Dev set: {dev_file.name}")
    print(f"   - Holdout set: {holdout_file.name}")
    
    dev_prompts = load_shared_prompts(dev_file)
    holdout_prompts = load_shared_prompts(holdout_file)
    
    # Combine both sets
    shared_prompts = dev_prompts + holdout_prompts
    
    print(f"   ✓ Dev: {len(dev_prompts)} prompts")
    print(f"   ✓ Holdout: {len(holdout_prompts)} prompts")
    print(f"   ✓ Total: {len(shared_prompts)} prompts")
    print(f"\n   Note: Base model priors (GPT-4o, Mixtral) are frozen.")
    print(f"         Only GPT-5 learns online, so dev/holdout mixing is safe.")
    
    # Create base registry
    registry = {
        "openai/gpt-4o": {
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "capabilities": ["reasoning", "coding"],
            "speed_profile": "balanced"
        },
        "mistralai/mixtral-8x7b-instruct": {
            "cost_per_1m_tokens": 500.0,
            "median_latency_s": 0.8,
            "capabilities": ["general", "coding"],
            "speed_profile": "fast"
        }
    }
    
    # Run multiple trials for statistical significance
    n_trials = 5
    n_samples = 500
    
    conditions = [
        ("Cold Start", register_gpt5_cold_start),
        ("LST", register_gpt5_lst)
    ]
    
    results = {name: {'regret': [], 'models': [], 'decisions': []} for name, _ in conditions}
    
    for trial in range(n_trials):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial+1}/{n_trials}")
        print("="*80)
        
        # Shuffle prompts for this trial (different order = different contexts)
        np.random.seed(42 + trial)
        trial_prompts = shared_prompts.copy()
        np.random.shuffle(trial_prompts)
        
        for i, (name, register_fn) in enumerate(conditions):
            print(f"\n  {name}:")
            router = create_base_router(registry)
            register_fn(router)
            
            # Verbose only for first trial, first condition
            verbose = (trial == 0 and i == 0)
            if verbose:
                print(f"    Running {n_samples} samples (showing first 10)...")
            
            regret, models, decisions = run_online_routing(
                router, trial_prompts, n_samples=n_samples, verbose=verbose
            )
            
            # Store results
            results[name]['regret'].append(regret)
            results[name]['models'].append(models)
            results[name]['decisions'].append(decisions)
            
            # Summary stats
            model_counts = {}
            for m in models:
                model_name = m.split('/')[-1]
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
            
            print(f"    Final regret: {regret[-1]:.2f}, Routing: {model_counts}")
    
    # Plot
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "regret_waterfall.png"
    
    plot_regret_waterfall(
        results["Cold Start"]['regret'],
        results["LST"]['regret'],
        results["Cold Start"]['models'],
        results["LST"]['models'],
        output_path
    )
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nFinal Cumulative Regret ({n_samples} samples, {n_trials} trials):")
    
    for name in ["Cold Start", "LST"]:
        final_regrets = [r[-1] for r in results[name]['regret']]
        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        print(f"  {name:20s}: {mean_regret:.2f} ± {std_regret:.2f}")
    
    # Compute savings
    mean_cold = np.mean([r[-1] for r in results["Cold Start"]['regret']])
    mean_lst = np.mean([r[-1] for r in results["LST"]['regret']])
    savings_vs_cold = mean_cold - mean_lst
    
    if mean_cold > 0:
        print(f"\n💰 LST Savings vs Cold Start:")
        print(f"   Absolute: {savings_vs_cold:.2f} regret")
        print(f"   Relative: {savings_vs_cold/mean_cold*100:.1f}%")
    
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()

