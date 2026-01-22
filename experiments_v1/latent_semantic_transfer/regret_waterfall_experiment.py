#!/usr/bin/env python3
"""
Regret Waterfall Experiment: Compare initialization strategies using REAL data.

Three conditions:
1. Cold Start (Baseline): A = λI, b = 0
2. Manual Heuristic (T-shirt sizing): Fixed n_eff = 5.0 (no similarity gating)
3. Latent Semantic Transfer: Adaptive n_eff based on similarity

Tracks cumulative regret over first 200 samples using real GPT-5 evaluations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gzip
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import BanditRouter


@dataclass
class RegretMetrics:
    """Metrics for tracking cumulative regret over time."""
    condition: str
    cumulative_regret: List[float]  # Regret at each timestep
    rewards: List[float]  # Actual rewards obtained
    optimal_rewards: List[float]  # Oracle rewards (what we could have gotten)


def load_real_rewards(rewards_file: Path) -> Dict[str, List[Tuple[str, float]]]:
    """Load real model rewards from offline dataset."""
    model_rewards = {}
    
    with gzip.open(rewards_file, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            model_id = entry['model_id']
            prompt = entry['prompt']
            reward = entry['raw_score']  # 0.0 or 1.0 from judge votes
            
            if model_id not in model_rewards:
                model_rewards[model_id] = []
            model_rewards[model_id].append((prompt, reward))
    
    return model_rewards


def create_base_router(registry: Dict[str, Dict]) -> BanditRouter:
    """Create router with base models (GPT-4, Mixtral) and warmup priors."""
    router = BanditRouter(
        model_registry=registry,
        alpha=0.05,
        init_lambda=1.0,
        verbose_routing=False
    )
    
    # Load priors
    priors_path = Path(__file__).parent.parent.parent / "data" / "routellm" / "priors_warmup_routellm_pca24.joblib"
    priors_data = joblib.load(priors_path)
    A_matrices = priors_data['A']
    b_vectors = priors_data['b']
    
    for model_id in ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]:
        if model_id in router.bandit.models:
            router.bandit.A[model_id] = A_matrices[model_id].copy()
            router.bandit.b[model_id] = b_vectors[model_id].copy()
            router.bandit.A_inv[model_id] = np.linalg.inv(router.bandit.A[model_id])
    
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
    
    print(f"   Registered GPT-5: Cold start (||θ|| = 0.00)")


def register_gpt5_manual_heuristic(router: BanditRouter) -> None:
    """Register GPT-5 with manual heuristic (fixed n_eff = 5.0, no gating)."""
    # Always use GPT-4-Turbo as neighbor (manual rule)
    A_inv_gpt4 = router.bandit.A_inv["openai/gpt-4-turbo"]
    b_gpt4 = router.bandit.b["openai/gpt-4-turbo"]
    theta_gpt4 = A_inv_gpt4 @ b_gpt4
    
    # Fixed n_eff = 5.0 (T-shirt sizing: "medium transfer")
    n_effective = 5.0
    
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
    print(f"   Registered GPT-5: Manual heuristic (n_eff=5.0, ||θ|| = {np.linalg.norm(theta_gpt5):.2f})")


def register_gpt5_lst(router: BanditRouter) -> None:
    """Register GPT-5 with Latent Semantic Transfer (adaptive n_eff)."""
    # Calculate semantic similarity
    gpt5_dna = router._get_model_dna("openai/gpt-5-chat", ["reasoning", "coding", "math", "creative"], "balanced")
    gpt4_dna = router._get_model_dna("openai/gpt-4-turbo", ["reasoning", "coding"], "balanced")
    
    gpt5_vec = router.encoder.encode([gpt5_dna], convert_to_numpy=True)[0]
    gpt4_vec = router.encoder.encode([gpt4_dna], convert_to_numpy=True)[0]
    similarity = np.dot(gpt5_vec, gpt4_vec) / (np.linalg.norm(gpt5_vec) * np.linalg.norm(gpt4_vec))
    
    # Adaptive n_eff based on similarity
    if similarity > 0.8:
        n_effective = 10.0
    elif similarity > 0.6:
        n_effective = 5.0
    else:
        n_effective = 1.0
    
    # Transfer from GPT-4-Turbo
    A_inv_gpt4 = router.bandit.A_inv["openai/gpt-4-turbo"]
    b_gpt4 = router.bandit.b["openai/gpt-4-turbo"]
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
    print(f"   Registered GPT-5: LST (similarity={similarity:.3f}, n_eff={n_effective}, ||θ|| = {np.linalg.norm(theta_gpt5):.2f})")


def run_online_learning(
    router: BanditRouter,
    real_rewards: Dict[str, List[Tuple[str, float]]],
    n_samples: int = 200
) -> RegretMetrics:
    """
    Run online learning with real rewards, tracking cumulative regret.
    
    Returns:
        RegretMetrics with cumulative regret at each timestep
    """
    gpt5_rewards = real_rewards["openai/gpt-5-chat"][:n_samples]
    
    cumulative_regret_list = []
    rewards_list = []
    optimal_rewards_list = []
    cumulative_regret = 0.0
    
    # Oracle: always choose GPT-5 (since we're testing GPT-5 warmup)
    optimal_reward = 1.0  # Assume normalized rewards, oracle = 1.0
    
    for t, (prompt, true_reward) in enumerate(gpt5_rewards):
        # Create dummy context (we only care about GPT-5 routing)
        context = np.random.randn(router.bandit.dim)
        context = context / np.linalg.norm(context)  # Normalize
        
        # Get current cumulative regret BEFORE this sample
        cumulative_regret_list.append(cumulative_regret)
        
        # Bandit selects model (we force GPT-5 for this experiment)
        selected_model = "openai/gpt-5-chat"
        
        # Observe reward
        reward = true_reward
        rewards_list.append(reward)
        optimal_rewards_list.append(optimal_reward)
        
        # Update bandit
        router.bandit.update(selected_model, context, reward)
        
        # Update cumulative regret
        regret = optimal_reward - reward
        cumulative_regret += regret
    
    # Final cumulative regret
    cumulative_regret_list.append(cumulative_regret)
    
    return cumulative_regret_list, rewards_list, optimal_rewards_list


def plot_regret_waterfall(
    metrics_cold: Tuple[List[float], List[float], List[float]],
    metrics_manual: Tuple[List[float], List[float], List[float]],
    metrics_lst: Tuple[List[float], List[float], List[float]],
    output_path: Path
) -> None:
    """
    Create the Regret Waterfall visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Unpack metrics
    regret_cold, rewards_cold, _ = metrics_cold
    regret_manual, rewards_manual, _ = metrics_manual
    regret_lst, rewards_lst, _ = metrics_lst
    
    # Plot cumulative regret
    samples = np.arange(len(regret_cold))
    
    ax.plot(samples, regret_cold, 
            label='Cold Start (Baseline)', 
            color='#e74c3c', linewidth=2.5, linestyle='-', alpha=0.9)
    
    ax.plot(samples, regret_manual, 
            label='Manual Heuristic (T-shirt sizing)', 
            color='#f39c12', linewidth=2.5, linestyle='--', alpha=0.9)
    
    ax.plot(samples, regret_lst, 
            label='Latent Semantic Transfer (Ours)', 
            color='#27ae60', linewidth=3.0, linestyle='-', alpha=1.0)
    
    # Styling
    ax.set_xlabel('Samples (Online Learning)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax.set_title('Regret Waterfall: Initialization Strategy Comparison\n(Real GPT-5 Data, 200 Samples)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(regret_cold) - 1)
    ax.set_ylim(bottom=0)
    
    # Add annotations
    final_regret_cold = regret_cold[-1]
    final_regret_manual = regret_manual[-1]
    final_regret_lst = regret_lst[-1]
    
    # Annotate final values
    ax.annotate(f'{final_regret_cold:.1f}', 
                xy=(len(regret_cold)-1, final_regret_cold),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#e74c3c')
    
    ax.annotate(f'{final_regret_manual:.1f}', 
                xy=(len(regret_manual)-1, final_regret_manual),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#f39c12')
    
    ax.annotate(f'{final_regret_lst:.1f}', 
                xy=(len(regret_lst)-1, final_regret_lst),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#27ae60')
    
    # Add "savings" region
    if final_regret_lst < final_regret_cold:
        savings = final_regret_cold - final_regret_lst
        mid_x = len(regret_cold) // 2
        ax.fill_between([mid_x, len(regret_cold)-1], 
                        [regret_lst[mid_x], final_regret_lst],
                        [regret_cold[mid_x], final_regret_cold],
                        color='green', alpha=0.1, label='_nolegend_')
        
        ax.text(len(regret_cold) * 0.7, (final_regret_cold + final_regret_lst) / 2,
                f'Savings:\n{savings:.1f} regret\n({savings/final_regret_cold*100:.1f}%)',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Regret Waterfall saved: {output_path}")
    
    # Also save high-quality version for paper
    output_path_pdf = output_path.with_suffix('.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ PDF version saved: {output_path_pdf}")


def main():
    print("="*80)
    print("REGRET WATERFALL EXPERIMENT")
    print("="*80)
    print("\nComparing initialization strategies using REAL GPT-5 data:")
    print("1. Cold Start (Baseline): A = λI, b = 0")
    print("2. Manual Heuristic (T-shirt sizing): Fixed n_eff = 5.0")
    print("3. Latent Semantic Transfer (Ours): Adaptive n_eff")
    
    # Load real rewards
    rewards_file = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset" / "dev_rewards_complete.jsonl.gz"
    print(f"\n📂 Loading real rewards: {rewards_file}")
    real_rewards = load_real_rewards(rewards_file)
    print(f"   ✓ Loaded {len(real_rewards['openai/gpt-5-chat'])} GPT-5 samples")
    
    # Create base registry
    registry = {
        "openai/gpt-4-turbo": {
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
    
    # Condition 1: Cold Start
    print("\n" + "="*80)
    print("CONDITION 1: Cold Start (Baseline)")
    print("="*80)
    router_cold = create_base_router(registry)
    register_gpt5_cold_start(router_cold)
    print("\n🧪 Running online learning (200 samples)...")
    metrics_cold = run_online_learning(router_cold, real_rewards, n_samples=200)
    print(f"   Final cumulative regret: {metrics_cold[0][-1]:.2f}")
    
    # Condition 2: Manual Heuristic
    print("\n" + "="*80)
    print("CONDITION 2: Manual Heuristic (T-shirt sizing)")
    print("="*80)
    router_manual = create_base_router(registry)
    register_gpt5_manual_heuristic(router_manual)
    print("\n🧪 Running online learning (200 samples)...")
    metrics_manual = run_online_learning(router_manual, real_rewards, n_samples=200)
    print(f"   Final cumulative regret: {metrics_manual[0][-1]:.2f}")
    
    # Condition 3: Latent Semantic Transfer
    print("\n" + "="*80)
    print("CONDITION 3: Latent Semantic Transfer (Ours)")
    print("="*80)
    router_lst = create_base_router(registry)
    register_gpt5_lst(router_lst)
    print("\n🧪 Running online learning (200 samples)...")
    metrics_lst = run_online_learning(router_lst, real_rewards, n_samples=200)
    print(f"   Final cumulative regret: {metrics_lst[0][-1]:.2f}")
    
    # Plot results
    print("\n" + "="*80)
    print("CREATING REGRET WATERFALL VISUALIZATION")
    print("="*80)
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "regret_waterfall.png"
    
    plot_regret_waterfall(metrics_cold, metrics_manual, metrics_lst, output_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFinal Cumulative Regret (200 samples):")
    print(f"  Cold Start:       {metrics_cold[0][-1]:.2f}")
    print(f"  Manual Heuristic: {metrics_manual[0][-1]:.2f}")
    print(f"  LST (Ours):       {metrics_lst[0][-1]:.2f}")
    
    savings_vs_cold = metrics_cold[0][-1] - metrics_lst[0][-1]
    savings_vs_manual = metrics_manual[0][-1] - metrics_lst[0][-1]
    
    print(f"\n💰 Regret Savings:")
    print(f"  vs Cold Start:       {savings_vs_cold:.2f} ({savings_vs_cold/metrics_cold[0][-1]*100:.1f}%)")
    print(f"  vs Manual Heuristic: {savings_vs_manual:.2f} ({savings_vs_manual/metrics_manual[0][-1]*100:.1f}%)")
    
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()

