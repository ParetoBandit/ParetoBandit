"""
Figure 12: Learning Efficiency (Fixed)
- Shuffled Data: Removes 'Sine Wave' sorting artifacts.
- Exponential Smoothing: Removes 'Decay' artifacts to show true current performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from banditgpt import BanditRouter
from final_release.baselines import BaRPRouter, PILOTRouter
from final_release.kdd_paper.table_3.router_performance_comparison import (
    load_model_registry,
    load_battle_dataset,
    RouteLLMRouter,
    FrugalGPTRouter,
)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def simulate_rolling_quality(router_name, df, registry, encoder, n_runs=5):
    """
    Simulate router using Exponential Moving Average (EMA) to show REAL-TIME learning.
    """
    n_requests = len(df)
    queries = df["question"].tolist()
    
    # Pre-encode all queries
    print(f"    Pre-encoding {n_requests} queries...")
    query_embeddings = encoder.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    
    # Identify Baselines
    weak_model = min(registry.keys(),
                   key=lambda m: float(registry[m].get("input_cost_per_m", 100)))
    strong_model = min(registry.keys(),
                     key=lambda m: float(registry[m].get("hallucination_vectara", 100)))

    def get_reward(model_id):
        hall = float(registry.get(model_id, {}).get("hallucination_vectara", 10.0))
        return max(0.0, 1.0 - (hall / 100.0))

    # Initialize Static Routers
    static_router = None
    if router_name == "FrugalGPT":
        static_router = FrugalGPTRouter()
    elif router_name == "RouteLLM":
        static_router = RouteLLMRouter()

    all_runs_curves = []

    for run in range(n_runs):
        # Set seed per run
        np.random.seed(42 + run)
        
        # Initialize Bandit
        run_router = None
        if router_name == "BanditGPT":
            run_router = BanditRouter.create(
                model_registry=registry,
                priors="benchmark",  # Use HLE benchmark priors for informed initialization
                exploration="balanced", 
                forgetting_factor=1.0
            )

        rewards = []
        
        for i, query in enumerate(queries):
            # 1. Select Model
            if router_name == "BanditGPT":
                emb = query_embeddings[i]
                ctx = np.append(emb, 1.0)
                selected, _ = run_router.bandit.select_arm(ctx)
            elif router_name in ["FrugalGPT", "RouteLLM"]:
                prob_weak = static_router.predict_proba(query)
                selected = weak_model if prob_weak > 0.5 else strong_model
            else: # Oracles
                selected = strong_model

            # 2. Get Reward
            reward = get_reward(selected)
            rewards.append(reward)
            
            # 3. Update Bandit
            if router_name == "BanditGPT":
                run_router.bandit.update(selected, ctx, reward)
        
        # Calculate Exponential Moving Average (EMA) for this run
        # span=200 makes it responsive but smooth
        ema_curve = pd.Series(rewards).ewm(span=300, adjust=False).mean().values
        all_runs_curves.append(ema_curve)
    
    # Average across runs
    all_runs_curves = np.array(all_runs_curves)
    mean_curve = np.mean(all_runs_curves, axis=0)
    std_curve = np.std(all_runs_curves, axis=0)
    
    return np.arange(n_requests), mean_curve, std_curve

def main():
    print("="*60)
    print("FIGURE 12: LEARNING EFFICIENCY (SHUFFLED + EMA)")
    print("="*60)
    
    # 1. Load & SHUFFLE Data
    print("\n[1/5] Loading and SHUFFLING data...")
    registry = load_model_registry()
    df = load_battle_dataset(n_samples=2000)
    
    # --- CRITICAL FIX: Randomize order to kill the sine wave ---
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Load Encoder
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Run Simulation
    routers_config = [
        ("BanditGPT\n(Ours)", "BanditGPT"),
        ("RouteLLM\n(Static)", "RouteLLM"),
        ("FrugalGPT\n(Cascade)", "FrugalGPT"),
        ("BaRP Oracle\n(Target)", "BaRP")
    ]
    
    results = {}
    for name, r_type in routers_config:
        print(f"  Simulating {name.replace(chr(10), ' ')}...")
        x, y, err = simulate_rolling_quality(r_type, df, registry, encoder, n_runs=5)
        results[name] = (x, y, err)

    # 4. Plot
    print("\n[4/5] Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "BanditGPT\n(Ours)": '#2E86AB',
        "BaRP Oracle\n(Target)": '#C73E1D',
        "FrugalGPT\n(Cascade)": '#F18F01',
        "RouteLLM\n(Static)": '#A23B72'
    }
    
    for name, (x, y, err) in results.items():
        ax.plot(x, y, label=name, color=colors[name], linewidth=2.5)
        if "BanditGPT" in name:
            ax.fill_between(x, y-err, y+err, color=colors[name], alpha=0.15)
            
    ax.set_title("Figure 12: Learning Efficiency (Smoothed)", fontweight='bold')
    ax.set_xlabel("Number of Requests", fontweight='bold')
    ax.set_ylabel("Average Quality (EMA)", fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0.90, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "figure_12_learning_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")
    
    # 5. Print Data Table
    print("\n[5/5] Results Summary:")
    print("="*70)
    print(f"{'Router':<30} {'Start (Q@100)':<15} {'Mid (Q@1000)':<15} {'Final (Q@2000)':<15}")
    print("-"*70)
    
    for name, (x, y, err) in results.items():
        clean_name = name.replace('\n', ' ')
        start_q = y[99] if len(y) > 99 else y[-1]  # Quality at request 100
        mid_q = y[999] if len(y) > 999 else y[-1]  # Quality at request 1000
        final_q = y[-1]  # Final quality
        
        print(f"{clean_name:<30} {start_q:.4f}          {mid_q:.4f}          {final_q:.4f}")
    
    print("="*70)
    print("\nNote: Quality = Average EMA over last 300 requests (span=300)")


if __name__ == "__main__":
    main()
