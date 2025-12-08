
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import LogNorm
import sys
import os

# Ensure we can import from llm_jury
sys.path.append(os.getcwd())

from llm_jury.core.models import (
    ModelMetadata, RoutingDecision, ProductArchetype, PromptCategory
)
from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy

def load_models():
    try:
        with open('model_registry_cache_enhanced.json', 'r') as f:
            cache_data = json.load(f)
            data = cache_data.get('models', [])
        
        models = []
        for d in data:
            # Convert dict to ModelMetadata
            # We need to filter keys that match ModelMetadata fields
            # For simplicity, we'll instantiate with required fields and update
            
            # Handle potential missing fields with defaults
            m = ModelMetadata(
                name=d.get('name', 'Unknown'),
                mmlu_score=d.get('mmlu_score', 0),
                gpqa_score=d.get('gpqa_score', 0),
                math_score=d.get('math_score', 0),
                ifeval_score=d.get('ifeval_score', 0),
                tool_use_ability=d.get('tool_use_ability', 0),
                context_window_k=d.get('context_window_k', 0),
                hallucination_rate=d.get('hallucination_rate', 0),
                ethics_score=d.get('ethics_score', 0),
                hf_downloads=d.get('hf_downloads', 0),
                hf_likes=d.get('hf_likes', 0),
                hf_created_at=d.get('hf_created_at', ""),
                archetype=ProductArchetype(d.get('archetype', ProductArchetype.FRONTIER.value)),
                input_cost_per_m=d.get('input_cost_per_m', 0),
                output_cost_per_m=d.get('output_cost_per_m', 0),
                median_latency_ms=d.get('median_latency_ms', 0),
                param_count_b=d.get('param_count_b', 0)
            )
            if m.mmlu_score > 0: # Filter out invalid models
                models.append(m)
                
        return models
    except FileNotFoundError:
        print("❌ Cache file not found!")
        return []

def visualize_optimization():
    print("⚖️  Generating optimization landscape visualization (Faceted by Archetype)...")
    
    models = load_models()
    if not models: return

    # 1. Setup Ranker with Global Baseline
    # Find baseline model (GPT-4o)
    baseline = next((m for m in models if "GPT-4o" in m.name and "ChatGPT" not in m.name), None)
    if not baseline:
        baseline = models[0]
        print(f"⚠️ Baseline GPT-4o not found, using {baseline.name}")
    
    ranker = Optimizer(baseline_model=baseline, all_models_data=[], strategy=OptimizationStrategy.BALANCED)
    
    # Define target archetypes
    archetypes = [
        (ProductArchetype.FRONTIER, "Frontier (Complex Reasoning)"),
        (ProductArchetype.REASONING_SPECIALIST, "Reasoning Specialist"),
        (ProductArchetype.RAG_SPECIALIST, "RAG Specialist (Context Aware)"),
        (ProductArchetype.BULK_OPS, "Bulk Ops (High Throughput)")
    ]
    
    # Prepare plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (arch_enum, arch_name) in enumerate(archetypes):
        ax = axes[idx]
        
        # Note: We match on the string value of the enum in the metadata
        arch_models = [m for m in models if m.archetype.value == arch_name]
        
        if not arch_models:
            print(f"⚠️ No models found for {arch_name}")
        # Prepare Data for Plotting
    plot_data = []
    for m in models:
        # Calculate scores for a generic decision
        decision = RoutingDecision(archetype=m.archetype, category=PromptCategory.GENERAL, reason="Visualization")
        weights = ranker._get_weights(decision)
        
        # Calculate Proxy Latency
        proxy_lat = calc_proxy_latency(m)
        
        plot_data.append({
            'name': m.name,
            'archetype': m.archetype.value,
            'quality_score': ranker._calc_quality(m, weights),
            'cost': (m.input_cost_per_m * 0.75) + (m.output_cost_per_m * 0.25),
            'latency': proxy_lat,
            'chebyshev_score': ranker.rank([m], decision, top_k=1)[0].score if ranker.rank([m], decision, top_k=1) else 0
        })
        
    df_all = pd.DataFrame(plot_data)
    
    # Calculate global x-axis limits (Cost) for consistency
    global_min_cost = df_all['cost'].replace(0, 0.01).min()
    global_max_cost = df_all['cost'].replace(0, 0.01).max()
    
    # Add padding on log scale
    x_min = global_min_cost * 0.5
    x_max = global_max_cost * 2.0
    
    # Create Faceted Plot (2x2)
    archetypes = [
        (ProductArchetype.FRONTIER, "Frontier Models"),
        (ProductArchetype.REASONING_SPECIALIST, "Reasoning Models"),
        (ProductArchetype.RAG_SPECIALIST, "RAG Specialists"),
        (ProductArchetype.BULK_OPS, "Bulk Ops Models")
    ]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    for idx, (arch_enum, arch_name) in enumerate(archetypes):
        ax = axes[idx]
        df = df_all[df_all['archetype'] == arch_enum.value].copy()
        
        if df.empty: continue
            
        # Rank within archetype
        df['rank'] = df['chebyshev_score'].rank(ascending=True).astype(int)
        
        # Handle log scale zeros
        df['cost_plot'] = df['cost'].replace(0, 0.01)
        
        # Plot
        # Color by Latency (Proxy Data)
        sc = ax.scatter(
            df['cost_plot'], 
            df['quality_score'], 
            c=df['latency'], 
            cmap='RdYlGn_r', # Red=Slow, Green=Fast
            norm=LogNorm(vmin=50, vmax=2000), # Adjusted for proxy range (ms)
            s=150, 
            alpha=0.8, 
            edgecolors='w', 
            linewidth=1
        )
        
        # Scales and Labels
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_xlim(x_min, x_max)  # Shared x-axis range
        
        # Force same tick locations for all subplots
        tick_locs = [0.01, 0.1, 1.0, 10.0]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f'${t}' for t in tick_locs])
        
        ax.set_title(arch_name, fontsize=16, weight='bold')
        ax.set_xlabel("Blended Cost ($/M) - Log", fontsize=14)
        ax.set_ylabel("Quality Score (vs GPT-4o)", fontsize=14)
        ax.tick_params(labelsize=12)
        
        # Highlight Top 3
        top_3 = df.nsmallest(3, 'chebyshev_score')
        texts = []
        for _, row in top_3.iterrows():
            parts = row['name'].split(': ')
            short_name = parts[1] if len(parts) > 1 else row['name']
            label = f"#{row['rank']} {short_name}"
            texts.append(ax.text(row['cost_plot'], row['quality_score'], label, 
                                fontsize=11, weight='bold', color='black'))
            ax.plot(row['cost_plot'], row['quality_score'], 'o', ms=20, mfc='none', mec='gold', mew=2)
            
        # Adjust text
        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        except: pass
        
        # Utopia Annotation (only on first plot to save space)
        if idx == 0:
            ax.annotate("UTOPIA", xy=(0.05, 98), xytext=(0.5, 98),
                       arrowprops=dict(facecolor='black', shrink=0.05),
                       fontsize=12, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="green", alpha=0.9))

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Est. Latency (ms) - Proxy based on Size', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.suptitle("The Optimization Frontier by Archetype\n(Quality vs. Cost vs. Proxy Latency [Size-based])", 
                 fontsize=22, weight='bold', color='#1A237E', y=0.95)
    
    plt.subplots_adjust(right=0.9, wspace=0.2, hspace=0.3)
    
    output_file = 'optimization_landscape.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved faceted optimization visualization to {output_file}")

def calc_proxy_latency(model):
    """
    Calculate proxy latency based on model size (Physics-based approximation).
    Formula: Latency ≈ k * Size + b
    k = 5 ms/B (slope for typical GPU inference)
    b = 50 ms (fixed overhead)
    """
    params = model.param_count_b if model.param_count_b > 0 else 7.0 # Default to 7B if missing
    return (5.0 * params) + 50.0

def visualize_regrets():
    print("📊 Generating regret scorecard visualization (Proxy Data)...")
    
    models = load_models()
    if not models: return
    
    # Setup Ranker
    baseline = next((m for m in models if "GPT-4o" in m.name and "ChatGPT" not in m.name), models[0])
    ranker = Optimizer(baseline_model=baseline, all_models_data=[], strategy=OptimizationStrategy.BALANCED)
    
    archetypes = [
        (ProductArchetype.FRONTIER, "Frontier"),
        (ProductArchetype.REASONING_SPECIALIST, "Reasoning"),
        (ProductArchetype.RAG_SPECIALIST, "RAG Specialist"),
        (ProductArchetype.BULK_OPS, "Bulk Ops")
    ]
    
    # Okabe-Ito Palette (Color Blind Friendly)
    colors = ['#56B4E9', '#D55E00', '#CC79A7']
    metrics = ['Quality Gap', 'Cost Gap', 'Latency Gap']
    
    # First pass: Calculate all regrets to find global max
    all_regret_data = []
    
    for arch_enum, arch_name in archetypes:
        # Filter and Rank
        arch_models = [m for m in models if m.archetype.value == arch_enum.value]
        if not arch_models: continue
            
        decision = RoutingDecision(archetype=arch_enum, category=PromptCategory.GENERAL, reason="Regret Analysis")
        
        weights = ranker._get_weights(decision)
        base_q = ranker._calc_quality(baseline, weights)
        base_cost = (baseline.input_cost_per_m * 0.75) + (baseline.output_cost_per_m * 0.25)
        
        # Baseline Proxy Latency
        base_lat = calc_proxy_latency(baseline)
        
        regret_data = []
        
        for m in arch_models:
            # Calculate Proxy Latency
            proxy_lat = calc_proxy_latency(m)

            # 1. Calculate raw regrets
            q_score = ranker._calc_quality(m, weights)
            q_regret = max(0, 1.0 - (q_score / base_q)) if base_q > 0 else 0
            
            m_cost = (m.input_cost_per_m * 0.75) + (m.output_cost_per_m * 0.25)
            c_regret = np.log1p(m_cost / base_cost) / np.log1p(1.0) if base_cost > 0 else 0
            
            l_regret = np.log1p(proxy_lat / base_lat) / np.log1p(1.0) if base_lat > 0 else 0
            
            # 2. Apply Chebyshev weights
            w_q_regret = 0.6 * q_regret
            w_c_regret = 0.2 * c_regret
            w_l_regret = 0.2 * l_regret
            
            chebyshev_score = max(w_q_regret, w_c_regret, w_l_regret)
            
            regret_data.append({
                'archetype': arch_name,
                'name': m.name,
                'Quality Gap': w_q_regret,
                'Cost Gap': w_c_regret,
                'Latency Gap': w_l_regret,
                'Total Score': chebyshev_score
            })
        
        all_regret_data.extend(regret_data)
    
    # Calculate global max regret
    if all_regret_data:
        all_df = pd.DataFrame(all_regret_data)
        global_max_regret = all_df[metrics].max().max()
    else:
        global_max_regret = 0.2
    
    # Prepare plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    for idx, (arch_enum, arch_name) in enumerate(archetypes):
        ax = axes[idx]
        
        # Filter data for this archetype
        arch_data = [d for d in all_regret_data if d['archetype'] == arch_name]
        if not arch_data: continue
        
        # Sort by Total Score (Ascending) and take Top 5
        df = pd.DataFrame(arch_data).sort_values('Total Score').head(5)
        
        # Melt for grouped bar chart
        df_melt = df.melt(id_vars=['name', 'Total Score'], value_vars=metrics, 
                          var_name='Metric', value_name='Weighted Regret')
        
        # Shorten names
        df_melt['Short Name'] = df_melt['name'].apply(lambda x: x.split(': ')[1] if ': ' in x else x)
        
        # Plot
        sns.barplot(data=df_melt, y='Short Name', x='Weighted Regret', hue='Metric', 
                    palette=colors, ax=ax, orient='h')
        
        # Add value labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        
        # Set shared x-axis limit (fixed at 0.2 for consistency)
        ax.set_xlim(0, 0.2)
        
        ax.set_title(f"{arch_name} (Top 5)", fontsize=14, weight='bold')
        ax.set_xlabel("Weighted Regret (Lower is Better)", fontsize=10)
        ax.set_ylabel("")
        
        # Add rank numbers to Y-labels
        ax.set_yticklabels([f"#{i+1} {name}" for i, name in enumerate(df['name'].apply(lambda x: x.split(': ')[1] if ': ' in x else x))])

    # Legend (Shared)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    # Remove individual legends
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    plt.suptitle("Why They Win: The Regret Scorecard\n(Optimizing for Quality, Cost, and Latency [Size Proxy])", 
                 fontsize=20, weight='bold', color='#1A237E', y=0.95)
    
    # Add Insight
    insight_text = (
        "HOW TO READ:\n"
        "• The 'Score' is determined by the LONGEST bar.\n"
        "• The Winner (#1) is the model with the shortest 'longest bar'.\n"
        "• Blue Bar = Quality Issue | Orange Bar = Cost Issue | Purple Bar = Latency Issue"
    )
    fig.text(0.5, 0.02, insight_text, fontsize=12, ha='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='gray'))

    plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.3, hspace=0.3)
    
    output_file = 'optimization_scorecard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved regret scorecard to {output_file}")

if __name__ == "__main__":
    visualize_optimization()
    visualize_regrets()
