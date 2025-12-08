
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text

def visualize_landscape():
    print("🎨 Generating model landscape visualization...")
    
    # Load data
    try:
        with open('model_registry_cache_enhanced.json', 'r') as f:
            cache_data = json.load(f)
            data = cache_data.get('models', [])
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print("❌ Cache file not found!")
        return

    # Filter to models with real MMLU scores
    df = df[df['mmlu_score'] > 0].copy()
    
    # Set up the plot style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define colors for archetypes
    palette = {
        "Frontier (Complex Reasoning)": "#FF4B4B",      # Red
        "Reasoning Specialist": "#FFA500",              # Orange
        "RAG Specialist (Context Aware)": "#4B0082",    # Indigo
        "Edge/Local (Privacy First)": "#008000",        # Green
        "Bulk Ops (High Throughput)": "#0000FF"         # Blue
    }
    
    # Create scatter plot
    g = sns.scatterplot(
        data=df,
        x="input_cost_per_m",
        y="mmlu_score",
        hue="archetype",
        style="archetype",
        size="param_count_b",
        sizes=(100, 500),
        palette=palette,
        alpha=0.7,
        edgecolor="w",
        linewidth=1.5,
        ax=ax
    )
    
    # Set x-axis to symlog to handle 0s and large outliers
    ax.set_xscale('symlog', linthresh=0.01)
    
    # Add horizontal bands to show "similar quality" zones
    quality_bands = [
        (80, 90, "Elite Performance (80-90)", '#FFD0D0'),        # Darker red
        (70, 80, "High Performance (70-80)", '#FFE8B3'),         # Darker yellow
        (60, 70, "Good Performance (60-70)", '#B3D9FF'),         # Darker blue
        (50, 60, "Moderate Performance (50-60)", '#D0E8D0'),     # Light green
        (0, 50, "Basic Performance (0-50)", '#E8E8E8'),          # Light gray
    ]
    
    for y_min, y_max, label, color in quality_bands:
        ax.axhspan(y_min, y_max, alpha=0.4, color=color, zorder=0)
    
    # Smart labeling: only label most important models to avoid overlap
    texts = []
    labeled_models = set()
    labeled_positions = set()  # Track (cost, mmlu_score) to avoid duplicate positions
    
    def add_label(row, fontsize=8, color='black', weight='normal', suffix=''):
        """Helper to add a label if not already labeled and position not taken"""
        position = (round(row['input_cost_per_m'], 4), round(row['mmlu_score'], 2))
        
        if row['name'] not in labeled_models and position not in labeled_positions:
            labeled_models.add(row['name'])
            labeled_positions.add(position)
            label = f"{row['name']}{suffix}"
            texts.append(ax.text(row['input_cost_per_m'], row['mmlu_score'], 
                               label, fontsize=fontsize, color=color, weight=weight))
            return True
        return False
    
    # 1. Top performers in each quality band (to show cost differences)
    # Only label the cheapest and most expensive to highlight the cost range
    for y_min, y_max, _, _ in quality_bands:
        band_models = df[(df['mmlu_score'] >= y_min) & (df['mmlu_score'] < y_max)]
        if len(band_models) > 0:
            # Label cheapest and most expensive in each band
            cheapest = band_models[band_models['input_cost_per_m'] > 0].nsmallest(1, 'input_cost_per_m')
            most_expensive = band_models.nlargest(1, 'input_cost_per_m')
            
            for _, row in cheapest.iterrows():
                add_label(row, fontsize=8, color='green', weight='bold', 
                         suffix=f' ${row["input_cost_per_m"]:.2f}')
            
            for _, row in most_expensive.iterrows():
                add_label(row, fontsize=8, color='red', weight='bold',
                         suffix=f' ${row["input_cost_per_m"]:.2f}')
    
    # 2. Top 3 overall models only
    top_models = df.nlargest(3, 'mmlu_score')
    for _, row in top_models.iterrows():
        add_label(row, fontsize=9, weight='bold')
    
    # 3. Top 2 free models (high value) - sorted to get consistent selection
    free = df[df['input_cost_per_m'] == 0].sort_values(['mmlu_score', 'name'], ascending=[False, True])
    labeled_count = 0
    for _, row in free.iterrows():
        if add_label(row, fontsize=8, color='darkgreen', weight='bold', suffix=' FREE'):
            labeled_count += 1
            if labeled_count >= 2:
                break
    
    # 4. Highlight "High Value Zone" (Top Left)
    # High Performance (>70 MMLU) & Low Cost (<$0.50)
    from matplotlib.patches import Rectangle
    
    # Add a subtle green background for the high value zone
    rect = Rectangle((0.008, 70), 0.492, 25, 
                    linewidth=2, edgecolor='green', facecolor='#E8F5E9', 
                    alpha=0.3, linestyle='--', zorder=0)
    ax.add_patch(rect)
    
    ax.text(0.06, 92, "High Value Zone\n(Top Tier Performance @ Commodity Prices)", 
            fontsize=11, color='darkgreen', weight='bold', ha='center', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round,pad=0.5'))

    # 5. Add Explicit Value Comparison Arrows
    # Find pairs of models with similar MMLU (<2 diff) but huge cost diff (>10x)
    
    # Define specific comparisons we want to highlight (Expensive -> Cheap)
    comparisons = [
        ("OpenAI: GPT-4o", "DeepSeek: DeepSeek V3"),  # Example pair
        ("Anthropic: Claude 3.5 Sonnet", "DeepSeek: R1"), # Example pair
    ]
    
    # Helper to find model data
    def get_model_data(name_part):
        matches = df[df['name'].str.contains(name_part, case=False, regex=False)]
        if len(matches) > 0:
            return matches.iloc[0]
        return None

    # Draw arrows for high-value comparisons
    # We'll programmatically find the best examples in the Elite/High bands
    
    # Strategy: For the most expensive model in Elite band, find the cheapest in same band
    elite_models = df[df['mmlu_score'] >= 80]
    if len(elite_models) > 1:
        expensive = elite_models.nlargest(1, 'input_cost_per_m').iloc[0]
        cheapest = elite_models[elite_models['input_cost_per_m'] > 0].nsmallest(1, 'input_cost_per_m').iloc[0]
        
        if expensive['input_cost_per_m'] > cheapest['input_cost_per_m'] * 5:
            # Draw arrow
            savings = (expensive['input_cost_per_m'] - cheapest['input_cost_per_m']) / expensive['input_cost_per_m'] * 100
            multiple = expensive['input_cost_per_m'] / cheapest['input_cost_per_m']
            
            ax.annotate(f"{multiple:.0f}x Cheaper",
                        xy=(cheapest['input_cost_per_m'], cheapest['mmlu_score']), 
                        xytext=(expensive['input_cost_per_m'], expensive['mmlu_score']),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", 
                                      color="#D32F2F", lw=2, linestyle='-'),
                        fontsize=10, color="#D32F2F", weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", alpha=0.9))

    # Strategy: Same for High Performance band
    high_models = df[(df['mmlu_score'] >= 70) & (df['mmlu_score'] < 80)]
    if len(high_models) > 1:
        expensive = high_models.nlargest(1, 'input_cost_per_m').iloc[0]
        cheapest = high_models[high_models['input_cost_per_m'] > 0].nsmallest(1, 'input_cost_per_m').iloc[0]
        
        if expensive['input_cost_per_m'] > cheapest['input_cost_per_m'] * 5:
             # Draw arrow
            multiple = expensive['input_cost_per_m'] / cheapest['input_cost_per_m']
            
            ax.annotate(f"{multiple:.0f}x Cheaper",
                        xy=(cheapest['input_cost_per_m'], cheapest['mmlu_score']), 
                        xytext=(expensive['input_cost_per_m'], expensive['mmlu_score']),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", 
                                      color="#D32F2F", lw=2, linestyle='-'),
                        fontsize=10, color="#D32F2F", weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D32F2F", alpha=0.9))

    
    # Adjust text to avoid overlap with aggressive parameters
    try:
        adjust_text(texts, 
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.7, alpha=0.5),
                   expand_points=(2.0, 2.2),  # More aggressive expansion
                   expand_text=(1.8, 1.8),    # More space between labels
                   force_points=(0.8, 0.8),   # Stronger repulsion from points
                   force_text=(0.8, 0.8),     # Stronger repulsion between labels
                   lim=500,                    # More iterations
                   ax=ax)
    except ImportError:
        print("⚠️ adjustText not installed, skipping text adjustment")
    except Exception as e:
        print(f"⚠️ Text adjustment failed: {e}")
    
    # Add cost comparison annotations
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=2)
    ax.text(1.0, ax.get_ylim()[0] + 2, '$1/M tokens', 
           fontsize=10, ha='center', va='bottom', 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Customize plot
    ax.set_title("Price ≠ Performance: The LLM Value Landscape\n(Identical quality models exist at >100x lower cost)", 
                fontsize=18, weight='bold', pad=20, color='#1A237E')
    ax.set_xlabel("Input Cost ($ per Million Tokens) - Log Scale", fontsize=13, weight='bold')
    ax.set_ylabel("MMLU Score (Knowledge & Reasoning)", fontsize=13, weight='bold')
    
    # Create custom legend for performance tiers
    from matplotlib.patches import Patch
    
    # Get existing archetype legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Add performance tier patches
    tier_handles = [Patch(facecolor=color, alpha=0.4, label=label) 
                   for _, _, label, color in quality_bands]
    
    # Create archetype legend (top right)
    archetype_legend = ax.legend(handles, labels, 
                                bbox_to_anchor=(1.02, 1), 
                                loc='upper left',
                                title='Model Archetype',
                                frameon=True, 
                                shadow=True, 
                                fontsize=9,
                                title_fontsize=10)
    
    # Add archetype legend to the plot
    ax.add_artist(archetype_legend)
    
    # Create performance tier legend (below archetype legend)
    tier_legend = ax.legend(handles=tier_handles,
                           bbox_to_anchor=(1.02, 0.5), 
                           loc='upper left',
                           title='Performance Tier\n(MMLU Score)',
                           frameon=True, 
                           shadow=True, 
                           fontsize=9,
                           title_fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_file = 'model_landscape.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_file}")
    
    # Print summary statistics to highlight the value proposition
    print("\n📊 Cost Analysis by Performance Tier:")
    for y_min, y_max, label, _ in quality_bands:
        band_models = df[(df['mmlu_score'] >= y_min) & (df['mmlu_score'] < y_max) & (df['input_cost_per_m'] > 0)]
        if len(band_models) > 0:
            min_cost = band_models['input_cost_per_m'].min()
            max_cost = band_models['input_cost_per_m'].max()
            print(f"  {label}: ${min_cost:.2f} - ${max_cost:.2f} per M tokens ({max_cost/min_cost:.1f}x difference)")

if __name__ == "__main__":
    visualize_landscape()
