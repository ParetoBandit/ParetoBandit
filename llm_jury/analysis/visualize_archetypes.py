
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text
from matplotlib.patches import Rectangle, Patch

def load_data():
    try:
        with open('model_registry_cache_enhanced.json', 'r') as f:
            cache_data = json.load(f)
            data = cache_data.get('models', [])
        df = pd.DataFrame(data)
        # Filter to models with real MMLU scores
        return df[df['mmlu_score'] > 0].copy()
    except FileNotFoundError:
        print("❌ Enhanced cache file not found! Run 'python generate_enhanced_cache.py'")
        return None

def visualize_archetypes():
    print("🎨 Generating archetype landscape visualization...")
    df = load_data()
    if df is None: return

    # Filter for the 3 main archetypes we want to show
    target_archetypes = [
        "Frontier (Complex Reasoning)",
        "Reasoning Specialist",
        "RAG Specialist (Context Aware)"
    ]
    
    df_filtered = df[df['archetype'].isin(target_archetypes)].copy()
    
    # Set up the plot style
    sns.set_theme(style="whitegrid")
    
    # Create a FacetGrid - adjust col_wrap to 3 for single row
    g = sns.FacetGrid(df_filtered, col="archetype", col_wrap=3, height=6, aspect=1.3,
                      sharex=True, sharey=True)
    
    # Define colors for archetypes (matching main plot)
    palette = {
        "Frontier (Complex Reasoning)": "#FF4B4B",      # Red
        "Reasoning Specialist": "#FFA500",              # Orange
        "RAG Specialist (Context Aware)": "#4B0082",    # Indigo
    }
    
    def scatter_plot(data, **kwargs):
        ax = plt.gca()
        archetype = data['archetype'].iloc[0]
        color = palette.get(archetype, 'blue')
        
        # Plot points
        sns.scatterplot(data=data, x="input_cost_per_m", y="mmlu_score", 
                        size="param_count_b", sizes=(50, 300),
                        color=color, alpha=0.7, edgecolor="w", linewidth=1, ax=ax, legend=False)
        
        # Set log scale
        ax.set_xscale('symlog', linthresh=0.01)
        
        # Highlight High Value Zone
        rect = Rectangle((0.008, 70), 0.492, 25, 
                        linewidth=1, edgecolor='green', facecolor='#E8F5E9', 
                        alpha=0.3, linestyle='--', zorder=0)
        ax.add_patch(rect)
        
        # Label top value models in this archetype
        # Find models in top 20% of MMLU for this archetype, then cheapest of those
        if len(data) > 0:
            max_score = data['mmlu_score'].max()
            top_tier = data[data['mmlu_score'] >= max_score * 0.9]
            
            texts = []
            # Label the absolute best performer
            best = data.nlargest(1, 'mmlu_score').iloc[0]
            texts.append(ax.text(best['input_cost_per_m'], best['mmlu_score'], 
                                best['name'], fontsize=10, weight='bold', color=color))
            
            # Label the best value (cheapest in top tier)
            if len(top_tier) > 0:
                best_value = top_tier[top_tier['input_cost_per_m'] > 0].nsmallest(1, 'input_cost_per_m')
                if not best_value.empty:
                    val = best_value.iloc[0]
                    if val['name'] != best['name']:
                        texts.append(ax.text(val['input_cost_per_m'], val['mmlu_score'], 
                                           f"{val['name']}\n(Best Value)", fontsize=10, weight='bold', color='green'))
            
            # Adjust text
            try:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            except: pass

    g.map_dataframe(scatter_plot)
    
    # Titles and Labels - Increased Font Sizes
    g.set_titles("{col_name}", size=16, weight='bold')
    g.set_axis_labels("Input Cost ($/M Tokens)", "MMLU Score")
    
    # Increase axis label font sizes manually since set_axis_labels doesn't take size arg directly in older seaborn
    for ax in g.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(labelsize=12)
    
    # Add main title
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Archetype Value Analysis: Finding the Sweet Spot by Use Case", 
                   fontsize=22, weight='bold', color='#1A237E')
    
    # Save
    output_file = 'archetype_landscape.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved archetype visualization to {output_file}")


def visualize_savings():
    print("💰 Generating savings potential visualization...")
    df = load_data()
    if df is None: return
    
    # Define comparisons: Flagship vs Best Value Alternative
    # (Flagship Name, Alternative Name)
    comparisons = [
        ("OpenAI: ChatGPT-4o", "DeepSeek: DeepSeek V3"), 
        ("OpenAI: GPT-4o", "DeepSeek: DeepSeek V3"),     
        ("Anthropic: Claude 3.7 Sonnet", "DeepSeek: R1"), 
        ("Google: Gemini 2.5 Pro", "DeepSeek: DeepSeek V3"), 
        ("Mistral: Mistral Large 2", "DeepSeek: DeepSeek V3") 
    ]
    
    plot_data = []
    
    for expensive_name, cheap_name in comparisons:
        # Find models
        expensive = df[df['name'].str.contains(expensive_name, case=False, regex=False)]
        cheap = df[df['name'].str.contains(cheap_name, case=False, regex=False)]
        
        if not expensive.empty and not cheap.empty:
            exp_row = expensive.iloc[0]
            cheap_row = cheap.iloc[0]
            
            # Calculate savings
            input_savings = (exp_row['input_cost_per_m'] - cheap_row['input_cost_per_m']) / exp_row['input_cost_per_m'] * 100
            output_savings = (exp_row['output_cost_per_m'] - cheap_row['output_cost_per_m']) / exp_row['output_cost_per_m'] * 100
            
            pair_label = f"{exp_row['name'].split(': ')[1]}\nvs\n{cheap_row['name'].split(': ')[1]}"
            
            # Add data for Expensive model
            plot_data.append({
                'Pair': pair_label,
                'Model': exp_row['name'].split(': ')[1],
                'Input Cost': exp_row['input_cost_per_m'],
                'Output Cost': exp_row['output_cost_per_m'],
                'Type': 'Expensive',
                'MMLU': exp_row['mmlu_score']
            })
            
            # Add data for Value Alternative
            plot_data.append({
                'Pair': pair_label,
                'Model': cheap_row['name'].split(': ')[1],
                'Input Cost': cheap_row['input_cost_per_m'],
                'Output Cost': cheap_row['output_cost_per_m'],
                'Type': 'Value Alternative',
                'MMLU': cheap_row['mmlu_score'],
                'Input Savings': f"{input_savings:.0f}%",
                'Output Savings': f"{output_savings:.0f}%"
            })

    if not plot_data:
        print("⚠️ No matching models found for savings comparison")
        return

    df_plot = pd.DataFrame(plot_data)
    
    # Plot with 2 subplots (Input vs Output)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sns.set_theme(style="whitegrid")
    
    # Palette
    palette = {'Expensive': '#FF5252', 'Value Alternative': '#4CAF50'}
    
    # Plot 1: Input Costs
    sns.barplot(data=df_plot, x='Pair', y='Input Cost', hue='Type', palette=palette, ax=ax1)
    ax1.set_title("Input Cost Comparison ($/M Tokens)", fontsize=14, weight='bold')
    ax1.set_ylabel("Cost ($)", fontsize=12)
    ax1.set_xlabel("")
    ax1.legend().remove() # Remove legend from first plot to avoid duplication
    
    # Add labels for Input
    for container in ax1.containers:
        ax1.bar_label(container, fmt='$%.2f', padding=3, fontsize=9, weight='bold')
        
    # Plot 2: Output Costs
    sns.barplot(data=df_plot, x='Pair', y='Output Cost', hue='Type', palette=palette, ax=ax2)
    ax2.set_title("Output Cost Comparison ($/M Tokens)", fontsize=14, weight='bold')
    ax2.set_ylabel("Cost ($)", fontsize=12)
    ax2.set_xlabel("")
    ax2.legend(title="Model Type", loc='upper right')
    
    # Add labels for Output
    for container in ax2.containers:
        ax2.bar_label(container, fmt='$%.2f', padding=3, fontsize=9, weight='bold')

    # Main Title
    plt.suptitle("The Savings Opportunity: Flagship Performance at a Fraction of the Cost\n(Comparing Input and Output Token Costs)", 
                 fontsize=18, weight='bold', color='#1A237E', y=0.98)
    
    # Add insight text
    insight_text = (
        "INSIGHT:\n"
        "Savings are consistent across both Input and Output tokens.\n"
        "Output tokens are typically 3-4x more expensive,\n"
        "making the absolute dollar savings even larger."
    )
    # Place insight on the figure
    fig.text(0.5, 0.02, insight_text, fontsize=12, ha='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='green'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15) # Make room for title and footer
    
    output_file = 'savings_potential.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved savings visualization to {output_file}")

if __name__ == "__main__":
    visualize_archetypes()
    visualize_savings()
