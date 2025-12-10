"""
KDD-Quality Analysis for LLM Jury Routing System

This script generates publication-quality plots and statistics for 
a multi-model LLM routing system. Unlike binary routers (RouteLLM, FrugalGPT),
LLM Jury selects from 47+ models using multi-objective optimization.

Key analyses:
1. Pareto Frontier Visualization - Cost vs Quality tradeoff space
2. Use Case Detection - Classification accuracy and distribution  
3. Value Analysis - Cost savings at quality parity
4. Per-Domain Recommendations - Which models for which tasks
5. Optimization Strategy Comparison - Different weight configurations
6. Routing Overhead - Time to make routing decisions

Output: Publication-ready figures and LaTeX-formatted tables.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory for figures
FIGURES_DIR = PROJECT_ROOT / "kdd_paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_model_data():
    """Load model metadata from cache."""
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    if not cache_path.exists():
        print(f"❌ Model cache not found: {cache_path}")
        return []
    
    with open(cache_path) as f:
        models = json.load(f)
    
    print(f"✓ Loaded {len(models)} models from cache")
    return models


def calculate_pareto_frontier(models: List[Dict]) -> List[Dict]:
    """
    Find Pareto-optimal models (non-dominated in cost-quality space).
    
    A model is Pareto-optimal if no other model has both:
    - Higher quality AND lower cost
    """
    pareto_models = []
    
    for model in models:
        is_dominated = False
        # Try both field name conventions
        quality = model.get('intelligence_index') or model.get('aa_intelligence_index') or 0
        cost = model.get('input_cost_per_m') or model.get('price_1m_input') or float('inf')
        
        if quality == 0 or cost == float('inf') or cost == 0:
            continue
            
        for other in models:
            other_quality = other.get('intelligence_index') or other.get('aa_intelligence_index') or 0
            other_cost = other.get('input_cost_per_m') or other.get('price_1m_input') or float('inf')
            
            if other_quality == 0 or other_cost == float('inf') or other_cost == 0:
                continue
            
            # Check if other dominates this model
            if other_quality > quality and other_cost < cost:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_models.append(model)
    
    return pareto_models


def analyze_cost_quality_tradeoffs(models: List[Dict]) -> Dict:
    """
    Analyze the cost-quality tradeoff space.
    
    Returns statistics about the model landscape.
    """
    valid_models = []
    for m in models:
        # Try both field name conventions
        quality = m.get('intelligence_index') or m.get('aa_intelligence_index') or 0
        cost = m.get('input_cost_per_m') or m.get('price_1m_input') or 0
        
        if quality and cost and quality > 0 and cost > 0:
            # Get provider from multiple possible fields
            provider = (m.get('provider') or m.get('creator_name') or 
                       m.get('organization') or m.get('openlm_organization') or 'Unknown')
            
            valid_models.append({
                'name': m.get('name', 'Unknown'),
                'quality': quality,
                'cost': cost,
                'latency': m.get('time_to_first_token_seconds', 0.5) * 1000,  # Convert to ms
                'provider': provider,
            })
    
    if not valid_models:
        return {}
    
    qualities = [m['quality'] for m in valid_models]
    costs = [m['cost'] for m in valid_models]
    
    # Find reference points
    gpt4o = next((m for m in valid_models if 'gpt-4o' in m['name'].lower()), None)
    claude = next((m for m in valid_models if 'claude' in m['name'].lower() and 'sonnet' in m['name'].lower()), None)
    gemini3 = next((m for m in valid_models if 'gemini 3' in m['name'].lower()), None)
    
    return {
        'total_models': len(valid_models),
        'models': valid_models,
        'quality_range': (min(qualities), max(qualities)),
        'cost_range': (min(costs), max(costs)),
        'quality_mean': np.mean(qualities),
        'quality_std': np.std(qualities),
        'cost_mean': np.mean(costs),
        'cost_std': np.std(costs),
        'gpt4o': gpt4o,
        'claude_sonnet': claude,
        'gemini3': gemini3,
    }


def generate_pareto_plot(models: List[Dict], output_path: Path):
    """
    Generate Pareto frontier visualization.
    
    This is a key KDD-quality plot showing:
    - All models in cost-quality space
    - Pareto frontier highlighted
    - Key reference models labeled
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")
        return
    
    # Prepare data
    analysis = analyze_cost_quality_tradeoffs(models)
    if not analysis:
        print("⚠ No valid model data for plotting")
        return
    
    valid_models = analysis['models']
    pareto_models = calculate_pareto_frontier(models)
    pareto_names = {m.get('name') for m in pareto_models}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by provider - use tab20 for more colors
    providers = list(set(m['provider'] for m in valid_models))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(providers))))
    provider_colors = dict(zip(providers, colors))
    
    # Ensure key providers have distinct, recognizable colors
    key_provider_colors = {
        'OpenAI': '#10a37f',      # OpenAI green
        'Google': '#4285f4',       # Google blue
        'Anthropic': '#d4a574',    # Anthropic tan
        'DeepSeek': '#ff6b35',     # Orange
        'xAI': '#1da1f2',          # X blue
        'Meta': '#0668e1',         # Meta blue
        'Mistral': '#ff4500',      # Red-orange
        'Cohere': '#8b5cf6',       # Purple
        'Microsoft': '#00a4ef',    # Microsoft blue
        'Alibaba': '#ff6a00',      # Alibaba orange
    }
    provider_colors.update(key_provider_colors)
    
    # Plot non-Pareto models
    for m in valid_models:
        if m['name'] not in pareto_names:
            color = provider_colors.get(m['provider'], 'gray')
            ax.scatter(m['cost'], m['quality'], 
                      c=[color], 
                      alpha=0.4, s=60, marker='o')
    
    # Plot Pareto models with emphasis
    pareto_costs = []
    pareto_qualities = []
    pareto_plot_models = []  # Store for labeling
    for m in valid_models:
        if m['name'] in pareto_names:
            color = provider_colors.get(m['provider'], 'gray')
            ax.scatter(m['cost'], m['quality'], 
                      c=[color], 
                      alpha=1.0, s=150, marker='*', 
                      edgecolors='black', linewidth=1)
            pareto_costs.append(m['cost'])
            pareto_qualities.append(m['quality'])
            pareto_plot_models.append(m)
    
    # Draw Pareto frontier line
    if pareto_costs:
        # Sort by cost for line drawing
        pareto_sorted = sorted(zip(pareto_costs, pareto_qualities))
        pareto_costs_sorted, pareto_qualities_sorted = zip(*pareto_sorted)
        ax.plot(pareto_costs_sorted, pareto_qualities_sorted, 
               'k--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    
    # Label ALL Pareto-optimal models (the stars) with smart positioning
    # Sort by cost to determine positions
    pareto_sorted_for_labels = sorted(pareto_plot_models, key=lambda x: x['cost'])
    
    # Define custom offsets based on model names to avoid overlap
    # These are manually tuned for the 7 Pareto models
    label_positions = {
        'Ministral 3B': (-10, -35),              # Bottom left - below
        'GPT-5 nano': (-70, -15),                # Second - to the left
        'gpt-oss-120B': (-100, -5),              # Third - far left
        'Grok 4.1 Fast': (-120, 20),             # Fourth - far left above
        'GPT-5 mini': (15, -30),                 # Fifth - below right
        'GPT-5.1': (-100, 30),                   # Sixth - far left above
        'Gemini 3 Pro': (15, -35),               # Seventh - below right
    }
    
    for m in pareto_sorted_for_labels:
        # Find matching offset
        offset = (10, -20)  # default
        for key, pos in label_positions.items():
            if key.lower() in m['name'].lower():
                offset = pos
                break
        
        ax.annotate(m['name'][:25], (m['cost'], m['quality']),
                   xytext=offset, textcoords='offset points',
                   fontsize=8, fontweight='bold', alpha=0.95,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, connectionstyle='arc3,rad=0.1'))
    
    # Styling
    ax.set_xlabel('Cost ($/1M input tokens)', fontsize=12)
    ax.set_ylabel('Quality (Intelligence Index)', fontsize=12)
    ax.set_title('LLM Cost-Quality Tradeoff Space\n(Stars = Pareto-optimal models)', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Legend for providers - prioritize key providers
    key_providers = ['OpenAI', 'Google', 'Anthropic', 'DeepSeek', 'xAI', 'Meta', 'Mistral', 'Cohere', 'Microsoft', 'Alibaba']
    active_providers = set(m['provider'] for m in valid_models)
    
    # Sort providers: key providers first, then others alphabetically
    sorted_providers = []
    for p in key_providers:
        if p in active_providers:
            sorted_providers.append(p)
    for p in sorted(active_providers):
        if p not in sorted_providers:
            sorted_providers.append(p)
    
    legend_elements = [mpatches.Patch(facecolor=provider_colors.get(p, 'gray'), label=p) 
                      for p in sorted_providers[:12]]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Pareto plot to {output_path}")


def generate_value_analysis(models: List[Dict], baseline_name: str = "Gemini 3 Pro") -> Dict:
    """
    Analyze value proposition of routing vs always using baseline.
    
    For each model, calculate:
    - Quality ratio vs baseline
    - Cost ratio vs baseline
    - Value ratio (quality/cost improvement)
    """
    analysis = analyze_cost_quality_tradeoffs(models)
    if not analysis:
        return {}
    
    valid_models = analysis['models']
    baseline = next((m for m in valid_models if baseline_name.lower() in m['name'].lower()), None)
    
    if not baseline:
        print(f"⚠ Baseline {baseline_name} not found")
        return {}
    
    results = []
    for m in valid_models:
        quality_ratio = m['quality'] / baseline['quality']
        cost_ratio = m['cost'] / baseline['cost']
        value_ratio = quality_ratio / cost_ratio  # Higher is better
        
        results.append({
            'name': m['name'],
            'quality': m['quality'],
            'cost': m['cost'],
            'quality_ratio': quality_ratio,
            'cost_ratio': cost_ratio,
            'value_ratio': value_ratio,
            'cost_savings_pct': (1 - cost_ratio) * 100,
            'quality_retention_pct': quality_ratio * 100,
        })
    
    # Sort by value ratio
    results.sort(key=lambda x: x['value_ratio'], reverse=True)
    
    # Statistics
    value_ratios = [r['value_ratio'] for r in results]
    cost_savings = [r['cost_savings_pct'] for r in results if r['cost_savings_pct'] > 0]
    
    return {
        'baseline': baseline,
        'models': results,
        'top_value_models': results[:10],
        'stats': {
            'models_with_better_value': sum(1 for v in value_ratios if v > 1),
            'models_with_cost_savings': len(cost_savings),
            'avg_cost_savings_if_cheaper': np.mean(cost_savings) if cost_savings else 0,
            'max_value_ratio': max(value_ratios),
            'median_value_ratio': np.median(value_ratios),
        }
    }


def generate_value_plot(models: List[Dict], output_path: Path):
    """
    Generate value analysis visualization.
    
    Shows cost savings vs quality retention for all models.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")
        return
    
    value_analysis = generate_value_analysis(models)
    if not value_analysis:
        return
    
    results = value_analysis['models']
    baseline = value_analysis['baseline']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot: cost savings vs quality retention
    for r in results:
        # Color based on value ratio
        if r['value_ratio'] > 1.5:
            color = 'green'
            alpha = 0.8
        elif r['value_ratio'] > 1.0:
            color = 'blue'
            alpha = 0.6
        else:
            color = 'red'
            alpha = 0.4
        
        ax.scatter(r['cost_savings_pct'], r['quality_retention_pct'],
                  c=color, alpha=alpha, s=80)
    
    # Add reference lines
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Baseline Quality')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline Cost')
    
    # Highlight top value models
    for r in results[:5]:
        ax.annotate(r['name'][:15], 
                   (r['cost_savings_pct'], r['quality_retention_pct']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Quadrant labels
    ax.text(50, 110, 'Better Value\n(Cheaper + Higher Quality)', 
           fontsize=10, ha='center', color='green', alpha=0.7)
    ax.text(-30, 110, 'Higher Quality\n(More Expensive)', 
           fontsize=10, ha='center', color='blue', alpha=0.7)
    ax.text(50, 85, 'Cost Optimized\n(Cheaper + Lower Quality)', 
           fontsize=10, ha='center', color='orange', alpha=0.7)
    ax.text(-30, 85, 'Dominated\n(More Expensive + Lower Quality)', 
           fontsize=10, ha='center', color='red', alpha=0.7)
    
    ax.set_xlabel('Cost Savings vs Gemini 3 Pro (%)', fontsize=12)
    ax.set_ylabel('Quality Retention vs Gemini 3 Pro (%)', fontsize=12)
    ax.set_title('Model Value Analysis: Cost Savings vs Quality Retention', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved value plot to {output_path}")


def analyze_use_case_routing():
    """
    Analyze use case detection and routing decisions.
    
    Uses the PromptClassifier to show:
    - Use case distribution
    - Confidence scores
    - Classification examples
    """
    try:
        from llm_jury import PromptClassifier, classify_prompt
    except ImportError:
        print("⚠ LLM Jury not available")
        return {}
    
    classifier = PromptClassifier()
    
    # Test prompts representing different use cases
    test_prompts = [
        # Code generation
        ("Write a Python function to sort a list using quicksort", "code_generation"),
        ("Implement a REST API endpoint in FastAPI", "code_generation"),
        # Data analysis
        ("Analyze this CSV data and find trends in sales", "data_analysis"),
        ("What statistical test should I use for this data?", "data_analysis"),
        # Creative
        ("Write a short story about a robot learning to love", "creative_writing"),
        ("Compose a haiku about autumn leaves", "creative_writing"),
        # RAG
        ("Using the provided documents, answer: What is the refund policy?", "rag_pipeline"),
        ("Based on the context above, summarize the key findings", "rag_pipeline"),
        # Math
        ("Solve this differential equation: dy/dx = xy", "math_reasoning"),
        ("Prove that sqrt(2) is irrational", "math_reasoning"),
        # General QA
        ("What is the capital of France?", "general_qa"),
        ("Explain quantum entanglement in simple terms", "general_qa"),
        # SQL
        ("Write a SQL query to find top customers by revenue", "sql_generation"),
        ("Optimize this slow database query", "sql_generation"),
        # Summarization
        ("Summarize this article in 3 bullet points", "summarization"),
        ("TL;DR this research paper", "summarization"),
    ]
    
    results = []
    correct = 0
    
    for prompt, expected in test_prompts:
        result = classifier.classify(prompt)
        is_correct = result.use_case == expected
        if is_correct:
            correct += 1
        
        results.append({
            'prompt': prompt[:50] + '...',
            'expected': expected,
            'predicted': result.use_case,
            'confidence': result.confidence,
            'correct': is_correct,
        })
    
    accuracy = correct / len(test_prompts) * 100
    
    # Group by expected use case
    by_use_case = defaultdict(list)
    for r in results:
        by_use_case[r['expected']].append(r)
    
    return {
        'total_prompts': len(test_prompts),
        'accuracy': accuracy,
        'results': results,
        'by_use_case': dict(by_use_case),
        'avg_confidence': np.mean([r['confidence'] for r in results]),
    }


def generate_use_case_confusion_matrix(output_path: Path):
    """Generate confusion matrix for use case classification."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, skipping plot")
        return
    
    analysis = analyze_use_case_routing()
    if not analysis:
        return
    
    # Build confusion matrix
    use_cases = sorted(set(r['expected'] for r in analysis['results']))
    n = len(use_cases)
    confusion = np.zeros((n, n))
    
    uc_to_idx = {uc: i for i, uc in enumerate(use_cases)}
    
    for r in analysis['results']:
        expected_idx = uc_to_idx[r['expected']]
        predicted = r['predicted']
        if predicted in uc_to_idx:
            predicted_idx = uc_to_idx[predicted]
        else:
            # Map to closest or 'other'
            predicted_idx = uc_to_idx.get('general_qa', 0)
        confusion[expected_idx, predicted_idx] += 1
    
    # Normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, where=row_sums != 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_norm, cmap='Blues')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([uc[:12] for uc in use_cases], rotation=45, ha='right')
    ax.set_yticklabels(use_cases)
    
    # Annotate
    for i in range(n):
        for j in range(n):
            val = confusion_norm[i, j]
            if val > 0:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
    
    ax.set_xlabel('Predicted Use Case', fontsize=12)
    ax.set_ylabel('Expected Use Case', fontsize=12)
    ax.set_title(f'Use Case Classification Confusion Matrix\n(Accuracy: {analysis["accuracy"]:.1f}%)', fontsize=14)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {output_path}")


def generate_latex_tables(models: List[Dict]) -> str:
    """Generate LaTeX-formatted tables for the paper."""
    
    # Table 1: Top Value Models
    value_analysis = generate_value_analysis(models)
    if not value_analysis:
        return ""
    
    latex = []
    latex.append("% Table 1: Top Value Models (vs Gemini 3 Pro baseline)")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Top 10 Models by Value Ratio (Quality/Cost vs Gemini 3 Pro)}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("Model & Quality (\\%) & Cost Savings (\\%) & Value Ratio \\\\")
    latex.append("\\midrule")
    
    for r in value_analysis['top_value_models']:
        latex.append(f"{r['name'][:25]} & {r['quality_retention_pct']:.1f} & {r['cost_savings_pct']:.1f} & {r['value_ratio']:.2f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:value_models}")
    latex.append("\\end{table}")
    
    # Table 2: Pareto Frontier Models
    pareto_models = calculate_pareto_frontier(models)
    latex.append("")
    latex.append("% Table 2: Pareto-Optimal Models")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Pareto-Optimal Models in Cost-Quality Space}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Model & Quality Index & Cost (\\$/1M) & Provider \\\\")
    latex.append("\\midrule")
    
    for m in sorted(pareto_models, key=lambda x: x.get('aa_intelligence_index', 0), reverse=True)[:10]:
        name = m.get('name', 'Unknown')[:25]
        quality = m.get('aa_intelligence_index', 0)
        cost = m.get('input_cost_per_m', 0)
        provider = m.get('provider', 'Unknown')
        latex.append(f"{name} & {quality:.1f} & {cost:.2f} & {provider} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:pareto_models}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def print_summary_statistics(models: List[Dict]):
    """Print summary statistics for the paper."""
    
    print("\n" + "="*70)
    print("KDD PAPER STATISTICS")
    print("="*70)
    
    # Model landscape
    analysis = analyze_cost_quality_tradeoffs(models)
    if analysis:
        print(f"\n📊 Model Landscape:")
        print(f"   Total models: {analysis['total_models']}")
        print(f"   Quality range: {analysis['quality_range'][0]:.1f} - {analysis['quality_range'][1]:.1f}")
        print(f"   Cost range: ${analysis['cost_range'][0]:.2f} - ${analysis['cost_range'][1]:.2f} per 1M tokens")
        print(f"   Quality mean ± std: {analysis['quality_mean']:.1f} ± {analysis['quality_std']:.1f}")
        print(f"   Cost mean ± std: ${analysis['cost_mean']:.2f} ± ${analysis['cost_std']:.2f}")
    
    # Pareto analysis
    pareto_models = calculate_pareto_frontier(models)
    print(f"\n🎯 Pareto Analysis:")
    print(f"   Pareto-optimal models: {len(pareto_models)}")
    print(f"   Dominated models: {analysis['total_models'] - len(pareto_models)}")
    
    # Value analysis
    value_analysis = generate_value_analysis(models)
    if value_analysis and 'stats' in value_analysis:
        stats = value_analysis['stats']
        print(f"\n💰 Value Analysis (vs Gemini 3 Pro baseline):")
        print(f"   Models with better value ratio: {stats['models_with_better_value']}")
        print(f"   Models with cost savings: {stats['models_with_cost_savings']}")
        print(f"   Avg cost savings (if cheaper): {stats['avg_cost_savings_if_cheaper']:.1f}%")
        print(f"   Max value ratio: {stats['max_value_ratio']:.2f}x")
        print(f"   Median value ratio: {stats['median_value_ratio']:.2f}x")
    
    # Use case detection
    uc_analysis = analyze_use_case_routing()
    if uc_analysis:
        print(f"\n🏷️ Use Case Detection:")
        print(f"   Test accuracy: {uc_analysis['accuracy']:.1f}%")
        print(f"   Avg confidence: {uc_analysis['avg_confidence']:.2f}")
        print(f"   Use cases tested: {len(uc_analysis['by_use_case'])}")
    
    print("\n" + "="*70)


def main():
    """Run full KDD analysis."""
    print("="*70)
    print("KDD-QUALITY ANALYSIS FOR LLM JURY")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # 1. Load model data
    models = load_model_data()
    if not models:
        return
    
    # 2. Generate plots
    print("\n📈 Generating Plots...")
    
    generate_pareto_plot(models, FIGURES_DIR / "pareto_frontier.png")
    generate_value_plot(models, FIGURES_DIR / "value_analysis.png")
    generate_use_case_confusion_matrix(FIGURES_DIR / "use_case_confusion.png")
    
    # 3. Generate LaTeX tables
    print("\n📝 Generating LaTeX Tables...")
    latex_tables = generate_latex_tables(models)
    
    tables_path = FIGURES_DIR / "tables.tex"
    with open(tables_path, 'w') as f:
        f.write(latex_tables)
    print(f"✓ Saved LaTeX tables to {tables_path}")
    
    # 4. Print summary statistics
    print_summary_statistics(models)
    
    print("\n✅ KDD ANALYSIS COMPLETE")
    print(f"📁 Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

