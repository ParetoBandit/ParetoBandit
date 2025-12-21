#!/usr/bin/env python3
"""
Visualize the routing decisions from the full 81-model BanditRouter run.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the selections
selections_path = Path(__file__).parent / "full_81_models_selections.json"
with open(selections_path, "r") as f:
    data = json.load(f)

selections = data["selections"]  # List of model names
total_rounds = len(selections)

# Load the benchmark data to get rewards
rewards_path = Path("/Users/annette/repostitories/llm_jury/banditgpt/data/priors/archetype_grid_dense_run.jsonl")

# Load all rewards (costs are uniform at 1.0 in this simulation)
model_rewards = {}

with open(rewards_path, "r") as f:
    for line in f:
        record = json.loads(line)
        if not record.get("ok", False):
            continue
            
        model_id = record["model_id"]
        logit = record.get("reward_logit", 0.0)
        reward = 1.0 / (1.0 + np.exp(-logit))  # Convert logit to probability
        
        if model_id not in model_rewards:
            model_rewards[model_id] = []
        model_rewards[model_id].append(reward)

# Calculate average rewards per model
model_avg_rewards = {model: np.mean(rewards) for model, rewards in model_rewards.items()}

# Assign realistic costs based on model size/tier ($/1M tokens input)
# Reference: Typical API pricing as of Dec 2024
def get_model_cost(model_name):
    """Estimate realistic cost based on model name and size."""
    model_lower = model_name.lower()
    
    # Frontier models (most expensive)
    if any(x in model_lower for x in ['claude-opus-4', 'gpt-5', 'gemini-ultra', 'grok-3']):
        return 15.0
    if any(x in model_lower for x in ['claude-sonnet-4.5', 'gpt-4.1', 'gemini-3-pro']):
        return 10.0
    if any(x in model_lower for x in ['claude-opus', 'gpt-4o', 'claude-3.5-sonnet']):
        return 5.0
    
    # Very large models (400B+)
    if any(x in model_lower for x in ['405b', '400b']):
        return 3.0
    
    # Large models (70B-200B)
    if any(x in model_lower for x in ['mixtral-8x22b', 'llama-3-70b', 'llama-3.1-70b', '90b']):
        return 1.0
    if any(x in model_lower for x in ['mistral-large', 'llama-3.3', 'qwq']):
        return 1.5
    
    # Medium models (20-50B)
    if any(x in model_lower for x in ['mistral-small', 'phi-4', 'nemotron']):
        return 0.4
    if any(x in model_lower for x in ['command-r-plus', 'llama-4-scout']):
        return 0.5
    
    # Small models (7B-20B)
    if any(x in model_lower for x in ['gpt-4.1-mini', 'gpt-4o-mini', 'claude-haiku', 'gemma-3']):
        return 0.15
    if any(x in model_lower for x in ['command-r', 'mistral-medium']):
        return 0.2
    
    # Very small models (<7B)
    if any(x in model_lower for x in ['ministral-3b', 'nova-lite', 'nova-micro', 'gemma-2-2b']):
        return 0.06
    
    # Default for medium-tier models
    return 0.5

model_costs = {model: get_model_cost(model) for model in model_avg_rewards.keys()}

# Count selections per model
model_counts = {}
for model in selections:
    model_counts[model] = model_counts.get(model, 0) + 1

# Sort by frequency
sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)

# Create figure with 3 panels
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 1, hspace=0.4)

# ============================================================================
# Panel A: Model Selection Over Time (Top 5 models only)
# ============================================================================
ax1 = fig.add_subplot(gs[0])

top_5_models = [m[0] for m in sorted_models[:5]]
window_size = 50

# Create rolling window data
time_steps = np.arange(window_size, total_rounds + 1)
model_proportions = {model: [] for model in top_5_models}
model_proportions["Other"] = []

for t in time_steps:
    window_data = selections[t - window_size:t]  # window_data is a list of model names
    window_counts = {model: 0 for model in top_5_models}
    other_count = 0
    
    for model_name in window_data:
        if model_name in top_5_models:
            window_counts[model_name] += 1
        else:
            other_count += 1
    
    for model in top_5_models:
        model_proportions[model].append(window_counts[model] / window_size)
    model_proportions["Other"].append(other_count / window_size)

# Create stacked area chart
colors = plt.cm.tab10(np.linspace(0, 1, 6))
bottom = np.zeros(len(time_steps))

for i, model in enumerate(top_5_models + ["Other"]):
    ax1.fill_between(
        time_steps,
        bottom,
        bottom + model_proportions[model],
        label=model,
        alpha=0.8,
        color=colors[i]
    )
    bottom += model_proportions[model]

ax1.set_xlabel("Routing Decision", fontsize=12, fontweight='bold')
ax1.set_ylabel("Model Selection Proportion", fontsize=12, fontweight='bold')
ax1.set_title(f"Panel A: Model Selection Over Time (Top 5 + Other, {window_size}-step rolling window)", 
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3)

# ============================================================================
# Panel B: Model Selection Heatmap (All 800 decisions, showing top 10 models)
# ============================================================================
ax2 = fig.add_subplot(gs[1])

top_10 = sorted_models[:10]
top_10_names = [m[0] for m in top_10]

# Create a matrix: rows = top 10 models, columns = decision index
# We'll show every 8th decision to fit 100 columns (800/8 = 100)
sample_rate = 8
n_samples = total_rounds // sample_rate
heatmap_data = np.zeros((len(top_10_names), n_samples))

for col_idx in range(n_samples):
    decision_idx = col_idx * sample_rate
    if decision_idx < len(selections):
        model_name = selections[decision_idx]
        if model_name in top_10_names:
            row_idx = top_10_names.index(model_name)
            heatmap_data[row_idx, col_idx] = 1

# Shorten model names for y-axis
short_names = []
for name in top_10_names:
    if '/' in name:
        name = name.split('/')[-1]
    if len(name) > 25:
        name = name[:22] + '...'
    short_names.append(name)

# Create heatmap
im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')

# Set ticks
ax2.set_yticks(range(len(short_names)))
ax2.set_yticklabels(short_names, fontsize=9)
ax2.set_xlabel(f"Routing Decision (every {sample_rate}th shown, total={total_rounds})", fontsize=12, fontweight='bold')
ax2.set_title(f"Panel B: Model Selection Pattern (Top 10 models, all {total_rounds} decisions)", 
              fontsize=13, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Selected', rotation=270, labelpad=15, fontweight='bold')

# ============================================================================
# Panel C: Cost vs. Quality Scatter (Top 10 models)
# ============================================================================
ax3 = fig.add_subplot(gs[2])

# Get top 10 models
top_10 = sorted_models[:10]
top_10_names = [m[0] for m in top_10]

costs = []
avg_rewards = []
counts = []
labels = []

for model in top_10_names:
    # Get cost and average reward from benchmark data
    cost = model_costs.get(model, 1.0)
    avg_reward = model_avg_rewards.get(model, 0.5)
    count = model_counts[model]
    
    costs.append(cost)
    avg_rewards.append(avg_reward)
    counts.append(count)
    
    # Shorten label
    label = model
    if '/' in label:
        label = label.split('/')[-1]
    if len(label) > 25:
        label = label[:22] + '...'
    labels.append(label)

# Normalize bubble sizes
max_count = max(counts)
sizes = [1000 * (c / max_count) for c in counts]

# Create scatter plot
scatter = ax3.scatter(costs, avg_rewards, s=sizes, alpha=0.6, 
                     c=range(len(costs)), cmap='tab10', edgecolors='black', linewidth=1)

# Add labels
for i, (x, y, label, count) in enumerate(zip(costs, avg_rewards, labels, counts)):
    ax3.annotate(f"{label}\n(n={count})", 
                (x, y), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

ax3.set_xlabel("Cost ($/1M tokens)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Average Reward", fontsize=12, fontweight='bold')
ax3.set_title("Panel C: Cost vs. Quality Trade-off (Top 10 models, bubble size = usage)", 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)

# Save figure
output_path = Path(__file__).parent / "routing_analysis_81_models.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("MODEL SELECTION SUMMARY")
print("="*80)
print(f"\nTotal routing decisions: {total_rounds}")
print(f"Unique models selected: {len(model_counts)}")
print(f"Models never selected: {81 - len(model_counts)}")

print("\n" + "-"*80)
print("TOP 10 MODELS:")
print("-"*80)
for i, (model, count) in enumerate(sorted_models[:10], 1):
    pct = 100 * count / total_rounds
    avg_reward = model_avg_rewards.get(model, 0.5)
    cost = model_costs.get(model, 1.0)
    print(f"{i:2d}. {model:45s} {count:4d} ({pct:5.1f}%) | Avg Reward: {avg_reward:.3f} | Cost: ${cost:.2f}/1M")

print("\n" + "-"*80)
print("COST EFFECTIVENESS INSIGHTS:")
print("-"*80)

# Calculate weighted average cost
total_cost = sum(model_costs.get(model, 1.0) for model in selections)
avg_cost = total_cost / total_rounds
print(f"Average cost per request: ${avg_cost:.4f}/1M tokens")

# Calculate if we always used the most expensive model
most_expensive_cost = max(model_costs.values())
print(f"Cost if always using most expensive model: ${most_expensive_cost:.4f}/1M tokens")

savings = 100 * (1 - avg_cost / most_expensive_cost)
print(f"Cost savings: {savings:.1f}%")

# Calculate weighted average reward
weighted_reward = 0
for model in selections:
    weighted_reward += model_avg_rewards.get(model, 0.5)
avg_reward_all = weighted_reward / total_rounds
print(f"\nAverage reward achieved: {avg_reward_all:.3f}")

plt.show()

