#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
# Production Weights
LAMBDA_COST = 1.42
LAMBDA_LATENCY = 0.1

# The 8-Model "Pareto Frontier" (Acc vs Cost vs Risk vs Latency)
PARETO_MODELS = [
    "google/gemma-3-1b-it",             # Ultra-Cheap & Fast
    "meta-llama/llama-3.2-1b-instruct", # Low Latency
    "deepseek/deepseek-r1-0528-qwen3-8b", # Cost-Efficiency King
    "openai/gpt-oss-20b",               # Balanced / SafeWrapper
    "google/gemini-2.0-flash-001",      # General Purpose
    "openai/gpt-4o",                    # Safety Expert
    "x-ai/grok-3",                      # Premium Safety
    "anthropic/claude-sonnet-4.5"       # Reasoning Powerhouse
]

# Color Map for Consistency
COLOR_MAP = {
    "google/gemma-3-1b-it": "#1f77b4", # blue
    "deepseek/deepseek-r1-0528-qwen3-8b": "#ff7f0e", # orange
    "openai/gpt-oss-20b": "#2ca02c", # green 
    "openai/gpt-4o": "#d62728", # red
    "x-ai/grok-3": "#9467bd", # purple
    "meta-llama/llama-3.2-1b-instruct": "#8c564b", # brown
    "google/gemini-2.0-flash-001": "#e377c2", # pink
    "anthropic/claude-sonnet-4.5": "#7f7f7f" # gray
}

def load_data():
    # Script is in final_release/kdd_paper/
    # models.json is in banditgpt/models.json
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "banditgpt" / "models.json") as f:
        data = json.load(f)
    
    registry = {m["openrouter_id"]: m for m in data["models"]}
    subset = []
    
    for pid in PARETO_MODELS:
        if pid in registry:
            subset.append(registry[pid])
            
    return subset

def get_utility_components(models):
    """
    Calculate normalized components for all models.
    Returns DataFrame with columns: [Name, Quality, NormCost, NormLat, RiskFactor, Base_Value]
    """
    # 1. Extract raw values
    raw_q = [] # Quality (HLE)
    raw_c = [] # Cost ($/1m) 
    raw_l = [] # Latency (s)
    raw_r = [] # Risk (%)
    names = []
    ids = []
    
    for m in models:
        names.append(m.get("display_name", m["openrouter_id"]).split("/")[-1])
        ids.append(m["openrouter_id"])
        
        # Quality: Use HLE as proxy for UCB-Quality
        # Shifted to be > 0 roughly for visualization
        raw_q.append(m.get("hle", 0.1) * 100) 
        
        # Cost
        raw_c.append(m.get("price_1m_blended", 1.0))
        
        # Latency
        raw_l.append(m.get("time_to_first_token_seconds", 1.0))
        
        # Risk (Percentage)
        # Note: In utility, it is (Rate/100) * lambda
        # Here we keep as rate (0-100) for easier x-axis interpretation
        raw_r.append(m.get("hallucination_composite", 8.0))

    # 2. Normalize Log Params (Matches Bandit Logic)
    # Log transform
    log_c = np.log(np.maximum(raw_c, 1e-9))
    log_l = np.log(np.maximum(raw_l, 1e-9))
    
    # Min-Max Normalization
    min_c, range_c = log_c.min(), (log_c.max() - log_c.min())
    if range_c == 0: range_c = 1.0
    
    min_l, range_l = log_l.min(), (log_l.max() - log_l.min())
    if range_l == 0: range_l = 1.0
    
    norm_c = (log_c - min_c) / range_c
    norm_l = (log_l - min_l) / range_l
    
    # 3. Calculate Base Value
    # Base = Quality - (w_c * NormCost) - (w_l * NormLatency)
    # Scaled to look nice on plot (Bandit uses 0-1 range roughly, we multiply for readability)
    
    # Actually, let's stick to raw bandit scale for correctness
    # Quality ~ [0, 1] (HLE) 
    # Cost/Lat ~ [0, 1]
    
    base_values = []
    risk_factors = [] # This is 'Rate / 100' so lambda scales naturally
    
    for i in range(len(models)):
        q = models[i].get("hle", 0.0) 
        # Boost Quality magnitude to match production UCB magnitude (~0.5 - 2.0 with priors)
        # Let's assume sigmoided prior + alpha * sigma -> roughly [0.5, 0.9]
        # For viz, raw HLE is fine.
        
        cost_penalty = LAMBDA_COST * norm_c[i]
        lat_penalty = LAMBDA_LATENCY * norm_l[i]
        
        base = q - cost_penalty - lat_penalty
        
        # Risk Factor: The value multiplied by lambda_risk
        # Penalty = (Rate/100) * lambda
        # So factor = Rate/100
        r_factor = raw_r[i] / 100.0
        
        base_values.append(base)
        risk_factors.append(r_factor)
        
    df = pd.DataFrame({
        "ID": ids,
        "Name": names,
        "Base_Value": base_values,
        "Risk_Factor": risk_factors,
        "Risk_Display": raw_r # 0-100%
    })
    return df

def main():
    models = load_data()
    df = get_utility_components(models)
    
    # Setup Lambdas
    lambdas = np.linspace(0, 100, 500)
    
    # Plotting
    plt.figure(figsize=(16, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- PLOT 1: Crossover / Rank Reversal ---
    ax1 = plt.subplot(1, 2, 1)
    
    # Identify "winners" to bold them
    # For every lambda, find max utility
    winners = set()
    for l in np.linspace(0, 100, 20):
        utils = df["Base_Value"] - (df["Risk_Factor"] * l)
        winners.add(df.iloc[utils.idxmax()]["ID"])
        
    for index, row in df.iterrows():
        # U = Base - Lambda * (Risk/100)
        utility_scores = row['Base_Value'] - (lambdas * row['Risk_Factor'])
        
        mid = row['ID']
        color = COLOR_MAP.get(mid, 'gray')
        
        # Highlight winners
        alpha = 1.0 if mid in winners else 0.3
        width = 3.0 if mid in winners else 1.0
        label = row['Name'] if mid in winners else None
        
        ax1.plot(lambdas, utility_scores, label=label, color=color, linewidth=width, alpha=alpha)

    ax1.set_title('Rank Reversal: Utility vs. $\lambda_{risk}$', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Risk Aversion ($\lambda_{risk}$)', fontsize=12)
    ax1.set_ylabel('Total Utility Score', fontsize=12)
    
    # Add Critical Threshold Lines
    # Manually identified phase transitions from Table 2
    ax1.axvline(x=12.5, color='gray', linestyle='--', alpha=0.5)
    ax1.text(13.5, ax1.get_ylim()[0], 'Phase 1: Shift to Mid', rotation=90, color='gray')
    
    ax1.axvline(x=60.0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(61.0, ax1.get_ylim()[0], 'Phase 2: Shift to Safety', rotation=90, color='gray')

    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # --- PLOT 2: Efficient Frontier ---
    # X: Risk Score (Lower better)
    # Y: Base Value (Higher better)
    ax2 = plt.subplot(1, 2, 2)
    
    for index, row in df.iterrows():
        mid = row['ID']
        color = COLOR_MAP.get(mid, 'gray')
        alpha = 1.0 if mid in winners else 0.3
        size = 150 if mid in winners else 50
        
        ax2.scatter(row['Risk_Display'], row['Base_Value'], s=size, c=color, alpha=alpha, edgecolors='white')
        
        # Annotate winners
        if mid in winners:
             ax2.annotate(row['Name'], (row['Risk_Display'], row['Base_Value']), 
                 xytext=(8, -5), textcoords='offset points', fontsize=10, fontweight='bold')

    ax2.set_title('Evidence of Trade-offs: Base Value vs. Risk', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hallucination Risk % (Lower is Better)', fontsize=12)
    ax2.set_ylabel('Base Utility (Quality - Cost - Latency)', fontsize=12)
    ax2.invert_xaxis() # Lower risk is better, so move 0 to right or keep standard? 
    # Ideally standard: 0 on left. User said "Slope lambda". 
    # Let's keep 0 on left. No invert. 
    # Actually user said "Candidate touches last as you move line up and left". 
    # So Left (0) is good. 
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = Path(__file__).parent / "figure_rank_reversal.png"
    plt.savefig(out_path, dpi=300)
    print(f"Generated plot: {out_path}")

if __name__ == "__main__":
    main()
