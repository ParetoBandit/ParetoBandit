import pandas as pd
import numpy as np
from datasets import load_dataset

# --- STEP 1: LOAD THE GOLDEN DATASET ---
print("Loading RouterBench...")
# We use the 'train' split for simulation to get enough data points
# In your paper, use 'test' or 'validation' splits rigorously
ds = load_dataset("withmartian/routerbench", split="train[:10000]") # Limit for demo speed
df = ds.to_pandas()

# RouterBench Structure:
# Each row is a specific model's response to a prompt.
# Columns: 'prompt', 'model', 'cost', 'score' (0.0 to 1.0)
# We need to pivot this so each row is a PROMPT, and columns are model details.
print("Pivoting data (this allows us to 'choose' models)...")
df_pivot = df.pivot(index='prompt', columns='model', values=['cost', 'score'])

# Helper: Get cost/score for a specific model on a specific prompt
def get_outcome(prompt_row, model_name):
    cost = prompt_row[('cost', model_name)]
    score = prompt_row[('score', model_name)]
    return cost, score

# --- STEP 2: DEFINE THE ROUTERS ---

# A. The Baseline: RouteLLM (Simulated via Thresholding)
# RouteLLM typically uses a "Router Model" to predict a score. 
# Here we simulate its "outcome" by assuming it routes based on a "Predicted Win Rate".
# For a pure simulation, we can assume a "Strong" vs "Weak" binary choice often used.
def routellm_baseline(row, threshold, strong_model="gpt-4", weak_model="mixtral-8x7b"):
    # Real RouteLLM predicts this. We simulate a "Difficulty Proxy" using prompt length 
    # or a mock 'classifier_score' you would generate.
    # IN PAPER: You would load actual RouteLLM predictions here.
    difficulty_proxy = len(str(row.name)) / 1000.0 # Mock difficulty (0.0 to 1.0)
    
    if difficulty_proxy > threshold:
        return get_outcome(row, strong_model)
    else:
        return get_outcome(row, weak_model)

# B. Your Method: The "Optimization" Router
# This is where your fancy math goes.
def my_optimization_router(row, risk_tolerance, hallucination_priors):
    # Example Logic: 
    # Minimize Cost s.t. Expected_Score > X AND Hallucination_Risk < Y
    
    best_model = None
    min_cost = float('inf')
    
    # Iterate through ALL available models (not just strong/weak)
    # This is your "Optimization" advantage: you consider the full portfolio.
    available_models = ["gpt-4", "mixtral-8x7b", "llama-3-70b", "haiku"]
    
    for model in available_models:
        # Check if model data exists for this prompt
        if model not in row['cost'].index: continue
        
        cost = row[('cost', model)]
        
        # CONSTRAINT: Hallucination Check
        # Using the "Priors" dictionary you built from Artificial Analysis
        h_rate = hallucination_priors.get(model, 1.0)
        if h_rate > risk_tolerance:
            continue # Skip this model, it's too risky
            
        # OBJECTIVE: Cheapest model that passes the check
        if cost < min_cost:
            min_cost = cost
            best_model = model
            
    # Fallback to safest model if all fail
    if best_model is None:
        best_model = "gpt-4"
        
    return get_outcome(row, best_model)

# --- STEP 3: RUN THE SIMULATION ---
print("Running Simulation...")

# Mock Hallucination Priors (From Artificial Analysis / Your External Data)
h_priors = {
    "gpt-4": 0.01,
    "llama-3-70b": 0.05,
    "mixtral-8x7b": 0.08,
    "haiku": 0.10
}

results = []

# Sweep thresholds to generate the Pareto Curve
thresholds = np.linspace(0, 1, 10) # 0.0, 0.1 ... 1.0

for t in thresholds:
    # 1. Evaluate Baseline
    base_costs, base_scores = [], []
    for _, row in df_pivot.iterrows():
        c, s = routellm_baseline(row, t)
        base_costs.append(c)
        base_scores.append(s)
    
    results.append({
        "method": "RouteLLM",
        "param": t,
        "avg_cost": np.mean(base_costs),
        "avg_score": np.mean(base_scores)
    })

    # 2. Evaluate Your Method
    my_costs, my_scores = [], []
    for _, row in df_pivot.iterrows():
        # Note: Your param 't' might map to 'risk_tolerance' differently
        c, s = my_optimization_router(row, risk_tolerance=t, hallucination_priors=h_priors)
        my_costs.append(c)
        my_scores.append(s)

    results.append({
        "method": "My_Optimizer",
        "param": t,
        "avg_cost": np.mean(my_costs),
        "avg_score": np.mean(my_scores)
    })

print(pd.DataFrame(results))
# Save this dataframe to CSV -> This is the input for your "Money Plot"