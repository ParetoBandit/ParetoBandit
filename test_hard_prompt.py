
import numpy as np
import json
from pathlib import Path
from final_release.bandit import BanditRouter, l2_normalize, OptimizationProfile

# ==============================================================================
# Helper to simulate "Real" performance lookup
# ==============================================================================
def get_simulated_metrics(model_id, domain, registry):
    """
    Returns the ground-truth accuracy and cost for a model from the registry.
    In a real system, this would come from a Judge or Reward Model.
    """
    if model_id not in registry:
        return 0.0, 100.0
    
    # Get Metadata
    meta = registry[model_id]
    acc = meta.get(f"acc_{domain}", 0.5)
    cost = meta.get("price_1m_blended", 1.0) # Cost per 1M tokens
    lat = meta.get("time_to_first_token_seconds", 0.5)
    
    return acc, cost, lat

# ==============================================================================
# Main Analysis
# ==============================================================================
def analyze_tough_prompt():
    print("Initializing Bandit...")
    # Load defaults (alpha=1.0, etc.)
    bandit_router = BanditRouter.create()
    
    # 1. ESTABLISH THE "TOUGH" SCENARIO
    # A complex Math/Reasoning prompt explicitly triggering the 'Math' domain
    prompt = "Calculate the trajectory of a particle in a cyclotron with magnetic field B=1.5T and radius R=0.5m, accounting for relativistic effects."
    domain = "math" 
    
    print(f"\n--- Scenario: Tough Prompt ({domain.upper()}) ---")
    print(f"Prompt: \"{prompt}\"")
    
    # Pre-compute query embedding
    x_vec = bandit_router.encoder.encode(prompt)
    x_vec = l2_normalize(x_vec)
    x_vec = np.append(x_vec, 1.0) # Bias
    
    # 2. RUN BANDIT SELECTION
    # The bandit selects based on UCB of Expected Value
    selected_model, ucb_score = bandit_router.bandit.select_arm(x_vec)
    
    # 3. COMPARE VS OPEN SOURCE GIANTS
    # We'll explicitly check these popular open weights models
    # Note: Using keys from models.json (approximate IDs)
    os_competitors = [
        "deepseek/deepseek-r1-0528-qwen3-8b", # The efficent specialist (Bandit favorite)
        "meta-llama/llama-3.3-70b-instruct",  # Strong OS Generalist
        "google/gemma-2-27b-it",             # Mid-weight OS
        "mistralai/mixtral-8x22b-instruct",   # Heavy OS
        "qwen/qwen-2.5-72b-instruct",         # Statistically strongest OS base
        "deepseek/deepseek-v3",       # The new heavyweight
    ]
    
    print(f"\n{'Model':<40} | {'Acc (Math)':<10} | {'Cost ($/1M)':<12} | {'Bandit UCB':<10} | {'Outcome'}")
    print("-" * 100)
    
    # We need to access internal bandit state to see UCBs for all models
    # UCB = Mean + Alpha * Std
    
    winner_ucb = -999.0
    winner_name = ""
    
    for m_id in os_competitors:
        if m_id not in bandit_router.bandit.A_inv:
            print(f"{m_id:<40} | {'N/A':<10} | {'N/A':<12} | {'Not in Arms'}")
            continue
            
        # Get Ground Truth Stats
        acc, cost, lat = get_simulated_metrics(m_id, domain, bandit_router.registry)
        
        # Calculate Bandit's Internal Score
        theta = bandit_router.bandit.A_inv[m_id] @ bandit_router.bandit.b[m_id]
        mean = float(theta.dot(x_vec))
        var = float(x_vec.dot(bandit_router.bandit.A_inv[m_id]).dot(x_vec))
        std = float(np.sqrt(max(var, 1e-12)))
        ucb = mean + bandit_router.bandit.alpha * std
        
        is_selected = (m_id == selected_model)
        status = "** PICKED **" if is_selected else ""
        
        print(f"{m_id:<40} | {acc:<10.2f} | ${cost:<11.3f} | {ucb:<10.3f} | {status}")
        
    print("\n--- Summary ---")
    print(f"Bandit Selected: {selected_model}")
    print("Why? The bandit balances predicted High Accuracy with Low Cost.")
    
    # 4. SIMULATE "HARD PROMPT" FAILURE (Optional)
    # What if the DeepSeek R1 fails specifically on *this* prompt?
    # This requires running the update loop which we won't do here, 
    # but we can show the "Best Possible" OS model for Math.
    
    best_os = max(os_competitors, key=lambda m: bandit_router.registry.get(m, {}).get(f"acc_{domain}", 0) if m in bandit_router.registry else 0)
    best_acc = bandit_router.registry[best_os][f"acc_{domain}"]
    print(f"Top OSS Accuracy Model: {best_os} ({best_acc:.1%})")

if __name__ == "__main__":
    analyze_tough_prompt()
