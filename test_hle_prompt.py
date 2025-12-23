
import numpy as np
import json
from pathlib import Path
from final_release.bandit import BanditRouter, l2_normalize

# ==============================================================================
# Helper to simulate "Real" performance lookup
# ==============================================================================
def get_metrics_from_registry(model_id, registry):
    """
    Returns the HLE Accuracy and Cost from the registry.
    """
    if model_id not in registry:
        return 0.0, 100.0, 0.0
    
    meta = registry[model_id]
    # Use HLE score if available, else fallback to math_500 or generic quality
    acc = meta.get("hle", 0.0)
    if acc is None: acc = meta.get("math_500", 0.0)
    if acc is None: acc = meta.get("quality_score", 0.0)
    
    cost = meta.get("price_1m_blended", 1.0)
    lat = meta.get("time_to_first_token_seconds", 0.5)
    
    return acc, cost, lat

# ==============================================================================
# Main Analysis
# ==============================================================================
def analyze_hle_prompt():
    print("Initializing Bandit with HLE Priors...")
    
    # Load Registry with HLE data
    base_dir = Path("final_release")
    data_path = base_dir / "data" / "models_cache_with_hle.json"
    
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    with open(data_path) as f:
        data = json.load(f)
    
    # Flatten registry
    registry = {m["openrouter_id"]: m for m in data["models"] if "openrouter_id" in m}
    
    # Initialize Bandit (Cold Start for demo, or attempt to load state if available)
    # We use 'create' which loads defaults.
    bandit_router = BanditRouter.create(model_registry=registry)
    
    # 1. ESTABLISH THE "TOUGH" SCENARIO (Real prompt from Cluster 103)
    prompt = """You are an expert in regulating heat networks and you need to decide on the supply temperature of the heat exchange station based on the known information...
    Current time: 2023-01-20 12:03;
    Predicted heat load for the next hour: 33 w/m2;
    Weather forecast for the next hour: 13 °C;
    current supply water temperature: 48°C;
    ...
    You need to decide the water supply temperature... make sure that 80% of your customers have a room temperature between 18 and 22°C."""
    
    print(f"\n--- Scenario: Complex Reasoning (HLE-style) ---")
    print(f"Prompt Segment: \"{prompt[:200]}...\"")
    
    # Pre-compute query embedding
    x_vec = bandit_router.encoder.encode(prompt)
    x_vec = l2_normalize(x_vec)
    x_vec = np.append(x_vec, 1.0) # Bias
    
    # 2. RUN BANDIT SELECTION
    selected_model, ucb_score = bandit_router.bandit.select_arm(x_vec)
    
    # 3. COMPARE VS OPEN SOURCE & SOTA
    competitors = [
        "deepseek/deepseek-r1",             # The Reasoning King
        "anthropic/claude-3.5-sonnet",      # The Generalist King
        "deepseek/deepseek-r1-0528-qwen3-8b", # The Efficient Specialist
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-2.0-flash-001",      # The Static Balance Choice
        "openai/gpt-4o"
    ]
    
    print(f"\n{'Model':<40} | {'HLE Acc':<10} | {'Cost ($/1M)':<12} | {'Bandit Score':<12} | {'Outcome'}")
    print("-" * 105)
    
    # Sort by HLE Acc for better visualization
    competitors.sort(key=lambda m: get_metrics_from_registry(m, registry)[0], reverse=True)
    
    for m_id in competitors:
        if m_id not in bandit_router.bandit.A_inv:
            print(f"{m_id:<40} | {'N/A':<10} | {'N/A':<12} | {'Not in Arms'}")
            continue
            
        acc, cost, lat = get_metrics_from_registry(m_id, registry)
        
        # Calculate Bandit's Internal Score (UCB)
        theta = bandit_router.bandit.A_inv[m_id] @ bandit_router.bandit.b[m_id]
        mean = float(theta.dot(x_vec))
        var = float(x_vec.dot(bandit_router.bandit.A_inv[m_id]).dot(x_vec))
        std = float(np.sqrt(max(var, 1e-12)))
        ucb = mean + bandit_router.bandit.alpha * std
        
        # Check actual selection
        is_selected = (m_id == selected_model)
        status = "** PICKED **" if is_selected else ""
        
        # Highlight if it's the efficient specialist
        if m_id == "deepseek/deepseek-r1-0528-qwen3-8b":
            status += " (Eff. Spec)"
            
        print(f"{m_id:<40} | {acc:<10.3f} | ${cost:<11.3f} | {ucb:<12.3f} | {status}")
        
    print("\n--- Summary ---")
    print(f"Bandit Selected: {selected_model}")
    
    # Get stats for selected
    sel_acc, sel_cost, _ = get_metrics_from_registry(selected_model, registry)
    
    # Get stats for Top HLE (Oracle)
    best_hle_model = max(registry.keys(), key=lambda m: registry[m].get("hle", 0) if registry[m].get("hle") else 0)
    best_acc, best_cost, _ = get_metrics_from_registry(best_hle_model, registry)
    
    print(f"Selected Model HLE: {sel_acc:.1%} (Cost: ${sel_cost:.3f})")
    print(f"Top Possible HLE:   {best_acc:.1%} ({best_hle_model}, Cost: ${best_cost:.3f})")
    
    if sel_cost < best_cost and sel_acc > 0.8 * best_acc:
        print("\nSUCCESS: Bandit found a cost-efficient alternative to the frontier model.")
    elif selected_model == best_hle_model:
        print("\nSUCCESS: Bandit correctly identified the SOTA model for this tough task.")
    else:
        print("\nObservation: Bandit made a trade-off.")

if __name__ == "__main__":
    analyze_hle_prompt()
