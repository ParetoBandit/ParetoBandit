
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from final_release.bandit import BanditRouter, l2_normalize, OptimizationProfile

def load_ground_truth():
    """Load the production registry with high-fidelity scores."""
    base_dir = Path("final_release")
    data_path = base_dir / "models.json"
    
    with open(data_path) as f:
        data = json.load(f)
    
    registry = {m["openrouter_id"]: m for m in data["models"] if "openrouter_id" in m}
    return registry

def get_metrics(model_id, registry, task_type="general"):
    """
    Get (Accuracy, Cost) for a model on a specific task type.
    task_type: 'general' (Easy), 'mid' (Balanced), or 'hle' (Hard)
    """
    if model_id not in registry:
        return 0.0, 100.0
        
    m = registry[model_id]
    cost = m.get("price_1m_blended", 1.0)
    base_hle = m.get("hle", 0.0) or 0.0
    
    if task_type == "hle":
        # Hard Task: Use HLE Score
        acc = base_hle
        if acc == 0:
             # Fallback: Math 500 is easier than HLE
             math_score = m.get("math_500", 0.0) or 0.0
             acc = math_score * 0.3

    elif task_type == "mid":
        # Mid Task: Use Math 500 or Humanity score
        acc = m.get("math_500")
        if acc is None:
            acc = base_hle * 1.5 # Assume mid tasks are 1.5x easier than HLE
        acc = min(0.95, acc)

    else:
        base_qual = m.get("quality_score", 0.8)
        # Boost it because "Easy" tasks are... easy
        acc = min(0.99, base_qual * 1.2)
        
    return acc, cost


# ==========================================
# Baseline Routers (Simulated)
# ==========================================
class RouteLLM_Wrapper:
    def route(self, prompt: str) -> str:
        # Simple keyword-based routing (simulating BERT classifier)
        keywords = ["calculate", "code", "function", "integral", "solve", "math", "reasoning"]
        if any(k in prompt.lower() for k in keywords):
            return "openai/gpt-4o" # Strong
        return "google/gemini-2.0-flash-001" # Weak

class FrugalGPT_Wrapper:
    def route(self, prompt: str, registry, task_type) -> str:
        # Cascade: Try Cheap -> Check Score -> Try Strong
        cheap = "google/gemini-2.0-flash-001"
        strong = "openai/gpt-4o"
        
        # Simulate Cheap Model Performance
        # In reality, FrugalGPT uses an LLM-as-a-Judge or scoring function
        acc_cheap, _ = get_metrics(cheap, registry, task_type)
        
        # Threshold: 0.9 (If cheap is likely >90% confident, stop)
        if acc_cheap > 0.9:
            return cheap
            
        return strong

def run_simulation():
    print("Generating Table 3 Data...")
    
    registry = load_ground_truth()
    
    # Filter anomalies (Free/Preview models that skew economics)
    registry = {k: v for k, v in registry.items() if v.get("price_1m_blended", 0) > 0.01}
    
    # Prior strength 40 is standard for HLE transfer
    router = BanditRouter.create(model_registry=registry, prior_strength=40.0)
    
    # Baselines
    routellm = RouteLLM_Wrapper()
    frugalgpt = FrugalGPT_Wrapper()
    
    # Define Scenarios
    scenarios = [
        {
            "name": "Simple Query (Easy)",
            "prompt": "Write a python function to print 'Hello World'.",
            "type": "general",
            "static_choice": "google/gemini-2.0-flash-001" 
        },
        {
            "name": "Standard Logic (Mid)",
            "prompt": "Solve for x: 3x + 5 = 20. Explain your steps.",
            "type": "mid",
            "static_choice": "google/gemini-2.0-flash-001"
        },
        {
            "name": "Complex Reasoning (Hard)",
            "prompt": "You are an expert in regulating heat networks... [HLE Prompt context] ... decide the supply temperature.",
            "type": "hle",
            "static_choice": "google/gemini-2.0-flash-001" 
        }
    ]
    
    results = []
    
    for scen in scenarios:
        print(f"\n--- Scenario: {scen['name']} ---")
        
        # --------------------------------------------------------------------------
        # TWO-PHASE EVALUATION PROTOCOL (Burn-In + Convergence)
        # --------------------------------------------------------------------------
        # We run 100 requests to allow the Bandit to learn from feedback.
        # Phase 1 (0-50): Burn-In (Learning from Priors + Feedback)
        # Phase 2 (50-100): Converged Performance (Used for Table 3)
        
        bandit_choice = None
        for i in range(100):
            # 1. Bandit Selection
            # We use a dummy prompt since we are simulating feedback
            resp, log = router.route(scen['prompt'], profile="value_efficient")
            # resp is the model_id string
            selected_model_id = resp
            
            # 2. Simulate Feedback
            # Re-calculating score for the selected model to generate feedback
            # Use get_metrics to handle missing keys/fallbacks safely
            acc, _ = get_metrics(selected_model_id, registry, scen['type'])
            
            # Simulate a binary reward (Bernoulli trial based on accuracy)
            import random
            reward = 1.0 if random.random() < acc else 0.0
            
            # Feedback to Bandit
            # Update the bandit with the observed reward (Quality)
            # The bandit learns Quality correlation. Cost is handled at routing time via static lookup.
            router.update(selected_model_id, scen['prompt'], float(reward))
            
            # Store the *Final* selection (Request #99) for the Table
            if i == 99:
                bandit_choice = selected_model_id
        
        # 2. LiteLLM (Static)
        litellm_choice = scen['static_choice']
        
        # 3. RouteLLM (Classifier)
        routellm_choice = routellm.route(scen['prompt'])
        
        # 4. FrugalGPT (Cascade)
        frugal_choice = frugalgpt.route(scen['prompt'], registry, scen['type'])
        
        
        # 5. Oracle (Upper Bound)
        all_models = [m for m in registry.keys() if m in router.bandit.A]
        oracle_choice = max(all_models, key=lambda m: get_metrics(m, registry, scen['type'])[0])

        systems = [
            ("BanditGPT (Ours)", bandit_choice),
            ("LiteLLM (Static)", litellm_choice),
            ("RouteLLM (Classifier)", routellm_choice),
            ("FrugalGPT (Cascade)", frugal_choice),
            ("Oracle (Upper Bound)", oracle_choice)
        ]
        
        for sys_name, model in systems:
            acc, cost = get_metrics(model, registry, scen['type'])
            
            # FrugalGPT Cost Penalty: It pays for Cheap + Strong if it cascaded
            if sys_name == "FrugalGPT (Cascade)" and model == "openai/gpt-4o":
                 cheap_cost = get_metrics("google/gemini-2.0-flash-001", registry, scen['type'])[1]
                 cost += cheap_cost
            
            print(f"  {sys_name}: {model} (Acc={acc:.3f}, Cost=${cost:.3f})")
            
            results.append({
                "Scenario": scen['name'],
                "System": sys_name,
                "Model": model,
                "Accuracy": acc,
                "Cost": cost
            })

    # Format Table
    df = pd.DataFrame(results)
    
    md_table = "| Scenario | System | Selected Model | Accuracy | Cost ($/1M) |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in df.iterrows():
        md_table += f"| {row['Scenario']} | **{row['System']}** | `{row['Model']}` | {row['Accuracy']:.1%} | ${row['Cost']:.3f} |\n"
        
    print("\nGenerated Table 3:")
    print(md_table)
    
    # Save to file
    out_path = Path(__file__).parent / "table_3_generated.md"
    with open(out_path, "w") as f:
        f.write("# Table 3: Performance Dynamics on Challenging Prompts\n\n")
        f.write(md_table)
        f.write("\n\n*Note: 'Legacy Router' represents static routing rules (e.g. LiteLLM) which default to cost-efficient models regardless of complexity.*")
    
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_simulation()
