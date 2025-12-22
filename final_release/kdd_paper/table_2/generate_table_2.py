#!/usr/bin/env python3
"""
Table 2: SOTA Router Comparison
Compares BanditGPT against RouteLLM, FrugalGPT, Aurelio AI, and LiteLLM.
"""

import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_release.bandit import BanditRouter

# ==============================================================================
# 1. LOAD REAL MODEL DATA
# ==============================================================================

def load_model_data():
    cache_path = Path(__file__).parent.parent.parent / "models.json"
    with open(cache_path) as f:
        data = json.load(f)
    
    full_registry = {m["openrouter_id"]: m for m in data["models"] if "openrouter_id" in m}
    
    sim_registry = {}
    for oid, m in full_registry.items():
        sim_registry[oid] = {
            "price": m.get("price_1m_blended", 1.0),
            "acc_math": m.get("math_500", 0.5),
            "acc_code": m.get("humaneval_score", 50.0) / 100.0,
            "acc_instr": m.get("mmlu_pro", 0.5),
        }
    return full_registry, sim_registry

FULL_REGISTRY, MODEL_REGISTRY = load_model_data()

MODELS = {
    "frontier": "openai/gpt-4o", 
    "budget": "amazon/nova-lite-v1",
    "balanced": "google/gemini-2.0-flash-001",
    "specialist_math": "deepseek/deepseek-r1-0528-qwen3-8b",
    "specialist_code": "deepseek/deepseek-r1-0528-qwen3-8b"
}

# ==============================================================================
# 2. BASELINE WRAPPERS
# ==============================================================================

class RouteLLM_Wrapper:
    def __init__(self):
        self.strong = MODELS["frontier"]
        self.weak = MODELS["budget"]
        self.keywords = ["calculate", "code", "function", "integral", "derivative", "solve", "implement", "python", "math"]

    def route(self, prompt: str) -> str:
        if any(word in prompt.lower() for word in self.keywords):
            return self.strong
        return self.weak

class AurelioAI_Wrapper:
    def __init__(self):
        self.routes = {
            "math": ["calculate", "derivative", "integral", "solve", "equation", "math", "volume"],
            "code": ["function", "python", "script", "debug", "class", "import", "implement", "typescript"]
        }

    def route(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if random.random() < 0.15: return MODELS["frontier"]
        if any(u in prompt_lower for u in self.routes["math"]): return MODELS["specialist_math"]
        if any(u in prompt_lower for u in self.routes["code"]): return MODELS["specialist_code"]
        return MODELS["frontier"]

class FrugalGPT_Wrapper:
    def execute(self, domain: str) -> Dict:
        cheap, strong = MODELS["budget"], MODELS["frontier"]
        acc_cheap = MODEL_REGISTRY[cheap][f"acc_{domain}"]
        if random.random() < acc_cheap:
            return {"model": cheap, "cost": MODEL_REGISTRY[cheap]["price"]/1000, "acc": acc_cheap}
        return {"model": strong, "cost": (MODEL_REGISTRY[cheap]["price"] + MODEL_REGISTRY[strong]["price"])/1000, "acc": MODEL_REGISTRY[strong][f"acc_{domain}"]}

class LiteLLM_Wrapper:
    def route(self) -> str: return MODELS["balanced"]

# ==============================================================================
# 3. EXPERIMENT RUNNER
# ==============================================================================

def run_comparison(n_samples=1000):
    results = []
    routellm = RouteLLM_Wrapper()
    aurelio = AurelioAI_Wrapper()
    frugal = FrugalGPT_Wrapper()
    litellm = LiteLLM_Wrapper()
    
    bandit_router = BanditRouter.create(
        model_registry=FULL_REGISTRY, 
        exploration="balanced",
        prior_strength=40.0,
        forgetting_factor=0.9,
        benchmark_key="hle"
    )
    
    # Debug initial scores
    test_prompt = "Solve the differential equation dy/dx = y*cos(x)."
    x = bandit_router.encoder.encode(test_prompt)
    from final_release.bandit import l2_normalize
    x = l2_normalize(x)
    x = np.append(x, 1.0) # Add bias term
    print("\nInitial UCB Scores (Math Prompt):")
    for m_id in ["openai/gpt-4o", "meta-llama/llama-3.2-1b-instruct", "google/gemini-3-pro-preview", "openai/gpt-5.1", "google/gemini-2.0-flash-001"]:
        theta = bandit_router.bandit.A_inv[m_id] @ bandit_router.bandit.b[m_id]
        mean = float(theta.dot(x))
        var = float(x.dot(bandit_router.bandit.A_inv[m_id]).dot(x))
        std = float(np.sqrt(max(var, 1e-12)))
        ucb = mean + bandit_router.bandit.alpha * std
        print(f"  {m_id}: {ucb:.4f} (mean={mean:.4f}, std={std:.4f})")

    prompts = [
        ("math", "Solve the differential equation dy/dx = y*cos(x)."),
        ("code", "Implement a fast Fourier transform in Python."),
        ("instr", "Who was the first person to walk on the moon?")
    ]
    
    bandit_counts = {}
    for i in range(n_samples):
        domain, prompt = random.choice(prompts)
        
        # RouteLLM
        m = routellm.route(prompt)
        results.append({"System": "RouteLLM", "Acc": MODEL_REGISTRY[m][f"acc_{domain}"], "Cost": MODEL_REGISTRY[m]["price"]/1000})
        
        # Aurelio AI
        m = aurelio.route(prompt)
        results.append({"System": "Aurelio AI", "Acc": MODEL_REGISTRY[m][f"acc_{domain}"], "Cost": MODEL_REGISTRY[m]["price"]/1000})
        
        # FrugalGPT
        res = frugal.execute(domain)
        results.append({"System": "FrugalGPT", "Acc": res["acc"], "Cost": res["cost"]})
        
        # LiteLLM
        m = litellm.route()
        results.append({"System": "LiteLLM", "Acc": MODEL_REGISTRY[m][f"acc_{domain}"], "Cost": MODEL_REGISTRY[m]["price"]/1000})
        
        # BanditGPT (Ours)
        m, _ = bandit_router.route(prompt, profile="balanced")
        bandit_counts[m] = bandit_counts.get(m, 0) + 1
        acc = MODEL_REGISTRY[m][f"acc_{domain}"]
        cost = MODEL_REGISTRY[m]["price"]/1000
        results.append({"System": "BanditGPT (Ours)", "Acc": acc, "Cost": cost})
        
        # UPDATE THE ROUTER!
        bandit_router.update(m, prompt, acc)
        
        if i % 100 == 0:
            # Debug UCBs
            x_debug = bandit_router._get_context_vector(prompt)
            print(f"\nStep {i}: Selected {m} (Rew={acc:.2f})")
            for debug_m in ["meta-llama/llama-3.2-1b-instruct", "google/gemini-2.0-flash-001"]:
                _, ucb = bandit_router.bandit.select_arm(x_debug, candidates=[debug_m])
                print(f"  {debug_m}: UCB={ucb:.4f}")

    df = pd.DataFrame(results)
    print("\nBanditGPT Top 10 Models:")
    sorted_counts = sorted(bandit_counts.items(), key=lambda x: x[1], reverse=True)
    for m, c in sorted_counts[:10]:
        print(f"  {m}: {c}")

    summary = df.groupby("System").mean().reset_index()
    
    print("\nTable 2: SOTA Router Comparison")
    print("| System | Accuracy | Cost ($/1k) |")
    print("| :--- | :--- | :--- |")
    for _, row in summary.iterrows():
        print(f"| {row['System']} | {row['Acc']:.4f} | {row['Cost']:.6f} |")
    
    with open(Path(__file__).parent / "table_2_results.json", "w") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2)

if __name__ == "__main__":
    run_comparison()
