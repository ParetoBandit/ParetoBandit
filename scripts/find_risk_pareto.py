#!/usr/bin/env python3
import json
from pathlib import Path

def load_models():
    base_dir = Path("/Users/annette/repostitories/llm_jury/final_release")
    with open(base_dir / "models.json") as f:
        data = json.load(f)
    return data["models"]

def is_dominated(candidate, others):
    """
    Check if candidate is dominated by ANY other model in 'others'.
    A dominates B if:
       Attributes:
    - Acc: 'hle' (Reasoning) or 'math_500' if HLE is close. Let's use 'hle'.
    - Cost: 'price_1m_blended'
    - Risk: 'hallucination_composite'
    - Latency: 'time_to_first_token_seconds'
    """
    # Attributes for 4D Pareto:
    # 1. Acc (HLE) - Maximize
    # 2. Cost ($/1M) - Minimize
    # 3. Risk (Hallucination %) - Minimize
    # 4. Latency (TTFT s) - Minimize
    c_acc = candidate.get("hle", 0.0) or 0.0
    c_cost = candidate.get("price_1m_blended", 100.0) or 100.0
    c_risk = candidate.get("hallucination_composite", candidate.get("hallucination_rate", 100.0))
    c_lat = candidate.get("time_to_first_token_seconds", 10.0) or 10.0
    
    for other in others:
        if other["openrouter_id"] == candidate["openrouter_id"]:
            continue
            
        o_acc = other.get("hle", 0.0) or 0.0
        o_cost = other.get("price_1m_blended", 100.0) or 100.0
        o_risk = other.get("hallucination_composite", other.get("hallucination_rate", 100.0))
        o_lat = other.get("time_to_first_token_seconds", 10.0) or 10.0
        
        # Check domination (Better = Higher Acc, Lower Cost, Lower Risk, Lower Latency)
        better_acc = o_acc >= c_acc
        better_cost = o_cost <= c_cost
        better_risk = o_risk <= c_risk
        better_lat = o_lat <= c_lat
        
        strict = (o_acc > c_acc) or (o_cost < c_cost) or (o_risk < c_risk) or (o_lat < c_lat)
        
        if better_acc and better_cost and better_risk and better_lat and strict:
            return True, other["openrouter_id"]
            
    return False, None

def main():
    models = load_models()
    # Filter only models with valid data to avoid noise
    valid_models = [m for m in models if m.get("price_1m_blended") is not None]
    
    pareto_frontier = []
    
    for m in valid_models:
        dominated, dominator = is_dominated(m, valid_models)
        if not dominated:
            pareto_frontier.append(m)
        else:
            # excessive print
            # print(f"Dominated: {m['display_name']} by {dominator}")
            pass
            
    # Sort for readability: Cost ascending
    pareto_frontier.sort(key=lambda x: x.get("price_1m_blended", 0))
    
    print(f"Found {len(pareto_frontier)} Pareto-Optimal Models (Acc vs Cost vs Risk)")
    print("-" * 100)
    print(f"{'Model':<40} {'HLE':<8} {'$/1M':<10} {'Risk %':<8} {'Notes'}")
    print("-" * 100)
    
    for m in pareto_frontier:
        name = m.get("display_name", m["openrouter_id"])[:38]
        acc = m.get("hle", 0.0)
        cost = m.get("price_1m_blended", 0.0)
        risk = m.get("hallucination_composite", 0.0)
        
        # Tag potential roles
        tags = []
        if cost < 0.1: tags.append("Cheap")
        if acc > 0.05 and risk < 5.0: tags.append("SafeWrapper")
        if risk < 2.5: tags.append("SafetyExpert")
        
        print(f"{name:<40} {acc:<8.3f} ${cost:<9.4f} {risk:<8.2f} {', '.join(tags)}")

    # Print python list for copy-paste
    print("-" * 100)
    ids = [m["openrouter_id"] for m in pareto_frontier]
    print("PARETO_MODELS = [")
    for mid in ids:
        print(f'    "{mid}",')
    print("]")

if __name__ == "__main__":
    main()
