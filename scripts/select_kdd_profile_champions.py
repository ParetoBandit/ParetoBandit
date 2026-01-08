import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# -----------------------------------------------------------------------------
# 1. KDD-Rigorous Constants (From router.py)
# -----------------------------------------------------------------------------
MARKET_COST_FLOOR = 0.0005     # $/1k
MARKET_COST_CEILING = 10.00    # $/1k
COST_RANGE_LOG = np.log(MARKET_COST_CEILING) - np.log(MARKET_COST_FLOOR)

MARKET_LATENCY_FLOOR = 0.05    # seconds
MARKET_LATENCY_CEILING = 5.0   # seconds
LATENCY_RANGE_LOG = np.log(MARKET_LATENCY_CEILING) - np.log(MARKET_LATENCY_FLOOR)

# HLE Calibration
EASY_FLOOR = 0.95
EASY_SLOPE = 0.05
HARD_MAX_BENCHMARK = 0.35
HARD_EXPONENT = 2.0

@dataclass
class ModelCandidate:
    id: str
    display_name: str
    hle: float
    cost_1m: float  # Blended $/1M
    latency: float  # TTFT
    
    @property
    def cost_per_1k(self) -> float:
        return self.cost_1m / 1000.0

# -----------------------------------------------------------------------------
# 2. Router Emulation Logic
# -----------------------------------------------------------------------------
def calculate_penalties(cost_per_1k: float, latency: float) -> Tuple[float, float]:
    # Cost Penalty
    safe_cost = max(cost_per_1k, MARKET_COST_FLOOR)
    log_cost = np.log(safe_cost)
    norm_cost = (log_cost - np.log(MARKET_COST_FLOOR)) / COST_RANGE_LOG
    cost_pen = max(0.0, min(1.0, norm_cost))
    
    # Latency Penalty
    safe_lat = max(latency, MARKET_LATENCY_FLOOR)
    log_lat = np.log(safe_lat)
    norm_lat = (log_lat - np.log(MARKET_LATENCY_FLOOR)) / LATENCY_RANGE_LOG
    lat_pen = max(0.0, min(1.0, norm_lat))
    
    return cost_pen, lat_pen

def transform_hle(hle: float, difficulty: float) -> float:
    # Expert A (Easy)
    u_easy = EASY_FLOOR + (EASY_SLOPE * hle)
    
    # Expert B (Hard) - Quadratic Elite Advantage
    linear_score = hle / HARD_MAX_BENCHMARK
    u_hard = max(0.01, min(0.99, linear_score ** HARD_EXPONENT))
    
    # Mixture
    return ((1.0 - difficulty) * u_easy) + (difficulty * u_hard)

def calculate_utility(model: ModelCandidate, weights: Dict[str, float], difficulty: float) -> float:
    # 1. Get Predicted Quality (Transformed HLE)
    # Note: In router, this is UCB. Here we use the PRIOR mean (the HLE transform).
    quality = transform_hle(model.hle, difficulty)
    
    # 2. Get Penalties
    cost_pen, lat_pen = calculate_penalties(model.cost_per_1k, model.latency)
    
    # 3. Weighted Utility
    # Utility = w_q * Q + w_c * (1-C) + w_l * (1-L)
    w_q = weights.get("w_q", 0.0)
    w_c = weights.get("w_c", 0.0)
    w_l = weights.get("w_l", 0.0)
    
    return (w_q * quality) + (w_c * (1.0 - cost_pen)) + (w_l * (1.0 - lat_pen))

# -----------------------------------------------------------------------------
# 3. Main Analysis
# -----------------------------------------------------------------------------
def main():
    print("Loading models...")
    with open('src/bandit_gpt/config/models.json', 'r') as f:
        data = json.load(f)
        
    candidates = []
    for m in data['models']:
        # Fallbacks for missing data
        hle = m.get('hle', 0.5) or 0.5
        # Use blending for cost selection
        cost = m.get('price_1m_blended', 0.5) 
        if cost == 0: cost = 0.01 # Avoid log(0)
        
        lat = m.get('time_to_first_token_seconds', 1.0)
        if lat == 0: lat = 0.05
        
        candidates.append(ModelCandidate(
            id=m['openrouter_id'],
            display_name=m['display_name'],
            hle=hle,
            cost_1m=cost,
            latency=lat
        ))
        
    print(f"Loaded {len(candidates)} models.")
    
    # Define Profiles (from router.py)
    PROFILES = {
        "MAX_QUALITY":  {"w_q": 0.97, "w_c": 0.02, "w_l": 0.01, "z": 1.0}, # Test on Hard
        "ARBITRAGE":    {"w_q": 0.75, "w_c": 0.20, "w_l": 0.05, "z": 1.0}, # Test on Hard (smart & cheap)
        "BEST_VALUE":   {"w_q": 0.40, "w_c": 0.55, "w_l": 0.05, "z": 0.5}, # Test on Mid
        "COST_SAVER":   {"w_q": 0.10, "w_c": 0.85, "w_l": 0.05, "z": 0.0}, # Test on Easy
        "LOW_LATENCY":  {"w_q": 0.20, "w_c": 0.10, "w_l": 0.70, "z": 0.5}, # Test on Mid (speed focus)
    }
    
    champions = set()
    
    print("\n" + "="*100)
    print(f"{'PROFILE':<15} | {'WINNER':<40} | {'SCORE':<6} | {'HLE':<5} | {'COST':<7} | {'LATENCY':<5}")
    print("="*100)
    
    for name, params in PROFILES.items():
        difficulty = params.pop("z")
        weights = params
        
        # Score all models
        scored = []
        for m in candidates:
            u = calculate_utility(m, weights, difficulty)
            scored.append((u, m))
            
        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Pick top 2
        winner = scored[0][1]
        runner_up = scored[1][1]
        
        champions.add(winner.id)
        champions.add(runner_up.id)
        
        u_win = scored[0][0]
        print(f"{name:<15} | {winner.id:<40} | {u_win:.4f} | {winner.hle:.3f} | ${winner.cost_1m:<6.2f} | {winner.latency:.2f}s")
        print(f"{'(runner-up)':<15} | {runner_up.id:<40} | {scored[1][0]:.4f} | {runner_up.hle:.3f} | ${runner_up.cost_1m:<6.2f} | {runner_up.latency:.2f}s")
        print("-" * 100)
        
    print("\n" + "="*100)
    print("FINAL KDD PORTFOLIO (Union of Champions)")
    print("="*100)
    
    # Sort by cost for display
    final_list = [c for c in candidates if c.id in champions]
    final_list.sort(key=lambda x: x.cost_1m)
    
    for m in final_list:
        print(f"- {m.id:<40} (HLE={m.hle:.2f}, ${m.cost_1m:.2f}, {m.latency:.2f}s)")
        
if __name__ == "__main__":
    main()
