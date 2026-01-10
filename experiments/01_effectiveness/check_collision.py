import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path("/Users/annette/repostitories/banditGPT")
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "experiments"))

from experiments.utils.data_loader import load_oracle_rewards, load_model_registry

def check_first_model_regret():
    reg = load_model_registry()
    test_oracle = load_oracle_rewards("test_rewards_hle_models.jsonl")
    
    model_coverage = defaultdict(int)
    for p in test_oracle.values():
        for m in p:
            model_coverage[m] += 1
            
    min_cov = len(test_oracle) * 0.5
    available_models = [m for m in reg.keys() if model_coverage.get(m, 0) >= min_cov]
    
    if not available_models:
        print("No available models found.")
        return
        
    first_model = available_models[0]
    print(f"First available model: {first_model}")
    
    selected_rewards = []
    oracle_best = []
    
    for prompt, rewards in test_oracle.items():
        if not rewards:
            continue
            
        # Filter rewards for available models ONLY (same as main script)
        available_rewards = [rewards.get(m, 0.0) for m in available_models if m in rewards]
        
        if available_rewards:
            oracle_best.append(max(available_rewards))
            selected_rewards.append(rewards.get(first_model, 0.0))
            
    regret = sum(oracle_best) - sum(selected_rewards)
    print(f"Cumulative Regret of {first_model}: {regret}")

if __name__ == "__main__":
    check_first_model_regret()
