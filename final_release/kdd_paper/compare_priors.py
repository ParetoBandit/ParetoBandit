import json
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
try:
    from banditgpt import BanditRouter, l2_normalize
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from banditgpt import BanditRouter, l2_normalize

def run_prior_comparison():
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "banditgpt" / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Target: Math specialist
    target_model = "google/gemini-2.5-flash-lite"
    math_prompt = "Solve the integral of x^2 * cos(x) dx."
    
    results = {}
    
    for key in ["hle", "math_500"]:
        print(f"\nTesting Prior: {key}")
        router = BanditRouter.create(
            model_registry=registry,
            benchmark_key=key,
            prior_strength=20.0, # Standard strength
            exploration="safe"
        )
        
        # Check initial UCB scores for the target model vs others
        x = router.encoder.encode(math_prompt)
        x = l2_normalize(x)
        
        # Get top 5 models by initial UCB
        scores = []
        for m_id in router.bandit.models:
            theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
            mean = float(theta.dot(x))
            var = float(x.dot(router.bandit.A_inv[m_id]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + router.bandit.alpha * std
            scores.append((m_id, ucb))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Top 5 models by initial UCB ({key}):")
        for i, (m_id, score) in enumerate(scores[:5]):
            print(f"  {i+1}. {m_id}: {score:.4f}")
            
        # Simulate convergence: How many steps to pick the target model consistently?
        picks = []
        for _ in range(50):
            chosen, _ = router.route(math_prompt)
            picks.append(chosen == target_model)
            # Update with "truth" (target is best)
            reward = 1.0 if chosen == target_model else 0.5
            router.bandit.update(chosen, x, reward)
            
        conv_step = next((i for i, p in enumerate(picks) if all(picks[i:])), "Never")
        print(f"Converged to {target_model} at step: {conv_step}")
        results[key] = conv_step

if __name__ == "__main__":
    run_prior_comparison()
