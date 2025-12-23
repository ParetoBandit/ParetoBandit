#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add final_release to path
sys.path.append(str(Path(__file__).parent.parent / "final_release"))

from bandit import BanditRouter

def main():
    print(">>> Testing Context Sensitivity Classifier (Auto-Detection)...")
    
    # Load Models
    base_dir = Path(__file__).parent.parent / "final_release"
    with open(base_dir / "models.json") as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    router = BanditRouter(registry)
    
    prompts = [
        "Write a haiku about a robot.",
        "What are the symptoms of appendicitis?",
        "Help me debug this Python function: def sort(x):...",
        "Who uses the most bandwidth on the internet?",
        "Draft a non-disclosure agreement for a contractor.",
        "Tell me a funny dad joke.",
        "Calculate the derivative of x^2 + 4x."
    ]
    
    print(f"\n{'Detected Sensitivity':<20} | {'Selected Model':<35} | {'Risk %':<8} | {'Prompt Snippet'}")
    print("-" * 110)
    
    for p in prompts:
        # No manual sensitivity override - let the classifier decide!
        model, log = router.route(p, profile="balanced")
        
        # We need to hack/peek to see what sensitivity was detected 
        # (The log object captures it in my implementation? Let's check log structure)
        # Ah, I returned (best_model, log). The log has 'sensitivity' field if I added it?
        # Let's check bandit.py... I added `self.logs.append(log)` but log struct might not have it.
        # Wait, I added it to the dict return in my first attempt, but `route` returns `Tuple[str, RoutingLog]`.
        # I suspect RoutingLog doesn't have 'sensitivity' field unless I added it to dataclass.
        
        # Let's just re-run classification for display since it's deterministic
        detected = router._classify_sensitivity(p)
        
        meta = registry[model]
        risk = float(meta.get("hallucination_composite", 8.0))
        
        print(f"{detected:<20} | {model:<35} | {risk:<8.2f} | {p[:40]}...")

    print("-" * 110)

if __name__ == "__main__":
    main()
