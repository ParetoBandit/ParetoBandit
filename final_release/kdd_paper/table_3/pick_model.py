
import json
import numpy as np
from pathlib import Path
from final_release.bandit import BanditRouter, l2_normalize

def main():
    # Load Registry
    base_dir = Path("final_release")
    data_path = base_dir / "data" / "models_cache_with_hle.json"
    
    with open(data_path) as f:
        data = json.load(f)
    
    registry = {m["openrouter_id"]: m for m in data["models"] if "openrouter_id" in m}
    
    # Initialize Bandit (with optimal production defaults set in previous task)
    bandit_router = BanditRouter.create(model_registry=registry)
    
    prompt = """You are given a binary classifier evaluated on a test set of 1,000 samples.
The classifier predicted “positive” on 260 samples.
Of those predicted positives, 195 were actually positive.
In total, there are 240 actual positive samples in the dataset.
Questions:
How many true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) did the classifier produce?
What are the precision, recall, and F1 score of the classifier?
Provide the values of TP, FP, TN, FN, precision, recall, and F1 score, each rounded to two decimal places where applicable, with no intermediate explanation."""

    print(f"Prompt: {prompt[:100]}...")
    
    # 1. BanditGPT Selection
    x_vec = bandit_router.encoder.encode(prompt)
    x_vec = l2_normalize(x_vec)
    x_vec = np.append(x_vec, 1.0) # Bias
    
    selected_model, ucb_score = bandit_router.bandit.select_arm(x_vec)
    print(f"BanditGPT Selected: {selected_model}")
    
if __name__ == "__main__":
    main()
