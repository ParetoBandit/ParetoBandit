#!/usr/bin/env python3
"""
generate_warmup.py

End-to-End Synthetic Warmup Generator.
1. Generates 20,000 synthetic prompts (Math, Code, Chat).
2. Simulates rewards using Item Response Theory (IRT).
3. Updates a BanditRouter to build dense A matrices (Covariance) and b vectors (Beliefs).
4. Saves the resulting state to 'data/priors_warmup.joblib'.
"""

import sys
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import math

# Add project root to path so we can import src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.bandit_gpt.router import BanditRouter
from experiments.utils.data_loader import load_model_registry
from sentence_transformers import SentenceTransformer

# CONFIGURATION
# Export to root/data directory (absolute path for clarity)
OUTPUT_PATH = PROJECT_ROOT / "data" / "priors_warmup.joblib"
N_SAMPLES = 20000
SEED = 42

def ir_theory_reward(model_skill: float, difficulty: float) -> float:
    """
    Simulates a probability of success using Item Response Theory (IRT).
    
    **Mathematical Foundation:**
    
    The transformation from [0, 1] to [-3, +3] is REQUIRED for realistic probabilities.
    
    Without transformation (naive approach):
    - Opus (0.9) vs Easy (0.1): diff = 0.8 → sigmoid(0.8) ≈ 0.69 (69% success)
    - WRONG: Opus should have ~100% success on easy prompts!
    
    With transformation (this implementation):
    - Opus (0.9): (0.9 - 0.5) × 6 = +2.4 (strong skill)
    - Easy (0.1): (0.1 - 0.5) × 6 = -2.4 (low difficulty)
    - Logit: 1.5 × (2.4 - (-2.4)) = 1.5 × 4.8 = 7.2
    - Probability: sigmoid(7.2) ≈ 0.991 (99.1% success)
    - CORRECT: Opus almost always solves easy prompts!
    
    **Why This Matters:**
    - Small gaps (HLE 0.5 vs 0.55) → ~50/50 outcomes (uncertainty)
    - Large gaps (HLE 0.2 vs 0.8) → decisive outcomes (0% or 100%)
    - Without this, synthetic data is "muddy" and bandit can't learn sharp distinctions
    
    **1-Parameter Logistic Model (Rasch):**
    P(success) = 1 / (1 + exp(-a(θ - β)))
    
    where:
    - θ (theta): person ability = (model_skill - 0.5) × 6
    - β (beta): item difficulty = (difficulty - 0.5) × 6
    - a: discrimination parameter = 1.5 (controls curve steepness)
    
    Args:
        model_skill (float): Normalized skill from HLE score (0.0 to 1.0)
        difficulty (float): Normalized difficulty from router (0.0 to 1.0)
        
    Returns:
        float: Probability of success (0.0 to 1.0)
    
    Example:
        >>> ir_theory_reward(model_skill=0.9, difficulty=0.1)  # Opus on easy
        0.991  # 99.1% success
        >>> ir_theory_reward(model_skill=0.2, difficulty=0.9)  # Weak model on hard
        0.009  # 0.9% success
        >>> ir_theory_reward(model_skill=0.5, difficulty=0.5)  # Equal match
        0.500  # 50% success
    """
    # 1. Transform from [0, 1] to [-3, +3] logit space
    #    This "stretches" the probability space to enable extreme values
    theta = (model_skill - 0.5) * 6.0      # Model ability in logit space
    beta = (difficulty - 0.5) * 6.0        # Task difficulty in logit space
    
    # 2. Apply 1-Parameter Logistic Model with discrimination
    #    Discriminability (1.5) controls steepness of the sigmoid curve
    #    Higher values → sharper transitions between success/failure
    discriminability = 1.5
    logit = discriminability * (theta - beta)
    
    # 3. Convert logit to probability via sigmoid
    #    sigmoid(x) = 1 / (1 + exp(-x))
    probability = 1.0 / (1.0 + math.exp(-logit))
    
    return probability

def main():
    print(f"🚀 Starting Synthetic Warmup Generator (N={N_SAMPLES})...")
    
    # 1. Setup
    registry = load_model_registry()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize a "Blank" Router
    # We set priors="none" because we are ABOUT to build the priors manually.
    print("   Initializing blank router (cold start)...")
    router = BanditRouter.create(
        registry,
        exploration="safe",
        priors="none", 
        context_encoder=encoder
    )
    
    # 2. Generate Synthetic Prompts
    # We need a mix to cover the feature space
    print("   Generating synthetic prompts...")
    prompts = router._generate_synthetic_data(n=N_SAMPLES)
    
    print(f"   Simulating {len(prompts)} interactions across {len(router.bandit.models)} models...")
    
    # 3. Analyze HLE Score Coverage
    models_with_hle = 0
    models_with_fallback = 0
    hle_scores = []
    
    for model_id in router.bandit.models:
        hle = router.registry.get(model_id, {}).get("hle", None)
        if hle is not None:
            models_with_hle += 1
            hle_scores.append(hle)
        else:
            models_with_fallback += 1
    
    print(f"   HLE Coverage:")
    print(f"     ✓ Models with HLE scores: {models_with_hle}/{len(router.bandit.models)}")
    if models_with_fallback > 0:
        print(f"     ⚠ Models using fallback (0.5): {models_with_fallback}")
    if hle_scores:
        print(f"     HLE range: [{min(hle_scores):.3f}, {max(hle_scores):.3f}], mean={np.mean(hle_scores):.3f}")
    
    # 4. Training Loop (The "Simulation")
    updates_count = 0
    
    for prompt in tqdm(prompts, desc="Warming Up"):
        # A. Analyze Context (The "Map")
        # Get difficulty score for IRT calculation
        difficulty = router._detect_difficulty_score(prompt)
        
        # B. Update Every Model (The "Compass")
        for model_id in router.bandit.models:
            # Get Model Skill from Registry (HLE Score)
            # Fallback to 0.5 if missing
            hle = router.registry.get(model_id, {}).get("hle", 0.5)
            if hle is None:
                hle = 0.5
            
            # Calculate IRT Reward
            # "Would this model succeed on this prompt?"
            prob_success = ir_theory_reward(model_skill=hle, difficulty=difficulty)
            
            # Apply Cost Penalty?
            # OPTIONAL: If you want the priors to bake in cost-efficiency:
            # cost = router._estimate_cost(model_id, len(prompt)*1.3, 100)
            # reward = prob_success / (1.0 + cost) 
            # FOR NOW: Let's stick to pure Capability (Prob Success) 
            # and let the router's UCB cost logic handle the rest during runtime.
            reward = prob_success
            
            # C. Update the Bandit State
            # IMPORTANT: Pass the prompt STRING, not the context vector!
            # The router.update() method will call _get_context_vector() internally,
            # which adds the bias term. If we pass log.context_vector, we'd get
            # double bias (dimension mismatch error).
            router.update(model_id, prompt, reward)
            updates_count += 1

    # 5. Save the Artifact
    print(f"✅ Training complete. Processed {updates_count} simulated updates.")
    
    # We extract strictly the LinUCB matrices
    state_to_save = {
        "A": router.bandit.A,  # The Covariance Matrices (The Map)
        "b": router.bandit.b,  # The Reward Vectors (The Compass)
        "n": N_SAMPLES         # Metadata
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state_to_save, OUTPUT_PATH)
    
    print(f"💾 Saved Warmup Priors to: {OUTPUT_PATH}")
    print("   You can now use priors='warmup' in your experiments.")

if __name__ == "__main__":
    main()
