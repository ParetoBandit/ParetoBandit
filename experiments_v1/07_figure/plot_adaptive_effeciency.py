"""
Figure 6: Adaptive Efficiency (Zero-Shot Readiness)

Visualizes the "Latent Semantic Transfer" capability where a new model
inherits the intuition of its nearest semantic neighbor, avoiding the
catastrophic "Cold Start" dip.

Scenario:
1. Train on standard portfolio (Mixtral, GPT-4-Turbo) for 300 steps.
2. At t=300, "Release" a new superior model (GPT-5.1).
3. Compare:
   - Cold Start: Initializes with Identity matrix (learns from 0).
   - Semantic Transfer: Inherits θ from GPT-4-Turbo but resets A (confidence).

Outcome:
The Semantic Transfer line should maintain high reward immediately,
while Cold Start dips and slowly climbs.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import copy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CostAwareLinUCBRouter, CostAwareTabulaRasaRouter
from bandit_gpt.calibration import embed_prompt
# We use SentenceTransformer directly to simulate the embedding part of "Transfer"
from sentence_transformers import SentenceTransformer
import joblib
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER, 
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS  # Use all-models dataset which includes GPT-5.1
)
import gzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
OLD_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEIGHBOR_MODEL = "openai/gpt-4-turbo" # The "Teacher" for transfer (semantically closest to GPT-5.1)
NEW_MODEL = "openai/gpt-5.1"          # The "New Release" (Superior) - Real model from dataset!

# Simulation Params
TOTAL_STEPS = 1000
RELEASE_STEP = 300  # Release new model early to show the contrast
WINDOW_SIZE = 50    # Smoothing window for plots

# ============================================================================
# DATA LOADING
# ============================================================================
def load_real_data():
    """Load LMSYS Dev Data from all-models dataset"""
    prompts = []
    rewards_map = {}
    
    # Load from all-models dataset (includes GPT-5.1 and many other models)
    with gzip.open(DEV_DATA_PATH_ALL_MODELS, 'rt') as f:
        for line in f:
            d = json.loads(line)
            
            # Extract prompt (only add once per sample)
            if len(prompts) < d.get("sample_id", len(prompts)) + 1:
                prompts.append(d.get("prompt", ""))
            
            # Extract reward for this model
            # Use reward_logit as the reward signal (ranges from ~-5 to +5)
            mid = d.get("model_id")
            score = d.get("reward_logit", d.get("reward", 0.0))
            if mid not in rewards_map:
                rewards_map[mid] = []
            rewards_map[mid].append(score)
            
    logger.info(f"Loaded from {DEV_DATA_PATH_ALL_MODELS}")
    logger.info(f"Models needed: {OLD_MODELS + [NEW_MODEL]}")
    logger.info(f"Models available in dataset: {sorted([m for m in rewards_map.keys() if m in OLD_MODELS + [NEW_MODEL]])}")
    return prompts, rewards_map

# ============================================================================
# EXPERIMENT LOOP
# ============================================================================
def run_adaptation_experiment():
    # 1. Setup
    prompts, rewards_map = load_real_data()
    logger.info(f"Loaded {len(prompts)} prompts")
    
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    # Context dimension (PCA + Bias)
    context_dim = pca.n_components_ + 1
    
    # Mock Costs (Normalized)
    model_costs = {
        m: {"normalized_cost": 0.1 if "mixtral" in m else 0.5} 
        for m in OLD_MODELS + [NEW_MODEL]
    }

    # 2. Initialize Routers (Start with OLD_MODELS only)
    # Both start identical
    priors_dummy = {"A": {}, "b": {}, "context_dim": context_dim}
    for m in OLD_MODELS:
        priors_dummy["A"][m] = np.eye(context_dim)
        priors_dummy["b"][m] = np.zeros(context_dim)

    # Router A: Cold Start
    router_cold = CostAwareLinUCBRouter(
        models=list(OLD_MODELS),
        warmup_priors=copy.deepcopy(priors_dummy),
        model_costs=model_costs,
        alpha_start=0.1, alpha_end=0.1, cost_penalty=0.0
    )

    # Router B: Semantic Transfer
    router_transfer = CostAwareLinUCBRouter(
        models=list(OLD_MODELS),
        warmup_priors=copy.deepcopy(priors_dummy),
        model_costs=model_costs,
        alpha_start=0.1, alpha_end=0.1, cost_penalty=0.0
    )
    
    # Metrics
    history_cold = []
    history_transfer = []
    
    # 3. Run Simulation
    logger.info(f"Starting simulation. Release at t={RELEASE_STEP}...")
    
    for t in range(min(TOTAL_STEPS, len(prompts))):
        prompt = prompts[t]
        x_emb = encoder.encode([prompt])[0]
        x_pca = pca.transform([x_emb])[0]
        x = np.concatenate([x_pca, [1.0]]) # Add bias
        
        # --- THE EVENT: MODEL RELEASE ---
        if t == RELEASE_STEP:
            logger.info(f"🚀 RELEASE EVENT! Adding {NEW_MODEL}...")
            
            # Branch A: Cold Start (Identity Init)
            router_cold.models.append(NEW_MODEL)
            router_cold.A[NEW_MODEL] = np.eye(context_dim)
            router_cold.b[NEW_MODEL] = np.zeros(context_dim)
            
            # Branch B: Semantic Transfer (The Algorithm)
            # 1. Extract Neighbor's Intuition (Theta)
            # In production, we'd use embedding similarity to find neighbor.
            # Here we hardcode 'gpt-4-turbo' as the semantic match to GPT-5.1.
            A_neighbor = router_transfer.A[NEIGHBOR_MODEL]
            b_neighbor = router_transfer.b[NEIGHBOR_MODEL]
            theta_neighbor = np.linalg.inv(A_neighbor) @ b_neighbor
            
            # 2. Transfer Intuition with Proper Bayesian Prior
            # [KDD FIX] Scale BOTH A and b to preserve mean while scaling confidence
            # A = n_eff * I (Precision scaled by effective sample size)
            # b = n_eff * theta (Moment scaled to preserve mean: theta_hat = theta)
            # This ensures variance ~ 1/n_eff (confidence increases with n_eff)
            N_effective = 5.0 # We trust the neighbor ~5 samples worth
            
            router_transfer.models.append(NEW_MODEL)
            router_transfer.A[NEW_MODEL] = N_effective * np.eye(context_dim)
            router_transfer.b[NEW_MODEL] = N_effective * theta_neighbor
            
            logger.info(f"   ✓ Transfer complete. Inherited θ from {NEIGHBOR_MODEL}")

        # --- ROUTING ---
        # Router A
        sel_cold = router_cold.select_model(x)
        # Router B
        sel_transfer = router_transfer.select_model(x)
        
        # --- REWARDS ---
        # Cycle through ground truth rewards for this model
        def get_reward(model):
            return rewards_map[model][t % len(rewards_map[model])]

        r_cold = get_reward(sel_cold)
        r_transfer = get_reward(sel_transfer)
        
        # --- UPDATE ---
        router_cold.update(x, sel_cold, r_cold)
        router_transfer.update(x, sel_transfer, r_transfer)
        
        history_cold.append(r_cold)
        history_transfer.append(r_transfer)
        
        if t % 100 == 0:
            logger.info(f"Step {t}: Cold={np.mean(history_cold[-50:]):.3f}, Transfer={np.mean(history_transfer[-50:]):.3f}")

    return history_cold, history_transfer

# ============================================================================
# PLOTTING
# ============================================================================
def plot_results(hist_cold, hist_transfer):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Moving Average
    def smooth(data, w=WINDOW_SIZE):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    y_cold = smooth(hist_cold)
    y_transfer = smooth(hist_transfer)
    x_axis = range(len(y_cold))
    
    # Shift x-axis to align with smoothing
    x_axis = [i + WINDOW_SIZE//2 for i in x_axis]
    
    plt.figure(figsize=(12, 7))
    
    # Plot Lines
    plt.plot(x_axis, y_cold, label="Baseline: Cold Start", color="#e74c3c", linewidth=2.5, alpha=0.8)
    plt.plot(x_axis, y_transfer, label="Proposed: Latent Semantic Transfer", color="#2ecc71", linewidth=3.0)
    
    # Add Release Line
    plt.axvline(x=RELEASE_STEP, color='black', linestyle='--', alpha=0.5, label="Model Release (GPT-5)")
    
    # Annotations
    plt.annotate("Cold Start Dip\n(Exploration Cost)", 
                 xy=(RELEASE_STEP + 30, np.min(y_cold[RELEASE_STEP:])), 
                 xytext=(RELEASE_STEP + 100, np.min(y_cold[RELEASE_STEP:]) - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.annotate("Zero-Shot Readiness\n(Inherited Intuition)", 
                 xy=(RELEASE_STEP + 10, y_transfer[RELEASE_STEP-WINDOW_SIZE]), 
                 xytext=(RELEASE_STEP - 200, y_transfer[RELEASE_STEP-WINDOW_SIZE] + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Figure 6: Adaptive Efficiency - Zero-Shot Readiness", fontsize=16, fontweight='bold')
    plt.xlabel("Routing Steps (t)", fontsize=12)
    plt.ylabel("Moving Average Reward (Quality)", fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.savefig(output_dir / "figure6_adaptive_efficiency.png", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved plot to {output_dir}/figure6_adaptive_efficiency.png")

if __name__ == "__main__":
    h_c, h_t = run_adaptation_experiment()
    plot_results(h_c, h_t)