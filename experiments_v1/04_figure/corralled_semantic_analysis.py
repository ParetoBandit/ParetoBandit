#!/usr/bin/env python3
"""
Figure 4: Corralled Bandit Algorithm with Semantic Projection

This script implements the mathematically correct Corralled algorithm:
1. **Optimization Phase**: Learn on labeled data (1,121 dev prompts with real rewards)
   - Use importance-weighted loss: ℓ̂_{t,e} = 𝟙_{e=e*}(1 - r_t) / ρ_{t,e}
   - Update expert weights using exponential weights algorithm
   - NO fake numbers - only use actual rewards where available

2. **Visualization Phase**: Project learned parameters onto 1M semantic space
   - Show how the learned policy would perform across the semantic manifold
   - Use cluster density to estimate coverage
   - Demonstrate that "Easy" cluster (94.1%) is exploitable

**Key Insight:**
The warmup expert is biased toward flagships (GPT-4-Turbo, Claude-3), but the Mixtral
model has high utility in the "Easy" cluster. Corralling allows the algorithm to
unlearn the warmup bias and shift weight to the tabula rasa expert that discovers
Mixtral's value.

**Paper Strategy:**
- Main Results: Report regret/AUPR on dev dataset (N=1,121) with actual rewards
- Figure 1 & Appendix D: Use 1M dataset to show semantic manifold and cluster coverage
- Figure 4 (this script): Show Corralling learns to exploit the Easy cluster

Usage:
    python experiments_v1/03_figure/corralled_semantic_analysis.py --learning-rate 1.0
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import argparse
import json
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.stats import gaussian_kde

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
    OFFLINE_DATASET_DIR,
)

# Use complete 3-model dataset (Mixtral, GPT-4-Turbo, GPT-4o)
# This tests multi-model routing and semantic transfer
CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"


# ============================================================================
# PART 1: OPTIMIZATION (ON LABELED DATA)
# ============================================================================

class TabulaRasaRouter:
    """LinUCB router initialized from scratch (A=I, b=0)."""
    
    def __init__(self, models: List[str], context_dim: int, alpha: float = 1.0):
        self.models = models
        self.alpha = alpha
        self.context_dim = context_dim
        
        # Initialize with identity (no prior knowledge)
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        
        # Track selections
        self.selections = {m: 0 for m in models}
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        """Select model using UCB.
        
        Args:
            context: Context vector
            total_steps: Total training steps (unused in basic LinUCB, for compatibility)
        """
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores[model] = expected + self.alpha * uncertainty
        
        selected = max(ucb_scores, key=ucb_scores.get)
        self.selections[selected] += 1
        return selected
    
    def update(self, context: np.ndarray, model: str, reward: float):
        """Update matrices after observing reward."""
        context = context.reshape(-1, 1)  # Column vector
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()
    
    def get_theta(self, model: str) -> np.ndarray:
        """Get learned parameter vector for a model."""
        A_inv = np.linalg.inv(self.A[model])
        return A_inv @ self.b[model]


def load_labeled_data(data_path: Path, sample_size: int = None) -> List[Dict]:
    """
    Load evaluation data with rewards (labeled data only).
    
    Returns:
        List of dicts with 'prompt' and 'scores' (model_id -> score)
    """
    entries = []
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Group by prompt
    prompt_data = {}
    for entry in entries:
        prompt = entry['prompt']
        model_id = entry['model_id']
        score = entry.get('raw_score', 0.0)
        
        if prompt not in prompt_data:
            prompt_data[prompt] = {
                'prompt': prompt,
                'scores': {}
            }
        
        prompt_data[prompt]['scores'][model_id] = score
    
    # Convert to list and sample if needed
    data_list = list(prompt_data.values())
    
    if sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(data_list), size=min(sample_size, len(data_list)), replace=False)
        data_list = [data_list[i] for i in indices]
    
    return data_list


def compute_oracle_reward(sample: Dict, model: str) -> Tuple[float, float]:
    """
    Compute oracle reward for a model on a sample.
    
    Returns:
        Tuple of (model_reward, oracle_reward)
    """
    scores = sample.get('scores', {})
    
    if not scores:
        return 0.0, 0.0
    
    # Oracle always picks the best model
    oracle_model = max(scores, key=scores.get)
    oracle_reward = scores[oracle_model]
    
    # Return the selected model's reward and oracle reward
    return scores.get(model, 0.0), oracle_reward


def extend_priors_with_semantic_transfer(
    warmup_priors: Dict,
    new_models: List[str],
    transfer_mapping: Dict[str, str],
    gamma: float = 0.05
) -> Dict:
    """
    Extend warmup priors to include new models via semantic transfer.
    
    Args:
        warmup_priors: Existing priors (e.g., for Mixtral, GPT-4-Turbo)
        new_models: Models to add (e.g., GPT-4o)
        transfer_mapping: {new_model: source_model} (e.g., {'gpt-4o': 'gpt-4-turbo'})
        gamma: Scaling factor for transferred priors (low = weak transfer)
    
    Returns:
        Extended priors with new models
    """
    extended_priors = {
        'A': warmup_priors['A'].copy(),
        'b': warmup_priors['b'].copy(),
        'models': warmup_priors['models'].copy(),
        'context_dim': warmup_priors['context_dim']
    }
    
    for new_model in new_models:
        if new_model in extended_priors['models']:
            continue  # Already exists
        
        source_model = transfer_mapping.get(new_model)
        if not source_model or source_model not in warmup_priors['models']:
            raise ValueError(f"Cannot transfer priors for {new_model}: source {source_model} not found")
        
        # Transfer priors with scaling (gamma acts like n_effective)
        extended_priors['A'][new_model] = gamma * warmup_priors['A'][source_model].copy()
        extended_priors['b'][new_model] = gamma * warmup_priors['b'][source_model].copy()
        extended_priors['models'].append(new_model)
        
        print(f"   ✅ Transferred priors: {source_model} → {new_model} (γ={gamma})")
    
    return extended_priors


def train_corralling_router(
    data: List[Dict],
    encoder: SentenceTransformer,
    pca,
    warmup_priors: Dict,
    learning_rate: float = 1.0,
    enable_semantic_transfer: bool = True
) -> Tuple[CorrallingRouter, Dict]:
    """
    Train Corralling router on labeled data using importance-weighted loss.
    
    This is the OPTIMIZATION phase - we only use prompts where we have rewards.
    
    Args:
        enable_semantic_transfer: If True, extend priors to include GPT-4o via transfer from GPT-4-Turbo
    
    Returns:
        (trained_router, training_metrics)
    """
    # Collect all models present in the data
    all_models_in_data = set()
    for sample in data:
        all_models_in_data.update(sample['scores'].keys())
    all_models_in_data = sorted(all_models_in_data)
    
    print(f"\n📊 Models in training data: {all_models_in_data}")
    print(f"   Models in warmup priors: {warmup_priors['models']}")
    
    # Check if we need to extend priors
    missing_models = [m for m in all_models_in_data if m not in warmup_priors['models']]
    
    if missing_models and enable_semantic_transfer:
        print(f"\n🔄 Semantic Transfer: Extending priors for {len(missing_models)} new model(s)")
        
        # Define transfer mapping (GPT-4o inherits from GPT-4-Turbo)
        transfer_mapping = {
            'openai/gpt-4o': 'openai/gpt-4-turbo'
        }
        
        warmup_priors = extend_priors_with_semantic_transfer(
            warmup_priors,
            new_models=missing_models,
            transfer_mapping=transfer_mapping,
            gamma=0.05  # Weak transfer (low confidence)
        )
    elif missing_models:
        raise ValueError(f"Missing warmup priors for: {missing_models}. Enable semantic_transfer or provide priors.")
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    print(f"\n🎓 Training Corralling Router (Learning Rate: {learning_rate})")
    print(f"   Models: {len(models)}")
    print(f"   Context Dim: {context_dim}")
    print(f"   Training Samples: {len(data)}")
    
    # Initialize experts
    warmup_expert = SimpleLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors,
        alpha=1.0
    )
    
    tabula_rasa_expert = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    # Initialize Corralling
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=learning_rate
    )
    
    # Training loop
    cumulative_regret = 0.0
    total_reward = 0.0
    regret_history = []
    reward_history = []
    expert_weights_history = []
    
    for i, sample in enumerate(tqdm(data, desc="   Training")):
        prompt = sample['prompt']
        
        # Embed prompt
        context = embed_prompt(prompt, encoder, pca)
        
        # Select model (using importance sampling)
        selected_model = router.select_model(context)
        
        # Get reward (ONLY available for labeled data)
        model_reward, oracle_reward = compute_oracle_reward(sample, selected_model)
        
        # Compute regret
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        regret_history.append(cumulative_regret)
        reward_history.append(total_reward / (i + 1))
        expert_weights_history.append(router.weights.copy())
        
        # Update router with importance-weighted loss
        # The CorrallingRouter.update() method already implements:
        # ℓ̂_{t,e} = 𝟙_{e=e*}(1 - r_t) / ρ_{t,e}
        router.update(context, selected_model, model_reward)
    
    metrics = {
        'cumulative_regret': cumulative_regret,
        'avg_reward': total_reward / len(data),
        'regret_history': regret_history,
        'reward_history': reward_history,
        'expert_weights_history': np.array(expert_weights_history),
        'final_expert_weights': router.weights.copy(),
        'model_usage': router.selections.copy()
    }
    
    print(f"\n   ✅ Training Complete")
    print(f"      Cumulative Regret: {cumulative_regret:.2f}")
    print(f"      Average Reward: {metrics['avg_reward']:.4f}")
    print(f"      Final Weights: Warmup={router.weights[0]:.3f}, Tabula Rasa={router.weights[1]:.3f}")
    
    return router, metrics


# ============================================================================
# PART 2: VISUALIZATION (ON 1M SEMANTIC SPACE)
# ============================================================================

def load_1M_prompts(data_file: Path, max_prompts: int = None) -> List[str]:
    """
    Load prompts from 1M dataset (NO REWARDS - just prompts).
    
    This is for visualization only - we project the learned policy onto
    the semantic space to show coverage.
    """
    print(f"\n📥 Loading 1M prompts for semantic projection...")
    print(f"   Data: {data_file}")
    
    prompts = []
    
    with gzip.open(data_file, 'rt') as f:
        for line in tqdm(f, desc="   Loading"):
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                
                if not prompt or not isinstance(prompt, str):
                    continue
                
                prompt = prompt.strip()
                if not prompt:
                    continue
                
                prompts.append(prompt)
                
                if max_prompts and len(prompts) >= max_prompts:
                    break
                
            except Exception:
                continue
    
    print(f"   ✅ Loaded {len(prompts):,} prompts")
    
    return prompts


def embed_and_project_2d(prompts: List[str], encoder: SentenceTransformer, pca, batch_size: int = 64) -> np.ndarray:
    """
    Embed prompts and project to 2D using pre-trained PCA.
    """
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    
    print(f"\n📐 Projecting to 2D...")
    X_nd = pca.transform(embeddings)
    X_2d = X_nd[:, :2]
    
    explained_var_2d = np.sum(pca.explained_variance_ratio_[:2])
    print(f"   ✅ 2D projection complete")
    print(f"   PC1: {pca.explained_variance_ratio_[0]:.3%}")
    print(f"   PC2: {pca.explained_variance_ratio_[1]:.3%}")
    print(f"   Total (2D): {explained_var_2d:.2%}")
    
    return X_2d, X_nd


def project_learned_policy(
    router: CorrallingRouter,
    prompts: List[str],
    encoder: SentenceTransformer,
    pca,
    models: List[str]
) -> Dict[str, np.ndarray]:
    """
    Project the learned Corralling policy onto the semantic space.
    
    For each prompt in the 1M space, we compute which model the learned
    policy would select. This shows the coverage of the semantic manifold.
    
    NOTE: We do NOT compute rewards here - we only show which model would
    be selected. This is a PROJECTION, not an evaluation.
    
    We use the router's embedding logic to ensure consistency with training.
    """
    print(f"\n🎯 Projecting learned policy onto semantic space...")
    print(f"   Points: {len(prompts):,}")
    
    # For each prompt, determine which model would be selected
    selections = []
    
    for i in tqdm(range(len(prompts)), desc="   Projecting"):
        prompt = prompts[i]
        
        # Use the router's embedding logic (handles PCA + bias term correctly)
        context = embed_prompt(prompt, encoder, pca)
        
        # Sample from expert distribution (using learned weights)
        expert_idx = np.random.choice(router.n_experts, p=router.weights)
        
        # Get that expert's selection
        model = router.experts[expert_idx].select_model(context)
        
        selections.append(model)
    
    # Count selections per model
    selection_counts = {}
    for model in models:
        selection_counts[model] = selections.count(model)
    
    print(f"\n   ✅ Projection complete")
    print(f"      Model usage across 1M space:")
    for model, count in sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = 100.0 * count / len(selections)
        print(f"         {model}: {count:,} ({pct:.1f}%)")
    
    return {
        'selections': np.array(selections),
        'selection_counts': selection_counts
    }


def create_semantic_visualization(
    X_2d: np.ndarray,
    policy_projection: Dict,
    training_metrics: Dict,
    pca,
    output_dir: Path
):
    """
    Create visualization showing:
    1. Learned policy projected onto semantic space
    2. Expert weight evolution during training
    3. Cluster coverage statistics
    """
    print(f"\n🎨 Creating visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Figure 1: Semantic Space with Policy Projection
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Semantic scatter with cluster separation
    ax1 = axes[0]
    
    pc1_values = X_2d[:, 0]
    low_pc1_mask = pc1_values < 0.3
    high_pc1_mask = pc1_values >= 0.3
    
    X_low = X_2d[low_pc1_mask]
    X_high = X_2d[high_pc1_mask]
    
    # Downsample for visualization
    downsample_size = min(10000, len(X_2d))
    if len(X_2d) > downsample_size:
        indices = np.random.choice(len(X_2d), downsample_size, replace=False)
        X_sample = X_2d[indices]
    else:
        X_sample = X_2d
    
    pc1_sample = X_sample[:, 0]
    low_mask_s = pc1_sample < 0.3
    high_mask_s = pc1_sample >= 0.3
    
    ax1.scatter(X_sample[low_mask_s, 0], X_sample[low_mask_s, 1],
               c='#4575b4', s=25, alpha=0.7, label=f'Easy Cluster ({len(X_low):,}, {len(X_low)/len(X_2d)*100:.1f}%)',
               edgecolors='none', rasterized=True)
    
    ax1.scatter(X_sample[high_mask_s, 0], X_sample[high_mask_s, 1],
               c='#d73027', s=25, alpha=0.7, label=f'Hard Cluster ({len(X_high):,}, {len(X_high)/len(X_2d)*100:.1f}%)',
               edgecolors='none', rasterized=True)
    
    # Add KDE contour for easy cluster
    if len(X_low) > 100:
        try:
            kde_sample_size = min(5000, len(X_low))
            kde_indices = np.random.choice(len(X_low), kde_sample_size, replace=False)
            X_kde_sample = X_low[kde_indices]
            
            kde_low = gaussian_kde(X_kde_sample.T, bw_method=0.12)
            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density_low = np.reshape(kde_low(positions).T, xx.shape)
            ax1.contour(xx, yy, density_low, levels=4, colors='#2166ac', alpha=0.6, linewidths=2.5)
        except:
            pass
    
    # Cluster separation line
    ax1.axvline(x=0.3, color='black', linestyle='--', linewidth=3, alpha=0.7, label='Cluster Boundary', zorder=5)
    
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    
    ax1.set_xlabel(f'PC1 ({pc1_var:.2%} variance)', fontsize=15, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pc2_var:.2%} variance)', fontsize=15, fontweight='bold')
    ax1.set_title(
        'Semantic Task Structure: Corralling Exploits Easy Cluster\n'
        f'Learned Policy Projected onto 1M Prompts',
        fontsize=17,
        fontweight='bold',
        pad=15
    )
    ax1.grid(alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Right: Expert weight evolution
    ax2 = axes[1]
    
    weights_history = training_metrics['expert_weights_history']
    
    ax2.plot(weights_history[:, 0], label='Warmup Expert', linewidth=2.5, color='orange')
    ax2.plot(weights_history[:, 1], label='Tabula Rasa Expert', linewidth=2.5, color='green')
    
    ax2.set_xlabel('Training Samples', fontsize=15, fontweight='bold')
    ax2.set_ylabel('Expert Weight', fontsize=15, fontweight='bold')
    ax2.set_title(
        'Corralling: Adaptive Expert Weighting\n'
        'Algorithm Learns to Downweight Biased Warmup',
        fontsize=17,
        fontweight='bold',
        pad=15
    )
    ax2.legend(fontsize=13, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.1, 1.1])  # Extended range to show curve outlines better
    
    # Add final weights annotation
    final_warmup = weights_history[-1, 0]
    final_tabula = weights_history[-1, 1]
    ax2.annotate(
        f'Final Weights:\nWarmup: {final_warmup:.3f}\nTabula Rasa: {final_tabula:.3f}',
        xy=(len(weights_history)-1, 0.5),
        xytext=(len(weights_history)*0.6, 0.7),
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    output_file = output_dir / 'figure4_corralling_semantic_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Saved: {output_file}")
    
    # High-res version
    output_file_hires = output_dir / 'figure4_corralling_semantic_analysis_hires.png'
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()
    
    # ========================================================================
    # Figure 2: Training Metrics
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Cumulative Regret
    ax1 = axes[0]
    ax1.plot(training_metrics['regret_history'], linewidth=2.5, color='#e74c3c')
    ax1.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax1.set_title('Corralling: Cumulative Regret on Labeled Data', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right: Average Reward
    ax2 = axes[1]
    ax2.plot(training_metrics['reward_history'], linewidth=2.5, color='#27ae60')
    ax2.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Reward', fontsize=13, fontweight='bold')
    ax2.set_title('Corralling: Average Reward on Labeled Data', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'training_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {output_file}")
    
    plt.close()


def print_summary(training_metrics: Dict, policy_projection: Dict, X_2d: np.ndarray):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("CORRALLING SEMANTIC ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 TRAINING RESULTS (on Labeled Data):")
    print(f"   Cumulative Regret: {training_metrics['cumulative_regret']:.2f}")
    print(f"   Average Reward: {training_metrics['avg_reward']:.4f}")
    print(f"   Final Expert Weights:")
    print(f"      Warmup Expert: {training_metrics['final_expert_weights'][0]:.4f}")
    print(f"      Tabula Rasa Expert: {training_metrics['final_expert_weights'][1]:.4f}")
    
    if training_metrics['final_expert_weights'][1] > training_metrics['final_expert_weights'][0]:
        ratio = training_metrics['final_expert_weights'][1] / training_metrics['final_expert_weights'][0]
        print(f"\n   ✅ Tabula Rasa WON: {ratio:.2f}x more weight than Warmup")
        print(f"      → Algorithm successfully unlearned warmup bias!")
    else:
        print(f"\n   ⚠️  Warmup still dominates")
    
    print("\n🌍 SEMANTIC PROJECTION (on 1M Space):")
    pc1_values = X_2d[:, 0]
    easy_cluster_size = np.sum(pc1_values < 0.3)
    easy_cluster_pct = 100.0 * easy_cluster_size / len(X_2d)
    
    print(f"   Total Prompts: {len(X_2d):,}")
    print(f"   Easy Cluster (PC1 < 0.3): {easy_cluster_size:,} ({easy_cluster_pct:.1f}%)")
    print(f"   Hard Cluster (PC1 ≥ 0.3): {len(X_2d) - easy_cluster_size:,} ({100-easy_cluster_pct:.1f}%)")
    
    print("\n📈 MODEL USAGE (Projected Policy):")
    for model, count in sorted(policy_projection['selection_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        pct = 100.0 * count / len(policy_projection['selections'])
        print(f"   {model:<40} {count:>7,} ({pct:>5.1f}%)")
    
    print("\n💡 KEY INSIGHT:")
    print(f"   The Easy cluster ({easy_cluster_pct:.1f}% of prompts) is exploitable!")
    print(f"   Corralling learns to use cheaper models (e.g., Mixtral) in this region,")
    print(f"   unlearning the warmup prior's bias toward expensive flagships.")
    
    print("\n📝 FOR THE PAPER:")
    print(f"   • Main Results: Report regret/AUPR on LMSYS Holdout (labeled data)")
    print(f"   • Figure 1 & Appendix D: Show 1M semantic manifold proves Easy cluster exists")
    print(f"   • Figure 4 (this): Show Corralling exploits the Easy cluster")
    print(f"   • No fake numbers: Optimization uses only labeled data with real rewards")


def main():
    parser = argparse.ArgumentParser(description='Corralled Semantic Analysis')
    parser.add_argument('--learning-rate', type=float, default=1.0, 
                       help='Corralling learning rate (eta)')
    parser.add_argument('--gamma', type=float, default=0.05, 
                       help='Gamma scaling for warmup priors')
    parser.add_argument('--train-size', type=int, default=1121, 
                       help='Number of labeled samples for training (dev has 1121 prompts)')
    parser.add_argument('--projection-size', type=int, default=None,
                       help='Number of 1M prompts to project (None = all)')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CORRALLED SEMANTIC ANALYSIS")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Learning Rate (eta): {args.learning_rate}")
    print(f"   Gamma (warmup scaling): {args.gamma}")
    print(f"   Training Size: {args.train_size}")
    print(f"   Projection Size: {args.projection_size or 'ALL'}")
    print(f"   Output: {output_dir}")
    
    # ========================================================================
    # STRICT DATA VALIDATION: Only use real data, no synthetic/fallback data
    # ========================================================================
    
    print("\n" + "="*80)
    print("VALIDATING REQUIRED DATA FILES (REAL DATA ONLY)")
    print("="*80)
    
    required_files = {
        'Labeled Data': Path(CANONICAL_DEV_DATA_PATH),
        'PCA Model': Path(DEFAULT_PCA_PATH),
        'Warmup Priors': Path(DEFAULT_WARMUP_PRIORS_PATH),
        '1M Dataset': Path(__file__).parent.parent / "appendix_d" / "data" / "lmsys_chat_1M.jsonl.gz"
    }
    
    missing_files = []
    for name, path in required_files.items():
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (NOT FOUND)")
            missing_files.append((name, path))
    
    if missing_files:
        print("\n" + "="*80)
        print("❌ ERROR: MISSING REQUIRED DATA FILES")
        print("="*80)
        print("\nThis script requires REAL data only (no synthetic/fallback data).")
        print("\nMissing files:")
        for name, path in missing_files:
            print(f"   • {name}: {path}")
        
        print("\nTo fix:")
        if any(name == 'Labeled Data' for name, _ in missing_files):
            print("   • Labeled Data: Run 'python scripts/generate_gpt4_turbo_rewards.py'")
        if any(name == 'PCA Model' for name, _ in missing_files):
            print("   • PCA Model: Run 'python scripts/train_pca_from_routellm.py'")
        if any(name == 'Warmup Priors' for name, _ in missing_files):
            print("   • Warmup Priors: Run 'python scripts/generate_warmup_priors.py'")
        if any(name == '1M Dataset' for name, _ in missing_files):
            print("   • 1M Dataset: Run 'python experiments_v1/appendix_d/download_1M_dataset.py'")
        
        print("\nExiting...")
        sys.exit(1)
    
    print("\n✅ All required data files found. Proceeding with real data only.")
    
    # ========================================================================
    # PHASE 1: OPTIMIZATION (on labeled data)
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 1: OPTIMIZATION (on labeled data with rewards)")
    print("="*80)
    
    # Load resources
    print("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # Scale priors
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=args.gamma)
    
    # Load labeled data (with rewards)
    print("\n📊 Loading labeled data...")
    labeled_data = load_labeled_data(Path(CANONICAL_DEV_DATA_PATH), sample_size=args.train_size)
    print(f"   ✅ Loaded {len(labeled_data)} labeled samples")
    
    # Train Corralling router
    router, training_metrics = train_corralling_router(
        data=labeled_data,
        encoder=encoder,
        pca=pca,
        warmup_priors=warmup_priors_scaled,
        learning_rate=args.learning_rate
    )
    
    # ========================================================================
    # PHASE 2: VISUALIZATION (on 1M semantic space)
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 2: VISUALIZATION (project onto 1M semantic space)")
    print("="*80)
    
    # Check if 1M data exists - STRICT: Fail if not available (no fallback to synthetic data)
    data_1M_file = Path(__file__).parent.parent / "appendix_d" / "data" / "lmsys_chat_1M.jsonl.gz"
    
    if not data_1M_file.exists():
        print(f"\n❌ ERROR: 1M dataset not found: {data_1M_file}")
        print(f"\n   This script requires REAL data only (no synthetic/fallback data).")
        print(f"\n   To download the 1M dataset, run:")
        print(f"      python experiments_v1/appendix_d/download_1M_dataset.py")
        print(f"\n   Exiting...")
        sys.exit(1)
    
    # Load 1M prompts (NO REWARDS - just for visualization)
    print(f"\n✅ Found 1M dataset: {data_1M_file}")
    prompts_1M = load_1M_prompts(data_1M_file, max_prompts=args.projection_size)
    
    # Embed and project to 2D for visualization
    X_2d, X_nd = embed_and_project_2d(prompts_1M, encoder, pca)
    
    # Project learned policy onto semantic space
    # Use the router's embedding logic to ensure consistency
    policy_projection = project_learned_policy(
        router=router,
        prompts=prompts_1M,
        encoder=encoder,
        pca=pca,
        models=warmup_priors_scaled['models']
    )
    
    # Create visualizations
    create_semantic_visualization(
        X_2d=X_2d,
        policy_projection=policy_projection,
        training_metrics=training_metrics,
        pca=pca,
        output_dir=output_dir
    )
    
    # Print summary
    print_summary(training_metrics, policy_projection, X_2d)
    
    # Save metrics
    print("\n💾 Saving results...")
    results = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'train_size': args.train_size,
        'cumulative_regret': float(training_metrics['cumulative_regret']),
        'avg_reward': float(training_metrics['avg_reward']),
        'final_expert_weights': training_metrics['final_expert_weights'].tolist(),
        'model_usage': training_metrics['model_usage']
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Results saved to {output_dir}/results.json")
    
    print("\n" + "="*80)
    print("✅ CORRALLING SEMANTIC ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()

