"""
Diagnostic Analysis: Why Does Semantic Transfer Work?
======================================================

Despite lack of correlation between semantic similarity and performance correlation,
semantic transfer might still provide benefit through alternative mechanisms:

1. MAGNITUDE TRANSFER: Inheriting "task difficulty" signals (absolute values)
2. IMPLICIT REGULARIZATION: Any warm start better than cold start
3. DIRECTIONAL TRANSFER: Inheriting "this model is generally better" signal
4. COINCIDENTAL ALIGNMENT: Transfer helps on the 20% of tasks where GPT-5.1 wins

This script diagnoses which mechanism(s) explain any observed benefit.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import logging
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS
)
import joblib
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
MODELS_3 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo", "openai/gpt-5.1"]
NEW_MODEL = "openai/gpt-5.1"

# ============================================================================
# HYPOTHESIS 1: MAGNITUDE TRANSFER
# ============================================================================
def test_magnitude_transfer(data: List):
    """
    Test if semantic transfer works by transferring MAGNITUDES (task difficulty)
    rather than DIRECTIONS (relative preferences).
    
    Hypothesis: Even if models have uncorrelated preferences, transferring
    "this is a hard task" signal (high uncertainty) might help.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 HYPOTHESIS 1: MAGNITUDE TRANSFER")
    logger.info("="*80)
    logger.info("Testing: Does transfer help by inheriting 'task difficulty' signals?")
    logger.info("="*80)
    
    # Load warmup priors
    priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    
    # Get GPT-4's preference vector (what would be transferred)
    gpt4_model = "openai/gpt-4-turbo"
    if gpt4_model in priors['A'] and gpt4_model in priors['b']:
        A_gpt4 = priors['A'][gpt4_model]
        b_gpt4 = priors['b'][gpt4_model]
        theta_gpt4 = np.linalg.solve(A_gpt4, b_gpt4)  # Preference vector
        
        logger.info(f"\n📊 GPT-4-Turbo Preference Vector Statistics:")
        logger.info(f"   Mean: {np.mean(theta_gpt4):.4f}")
        logger.info(f"   Std:  {np.std(theta_gpt4):.4f}")
        logger.info(f"   Min:  {np.min(theta_gpt4):.4f}")
        logger.info(f"   Max:  {np.max(theta_gpt4):.4f}")
        logger.info(f"   L2 norm: {np.linalg.norm(theta_gpt4):.4f}")
    
    # Analyze how GPT-4 and GPT-5.1 rewards correlate with preference magnitudes
    gpt4_rewards = []
    gpt5_rewards = []
    context_features = []
    
    for item in data[:200]:  # Sample for efficiency
        # Get context features
        embedding = encoder.encode(item.prompt)
        pca_features = pca.transform(embedding.reshape(1, -1))[0]
        context_features.append(pca_features)
        
        gpt4_rewards.append(item.rewards.get(gpt4_model, 0))
        gpt5_rewards.append(item.rewards.get(NEW_MODEL, 0))
    
    context_features = np.array(context_features)
    
    # Compute "predicted difficulty" using GPT-4's preference vector
    if gpt4_model in priors['A']:
        # Handle dimension mismatch (theta might have bias term)
        if len(theta_gpt4) != context_features.shape[1]:
            logger.info(f"\n⚠️  Dimension mismatch: theta={len(theta_gpt4)}, features={context_features.shape[1]}")
            logger.info("   Adjusting by truncating theta to match PCA dimensions")
            theta_gpt4 = theta_gpt4[:context_features.shape[1]]
        
        predicted_values = context_features @ theta_gpt4
        
        # Test 1: Do high absolute predicted values correlate with actual performance?
        logger.info("\n📊 Magnitude Transfer Analysis:")
        logger.info("   Testing if |θ_GPT4 · x| predicts task outcomes...")
        
        abs_predictions = np.abs(predicted_values)
        
        # Correlation between magnitude and actual rewards
        corr_gpt4_mag, p_gpt4 = stats.pearsonr(abs_predictions, gpt4_rewards)
        corr_gpt5_mag, p_gpt5 = stats.pearsonr(abs_predictions, gpt5_rewards)
        
        logger.info(f"\n   GPT-4 Reward vs |Prediction|: r={corr_gpt4_mag:.3f}, p={p_gpt4:.4f}")
        logger.info(f"   GPT-5.1 Reward vs |Prediction|: r={corr_gpt5_mag:.3f}, p={p_gpt5:.4f}")
        
        if abs(corr_gpt5_mag) > 0.1 and p_gpt5 < 0.05:
            logger.info("\n   ✅ MAGNITUDE TRANSFER MAY BE ACTIVE")
            logger.info("      GPT-4's certainty about tasks transfers to GPT-5.1")
        else:
            logger.info("\n   ❌ NO EVIDENCE FOR MAGNITUDE TRANSFER")
    
    logger.info("="*80 + "\n")

# ============================================================================
# HYPOTHESIS 2: DIRECTIONAL TRANSFER
# ============================================================================
def test_directional_transfer(data: List):
    """
    Test if transfer works by inheriting directional preference
    (e.g., "GPT models generally better than Mixtral on these features").
    
    Even if GPT-4 and GPT-5.1 preferences are uncorrelated, maybe
    "anti-Mixtral" signal helps (positive values = good for GPT models).
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 HYPOTHESIS 2: DIRECTIONAL TRANSFER")
    logger.info("="*80)
    logger.info("Testing: Does transfer help by inheriting 'GPT > Mixtral' signal?")
    logger.info("="*80)
    
    # Analyze task subsets where different models win
    mixtral_wins_tasks = []
    gpt4_wins_tasks = []
    gpt5_wins_tasks = []
    ties = []
    
    mixtral = "mistralai/mixtral-8x7b-instruct"
    gpt4 = "openai/gpt-4-turbo"
    gpt5 = "openai/gpt-5.1"
    
    for item in data:
        r_m = item.rewards.get(mixtral, 0)
        r_g4 = item.rewards.get(gpt4, 0)
        r_g5 = item.rewards.get(gpt5, 0)
        
        if r_g5 > r_m and r_g5 > r_g4:
            gpt5_wins_tasks.append(item)
        elif r_g4 > r_m and r_g4 >= r_g5:
            gpt4_wins_tasks.append(item)
        elif r_m > r_g4 and r_m > r_g5:
            mixtral_wins_tasks.append(item)
        else:
            ties.append(item)
    
    logger.info(f"\n📊 Task Breakdown:")
    logger.info(f"   GPT-5.1 solo wins: {len(gpt5_wins_tasks)}")
    logger.info(f"   GPT-4 solo wins: {len(gpt4_wins_tasks)}")
    logger.info(f"   Mixtral solo wins: {len(mixtral_wins_tasks)}")
    logger.info(f"   Ties: {len(ties)}")
    
    # Key test: On tasks where GPT-5.1 wins, did GPT-4 also beat Mixtral?
    if gpt5_wins_tasks:
        gpt4_also_beats_mixtral = 0
        for item in gpt5_wins_tasks:
            if item.rewards.get(gpt4, 0) > item.rewards.get(mixtral, 0):
                gpt4_also_beats_mixtral += 1
        
        alignment_rate = gpt4_also_beats_mixtral / len(gpt5_wins_tasks)
        logger.info(f"\n📊 Directional Alignment:")
        logger.info(f"   On {len(gpt5_wins_tasks)} tasks where GPT-5.1 wins,")
        logger.info(f"   GPT-4 also beats Mixtral: {gpt4_also_beats_mixtral} ({alignment_rate:.1%})")
        
        if alignment_rate > 0.7:
            logger.info("\n   ✅ STRONG DIRECTIONAL ALIGNMENT")
            logger.info("      'GPT > Mixtral' signal transfers effectively")
        elif alignment_rate > 0.5:
            logger.info("\n   ⚠️  WEAK DIRECTIONAL ALIGNMENT")
            logger.info("      Some directional transfer, but inconsistent")
        else:
            logger.info("\n   ❌ NO DIRECTIONAL ALIGNMENT")
    
    logger.info("="*80 + "\n")

# ============================================================================
# HYPOTHESIS 3: IMPLICIT REGULARIZATION
# ============================================================================
def test_implicit_regularization(data: List):
    """
    Test if ANY warm start (even random) is better than cold start,
    suggesting benefit comes from regularization not semantic content.
    
    Compare:
    - Cold start (A=λI, b=0)
    - Random prior (A=λI, b=random)
    - Semantic transfer (A=λI, b=N_eff * θ_GPT4)
    
    If Random ≈ Semantic > Cold, then benefit is regularization.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 HYPOTHESIS 3: IMPLICIT REGULARIZATION")
    logger.info("="*80)
    logger.info("Testing: Is ANY warm start better than cold start?")
    logger.info("="*80)
    
    # Load priors
    priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    gpt4_model = "openai/gpt-4-turbo"
    
    if gpt4_model in priors['A'] and gpt4_model in priors['b']:
        A_gpt4 = priors['A'][gpt4_model]
        b_gpt4 = priors['b'][gpt4_model]
        theta_gpt4 = np.linalg.solve(A_gpt4, b_gpt4)
        
        # Match PCA dimensions
        pca_dim = 32
        theta_gpt4_truncated = theta_gpt4[:pca_dim]
        
        dim = pca_dim
        lambda_init = 0.1  # Typical initialization
        N_eff = 5.0  # Semantic transfer strength
        
        # Three initialization strategies
        cold_b = np.zeros(dim)
        random_b = np.random.randn(dim) * np.std(theta_gpt4_truncated)  # Random with same scale
        semantic_b = N_eff * theta_gpt4_truncated
        
        logger.info(f"\n📊 Prior Characteristics (dim={dim}):")
        logger.info(f"   Cold Start: ||b|| = {np.linalg.norm(cold_b):.4f}")
        logger.info(f"   Random: ||b|| = {np.linalg.norm(random_b):.4f}")
        logger.info(f"   Semantic: ||b|| = {np.linalg.norm(semantic_b):.4f}")
        
        # Compute initial predictions for sample contexts
        encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
        pca = joblib.load(DEFAULT_PCA_PATH)
        
        sample_contexts = []
        for item in data[:100]:
            embedding = encoder.encode(item.prompt)
            pca_features = pca.transform(embedding.reshape(1, -1))[0]
            sample_contexts.append(pca_features)
        
        sample_contexts = np.array(sample_contexts)
        
        # Compute initial predictions (before any learning)
        cold_preds = sample_contexts @ cold_b
        random_preds = sample_contexts @ random_b
        semantic_preds = sample_contexts @ semantic_b
        
        logger.info(f"\n📊 Initial Prediction Variance (before learning):")
        logger.info(f"   Cold Start: σ² = {np.var(cold_preds):.4f}")
        logger.info(f"   Random: σ² = {np.var(random_preds):.4f}")
        logger.info(f"   Semantic: σ² = {np.var(semantic_preds):.4f}")
        
        # Hypothesis: Non-zero variance provides implicit regularization
        # by breaking symmetry and allowing exploration
        if np.var(semantic_preds) > 10 * np.var(cold_preds):
            logger.info("\n   ✅ STRONG REGULARIZATION EFFECT")
            logger.info("      Semantic prior provides meaningful initial variance")
        elif np.var(random_preds) > np.var(cold_preds):
            logger.info("\n   ⚠️  REGULARIZATION FROM ANY PRIOR")
            logger.info("      Random prior also breaks symmetry")
        else:
            logger.info("\n   ❌ NO CLEAR REGULARIZATION BENEFIT")
    
    logger.info("="*80 + "\n")

# ============================================================================
# HYPOTHESIS 4: COINCIDENTAL ALIGNMENT
# ============================================================================
def test_coincidental_alignment(data: List):
    """
    Test if benefit comes from coincidental alignment on the 20% of tasks
    where GPT-5.1 actually outperforms GPT-4.
    
    If semantic transfer helps mainly on these tasks (not the 71.5% ties),
    then benefit is task-specific coincidence rather than general principle.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 HYPOTHESIS 4: COINCIDENTAL ALIGNMENT")
    logger.info("="*80)
    logger.info("Testing: Does transfer mainly help on tasks where GPT-5.1 excels?")
    logger.info("="*80)
    
    # Categorize tasks
    gpt5_better = []
    gpt5_worse = []
    ties = []
    
    gpt4 = "openai/gpt-4-turbo"
    gpt5 = "openai/gpt-5.1"
    
    for item in data:
        r_g4 = item.rewards.get(gpt4, 0)
        r_g5 = item.rewards.get(gpt5, 0)
        
        if r_g5 > r_g4:
            gpt5_better.append(item)
        elif r_g4 > r_g5:
            gpt5_worse.append(item)
        else:
            ties.append(item)
    
    logger.info(f"\n📊 Task Categories:")
    logger.info(f"   GPT-5.1 > GPT-4: {len(gpt5_better)} ({len(gpt5_better)/len(data):.1%})")
    logger.info(f"   GPT-4 > GPT-5.1: {len(gpt5_worse)} ({len(gpt5_worse)/len(data):.1%})")
    logger.info(f"   Ties: {len(ties)} ({len(ties)/len(data):.1%})")
    
    # Load GPT-4 preference and compute alignment on each category
    priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    if gpt4 in priors['A'] and gpt4 in priors['b']:
        encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
        pca = joblib.load(DEFAULT_PCA_PATH)
        
        A_gpt4 = priors['A'][gpt4]
        b_gpt4 = priors['b'][gpt4]
        theta_gpt4 = np.linalg.solve(A_gpt4, b_gpt4)
        
        # Match PCA dimensions
        pca_dim = 32
        theta_gpt4 = theta_gpt4[:pca_dim]
        
        def compute_alignment_score(tasks):
            if not tasks:
                return 0, 0
            scores = []
            for item in tasks[:50]:  # Sample for efficiency
                embedding = encoder.encode(item.prompt)
                pca_features = pca.transform(embedding.reshape(1, -1))[0]
                score = pca_features @ theta_gpt4
                scores.append(score)
            return np.mean(scores), np.std(scores)
        
        mean_better, std_better = compute_alignment_score(gpt5_better)
        mean_worse, std_worse = compute_alignment_score(gpt5_worse)
        mean_ties, std_ties = compute_alignment_score(ties)
        
        logger.info(f"\n📊 GPT-4 Preference Alignment:")
        logger.info(f"   Tasks where GPT-5.1 wins: μ={mean_better:.3f}, σ={std_better:.3f}")
        logger.info(f"   Tasks where GPT-4 wins:   μ={mean_worse:.3f}, σ={std_worse:.3f}")
        logger.info(f"   Ties:                     μ={mean_ties:.3f}, σ={std_ties:.3f}")
        
        if mean_better > mean_worse + std_worse:
            logger.info("\n   ✅ COINCIDENTAL ALIGNMENT DETECTED")
            logger.info("      GPT-4's preference aligns with GPT-5.1's strengths")
        else:
            logger.info("\n   ❌ NO COINCIDENTAL ALIGNMENT")
    
    logger.info("="*80 + "\n")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("🔬 DIAGNOSTIC ANALYSIS: WHY DOES SEMANTIC TRANSFER WORK?")
    logger.info("="*80)
    logger.info("Testing 4 alternative mechanisms despite lack of correlation validation")
    logger.info("="*80 + "\n")
    
    # Load data
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=MODELS_3
    )
    data = [item for item in evaluator if all(m in item.rewards for m in MODELS_3)]
    logger.info(f"✅ Loaded {len(data)} samples\n")
    
    # Run all diagnostic tests
    test_magnitude_transfer(data)
    test_directional_transfer(data)
    test_implicit_regularization(data)
    test_coincidental_alignment(data)
    
    logger.info("="*80)
    logger.info("🔬 DIAGNOSTIC ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info("Review results above to understand which mechanism(s) explain")
    logger.info("any observed benefit of semantic transfer.")
    logger.info("="*80)
