#!/usr/bin/env python3
"""
Pareto Frontier Data Generation — banditGPT vs RouteLLM-MF

Sweeps cost-penalty (λ) for banditGPT and threshold for RouteLLM-MF
to produce (cost, quality) operating points.  The resulting JSON is
consumed by generate_figure4.py, which renders the two-panel publication
figure (Pareto frontier + learning curve).

Dataset: Combined dev + holdout (N=1,871 prompts) — real data only.
Models: GPT-4-turbo and Mixtral-8x7B (actual reward data, no simulation).
Costs: From models.json (real pricing).
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.calibration import embed_prompt
import copy
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)
from sentence_transformers import SentenceTransformer
from routellm.controller import Controller
import joblib

sys.path.insert(0, str(project_root / "experiments"))
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING - REAL DATA ONLY, NO FALLBACKS OR SYNTHETIC DATA
# =============================================================================

def load_model_costs() -> Dict:
    """
    Load model costs from models.json - REAL PRICING ONLY.
    
    Returns:
        Dict mapping model_id to cost per request
    """
    if not DEFAULT_MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"models.json not found at: {DEFAULT_MODEL_REGISTRY_PATH}")
    
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    
    # Build cost lookup (100 input tokens, 400 output tokens)
    # NO FALLBACKS - strict validation
    costs = {}
    for model in data["models"]:
        model_id = model["model_id"]
        
        # Require pricing data (no defaults)
        if "price_1m_input" not in model:
            raise ValueError(f"Missing price_1m_input for model {model_id}")
        if "price_1m_output" not in model:
            raise ValueError(f"Missing price_1m_output for model {model_id}")
        if "display_name" not in model:
            raise ValueError(f"Missing display_name for model {model_id}")
        
        input_cost = model["price_1m_input"]
        output_cost = model["price_1m_output"]
        cost_per_request = (100 * input_cost + 400 * output_cost) / 1_000_000
        
        costs[model_id] = {
            "name": model["display_name"],
            "cost": cost_per_request
        }
    
    logger.info(f"✓ Loaded costs for {len(costs)} models from models.json")
    return costs


def load_dataset_with_split() -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Load dataset with proper train/test split.
    NO SYNTHETIC DATA OR FALLBACKS.
    
    Returns:
        (train_data, eval_data, stats)
        - train_data: Dev set for training (N~1,121)
        - eval_data: Holdout set for evaluation (N~750)
        - stats: Statistics about the dataset
    """
    # Strict validation - no fallbacks
    if not CANONICAL_DEV_DATA_PATH.exists():
        raise FileNotFoundError(f"Dev data not found: {CANONICAL_DEV_DATA_PATH}")
    if not CANONICAL_HOLDOUT_DATA_PATH.exists():
        raise FileNotFoundError(f"Holdout data not found: {CANONICAL_HOLDOUT_DATA_PATH}")
    
    logger.info("\n📥 Loading dataset with TRAIN/TEST split...")
    logger.info("   Train: Dev set (for banditGPT online learning)")
    logger.info("   Test: Holdout set (for all methods evaluation)")
    logger.info("   Using REAL data only - no synthetic fallbacks")
    
    def load_split(filepath, label):
        """Load one split (dev or holdout)."""
        prompt_rewards = defaultdict(lambda: {})
        count = 0
        
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    # NO FALLBACKS - strict validation
                    if "prompt" not in entry:
                        raise ValueError(f"Missing 'prompt' field in {filepath}")
                    if "model_id" not in entry:
                        raise ValueError(f"Missing 'model_id' field in {filepath}")
                    if "raw_score" not in entry:
                        raise ValueError(f"Missing 'raw_score' field for prompt in {filepath}")
                    
                    prompt = entry["prompt"]
                    model_id = entry["model_id"]
                    score = entry["raw_score"]
                    prompt_rewards[prompt][model_id] = score
                    count += 1
        
        # Convert to list, only keep prompts with BOTH models
        prompts_data = []
        for prompt, rewards in prompt_rewards.items():
            if len(rewards) == 2:
                prompts_data.append({
                    "prompt": prompt,
                    "rewards": rewards
                })
        
        logger.info(f"  {label}: {len(prompts_data)} prompts (from {count} entries)")
        return prompts_data
    
    # Load both splits
    train_data = load_split(CANONICAL_DEV_DATA_PATH, "Train (Dev)")
    eval_data = load_split(CANONICAL_HOLDOUT_DATA_PATH, "Test (Holdout)")
    
    stats = {
        "train_prompts": len(train_data),
        "eval_prompts": len(eval_data),
        "total_prompts": len(train_data) + len(eval_data),
        "models": list(train_data[0]["rewards"].keys()) if train_data else []
    }
    
    logger.info(f"  Models: {stats['models']}")
    logger.info(f"  Total: {stats['total_prompts']} prompts")
    
    if stats['train_prompts'] < 100 or stats['eval_prompts'] < 100:
        raise ValueError(f"Insufficient data: train={stats['train_prompts']}, eval={stats['eval_prompts']}")
    
    return train_data, eval_data, stats


# =============================================================================
# ROUTING STRATEGIES - REAL IMPLEMENTATIONS ONLY
# =============================================================================

def oracle_routing(prompts_data: List[Dict], model_costs: Dict) -> Tuple[float, float]:
    """Oracle: Always select the best model for each prompt."""
    total_reward = 0.0
    total_cost = 0.0
    
    for prompt_data in prompts_data:
        rewards = prompt_data["rewards"]
        # Select best model
        best_model = max(rewards.items(), key=lambda x: x[1])
        model_id, reward = best_model
        
        total_reward += reward
        total_cost += model_costs[model_id]["cost"]
    
    n = len(prompts_data)
    return total_reward / n, total_cost / n


def static_routing(prompts_data: List[Dict], model_id: str, 
                   model_costs: Dict) -> Tuple[float, float]:
    """Static: Always use the same model."""
    total_reward = 0.0
    total_cost = 0.0
    count = 0
    
    for prompt_data in prompts_data:
        rewards = prompt_data["rewards"]
        if model_id in rewards:
            total_reward += rewards[model_id]
            total_cost += model_costs[model_id]["cost"]
            count += 1
    
    if count == 0:
        return 0.0, 0.0
    
    return total_reward / count, total_cost / count


def route_single_prompt(prompt_data: Dict, controller: Controller, router_name: str, 
                       threshold: float, model_costs: Dict, max_retries: int = 5) -> Tuple[float, float]:
    """
    Route a single prompt and return (reward, cost).
    Includes exponential backoff for rate limit handling.
    """
    prompt = prompt_data["prompt"]
    rewards = prompt_data["rewards"]
    
    # Use REAL RouteLLM to select model (with retry logic)
    for attempt in range(max_retries):
        try:
            selected_model = controller.route(prompt, router=router_name, threshold=threshold)
            
            # Get reward and cost
            if selected_model in rewards:
                reward = rewards[selected_model]
                cost = model_costs[selected_model]["cost"]
                return reward, cost
            
            return 0.0, 0.0
        
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = (2 ** attempt) * 0.2  # 0.2s, 0.4s, 0.8s, 1.6s, 3.2s
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
            raise e
    
    return 0.0, 0.0


def routellm_routing_sequential(prompts_data: List[Dict], controller: Controller, 
                                router_name: str, threshold: float,
                                weak_model: str, strong_model: str, 
                                model_costs: Dict) -> Tuple[float, float]:
    """
    REAL RouteLLM routing using sequential processing (rate limit safe).
    
    Args:
        prompts_data: Evaluation prompts
        controller: Pre-initialized RouteLLM controller
        router_name: RouteLLM router type ('mf', 'sw_ranking', 'bert', etc.)
        threshold: Routing threshold (0-1)
        weak_model: Cheap model ID
        strong_model: Expensive model ID
        model_costs: Cost lookup
    
    Returns:
        (avg_reward, avg_cost) on evaluation data
    """
    total_reward = 0.0
    total_cost = 0.0
    count = 0
    
    # Sequential processing with retry logic (rate limit safe)
    for i, prompt_data in enumerate(prompts_data):
        try:
            reward, cost = route_single_prompt(
                prompt_data, controller, router_name, threshold, model_costs
            )
            if reward > 0 or cost > 0:
                total_reward += reward
                total_cost += cost
                count += 1
        except Exception as e:
            logger.warning(f"      Prompt {i+1}/{len(prompts_data)} failed: {str(e)[:100]}")
            continue
    
    if count == 0:
        return 0.0, 0.0
    
    return total_reward / count, total_cost / count


# =============================================================================
# PARETO GENERATION LOGIC
# =============================================================================

def normalize_prior_strength(priors: Dict, target_sample_size: float = 10.0) -> Dict:
    """
    Normalize prior strength to a specific effective sample size.
    
    This is more precise than gamma scaling because it directly sets
    the "confidence mass" (Trace of A) to represent a specific number
    of samples, rather than using an arbitrary scaling factor.
    
    Args:
        priors: Original priors with potentially massive A matrices
        target_sample_size: Desired effective sample count (default: 10)
    
    Returns:
        New priors with normalized confidence
    """
    dim = priors['context_dim']
    new_priors = copy.deepcopy(priors)
    
    logger.info(f"\n🔧 Normalizing Prior Strength to {target_sample_size} effective samples:")
    
    for m in priors['A']:
        A = priors['A'][m]
        b = priors['b'][m]
        
        # 1. Calculate current "mass" (approximate sample count)
        current_mass = np.trace(A) / dim
        
        # 2. Calculate scaling factor to reach target
        # If mass is 80,000 and we want 10, scale = 10 / 80,000
        if current_mass > 1e-6:
            scale = target_sample_size / current_mass
        else:
            scale = 1.0
        
        # 3. Scale BOTH A and b to preserve θ while changing confidence
        new_priors['A'][m] = A * scale
        new_priors['b'][m] = b * scale
        
        # Verify
        new_mass = np.trace(new_priors['A'][m]) / dim
        logger.info(f"  {m}: {current_mass:.0f} → {new_mass:.0f} samples (scale={scale:.2e})")
    
    return new_priors


def banditgpt_hybrid_routing(train_data: List[Dict], eval_data: List[Dict],
                             encoder: SentenceTransformer, pca, warmup_priors: Dict,
                             model_costs: Dict, lambda_penalty: float,
                             cold_start: bool = False) -> Tuple[float, float]:
    """
    banditGPT Hybrid using the **production BanditRouter**.

    Exercises the full ``BanditRouter.create()`` → ``route()`` →
    ``process_feedback()`` code path.

    PHASE 1 (BURN-IN): Train on dev set WITH cost penalty λ (skipped if cold_start=True)
    PHASE 2 (EVALUATION): Test on holdout set, NO UPDATES

    Returns:
        (avg_reward, avg_cost) on eval_data
    """
    # Resolve warmup path
    sanitized_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_path = str(sanitized_path if sanitized_path.exists() else DEFAULT_WARMUP_PRIORS_PATH)

    # Embed all prompts with provided encoder+PCA
    train_emb = [embed_prompt(p["prompt"], encoder, pca) for p in train_data]
    eval_emb = [embed_prompt(p["prompt"], encoder, pca) for p in eval_data]
    dim = len(train_emb[0])

    router = create_experiment_router(
        model_registry=None,
        feature_dim=dim,
        prior_n_effective=10.0,
        alpha=2.0,
        warmup_path=warmup_path,
        cost_penalty=lambda_penalty,
    )

    # Zero-leakage normalization bounds (train data only)
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in_steps = len(train_data)

    # Phase 1: Burn-in (skipped for cold-start)
    if not cold_start:
        for i, p in enumerate(train_data):
            model, log = router.route(train_emb[i], total_steps=burn_in_steps)
            norm_r = (p["rewards"][model] - r_min) / r_range
            router.process_feedback(log.request_id, norm_r)

    # Phase 2: Evaluation (no updates)
    total_reward, total_cost = 0.0, 0.0
    for i, p in enumerate(eval_data):
        model, _log = router.route(eval_emb[i], total_steps=burn_in_steps)
        total_reward += p["rewards"][model]
        total_cost += model_costs[model]["cost"]

    return total_reward / len(eval_data), total_cost / len(eval_data)


# =============================================================================
# PARETO FRONTIER GENERATION
# =============================================================================

def generate_pareto_frontier(train_data: List[Dict], eval_data: List[Dict], 
                            model_costs: Dict, encoder: SentenceTransformer, 
                            pca, warmup_priors: Dict) -> Tuple[Dict[str, List[Tuple[float, float]]], Dict]:
    """
    Generate Pareto frontier using REAL data only.
    
    IMPORTANT: All methods evaluated on eval_data (holdout).
               banditGPT trains on train_data first, then evaluates on eval_data.
    
    Returns:
        (results, stats): Results dict with (cost, reward) tuples and stats dict with standard deviations
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING PARETO FRONTIER (REAL DATA ONLY)")
    logger.info("="*70)
    logger.info(f"  Train set: {len(train_data)} prompts (for banditGPT learning)")
    logger.info(f"  Eval set: {len(eval_data)} prompts (for all methods)")
    
    results = {}
    models = list(eval_data[0]["rewards"].keys())
    cheap_model = "mistralai/mixtral-8x7b-instruct"
    expensive_model = "openai/gpt-4-turbo"
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Oracle (evaluated on eval_data)
    logger.info("\n1. Oracle (Perfect Routing)...")
    oracle_reward, oracle_cost = oracle_routing(eval_data, model_costs)
    results["Oracle"] = [(oracle_cost, oracle_reward)]
    logger.info(f"   Reward: {oracle_reward:.4f}, Cost: ${oracle_cost:.6f}")
    
    # Save intermediate
    save_results(results, output_dir, len(eval_data), prefix="intermediate_")
    
    # 2. Static baselines (evaluated on eval_data)
    logger.info("\n2. Static Routing Baselines...")
    for model_id in models:
        reward, cost = static_routing(eval_data, model_id, model_costs)
        name = model_costs[model_id]["name"]
        results[f"Static-{name}"] = [(cost, reward)]
        logger.info(f"   {name}: Reward={reward:.4f}, Cost=${cost:.6f}")
    
    # Save intermediate
    save_results(results, output_dir, len(eval_data), prefix="intermediate_")
    
    # 3. RouteLLM Matrix Factorization Router (REAL, evaluated on eval_data)
    # SMART AUGMENTATION: Check for existing results and only run missing thresholds
    logger.info("\n3. RouteLLM Matrix Factorization Router (REAL)...")
    
    existing_results_path = output_dir / "pareto_results.json"
    routellm_points = []
    
    # Try to load existing RouteLLM results
    if existing_results_path.exists():
        logger.info("   📥 Found existing results - checking for gaps...")
        try:
            with open(existing_results_path) as f:
                existing_data = json.load(f)
            
            if "RouteLLM-MF" in existing_data.get("strategies", {}):
                existing_points = existing_data["strategies"]["RouteLLM-MF"]
                routellm_points = [(p["cost"], p["reward"]) for p in existing_points]
                logger.info(f"   ✓ Loaded {len(routellm_points)} existing RouteLLM points")
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to load existing results: {e}")
    
    # Dense coverage in the low-threshold / high-cost region (0.0–0.15), plus
    # coarser coverage across the rest of the range.
    target_thresholds = sorted(set(
        list(np.arange(0.0, 0.16, 0.01))          # 16 pts in [0.00, 0.15]
        + list(np.arange(0.20, 1.01, 1/12))        # ~10 pts in [0.20, 1.0]
    ))

    # Track which thresholds are already covered (within tolerance)
    existing_thresholds: set[float] = set()
    if existing_results_path.exists():
        try:
            with open(existing_results_path) as f:
                meta = json.load(f).get("metadata", {})
            for t in meta.get("routellm_thresholds_evaluated", []):
                existing_thresholds.add(round(t, 4))
        except Exception:
            pass

    thresholds_to_run = [
        t for t in target_thresholds
        if round(t, 4) not in existing_thresholds
    ]

    if thresholds_to_run:
        logger.info(f"   🎯 Need {len(thresholds_to_run)} more points for smooth curve")
        logger.info("   Initializing RouteLLM controller...")
        
        try:
            routellm_controller = Controller(
                routers=['mf'],
                strong_model=expensive_model,
                weak_model=cheap_model
            )
            logger.info("   ✓ RouteLLM initialized")
            
            logger.info(f"   Sweeping {len(thresholds_to_run)} thresholds (sequential, rate-limit safe)...")
            logger.info("   ⏱️  Estimated time: ~{:.1f} minutes".format(len(thresholds_to_run) * len(eval_data) * 0.002 / 60))
            
            new_points = []
            for i, threshold in enumerate(thresholds_to_run, 1):
                try:
                    reward, cost = routellm_routing_sequential(
                        eval_data, routellm_controller, 'mf', threshold, 
                        cheap_model, expensive_model, model_costs
                    )
                    if reward > 0:
                        new_points.append((cost, reward))
                    logger.info(f"   [{i}/{len(thresholds_to_run)}] Threshold {threshold:.3f}: R={reward:.4f}, C=${cost:.6f}")
                except Exception as e:
                    logger.warning(f"   [{i}/{len(thresholds_to_run)}] Threshold {threshold:.3f} failed: {e}")
                    continue
                
                if i < len(thresholds_to_run):
                    time.sleep(0.3)
            
            routellm_points.extend(new_points)
            logger.info(f"   ✓ Added {len(new_points)} new points (total: {len(routellm_points)})")
            
        except Exception as e:
            logger.error(f"   ✗ RouteLLM processing failed: {e}")
            logger.warning("   Using existing points only")
    else:
        logger.info(f"   ✓ Already have sufficient coverage ({len(routellm_points)} points)")
    
    results["RouteLLM-MF"] = routellm_points
    all_evaluated_thresholds = sorted(existing_thresholds | {round(t, 4) for t in thresholds_to_run})
    routellm_meta = {"routellm_thresholds_evaluated": all_evaluated_thresholds}
    
    # Save intermediate
    save_results(results, output_dir, len(eval_data), prefix="intermediate_",
                 metadata_extra=routellm_meta)
    
    # 4. banditGPT Hybrid with Cost-Aware Policy
    # Sweep cost penalty λ to create Pareto frontier
    # λ = 0: Pure quality optimization (expensive)
    # λ = high: Cost-conscious (cheap)
    logger.info("\n4. banditGPT Hybrid (Cost-Aware Corralling)...")
    logger.info("   Sweeping cost penalty λ to create Pareto frontier...")
    
    # Normalize costs to [0, 1] scale (same as rewards)
    # This ensures λ values are interpretable
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    
    logger.info(f"   Cost normalization: min=${min_cost:.6f}, max=${max_cost:.6f}, range=${cost_range:.6f}")
    
    # Normalize model costs
    normalized_costs = {}
    for model_id in models:
        raw_cost = model_costs[model_id]["cost"]
        normalized = (raw_cost - min_cost) / cost_range if cost_range > 0 else 0.0
        normalized_costs[model_id] = {
            "cost": raw_cost,  # Keep original for accounting
            "normalized_cost": normalized  # Use for UCB
        }
        logger.info(f"   {model_costs[model_id]['name']}: ${raw_cost:.6f} → {normalized:.4f}")
    
    hybrid_points = []
    hybrid_stats = []  # Track standard deviations
    
    # Sweep cost penalties (normalized)
    # Note: λ should be commensurate with the reward range [0, 1]
    # λ=0.0: Pure quality (ignore cost)
    # λ=0.1-0.5: Balanced trade-off
    # λ=1.0+: Cost-conscious (heavily penalize expensive models)
    cost_penalties = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    for i, lambda_val in enumerate(cost_penalties, 1):
        # Run 5 trials and average to smooth out noise/outliers
        trial_rewards = []
        trial_costs = []
        
        for trial in range(20):
            np.random.seed(42 + trial)
            r, c = banditgpt_hybrid_routing(
                train_data, eval_data, encoder, pca, warmup_priors, normalized_costs, 
                lambda_penalty=lambda_val,
            )
            trial_rewards.append(r)
            trial_costs.append(c)
        
        avg_reward = np.mean(trial_rewards)
        avg_cost = np.mean(trial_costs)
        std_reward = np.std(trial_rewards, ddof=1) if len(trial_rewards) > 1 else 0.0
        std_cost = np.std(trial_costs, ddof=1) if len(trial_costs) > 1 else 0.0
        
        if avg_reward > 0:
            hybrid_points.append((avg_cost, avg_reward))
            hybrid_stats.append({
                "cost_std": std_cost,
                "reward_std": std_reward,
                "n_trials": len(trial_rewards)
            })
        
        logger.info(f"   [{i}/{len(cost_penalties)}] λ={lambda_val:.1f}: "
                   f"Reward={avg_reward:.4f}±{std_reward:.4f}, "
                   f"Cost=${avg_cost:.6f}±${std_cost:.6f} (20 trials)")
    
    results["banditGPT-Hybrid"] = hybrid_points
    logger.info(f"   ✓ Generated {len(hybrid_points)} points")
    
    # Prepare stats dictionary
    stats = {
        "banditGPT-Hybrid": hybrid_stats
    }
    
    # Save final results with statistics
    save_results(results, output_dir, len(eval_data), prefix="intermediate_",
                include_stats=True, stats_data=stats,
                metadata_extra=routellm_meta)
    
    return results, stats, routellm_meta



def save_results(results: Dict, output_dir: Path, n_prompts: int, prefix: str = "", 
                 include_stats: bool = False, stats_data: Dict = None,
                 metadata_extra: Dict | None = None):
    """
    Save results to JSON with optional statistical information.
    
    Args:
        results: Dict of strategy -> list of (cost, reward) tuples
        output_dir: Output directory path
        n_prompts: Number of prompts in evaluation
        prefix: Optional filename prefix
        include_stats: Whether to include standard deviation data
        stats_data: Optional dict with strategy -> list of statistics per point
        metadata_extra: Optional extra metadata fields to persist
    """
    filename = f'{prefix}pareto_results.json' if prefix else 'pareto_results.json'
    output_file = output_dir / filename
    
    strategies_data = {}
    for strategy, points in results.items():
        if include_stats and stats_data and strategy in stats_data:
            strategies_data[strategy] = [
                {
                    "cost": float(c), 
                    "reward": float(r),
                    "cost_std": float(stats_data[strategy][i].get("cost_std", 0.0)),
                    "reward_std": float(stats_data[strategy][i].get("reward_std", 0.0))
                }
                for i, (c, r) in enumerate(points)
            ]
        else:
            strategies_data[strategy] = [
                {"cost": float(c), "reward": float(r)} 
                for c, r in points
            ]
    
    metadata = {
        "n_prompts": n_prompts,
        "description": "Pareto frontier - REAL DATA ONLY",
        "includes_statistics": include_stats
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    
    results_serializable = {
        "metadata": metadata,
        "strategies": strategies_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"✅ Saved results: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("="*70)
    logger.info("FIGURE 4: PARETO FRONTIER ANALYSIS")
    logger.info("="*70)
    logger.info("\n⚠️  REAL DATA ONLY - NO SYNTHETIC FALLBACKS")
    logger.info("📊 Proper Train/Test Split:")
    logger.info("   - banditGPT trains on DEV set")
    logger.info("   - All methods evaluate on HOLDOUT set")
    
    # Load costs
    logger.info("\n📦 Loading model costs...")
    model_costs = load_model_costs()
    
    # Load data with train/test split
    train_data, eval_data, stats = load_dataset_with_split()
    
    # Load encoder and PCA
    logger.info("\n📦 Loading encoder and PCA...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    # Use SANITIZED priors (b vectors scaled to [0,1] range)
    sanitized_priors_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    if sanitized_priors_path.exists():
        warmup_priors = joblib.load(sanitized_priors_path)
        logger.info(f"  ✓ Using SANITIZED priors: {sanitized_priors_path}")
    else:
        logger.warning(f"  ⚠️  Sanitized priors not found, using original (may cause scale issues)")
        warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    logger.info(f"  ✓ Encoder: {DEFAULT_SENTENCE_TRANSFORMER}")
    logger.info(f"  ✓ PCA: {DEFAULT_PCA_PATH}")
    
    # Generate Pareto frontier
    results, result_stats, routellm_meta = generate_pareto_frontier(train_data, eval_data, model_costs, encoder, pca, warmup_priors)
    
    output_dir = Path(__file__).parent / "results"
    
    # Save final results with statistics (figure is generated by generate_figure4.py)
    save_results(results, output_dir, stats["eval_prompts"], 
                include_stats=True, stats_data=result_stats,
                metadata_extra=routellm_meta)
    
    logger.info("\n" + "="*70)
    logger.info("✅ PARETO FRONTIER ANALYSIS COMPLETE!")
    logger.info("="*70)
    
    # Summary
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"  Train set: {stats['train_prompts']:,} prompts (dev)")
    logger.info(f"  Eval set: {stats['eval_prompts']:,} prompts (holdout)")
    logger.info(f"\n📈 Results on HOLDOUT set:")
    for strategy, points in results.items():
        if points:
            costs = [p[0] for p in points]
            rewards = [p[1] for p in points]
            logger.info(f"\n{strategy}:")
            logger.info(f"  Cost: ${min(costs):.6f} - ${max(costs):.6f}")
            logger.info(f"  Reward: {min(rewards):.4f} - {max(rewards):.4f}")


if __name__ == "__main__":
    main()
