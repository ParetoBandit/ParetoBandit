#!/usr/bin/env python3
"""
Experiment 09: Custom Weights Pareto Demonstration

Demonstrates how custom quality/cost/latency weights affect model selection
and creates a Pareto curve showing the cost-quality tradeoff.

This experiment shows:
1. "cost_saver" profile (minimize cost) → selects cheap models
2. "high_quality" profile (maximize quality) → selects premium models
3. Pareto curve of all models showing cost vs quality tradeoff
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(n_samples: int = 100, model_filter: set = None) -> List[Dict]:
    """
    Load test prompts with ground truth rewards from holdout dataset.
    
    Args:
        n_samples: Number of test samples to load
        model_filter: Set of model IDs to filter for (only include these models)
        
    Returns:
        List of dicts with keys: prompt, rewards (dict of model_id -> reward)
    """
    import gzip
    
    data_dir = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    test_path = data_dir / "holdout_rewards_complete.jsonl.gz"
    
    if not test_path.exists():
        raise FileNotFoundError(f"Holdout test data not found at {test_path}")
    
    logger.info(f"📦 Loading holdout test data from {test_path}")
    
    # Group by prompt to get all model rewards for each prompt
    prompt_data = defaultdict(lambda: {"rewards": {}})
    
    with gzip.open(test_path, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                reward = entry.get("raw_score", 0.0)
                
                # Filter by model if specified
                if model_filter and model_id not in model_filter:
                    continue
                
                prompt_data[prompt]["rewards"][model_id] = reward
    
    # Convert to list format, only keeping prompts with complete coverage
    test_samples = []
    required_models = model_filter if model_filter else set()
    
    for prompt, data in prompt_data.items():
        # Only include prompts that have rewards for all required models
        if required_models and not required_models.issubset(data["rewards"].keys()):
            continue
        
        test_samples.append({
            "prompt": prompt,
            "rewards": data["rewards"]
        })
        
        if len(test_samples) >= n_samples:
            break
    
    logger.info(f"  ✓ Loaded {len(test_samples)} test samples with complete coverage")
    logger.info(f"  Models per sample: {len(test_samples[0]['rewards']) if test_samples else 0}")
    
    return test_samples


def generate_synthetic_data(n_samples: int = 100, registry: Dict = None) -> List[Dict]:
    """
    Generate synthetic test data using model quality from registry.
    
    Uses initial_quality from registry to simulate realistic rewards.
    
    Args:
        n_samples: Number of samples to generate
        registry: Model registry with initial_quality scores
    """
    logger.info("🔧 Generating synthetic test data from registry quality scores")
    
    prompts = [
        "Write a Python function to sort a list",
        "Explain quantum mechanics in simple terms",
        "Debug this code that throws a TypeError",
        "Solve the equation x^2 + 3x + 2 = 0",
        "Write a creative story about time travel",
        "What is the capital of France?",
        "How do neural networks work?",
        "Calculate the derivative of f(x) = x^3",
        "Explain the difference between TCP and UDP",
        "Write a poem about artificial intelligence"
    ]
    
    if not registry:
        raise ValueError("Registry required for synthetic data generation")
    
    model_ids = list(registry.keys())
    logger.info(f"  Generating rewards for {len(model_ids)} models")
    
    # Simulate rewards for different models based on their initial_quality
    test_samples = []
    random.seed(42)
    
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)] + f" (variation {i})"
        
        # Simulate rewards using initial_quality from registry
        rewards = {}
        for model_id in model_ids:
            model_info = registry[model_id]
            
            # Use initial_quality as base (this comes from dev_rewards_complete)
            base_quality = model_info.get("initial_quality", 0.85)
            
            # Add small random variance to simulate real-world variation
            # Quality varies more on harder prompts
            prompt_difficulty = (i % 10) / 10.0  # Simple difficulty proxy
            variance = random.uniform(-0.05, 0.05) * (1 + prompt_difficulty)
            
            reward = base_quality + variance
            reward = np.clip(reward, 0.60, 1.0)
            
            rewards[model_id] = reward
        
        test_samples.append({
            "prompt": prompt,
            "rewards": rewards
        })
    
    logger.info(f"  ✓ Generated {len(test_samples)} synthetic samples with {len(model_ids)} models each")
    logger.info(f"  Quality range: {min(model_info.get('initial_quality', 0.85) for model_info in registry.values()):.3f} - {max(model_info.get('initial_quality', 0.85) for model_info in registry.values()):.3f}")
    
    return test_samples


def load_model_registry() -> Dict:
    """Load model registry with cost/latency data (Pareto-optimal models)."""
    # Use the Pareto-optimal models for this experiment
    models_path = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "config" / "models_pareto.json"
    
    if not models_path.exists():
        logger.error(f"Models registry not found at {models_path}")
        raise FileNotFoundError(f"Models registry not found: {models_path}")
    
    logger.info(f"📦 Loading Pareto-optimal model registry from {models_path}")
    
    with open(models_path) as f:
        data = json.load(f)
    
    registry = {m["openrouter_id"]: m for m in data["models"]}
    logger.info(f"  ✓ Registry: {len(registry)} Pareto-optimal models")
    
    # Print model names for verification
    for model_id in registry.keys():
        logger.info(f"    - {model_id}")
    
    return registry


def get_model_cost(model: Dict) -> float:
    """Calculate blended cost per 1k tokens in USD."""
    # Support both naming conventions
    input_cost = model.get("price_1m_input") or model.get("input_cost_per_m")
    output_cost = model.get("price_1m_output") or model.get("output_cost_per_m")
    
    if input_cost is None or output_cost is None:
        return 0.0
    
    # Blended cost per 1k tokens (50/50 split)
    # Convert from per-1M to per-1k
    cost_per_1k = (0.5 * input_cost + 0.5 * output_cost) / 1000.0
    return cost_per_1k


# =============================================================================
# CUSTOM WEIGHT PROFILES
# =============================================================================

def get_custom_profiles() -> Dict[str, Dict]:
    """
    Define custom weight profiles for the experiment.
    
    Returns:
        Dict mapping profile name to weight dict
    """
    profiles = {
        "cost_saver": {
            "name": "Cost Saver",
            "description": "Minimize cost (best low-cost option)",
            "weights": {
                "w_q": 1.0,   # Still care about quality (avoid terrible models)
                "w_c": 10.0,  # Heavily penalize expensive models
                "w_l": 0.0    # Don't care about latency
            }
        },
        "high_quality": {
            "name": "High Quality",
            "description": "Maximize quality (independent of cost)",
            "weights": {
                "w_q": 10.0,  # Maximize quality
                "w_c": 0.0,   # Don't penalize cost
                "w_l": 0.0    # Don't care about latency
            }
        },
        "balanced": {
            "name": "Balanced",
            "description": "Balance quality and cost",
            "weights": {
                "w_q": 5.0,   # Moderate quality preference
                "w_c": 5.0,   # Moderate cost penalty
                "w_l": 0.0    # Don't care about latency
            }
        }
    }
    
    return profiles


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_profile_experiment(
    router: BanditRouter,
    test_samples: List[Dict],
    profile_weights: Dict,
    profile_name: str,
    registry: Dict
) -> Dict:
    """
    Run routing with a specific weight profile and measure outcomes.
    
    Args:
        router: Initialized BanditRouter
        test_samples: List of test prompts with ground truth rewards
        profile_weights: Weight dict {"w_q": ..., "w_c": ..., "w_l": ...}
        profile_name: Name of profile for logging
        registry: Model registry for cost lookup
        
    Returns:
        Results dict with metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running profile: {profile_name}")
    logger.info(f"Weights: {profile_weights}")
    logger.info(f"{'='*70}")
    
    # Disable exploration for deterministic evaluation
    original_alpha = router.bandit.alpha
    router.bandit.alpha = 0.0
    
    costs = []
    qualities = []
    selections = Counter()
    routing_details = []
    
    for i, sample in enumerate(test_samples):
        prompt = sample["prompt"]
        ground_truth = sample["rewards"]
        
        # Route with custom profile
        selected_model, log = router.route(
            prompt,
            profile=profile_weights,
            input_tokens=100,
            output_tokens=600
        )
        
        # Get ground truth reward for selected model
        if selected_model in ground_truth:
            reward = ground_truth[selected_model]
            qualities.append(reward)
            
            # Get model cost
            model_info = registry.get(selected_model, {})
            cost = get_model_cost(model_info)
            costs.append(cost)
            
            selections[selected_model] += 1
            
            routing_details.append({
                "prompt": prompt[:50] + "...",
                "selected_model": selected_model,
                "reward": reward,
                "cost": cost
            })
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i+1}/{len(test_samples)} samples...")
    
    # Restore exploration
    router.bandit.alpha = original_alpha
    
    # Calculate metrics
    avg_cost = float(np.mean(costs)) if costs else 0.0
    avg_quality = float(np.mean(qualities)) if qualities else 0.0
    
    # Calculate success rate (% of prompts with reward > 0.8)
    success_rate = sum(1 for q in qualities if q > 0.8) / len(qualities) if qualities else 0.0
    
    results = {
        "profile_name": profile_name,
        "weights": profile_weights,
        "avg_cost": avg_cost,
        "avg_quality": avg_quality,
        "success_rate": success_rate,
        "n_samples": len(qualities),
        "model_selections": dict(selections),
        "routing_details": routing_details[:10]  # Save first 10 for inspection
    }
    
    logger.info(f"\n📊 Results for {profile_name}:")
    logger.info(f"  Average Cost:    ${avg_cost:.5f} per 1k tokens")
    logger.info(f"  Average Quality: {avg_quality*100:.2f}%")
    logger.info(f"  Success Rate:    {success_rate*100:.2f}%")
    logger.info(f"  Model Distribution:")
    for model, count in selections.most_common():
        pct = 100 * count / len(test_samples)
        logger.info(f"    {model:30s} {count:3d} ({pct:5.1f}%)")
    
    return results


def compute_model_baselines(test_samples: List[Dict], registry: Dict) -> List[Dict]:
    """
    Compute cost and quality (FCI) for each individual model.
    These form the Pareto frontier baseline.
    
    Uses theoretical FCI scores from model registry (initial_quality),
    not empirical test performance.
    """
    logger.info("\n📈 Computing individual model baselines using FCI scores...")
    
    model_points = []
    
    for model_id, model_info in registry.items():
        cost = get_model_cost(model_info)
        if cost == 0.0:
            continue
        
        # Use FCI score from registry (initial_quality)
        fci_quality = model_info.get("initial_quality")
        
        if fci_quality is None:
            logger.warning(f"  ⚠️  {model_id} missing initial_quality (FCI), skipping")
            continue
        
        # Also calculate empirical performance for comparison
        empirical_qualities = []
        for sample in test_samples:
            if model_id in sample["rewards"]:
                empirical_qualities.append(sample["rewards"][model_id])
        
        if empirical_qualities:
            empirical_avg = float(np.mean(empirical_qualities))
            success_rate = sum(1 for q in empirical_qualities if q > 0.8) / len(empirical_qualities)
        else:
            empirical_avg = fci_quality  # Fallback to FCI if no test data
            success_rate = 0.0
        
        model_points.append({
            "model_id": model_id,
            "model_name": model_info.get("display_name", model_id),
            "cost": cost,
            "quality": fci_quality,  # Use FCI for Pareto analysis
            "empirical_quality": empirical_avg,  # Track empirical for comparison
            "success_rate": success_rate,
            "n_samples": len(empirical_qualities)
        })
        
        logger.info(f"  {model_id:30s} Cost=${cost:.5f}, FCI={fci_quality*100:.2f}%, "
                   f"Empirical={empirical_avg*100:.2f}% (Δ={abs(fci_quality-empirical_avg)*100:.1f}%)")
    
    logger.info(f"  ✓ Computed {len(model_points)} model baselines using FCI scores")
    return model_points


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run custom weights Pareto experiment."""
    logger.info("="*70)
    logger.info("EXPERIMENT 09: CUSTOM WEIGHTS PARETO DEMONSTRATION")
    logger.info("="*70)
    
    # Load Pareto-optimal model registry
    registry = load_model_registry()
    
    # Load real holdout test data with complete coverage for Pareto models
    logger.info("\n📊 Loading holdout test data...")
    pareto_models = set(registry.keys())
    test_samples = load_test_data(n_samples=100, model_filter=pareto_models)
    
    # Initialize router with Pareto-specific warmup priors
    logger.info("\n🔧 Initializing BanditRouter with Pareto warmup priors...")
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    warmup_path = PROJECT_ROOT / "artifacts" / "priors_warmup_pareto.joblib"
    
    router = BanditRouter.create(
        model_registry=registry,
        priors=str(warmup_path),  # Use Pareto-specific warmup priors
        alpha=0.0  # Pure exploitation for evaluation (greedy)
    )
    logger.info(f"  ✓ Router initialized with {len(router.bandit.models)} models")
    logger.info(f"  ✓ Using warmup priors from: {warmup_path}")
    
    # Verify all models have warmup priors loaded
    logger.info("\n🔍 Verifying warmup priors for all models...")
    for model_id in router.bandit.models:
        theta = router.bandit.A_inv[model_id] @ router.bandit.b[model_id]
        theta_norm = np.linalg.norm(theta)
        logger.info(f"    {model_id:40s} ||θ|| = {theta_norm:.4f}")
    
    if any(np.linalg.norm(router.bandit.A_inv[m] @ router.bandit.b[m]) < 0.01 
           for m in router.bandit.models):
        logger.warning("⚠️ Some models have very small theta norms - may not have warmup priors")
    
    # Get custom profiles
    profiles = get_custom_profiles()
    
    # Run experiments for each profile
    profile_results = []
    for profile_key, profile_config in profiles.items():
        result = run_profile_experiment(
            router=router,
            test_samples=test_samples,
            profile_weights=profile_config["weights"],
            profile_name=profile_config["name"],
            registry=registry
        )
        result["description"] = profile_config["description"]
        profile_results.append(result)
    
    # Compute model baselines
    model_baselines = compute_model_baselines(test_samples, registry)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "custom_weights_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "09_custom_weights_pareto",
            "description": "Custom quality/cost/latency weight demonstration",
            "n_test_samples": len(test_samples),
            "profile_results": profile_results,
            "model_baselines": model_baselines
        }, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {results_path}")
    
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("SUMMARY TABLE")
    logger.info("="*70)
    logger.info(f"\n{'Profile':<20} {'Cost ($/1k)':<15} {'Quality':<12} {'Success Rate':<15}")
    logger.info("-" * 70)
    for result in profile_results:
        logger.info(f"{result['profile_name']:<20} "
                   f"${result['avg_cost']:<14.5f} "
                   f"{result['avg_quality']*100:<11.2f}% "
                   f"{result['success_rate']*100:<14.2f}%")
    
    logger.info("\n📁 Next step: Run plot_custom_weights.py to visualize the Pareto curve")


if __name__ == "__main__":
    main()

