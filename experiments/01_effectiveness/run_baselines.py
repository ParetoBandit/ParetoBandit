#!/usr/bin/env python3
"""
Experiment 01: Effectiveness Comparison (KDD-Compliant Offline Replay)

Compares BanditGPT against baselines using REAL data:
- Random selection (no learning)
- ε-greedy (ε=0.1, learns running averages)
- Vanilla LinUCB (α=0.1, bias-only context, learns)
- RouteLLM (SOTA static router, no learning) [TODO: Install library]
- BanditGPT (full semantic features, learns)

Train/Test Split Methodology:
- Data Source: LMSYS Arena dataset (first-turn prompts only)
- Split Type: Random I.I.D. split (976 train, 976 test)
- Distribution: Both splits from same timeframe/distribution
- Burn-in: BanditGPT methods learn on training data before test evaluation
- Oracle: Calculated only over available models (prevents inflated regret)

Critical: All methods use ORACLE REWARD LOOKUP, not random generation.
Output: results/effectiveness_results.json
"""

from typing import Dict, List, Tuple, Any
import argparse
import sys
import numpy as np
import json
import joblib
import random
import time
import logging
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_loader import load_oracle_rewards, load_model_registry
from utils.metrics import calculate_cumulative_regret

# Import BanditRouter for full system evaluation
from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from src.bandit_gpt.storage import SqliteContextStore
from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from sentence_transformers import SentenceTransformer


# =============================================================================
# BASELINE IMPLEMENTATIONS (All use Oracle Lookup)
# =============================================================================

def run_random_baseline(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    seed: int = 42
) -> Dict:
    """
    Random model selection baseline (no learning).
    
    The simplest baseline: uniform random selection across all models.
    No update step since random has no state to learn.
    """
    print(f"Running Random baseline (seed={seed})...")
    rng = np.random.RandomState(seed)
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="Random", leave=False):
        # Select random model
        model_id = rng.choice(available_models)
        
        # ORACLE LOOKUP (not random generation!)
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": "random",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_epsilon_greedy(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    ε-greedy baseline with online learning.
    
    Maintains running mean reward for each model.
    With probability ε: explore (random)
    With probability 1-ε: exploit (best empirical mean)
    """
    print(f"Running ε-greedy (ε={epsilon}, seed={seed})...")
    rng = np.random.RandomState(seed)
    
    # Initialize running statistics
    model_counts = {m: 0 for m in available_models}
    model_means = {m: 0.5 for m in available_models}  # Optimistic prior
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="ε-greedy", leave=False):
        # ε-greedy selection
        if rng.random() < epsilon:
            # Explore: random selection
            model_id = rng.choice(available_models)
        else:
            # Exploit: best empirical mean
            model_id = max(available_models, key=lambda m: model_means[m])
        
        # ORACLE LOOKUP
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # UPDATE (incremental mean)
        model_counts[model_id] += 1
        n = model_counts[model_id]
        model_means[model_id] += (reward - model_means[model_id]) / n
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": f"epsilon_greedy_{epsilon}",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_vanilla_linucb(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    alpha: float = 1.0,  # Fix C: Increased from 0.1 to 1.0 for aggressive exploration
    seed: int = 42
) -> Dict:
    """
    Vanilla LinUCB baseline (bias-only context, no semantic features).
    
    Uses context vector x = [1.0] (just bias term).
    This tests whether semantic features provide lift over context-blind LinUCB.
    """
    print(f"Running Vanilla LinUCB (α={alpha}, seed={seed})...")
    
    # Disjoint LinUCB with d=1 (bias only)
    d = 1
    A = {m: np.eye(d) for m in available_models}  # d×d identity matrices
    b = {m: np.zeros(d) for m in available_models}  # d-dimensional vectors
    
    selected_models = []
    selected_rewards = []
    
    # DEBUG: Track model selection distribution
    from collections import Counter
    model_selections = Counter()
    
    for i, prompt in enumerate(tqdm(prompts, desc="LinUCB", leave=False)):
        # Context: bias-only
        x = np.array([1.0])
        
        # Compute UCB for each arm
        ucb_scores = {}
        for m in available_models:
            A_inv = np.linalg.inv(A[m])
            theta = A_inv @ b[m]
            ucb = theta @ x + alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[m] = float(ucb)
        
        # Fix B: Randomized Tie-Breaking
        # Prevent "Index 0" curse where the first model always wins ties
        best_score = max(ucb_scores.values())
        candidates = [m for m, s in ucb_scores.items() if s == best_score]
        
        # Use stable hash for reproducibility across processes
        import zlib
        prompt_hash = zlib.adler32(prompt.encode('utf-8'))
        rng_step = np.random.RandomState(seed + (prompt_hash % 10000))
        model_id = rng_step.choice(candidates)
        model_selections[model_id] += 1
        
        # ORACLE LOOKUP
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # DEBUG: Log first 10 selections
        if i < 10:
            print(f"    [{i}] Selected: {model_id[:30]:30s} | UCB: {ucb_scores[model_id]:.3f} | Reward: {reward:.3f}")
        
        # UPDATE (Sherman-Morrison style, but simple for d=1)
        A[model_id] += np.outer(x, x)
        b[model_id] += reward * x
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    # DEBUG: Print selection distribution
    print(f"\n  Model Selection Distribution (Vanilla LinUCB):")
    for model, count in model_selections.most_common(5):
        pct = 100 * count / len(prompts)
        print(f"    {model[:40]:40s}: {count:4d} ({pct:5.1f}%)")
    
    return {
        "method": "vanilla_linucb",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_routellm_baseline(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    registry: Dict[str, Dict],
    available_models: List[str],
    cached_scores: Dict[str, float] = None,
    seed: int = 42
) -> Dict:
    """
    Optimized RouteLLM Baseline: Deterministic Anchors + Fast Execution.
    """
    print(f"Running RouteLLM (Deterministic Anchors, seed={seed})...")
    
    # 1. SETUP ANCHORS FOR HLE (Fixed, not median-based)
    # For HLE metric, the difficulty is shifted far right, so we need:
    # - Strong: Best model (only one that can solve hardest queries)
    # - Weak: A "historically strong" model that can solve easiest 5-10%
    #   This creates a meaningful routing boundary, not a trivial "always pick best"
    
    # Find the best model by HLE
    sorted_models = sorted(
        available_models, 
        key=lambda m: registry[m].get("hle", 0), 
        reverse=True
    )
    strong_anchor = sorted_models[0]  # Gemini 3 Pro (HLE=0.372)
    
    # Weak anchor: Look for GPT-OSS-120B specifically (mid-tier model for meaningful routing)
    # If not available, use the model closest to HLE=0.185
    weak_anchor = None
    for model_id in available_models:
        if "gpt-oss-120b" in model_id.lower():
            weak_anchor = model_id
            break
    
    # Fallback: Use model with HLE closest to 0.185 (the "mid-tier" for routing)
    if weak_anchor is None:
        target_hle = 0.185
        weak_anchor = min(available_models, key=lambda m: abs(registry[m].get("hle", 0) - target_hle))
    
    if seed == 0:
        print(f"  ⚓ Anchors: Strong={strong_anchor} ({registry[strong_anchor].get('hle', 0):.3f}), "
              f"Weak={weak_anchor} ({registry[weak_anchor].get('hle', 0):.3f})")

    # 2. INITIALIZE CONTROLLER (Do this ONCE, outside the loop)
    # This prevents reloading weights 1,000 times
    try:
        from routellm.controller import Controller
        # We use a lightweight router (mf) to avoid downloading huge models
        controller = Controller(
            routers=["mf"], 
            strong_model="gpt-4-1106-preview", 
            weak_model="mixtral-8x7b-instruct-v0.1"
        )
        router = controller.routers["mf"]
        has_library = True
    except ImportError:
        print("  ⚠️  RouteLLM library not installed. Returning empty.")
        return {"method": "routellm_mf", "selected_models": [], "rewards": []}

    selected_models = []
    selected_rewards = []
    THRESHOLD = 0.5 
    
    # 3. FAST LOOP
    for prompt in tqdm(prompts, desc="RouteLLM", leave=False):
        # A. Get Score (Use Cache or Compute)
        if cached_scores is not None and prompt in cached_scores:
            win_rate = cached_scores[prompt]
        else:
            # This is now fast because 'router' is already loaded
            try:
                win_rate = router.calculate_strong_win_rate(prompt)
            except Exception:
                win_rate = 0.5 # Default to uncertainty on error
                
            if cached_scores is not None:
                cached_scores[prompt] = win_rate
        
        # B. Deterministic Selection
        prompt_rewards = oracle_rewards.get(prompt, {})
        
        # Filter to only models that are in our registry (available_models)
        valid_models = [m for m in prompt_rewards if m in available_models]
        
        if win_rate > THRESHOLD:
            # Router wants Strong
            if strong_anchor in valid_models:
                choice = strong_anchor
            elif valid_models:
                choice = max(valid_models, key=lambda m: registry[m].get("hle", 0))
            else:
                # No valid models, fallback to strong anchor anyway
                choice = strong_anchor
        else:
            # Router wants Weak
            if weak_anchor in valid_models:
                choice = weak_anchor
            elif valid_models:
                choice = min(valid_models, key=lambda m: registry[m].get("hle", 0))
            else:
                # No valid models, fallback to weak anchor anyway
                choice = weak_anchor
        
        reward = prompt_rewards.get(choice, 0.0)
        selected_models.append(choice)
        selected_rewards.append(reward)
        
    return {
        "method": "routellm_mf",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }



def run_banditgpt(
    test_prompts: List[str],
    test_oracle_rewards: Dict[str, Dict[str, float]],
    train_prompts: List[str],
    train_oracle_rewards: Dict[str, Dict[str, float]],
    registry: Dict[str, Dict],
    encoder,
    priors: str = "hle",
    prior_n_effective: float = None,
    seed: int = 42
) -> Dict:
    """
    Full BanditGPT system with semantic features and online learning.
    
    Uses the complete feature set:
    - BERT embeddings (384-d) → PCA (32-d)
    - Handcrafted features (code blocks, LaTeX, length, etc.)
    - Virtual anchor similarities
    - Complexity score
    
    Args:
        test_prompts: Test prompts for evaluation
        test_oracle_rewards: Oracle rewards for test prompts
        train_prompts: Training prompts for burn-in
        train_oracle_rewards: Oracle rewards for training prompts
        priors: Prior initialization strategy ("none", "hle", "warmup")
        prior_n_effective: Effective sample size (None = use default for strategy)
    
    This is the method we're trying to prove works!
    """
    print(f"Running BanditGPT (priors={priors}, N_eff={prior_n_effective}, seed={seed})...")
    
    # Fresh router for this trial (clean slate)
    create_kwargs = {
        "exploration": "balanced",  # Fix: align with LinUCB alpha=1.0
        "priors": priors,
        "context_encoder": encoder,
    }
    
    # Add prior_n_effective only if specified (otherwise use router defaults)
    if prior_n_effective is not None:
        create_kwargs["prior_n_effective"] = prior_n_effective
    
    router = BanditRouter.create(registry, **create_kwargs)
    
    # BURN-IN PHASE: Learn from training data (don't count regret)
    print(f"  🔥 Burn-in on {len(train_prompts)} training prompts...")
    for prompt in tqdm(train_prompts, desc=f"  Burn-in-{priors}", leave=False):
        model_id, log = router.route(prompt, profile="arbitrage")
        reward = train_oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        router.update(model_id, prompt, reward)
    
    # TEST PHASE: Evaluate on held-out test data (count regret)
    print(f"  ✅ Testing on {len(test_prompts)} test prompts...")
    selected_models = []
    selected_rewards = []
    
    # DEBUG: Track model selection distribution
    from collections import Counter
    model_selections = Counter()
    
    for i, prompt in enumerate(tqdm(test_prompts, desc=f"BanditGPT-{priors}", leave=False)):
        # ROUTE (uses full semantic features)
        model_id, log = router.route(prompt, profile="max_quality")
        model_selections[model_id] += 1
        
        # ORACLE LOOKUP
        reward = test_oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # DEBUG: Log first 10 selections
        if i < 10:
            print(f"    [{i}] Selected: {model_id[:30]:30s} | Reward: {reward:.3f}")
        
        # UPDATE (continue learning during test - online learning)
        # DISABLE for offline evaluation match with tuning script
        # router.update(model_id, prompt, reward)
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    # DEBUG: Print selection distribution
    print(f"\n  Model Selection Distribution ({priors}):")
    for model, count in model_selections.most_common(5):
        pct = 100 * count / len(test_prompts)
        print(f"    {model[:40]:40s}: {count:4d} ({pct:5.1f}%)")
    
    method_name = f"banditgpt_{priors}"
    if prior_n_effective is not None:
        method_name += f"_n{int(prior_n_effective)}"
    
    return {
        "method": method_name,
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def burn_in_router(
    train_prompts: List[str],
    train_oracle_rewards: Dict[str, Dict[str, float]],
    registry: Dict[str, Dict],
    encoder,
    priors: str = "hle",
    prior_n_effective: float = None
) -> BanditRouter:
    """
    Burn-in phase: Train router on training data.
    
    Returns a router that has learned from training data.
    This router can then be cloned for multiple test runs.
    """
    print(f"  🔥 Burn-in BanditGPT (priors={priors})...")
    
    # Fresh router
    create_kwargs = {
        "exploration": "balanced",  # Fix: align with LinUCB alpha=1.0
        "priors": priors,
        "context_encoder": encoder,
    }
    
    if prior_n_effective is not None:
        create_kwargs["prior_n_effective"] = prior_n_effective
    
    router = BanditRouter.create(registry, **create_kwargs)
    
    # Learn from training data
    for prompt in tqdm(train_prompts, desc=f"  Burn-in-{priors}", leave=False):
        model_id, log = router.route(prompt, profile="max_quality")
        reward = train_oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        router.update(model_id, prompt, reward)
    
    print(f"  ✅ Burn-in complete ({len(train_prompts)} training prompts)")
    return router


def test_router(
    router: BanditRouter,
    test_prompts: List[str],
    test_oracle_rewards: Dict[str, Dict[str, float]],
    priors: str = "hle",
    seed: int = 42
) -> Dict:
    """
    Test phase: Evaluate burned-in router on test data.
    
    The router continues to learn online during testing.
    """
    selected_models = []
    selected_rewards = []
    
    # Strict Greedy for Evaluation (Test the learned policy)
    # router.bandit.alpha = 0.0 # Disabled: Maintain LinUCB exploration for online learning
    
    for prompt in tqdm(test_prompts, desc=f"BanditGPT-{priors}", leave=False):
        # ROUTE
        model_id, log = router.route(prompt, profile="max_quality")
        
        # ORACLE LOOKUP
        reward = test_oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # UPDATE (continue learning online)
        router.update(model_id, prompt, reward)
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    method_name = f"banditgpt_{priors}"
    
    return {
        "method": method_name,
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


# Removed local implementations in favor of src.bandit_gpt.utils.experiment.ExperimentBurnIn

def analyze_by_difficulty(
    all_method_rewards: Dict[str, np.ndarray],
    test_prompts: List[str],
    oracle_best: np.ndarray,
    available_models: List[str],
    test_oracle_rewards: Dict[str, Dict[str, float]]
):
    """
    Categorizes prompts into 'Easy' vs 'Hard' and reports bucketed regret.
    
    'Hard' = Context Matters (High Variance in model performance)
    'Easy' = Safe Pick (Low Variance, most models give similar rewards)
    """
    print("\n" + "="*70)
    print("🔍 BREAKDOWN BY DIFFICULTY (Contextual Lift Analysis)")
    print("="*70)
    
    # 1. Classify Prompts by Oracle Variance
    hard_indices = []
    easy_indices = []
    
    for i, prompt in enumerate(test_prompts):
        rewards = [test_oracle_rewards[prompt].get(m, 0.0) for m in available_models]
        # High variance means routing is critical (models disagree)
        if np.var(rewards) > 0.05:
            hard_indices.append(i)
        else:
            easy_indices.append(i)
            
    print(f"  Bucket: EASY (Low Var)  |  n={len(easy_indices):4d} prompts")
    print(f"  Bucket: HARD (High Var) |  n={len(hard_indices):4d} prompts")
    print("-" * 70)
    
    # 2. Report Regret per Bucket
    print(f"{'Method':25s} | {'Easy Regret':12s} | {'Hard Regret':12s} | {'Total'}")
    print("-" * 70)
    
    for method, avg_rewards in all_method_rewards.items():
        # Calculate regret for each bucket
        # Regret = Oracle Best - Selected Reward
        easy_regret = np.sum(oracle_best[easy_indices] - avg_rewards[easy_indices])
        hard_regret = np.sum(oracle_best[hard_indices] - avg_rewards[hard_indices])
        total_regret = easy_regret + hard_regret
        
        print(f"{method:25s} | {easy_regret:12.1f} | {hard_regret:12.1f} | {total_regret:7.1f}")
    
    print("="*70)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    """Run all baseline comparisons with train/test split and burn-in."""
    parser = argparse.ArgumentParser(description="Run effectiveness baselines.")
    parser.add_argument("--models", type=str, help="Path to custom models.json")
    parser.add_argument("--warmup", type=str, help="Path to custom priors_warmup.joblib")
    parser.add_argument("--output", type=str, default="effectiveness_results.json", help="Output results filename")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 01: EFFECTIVENESS COMPARISON")
    print("Protocol: Curriculum Burn-In & Signal-Aware Oversampling")
    print("=" * 70)
    
    # 1. Load Data & Initialize Centralized Experiment Burn-In
    print("\n📦 Initializing Experiment Framework...")
    registry = load_model_registry(args.models)
    splits_path = Path(__file__).parent / "results" / "splits.json"
    
    # Pre-load full rewards pool for baseline lookups
    train_rewards_raw = load_oracle_rewards("lmsys_train_final_rewards_1k_clean.jsonl.gz")
    test_rewards_raw = load_oracle_rewards("lmsys_test_final_rewards_1k_clean.jsonl.gz")
    all_rewards = {**train_rewards_raw, **test_rewards_raw}
    
    # Initialize shared encoder
    print("🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    burner = ExperimentBurnIn(registry, all_rewards, splits_path, encoder=encoder)
    
    # 2. Get Canonical Splits with Rewards
    print("📊 Loading Canonical KDD Splits...")
    (dev_prompts, dev_rewards), (test_prompts_pool, test_rewards) = burner.get_splits(load_rewards=True)
    
    print(f"  ✓ Dev Set (Burn-in): {len(dev_prompts)}")
    print(f"  ✓ Test Set (Hold-out): {len(test_prompts_pool)}")
    
    # 3. Generate Curriculum from Dev
    print("\n🎓 Generating Signal-Aware Curriculum from Dev set...")
    burn_in_list = burner.generate_curriculum(dev_prompts)
    
    # 4. Identify 9-Model Portfolio based on Test Pool coverage
    print("⚖️  Identifying model portfolio...")
    model_coverage = defaultdict(int)
    for prompt in test_prompts_pool:
        # Check rewards in the combined pool for coverage check
        prompt_rewards = all_rewards.get(prompt, {})
        for model_id in prompt_rewards:
            model_coverage[model_id] += 1
    
    min_coverage = len(test_prompts_pool) * 0.5
    available_models = [
        m for m in registry.keys() 
        if model_coverage.get(m, 0) >= min_coverage
    ]
    print(f"  ✓ {len(available_models)} models with ≥50% coverage in test pool")
    
    # 5. Calculate Oracle Best for Hold-out set
    oracle_best = []
    valid_test_prompts = []
    
    print("  ⚖️  Calculating Fair Oracle (Hold-out Set)...")
    for prompt in test_prompts_pool:
        # Use the specific hold-out rewards for the test phase
        rewards = test_rewards.get(prompt, {})
        available_rewards = [
            rewards.get(m, 0.0) for m in available_models 
            if m in rewards
        ]
        
        if available_rewards:
            best_val = max(available_rewards)
            oracle_best.append(best_val)
            valid_test_prompts.append(prompt)
            
    oracle_best = np.array(oracle_best)
    test_prompts = valid_test_prompts
    print(f"  ✓ Oracle computed on {len(oracle_best)} valid hold-out prompts")

    # Experiment Configuration
    n_seeds = 10 # Robust stats
    results = {}
    method_raw_rewards = defaultdict(list)
    
    # Optimal Hyperparameters (from Gold Standard Tuning)
    # Typically found: N_eff=5.0, Alpha=0.05
    best_n_eff = 1.0
    best_alpha = 0.1
    
    experiments = [
        {"name": "Random", "priors": None, "strategy": "random"},
        {"name": "Cold Start (LinUCB)", "priors": "none", "strategy": "linucb"}, # Baseline A
        {"name": "HLE Priors (Diagonal)", "priors": "hle", "strategy": "linucb"}, # Baseline B
        {"name": "BanditGPT (Curriculum)", "priors": "warmup", "strategy": "banditgpt"} # OUR METHOD
    ]
    
    # -------------------------------------------------------------------------
    # BURN-IN PHASE (BanditGPT Only)
    # -------------------------------------------------------------------------
    # We create ONE burned-in router instance and clone it for seeds.
    # This represents the "Deployed Model" state.
    
    print("\n" + "="*40)
    print("🔥 BURN-IN PHASE (BanditGPT)")
    print("="*40)
    
    # KDD FIX: Explicitly pass the 23-dimension PCA path to avoid JIT trigger
    pca_path = Path(__file__).parent.parent.parent / "src" / "artifacts" / "pca_23.joblib"
    
    print(f"Initializing Master Router (PCA: {pca_path.name})...")
    master_router = BanditRouter.create(
        registry, 
        context_encoder=encoder,
        priors="warmup", 
        prior_n_effective=best_n_eff,
        alpha=best_alpha,
        pca_path=pca_path,
        warmup_path=args.warmup
    )
    
    burner.perform_burn_in(master_router, burn_in_list)
    
    
    # -------------------------------------------------------------------------
    # TEST PHASE
    # -------------------------------------------------------------------------
    print("\n" + "="*40)
    print(f"🚀 TEST PHASE ({n_seeds} Seeds)")
    print("="*40)
    
    # Pre-calculating RouteLLM scores (cached across seeds for efficiency)
    routellm_scores_cache = {}
    
    # Shuffle prompts ONCE for consistent ordering across all seeds
    test_data_combined = list(zip(test_prompts, oracle_best))
    rng_shuffle = np.random.RandomState(42)
    rng_shuffle.shuffle(test_data_combined)
    shuffled_prompts = [item[0] for item in test_data_combined]
    shuffled_oracle = np.array([item[1] for item in test_data_combined])

    for seed in range(n_seeds):
        print(f"\nSEED {seed + 1}/{n_seeds}")
        
        # 1. Random Baseline
        random_res = run_random_baseline(shuffled_prompts, all_rewards, available_models, seed=seed)
        
        # 2. Vanilla LinUCB (Starts COLD on Test Set)
        linucb_res = run_vanilla_linucb(shuffled_prompts, all_rewards, available_models, seed=seed)
        
        # 3. RouteLLM (Static) - Optimized with cache
        routellm_res = run_routellm_baseline(
            shuffled_prompts, all_rewards, registry, available_models, 
            cached_scores=routellm_scores_cache, seed=seed
        )
        
        # 4. BanditGPT with HLE (Starts COLD+HLE on Test Set, no burn-in)
        # To show benefit of burn-in vs just priors
        router_hle = BanditRouter.create(
            registry, 
            context_encoder=encoder,
            priors="hle", 
            prior_n_effective=10.0, # Default for HLE
            alpha=best_alpha,
            pca_path=pca_path,
            warmup_path=args.warmup
        )
        hle_res = test_router(router_hle, shuffled_prompts, all_rewards, priors="hle", seed=seed)

        # 5. BanditGPT (Curriculum Tuned)
        # Clone the Hot Router
        router_hot = copy.deepcopy(master_router)
        
        # CRITICAL: Reset random seed for this trial
        # The deepcopy preserves the RNG state, causing identical decisions across seeds
        # We need to reset the bandit's internal random state
        router_hot.bandit.rng = np.random.RandomState(seed)
        
        bandit_res = test_router(router_hot, shuffled_prompts, all_rewards, priors="warmup", seed=seed)
        
        # Collect Results
        run_results = [random_res, linucb_res, routellm_res, hle_res, bandit_res]
        
        for res in run_results:
            method = res["method"]
            cum_regret = calculate_cumulative_regret(res["rewards"], shuffled_oracle)
            
            if method not in results: results[method] = []
            results[method].append(cum_regret.tolist())
            
            method_raw_rewards[method].append(res["rewards"])
            
            print(f"  {method:25s}: Regret={cum_regret[-1]:7.1f}")

    # Analysis & Saving
    averaged_rewards = {
        m: np.mean(np.array(r), axis=0) for m, r in method_raw_rewards.items()
    }
    analyze_by_difficulty(
        averaged_rewards, 
        shuffled_prompts, # Last shuffle
        shuffled_oracle, # Last shuffle
        available_models, 
        all_rewards # Full lookup
    )
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_dir / args.output}")

if __name__ == "__main__":
    main()
