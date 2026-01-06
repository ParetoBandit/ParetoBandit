#!/usr/bin/env python3
"""
tune_n_lmsys.py

Hyperparameter Tuning for Prior Stiffness (N).
Uses REAL LMSYS prompts (barbell distribution) + SIMULATED Rewards (IRT).

Protocol:
1. Load LMSYS barbell data (CLEANED: First turn only, no data leakage)
2. Normalize Model HLE scores to 0.0-1.0 probability space
3. Grid Search N in [1, 10, 50, 100, 250, 500, 1000]
4. Metric: Cumulative Regret against IRT Oracle

KDD Compliance:
- First-turn only (credit assignment)
- Zero data leakage (verified separately)
- Barbell distribution (stress tests)
- Normalized HLE (mathematical validity)
"""

import sys
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Adjust path to find source code  
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from bandit_gpt.router import BanditRouter
from bandit_gpt.utils import calibrate_complexity

# =============================================================================
# CONFIGURATION  
# =============================================================================
LMSYS_PATH = repo_root / "src/bandit_gpt/data/lmsys_barbell_20k_minimal.jsonl"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_PLOT = OUTPUT_DIR / "sensitivity_n_lmsys.pdf"
OUTPUT_JSON = OUTPUT_DIR / "sensitivity_n_lmsys.json"

# The Grid to Search
# Looking for "Goldilocks" zone between Flexible (10) and Rigid (1000)
N_GRID = [0, 10, 50, 100, 250, 500, 1000]

# =============================================================================
# HELPERS: NORMALIZATION & IRT PHYSICS
# =============================================================================

def normalize_hle(score: float, metric="hle") -> float:
    """
    CRUCIAL: Maps arbitrary scales to Probability (0.0-1.0).
    
    Without normalization, IRT math explodes (sigmoid → 1.0 for everything).
    
    Args:
        score: Raw HLE score
        metric: Type of metric ("hle", "elo", "percentage")
    
    Returns:
        Normalized probability [0.0, 1.0]
    """
    if score is None:
        return 0.5  # Neutral default
    
    if metric == "hle":
        # HLE range from models.json: ~0.03 to ~0.35
        MIN_HLE, MAX_HLE = 0.03, 0.35
        val = max(MIN_HLE, min(score, MAX_HLE))
        return (val - MIN_HLE) / (MAX_HLE - MIN_HLE)
    
    elif metric == "elo":
        # Typical Elo range: 950 (weak) to 1350 (GPT-4 class)
        MIN_ELO, MAX_ELO = 950.0, 1350.0
        val = max(MIN_ELO, min(score, MAX_ELO))
        return (val - MIN_ELO) / (MAX_ELO - MIN_ELO)
    
    elif metric == "percentage":
        return score / 100.0 if score > 1.0 else score
    
    return score


def irt_reward(skill: float, difficulty: float) -> float:
    """
    Item Response Theory: Calculates P(Success).
    
    Mathematical Foundation:
        P(Success) = 1 / (1 + exp(-a(θ - β)))
        where:
            θ (theta) = skill level
            β (beta) = difficulty level  
            a = discrimination parameter (1.5)
    
    CRITICAL: Both inputs MUST be in [0.0, 1.0] range.
    
    Args:
        skill: Normalized model capability [0.0, 1.0]
        difficulty: Normalized prompt difficulty [0.0, 1.0]
    
    Returns:
        Success probability [0.0, 1.0]
    """
    # Transform to logit scale (-3 to +3)
    theta = (skill - 0.5) * 6.0
    beta = (difficulty - 0.5) * 6.0
    
    # Discrimination parameter (curve sharpness)
    a = 1.5
    
    # IRT formula
    logit = a * (theta - beta)
    
    return 1.0 / (1.0 + math.exp(-logit))


def load_barbell_prompts():
    """
    KDD-Grade Data Loader.
    
    Loads barbell-distributed LMSYS data:
    - First turn only (no credit assignment problem)
    - Zero train/test leakage (verified separately)
    - Balanced categories (STEM, CODE, GENERAL)
    - Stress tests included (easy→hard, hard→easy)
    
    Returns:
        List of prompt texts
    """
    prompts = []
    print(f"📂 Loading barbell dataset from {LMSYS_PATH}...")
    
    if not LMSYS_PATH.exists():
        raise FileNotFoundError(
            f"Missing data: {LMSYS_PATH}\n"
            f"Run: python scripts/sample_barbell_from_1m.py"
        )
    
    with open(LMSYS_PATH, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Extract prompt directly (minimal dataset has prompt as field)
                prompt = data.get('prompt', '')
                if prompt:
                    prompts.append(prompt)
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"✓ Loaded {len(prompts)} prompts")
    
    return prompts


def load_model_hle_map():
    """
    Load HLE scores from models.json.
    
    Returns:
        Dict mapping model_id -> normalized_hle
    """
    models_path = Path("src/bandit_gpt/config/models.json")
    
    with open(models_path) as f:
        models_data = json.load(f)
    
    hle_map = {}
    for model in models_data["models"]:
        model_id = model.get("openrouter_id") or model.get("name")
        hle = model.get("hle")
        
        if model_id and hle is not None:
            # Normalize to 0.0-1.0
            hle_map[model_id] = normalize_hle(hle, metric="hle")
    
    print(f"✓ Loaded HLE scores for {len(hle_map)} models")
    return hle_map


# =============================================================================
# MAIN TUNING LOOP
# =============================================================================

def main():
    print("="*70)
    print("🔬 Prior Strength (N) Tuning - LMSYS Barbell Dataset")
    print("="*70)
    print()
    
    # 1. Setup
    print("⚙️  Setup...")
    prompts = load_barbell_prompts()
    hle_map = load_model_hle_map()
    
    # Use subset for faster tuning (3K is enough to see U-curve)
    # For final paper, use all prompts
    validation_prompts = prompts[:3000]
    print(f"✓ Using {len(validation_prompts)} prompts for grid search")
    print()
    
    results = {}
    
    # 2. Grid Search
    for n in N_GRID:
        print(f"{'='*70}")
        print(f"Testing N = {n}")
        print(f"{'='*70}")
        
        # Initialize Router with specific N
        try:
            router = BanditRouter.create(
                registry="default",
                priors="warmup",              # Load warmup A-matrix
                prior_n_effective=float(n),   # Override N (stiffness)
                exploration="safe"            # α=0.1
            )
        except Exception as e:
            print(f"❌ Failed to initialize router: {e}")
            continue
        
        cumulative_regret = 0.0
        
        # 3. Validation Stream
        for i, prompt in enumerate(tqdm(validation_prompts, desc=f"N={n}", leave=False)):
            
            # A. Router Decision
            model_id, log = router.route(prompt, profile="arbitrage")
            
            # B. Ground Truth via IRT Physics
            difficulty = calibrate_complexity(prompt)  # Returns 0.0-1.0
            
            # Get normalized skill for chosen model
            sel_skill = hle_map.get(model_id, 0.5)  # Default to median
            reward_actual = irt_reward(sel_skill, difficulty)
            
            # C. Oracle Baseline (Best Possible Reward)
            max_reward = 0.0
            for m_id in router.models:
                m_skill = hle_map.get(m_id, 0.5)
                possible_reward = irt_reward(m_skill, difficulty)
                if possible_reward > max_reward:
                    max_reward = possible_reward
            
            # D. Calculate Regret
            regret = max_reward - reward_actual
            cumulative_regret += regret
            
            # E. Update Router (Does N allow learning?)
            router.update(model_id, log.context_vector, reward_actual)
            
            # Progress update every 500 prompts
            if (i + 1) % 500 == 0:
                avg_regret = cumulative_regret / (i + 1)
                print(f"  Prompt {i+1}/{len(validation_prompts)}: "
                      f"Cumulative Regret = {cumulative_regret:.2f}, "
                      f"Avg = {avg_regret:.4f}")
        
        # Log results
        avg_regret = cumulative_regret / len(validation_prompts)
        results[n] = cumulative_regret
        
        print(f"\n✓ N={n}: Total Regret = {cumulative_regret:.2f}, Avg = {avg_regret:.4f}")
        print()
    
    # =========================================================================
    # ANALYSIS & PLOTTING
    # =========================================================================
    
    if not results:
        print("❌ No results generated.")
        return
    
    best_n = min(results, key=results.get)
    print("="*70)
    print(f"🏆 OPTIMAL HYPERPARAMETER FOUND: N = {best_n}")
    print(f"   Minimum Regret: {results[best_n]:.2f}")
    print("="*70)
    print()
    
    # Save numerical results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            'n_grid': N_GRID,
            'cumulative_regret': {str(k): v for k, v in results.items()},
            'optimal_n': best_n,
            'n_prompts': len(validation_prompts)
        }, f, indent=2)
    
    print(f"💾 Saved results to {OUTPUT_JSON}")
    
    # Generate KDD-Style Plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_vals = sorted(results.keys())
    y_vals = [results[x] for x in x_vals]
    
    # Plot curve
    ax.plot(x_vals, y_vals, marker='o', linewidth=2.5, markersize=8, 
            color='#2563eb', label='Cumulative Regret')
    ax.set_xscale('log')
    
    # Highlight optimal N
    min_y = results[best_n]
    ax.scatter([best_n], [min_y], color='#dc2626', s=200, zorder=5,
               marker='*', edgecolors='black', linewidths=1.5,
               label=f'Optimal N = {best_n}')
    
    # Annotate minimum
    ax.annotate(
        f'Optimal N = {best_n}\nRegret = {min_y:.1f}',
        xy=(best_n, min_y),
        xytext=(30, 30),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5)
    )
    
    # Styling
    ax.set_xlabel('Prior Strength ($N_{eff}$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Regret (Lower is Better)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Prior Strength on Real-World Data (LMSYS Barbell)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add region labels
    if best_n > 0:
        # Left: Over-exploration
        left_n = [n for n in x_vals if n < best_n and n > 0]
        if left_n:
            ax.axvspan(min([n for n in x_vals if n > 0]), max(left_n), 
                      alpha=0.1, color='red', label='Too Nervous')
        
        # Right: Under-exploration
        right_n = [n for n in x_vals if n > best_n]
        if right_n:
            ax.axvspan(min(right_n), max(x_vals), 
                      alpha=0.1, color='orange', label='Too Rigid')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "sensitivity_n_lmsys.png", dpi=300, bbox_inches='tight')
    
    print(f"📊 Plot saved to {OUTPUT_PLOT}")
    print()
    
    # Interpretation
    print("📈 Curve Interpretation:")
    print("   U-Shaped Curve = GOOD (Goldilocks zone exists)")
    print("   Flat Curve = BAD (N doesn't matter → warmup too strong/weak)")
    print()
    print("   Left side (low N) high regret = Router too jittery (over-learns noise)")
    print("   Right side (high N) high regret = Router too rigid (ignores data)")
    print(f"   Bottom (N={best_n}) = Optimal balance")
    print()
    print("✅ Include this plot in your 'Hyperparameter Sensitivity' section")

if __name__ == "__main__":
    main()
