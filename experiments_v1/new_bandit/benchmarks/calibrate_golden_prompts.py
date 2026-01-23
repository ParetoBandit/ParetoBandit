#!/usr/bin/env python3
"""
Golden Prompt Calibration Experiment

Derives empirically-backed regression coefficients for the HLE → Prior transformation.

This script addresses the KDD reviewer critique about "unvalidated assumptions" by:
1. Loading real test data with actual success rates per model
2. Loading HLE scores from the model registry
3. Fitting regression curves (linear, polynomial, sigmoid)
4. Outputting calibrated parameters for RouterConfig

Usage:
    python calibrate_golden_prompts.py
"""

import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "offline_dataset"
MODELS_PATH = PROJECT_ROOT / "models.json"
TEST_REWARDS_PATH = DATA_DIR / "test_rewards_pareto_dedup.jsonl"
OUTPUT_PATH = Path(__file__).parent / "calibration_results.json"


def load_hle_scores() -> dict:
    """Load HLE (benchmark) scores from model registry."""
    with open(MODELS_PATH) as f:
        data = json.load(f)
    
    hle_scores = {}
    for m in data["models"]:
        model_id = m["openrouter_id"]
        # Use the 'hle' field directly
        hle = m.get("hle")
        if hle is not None:
            hle_scores[model_id] = float(hle)
    
    return hle_scores


def load_empirical_success_rates() -> dict:
    """
    Calculate empirical success rates from real test data.
    
    Returns:
        Dict: {model_id: {total: N, wins: W, success_rate: W/N}}
    """
    model_stats = defaultdict(lambda: {"total": 0, "wins": 0})
    
    print(f"   Loading from {TEST_REWARDS_PATH}...")
    with open(TEST_REWARDS_PATH) as f:
        for line_num, line in enumerate(f):
            entry = json.loads(line)
            if entry.get("ok"):
                model_id = entry["model_id"]
                raw_score = entry["raw_score"]
                
                model_stats[model_id]["total"] += 1
                if raw_score > 0.5:  # Binary success criterion
                    model_stats[model_id]["wins"] += 1
            
            if (line_num + 1) % 10000 == 0:
                print(f"   Processed {line_num + 1} entries...")
    
    # Calculate success rates
    for model_id in model_stats:
        stats = model_stats[model_id]
        if stats["total"] > 0:
            stats["success_rate"] = stats["wins"] / stats["total"]
        else:
            stats["success_rate"] = 0.0
    
    return dict(model_stats)


# Regression functions to fit
def linear(x, a, b):
    """Linear: y = a*x + b"""
    return a * x + b


def quadratic(x, a, b, c):
    """Quadratic: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c


def sigmoid(x, k, x0, L, b):
    """Sigmoid: y = L / (1 + exp(-k*(x-x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def two_tier(x, easy_floor, easy_slope, hard_threshold, hard_max):
    """Two-tiered (current implementation approximation)"""
    # Simple piecewise: linear for easy, power for hard
    result = np.where(
        x < hard_threshold,
        easy_floor + easy_slope * x,
        (x / hard_max) ** 2
    )
    return np.clip(result, 0.0, 1.0)


def main():
    print("=" * 70)
    print("GOLDEN PROMPT CALIBRATION EXPERIMENT")
    print("=" * 70)
    
    # 1. Load HLE scores
    print("\n📊 Loading HLE scores from model registry...")
    hle_scores = load_hle_scores()
    print(f"   ✓ Loaded HLE scores for {len(hle_scores)} models")
    
    # 2. Load empirical success rates
    print("\n📈 Calculating empirical success rates from test data...")
    success_rates = load_empirical_success_rates()
    print(f"   ✓ Calculated success rates for {len(success_rates)} models")
    
    # 3. Prepare data for regression
    # Only use models that have both HLE score and test results
    X = []  # HLE scores
    Y = []  # Empirical success rates
    models = []
    
    for model_id in success_rates:
        if model_id in hle_scores and success_rates[model_id]["total"] >= 10:
            hle = hle_scores[model_id]
            sr = success_rates[model_id]["success_rate"]
            
            # Filter valid HLE scores (0-1 range typically)
            if 0.0 <= hle <= 1.0:
                X.append(hle)
                Y.append(sr)
                models.append(model_id)
    
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"\n📐 Fitting regression curves on {len(X)} models...")
    print(f"   HLE range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Success rate range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    # 4. Fit regression curves
    results = {}
    
    # Linear fit
    try:
        popt, _ = curve_fit(linear, X, Y, p0=[1.0, 0.5], maxfev=10000)
        y_pred = linear(X, *popt)
        r2 = 1 - np.sum((Y - y_pred)**2) / np.sum((Y - np.mean(Y))**2)
        pearson_r, p_val = pearsonr(Y, y_pred)
        
        results["linear"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "p_value": float(p_val),
            "formula": f"y = {popt[0]:.4f} * x + {popt[1]:.4f}"
        }
        print(f"\n   Linear: R² = {r2:.4f}, r = {pearson_r:.4f}")
        print(f"           y = {popt[0]:.4f} * x + {popt[1]:.4f}")
    except Exception as e:
        print(f"\n   Linear fit failed: {e}")
    
    # Quadratic fit
    try:
        popt, _ = curve_fit(quadratic, X, Y, p0=[1.0, 1.0, 0.5], maxfev=10000)
        y_pred = quadratic(X, *popt)
        r2 = 1 - np.sum((Y - y_pred)**2) / np.sum((Y - np.mean(Y))**2)
        pearson_r, _ = pearsonr(Y, y_pred)
        
        results["quadratic"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])},
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "formula": f"y = {popt[0]:.4f} * x² + {popt[1]:.4f} * x + {popt[2]:.4f}"
        }
        print(f"\n   Quadratic: R² = {r2:.4f}, r = {pearson_r:.4f}")
        print(f"             y = {popt[0]:.4f}*x² + {popt[1]:.4f}*x + {popt[2]:.4f}")
    except Exception as e:
        print(f"\n   Quadratic fit failed: {e}")
    
    # Sigmoid fit
    try:
        # Initial guess: k=10, x0=0.2, L=0.5, b=0.5
        popt, _ = curve_fit(sigmoid, X, Y, p0=[10.0, 0.2, 0.5, 0.5], 
                           bounds=([0.1, 0.0, 0.0, 0.0], [200.0, 1.0, 1.0, 1.0]),
                           maxfev=10000)
        y_pred = sigmoid(X, *popt)
        r2 = 1 - np.sum((Y - y_pred)**2) / np.sum((Y - np.mean(Y))**2)
        pearson_r, _ = pearsonr(Y, y_pred)
        
        results["sigmoid"] = {
            "params": {"k": float(popt[0]), "x0": float(popt[1]), 
                      "L": float(popt[2]), "b": float(popt[3])},
            "r2": float(r2),
            "pearson_r": float(pearson_r),
            "formula": f"y = {popt[2]:.4f} / (1 + exp(-{popt[0]:.2f}*(x-{popt[1]:.4f}))) + {popt[3]:.4f}"
        }
        print(f"\n   Sigmoid: R² = {r2:.4f}, r = {pearson_r:.4f}")
        print(f"           k = {popt[0]:.2f}, x0 = {popt[1]:.4f}, L = {popt[2]:.4f}, b = {popt[3]:.4f}")
    except Exception as e:
        print(f"\n   Sigmoid fit failed: {e}")
    
    # 5. Determine best model
    best_model = max(results.items(), key=lambda x: x[1].get("r2", 0))
    
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    
    print(f"\n🏆 Best Fit: {best_model[0].upper()}")
    print(f"   R² = {best_model[1]['r2']:.4f}")
    print(f"   Formula: {best_model[1]['formula']}")
    
    # 6. Generate recommendations for RouterConfig
    print("\n" + "=" * 70)
    print("RECOMMENDED UPDATES TO RouterConfig")
    print("=" * 70)
    
    if "sigmoid" in results:
        sig = results["sigmoid"]["params"]
        print(f"""
# Calibrated Sigmoid Parameters (from Golden Prompt Experiment)
# R² = {results['sigmoid']['r2']:.4f}
prior_sigmoid_k: float = {sig['k']:.2f}
prior_sigmoid_center: float = {sig['x0']:.4f}  # x0
prior_sigmoid_L: float = {sig['L']:.4f}  # Scale
prior_sigmoid_b: float = {sig['b']:.4f}  # Offset
calibration_validated: bool = True
""")
    
    if "linear" in results:
        lin = results["linear"]["params"]
        print(f"""
# Alternative: Calibrated Linear Parameters
# R² = {results['linear']['r2']:.4f}
# easy_floor ≈ {lin['b']:.4f} (intercept)
# easy_slope ≈ {lin['a']:.4f} (slope)
""")
    
    # 7. Save results
    output = {
        "n_models": len(X),
        "hle_range": [float(X.min()), float(X.max())],
        "success_rate_range": [float(Y.min()), float(Y.max())],
        "models_used": models,
        "fits": results,
        "best_fit": best_model[0],
        "recommendation": f"Use {best_model[0]} with R²={best_model[1]['r2']:.4f}"
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
