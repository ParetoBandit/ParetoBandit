#!/usr/bin/env python3
"""
HLE Calibration: Fit HLE → Utility Curve Using Training Data

Uses:
- models.json: HLE scores for each model
- train_rewards_1k.jsonl: Actual success rates on TRAINING prompts (~1000)

Goal: Empirically validate/replace the arbitrary sigmoid transformation.

IMPORTANT: We use TRAINING data for calibration (prior design).
           Test data is reserved for evaluation (no leakage).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Paths
project_root = Path(__file__).parent.parent.parent
models_path = project_root / "models.json"
train_rewards_path = project_root / "data" / "train_rewards_1k.jsonl"  # Use TRAINING data

print("="*80)
print("HLE CALIBRATION: Empirical Validation (Using Training Data)")
print("="*80)

# Load models.json
print("\n1. Loading models.json...")
with open(models_path) as f:
    models_data = json.load(f)
    
registry = {m["openrouter_id"]: m for m in models_data["models"]}
print(f"   ✓ Loaded {len(registry)} models")

# Load training rewards and calculate success rates
print("\n2. Calculating actual success rates from training data...")
model_scores = defaultdict(lambda: {"total": 0, "successes": 0})

with open(train_rewards_path) as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("ok"):
            model_id = entry["model_id"]
            raw_score = entry["raw_score"]
            
            # Binary success: score > 0.5 (or use your threshold)
            is_success = raw_score > 0.5
            
            model_scores[model_id]["total"] += 1
            if is_success:
                model_scores[model_id]["successes"] += 1

# Calculate success rates and pair with HLE scores
calibration_data = []
for model_id, scores in model_scores.items():
    if model_id in registry and scores["total"] >= 10:  # Min 10 samples
        hle_score = registry[model_id].get("hle")
        if hle_score is not None and hle_score > 0:
            success_rate = scores["successes"] / scores["total"]
            model_name = registry[model_id].get("display_name", model_id)
            
            calibration_data.append({
                "model": model_name,
                "model_id": model_id,
                "hle": hle_score,
                "success_rate": success_rate,
                "n_samples": scores["total"]
            })

# Sort by HLE for visualization
calibration_data.sort(key=lambda x: x["hle"])

print(f"   ✓ Collected data for {len(calibration_data)} models")
print(f"\n   Model samples (showing first 10):")
print(f"   {'Model':<30} {'HLE':<8} {'Success Rate':<15} {'N':<8}")
print(f"   {'-'*70}")
for d in calibration_data[:10]:
    print(f"   {d['model'][:28]:<30} {d['hle']:<8.3f} {d['success_rate']:<15.3f} {d['n_samples']:<8}")

# Extract arrays for curve fitting
hle_scores = np.array([d["hle"] for d in calibration_data])
success_rates = np.array([d["success_rate"] for d in calibration_data])
weights = np.array([d["n_samples"] for d in calibration_data])  # Weight by sample size

print("\n3. Fitting curves...")

# Current sigmoid (for comparison)
def current_sigmoid(x):
    k = 80.0
    x0 = 0.20
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

# Option A: Linear fit
def linear(x, m, b):
    return m * x + b

try:
    params_linear, cov_linear = curve_fit(linear, hle_scores, success_rates, sigma=1/np.sqrt(weights))
    r2_linear = 1 - (np.sum((success_rates - linear(hle_scores, *params_linear))**2) / 
                     np.sum((success_rates - np.mean(success_rates))**2))
    print(f"\n   Linear Fit:")
    print(f"      utility = {params_linear[0]:.4f} * hle + {params_linear[1]:.4f}")
    print(f"      R² = {r2_linear:.4f}")
except Exception as e:
    print(f"   Linear fit failed: {e}")
    params_linear = None

# Option B: Calibrated sigmoid
def sigmoid(x, k, x0, L, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b

try:
    # Initial guess: similar to current parameters
    p0 = [80.0, 0.20, 1.0, 0.0]
    params_sigmoid, cov_sigmoid = curve_fit(sigmoid, hle_scores, success_rates, 
                                            p0=p0, sigma=1/np.sqrt(weights), maxfev=5000)
    r2_sigmoid = 1 - (np.sum((success_rates - sigmoid(hle_scores, *params_sigmoid))**2) / 
                      np.sum((success_rates - np.mean(success_rates))**2))
    
    print(f"\n   Calibrated Sigmoid Fit:")
    print(f"      k = {params_sigmoid[0]:.2f} (steepness)")
    print(f"      x0 = {params_sigmoid[1]:.4f} (center)")
    print(f"      L = {params_sigmoid[2]:.4f} (max)")
    print(f"      b = {params_sigmoid[3]:.4f} (offset)")
    print(f"      R² = {r2_sigmoid:.4f}")
except Exception as e:
    print(f"   Sigmoid fit failed: {e}")
    params_sigmoid = None

# Calculate RMSE for current sigmoid
current_predictions = current_sigmoid(hle_scores)
rmse_current = np.sqrt(np.mean((success_rates - current_predictions)**2))
print(f"\n   Current Sigmoid (k=80, x0=0.20):")
print(f"      RMSE = {rmse_current:.4f}")

# Plot
print("\n4. Generating calibration plot...")
plt.figure(figsize=(12, 8))

# Scatter: actual data
plt.scatter(hle_scores, success_rates, s=weights/10, alpha=0.6, label="Actual Data (size=n_samples)")

# Current sigmoid
x_plot = np.linspace(0, 0.5, 200)
plt.plot(x_plot, current_sigmoid(x_plot), 'r--', linewidth=2, 
         label=f"Current Sigmoid (k=80, x0=0.20, RMSE={rmse_current:.3f})")

# Fitted curves
if params_linear is not None:
    plt.plot(x_plot, linear(x_plot, *params_linear), 'g-', linewidth=2,
             label=f"Linear Fit (R²={r2_linear:.3f})")

if params_sigmoid is not None:
    plt.plot(x_plot, sigmoid(x_plot, *params_sigmoid), 'b-', linewidth=2,
             label=f"Calibrated Sigmoid (R²={r2_sigmoid:.3f})")

plt.xlabel("HLE Score", fontsize=12)
plt.ylabel("Success Rate (on training prompts)", fontsize=12)
plt.title("HLE Calibration: Training Data Fit", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)
plt.ylim(0, 1.0)

output_dir = Path(__file__).parent
output_path = output_dir / "hle_calibration_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved plot to: {output_path}")

# Save results
results = {
    "current_sigmoid": {
        "k": 80.0,
        "x0": 0.20,
        "rmse": float(rmse_current)
    },
    "linear_fit": {
        "m": float(params_linear[0]) if params_linear is not None else None,
        "b": float(params_linear[1]) if params_linear is not None else None,
        "r2": float(r2_linear) if params_linear is not None else None
    } if params_linear is not None else None,
    "calibrated_sigmoid": {
        "k": float(params_sigmoid[0]),
        "x0": float(params_sigmoid[1]),
        "L": float(params_sigmoid[2]),
        "b": float(params_sigmoid[3]),
        "r2": float(r2_sigmoid)
    } if params_sigmoid is not None else None,
    "n_models": len(calibration_data),
    "total_samples": int(np.sum(weights))
}

results_path = output_dir / "hle_calibration_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
    
print(f"   ✓ Saved results to: {results_path}")

print("\n" + "="*80)
print("CALIBRATION COMPLETE")
print("="*80)

# Recommendations
print("\nRecommendations:")
if params_sigmoid is not None and r2_sigmoid > 0.85:
    print(f"✅ Calibrated sigmoid has good fit (R²={r2_sigmoid:.3f} > 0.85)")
    print(f"   Update RouterConfig:")
    print(f"      prior_sigmoid_k = {params_sigmoid[0]:.2f}")
    print(f"      prior_sigmoid_center = {params_sigmoid[1]:.4f}")
    print(f"      calibration_validated = True")
elif params_linear is not None and r2_linear > 0.85:
    print(f"✅ Linear fit has good fit (R²={r2_linear:.3f} > 0.85)")
    print(f"   Consider replacing sigmoid with: utility = {params_linear[0]:.4f} * hle + {params_linear[1]:.4f}")
else:
    print(f"⚠️  Current sigmoid (RMSE={rmse_current:.3f}) may be acceptable")
    print(f"   Collect more diverse test data if calibration is poor")
