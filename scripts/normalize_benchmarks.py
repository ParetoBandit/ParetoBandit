#!/usr/bin/env python3
"""
Normalize benchmark scores using Min-Max normalization.
Formula: Norm(S) = (S - S_floor) / (S_ceiling - S_floor)
"""
import json
from pathlib import Path

# Path to models.json
MODELS_PATH = Path(__file__).parent.parent / "src/bandit_gpt/config/models.json"

def normalize_benchmarks():
    # Load the models.json file
    with open(MODELS_PATH, 'r') as f:
        data = json.load(f)
    
    models = data['models']
    
    # Find floor and ceiling values for each benchmark
    # Based on user request: Floor = Llama-2-7b (approximated by cheapest/worst model)
    # Ceiling = Claude 3.5 Sonnet / o1 (approximated by best model)
    
    benchmarks = ['hle', 'GPQA', 'Livecode', 'IFbench']
    
    # Extract all values for each benchmark
    benchmark_values = {b: [] for b in benchmarks}
    for model in models:
        for benchmark in benchmarks:
            if benchmark in model:
                benchmark_values[benchmark].append(model[benchmark])
    
    # Calculate floor (min) and ceiling (max) for each benchmark
    floors = {b: min(benchmark_values[b]) for b in benchmarks}
    ceilings = {b: max(benchmark_values[b]) for b in benchmarks}
    
    print("Floor and Ceiling values:")
    for benchmark in benchmarks:
        print(f"  {benchmark}: floor={floors[benchmark]:.4f}, ceiling={ceilings[benchmark]:.4f}")
    print()
    
    # Apply Min-Max normalization to each model
    for model in models:
        for benchmark in benchmarks:
            if benchmark in model:
                score = model[benchmark]
                floor = floors[benchmark]
                ceiling = ceilings[benchmark]
                
                # Min-Max normalization formula
                if ceiling - floor > 0:
                    normalized_score = (score - floor) / (ceiling - floor)
                else:
                    normalized_score = 0.0  # Handle edge case where all values are the same
                
                # Add normalized field
                norm_field = f"norm_{benchmark}"
                model[norm_field] = round(normalized_score, 4)
        
        # Calculate weighted average quality score
        # Weights: 40% norm_hle, 25% norm_GPQA, 20% norm_Livecode, 15% norm_IFbench (sum = 100%)
        weights = {
            'norm_hle': 0.40,
            'norm_GPQA': 0.25,
            'norm_Livecode': 0.20,
            'norm_IFbench': 0.15
        }
        
        quality_score = 0.0
        for norm_field, weight in weights.items():
            if norm_field in model:
                quality_score += model[norm_field] * weight
        
        model['initial_quality'] = round(quality_score, 4)
    
    # Save the updated models.json
    with open(MODELS_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Successfully added normalized benchmark scores to {MODELS_PATH}")
    print("\nNormalized fields added:")
    for benchmark in benchmarks:
        print(f"  - norm_{benchmark}")
    print("  - initial_quality (weighted average: 40% hle, 25% GPQA, 20% Livecode, 15% IFbench)")

if __name__ == "__main__":
    normalize_benchmarks()
