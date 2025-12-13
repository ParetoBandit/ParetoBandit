#!/usr/bin/env python3
"""Check progress of running tests."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_full_spectrum_results.json"

if results_path.exists():
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    n_models = data['metadata']['n_models']
    n_problems = data['metadata']['n_problems']
    
    completed_models = len([m for m in data['models'] if m['total'] == n_problems])
    
    print(f"Progress: {completed_models}/{n_models} models completed")
    
    if completed_models > 0:
        print(f"\nRecent results:")
        for m in data['models'][-3:]:
            if m['total'] > 0:
                print(f"  {m['name'][:40]:<42} {m['correct']}/{m['total']} = {m['accuracy']:.1f}%")
else:
    print("Test not started yet or results file not created")
    print("Checking if process is running...")
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if 'run_arc_full_spectrum' in result.stdout:
        print("✓ Process is running")
        print("  (Datasets may be loading... this can take 1-2 minutes)")
    else:
        print("✗ Process not found")
