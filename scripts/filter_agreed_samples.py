#!/usr/bin/env python3
"""
Filter comparison results to keep only samples where HelpSteer2 and Gemini 3 Flash agree.

Usage:
    python scripts/filter_agreed_samples.py
"""

import json
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "results" / "helpsteer_gemini_comparison.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "final_release" / "data" / "helpsteer_gemini_agreed.jsonl"

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE}...")
    
    agreed_samples = []
    total = 0
    
    with open(INPUT_FILE, "r") as f:
        for line in f:
            total += 1
            try:
                data = json.loads(line)
                if data.get("match") is True:
                    # Keep relevant fields for training
                    sample = {
                        "prompt": data["prompt"],
                        "response": data["response"],
                        "correctness": data["helpsteer_score"], # Keep original score? Or just label?
                        # The trainer uses correctness score. 
                        # If they agree, it means Gemini said "Correct" (score > 2.5) or "Incorrect" (score <= 2.5).
                        # We should probably keep the original HelpSteer score so the trainer can still use it 
                        # (it might use regression or thresholding).
                        # But wait, if we filter by agreement, we are essentially confirming the label.
                        "is_bad": 1 if data["gemini_label"] == "Incorrect" else 0,
                        "source": "helpsteer_agreed"
                    }
                    agreed_samples.append(sample)
            except json.JSONDecodeError:
                continue
                
    print(f"Total samples processed: {total}")
    print(f"Agreed samples: {len(agreed_samples)}")
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        for sample in agreed_samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
