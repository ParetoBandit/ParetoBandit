#!/usr/bin/env python3
"""
Run 5-Fold Cross-Validation in parallel.
"""
import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "final_release" / "train_judge.py"

def main():
    processes = []
    print("Starting 5 parallel training jobs...")
    
    for fold in range(5):
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--fold", str(fold)]
        # Redirect output to separate log files
        log_file = open(f"train_fold_{fold+1}.log", "w")
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((p, log_file))
        print(f"Started Fold {fold+1} (PID: {p.pid})")
        
    print("\nWaiting for all jobs to complete...")
    
    try:
        for p, log_file in processes:
            p.wait()
            log_file.close()
            if p.returncode != 0:
                print(f"Job failed with return code {p.returncode}")
            else:
                print(f"Job finished successfully.")
                
        print("\nAll jobs completed. Checking results...")
        
        # Aggregate results
        import json
        results = []
        for fold in range(5):
            res_file = PROJECT_ROOT / "final_release" / f"results_fold_{fold+1}.json"
            if res_file.exists():
                with open(res_file) as f:
                    results.append(json.load(f))
            else:
                print(f"Missing result file for Fold {fold+1}")

        if results:
            import numpy as np
            avg_f1 = np.mean([r['f1'] for r in results])
            avg_acc = np.mean([r['acc'] for r in results])
            print(f"\nAVERAGE: F1={avg_f1:.3f} | Acc={avg_acc:.3f}")
            
    except KeyboardInterrupt:
        print("\nTerminating all jobs...")
        for p, log_file in processes:
            p.terminate()
            log_file.close()

if __name__ == "__main__":
    main()
