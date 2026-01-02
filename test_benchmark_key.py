#!/usr/bin/env python3
"""Quick test to see which code path is taken"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

# Temporarily add debug output to bandit.py
from banditgpt import BanditRouter

models_path = Path(__file__).parent / "banditgpt" / "models.json"
with open(models_path) as f:
    data = json.load(f)
registry = {m["openrouter_id"]: m for m in data["models"]}

print("Creating HLE router (should use benchmark_key=='hle' path)...")
hle = BanditRouter.create(registry, priors="hle", prior_n_effective=10.0)
print(f"HLE benchmark_key: {hle.benchmark_key}")

print("\nCreating CSR router (should use benchmark_key!='hle' path)...")
csr = BanditRouter.create(registry, priors="csr", prior_n_effective=10.0)
print(f"CSR benchmark_key: {csr.benchmark_key}")

# Check b vectors
sample_model = list(registry.keys())[0]
print(f"\nSample model: {sample_model}")
print(f"HLE b[:5]: {hle.bandit.b[sample_model][:5]}")
print(f"CSR b[:5]: {csr.bandit.b[sample_model][:5]}")
print(f"Are they different? {not all(hle.bandit.b[sample_model] == csr.bandit.b[sample_model])}")
