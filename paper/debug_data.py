
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_jury import ModelRegistry

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "models_cache.json"

def load_models_data():
    """Load raw model data from cache."""
    with open(CACHE_FILE, 'r') as f:
        return json.load(f)

def check_data():
    print(f"Checking data from {CACHE_FILE}")
    if not CACHE_FILE.exists():
        print("Cache file does not exist!")
        return

    models_data = load_models_data()
    print(f"Loaded {len(models_data)} models.")
    
    registry = ModelRegistry.load_cache(verbose=False)
    print(f"Registry loaded {len(registry)} models.")

    # Check baseline
    baseline_model = None
    for m in registry:
        if 'Gemini 2.5 Pro' in m.name and 'Flash' not in m.name:
            baseline_model = m
            break
    
    if baseline_model:
        print(f"Found baseline model: {baseline_model.name}")
    
    if models_data:
        print("Sample model keys:")
        print(models_data[0].keys())
        print("Sample model data:")
        print(json.dumps(models_data[0], indent=2))

if __name__ == "__main__":
    check_data()
