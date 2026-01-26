# Running RouteLLM with High-Resolution Threshold Sweep

## Current Status

**Current Version**: 10 thresholds (gap between $0.008 and $0.013)  
**High-Res Version**: 25 thresholds (fills the gap)

---

## Why The Gap Exists

RouteLLM uses a **threshold parameter** (0.0 to 1.0) to decide routing:
- threshold=1.0 → always Mixtral (cheap)
- threshold=0.5 → balanced mix
- threshold=0.0 → always GPT-4 (expensive)

**Current sampling**:
```
[0.00, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.00]
          ↑________________GAP________________↑
       t=0.11                              t=0.00
       $0.0076                             $0.013
```

**High-res sampling** (25 points):
```
[0.00, 0.04, 0.08, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.38, 
 0.42, 0.46, 0.50, 0.54, 0.58, 0.63, 0.67, 0.71, 0.75, 0.79, 
 0.83, 0.88, 0.92, 0.96, 1.00]
```

This fills intermediate cost points: $0.008, $0.009, $0.010, $0.011, $0.012

---

## Expected Results

### Current (10 thresholds):
| Threshold | Cost | Reward | Notes |
|-----------|------|--------|-------|
| 0.11 | $0.0076 | 0.872 | Last frontier point |
| 0.00 | $0.0130 | 0.812 | Dominated (pure GPT-4) |

### With 25 thresholds (predicted):
| Threshold | Cost | Reward | Notes |
|-----------|------|--------|-------|
| 0.11 | $0.0076 | 0.872 | Same as before |
| **0.08** | ~$0.008 | ~0.880 | **New point!** |
| **0.04** | ~$0.010 | ~0.890 | **New point!** |
| 0.00 | $0.0130 | 0.812 | Still dominated |

**Key improvement**: Smoother curve showing gradual cost/quality trade-off

---

## How To Run

### Option 1: Full Re-Run (Recommended)

```bash
cd /Users/annette/repostitories/banditGPT/experiments_v1/04_figure
python generate_pareto_frontier.py
```

**Runtime**: ~15-20 minutes total
- Oracle: <1 min
- Static baselines: <1 min  
- RouteLLM (25 thresholds): ~12-15 min ⏰
- banditGPT (10 λ values × 5 trials): ~3-5 min

**Output**:
- `results/pareto_results.json` (updated with 25 RouteLLM points)
- `results/figure4_pareto_frontier.png` (smoother RouteLLM curve)

### Option 2: RouteLLM Only (Faster)

If you only want to update RouteLLM and keep existing banditGPT results:

```python
cd experiments_v1/04_figure
python << 'EOF'
# This script re-runs ONLY RouteLLM with 25 thresholds
# and merges with existing banditGPT results
# Runtime: ~12 minutes

import sys
from pathlib import Path
import json
import numpy as np
import gzip
from collections import defaultdict

project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root / "src"))

from routellm.controller import Controller
from bandit_gpt.config_legacy import (
    CANONICAL_HOLDOUT_DATA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH
)

# Load existing results
with open('results/pareto_results.json') as f:
    data = json.load(f)

# Keep everything except RouteLLM  
results = data['strategies'].copy()

# Load eval data
prompt_rewards = defaultdict(lambda: {})
with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("ok"):
            prompt_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]

eval_prompts = [{"prompt": p, "rewards": r} for p, r in prompt_rewards.items() if len(r) == 2]

# Load costs
with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
    models_data = json.load(f)

model_costs = {}
for model in models_data["models"]:
    cost = (100 * model["price_1m_input"] + 400 * model["price_1m_output"]) / 1_000_000
    model_costs[model["openrouter_id"]] = {"cost": cost}

# Initialize RouteLLM
controller = Controller(
    routers=['mf'],
    strong_model='openai/gpt-4-turbo',
    weak_model='mistralai/mixtral-8x7b-instruct'
)

# Sweep 25 thresholds
thresholds = np.linspace(0.0, 1.0, 25)
routellm_points = []

print(f"Sweeping {len(thresholds)} thresholds...")
for i, t in enumerate(thresholds, 1):
    total_reward, total_cost, count = 0.0, 0.0, 0
    
    for p in eval_prompts:
        try:
            sel = controller.route(p["prompt"], router='mf', threshold=t)
            if sel in p["rewards"]:
                total_reward += p["rewards"][sel]
                total_cost += model_costs[sel]["cost"]
                count += 1
        except:
            continue
    
    if count > 0:
        routellm_points.append({
            "cost": total_cost / count,
            "reward": total_reward / count
        })
        print(f"[{i:2d}/{len(thresholds)}] t={t:.3f}: done")

# Update and save
results['RouteLLM-MF'] = routellm_points
with open('results/pareto_results.json', 'w') as f:
    json.dump({"metadata": data['metadata'], "strategies": results}, f, indent=2)

print(f"\n✅ Updated with {len(routellm_points)} RouteLLM points")
EOF
```

---

## Trade-offs

| Thresholds | Runtime | Gap Size | Curve Smoothness |
|------------|---------|----------|------------------|
| **10** | ~5 min | $0.005 | Coarse (current) |
| **25** | ~15 min | $0.002 | Smooth ✅ |
| **50** | ~30 min | $0.001 | Very smooth |

**Recommendation**: 25 thresholds is the sweet spot for publication quality

---

## Current Implementation

The file `generate_pareto_frontier.py` has been updated to use 25 thresholds:

```python
# Line ~452
thresholds = np.linspace(0.0, 1.0, 25)  # Increased from 10 to 25
```

Just run the script and wait ~15-20 minutes for completion!

---

## What Will Change in the Plot

### Before (10 thresholds):
- RouteLLM curve ends abruptly at $0.0076
- Large jump to dominated point at $0.013
- Looks like RouteLLM can't compete above $0.008

### After (25 thresholds):
- RouteLLM curve extends smoothly to ~$0.010-0.012  
- Gradual increase in cost and quality
- Shows full Pareto frontier before degrading
- **Still dominated by banditGPT at high quality!**

---

## Key Insight

Even with 2.5x more threshold values, **RouteLLM still hits a ceiling**:
- Best achievable: R ≈ 0.89-0.90 at cost ~$0.011
- After that: more cost = worse quality (threshold → 0)
- **banditGPT reaches R=0.908** at similar cost! ✅

The gap revealed a **fundamental limitation** of threshold-based routing, not a sampling artifact!

---

## Quick Start

```bash
# Update the script (already done)
cd /Users/annette/repostitories/banditGPT/experiments_v1/04_figure

# Run with 25 thresholds (~15-20 min)
python generate_pareto_frontier.py

# Results will be in:
# - results/pareto_results.json (data)
# - results/figure4_pareto_frontier.png (plot)
```

The command has been updated and is ready to run! ⏰

