# BanditGPT Scripts

Utility scripts for BanditGPT development and training.

## Warmup Generation

### `generate_warmup.py`

Generates synthetic "warmup" priors using Item Response Theory (IRT) simulation.

**Purpose**: Pre-train the LinUCB bandit with 20,000 simulated interactions to build dense covariance matrices and informed belief vectors. This enables "smart" router initialization without requiring expensive API calls.

**Usage**:
```bash
python scripts/generate_warmup.py
```

**Output**: Creates `data/priors_warmup.joblib` (~10-50 MB)

**Runtime**: ~2-5 minutes on modern hardware

**Scientific Foundation**:
- Uses 1-Parameter Logistic Model (Rasch Model) from psychometrics
- Simulates `P(success) = sigmoid(discriminability * (model_skill - task_difficulty))`
- Model skill derived from HLE scores in `models.json`
- Task difficulty computed from router's zero-shot complexity detection

**Configuration**:
- `N_SAMPLES`: Number of synthetic prompts (default: 20,000)
- `SEED`: Random seed for reproducibility (default: 42)
- `OUTPUT_PATH`: Save location (default: `data/priors_warmup.joblib`)

### Important Note on Estimated Fields

Since the source data table only provided Input Cost and Quality, Standard Industry Heuristics were used for the missing fields. You may want to adjust these if you have exact data:

- **Output Cost**: Estimated at **3x Input Cost** (Standard ratio for most providers).
- **Latency (`time_to_first_token_seconds`)**: Estimated based on parameter count (e.g., 4B models ≈ 0.18s, 70B+ models ≈ 0.5s+).
- **Context Length**: Defaulted to **128k** (Standard for modern models), except:
  - `gemini-2.5-flash-preview`: Set to **1M**
  - `gpt-oss-20b`: Set to **16k** (conservative estimate)

**Using Warmup in Experiments**:
```python
from src.bandit_gpt.router import BanditRouter

# Load pre-trained warmup state
router = BanditRouter.create(
    priors="warmup",  # Instead of "hle" or "none"
    exploration="safe"
)
```

**Verification**:
```python
import joblib

# Inspect the warmup artifact
data = joblib.load("data/priors_warmup.joblib")
print(f"Models trained: {len(data['A'])}")
print(f"Dimension: {data['A'][list(data['A'].keys())[0]].shape}")
print(f"Training samples: {data['n']}")
```

## Future Scripts

- `validate_priors.py`: Compare warmup vs. HLE priors on held-out data
- `benchmark_warmup.py`: Measure regret reduction from warmup initialization
- `export_weights.py`: Convert LinUCB state to human-readable format
