# Calibration Tools

Command-line tools for calibrating BanditGPT routers to new domains.

## Tools

### `find_gamma.py` - Find Optimal Calibration Factor

Determines the best gamma (covariance inflation) factor for your calibration dataset.

**Usage:**
```bash
python3 find_gamma.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --output results/
```

**Output:**
- `results/gamma_analysis.png` - Visualization of adaptation curves
- `results/gamma_results.json` - Numerical results
- Console recommendation for optimal gamma

### `calibrate_router.py` - Calibrate Router

Creates a production-ready router using the optimal gamma factor.

**Usage:**
```bash
python3 calibrate_router.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --gamma 0.010 \
  --output my_router.joblib
```

**Output:**
- `my_router.joblib` - Calibrated router ready for deployment

## Data Format

Both tools expect calibration data in JSONL format:

```jsonl
{"prompt": "How do I optimize React rendering?", "rewards": {"mistralai/mixtral-8x7b-instruct": 0.85, "openai/gpt-4-turbo": 0.95}}
{"prompt": "Explain async/await in Python", "rewards": {"mistralai/mixtral-8x7b-instruct": 0.90, "openai/gpt-4-turbo": 0.92}}
```

## Requirements

- Python 3.10+
- sentence-transformers
- joblib
- numpy
- matplotlib (for find_gamma.py)
- tqdm

Install:
```bash
pip install sentence-transformers joblib numpy matplotlib tqdm
```

## Using Calibrated Routers

```python
from bandit_gpt.calibration import CalibratedRouter
from sentence_transformers import SentenceTransformer
import joblib

# Load resources
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pca_model = joblib.load("artifacts/pca_23_routellm.joblib")

# Load calibrated router
router = CalibratedRouter.load("my_router.joblib", encoder, pca_model)

# Route a query
model = router.select_model("Explain quantum computing")
print(f"Selected: {model}")

# Update after observing reward (online learning)
router.update("Explain quantum computing", reward=0.95)

# Save updated router
router.save("my_router_updated.joblib")
```

## Documentation

- **Library API**: See `src/bandit_gpt/calibration.py` for detailed docstrings
- **Research artifacts**: See `experiments_v1/calibration/` for paper documentation
- **Complete guide**: See `experiments_v1/calibration/README.md` for detailed workflow

## File Structure

```
scripts/calibration/
├── README.md                 # This file
├── find_gamma.py             # CLI: Find optimal gamma
└── calibrate_router.py       # CLI: Calibrate router

src/bandit_gpt/
└── calibration.py            # Library: CalibratedRouter, helpers

experiments_v1/calibration/
├── README.md                 # Complete calibration guide
├── FINAL_RESULTS_SUMMARY.md  # KDD paper results
└── results/                  # Experimental outputs
```

## Examples

### Quick Start (Default Paths)

```bash
# From project root
cd scripts/calibration/

# Find gamma
python3 find_gamma.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl

# Calibrate with recommended gamma
python3 calibrate_router.py \
  --calibration-data ../../data/routellm/data/canonical_dev_calibration.jsonl \
  --gamma 0.010 \
  --output ../../results/my_router.joblib
```

### Custom Data

```bash
# Use your own calibration data
python3 find_gamma.py \
  --calibration-data /path/to/my_data.jsonl \
  --warmup-priors /path/to/priors.joblib \
  --pca /path/to/pca.joblib \
  --output my_results/

python3 calibrate_router.py \
  --calibration-data /path/to/my_data.jsonl \
  --warmup-priors /path/to/priors.joblib \
  --pca /path/to/pca.joblib \
  --gamma 0.005 \
  --output my_router.joblib
```

## Support

For issues or questions:
1. Check the complete guide: `experiments_v1/calibration/README.md`
2. Review library docstrings: `src/bandit_gpt/calibration.py`
3. Open an issue on GitHub

