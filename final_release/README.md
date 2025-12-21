# Bandit Router: Final Release

This folder contains the production-grade implementation of the Bandit Router (BanditGPT) as described in the KDD paper. It is designed to be a standalone, portable package.

## 1. Installation

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Quick Start

To test the router with default settings (HLE priors and the 80-model registry):

```bash
python test_router_simple.py
```

## 3. Core Components

- **`bandit.py`**: The primary entry point. Contains the `BanditRouter` class and multi-objective optimization logic.
- **`quality_predictor.py`**: The neural quality grader trained on HelpSteer2 and LMSYS Arena.
- **`models.json`**: The model registry containing quality (HLE), cost, and latency metadata for 80+ models.
- **`data/`**: Contains the pre-computed priors (`priors_meta_large.npz`) and evaluation datasets.

## 4. Advanced Usage

### Specifying Benchmarks
You can warm-start the router using different benchmarks from the registry:

```python
from bandit import BanditRouter

router = BanditRouter.create(
    priors="benchmark",
    benchmark_key="mmlu_pro" # or "hle", "math_500", etc.
)
```

### Persistence
Save and load the bandit's learned state:

```python
router.save_state("my_bandit_state.npz")
router = BanditRouter.create(state_path="my_bandit_state.npz")
```

## 5. Evaluation & Significance

To re-run the 5-fold validation, statistical significance tests, and adaptation curve:

```bash
python plot_regret.py
python validate_significance.py
python plot_adaptation.py
```

## 6. Data Preparation (Optional)

If you wish to re-generate the registry or priors from source data:

1. Update `data/models_cache_with_hle.json` or `data/lmsys_all_prompts.jsonl`.
2. Run `python setup_data.py` to update `models.json`.
3. Run `python calc_priors_large.py` to update `data/priors_meta_large.npz`.
