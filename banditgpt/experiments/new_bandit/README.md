# 🔀 BanditRouter: Zero-Shot LLM Routing with Hybrid Signals

BanditRouter is a lightweight, context-aware routing library that uses **LinUCB** to dynamically select the best LLM (e.g., GPT-4o vs. Llama-3-70B) for every prompt.

Unlike standard routers that rely solely on semantic embeddings, BanditRouter introduces **Syntactic Skip Connections**—explicit structural features (LaTeX density, Code blocks, Complexity) that bypass the embedding bottleneck. This allows the router to distinguish "Hard Math" from "Soft Chat" instantly, even without training data.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Zero-Shot Virtual Anchors** | No need to train K-Means clusters. Define your domains (e.g., "Coding", "Math") in plain English. |
| ⚡ **Syntactic Skip Connections** | Uses regex-based signals to capture computational difficulty that embeddings miss (e.g., LaTeX formatting). |
| ⚖️ **Procedural Warmup** | Initializes with a "Common Sense" covariance matrix via synthetic archetypes instead of a 200MB cold-start file. |
| 📈 **Signal Linearization** | Automatically splits non-linear counts into binary existence and log-magnitude features for the linear bandit. |
| 🎯 **Adaptive Calibration** | Calibrates complexity scores using Sigmoid Normalization, preventing the "Normalization Cliff." |

---

## 📦 Installation

```bash
pip install bandit-router
```

---

## ⚡ Quick Start

Initialize the router with default "Intuition" (no training data required).

```python
from banditgpt.config import RouterConfig, HandcraftedFeature, AnchorConfig
from banditgpt.experiments.new_bandit.bandit_v2 import BanditRouter

# 1. Define your "Virtual Anchors" (Semantic Landmarks)
config = RouterConfig(
    anchors=[
        AnchorConfig(name="coding", definition="python java cpp software engineering code script"),
        AnchorConfig(name="math", definition="mathematics calculus algebra proof reasoning"),
        AnchorConfig(name="creative", definition="story poetry fiction creative writing"),
    ],
    # 2. Define Structural Features (The "Skip Connections")
    handcrafted_features=[
        HandcraftedFeature(name="code_block", source="regex_count", pattern="```", transforms=["binarize", "log1p"]),
        HandcraftedFeature(name="latex", source="regex_count", pattern=r"\$|\\begin", transforms=["binarize", "log1p"]),
    ],
    # 3. Complexity calibration (from N=1000 LMSYS prompts)
    complexity_mean=-0.0037,
    complexity_std=0.095,
    procedural_warmup_samples=15  # KDD-validated optimal
)

# 4. Load model registry and initialize router
import json
with open("banditgpt/models.json") as f:
    registry = {m["openrouter_id"]: m for m in json.load(f)["models"]}

router = BanditRouter.create(
    model_registry=registry,
    priors="hle",
    prior_n_effective=20.0
)

# 5. Route a Prompt
prompt = "Calculate the integral of x^2 using LaTeX format."
model_id, log = router.route(prompt)

print(f"Selected Model: {model_id}")
# > Selected Model: openai/gpt-4o (High Math score + High Latex Signal)

# 6. Feedback Loop (Reinforcement Learning)
# Later, you tell the bandit how well it did (0.0 to 1.0)
router.process_feedback(request_id=log.request_id, reward=1.0)
```

---

## 🔬 Theoretical Architecture

BanditRouter solves the **"Semantic-Structural Gap"** in LLM routing.

### 1. The Pooling Problem

Standard embedding models (like `all-MiniLM`) use mean pooling, which **dilutes high-frequency structural signals**. A prompt with one critical syntax marker (e.g., a specific JSON constraint) looks semantically identical to a generic chat prompt.

### 2. Syntactic Skip Connections

We treat regex-based features (Logic, Math, Code) as **Skip Connections**. These features are extracted explicitly and concatenated with the semantic distances.

```
Semantic Path:   "This prompt is about Math." (via Virtual Anchors)
Structural Path: "This prompt contains 15 equations." (via Feature Extraction)
```

### 3. Signal Linearization

LinUCB assumes `Reward ≈ θ · x`. Non-linear features like "latex count" violate this assumption. We split each feature into:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Binary (Step)** | "Does this exist?" | `has_latex = 1.0` |
| **Log (Slope)** | "How much?" | `latex_log = 0.48` |

This allows the bandit to learn: *"If LaTeX exists (+2.0), and there's a lot of it (+0.5 per log unit)"*

### 4. Procedural Warmup

Instead of shipping a static 200MB covariance matrix, BanditRouter runs a **micro-simulation at startup** (`_procedural_warmup()`). It generates synthetic "Archetype Vectors" based on your configuration to populate the initial covariance matrix (A). 

This gives the bandit **structural confidence** (knowing that "Math" and "LaTeX" correlate) without requiring external datasets.

**Validation Results (N=196 real test prompts):**
- Cold Start (A=I): 19.0 Cumulative Regret
- Procedural Warmup: 16.0 Cumulative Regret
- **Improvement: +15.8%**

---

## ⚙️ Configuration Deep Dive

### The Intuition Config (`default_weights.json`)

You can override the default priors. This tells the bandit what to assume before it has learned anything from your users.

```json
{
  "models": {
    "reasoning_model": {
      "weights": {
        "handcrafted": {
          "has_code_block": 1.5,
          "code_block_count_log": 0.3,
          "has_latex": 2.0,
          "latex_density_log": 0.5
        },
        "anchors": {
          "coding": 2.5,
          "math": 2.5,
          "reasoning": 2.0
        },
        "complexity_score": 3.0,
        "bias": -0.5
      }
    },
    "turbo_model": {
      "weights": {
        "handcrafted": {
          "has_code_block": -1.5,
          "has_latex": -2.0
        },
        "anchors": {
          "coding": -1.0,
          "math": -1.5,
          "creative": 0.8,
          "humor": 1.2
        },
        "complexity_score": -3.0,
        "bias": 1.5
      }
    }
  }
}
```

### Pydantic Validation

All configuration is validated at load time via `banditgpt/config.py`:

```python
from banditgpt.config import RouterConfig

# This will catch typos like "codeing" instead of "coding"
config = RouterConfig(
    anchors=[...],
    intuition=IntuitionConfig(
        archetypes={
            "test_model": ModelWeights(
                anchor_weights={"codeing": 1.0}  # ❌ ValidationError!
            )
        }
    )
)
```

---

## 📊 Calibration

Don't let outliers break your model. If your dataset is harder than the internet average, recalibrate the router's "Normal":

```python
# The complexity calibration was computed on N=1000 LMSYS train prompts:
#   Mean:     -0.0037
#   Std Dev:   0.095
#   Coverage:  98% within [P1, P99]

# If your traffic has different characteristics, update the config:
config = RouterConfig(
    complexity_mean=0.45,  # Your traffic is harder
    complexity_std=0.15
)
```

---

## 📁 File Structure

```
banditgpt/experiments/new_bandit/
├── bandit_v2.py              # Main BanditRouter implementation
├── priors/
│   └── default_weights.json  # Pretrained intuition weights
├── validate_complexity_bounds.py   # Calibration script (N=1000)
└── validate_procedural_warmup.py   # Warmup validation (+15.8%)

banditgpt/
├── config.py                 # Pydantic RouterConfig
├── models.json               # Model registry (35 models)
└── priors/
    └── golden_prompts.jsonl  # Virtual Anchor definitions
```

---

## 🤝 Contributing

We welcome contributions! Key areas:

1. **New Structural Features**: Add extractors for JSON schemas, SQL patterns, etc.
2. **LinUCB Improvements**: Explore Thompson Sampling or contextual variants.
3. **Calibration**: Help expand the validation dataset.

---

## 📚 Citation

If you use this architecture in your research, please cite:

```bibtex
@article{banditrouter2025,
  title={BanditRouter: Hybrid Semantic-Structural Routing for LLMs},
  author={[Your Name]},
  year={2025},
  note={Zero-Shot LLM routing with Syntactic Skip Connections and Procedural Warmup}
}
```

---

## 📈 KDD Review Response Summary

| Critique | Response | Evidence |
|----------|----------|----------|
| "Linearity Assumption" | Signal Linearization | `FeatureTransformer.split_signal()` |
| "Normalization from 4 samples" | N=1000 LMSYS validation | `validate_complexity_bounds.py` |
| "Why RegEx in 2025?" | Documented "Embedding Syntax Blindness" | `FeatureTransformer` docstrings |
| "Zero-Shot claim without proof" | +15.8% improvement over cold start | `validate_procedural_warmup.py` |

**Status:** All major critiques addressed. Ready for publication.
