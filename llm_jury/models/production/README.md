# XGBoost Quality Prediction Models

## Overview

This directory contains production-ready XGBoost models that predict **P(success | prompt, model)** - the probability a specific LLM will successfully complete a specific prompt.

### The Key Insight

Benchmark scores (like LiveCodeBench = 38.1%) are **aggregate success rates**:
- A model with 38.1% LiveCodeBench score passed 38.1% of coding problems
- Our XGBoost predicts success probability **per problem**
- Averaged across problems, predictions should match the aggregate score

### Supported Intents

| Intent | Benchmark | What it measures |
|--------|-----------|------------------|
| **Reasoning** | GPQA Diamond | Graduate-level scientific reasoning accuracy |
| **Coding** | LiveCodeBench | Code generation pass rate (real execution) |
| **Summarization** | SummEdits | Factual consistency accuracy |
| **RAG** | MMLU-Pro | Knowledge retrieval accuracy |

## Files

Each intent has two files:

- `{intent}_xgboost_model.joblib` - Trained XGBoost model
- `{intent}_model_card.json` - Model metadata and feature specs

## Usage

### 1. Load Model

```python
import joblib
import json

# Load model
model = joblib.load('reasoning_xgboost_model.joblib')

# Load metadata
with open('reasoning_model_card.json') as f:
    metadata = json.load(f)
    
feature_names = metadata['feature_names']
```

### 2. Prepare Features

```python
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Get NVIDIA prompt features
nvidia_model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/prompt-task-and-complexity-classifier"
)
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/prompt-task-and-complexity-classifier"
)

def get_nvidia_features(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = nvidia_model(**inputs)
    # Extract features (implementation depends on NVIDIA model output)
    return {
        'nvidia_creativity': ...,
        'nvidia_reasoning': ...,
        'nvidia_constraint': ...,
        'nvidia_domain_knowledge': ...,
        'nvidia_contextual_knowledge': ...,
        'nvidia_few_shots': ...
    }

# Get model capability score
# For reasoning, coding, summarization: use model's aggregate score
# For RAG: use MMLU-Pro score from Artificial Analysis
model_capability = get_model_capability_score(model_name, intent)

# Combine features
features = [
    nvidia_features['nvidia_creativity'],
    nvidia_features['nvidia_reasoning'],
    nvidia_features['nvidia_constraint'],
    nvidia_features['nvidia_domain_knowledge'],
    nvidia_features['nvidia_contextual_knowledge'],
    nvidia_features['nvidia_few_shots'],
    model_capability
]
```

### 3. Make Predictions

```python
# Predict success probability
X = np.array([features])
success_probability = model.predict_proba(X)[0][1]

print(f"Success probability: {success_probability:.1%}")

# Routing decision
if success_probability > 0.7:
    route_to = 'primary_model'
elif success_probability > 0.5:
    route_to = 'secondary_model'
else:
    route_to = 'fallback_model'
```

## Model Performance

See individual model cards for detailed performance metrics including:

- Training accuracy and AUC
- Cross-validation results
- Feature importance
- Validation on proprietary models

## Feature Requirements

All models require 7 features:

1. **nvidia_creativity** (float 0-1): Creative scope
2. **nvidia_reasoning** (float 0-1): Reasoning complexity
3. **nvidia_constraint** (int): Number of constraints
4. **nvidia_domain_knowledge** (float 0-1): Domain expertise required
5. **nvidia_contextual_knowledge** (float 0-1): Context needed
6. **nvidia_few_shots** (int): Number of few-shot examples
7. **model_capability** (float 0-100): Model's capability score

### Capability Proxies by Intent

Each intent uses a specific benchmark as the capability proxy. These were selected for high uniqueness across models (enabling differentiation) and direct relevance to the task:

| Intent | Capability Field | Benchmark | Rationale |
|--------|-----------------|-----------|-----------|
| **Reasoning** | `gpqa` | GPQA Diamond | Graduate-level scientific reasoning (80/81 unique values) |
| **Coding** | `livecodebench` | LiveCodeBench | Continuously updated coding benchmark (80/81 unique values) |
| **Summarization** | `summedits_score` | SummEdits | Factual consistency in summaries (77/81 unique values) |
| **RAG** | `mmlu_pro` | MMLU-Pro | Knowledge retrieval and application (80/81 unique values) |

**Note**: We previously used `humaneval_score` for coding and `ifeval` for summarization, but these had limited differentiation across models (9/81 unique values for HumanEval). The new fields provide much better model differentiation.

## Production Deployment

### Real-Time Routing

```python
class LLMRouter:
    def __init__(self):
        self.models = {
            'reasoning': joblib.load('reasoning_xgboost_model.joblib'),
            'coding': joblib.load('coding_xgboost_model.joblib'),
            'summarization': joblib.load('summarization_xgboost_model.joblib'),
            'rag': joblib.load('rag_xgboost_model.joblib')
        }
        self.nvidia_classifier = load_nvidia_classifier()
        
    def route_request(self, prompt, intent, available_models):
        # Get prompt features
        nvidia_features = self.nvidia_classifier(prompt)
        
        # Score each available model
        scores = {}
        for model_name in available_models:
            capability = self.get_capability(model_name, intent)
            features = [...nvidia_features, capability]
            success_prob = self.models[intent].predict_proba([features])[0][1]
            scores[model_name] = success_prob
        
        # Route to best model
        best_model = max(scores, key=scores.get)
        confidence = scores[best_model]
        
        return best_model, confidence
```

## Training Methodology

### Data Sources (ALL REAL, NO SYNTHETIC)

| Intent | Training Data Source | Labels |
|--------|---------------------|--------|
| **Coding** | LiveCodeBench leaderboard | Real pass@1 from code execution |
| **Reasoning** | OpenCompass GPQA predictions | Real accuracy (answer matching) |
| **Summarization** | OpenCompass IFEval predictions | Real compliance scores |
| **RAG** | OpenCompass TriviaQA predictions | Real answer matching |

### How Training Works

```
Instance-level training data:
  (prompt_1, model_A, success=1)  ← Model A passed this problem
  (prompt_1, model_B, success=0)  ← Model B failed this problem
  (prompt_2, model_A, success=1)
  ...

Features per instance:
  - nvidia_creativity, reasoning, etc. (prompt complexity)
  - model_capability (benchmark score, e.g., 38.1%)

XGBoost learns:
  P(success) = f(prompt_complexity, model_capability)
```

### Zero-Shot Transfer: How We Predict for ALL Models

The XGBoost model learns **relationships**, not model-specific weights:

1. **Training**: 28 models from LiveCodeBench leaderboard (for coding)
2. **Capability feature**: Uses model's benchmark score as input
3. **Generalization**: Any model with a benchmark score can get predictions

```python
# Same model works for ANY model with a capability score
# Trained on: DeepSeek-V3, GPT-4o, Claude 3.5, etc.
# Can predict for: Mistral Large (not in training) using its livecodebench score
features = [prompt_complexity..., model_capability=42.3]
prediction = xgboost_model.predict(features)
```

### Why This Works

The `model_capability` feature importance is **31.3%** for coding - proving the model learned that capability scores are predictive of success. A model with 80% benchmark score will get higher predictions than one with 30%, regardless of whether either was in training.

## Training Results (December 2024)

| Intent | Train Size | Test Size | CV AUC | Test AUC | Capability Importance |
|--------|------------|-----------|--------|----------|----------------------|
| **Reasoning** | 4,950 | 990 | 0.733 | 0.786 | 12.0% |
| **Coding** | 6,400 | 2,000 | 0.875 | 0.695 | 31.3% |
| **Summarization** | 7,800 | 2,400 | 0.963 | 0.895 | 23.5% |
| **RAG** | 40,000 | 20,000 | 0.809 | 0.833 | 37.8% |

### Zero-Shot Transfer Validation

Predictions correlate with actual benchmark performance for models NOT in training:

| Intent | Pearson r | Spearman ρ | p-value | Status |
|--------|-----------|------------|---------|--------|
| **Coding** | 0.942 | 0.934 | <0.001 | ✅ Excellent |
| **Reasoning** | 0.993 | 0.953 | <0.001 | ✅ Excellent |
| **RAG** | 0.957 | 0.924 | <0.001 | ✅ Excellent |
| **Summarization** | 0.773 | 0.690 | <0.001 | ✅ Good |
| **Average** | **0.916** | **0.875** | - | ✅ Excellent |

## Key Files

- `training_summary.json` - Training metrics for all intents
- `{intent}_xgboost_model.joblib` - Trained model
- `{intent}_model_card.json` - Feature specs and metadata

For detailed methodology, see `KDD/data/documentation/methodology/`.
