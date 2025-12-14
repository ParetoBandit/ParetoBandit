# XGBoost Model Usage Guide

## Overview

This directory contains production-ready XGBoost models for LLM routing across 4 task intents:

1. **Reasoning** - Graduate-level reasoning tasks (GPQA)
2. **Coding** - Python function completion (HumanEval)
3. **Summarization** - Instruction following (IFEval)
4. **RAG** - Factual question answering (TriviaQA)

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

- **Reasoning**: Model's GPQA aggregate score
- **Coding**: Model's HumanEval aggregate score
- **Summarization**: Model's IFEval aggregate score
- **RAG**: Model's MMLU-Pro score (external benchmark)

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

## Notes

- Models trained on 133,394 labeled examples from 42 models
- Validated on 14,304 proprietary examples (GPT-4o, Claude, Gemini)
- Average zero-shot transfer correlation: r=0.564 (all p<0.0001)
- Ready for production deployment

For questions or issues, see documentation in the KDD/data/ directory.
