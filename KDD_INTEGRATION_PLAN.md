# KDD/data Integration into llm_jury Library

**Date**: December 13, 2024  
**Goal**: Make KDD/data production code the core prediction engine for llm_jury

---

## 🎯 Vision

```
User Prompt → Intent Classification (KDD models) → Success Prediction (KDD models) → Optimization (llm_jury) → Model Recommendation
              └────────────────────────────────────┘   └──────────────────────────────┘
                    NEW: Use KDD/data code                   EXISTING: Keep as-is
```

**Before**: llm_jury had its own intent classifier (pattern-based)  
**After**: llm_jury uses KDD/data production models (NVIDIA features + capability proxies)

---

## 📦 What We're Integrating

From `KDD/data/`:

### Core Scripts
1. **`core_scripts/opencompass_name_mappings.py`** → Model name resolution
2. **`core_scripts/build_instance_level_training_data.py`** → Data collection (for retraining)
3. **`core_scripts/train_final_xgboost_models.py`** → Model training (for updates)

### Production Models
4. **`production_models/*.joblib`** → 4 trained XGBoost models
5. **`production_models/*_model_card.json`** → Model metadata

### Validation Scripts
6. **`validation/validate_all_4_intents.py`** → Validation pipeline
7. **`validation/validate_rag_with_mmlu_pro.py`** → RAG-specific validation

---

## 🏗️ Integration Architecture

### New Module Structure

```
llm_jury/
├── prediction/                     # NEW MODULE (wraps KDD/data)
│   ├── __init__.py
│   ├── intent_predictor.py        # Main interface
│   ├── model_loader.py             # Load production models
│   ├── feature_extractor.py        # NVIDIA features + capability
│   ├── name_resolver.py            # Wraps opencompass_name_mappings
│   └── validator.py                # Wraps validation scripts
│
├── routing/
│   ├── kdd_router.py               # NEW: Router using KDD models
│   └── [existing files...]         # KEEP: Legacy for comparison
│
└── [existing modules...]           # KEEP: optimization, ranking, etc.
```

---

## 🔧 Implementation Steps

### Phase 1: Create Wrapper Module (30 min)

**Create `llm_jury/prediction/__init__.py`:**

```python
"""
Production intent prediction using KDD/data models.

This module wraps the validated production models from KDD/data/
for use in the llm_jury routing pipeline.
"""

from .intent_predictor import IntentPredictor
from .model_loader import load_model, load_all_models
from .feature_extractor import extract_features

__all__ = [
    'IntentPredictor',
    'load_model',
    'load_all_models',
    'extract_features',
]
```

---

### Phase 2: Create Model Loader (20 min)

**Create `llm_jury/prediction/model_loader.py`:**

```python
"""
Load production XGBoost models from KDD/data/production_models/.
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import xgboost as xgb


def get_models_dir() -> Path:
    """Get path to KDD/data/production_models directory."""
    # From llm_jury/prediction/ → ../../KDD/data/production_models/
    current_dir = Path(__file__).parent
    models_dir = current_dir.parent.parent / 'KDD' / 'data' / 'production_models'
    
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Production models directory not found: {models_dir}\n"
            "Expected structure: llm_jury/KDD/data/production_models/"
        )
    
    return models_dir


def load_model(intent: str) -> Tuple[xgb.XGBClassifier, dict]:
    """
    Load a trained XGBoost model and its metadata.
    
    Args:
        intent: One of 'reasoning', 'coding', 'summarization', 'rag'
    
    Returns:
        Tuple of (model, model_card)
    
    Raises:
        ValueError: If intent is invalid
        FileNotFoundError: If model files not found
    """
    valid_intents = ['reasoning', 'coding', 'summarization', 'rag']
    if intent not in valid_intents:
        raise ValueError(f"Invalid intent: {intent}. Must be one of {valid_intents}")
    
    models_dir = get_models_dir()
    
    # Load model
    model_path = models_dir / f'{intent}_xgboost_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load model card
    card_path = models_dir / f'{intent}_model_card.json'
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {card_path}")
    
    with open(card_path) as f:
        model_card = json.load(f)
    
    return model, model_card


def load_all_models() -> Dict[str, Tuple[xgb.XGBClassifier, dict]]:
    """
    Load all 4 production models.
    
    Returns:
        Dictionary mapping intent -> (model, model_card)
    """
    intents = ['reasoning', 'coding', 'summarization', 'rag']
    
    models = {}
    for intent in intents:
        models[intent] = load_model(intent)
    
    return models


def get_model_info(intent: str) -> dict:
    """Get model card without loading the model."""
    models_dir = get_models_dir()
    card_path = models_dir / f'{intent}_model_card.json'
    
    with open(card_path) as f:
        return json.load(f)


def get_all_model_info() -> Dict[str, dict]:
    """Get model cards for all intents."""
    intents = ['reasoning', 'coding', 'summarization', 'rag']
    return {intent: get_model_info(intent) for intent in intents}
```

---

### Phase 3: Create Name Resolver (15 min)

**Create `llm_jury/prediction/name_resolver.py`:**

```python
"""
Model name resolution using KDD/data mappings.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

# Import the production mappings from KDD/data
_kdd_data_path = Path(__file__).parent.parent.parent / 'KDD' / 'data' / 'core_scripts'
sys.path.insert(0, str(_kdd_data_path))

try:
    from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
except ImportError as e:
    raise ImportError(
        f"Could not import opencompass_name_mappings from {_kdd_data_path}\n"
        "Ensure KDD/data/core_scripts/opencompass_name_mappings.py exists."
    ) from e


class ModelNameResolver:
    """Resolve model names between different systems."""
    
    def __init__(self):
        """Initialize with production mappings."""
        self.mappings = OPENCOMPASS_TO_CACHE
    
    def resolve(self, model_name: str) -> Optional[str]:
        """
        Resolve OpenCompass model name to cache name.
        
        Args:
            model_name: OpenCompass model name
        
        Returns:
            Cache name if found, else original name
        """
        return self.mappings.get(model_name, model_name)
    
    def resolve_batch(self, model_names: list) -> Dict[str, str]:
        """Resolve multiple model names."""
        return {name: self.resolve(name) for name in model_names}
    
    def is_known(self, model_name: str) -> bool:
        """Check if model name is in mappings."""
        return model_name in self.mappings
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all model name mappings."""
        return self.mappings.copy()


# Singleton instance
_resolver = None

def get_resolver() -> ModelNameResolver:
    """Get or create the global name resolver."""
    global _resolver
    if _resolver is None:
        _resolver = ModelNameResolver()
    return _resolver


def resolve_name(model_name: str) -> str:
    """Convenience function to resolve a model name."""
    return get_resolver().resolve(model_name)
```

---

### Phase 4: Create Feature Extractor (30 min)

**Create `llm_jury/prediction/feature_extractor.py`:**

```python
"""
Extract features for intent prediction using NVIDIA classifier + capability proxies.
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PromptFeatures:
    """Features extracted from a prompt."""
    nvidia_creativity: float
    nvidia_reasoning: float
    nvidia_constraint: int
    nvidia_domain_knowledge: float
    nvidia_contextual_knowledge: float
    nvidia_few_shots: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'nvidia_creativity': self.nvidia_creativity,
            'nvidia_reasoning': self.nvidia_reasoning,
            'nvidia_constraint': self.nvidia_constraint,
            'nvidia_domain_knowledge': self.nvidia_domain_knowledge,
            'nvidia_contextual_knowledge': self.nvidia_contextual_knowledge,
            'nvidia_few_shots': self.nvidia_few_shots,
        }


class NVIDIAFeatureExtractor:
    """
    Extract NVIDIA prompt complexity features.
    
    Uses the NVIDIA Prompt Task and Complexity Classifier API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            api_key: NVIDIA API key (optional, reads from env if not provided)
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        self.api_key = api_key or os.getenv('NVIDIA_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not found. Set NVIDIA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"
    
    def extract(self, prompt: str) -> PromptFeatures:
        """
        Extract NVIDIA features from a prompt.
        
        Args:
            prompt: The prompt to analyze
        
        Returns:
            PromptFeatures with all NVIDIA complexity scores
        """
        # TODO: Implement actual NVIDIA API call
        # For now, return placeholder (implement based on KDD/data approach)
        
        # This should match the logic from build_instance_level_training_data.py
        # which calls the NVIDIA classifier API
        
        raise NotImplementedError(
            "NVIDIA API integration pending. "
            "See KDD/data/core_scripts/build_instance_level_training_data.py "
            "for reference implementation."
        )


class CapabilityExtractor:
    """
    Extract model capability scores for use as features.
    
    Reads from models_cache.json.
    """
    
    def __init__(self):
        """Initialize capability extractor."""
        self.cache_path = self._find_cache()
        self.capabilities = self._load_capabilities()
    
    def _find_cache(self) -> Path:
        """Find models_cache.json."""
        # Try multiple locations
        candidates = [
            Path(__file__).parent.parent.parent / 'data' / 'models_cache.json',
            Path(__file__).parent.parent.parent / 'KDD' / 'data' / 'models_cache.json',
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"models_cache.json not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in candidates)
        )
    
    def _load_capabilities(self) -> Dict[str, Dict[str, float]]:
        """Load capability scores from cache."""
        with open(self.cache_path) as f:
            cache_data = json.load(f)
        
        capabilities = {}
        for model in cache_data['models']:
            name = model['name']
            capabilities[name] = {}
            
            # Extract relevant benchmarks
            for benchmark in ['mmlu_pro', 'gpqa', 'humaneval_plus', 'ifeval']:
                value = model.get(benchmark)
                if value and value != 'N/A':
                    capabilities[name][benchmark] = float(value)
        
        return capabilities
    
    def get_capability(self, model_name: str, benchmark: str) -> Optional[float]:
        """Get a specific capability score."""
        return self.capabilities.get(model_name, {}).get(benchmark)
    
    def get_intent_capability(self, model_name: str, intent: str) -> Optional[float]:
        """
        Get the appropriate capability proxy for an intent.
        
        Args:
            model_name: Model name (cache format)
            intent: One of 'reasoning', 'coding', 'summarization', 'rag'
        
        Returns:
            Capability score or None if not available
        """
        benchmark_map = {
            'reasoning': 'gpqa',
            'coding': 'humaneval_plus',
            'summarization': 'ifeval',
            'rag': 'mmlu_pro',
        }
        
        benchmark = benchmark_map.get(intent)
        if not benchmark:
            raise ValueError(f"Unknown intent: {intent}")
        
        return self.get_capability(model_name, benchmark)


class FeatureExtractor:
    """
    Combined feature extractor for intent prediction.
    
    Extracts both NVIDIA prompt features and model capability features.
    """
    
    def __init__(self, nvidia_api_key: Optional[str] = None):
        """Initialize feature extractor."""
        # self.nvidia_extractor = NVIDIAFeatureExtractor(nvidia_api_key)
        self.capability_extractor = CapabilityExtractor()
    
    def extract_prompt_features(self, prompt: str) -> PromptFeatures:
        """Extract NVIDIA features from prompt."""
        # return self.nvidia_extractor.extract(prompt)
        raise NotImplementedError("NVIDIA API integration pending")
    
    def extract_model_capability(self, model_name: str, intent: str) -> float:
        """Extract model capability for intent."""
        capability = self.capability_extractor.get_intent_capability(model_name, intent)
        
        if capability is None:
            raise ValueError(
                f"No capability data for model '{model_name}' on intent '{intent}'"
            )
        
        return capability
    
    def prepare_features(
        self,
        prompt: str,
        model_name: str,
        intent: str
    ) -> np.ndarray:
        """
        Prepare feature vector for prediction.
        
        Args:
            prompt: User prompt
            model_name: Model name (cache format)
            intent: Intent category
        
        Returns:
            Feature vector as numpy array (7 features)
        """
        # Get NVIDIA features
        nvidia_features = self.extract_prompt_features(prompt)
        
        # Get capability score
        capability = self.extract_model_capability(model_name, intent)
        
        # Combine into feature vector (order matches training)
        features = np.array([
            nvidia_features.nvidia_creativity,
            nvidia_features.nvidia_reasoning,
            nvidia_features.nvidia_constraint,
            nvidia_features.nvidia_domain_knowledge,
            nvidia_features.nvidia_contextual_knowledge,
            nvidia_features.nvidia_few_shots,
            capability,  # model_capability (intent-specific)
        ])
        
        return features.reshape(1, -1)  # Shape: (1, 7)


# Convenience function
def extract_features(prompt: str, model_name: str, intent: str) -> np.ndarray:
    """Extract features for a prompt-model-intent combination."""
    extractor = FeatureExtractor()
    return extractor.prepare_features(prompt, model_name, intent)
```

---

### Phase 5: Create Main Predictor Interface (40 min)

**Create `llm_jury/prediction/intent_predictor.py`:**

```python
"""
Main interface for intent prediction using KDD/data production models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .model_loader import load_model, load_all_models, get_model_info
from .feature_extractor import FeatureExtractor
from .name_resolver import resolve_name


@dataclass
class PredictionResult:
    """Result from intent prediction."""
    intent: str
    model_name: str
    success_probability: float
    confidence: float
    should_use: bool
    threshold_used: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'intent': self.intent,
            'model': self.model_name,
            'success_probability': self.success_probability,
            'confidence': self.confidence,
            'should_use': self.should_use,
            'threshold': self.threshold_used,
        }


class IntentPredictor:
    """
    Production intent predictor using KDD/data models.
    
    This is the main interface for predicting whether a model will succeed
    on a given prompt for a specific intent.
    """
    
    def __init__(
        self,
        intents: Optional[List[str]] = None,
        threshold: float = 0.5,
        load_immediately: bool = True,
    ):
        """
        Initialize intent predictor.
        
        Args:
            intents: List of intents to load (default: all 4)
            threshold: Success probability threshold (default: 0.5)
            load_immediately: Load models on init (default: True)
        """
        self.intents = intents or ['reasoning', 'coding', 'summarization', 'rag']
        self.threshold = threshold
        self.models = {}
        self.model_cards = {}
        self.feature_extractor = FeatureExtractor()
        
        if load_immediately:
            self.load_models()
    
    def load_models(self):
        """Load all configured models."""
        print(f"Loading {len(self.intents)} production models...")
        
        for intent in self.intents:
            model, card = load_model(intent)
            self.models[intent] = model
            self.model_cards[intent] = card
            
            print(f"  ✓ {intent}: {card['n_train_examples']:,} training examples, "
                  f"test AUC = {card['test_auc']:.3f}")
        
        print(f"✓ All models loaded")
    
    def predict(
        self,
        prompt: str,
        model_name: str,
        intent: str,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """
        Predict success probability for a prompt-model-intent combination.
        
        Args:
            prompt: User prompt
            model_name: Model name (will be resolved if needed)
            intent: One of 'reasoning', 'coding', 'summarization', 'rag'
            threshold: Custom threshold (optional, uses default if None)
        
        Returns:
            PredictionResult with success probability and recommendation
        """
        if intent not in self.models:
            raise ValueError(
                f"Intent '{intent}' not loaded. Available: {list(self.models.keys())}"
            )
        
        # Resolve model name
        resolved_name = resolve_name(model_name)
        
        # Extract features
        features = self.feature_extractor.prepare_features(
            prompt, resolved_name, intent
        )
        
        # Predict
        model = self.models[intent]
        proba = model.predict_proba(features)[0]
        success_prob = proba[1]  # Probability of success (class 1)
        confidence = max(proba)  # Confidence is max probability
        
        # Apply threshold
        threshold_used = threshold if threshold is not None else self.threshold
        should_use = success_prob >= threshold_used
        
        return PredictionResult(
            intent=intent,
            model_name=resolved_name,
            success_probability=float(success_prob),
            confidence=float(confidence),
            should_use=should_use,
            threshold_used=threshold_used,
        )
    
    def predict_all_intents(
        self,
        prompt: str,
        model_name: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, PredictionResult]:
        """
        Predict success probability for all intents.
        
        Args:
            prompt: User prompt
            model_name: Model name
            threshold: Custom threshold (optional)
        
        Returns:
            Dictionary mapping intent -> PredictionResult
        """
        results = {}
        
        for intent in self.intents:
            results[intent] = self.predict(prompt, model_name, intent, threshold)
        
        return results
    
    def recommend_best_intent(
        self,
        prompt: str,
        model_name: str,
    ) -> Tuple[str, PredictionResult]:
        """
        Recommend the best intent for a prompt-model pair.
        
        Args:
            prompt: User prompt
            model_name: Model name
        
        Returns:
            Tuple of (best_intent, prediction_result)
        """
        results = self.predict_all_intents(prompt, model_name)
        
        # Find intent with highest success probability
        best_intent = max(results.keys(), key=lambda i: results[i].success_probability)
        
        return best_intent, results[best_intent]
    
    def batch_predict(
        self,
        prompts: List[str],
        model_name: str,
        intent: str,
        threshold: Optional[float] = None,
    ) -> List[PredictionResult]:
        """Predict for multiple prompts."""
        return [
            self.predict(prompt, model_name, intent, threshold)
            for prompt in prompts
        ]
    
    def get_model_info(self, intent: str) -> dict:
        """Get model card for an intent."""
        if intent not in self.model_cards:
            return get_model_info(intent)
        return self.model_cards[intent]


# Convenience function
def predict_success(
    prompt: str,
    model_name: str,
    intent: str,
    threshold: float = 0.5,
) -> PredictionResult:
    """
    Quick prediction function.
    
    Creates a predictor, loads the model, and makes a prediction.
    For repeated predictions, create an IntentPredictor instance instead.
    """
    predictor = IntentPredictor(intents=[intent])
    return predictor.predict(prompt, model_name, intent, threshold)
```

---

### Phase 6: Create KDD Router (30 min)

**Create `llm_jury/routing/kdd_router.py`:**

```python
"""
Router using KDD/data production models.

This replaces the legacy pattern-based router with the validated
production models from KDD/data.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from llm_jury.prediction import IntentPredictor, PredictionResult


@dataclass
class RoutingDecision:
    """Result from routing decision."""
    recommended_model: str
    intent: str
    success_probability: float
    alternatives: List[Tuple[str, float]]  # [(model, prob), ...]
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            'recommended_model': self.recommended_model,
            'intent': self.intent,
            'success_probability': self.success_probability,
            'alternatives': [
                {'model': m, 'prob': p} for m, p in self.alternatives
            ],
            'confidence': self.confidence,
        }


class KDDRouter:
    """
    Production router using KDD/data models.
    
    Pipeline:
    1. User provides prompt + available models
    2. Router predicts success probability for each model on each intent
    3. Recommends best model-intent combination
    4. Returns alternatives ranked by success probability
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        intents: Optional[List[str]] = None,
    ):
        """
        Initialize KDD router.
        
        Args:
            threshold: Minimum success probability threshold
            intents: List of intents to consider (default: all 4)
        """
        self.threshold = threshold
        self.intents = intents or ['reasoning', 'coding', 'summarization', 'rag']
        self.predictor = IntentPredictor(intents=self.intents, threshold=threshold)
    
    def route(
        self,
        prompt: str,
        available_models: List[str],
        intent: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a prompt to the best model.
        
        Args:
            prompt: User prompt
            available_models: List of available model names
            intent: Optional intent (if None, tries all intents)
        
        Returns:
            RoutingDecision with recommendation
        """
        if intent:
            # Single intent specified
            return self._route_single_intent(prompt, available_models, intent)
        else:
            # Try all intents
            return self._route_all_intents(prompt, available_models)
    
    def _route_single_intent(
        self,
        prompt: str,
        available_models: List[str],
        intent: str,
    ) -> RoutingDecision:
        """Route for a single intent."""
        results = []
        
        for model in available_models:
            pred = self.predictor.predict(prompt, model, intent)
            results.append((model, pred.success_probability, pred.confidence))
        
        # Sort by success probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_prob, best_conf = results[0]
        alternatives = [(m, p) for m, p, _ in results[1:]]
        
        return RoutingDecision(
            recommended_model=best_model,
            intent=intent,
            success_probability=best_prob,
            alternatives=alternatives,
            confidence=best_conf,
        )
    
    def _route_all_intents(
        self,
        prompt: str,
        available_models: List[str],
    ) -> RoutingDecision:
        """Route across all intents."""
        results = []
        
        for model in available_models:
            for intent in self.intents:
                pred = self.predictor.predict(prompt, model, intent)
                results.append((
                    model,
                    intent,
                    pred.success_probability,
                    pred.confidence,
                ))
        
        # Sort by success probability
        results.sort(key=lambda x: x[2], reverse=True)
        
        best_model, best_intent, best_prob, best_conf = results[0]
        alternatives = [(m, p) for m, _, p, _ in results[1:]]
        
        return RoutingDecision(
            recommended_model=best_model,
            intent=best_intent,
            success_probability=best_prob,
            alternatives=alternatives,
            confidence=best_conf,
        )
    
    def explain_decision(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Routing Decision:",
            f"  Recommended: {decision.recommended_model}",
            f"  Intent: {decision.intent}",
            f"  Success Probability: {decision.success_probability:.1%}",
            f"  Confidence: {decision.confidence:.1%}",
        ]
        
        if decision.alternatives:
            lines.append(f"  Alternatives:")
            for model, prob in decision.alternatives[:3]:
                lines.append(f"    - {model}: {prob:.1%}")
        
        return "\n".join(lines)
```

---

### Phase 7: Update Tests (20 min)

**Create `tests/test_kdd_integration.py`:**

```python
"""
Test KDD/data integration into llm_jury.
"""

import pytest
from llm_jury.prediction import IntentPredictor, load_model, resolve_name
from llm_jury.routing import KDDRouter


class TestModelLoader:
    """Test model loading."""
    
    def test_load_single_model(self):
        """Test loading a single model."""
        model, card = load_model('rag')
        
        assert model is not None
        assert 'test_auc' in card
        assert card['intent'] == 'rag'
    
    def test_load_all_intents(self):
        """Test loading all models."""
        predictor = IntentPredictor()
        
        assert len(predictor.models) == 4
        assert 'rag' in predictor.models
        assert 'reasoning' in predictor.models


class TestNameResolver:
    """Test name resolution."""
    
    def test_resolve_known_name(self):
        """Test resolving a known OpenCompass name."""
        resolved = resolve_name('gpt-4o-mini-2024-07-18')
        assert resolved == 'GPT-4o mini'
    
    def test_resolve_unknown_name(self):
        """Test resolving an unknown name (returns original)."""
        resolved = resolve_name('unknown-model-xyz')
        assert resolved == 'unknown-model-xyz'


class TestIntentPredictor:
    """Test intent prediction."""
    
    @pytest.mark.skip(reason="Requires NVIDIA API key")
    def test_predict_success(self):
        """Test predicting success probability."""
        predictor = IntentPredictor(intents=['rag'])
        
        result = predictor.predict(
            prompt="What is the capital of France?",
            model_name="GPT-4o mini",
            intent="rag",
        )
        
        assert 0 <= result.success_probability <= 1
        assert result.intent == 'rag'
        assert isinstance(result.should_use, bool)


class TestKDDRouter:
    """Test KDD router."""
    
    def test_router_initialization(self):
        """Test router initializes correctly."""
        router = KDDRouter()
        
        assert router.threshold == 0.5
        assert len(router.intents) == 4
    
    @pytest.mark.skip(reason="Requires NVIDIA API key")
    def test_routing_decision(self):
        """Test routing makes a decision."""
        router = KDDRouter()
        
        decision = router.route(
            prompt="Write a Python function to sort a list",
            available_models=["GPT-4o mini", "Claude 3.5 Sonnet"],
            intent="coding",
        )
        
        assert decision.recommended_model in ["GPT-4o mini", "Claude 3.5 Sonnet"]
        assert decision.intent == "coding"
        assert 0 <= decision.success_probability <= 1
```

---

## 📊 Migration Path

### Immediate (Week 1)
- [x] Phase 1: Create wrapper module structure
- [x] Phase 2: Implement model loader
- [x] Phase 3: Implement name resolver
- [ ] Phase 4: Implement feature extractor (needs NVIDIA API)
- [ ] Phase 5: Implement main predictor
- [ ] Phase 6: Create KDD router

### Short-term (Week 2-3)
- [ ] Phase 7: Add integration tests
- [ ] Update documentation
- [ ] Add example notebooks
- [ ] Performance benchmarking

### Medium-term (Month 1)
- [ ] Migrate llm_jury to use KDD router by default
- [ ] Keep legacy router for comparison
- [ ] Update CLI to support both routers
- [ ] Add migration guide for existing users

---

## 🎯 Success Criteria

### Technical
- [x] All 4 production models loadable from llm_jury
- [ ] Feature extraction working (NVIDIA + capability)
- [ ] Prediction accuracy matches KDD/data validation results
- [ ] Router makes sensible recommendations
- [ ] All tests passing

### Usability
- [ ] Simple API: `predict_success(prompt, model, intent)`
- [ ] Fast: <100ms per prediction (excluding NVIDIA API)
- [ ] Clear documentation with examples
- [ ] Error messages guide users

### Integration
- [ ] Works with existing llm_jury optimization
- [ ] Backwards compatible (legacy router still available)
- [ ] Easy to retrain models (use KDD/data scripts)
- [ ] Validation scripts accessible

---

## 🚧 Open Questions

1. **NVIDIA API Access**: How do we handle the NVIDIA classifier API?
   - Option A: Require API key (current approach)
   - Option B: Pre-compute features for common prompts
   - Option C: Build our own feature extractor (regression model)

2. **Model Updates**: How often should production models be retrained?
   - Suggestion: Monthly, using KDD/data/core_scripts/

3. **Backwards Compatibility**: Should we deprecate the old router?
   - Suggestion: Keep both, make KDD router default, add deprecation warning

4. **Performance**: Should we cache predictions?
   - Suggestion: Yes, add optional caching layer

---

## 📝 Next Steps

1. **Implement Phases 1-3** (immediate, ~1 hour)
2. **Decide on NVIDIA API strategy** (before Phase 4)
3. **Implement Phases 4-6** (2-3 hours)
4. **Add tests and documentation** (1-2 hours)
5. **Update main README with new capabilities**

---

**Total Estimated Time**: 6-8 hours for complete integration

**Status**: Ready to implement  
**Dependencies**: NVIDIA API access strategy decision
