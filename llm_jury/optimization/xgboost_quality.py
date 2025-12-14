"""
XGBoost-based Quality Prediction for Optimization.

Replaces static benchmark-weighted quality with dynamic predictions
from production XGBoost models trained on 113K real examples.

Quality = P(success | prompt, model, intent)

When NVIDIA features or capability scores are unavailable for the primary intent,
uses the intent fallback chain to try alternative XGBoost models.

Example:
    predictor = XGBoostQualityPredictor()
    
    # Predict for specific prompt
    quality = predictor.predict_quality(
        prompt="What is the capital of France?",
        model_data={'name': 'GPT-4o mini', ...},
        intent="rag"
    )
    # Returns: 0.87 (87% predicted success)
"""

import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import production XGBoost models
from llm_jury.prediction import load_all_models, get_all_model_info
from llm_jury.prediction.models import OPENCOMPASS_TO_CACHE


# =============================================================================
# Intent Mapping
# =============================================================================

# Map optimization intent names to XGBoost model intents
INTENT_MAPPING = {
    'coding': 'coding',
    'reasoning': 'reasoning',
    'factual_qa': 'rag',  # Factual QA uses RAG model
    'summarization': 'summarization',
    'agentic': 'coding',  # Agentic tasks closest to coding
    'general': 'reasoning',  # General uses reasoning as default
    'creative': 'summarization',  # Creative uses summarization
    'rag': 'rag',  # Direct mapping
}

# Fallback chain: if primary intent fails, try these in order
INTENT_FALLBACK_CHAIN = {
    'coding': ['reasoning', 'rag'],
    'reasoning': ['rag', 'coding'],
    'rag': ['reasoning', 'coding'],
    'summarization': ['reasoning', 'rag'],
}

# Map intents to capability score fields
# Use fields with high uniqueness (minimal duplicate values across models)
# to maximize prediction differentiation between models.
#
# Field selection rationale:
# - coding: livecodebench (80/81 unique) >> humaneval_score (45/67 unique, 6 share same)
# - reasoning: gpqa (78/81 unique) - good differentiation
# - summarization: summedits_score (77/81 unique) - ifeval has no data!
# - rag: mmlu_pro (75/81 unique) - matches training
CAPABILITY_FIELDS = {
    'coding': 'livecodebench',     # LiveCodeBench has much better model differentiation
    'reasoning': 'gpqa',           # GPQA has good coverage and uniqueness
    'summarization': 'summedits_score',  # SummEdits score for summarization capability
    'rag': 'mmlu_pro',             # MMLU-Pro matches training data
}


# =============================================================================
# XGBoost Quality Predictor
# =============================================================================

class XGBoostQualityPredictor:
    """
    Predict quality using production XGBoost models.
    
    Quality = P(success | prompt, model, intent)
    
    When NVIDIA features or capability scores are unavailable for the primary intent,
    uses the intent fallback chain to try alternative XGBoost models:
    
    - coding → reasoning → rag
    - reasoning → rag → coding
    - rag → reasoning → coding
    - summarization → reasoning → rag
    
    Attributes:
        models: Dict mapping intent -> (xgboost_model, model_card)
        nvidia_features_available: Whether NVIDIA API is accessible
    """
    
    def __init__(self):
        """Initialize XGBoost quality predictor with production models."""
        # Load all production XGBoost models
        try:
            self.models = load_all_models()
            self.model_info = get_all_model_info()
            print(f"✓ Loaded {len(self.models)} XGBoost models for quality prediction")
        except Exception as e:
            warnings.warn(f"Could not load XGBoost models: {e}")
            self.models = {}
            self.model_info = {}
        
        # Check if NVIDIA classifier is available
        try:
            from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
            self.nvidia_features_available = True
        except:
            self.nvidia_features_available = False
            warnings.warn("NVIDIA classifier not available. Install transformers and torch.")
    
    def _map_intent(self, intent: str) -> str:
        """Map optimization intent to XGBoost model intent."""
        mapped = INTENT_MAPPING.get(intent, 'reasoning')
        if mapped not in self.models:
            warnings.warn(
                f"Intent '{intent}' mapped to '{mapped}' but model not loaded. "
                f"Using 'reasoning' as fallback."
            )
            return 'reasoning'
        return mapped
    
    def _get_capability_score(
        self,
        model_data: Dict,
        intent: str,
    ) -> Optional[float]:
        """
        Extract capability score for intent from model data.
        
        Args:
            model_data: Model dict with benchmark scores
            intent: XGBoost intent (after mapping)
        
        Returns:
            Capability score or None if unavailable
        """
        # Get the capability field for this intent
        capability_field = CAPABILITY_FIELDS.get(intent)
        if not capability_field:
            return None
        
        # Try to get the score
        score = model_data.get(capability_field)
        if score is None or score == 'N/A':
            return None
        
        try:
            score_float = float(score)
            # Convert to 0-100 scale if needed
            if score_float <= 1.0:
                score_float *= 100
            return score_float
        except (ValueError, TypeError):
            return None
    
    def _extract_nvidia_features(self, prompt: str) -> Optional[Dict[str, float]]:
        """
        Extract NVIDIA prompt complexity features.
        
        Uses the NVIDIA prompt-task-and-complexity-classifier to extract
        6 complexity dimensions that were used in XGBoost training.
        
        Args:
            prompt: User prompt
        
        Returns:
            Dict with 6 NVIDIA features or None if classifier unavailable
        """
        try:
            from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
            
            # Lazy load classifier on first use
            if not hasattr(self, '_nvidia_classifier'):
                self._nvidia_classifier = NvidiaComplexityClassifier()
            
            # Get complexity features
            result = self._nvidia_classifier.classify(prompt)
            
            # Return features in the format expected by XGBoost models
            # These match the training feature names
            return {
                'nvidia_creativity': result.creativity_scope,
                'nvidia_reasoning': result.reasoning,
                'nvidia_constraint': result.constraint_ct,
                'nvidia_domain_knowledge': result.domain_knowledge,
                'nvidia_contextual_knowledge': result.contextual_knowledge,
                'nvidia_few_shots': result.number_of_few_shots,
            }
            
        except Exception as e:
            # If NVIDIA classifier fails (missing dependencies, etc.),
            # return None to trigger intent fallback
            warnings.warn(f"Could not extract NVIDIA features: {e}")
            return None
    
    def predict_quality(
        self,
        prompt: str,
        model_data: Dict,
        intent: str,
    ) -> float:
        """
        Predict success probability for (prompt, model, intent).
        
        If the primary intent fails (missing features/capability), tries fallback
        intents from INTENT_FALLBACK_CHAIN in order.
        
        Args:
            prompt: User prompt
            model_data: Model dict with benchmarks and metadata
            intent: Intent category (will be mapped to XGBoost intent)
        
        Returns:
            Quality score [0, 1] (success probability)
        """
        # Map intent
        xgb_intent = self._map_intent(intent)
        
        # Try primary intent
        result = self._try_predict_for_intent(prompt, model_data, xgb_intent)
        if result is not None:
            return result
        
        # Primary intent failed, try fallback chain
        fallback_intents = INTENT_FALLBACK_CHAIN.get(xgb_intent, [])
        for fallback_intent in fallback_intents:
            result = self._try_predict_for_intent(prompt, model_data, fallback_intent)
            if result is not None:
                warnings.warn(
                    f"Using fallback intent '{fallback_intent}' for primary intent '{xgb_intent}'"
                )
                return result
        
        # All intents failed, return neutral score
        warnings.warn(
            f"All XGBoost predictions failed for intent '{xgb_intent}'. "
            f"Returning neutral score."
        )
        return 0.5
    
    def _try_predict_for_intent(
        self,
        prompt: str,
        model_data: Dict,
        intent: str,
    ) -> Optional[float]:
        """
        Attempt XGBoost prediction for a specific intent.
        
        Args:
            prompt: User prompt
            model_data: Model dict
            intent: XGBoost intent (already mapped)
        
        Returns:
            Success probability [0, 1] or None if prediction fails
        """
        # Check if XGBoost model available
        if intent not in self.models:
            return None
        
        # Extract NVIDIA features
        nvidia_features = self._extract_nvidia_features(prompt)
        if nvidia_features is None:
            return None
        
        # Extract capability score
        capability = self._get_capability_score(model_data, intent)
        if capability is None:
            return None
        
        # Prepare feature vector (7 features)
        features = np.array([[
            nvidia_features['nvidia_creativity'],
            nvidia_features['nvidia_reasoning'],
            nvidia_features['nvidia_constraint'],
            nvidia_features['nvidia_domain_knowledge'],
            nvidia_features['nvidia_contextual_knowledge'],
            nvidia_features['nvidia_few_shots'],
            capability,  # Model capability for this intent
        ]])
        
        # Predict with XGBoost model
        model, card = self.models[intent]
        try:
            proba = model.predict_proba(features)[0]
            xgb_prob = proba[1]  # Probability of success (class 1)
            
            # Apply capability-based adjustment
            # The XGBoost model has low sensitivity to capability (only ~4% diff between cap 20-80)
            # because it was trained on model_aggregate (narrow range) not benchmark scores.
            # We blend the XGBoost prediction with a capability-based adjustment to give
            # models with higher capability scores appropriately higher predictions.
            #
            # Adjustment formula: final = (1 - alpha) * xgb_prob + alpha * capability_factor
            # where capability_factor = capability / 100 (normalized to 0-1)
            # alpha controls blend strength (0.3 = 30% from capability, 70% from XGBoost)
            
            alpha = 0.3  # Capability adjustment strength
            capability_factor = min(capability / 100.0, 1.0)  # Normalize to 0-1
            
            # Blend XGBoost prediction with capability factor
            adjusted_prob = (1 - alpha) * xgb_prob + alpha * capability_factor
            
            # Ensure result is in valid range
            return float(np.clip(adjusted_prob, 0.0, 1.0))
            
        except Exception as e:
            warnings.warn(f"XGBoost prediction failed for intent '{intent}': {e}")
            return None
    
    def predict_batch(
        self,
        prompt: str,
        models_data: List[Dict],
        intent: str,
    ) -> List[float]:
        """
        Predict quality for multiple models with same prompt and intent.
        
        Args:
            prompt: User prompt
            models_data: List of model dicts
            intent: Intent category
        
        Returns:
            List of quality scores [0, 1]
        """
        return [
            self.predict_quality(prompt, model, intent)
            for model in models_data
        ]
    
    def get_model_info_summary(self) -> str:
        """Get summary of loaded XGBoost models."""
        if not self.models:
            return "No XGBoost models loaded"
        
        lines = ["XGBoost Models Loaded:"]
        for intent, (model, card) in self.models.items():
            lines.append(
                f"  - {intent}: Test AUC={card['test_auc']:.3f}, "
                f"{card['n_train_examples']:,} examples"
            )
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_quality_xgboost(
    prompt: str,
    model_data: Dict,
    intent: str,
) -> float:
    """
    Convenience function to predict quality using XGBoost.
    
    Args:
        prompt: User prompt
        model_data: Model dict
        intent: Intent category
    
    Returns:
        Quality score [0, 1]
    """
    predictor = XGBoostQualityPredictor()
    return predictor.predict_quality(prompt, model_data, intent)


def create_quality_predictor() -> XGBoostQualityPredictor:
    """
    Factory function to create XGBoost quality predictor.
    
    Returns:
        Configured XGBoostQualityPredictor
    """
    return XGBoostQualityPredictor()
