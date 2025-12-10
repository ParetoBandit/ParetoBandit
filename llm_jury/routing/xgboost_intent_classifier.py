"""
XGBoost-based Intent Classifier.

Uses machine learning (XGBoost) with engineered features to classify intents.
Compared to regex-based approach, this learns patterns from data.
"""

import re
import json
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from llm_jury.routing.intent_classifier import IntentCategory


@dataclass
class XGBoostIntentResult:
    """Result from XGBoost intent classification."""
    category: IntentCategory
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: Optional[Dict[str, float]] = None
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
        }


class FeatureExtractor:
    """
    Extract features from prompts for ML classification.
    
    Features include:
        - Pattern-based signals (binary indicators)
        - Lexical features (word counts, lengths, etc.)
        - Statistical features (character distributions)
        - Linguistic features (POS patterns, question words)
    """
    
    def __init__(self):
        """Initialize feature extractor with pattern definitions."""
        # Pattern categories (from regex classifier)
        self.patterns = {
            'math_symbols': r'[∫∑∏∂∇√]|(\d+\s*[+\-×÷*/]\s*\d+)',
            'math_words': r'\b(equation|integral|derivative|formula|theorem|calculate|solve|compute|prove)\b',
            'logic_words': r'\b(logic|logical|reasoning|deduce|infer|conclude|if.*then|therefore)\b',
            
            'programming_langs': r'\b(python|javascript|typescript|java|c\+\+|rust|go|sql|html|css)\b',
            'code_keywords': r'\b(function|class|method|variable|array|def|import|return)\b',
            'code_actions': r'\b(implement|refactor|debug|optimize|compile|code)\b',
            'code_blocks': r'```[\s\S]*?```',
            
            'question_words': r'^(what|who|when|where|why|how|which|whose)\b',
            'knowledge_words': r'\b(explain|describe|define|fact|information|teach|learn)\b',
            
            'agent_keywords': r'\b(workflow|pipeline|agent|autonomous|orchestrate|automate)\b',
            'multistep': r'\b(plan|planning|multi[- ]step|first.*then|step\s+\d+)\b',
            'tool_use': r'\b(api|tool|function\s+call|execute|use\s+the)\b',
            
            'conversational': r'\b(chat|hello|hi|hey|thanks|help\s+me|can\s+you)\b',
            'opinion': r'\b(think|opinion|recommend|suggest|best|should\s+i)\b',
        }
        
        # Compile patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        
        # Question words
        self.question_words = [
            'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose'
        ]
    
    def extract_features(self, prompt: str) -> Dict[str, float]:
        """
        Extract features from a prompt.
        
        Args:
            prompt: The prompt to extract features from
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        prompt_lower = prompt.lower().strip()
        words = prompt_lower.split()
        
        # === Pattern-based features ===
        for name, pattern in self.compiled_patterns.items():
            features[f'has_{name}'] = float(bool(pattern.search(prompt)))
            matches = pattern.findall(prompt)
            features[f'count_{name}'] = float(len(matches))
        
        # === Lexical features ===
        features['length_chars'] = float(len(prompt))
        features['length_words'] = float(len(words))
        features['avg_word_length'] = float(np.mean([len(w) for w in words]) if words else 0)
        features['max_word_length'] = float(max([len(w) for w in words]) if words else 0)
        
        # === Character-based features ===
        features['num_digits'] = float(sum(c.isdigit() for c in prompt))
        features['num_upper'] = float(sum(c.isupper() for c in prompt))
        features['num_special'] = float(sum(not c.isalnum() and not c.isspace() for c in prompt))
        features['has_code_block'] = float('```' in prompt)
        features['has_parentheses'] = float('(' in prompt and ')' in prompt)
        features['has_brackets'] = float('[' in prompt or '{' in prompt)
        
        # === Question features ===
        features['starts_with_question'] = float(any(
            prompt_lower.startswith(qw) for qw in self.question_words
        ))
        features['has_question_mark'] = float('?' in prompt)
        features['num_question_marks'] = float(prompt.count('?'))
        
        # === Sentence structure ===
        sentences = [s.strip() for s in re.split(r'[.!?]', prompt) if s.strip()]
        features['num_sentences'] = float(len(sentences))
        features['has_multiple_sentences'] = float(len(sentences) > 1)
        
        # === Word frequency features (top keywords) ===
        coding_words = ['code', 'function', 'write', 'implement', 'class', 'method', 'script']
        math_words = ['calculate', 'solve', 'equation', 'formula', 'compute', 'derivative']
        question_indicators = ['what', 'who', 'explain', 'describe', 'tell', 'how']
        agent_words = ['plan', 'workflow', 'automate', 'execute', 'schedule', 'agent']
        
        features['coding_word_count'] = float(sum(words.count(w) for w in coding_words))
        features['math_word_count'] = float(sum(words.count(w) for w in math_words))
        features['question_word_count'] = float(sum(words.count(w) for w in question_indicators))
        features['agent_word_count'] = float(sum(words.count(w) for w in agent_words))
        
        # === Imperative vs interrogative ===
        imperative_verbs = ['write', 'create', 'implement', 'build', 'make', 'generate', 'design']
        features['has_imperative'] = float(any(
            prompt_lower.startswith(v) for v in imperative_verbs
        ))
        
        return features
    
    def extract_batch(self, prompts: List[str]) -> np.ndarray:
        """Extract features for multiple prompts."""
        features_list = [self.extract_features(p) for p in prompts]
        
        # Get all feature names (should be consistent)
        if not features_list:
            return np.array([])
        
        feature_names = sorted(features_list[0].keys())
        
        # Build feature matrix
        X = np.array([
            [feat_dict[name] for name in feature_names]
            for feat_dict in features_list
        ])
        
        return X, feature_names


class XGBoostIntentClassifier:
    """
    XGBoost-based intent classifier.
    
    Uses engineered features and gradient boosting to classify intents.
    Can be trained on labeled data or loaded from a saved model.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            model_path: Path to saved model (optional)
            feature_names: List of feature names (required if model_path provided)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = feature_names
        # 5 classes - GENERAL is now a trained class
        self.label_encoder = {
            'reasoning': 0,
            'coding': 1,
            'factual_qa': 2,
            'agentic': 3,
            'general': 4,
        }
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        self.confidence_threshold = 0.5  # Below this confidence = low trust
        
        if model_path:
            self.load(model_path)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **xgb_params
    ) -> Dict:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels (integers)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features
            **xgb_params: Additional XGBoost parameters
            
        Returns:
            Training history/metrics
        """
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Default XGBoost parameters (5 classes including GENERAL)
        params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss',
        }
        params.update(xgb_params)
        
        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        
        # Get training history
        history = {
            'train_loss': self.model.evals_result()['validation_0']['mlogloss'],
        }
        if len(eval_set) > 1:
            history['val_loss'] = self.model.evals_result()['validation_1']['mlogloss']
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes and probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def classify(self, prompt: str) -> XGBoostIntentResult:
        """
        Classify a single prompt.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            XGBoostIntentResult with category and confidence
        """
        start_time = time.perf_counter()
        
        # Extract features
        features = self.feature_extractor.extract_features(prompt)
        
        # Convert to array in correct order
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Predict
        pred_idx, probs = self.predict(X)
        pred_idx = pred_idx[0]
        probs = probs[0]
        
        # Get category and confidence
        category_str = self.label_decoder[pred_idx]
        # Handle both 'agentic' and legacy 'agentic_execution' 
        if category_str == 'agentic':
            category = IntentCategory.AGENTIC
        else:
            category = IntentCategory(category_str)
        confidence = float(probs[pred_idx])
        
        # Build probabilities dict
        prob_dict = {
            self.label_decoder[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return XGBoostIntentResult(
            category=category,
            confidence=confidence,
            probabilities=prob_dict,
            features=features,
            latency_ms=latency_ms,
        )
    
    def classify_batch(self, prompts: List[str]) -> List[XGBoostIntentResult]:
        """Classify multiple prompts."""
        return [self.classify(prompt) for prompt in prompts]
    
    def save(self, model_path: str):
        """Save the trained model and feature names."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(model_path))
        
        # Save feature names
        metadata_path = model_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'label_encoder': self.label_encoder,
            }, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
    
    def load(self, model_path: str):
        """Load a trained model and feature names."""
        model_path = Path(model_path)
        
        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = model_path.with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            self.label_encoder = metadata['label_encoder']
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        
        print(f"Model loaded from: {model_path}")


# Convenience function
def get_xgboost_classifier(model_path: Optional[str] = None) -> XGBoostIntentClassifier:
    """Get or create XGBoost classifier."""
    return XGBoostIntentClassifier(model_path=model_path)

