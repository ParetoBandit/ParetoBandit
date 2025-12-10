"""
Intent Classifier for prompt categorization.

Uses XGBoost + sentence embeddings for fast, accurate intent prediction.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class IntentClassifier:
    """
    Classify prompts into intent categories for routing.
    
    Features: Sentence embeddings (semantic representation)
    Model: XGBoost (fast, handles missing values)
    Training: 5-fold CV on ground-truth labeled data
    
    Attributes:
        intents: List of supported intent labels
        model: Trained XGBoost classifier
        embedder: Sentence embedding model
    """
    
    INTENT_LABELS = [
        'coding',
        'reasoning', 
        'factual_qa',
        'summarization',
        'agentic_execution',
        'general'
    ]
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize intent classifier.
        
        Args:
            model_path: Path to trained XGBoost model (auto-download if None)
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model_name = embedding_model
        self.model_path = model_path
        self._embedder = None
        self._model = None
        
    def _load_embedder(self):
        """Lazy load sentence embedder (only when needed)."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder
    
    def _load_model(self):
        """Lazy load XGBoost model."""
        if self._model is None:
            if self.model_path is None:
                raise ValueError(
                    "No trained model found. Please train a model first using "
                    "scripts/intent_classification/train_xgboost_classifier.py"
                )
            
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
        
        return self._model
    
    def predict(
        self, 
        prompt: Union[str, List[str]],
        return_probabilities: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict intent for one or more prompts.
        
        Args:
            prompt: Single prompt string or list of prompts
            return_probabilities: If True, include probability distribution
            
        Returns:
            Dictionary with:
                - intent: Predicted intent label
                - confidence: Confidence score (0-1)
                - probabilities: Full probability distribution (if requested)
        """
        # Handle single vs batch
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else prompt
        
        # Extract embeddings
        embedder = self._load_embedder()
        embeddings = embedder.encode(prompts, convert_to_numpy=True)
        
        # Predict
        model = self._load_model()
        predictions = model.predict(embeddings)
        probabilities = model.predict_proba(embeddings)
        
        # Format results
        results = []
        for pred, proba in zip(predictions, probabilities):
            result = {
                'intent': self.INTENT_LABELS[int(pred)],
                'confidence': float(np.max(proba))
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    label: float(p) 
                    for label, p in zip(self.INTENT_LABELS, proba)
                }
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def predict_batch(
        self,
        prompts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict intents for a large batch with progress tracking.
        
        Args:
            prompts: List of prompt strings
            batch_size: Batch size for embedding extraction
            
        Returns:
            List of prediction dictionaries
        """
        embedder = self._load_embedder()
        model = self._load_model()
        
        all_results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            embeddings = embedder.encode(batch, convert_to_numpy=True)
            predictions = model.predict(embeddings)
            probabilities = model.predict_proba(embeddings)
            
            for pred, proba in zip(predictions, probabilities):
                all_results.append({
                    'intent': self.INTENT_LABELS[int(pred)],
                    'confidence': float(np.max(proba))
                })
        
        return all_results
    
    @property
    def intents(self) -> List[str]:
        """Get list of supported intent labels."""
        return self.INTENT_LABELS.copy()
