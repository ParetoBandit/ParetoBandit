"""
Domain Calibration for Contextual Bandit Routers

This module provides tools for adapting pre-trained routers to new domains
using covariance inflation (gamma scaling) and minimal calibration data.

Key Classes:
    - CalibratedRouter: LinUCB router with domain-adapted priors
    - SimpleLinUCBRouter: Lightweight router for calibration experiments

Key Functions:
    - apply_gamma_scaling: Apply covariance inflation to priors
    - embed_prompt: Embed prompts with PCA for LinUCB context

Usage:
    >>> from bandit_gpt.calibration import CalibratedRouter
    >>> router = CalibratedRouter.load("my_router.joblib", encoder, pca_model)
    >>> model = router.select_model("How do I optimize React rendering?")
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def apply_gamma_scaling(priors: dict, gamma: float) -> dict:
    """
    Apply covariance inflation to warmup priors.
    
    This reduces the effective sample size of the warmup prior,
    allowing calibration data to have more influence on the final policy.
    
    Args:
        priors: Warmup priors dictionary with 'A', 'b', 'models', 'context_dim'
        gamma: Inflation factor ∈ (0, 1]. Lower = more inflation.
    
    Returns:
        Scaled priors dictionary with same structure.
    
    Mathematical Effect:
        A_adapted = A_warmup × γ
        N_eff = N_warmup × γ
    
    Example:
        >>> priors_scaled = apply_gamma_scaling(warmup_priors, gamma=0.01)
        >>> # Effective sample size reduced from 80,000 to 800
    """
    return {
        'A': {m: priors['A'][m] * gamma for m in priors['models']},
        'b': {m: priors['b'][m].copy() for m in priors['models']},
        'models': priors['models'],
        'context_dim': priors['context_dim'],
        'gamma': gamma,
        'n_prompts': priors.get('n_prompts', 0),
        'plasticity': priors.get('plasticity', 1.0)
    }


def embed_prompt(prompt: str, encoder: 'SentenceTransformer', pca_model) -> np.ndarray:
    """
    Embed prompt with PCA projection (must match warmup pipeline).
    
    Args:
        prompt: User query text
        encoder: SentenceTransformer model (e.g., all-MiniLM-L6-v2)
        pca_model: Fitted PCA model (joblib-loaded)
    
    Returns:
        Context vector: [23 PCA components, 1 bias term] ∈ ℝ^24
    
    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
        >>> encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
        >>> pca_model = joblib.load("pca_23.joblib")
        >>> context = embed_prompt("Hello world", encoder, pca_model)
        >>> context.shape  # (24,)
    """
    embedding = encoder.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
    embedding = pca_model.transform(embedding.reshape(1, -1)).flatten()
    return np.append(embedding, 1.0)  # Add bias term


class CalibratedRouter:
    """
    LinUCB router with domain-adapted priors via covariance inflation.
    
    This router is initialized from warmup priors (trained on large-scale data)
    and adapted to a specific domain using minimal calibration samples (100-200).
    
    Attributes:
        models: List of model IDs (e.g., ["mixtral-8x7b", "gpt-4-turbo"])
        alpha: Exploration parameter (UCB bonus multiplier)
        lambda_cost: Cost penalty for expensive models
        context_dim: Dimension of context vectors (typically 24 = 23 PCA + 1 bias)
        A: Dict of precision matrices (d×d) per model
        b: Dict of reward accumulators (d,) per model
        metadata: Calibration metadata (warmup size, gamma, samples)
    
    Example:
        >>> # Initialize from warmup priors
        >>> router = CalibratedRouter(
        ...     warmup_priors=warmup_priors,
        ...     encoder=encoder,
        ...     pca_model=pca_model,
        ...     alpha=1.0
        ... )
        >>> 
        >>> # Route a query
        >>> model = router.select_model("Explain async/await")
        >>> 
        >>> # Update after observing reward
        >>> router.update("Explain async/await", reward=0.95)
        >>> 
        >>> # Save for production
        >>> router.save("production_router.joblib")
    """
    
    def __init__(
        self,
        warmup_priors: dict,
        encoder: 'SentenceTransformer',
        pca_model,
        alpha: float = 1.0,
        lambda_cost: float = 0.0
    ):
        """
        Initialize router from warmup priors.
        
        Args:
            warmup_priors: Dictionary with 'A', 'b', 'models', 'context_dim'
            encoder: SentenceTransformer model for embedding prompts
            pca_model: Fitted PCA model for dimensionality reduction
            alpha: Exploration parameter (default: 1.0)
            lambda_cost: Cost penalty for expensive models (default: 0.0)
        """
        self.models = warmup_priors['models']
        self.alpha = alpha
        self.lambda_cost = lambda_cost
        self.context_dim = warmup_priors['context_dim']
        self.encoder = encoder
        self.pca_model = pca_model
        
        # Initialize from warmup priors
        self.A = {m: warmup_priors['A'][m].copy() for m in self.models}
        self.b = {m: warmup_priors['b'][m].copy() for m in self.models}
        
        # Metadata
        self.metadata = {
            'n_prompts_warmup': warmup_priors.get('n_prompts', 0),
            'gamma': warmup_priors.get('gamma', 1.0),
            'n_calibration_samples': 0
        }
    
    def select_model(self, prompt: str) -> str:
        """
        Select the best model for a given prompt using LinUCB.
        
        Args:
            prompt: User query text
        
        Returns:
            model_id: ID of the selected model
        
        Algorithm:
            UCB(m) = θ_m^T x + α √(x^T A_m^{-1} x) - λ_cost × cost(m)
            Select: argmax_m UCB(m)
        """
        context = embed_prompt(prompt, self.encoder, self.pca_model)
        
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
            # UCB = expected reward + exploration bonus - cost penalty
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            
            # Cost penalty (assume strong model = models[1])
            cost = self.lambda_cost if model == self.models[1] else 0.0
            
            ucb_scores[model] = expected + self.alpha * uncertainty - cost
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, prompt: str, reward: float):
        """
        Update router after observing reward (online learning).
        
        Args:
            prompt: The prompt that was routed
            reward: Observed reward (0.0-1.0)
        
        Updates:
            A_m ← A_m + xx^T
            b_m ← b_m + r·x
        """
        context = embed_prompt(prompt, self.encoder, self.pca_model)
        model = self.select_model(prompt)
        
        context = context.reshape(-1, 1)  # Column vector
        self.A[model] += context @ context.T
        self.b[model] += (reward * context).flatten()
        
        self.metadata['n_calibration_samples'] += 1
    
    def save(self, output_file: Path | str):
        """
        Save calibrated router to disk.
        
        Args:
            output_file: Path to save location (.joblib)
        
        Saved State:
            - A, b matrices (LinUCB parameters)
            - models, context_dim (configuration)
            - alpha, lambda_cost (hyperparameters)
            - metadata (calibration history)
        """
        state = {
            'A': self.A,
            'b': self.b,
            'models': self.models,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_cost': self.lambda_cost,
            'metadata': self.metadata
        }
        joblib.dump(state, Path(output_file))
    
    @classmethod
    def load(
        cls,
        router_file: Path | str,
        encoder: 'SentenceTransformer',
        pca_model
    ) -> 'CalibratedRouter':
        """
        Load a saved calibrated router from disk.
        
        Args:
            router_file: Path to .joblib file
            encoder: SentenceTransformer model (must match training)
            pca_model: PCA model (must match training)
        
        Returns:
            CalibratedRouter instance ready for inference
        
        Example:
            >>> from sentence_transformers import SentenceTransformer
            >>> from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
            >>> import joblib
            >>> 
            >>> encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
            >>> pca_model = joblib.load("pca_23.joblib")
            >>> router = CalibratedRouter.load("router.joblib", encoder, pca_model)
            >>> 
            >>> model = router.select_model("My query")
        """
        state = joblib.load(Path(router_file))
        router = cls.__new__(cls)
        router.A = state['A']
        router.b = state['b']
        router.models = state['models']
        router.context_dim = state['context_dim']
        router.alpha = state['alpha']
        router.lambda_cost = state['lambda_cost']
        router.metadata = state['metadata']
        router.encoder = encoder
        router.pca_model = pca_model
        return router


class SimpleLinUCBRouter:
    """
    Lightweight LinUCB router for calibration experiments.
    
    This is a simplified version of CalibratedRouter for use in
    gamma-finding experiments where we don't need full functionality.
    
    Attributes:
        models: List of model IDs
        alpha: Exploration parameter
        context_dim: Context vector dimension
        A: Precision matrices per model
        b: Reward accumulators per model
    """
    
    def __init__(self, models: List[str], warmup_priors: dict, alpha: float = 1.0):
        """Initialize from warmup priors."""
        self.models = models
        self.alpha = alpha
        self.context_dim = warmup_priors['context_dim']
        
        # Initialize from warmup priors
        self.A = {m: warmup_priors['A'][m].copy() for m in models}
        self.b = {m: warmup_priors['b'][m].copy() for m in models}
    
    def select_model(self, context: np.ndarray) -> str:
        """Select model using UCB (takes pre-computed context vector)."""
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
            # UCB = expected reward + exploration bonus
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores[model] = expected + self.alpha * uncertainty
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context: np.ndarray, model: str, reward: float):
        """Update matrices after observing reward."""
        context = context.reshape(-1, 1)  # Column vector
        self.A[model] += context @ context.T
        self.b[model] += (reward * context).flatten()
    
    def get_model_usage(self) -> Dict[str, float]:
        """Get cumulative model selection probabilities (approximation via trace)."""
        total_updates = sum(np.trace(self.A[m]) for m in self.models)
        if total_updates == 0:
            return {m: 100.0 / len(self.models) for m in self.models}
        return {m: (np.trace(self.A[m]) / total_updates) * 100 for m in self.models}

