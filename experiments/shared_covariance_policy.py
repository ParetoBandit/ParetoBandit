#!/usr/bin/env python3
"""
Shared Covariance LinUCB Policy

Key insight: One universal A matrix (shared covariance) + separate b vectors.

Parameters:
- Disjoint: 81 models × 384² = 12M parameters (overfits with 497 samples)
- Shared: 1 × 384² + 81 × 384 = 178K parameters (much more tractable)

With dense training (497 prompts × 81 models = 40K samples), we get:
- 40K samples / 178K params ≈ 0.22 ratio (still tight but feasible)
- With PCA d=16: 40K / (256 + 81×16) ≈ 25 samples/param (robust!)
"""

import numpy as np
from typing import Dict, List


class SharedCovarianceLinUCBPolicy:
    """
    LinUCB with Shared Covariance Matrix.
    
    Learns a single A matrix (universal context covariance) shared across
    all models, while maintaining separate b vectors (model-specific rewards).
    
    This dramatically reduces parameters and allows learning from dense
    evaluations (all models graded on each prompt).
    """
    
    def __init__(self, model_names: List[str], dim: int, alpha: float = 0.5):
        """
        Initialize shared covariance policy.
        
        Args:
            model_names: List of model identifiers
            dim: Embedding dimension
            alpha: Exploration parameter (UCB width)
        """
        self.models = model_names
        self.dim = dim
        self.alpha = alpha
        
        # Shared covariance matrix (ONE for all models)
        self.A_shared = np.eye(dim, dtype=np.float64)
        self.A_shared_inv = np.eye(dim, dtype=np.float64)
        
        # Per-model reward vectors
        self.b = {m: np.zeros(dim, dtype=np.float64) for m in model_names}
        
        # Counts for tracking
        self.counts = {m: 0 for m in model_names}
        self.total_updates = 0
    
    def update(self, model: str, context: np.ndarray, reward: float) -> None:
        """
        Update shared A and model-specific b.
        
        Args:
            model: Model identifier
            context: Context embedding vector
            reward: Observed reward (0.0 to 1.0)
        """
        x = np.asarray(context, dtype=np.float64)
        
        # Update shared covariance
        self.A_shared += np.outer(x, x)
        
        # Update model-specific reward accumulator
        self.b[model] += reward * x
        
        # Update inverse using Sherman-Morrison formula
        # A^{-1} -= (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        Ainv_x = self.A_shared_inv @ x
        denom = 1.0 + x.dot(Ainv_x)
        self.A_shared_inv -= np.outer(Ainv_x, Ainv_x) / denom
        
        self.counts[model] += 1
        self.total_updates += 1
    
    def predict(self, model: str, context: np.ndarray) -> tuple:
        """
        Compute UCB for a model on given context.
        
        Returns:
            (mean, ucb, std)
        """
        x = np.asarray(context, dtype=np.float64)
        
        # Mean: theta = A^{-1} b
        theta = self.A_shared_inv @ self.b[model]
        mean = float(theta.dot(x))
        
        # Variance: x^T A^{-1} x
        var = float(x.dot(self.A_shared_inv).dot(x))
        std = np.sqrt(max(var, 1e-12))
        
        # UCB
        ucb = mean + self.alpha * std
        
        return mean, ucb, std
    
    def select(self, context: np.ndarray, rng: np.random.Generator) -> str:
        """
        Select best model using UCB with tie-breaking.
        
        Args:
            context: Context embedding
            rng: Random number generator for tie-breaking
            
        Returns:
            Selected model name
        """
        best_model = None
        best_ucb = -float('inf')
        
        for model in self.models:
            _, ucb, _ = self.predict(model, context)
            
            # Add tiny random noise for tie-breaking
            ucb += rng.random() * 1e-8
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = model
        
        return best_model
    
    def apply_strength(self, strength: float) -> None:
        """
        Apply prior strength multiplier.
        
        Args:
            strength: Confidence multiplier (λ)
        """
        self.A_shared *= strength
        for m in self.models:
            self.b[m] *= strength
        
        # Recompute inverse
        self.A_shared_inv = np.linalg.inv(self.A_shared)
    
    def get_state(self) -> Dict:
        """Get policy state for serialization."""
        return {
            "model_names": self.models,
            "dim": self.dim,
            "alpha": self.alpha,
            "A_shared": self.A_shared,
            "b_dict": self.b,
            "counts": self.counts,
            "total_updates": self.total_updates,
        }
    
    @classmethod
    def from_state(cls, state: Dict) -> 'SharedCovarianceLinUCBPolicy':
        """Restore policy from saved state."""
        policy = cls(
            model_names=state["model_names"],
            dim=state["dim"],
            alpha=state["alpha"],
        )
        policy.A_shared = state["A_shared"]
        policy.A_shared_inv = np.linalg.inv(policy.A_shared)
        policy.b = state["b_dict"]
        policy.counts = state["counts"]
        policy.total_updates = state["total_updates"]
        return policy


if __name__ == "__main__":
    # Quick test
    print("Testing SharedCovarianceLinUCBPolicy...")
    
    models = ["gpt-4", "claude-3", "llama-3"]
    dim = 16
    policy = SharedCovarianceLinUCBPolicy(models, dim)
    
    rng = np.random.default_rng(42)
    
    # Simulate some updates
    for _ in range(100):
        ctx = rng.standard_normal(dim)
        model = rng.choice(models)
        reward = rng.random()
        policy.update(model, ctx, reward)
    
    # Test selection
    test_ctx = rng.standard_normal(dim)
    selected = policy.select(test_ctx, rng)
    
    print(f"✓ Policy initialized with {len(models)} models, dim={dim}")
    print(f"✓ Performed {policy.total_updates} updates")
    print(f"✓ Selected model: {selected}")
    print(f"✓ Shared A matrix shape: {policy.A_shared.shape}")
    print(f"✓ Parameters: {dim*dim} (A) + {len(models)*dim} (b) = {dim*dim + len(models)*dim}")

