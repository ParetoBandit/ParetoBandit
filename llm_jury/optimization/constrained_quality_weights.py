"""
Constrained Quality Weight Optimization via Lagrangian Dual.

Maximizes quality (correlation with Arena ELO) subject to constraints:
- Hallucination constraint: top models must have low hallucination
- Cost constraint: top models must be affordable

The shadow prices reveal the quality-constraint trade-offs:
- λ_safety: Quality lost per 1% stricter hallucination limit
- λ_cost: Quality lost per $1 tighter budget

Mathematical Formulation:
    
    max  Corr(Σ w_i × Benchmark_i, ArenaELO)
    s.t. E[Hallucination(top_k)] ≤ H_max
         E[Cost(top_k)] ≤ C_max
         Σ w_i = 1, w_i ≥ 0

Lagrangian:
    L = -Corr + λ_h(E[H] - H_max) + λ_c(E[C] - C_max)

Usage:
    from llm_jury.optimization import ConstrainedQualityOptimizer
    
    optimizer = ConstrainedQualityOptimizer(models_data)
    result = optimizer.optimize(
        hallucination_max=8.0,
        cost_max=0.01,
    )
    
    print(f"Optimal weights achieve r={result.correlation:.3f}")
    print(f"Shadow price of safety: {result.lambda_safety:.3f}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


BENCHMARKS = [
    'intelligence_index',
    'math_index',
    'mmlu_pro',
    'gpqa',
    'hle',
    'livecodebench',
    'scicode',
    'math_500',
    'aime',
]


@dataclass
class ConstrainedQualityResult:
    """Result from constrained quality optimization."""
    
    # Optimal weights
    weights: Dict[str, float]
    
    # Quality achieved
    correlation: float
    r_squared: float
    
    # Shadow prices (marginal quality cost of tightening constraints)
    lambda_safety: float  # Quality lost per 1% stricter hallucination
    lambda_cost: float    # Quality lost per $1 stricter budget
    
    # Constraint values
    hallucination_max: float
    cost_max: float
    achieved_hallucination: float
    achieved_cost: float
    
    # Convergence info
    converged: bool
    iterations: int
    
    # Models info
    n_models: int
    top_k: int
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "CONSTRAINED QUALITY OPTIMIZATION",
            "=" * 55,
            "",
            f"Quality: r = {self.correlation:.4f} (R² = {self.r_squared:.4f})",
            "",
            "Constraints & Shadow Prices:",
            f"  Hallucination: target ≤{self.hallucination_max:.1f}%, achieved {self.achieved_hallucination:.1f}%",
            f"    → λ_safety = {self.lambda_safety:.4f}",
        ]
        
        if self.lambda_safety > 0.01:
            lines.append(f"    → BINDING: Each 1% stricter costs {self.lambda_safety:.3f} correlation")
        else:
            lines.append(f"    → Slack (not binding)")
        
        lines.extend([
            f"  Cost: target ≤${self.cost_max:.4f}, achieved ${self.achieved_cost:.4f}",
            f"    → λ_cost = {self.lambda_cost:.4f}",
        ])
        
        if self.lambda_cost > 0.01:
            lines.append(f"    → BINDING: Each $0.01 stricter costs {self.lambda_cost*0.01:.4f} correlation")
        else:
            lines.append(f"    → Slack (not binding)")
        
        lines.extend([
            "",
            "Optimal Weights (top 5):",
        ])
        
        sorted_weights = sorted(self.weights.items(), key=lambda x: -x[1])[:5]
        for name, weight in sorted_weights:
            lines.append(f"  {name:<20} {weight:.3f}")
        
        return "\n".join(lines)
    
    def get_paper_statement(self) -> str:
        """Paper-ready statement."""
        return (
            f"Constrained optimization achieves r={self.correlation:.3f} with Arena ELO "
            f"while satisfying H≤{self.hallucination_max}% (λ={self.lambda_safety:.3f}) "
            f"and C≤${self.cost_max:.4f} (λ={self.lambda_cost:.3f}). "
            f"Shadow prices indicate {'safety' if self.lambda_safety > self.lambda_cost else 'cost'} "
            f"is the binding constraint."
        )


class ConstrainedQualityOptimizer:
    """
    Maximize quality correlation subject to safety and cost constraints.
    
    Uses projected gradient descent with Lagrangian dual updates.
    """
    
    def __init__(self, models_data: List[Dict]):
        """Initialize with model population."""
        self.models_data = models_data
        self.models = self._prepare_models()
        self.n_models = len(self.models)
        
        logger.info(f"Initialized ConstrainedQualityOptimizer with {self.n_models} models")
    
    def _safe_get(self, model: Dict, key: str, default=None):
        """Safely extract numeric value."""
        val = model.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _prepare_models(self) -> List[Dict]:
        """Extract model data."""
        models = []
        
        for m in self.models_data:
            name = m.get("name", "")
            if not name:
                continue
            
            # Need Arena ELO for quality target
            arena_elo = self._safe_get(m, "arena_elo")
            if arena_elo is None or arena_elo <= 0:
                continue
            
            # Extract benchmarks
            benchmarks = {}
            for b in BENCHMARKS:
                val = self._safe_get(m, b, 0.0)
                benchmarks[b] = val if val else 0.0
            
            # Extract constraint features
            halluc = self._safe_get(m, "hallucination_rate", 15.0)
            cost = self._safe_get(m, "price_1m_blended", 0.01)
            if cost is None or cost <= 0:
                cost = 0.01  # Default
            
            models.append({
                "name": name,
                "benchmarks": benchmarks,
                "arena_elo": arena_elo,
                "hallucination_rate": halluc,
                "cost": cost,
            })
        
        return models
    
    def _get_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get data matrices."""
        n = len(self.models)
        m = len(BENCHMARKS)
        
        X = np.zeros((n, m))  # Benchmarks
        y = np.zeros(n)        # Arena ELO
        h = np.zeros(n)        # Hallucination
        c = np.zeros(n)        # Cost
        
        for i, model in enumerate(self.models):
            for j, bench in enumerate(BENCHMARKS):
                X[i, j] = model["benchmarks"].get(bench, 0.0)
            y[i] = model["arena_elo"]
            h[i] = model["hallucination_rate"]
            c[i] = model["cost"]
        
        return X, y, h, c
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 scale."""
        X_norm = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            if col.max() > col.min():
                X_norm[:, j] = (col - col.min()) / (col.max() - col.min())
            else:
                X_norm[:, j] = 0.5
        return X_norm
    
    def _compute_quality(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Compute correlation between weighted score and Arena ELO."""
        score = X @ w
        if score.std() < 1e-10:
            return 0.0
        return np.corrcoef(score, y)[0, 1]
    
    def _get_top_k_mask(
        self, 
        X: np.ndarray, 
        w: np.ndarray, 
        k: int
    ) -> np.ndarray:
        """Get mask for top-k models by weighted score."""
        scores = X @ w
        top_indices = np.argsort(scores)[-k:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_indices] = True
        return mask
    
    def _project_simplex(self, w: np.ndarray) -> np.ndarray:
        """Project weights onto probability simplex (sum=1, non-negative)."""
        # Clip negatives
        w = np.maximum(w, 0)
        # Normalize
        if w.sum() > 0:
            return w / w.sum()
        return np.ones_like(w) / len(w)
    
    def optimize(
        self,
        hallucination_max: float = 10.0,
        cost_max: float = 0.01,
        top_k: int = 5,
        max_iterations: int = 200,
        lr_weights: float = 0.1,
        lr_dual: float = 0.05,
        verbose: bool = False,
    ) -> ConstrainedQualityResult:
        """
        Optimize weights to maximize quality subject to constraints.
        
        Args:
            hallucination_max: Max avg hallucination for top-k models (%)
            cost_max: Max avg cost for top-k models ($/1M tokens)
            top_k: Number of top models to consider for constraints
            max_iterations: Maximum optimization iterations
            lr_weights: Learning rate for weight updates
            lr_dual: Learning rate for dual (lambda) updates
            verbose: Print progress
            
        Returns:
            ConstrainedQualityResult with optimal weights and shadow prices
        """
        # Get data
        X, y, h, c = self._get_matrices()
        X_norm = self._normalize(X)
        
        n_benchmarks = X.shape[1]
        
        # Initialize weights (equal)
        w = np.ones(n_benchmarks) / n_benchmarks
        
        # Initialize dual variables (shadow prices)
        lambda_h = 0.0  # Hallucination constraint
        lambda_c = 0.0  # Cost constraint
        
        converged = False
        
        for iteration in range(max_iterations):
            # Get top-k models under current weights
            mask = self._get_top_k_mask(X_norm, w, top_k)
            
            # Check constraint violations
            avg_halluc = h[mask].mean()
            avg_cost = c[mask].mean()
            
            violation_h = avg_halluc - hallucination_max
            violation_c = avg_cost - cost_max
            
            # Current quality
            quality = self._compute_quality(X_norm, y, w)
            
            if verbose and iteration % 20 == 0:
                logger.info(
                    f"Iter {iteration}: quality={quality:.4f}, "
                    f"h={avg_halluc:.1f}% (viol={violation_h:+.1f}), "
                    f"c=${avg_cost:.4f} (viol={violation_c:+.4f}), "
                    f"λ_h={lambda_h:.4f}, λ_c={lambda_c:.4f}"
                )
            
            # === Dual update (gradient ascent on Lagrangian) ===
            lambda_h = max(0, lambda_h + lr_dual * violation_h)
            lambda_c = max(0, lambda_c + lr_dual * violation_c * 100)  # Scale cost
            
            # === Primal update (gradient descent on weights) ===
            # Gradient of correlation w.r.t. weights (approximate)
            score = X_norm @ w
            y_centered = y - y.mean()
            score_centered = score - score.mean()
            
            # d(corr)/dw ≈ X' @ y_centered / (std(score) * std(y) * n)
            grad_quality = X_norm.T @ y_centered / (score.std() * y.std() * len(y) + 1e-10)
            
            # Gradient of constraints (which benchmarks lead to high-halluc/cost models)
            # Penalize weights that push high-halluc models to top
            grad_h = np.zeros(n_benchmarks)
            grad_c = np.zeros(n_benchmarks)
            
            # Approximate: benchmarks correlated with hallucination among top models
            if mask.sum() > 0:
                X_top = X_norm[mask]
                h_top = h[mask]
                c_top = c[mask]
                
                # Weighted contribution to constraints
                for j in range(n_benchmarks):
                    grad_h[j] = np.corrcoef(X_top[:, j], h_top)[0, 1] if X_top[:, j].std() > 0 else 0
                    grad_c[j] = np.corrcoef(X_top[:, j], c_top)[0, 1] if X_top[:, j].std() > 0 else 0
            
            # Combined gradient: maximize quality - lambda * constraint_violation
            grad = grad_quality - lambda_h * grad_h - lambda_c * grad_c
            
            # Update weights
            w_new = w + lr_weights * grad
            w_new = self._project_simplex(w_new)
            
            # Check convergence
            if np.linalg.norm(w_new - w) < 1e-5:
                converged = True
                w = w_new
                break
            
            w = w_new
        
        # Final evaluation
        mask = self._get_top_k_mask(X_norm, w, top_k)
        final_quality = self._compute_quality(X_norm, y, w)
        final_halluc = h[mask].mean()
        final_cost = c[mask].mean()
        
        # Build result
        weights_dict = {bench: float(w[j]) for j, bench in enumerate(BENCHMARKS)}
        
        return ConstrainedQualityResult(
            weights=weights_dict,
            correlation=float(final_quality),
            r_squared=float(final_quality ** 2),
            lambda_safety=float(lambda_h),
            lambda_cost=float(lambda_c),
            hallucination_max=hallucination_max,
            cost_max=cost_max,
            achieved_hallucination=float(final_halluc),
            achieved_cost=float(final_cost),
            converged=converged,
            iterations=iteration + 1,
            n_models=len(self.models),
            top_k=top_k,
        )
    
    def analyze_tradeoffs(
        self,
        hallucination_range: List[float] = None,
        cost_max: float = 0.01,
    ) -> Dict[str, List]:
        """
        Analyze how quality changes as hallucination constraint varies.
        
        Returns dict with 'hallucination', 'quality', 'lambda' lists.
        """
        if hallucination_range is None:
            hallucination_range = [5, 7, 10, 12, 15, 20]
        
        results = {
            "hallucination_max": [],
            "quality": [],
            "lambda_safety": [],
        }
        
        for h_max in hallucination_range:
            result = self.optimize(
                hallucination_max=h_max,
                cost_max=cost_max,
            )
            results["hallucination_max"].append(h_max)
            results["quality"].append(result.correlation)
            results["lambda_safety"].append(result.lambda_safety)
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def get_constrained_weights(
    models_data: List[Dict],
    hallucination_max: float = 10.0,
    cost_max: float = 0.01,
) -> Dict[str, float]:
    """
    Get weights that maximize quality subject to constraints.
    
    Returns:
        Dict mapping benchmark -> weight
    """
    optimizer = ConstrainedQualityOptimizer(models_data)
    result = optimizer.optimize(
        hallucination_max=hallucination_max,
        cost_max=cost_max,
    )
    return result.weights

