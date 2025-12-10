"""
Correlation-Based Weight Optimization.

Derives benchmark weights that maximize correlation with actual quality
signals (Arena ELO, user preferences, etc.).

Key Insight:
    Instead of manually choosing weights, we find weights that best predict
    real-world quality as measured by Arena ELO (aggregated user preferences).

Methods:
1. Ridge Regression: Find weights minimizing MSE to Arena ELO
2. Correlation Maximization: Find weights maximizing Pearson correlation
3. Intent-Specific: Different weights per intent based on correlation

Usage:
    from llm_jury.optimization import CorrelationWeightOptimizer
    
    optimizer = CorrelationWeightOptimizer(models_data)
    
    # Get weights that predict Arena ELO
    result = optimizer.fit_weights(target="arena_elo")
    print(result.weights)
    # {'livecodebench': 0.35, 'intelligence_index': 0.28, ...}
    
    # Get intent-specific weights
    coding_weights = optimizer.fit_weights_for_intent("coding", target="arena_elo")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmarks Configuration
# =============================================================================

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

# Quality signals we can optimize against
QUALITY_SIGNALS = [
    'arena_elo',           # Chatbot Arena (user preferences)
    'quality_index',       # Artificial Analysis quality index
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CorrelationResult:
    """Result from correlation-based weight optimization."""
    
    # Optimal weights
    weights: Dict[str, float]
    
    # Quality signal used
    target: str
    
    # Correlation achieved
    correlation: float
    r_squared: float
    
    # Per-benchmark correlations (for insight)
    benchmark_correlations: Dict[str, float]
    
    # Number of models used
    n_models: int
    
    # Optional: intent-specific
    intent: Optional[str] = None
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"CORRELATION-BASED WEIGHTS",
            "=" * 50,
            f"Target: {self.target}",
            f"Models used: {self.n_models}",
            f"Correlation: r = {self.correlation:.4f}",
            f"R-squared: {self.r_squared:.4f}",
            "",
            "Optimal Weights (top 5):",
        ]
        
        sorted_weights = sorted(self.weights.items(), key=lambda x: -x[1])[:5]
        for name, weight in sorted_weights:
            corr = self.benchmark_correlations.get(name, 0)
            lines.append(f"  {name:<20} w={weight:.3f}  (r={corr:.3f})")
        
        return "\n".join(lines)
    
    def get_paper_statement(self) -> str:
        """Generate paper-ready statement."""
        top_weights = sorted(self.weights.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join([f"{b} ({w:.0%})" for b, w in top_weights])
        
        return (
            f"Weights derived via correlation maximization with {self.target} "
            f"achieve r={self.correlation:.3f} (R²={self.r_squared:.3f}). "
            f"Top predictors: {top_str}."
        )


# =============================================================================
# Correlation Weight Optimizer
# =============================================================================

class CorrelationWeightOptimizer:
    """
    Derives benchmark weights by maximizing correlation with quality signals.
    
    This finds weights that best predict real-world quality (Arena ELO)
    based on benchmark scores, rather than using arbitrary manual weights.
    """
    
    def __init__(self, models_data: List[Dict]):
        """
        Initialize with model population.
        
        Args:
            models_data: List of model dicts with benchmark scores and quality signals
        """
        self.models_data = models_data
        self.models = self._prepare_models()
        self.n_models = len(self.models)
        
        logger.info(f"Initialized CorrelationWeightOptimizer with {self.n_models} models")
    
    def _safe_get(self, model: Dict, key: str, default: float = None) -> Optional[float]:
        """Safely extract numeric value."""
        val = model.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _prepare_models(self) -> List[Dict]:
        """Extract relevant data from models."""
        models = []
        
        for m in self.models_data:
            name = m.get("name", "")
            if not name:
                continue
            
            # Extract benchmarks
            benchmarks = {}
            has_any_benchmark = False
            for b in BENCHMARKS:
                val = self._safe_get(m, b)
                if val is not None and val > 0:
                    benchmarks[b] = val
                    has_any_benchmark = True
            
            if not has_any_benchmark:
                continue
            
            # Extract quality signals
            signals = {}
            for s in QUALITY_SIGNALS:
                val = self._safe_get(m, s)
                if val is not None and val > 0:
                    signals[s] = val
            
            # Extract intent-relevant info
            provider = m.get("provider", "")
            
            models.append({
                "name": name,
                "benchmarks": benchmarks,
                "signals": signals,
                "provider": provider,
            })
        
        return models
    
    def _get_benchmark_matrix(
        self, 
        target: str,
        benchmarks: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Get benchmark matrix X and target vector y.
        
        Returns:
            X: (n_models, n_benchmarks) matrix
            y: (n_models,) target vector
            model_names: List of model names
            benchmark_names: List of benchmark names
        """
        if benchmarks is None:
            benchmarks = BENCHMARKS
        
        # Filter to models with target signal
        valid_models = [m for m in self.models if target in m["signals"]]
        
        if len(valid_models) < 10:
            raise ValueError(f"Not enough models with {target} signal: {len(valid_models)}")
        
        # Build matrices
        X = []
        y = []
        model_names = []
        
        for m in valid_models:
            row = []
            for b in benchmarks:
                val = m["benchmarks"].get(b, 0.0)
                row.append(val)
            
            X.append(row)
            y.append(m["signals"][target])
            model_names.append(m["name"])
        
        return np.array(X), np.array(y), model_names, benchmarks
    
    def _normalize_benchmarks(self, X: np.ndarray) -> np.ndarray:
        """Normalize benchmarks to 0-1 scale."""
        X_norm = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                X_norm[:, j] = (col - col_min) / (col_max - col_min)
            else:
                X_norm[:, j] = 0.5
        return X_norm
    
    def fit_weights(
        self,
        target: str = "arena_elo",
        method: str = "ridge",
        alpha: float = 1.0,
        benchmarks: Optional[List[str]] = None,
    ) -> CorrelationResult:
        """
        Fit weights that maximize correlation with target quality signal.
        
        Args:
            target: Quality signal to optimize for ('arena_elo', 'quality_index')
            method: 'ridge' (regularized regression) or 'ols' (ordinary least squares)
            alpha: Regularization strength for ridge regression
            benchmarks: Subset of benchmarks to use (default: all)
            
        Returns:
            CorrelationResult with optimal weights
        """
        # Get data
        X, y, model_names, bench_names = self._get_benchmark_matrix(target, benchmarks)
        n_models, n_benchmarks = X.shape
        
        logger.info(f"Fitting weights on {n_models} models with {n_benchmarks} benchmarks")
        
        # Normalize benchmarks to comparable scales
        X_norm = self._normalize_benchmarks(X)
        
        # Standardize target for regression
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std
        
        # Fit regression
        if method == "ridge":
            # Ridge regression: (X'X + αI)^-1 X'y
            XtX = X_norm.T @ X_norm
            Xty = X_norm.T @ y_norm
            I = np.eye(n_benchmarks)
            coeffs = np.linalg.solve(XtX + alpha * I, Xty)
        else:
            # OLS: (X'X)^-1 X'y
            coeffs, _, _, _ = np.linalg.lstsq(X_norm, y_norm, rcond=None)
        
        # Convert to non-negative weights that sum to 1
        # Use softmax-like transformation
        coeffs_pos = np.maximum(coeffs, 0)  # ReLU
        if coeffs_pos.sum() > 0:
            weights = coeffs_pos / coeffs_pos.sum()
        else:
            # Fallback: equal weights
            weights = np.ones(n_benchmarks) / n_benchmarks
        
        # Calculate correlation
        y_pred = X_norm @ weights
        correlation = np.corrcoef(y_pred, y)[0, 1]
        r_squared = correlation ** 2
        
        # Per-benchmark correlations
        bench_corrs = {}
        for j, bench in enumerate(bench_names):
            col = X_norm[:, j]
            if col.std() > 0:
                bench_corrs[bench] = np.corrcoef(col, y)[0, 1]
            else:
                bench_corrs[bench] = 0.0
        
        # Build result
        weights_dict = {bench: float(w) for bench, w in zip(bench_names, weights)}
        
        return CorrelationResult(
            weights=weights_dict,
            target=target,
            correlation=float(correlation),
            r_squared=float(r_squared),
            benchmark_correlations=bench_corrs,
            n_models=n_models,
        )
    
    def fit_weights_for_intent(
        self,
        intent: str,
        target: str = "arena_elo",
        method: str = "ridge",
        alpha: float = 1.0,
    ) -> CorrelationResult:
        """
        Fit intent-specific weights using relevant benchmarks.
        
        Args:
            intent: Intent category ('coding', 'reasoning', 'creative', etc.)
            target: Quality signal to optimize for
            method: Regression method
            alpha: Regularization strength
            
        Returns:
            CorrelationResult with intent-specific weights
        """
        # Intent-specific benchmark subsets
        intent_benchmarks = {
            "coding": ['livecodebench', 'scicode', 'intelligence_index', 'math_index'],
            "reasoning": ['math_index', 'math_500', 'aime', 'gpqa', 'intelligence_index'],
            "creative": ['intelligence_index', 'mmlu_pro', 'hle'],
            "factual_qa": ['mmlu_pro', 'gpqa', 'intelligence_index', 'hle'],
            "general": BENCHMARKS,  # Use all
        }
        
        benchmarks = intent_benchmarks.get(intent, BENCHMARKS)
        
        result = self.fit_weights(
            target=target,
            method=method,
            alpha=alpha,
            benchmarks=benchmarks,
        )
        result.intent = intent
        
        return result
    
    def analyze_benchmark_importance(
        self,
        target: str = "arena_elo",
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which benchmarks are most important for predicting quality.
        
        Returns:
            Dict with 'correlation', 'weight', 'rank' for each benchmark
        """
        result = self.fit_weights(target=target)
        
        analysis = {}
        for bench in BENCHMARKS:
            corr = result.benchmark_correlations.get(bench, 0)
            weight = result.weights.get(bench, 0)
            
            analysis[bench] = {
                "correlation": corr,
                "weight": weight,
                "importance_score": (abs(corr) + weight) / 2,
            }
        
        # Add ranks
        sorted_by_importance = sorted(
            analysis.keys(), 
            key=lambda b: -analysis[b]["importance_score"]
        )
        for rank, bench in enumerate(sorted_by_importance, 1):
            analysis[bench]["rank"] = rank
        
        return analysis


# =============================================================================
# Convenience Functions
# =============================================================================

def get_correlation_weights(
    models_data: List[Dict],
    target: str = "arena_elo",
) -> Dict[str, float]:
    """
    Get benchmark weights that maximize correlation with quality signal.
    
    Args:
        models_data: Model population data
        target: Quality signal to optimize for
        
    Returns:
        Dict mapping benchmark -> weight
    """
    optimizer = CorrelationWeightOptimizer(models_data)
    result = optimizer.fit_weights(target=target)
    return result.weights


def get_intent_correlation_weights(
    models_data: List[Dict],
    target: str = "arena_elo",
) -> Dict[str, Dict[str, float]]:
    """
    Get intent-specific weights based on correlation with quality.
    
    Returns:
        Dict mapping intent -> {benchmark: weight}
    """
    optimizer = CorrelationWeightOptimizer(models_data)
    
    intents = ["coding", "reasoning", "creative", "factual_qa", "general"]
    
    return {
        intent: optimizer.fit_weights_for_intent(intent, target=target).weights
        for intent in intents
    }

