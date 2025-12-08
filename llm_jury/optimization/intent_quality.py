"""
Intent-Specific Quality Scoring via Data-Driven Weights.

Weights are derived from regression, NOT magic numbers:
    
    Quality = β₀ + β₁×benchmark₁ + β₂×benchmark₂ + ... + ε
    
    Weights = normalize(β₁, β₂, ...) to sum to 1

Quality Targets (intent-specific):
- **coding**: CCS (`ccs_100` field, 0-100 scale) - Bayesian latent factor model 
  aggregating HumanEval, LiveCodeBench, SciCode, Arena Coding Rank (99% coverage)
- **reasoning**: CRS (`reasoning_score` field) - Bayesian latent factor model
  aggregating MATH-500, GPQA, HLE, AIME, Math Index (99% coverage)
- **factual_qa**: CFS (`cfs_100` field, 0-100 scale) - Bayesian latent factor model
  aggregating MMLU-Pro, GPQA, Arena Expert Rank (100% coverage)
- **summarization**: CSS (`css_100` field, 0-100 scale) - Bayesian latent factor model
  aggregating SummEdits, Hallucination Rate, Arena Longer Query Rank (62% coverage)
- **creative**: Arena Creative rank (`arena_rank_creative`, inverted - lower rank = better) (56% coverage)
- **general**: Calibrated proxy score (Intelligence Index → Arena scale, R²=0.47)

The CCS and CRS scores are computed using a principled Bayesian latent variable
model that handles missing data gracefully and learns benchmark importance from
data. See docs/COMPOSITE_CODING_SCORE.md and docs/COMPOSITE_REASONING_SCORE.md.

Key findings from our analysis:
- math_index is the best single predictor (r=0.45 with Arena ELO, p=0.003)
- All benchmarks are highly collinear (VIF > 10 for all)
- Intent doesn't change the math_index→quality relationship (no interaction)
- ~75% of performance is general capability, ~25% is specialization

Citation for MixEval:
    Ni et al. "MixEval: Deriving Wisdom of the Crowd from LLM Benchmark 
    Mixtures." NeurIPS 2024.

Usage:
    from llm_jury.optimization.intent_quality import IntentQualityScorer
    
    # Default: uses CCS for coding, CRS for reasoning, MixEval for others
    scorer = IntentQualityScorer(models_data)
    
    # Get data-derived weights for coding (uses CCS as quality target)
    weights = scorer.get_weights("coding")
    
    # Get data-derived weights for reasoning (uses CRS as quality target)
    weights = scorer.get_weights("reasoning")
    
    # Score a model
    score = scorer.score_model(model, intent="coding")
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Configuration
# =============================================================================

# All available benchmarks (ordered by coverage)
ALL_BENCHMARKS = [
    'intelligence_index',  # 98% coverage
    'reasoning_score',     # 99% coverage - CRS from Bayesian latent factor model
    'gpqa',                # 92% coverage, r=0.954 with intelligence_index
    'mmlu_pro',            # 91% coverage, r=0.861 with intelligence_index  
    'hle',                 # 90% coverage
    'livecodebench',       # 90% coverage
    'scicode',             # 90% coverage
    'math_index',          # 69% coverage
    'math_500',            # 57% coverage
    'aime',                # 51% coverage
]

# Fallback hierarchy for missing benchmarks
# If primary is missing, use fallback (based on correlation analysis)
BENCHMARK_FALLBACKS = {
    'intelligence_index': ['gpqa', 'mmlu_pro'],  # gpqa r=0.954, mmlu_pro r=0.861
    'math_index': ['math_500', 'aime'],
    'gpqa': ['intelligence_index', 'mmlu_pro'],
    'mmlu_pro': ['gpqa', 'intelligence_index'],
}

# Intent-specific benchmark subsets (domain knowledge: which benchmarks COULD matter)
# The regression will determine HOW MUCH each matters
# Using high-coverage benchmarks as primaries
INTENT_BENCHMARKS = {
    "coding": [
        'livecodebench',      # Direct coding measure (90%)
        'scicode',            # Scientific coding (90%)
        'math_index',         # Algorithmic thinking (69%)
        'gpqa',               # General reasoning (92%) - proxy for intelligence
    ],
    "reasoning": [
        'math_index',         # Mathematical reasoning (69%)
        'gpqa',               # Scientific reasoning (92%)
        'math_500',           # Hard math (57%)
        'aime',               # Competition math (51%)
        'hle',                # Expert-level reasoning (90%)
    ],
    "creative": [
        'gpqa',               # General capability proxy (92%)
        'hle',                # Language/nuance (90%)
        'mmlu_pro',           # Knowledge breadth (91%)
    ],
    "factual_qa": [
        'mmlu_pro',           # Knowledge breadth (91%)
        'gpqa',               # Domain expertise (92%)
        'hle',                # Language understanding (90%)
    ],
    # Note: "general" is a fallback intent - it uses the default quality target
    # (MixEval, Arena ELO, etc.) without domain-specific benchmark weighting
}

# =============================================================================
# Intent-Specific Quality Targets
# =============================================================================

# Map intents to their domain-specific quality targets
# - coding: CCS (Composite Coding Score) from Bayesian latent factor model
# - reasoning: CRS (Composite Reasoning Score) from Bayesian latent factor model
# - factual_qa: CFS (Composite Factual Score) from Bayesian latent factor model
# - summarization: CSS (Composite Summarization Score) from Bayesian latent factor model
# - creative: Arena Creative ranking (direct metric, not composite)
# - general: Calibrated proxy score (Intelligence Index → Arena scale via Theil-Sen regression)
INTENT_QUALITY_TARGETS = {
    "coding": "ccs_100",          # Composite Coding Score (HumanEval, LiveCodeBench, SciCode, Arena Coding Rank)
    "reasoning": "reasoning_score",  # CRS: Composite Reasoning Score (MATH-500, GPQA, HLE, AIME, Math Index)
    "factual_qa": "cfs_100",      # Composite Factual Score (MMLU-Pro, GPQA, Arena Expert Rank)
    "summarization": "css_100",   # Composite Summarization Score (SummEdits, Hallucination Rate, Arena Longer)
    "creative": "arena_rank_creative",  # LMArena Creative Writing rank (lower = better, inverted in scoring)
    "general": "general_quality",  # Calibrated proxy: Intelligence Index → Arena scale (R²=0.48, p<0.001)
}

# Metrics where lower values indicate better quality (e.g., ranks, error rates)
# These will be inverted when computing quality scores
INVERTED_METRICS = {
    "arena_rank_creative",  # Lower rank = better
    "arena_rank_coding",
    "arena_rank_math", 
    "arena_rank_expert",
    "arena_rank_longer",
    "arena_rank_overall",
    "hallucination_rate",   # Lower hallucination = better
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IntentWeights:
    """
    Data-derived weights for an intent.
    
    Weights are derived by regressing an intent-specific quality target on benchmarks:
    - coding: CCS (Composite Coding Score)
    - reasoning: CRS (Composite Reasoning Score)
    - others: MixEval or Arena ELO
    """
    intent: str
    weights: Dict[str, float]
    
    # Regression diagnostics
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    n_models: int
    
    # Per-benchmark info
    coefficients: Dict[str, float]  # Raw regression coefficients
    std_errors: Dict[str, float]    # Standard errors of coefficients
    t_statistics: Dict[str, float]  # t-statistics
    p_values: Dict[str, float]      # p-values for significance
    correlations: Dict[str, float]  # Individual correlations with quality target
    
    def summary(self) -> str:
        lines = [
            f"INTENT: {self.intent.upper()}",
            f"R² = {self.r_squared:.4f}, Adj R² = {self.adj_r_squared:.4f}",
            f"F-statistic = {self.f_statistic:.2f}, p = {self.f_pvalue:.4f}",
            f"Models: {self.n_models}",
            "",
            "Data-Derived Weights (with significance):",
            f"{'Benchmark':<18} {'Weight':>7} {'Coef':>9} {'SE':>8} {'t':>7} {'p-value':>9} {'Sig':>4}",
            "-" * 70,
        ]
        
        for bench in sorted(self.weights.keys(), key=lambda b: -self.weights[b]):
            w = self.weights[bench]
            coef = self.coefficients.get(bench, 0)
            se = self.std_errors.get(bench, 0)
            t = self.t_statistics.get(bench, 0)
            p = self.p_values.get(bench, 1)
            
            # Significance stars
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            elif p < 0.1:
                sig = "."
            else:
                sig = ""
            
            lines.append(f"{bench:<18} {w:>7.3f} {coef:>9.2f} {se:>8.2f} {t:>7.2f} {p:>9.4f} {sig:>4}")
        
        lines.extend([
            "",
            "Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1"
        ])
        
        return "\n".join(lines)


# =============================================================================
# Intent Quality Scorer
# =============================================================================

class IntentQualityScorer:
    """
    Score models by intent using data-derived weights.
    
    Weights are NOT magic numbers - they come from regressing
    a quality target on benchmark scores.
    
    Quality targets are intent-specific:
    - coding: CCS (Composite Coding Score) - Bayesian latent factor model
    - reasoning: CRS (Composite Reasoning Score) - Bayesian latent factor model
    - others: MixEval (r=0.96 with Arena ELO) or Arena ELO
    
    Key finding: math_index is the best single predictor (p=0.003).
    All benchmarks are highly collinear (VIF > 10).
    """
    
    # Supported general quality targets (in order of preference)
    # Used for intents without a domain-specific target
    GENERAL_QUALITY_TARGETS = [
        "mixeval_score",      # Best: r=0.96 with Arena, automated
        "arena_elo",          # Gold standard but limited coverage
        "intelligence_index", # Fallback: composite measure, high coverage
    ]
    
    def __init__(
        self, 
        models_data: List[Dict],
        default_quality_target: str = "auto",
        regularization: float = 1.0,
    ):
        """
        Initialize with model population.
        
        Args:
            models_data: List of model dicts with benchmarks
            default_quality_target: Default quality signal for intents without
                domain-specific targets:
                - "mixeval_score" (recommended, r=0.96 with user prefs)
                - "arena_elo" (real user prefs, limited coverage)
                - "intelligence_index" (high coverage fallback)
                - "auto" (pick best available)
                
                Note: "coding" always uses CCS and "reasoning" always uses CRS
                regardless of this setting.
            regularization: Ridge regression alpha (higher = more regularization)
        """
        self.models_data = models_data
        self.regularization = regularization
        
        # Determine default quality target for non-specialized intents
        self.default_quality_target = self._select_default_quality_target(default_quality_target)
        
        # Cache for prepared models by quality target
        self._models_cache: Dict[str, List[Dict]] = {}
        
        # Cache for computed weights
        self._weights_cache: Dict[str, IntentWeights] = {}
        
        logger.info(f"IntentQualityScorer initialized with {len(models_data)} models")
        logger.info(f"Default quality target: {self.default_quality_target}")
        logger.info(f"Intent-specific targets: coding→CCS, reasoning→CRS")
    
    def _select_default_quality_target(self, target: str) -> str:
        """Select default quality target for non-specialized intents."""
        if target != "auto":
            return target
        
        # Count coverage for each target
        for t in self.GENERAL_QUALITY_TARGETS:
            count = sum(1 for m in self.models_data 
                       if m.get(t) and float(m.get(t, 0) or 0) > 0)
            if count >= 20:  # Minimum for regression
                logger.info(f"Auto-selected default quality target: {t} ({count} models)")
                return t
        
        # Fallback
        logger.warning("No quality target with sufficient coverage, using intelligence_index")
        return "intelligence_index"
    
    def _get_quality_target_for_intent(self, intent: str) -> str:
        """
        Get the appropriate quality target for an intent.
        
        - coding: CCS (Composite Coding Score)
        - reasoning: CRS (Composite Reasoning Score)  
        - others: default quality target (MixEval, Arena ELO, etc.)
        """
        # Check for intent-specific target (use module-level constant)
        if intent in INTENT_QUALITY_TARGETS:
            target = INTENT_QUALITY_TARGETS[intent]
            # Verify the target has coverage
            # For inverted metrics (ranks), just check existence; for others check > 0
            is_inverted = target in INVERTED_METRICS
            if is_inverted:
                count = sum(1 for m in self.models_data if m.get(target) is not None)
            else:
                count = sum(1 for m in self.models_data 
                           if m.get(target) and float(m.get(target, 0) or 0) > 0)
            if count >= 10:  # Lower threshold for specialized scores
                return target
            else:
                logger.warning(
                    f"Intent-specific target '{target}' for '{intent}' has only {count} models, "
                    f"falling back to default target"
                )
        
        return self.default_quality_target
    
    def _safe_get(self, model: Dict, key: str, allow_zero: bool = False) -> Optional[float]:
        """Safely extract numeric value."""
        val = model.get(key)
        if val is None:
            return None
        try:
            v = float(val)
            if allow_zero:
                return v if v >= 0 else None
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None
    
    def _is_inverted_metric(self, metric: str) -> bool:
        """Check if a metric should be inverted (lower = better)."""
        return metric in INVERTED_METRICS
    
    def _get_with_fallback(self, model: Dict, key: str) -> Optional[float]:
        """Get benchmark value, using fallback if primary is missing."""
        val = self._safe_get(model, key)
        if val is not None:
            return val
        
        # Try fallbacks
        fallbacks = BENCHMARK_FALLBACKS.get(key, [])
        for fallback in fallbacks:
            val = self._safe_get(model, fallback)
            if val is not None:
                return val
        
        return None
    
    def _prepare_models(self, quality_target: str) -> List[Dict]:
        """
        Extract models with the specified quality target.
        
        Args:
            quality_target: The quality target field to use (e.g., 'ccs', 'crs', 'mixeval_score')
            
        Returns:
            List of model dicts with 'name', 'quality', and 'benchmarks' fields
        """
        # Check cache
        if quality_target in self._models_cache:
            return self._models_cache[quality_target]
        
        models = []
        is_inverted = self._is_inverted_metric(quality_target)
        
        for m in self.models_data:
            name = m.get("name", "")
            if not name:
                continue
            
            # Must have quality target (allow zero for inverted metrics like ranks)
            quality = self._safe_get(m, quality_target, allow_zero=is_inverted)
            if quality is None:
                continue
            
            # Invert if needed (for metrics where lower = better, like ranks)
            if is_inverted:
                quality = -quality
            
            # Extract all benchmarks
            benchmarks = {}
            for b in ALL_BENCHMARKS:
                val = self._safe_get(m, b)
                benchmarks[b] = val if val is not None else 0.0
            
            models.append({
                "name": name,
                "quality": quality,  # Generic quality field (negated if inverted metric)
                "benchmarks": benchmarks,
            })
        
        # Cache for reuse
        self._models_cache[quality_target] = models
        logger.debug(f"Prepared {len(models)} models with quality target '{quality_target}' (inverted={is_inverted})")
        
        return models
    
    def _fit_regression(
        self, 
        benchmarks: List[str],
        models: List[Dict],
    ) -> Dict:
        """
        Fit OLS regression: quality ~ benchmarks with full statistics.
        
        Args:
            benchmarks: List of benchmark field names to use as predictors
            models: List of model dicts (from _prepare_models) with 'quality' and 'benchmarks'
        
        Returns:
            Dict with coefficients, p_values, std_errors, t_stats, r_squared, etc.
        """
        # Build matrices
        X = []
        y = []
        
        for m in models:
            row = [m["benchmarks"].get(b, 0.0) for b in benchmarks]
            X.append(row)
            y.append(m["quality"])
        
        X = np.array(X)
        y = np.array(y)
        n, p = X.shape
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        # OLS: β = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            beta = XtX_inv @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        
        # Predictions and residuals
        y_pred = X_with_intercept @ beta
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        
        # Standard error of regression
        mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / max(n - 1, 1)
        
        # Standard errors of coefficients
        se_beta = np.sqrt(np.diag(XtX_inv) * mse)
        
        # t-statistics
        t_stats = beta / (se_beta + 1e-10)
        
        # p-values (two-tailed)
        df = n - p - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df)) if df > 0 else np.ones_like(t_stats)
        
        # F-statistic for overall model significance
        ms_reg = (ss_tot - ss_res) / p if p > 0 else 0
        ms_res = mse
        f_stat = ms_reg / ms_res if ms_res > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, df) if df > 0 and p > 0 else 1.0
        
        # Extract coefficients (skip intercept at index 0)
        coefficients = {b: float(beta[i+1]) for i, b in enumerate(benchmarks)}
        std_errors = {b: float(se_beta[i+1]) for i, b in enumerate(benchmarks)}
        t_statistics = {b: float(t_stats[i+1]) for i, b in enumerate(benchmarks)}
        p_vals = {b: float(p_values[i+1]) for i, b in enumerate(benchmarks)}
        
        # Per-benchmark correlations
        correlations = {}
        for j, b in enumerate(benchmarks):
            col = X[:, j]
            if col.std() > 0:
                correlations[b] = float(np.corrcoef(col, y)[0, 1])
            else:
                correlations[b] = 0.0
        
        return {
            "coefficients": coefficients,
            "std_errors": std_errors,
            "t_statistics": t_statistics,
            "p_values": p_vals,
            "correlations": correlations,
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "f_statistic": float(f_stat),
            "f_pvalue": float(f_pvalue),
        }
    
    def _coeffs_to_weights(self, coefficients: Dict[str, float]) -> Dict[str, float]:
        """
        Convert regression coefficients to normalized weights.
        
        Uses absolute values and normalizes to sum to 1.
        """
        # Use absolute values (negative coefficients still matter)
        abs_coeffs = {b: abs(c) for b, c in coefficients.items()}
        
        total = sum(abs_coeffs.values())
        if total > 0:
            return {b: c / total for b, c in abs_coeffs.items()}
        else:
            # Fallback: equal weights
            n = len(coefficients)
            return {b: 1.0 / n for b in coefficients}
    
    def get_weights(self, intent: str) -> IntentWeights:
        """
        Get data-derived weights for an intent.
        
        The quality target used for regression depends on the intent:
        - coding: CCS (Composite Coding Score)
        - reasoning: CRS (Composite Reasoning Score)
        - others: MixEval or Arena ELO (based on default_quality_target)
        
        Args:
            intent: One of 'coding', 'reasoning', 'creative', 'factual_qa', 'general'
            
        Returns:
            IntentWeights with regression-derived weights and p-values
        """
        # Check cache
        if intent in self._weights_cache:
            return self._weights_cache[intent]
        
        # Get intent-specific quality target
        quality_target = self._get_quality_target_for_intent(intent)
        
        # Prepare models with this quality target
        models = self._prepare_models(quality_target)
        
        if len(models) < 10:
            logger.warning(
                f"Only {len(models)} models have quality target '{quality_target}' for intent '{intent}'. "
                f"Results may be unreliable."
            )
        
        # Get relevant benchmarks
        benchmarks = INTENT_BENCHMARKS.get(intent, ALL_BENCHMARKS)
        
        # Fit regression with full statistics
        reg_results = self._fit_regression(benchmarks, models)
        
        # Convert to weights
        weights = self._coeffs_to_weights(reg_results["coefficients"])
        
        result = IntentWeights(
            intent=intent,
            weights=weights,
            r_squared=reg_results["r_squared"],
            adj_r_squared=reg_results["adj_r_squared"],
            f_statistic=reg_results["f_statistic"],
            f_pvalue=reg_results["f_pvalue"],
            n_models=len(models),
            coefficients=reg_results["coefficients"],
            std_errors=reg_results["std_errors"],
            t_statistics=reg_results["t_statistics"],
            p_values=reg_results["p_values"],
            correlations=reg_results["correlations"],
        )
        
        self._weights_cache[intent] = result
        
        logger.info(
            f"Intent '{intent}' weights computed using quality target '{quality_target}' "
            f"({len(models)} models, R²={reg_results['r_squared']:.3f})"
        )
        
        return result
    
    def get_all_weights(self) -> Dict[str, IntentWeights]:
        """Get weights for all intents."""
        return {
            intent: self.get_weights(intent)
            for intent in INTENT_BENCHMARKS.keys()
        }
    
    def score_model(
        self, 
        model_data: Dict, 
        intent: str,
    ) -> float:
        """
        Score a model for a specific intent.
        
        Args:
            model_data: Model dict with benchmark scores
            intent: Intent category
            
        Returns:
            Quality score (weighted sum of benchmarks)
        """
        weights = self.get_weights(intent).weights
        
        score = 0.0
        for bench, weight in weights.items():
            val = self._safe_get(model_data, bench)
            if val is not None:
                score += weight * val
        
        return score
    
    def rank_models(
        self, 
        intent: str, 
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rank all models by intent-specific quality.
        
        Returns:
            List of (model_name, score) tuples, sorted descending
        """
        scores = []
        for m in self.models_data:
            name = m.get("name", "")
            if name:
                score = self.score_model(m, intent)
                scores.append((name, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_n]


# =============================================================================
# Convenience Functions
# =============================================================================

def get_intent_weights(
    models_data: List[Dict],
    intent: str,
) -> Dict[str, float]:
    """
    Get data-derived weights for an intent.
    
    Weights are derived by regressing an intent-specific quality target on benchmarks:
    - coding: CCS (Composite Coding Score)
    - reasoning: CRS (Composite Reasoning Score)
    - others: MixEval or Arena ELO
    """
    scorer = IntentQualityScorer(models_data)
    return scorer.get_weights(intent).weights


def get_all_intent_weights(
    models_data: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Get data-derived weights for all intents.
    
    Each intent uses its appropriate quality target:
    - coding: CCS (Composite Coding Score)
    - reasoning: CRS (Composite Reasoning Score)
    - others: MixEval or Arena ELO
    """
    scorer = IntentQualityScorer(models_data)
    return {
        intent: result.weights
        for intent, result in scorer.get_all_weights().items()
    }

