"""
Bayesian Latent Factor Model for Computing Composite Scores.

This module provides a reusable implementation of the Bayesian latent factor
model used to compute composite scores (CRS, CCS) from multiple benchmarks.

The model assumes that observed standardized benchmark scores arise from a
latent factor (theta) with benchmark-specific loadings and noise:

    z_{i,b} ~ Normal(alpha_b + lambda_b * theta_i, sigma_b)

where:
    - theta_i: latent composite score for model i
    - alpha_b: benchmark-specific intercept
    - lambda_b: benchmark-specific loading (factor weight)
    - sigma_b: benchmark-specific residual standard deviation

Usage:
    from llm_jury.analysis.latent_factor import (
        BenchmarkConfig,
        BenchmarkSuite,
        REASONING_BENCHMARKS,
        CODING_BENCHMARKS,
        extract_benchmark_matrix,
        prepare_long_data,
        fit_latent_factor_model,
        summarize_latent_scores,
        compute_weighted_zscore,
        transform_to_0_100,
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd


# ============================================================================
# Benchmark Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark.
    
    Attributes:
        name: Benchmark field name in the data.
        description: Human-readable description.
        scale: Scale factor for normalization.
        invert: If True, lower values are better (e.g., ranks).
        weight: Weight for weighted z-score method.
        is_auxiliary: If True, this benchmark is used for covariance-based
            imputation but not as a primary quality signal. Auxiliary benchmarks
            help "borrow strength" for models missing primary benchmarks.
    """
    name: str
    description: str
    scale: float = 1.0
    invert: bool = False
    weight: float = 1.0  # Weight for weighted z-score method
    is_auxiliary: bool = False  # Auxiliary benchmarks for covariance imputation
    
    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'scale': self.scale,
            'invert': self.invert,
            'weight': self.weight,
            'is_auxiliary': self.is_auxiliary,
        }
    
    @classmethod
    def from_dict(cls, name: str, data: Dict) -> 'BenchmarkConfig':
        return cls(
            name=name,
            description=data.get('description', name),
            scale=data.get('scale', 1.0),
            invert=data.get('invert', False),
            weight=data.get('weight', 1.0),
            is_auxiliary=data.get('is_auxiliary', False),
        )


@dataclass
class BenchmarkSuite:
    """A collection of benchmarks for computing a composite score."""
    name: str
    description: str
    benchmarks: Dict[str, BenchmarkConfig] = field(default_factory=dict)
    score_prefix: str = "score"
    
    def get_configs(self) -> Dict[str, Dict]:
        """Get benchmark configs as dict of dicts."""
        return {name: cfg.to_dict() for name, cfg in self.benchmarks.items()}
    
    def get_weights(self) -> Dict[str, float]:
        """Get normalized weights for weighted z-score method."""
        total = sum(cfg.weight for cfg in self.benchmarks.values())
        if total == 0:
            total = 1
        return {name: cfg.weight / total for name, cfg in self.benchmarks.items()}
    
    def add_benchmark(self, name: str, description: str = None, 
                      scale: float = 1.0, invert: bool = False,
                      weight: float = 1.0):
        """Add a primary benchmark to the suite."""
        self.benchmarks[name] = BenchmarkConfig(
            name=name,
            description=description or name,
            scale=scale,
            invert=invert,
            weight=weight,
            is_auxiliary=False,
        )
        return self
    
    def add_auxiliary_benchmark(self, name: str, description: str = None,
                                scale: float = 1.0, invert: bool = False,
                                weight: float = 0.05):
        """Add an auxiliary benchmark for covariance-based imputation.
        
        Auxiliary benchmarks help "borrow strength" from correlated metrics
        to estimate latent factors for models missing primary benchmarks.
        They typically have high coverage and strong correlation with the
        domain being measured.
        
        Args:
            name: Benchmark field name.
            description: Human-readable description.
            scale: Scale factor for normalization.
            invert: If True, lower values are better.
            weight: Weight (typically small, e.g., 0.05) for weighted z-score.
        """
        self.benchmarks[name] = BenchmarkConfig(
            name=name,
            description=description or name,
            scale=scale,
            invert=invert,
            weight=weight,
            is_auxiliary=True,
        )
        return self
    
    def get_primary_benchmarks(self) -> List[str]:
        """Get names of primary (non-auxiliary) benchmarks."""
        return [name for name, cfg in self.benchmarks.items() if not cfg.is_auxiliary]
    
    def get_auxiliary_benchmarks(self) -> List[str]:
        """Get names of auxiliary benchmarks."""
        return [name for name, cfg in self.benchmarks.items() if cfg.is_auxiliary]
    
    def remove_benchmark(self, name: str):
        """Remove a benchmark from the suite."""
        if name in self.benchmarks:
            del self.benchmarks[name]
        return self
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'score_prefix': self.score_prefix,
            'benchmarks': {name: cfg.to_dict() for name, cfg in self.benchmarks.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkSuite':
        """Create from dictionary."""
        suite = cls(
            name=data.get('name', 'custom'),
            description=data.get('description', 'Custom benchmark suite'),
            score_prefix=data.get('score_prefix', 'score'),
        )
        for name, cfg_data in data.get('benchmarks', {}).items():
            suite.benchmarks[name] = BenchmarkConfig.from_dict(name, cfg_data)
        return suite
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'BenchmarkSuite':
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, path: Union[str, Path]):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Default Benchmark Suites
# ============================================================================

def create_reasoning_suite() -> BenchmarkSuite:
    """Create the default reasoning benchmark suite (CRS)."""
    suite = BenchmarkSuite(
        name="reasoning",
        description="Composite Reasoning Score benchmarks",
        score_prefix="crs",
    )
    suite.add_benchmark('math_500', 'MATH-500: Mathematical problem solving', scale=100, weight=0.30)
    suite.add_benchmark('gpqa', 'GPQA: Graduate-level science questions', scale=1, weight=0.25)
    suite.add_benchmark('hle', "HLE: Humanity's Last Exam", scale=100, weight=0.20)
    suite.add_benchmark('aime', 'AIME: Competition mathematics', scale=1, weight=0.15)
    suite.add_benchmark('math_index', 'Math Index: AA composite', scale=1, weight=0.10)
    return suite


def create_coding_suite() -> BenchmarkSuite:
    """Create the default coding benchmark suite (CCS).
    
    Primary Benchmarks:
    - humaneval_score: HumanEval function-level code generation (pass@1)
    - livecodebench: LiveCodeBench competitive programming tasks
    - scicode: SciCode scientific computing benchmark
    
    Auxiliary Benchmarks (for covariance-based imputation):
    - intelligence_index: General intelligence metric (r=0.96 with livecodebench)
    
    Note: Arena ranks are NOT included to enable independent external validation.
    We utilize a hierarchical prior with covariance between latent factors to borrow
    statistical strength for models with missing modalities. The auxiliary benchmark 
    has 100% coverage and high correlation with coding.
    """
    suite = BenchmarkSuite(
        name="coding",
        description="Composite Coding Score benchmarks",
        score_prefix="ccs",
    )
    # Primary benchmarks (Arena rank removed for independent validation)
    suite.add_benchmark('humaneval_score', 'HumanEval: Code generation pass@1', scale=1, weight=0.40)
    suite.add_benchmark('livecodebench', 'LiveCodeBench: Real-world coding tasks', scale=100, weight=0.40)
    suite.add_benchmark('scicode', 'SciCode: Scientific computing benchmark', scale=100, weight=0.20)
    # Auxiliary benchmark for covariance-based imputation
    suite.add_auxiliary_benchmark('intelligence_index', 
                                  'Intelligence Index: General capability (r=0.96 with coding)',
                                  scale=1, weight=0.05)
    return suite


def create_factual_qa_suite() -> BenchmarkSuite:
    """Create the default factual QA benchmark suite (CFS - Composite Factual Score).
    
    Benchmarks:
    - mmlu_pro: MMLU-Pro massive multitask language understanding (enhanced version)
    - gpqa: Graduate-level science QA requiring domain expertise
    
    Note: Arena ranks are NOT included to enable independent external validation.
    This composite measures a model's ability to answer factual questions,
    retrieve knowledge, and provide accurate information across domains.
    """
    suite = BenchmarkSuite(
        name="factual_qa",
        description="Composite Factual QA Score benchmarks",
        score_prefix="cfs",
    )
    suite.add_benchmark('mmlu_pro', 'MMLU-Pro: Massive multitask language understanding', scale=1, weight=0.50)
    suite.add_benchmark('gpqa', 'GPQA: Graduate-level science QA', scale=1, weight=0.50)
    return suite


def create_summarization_suite() -> BenchmarkSuite:
    """Create the default summarization benchmark suite (CSS - Composite Summarization Score).
    
    Benchmarks:
    - summedits_score: SummEdits benchmark measuring summary quality across domains
    - hallucination_rate: Hallucination rate (inverted: lower rate = better)
    
    Note: Arena ranks are NOT included to enable independent external validation.
    This composite measures a model's ability to produce accurate, faithful
    summaries without hallucinating information.
    """
    suite = BenchmarkSuite(
        name="summarization",
        description="Composite Summarization Score benchmarks",
        score_prefix="css",
    )
    suite.add_benchmark('summedits_score', 'SummEdits: Summary quality across domains', scale=1, weight=0.50)
    suite.add_benchmark('hallucination_rate', 'Hallucination Rate: Factual accuracy (lower is better)', 
                       scale=1, invert=True, weight=0.50)
    return suite


# Pre-built default suites
REASONING_BENCHMARKS = create_reasoning_suite()
CODING_BENCHMARKS = create_coding_suite()
FACTUAL_QA_BENCHMARKS = create_factual_qa_suite()
SUMMARIZATION_BENCHMARKS = create_summarization_suite()


def parse_benchmark_args(
    benchmark_args: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    default_suite: Optional[BenchmarkSuite] = None,
) -> BenchmarkSuite:
    """
    Parse benchmark configuration from arguments.
    
    Args:
        benchmark_args: List of benchmark specs in format "field:scale:weight" 
                       or just "field" (uses defaults). Example: 
                       ["math_500:100:0.3", "gpqa:1:0.25"]
        config_path: Path to JSON config file.
        default_suite: Default suite to use if no args provided.
    
    Returns:
        BenchmarkSuite configured from arguments.
    """
    # If config file provided, load it
    if config_path:
        return BenchmarkSuite.from_json(config_path)
    
    # If no benchmark args, use default
    if not benchmark_args:
        return default_suite or BenchmarkSuite(
            name="custom",
            description="Custom benchmark suite",
            score_prefix="score",
        )
    
    # Parse benchmark specs
    suite = BenchmarkSuite(
        name="custom",
        description="Custom benchmark suite from CLI",
        score_prefix=default_suite.score_prefix if default_suite else "score",
    )
    
    for spec in benchmark_args:
        parts = spec.split(':')
        field_name = parts[0]
        
        # Try to get defaults from default suite if available
        defaults = {}
        if default_suite and field_name in default_suite.benchmarks:
            defaults = default_suite.benchmarks[field_name].to_dict()
        
        scale = float(parts[1]) if len(parts) > 1 else defaults.get('scale', 1.0)
        weight = float(parts[2]) if len(parts) > 2 else defaults.get('weight', 1.0)
        description = defaults.get('description', field_name)
        invert = defaults.get('invert', False)
        
        suite.add_benchmark(field_name, description, scale=scale, invert=invert, weight=weight)
    
    return suite


def add_benchmark_args(parser):
    """Add common benchmark arguments to an argument parser."""
    parser.add_argument(
        "--benchmarks", nargs="*", metavar="SPEC",
        help="Custom benchmarks in format 'field:scale:weight' (e.g., 'math_500:100:0.3'). "
             "If scale/weight omitted, uses defaults."
    )
    parser.add_argument(
        "--config", type=str, metavar="PATH",
        help="Path to JSON config file defining benchmark suite."
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true",
        help="List available default benchmarks and exit."
    )
    parser.add_argument(
        "--save-config", type=str, metavar="PATH",
        help="Save current benchmark config to JSON file and exit."
    )
    return parser


def extract_benchmark_matrix(
    models: List[Dict],
    benchmark_configs: Dict[str, Dict],
    min_benchmarks: int = 2,
    model_name_field: str = 'name',
    fallback_name_field: str = 'openrouter_id',
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Extract benchmark scores from models into DataFrames.
    
    Args:
        models: List of model dictionaries with benchmark scores.
        benchmark_configs: Dict mapping benchmark field names to config dicts.
            Each config should have 'scale' and optionally 'invert'.
        min_benchmarks: Minimum number of benchmarks required for inclusion.
        model_name_field: Field to use for model name.
        fallback_name_field: Fallback field if model_name_field is missing.
    
    Returns:
        df_scores: DataFrame with scaled (possibly inverted) scores.
        df_z: DataFrame with z-score standardized values.
        model_names: List of model names included.
        benchmark_names: List of benchmark field names.
    """
    benchmark_names = list(benchmark_configs.keys())
    
    rows = []
    model_names = []
    
    for model in models:
        model_name = model.get(model_name_field, model.get(fallback_name_field, 'unknown'))
        scores = {}
        valid_count = 0
        
        for bench in benchmark_names:
            raw = model.get(bench)
            if raw is not None:
                config = benchmark_configs[bench]
                scale = config.get('scale', 1.0)
                invert = config.get('invert', False)
                
                scaled = float(raw) * scale
                if invert:
                    scaled = -scaled
                
                scores[bench] = scaled
                valid_count += 1
            else:
                scores[bench] = np.nan
        
        if valid_count >= min_benchmarks:
            rows.append(scores)
            model_names.append(model_name)
    
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), [], benchmark_names
    
    df_scores = pd.DataFrame(rows, columns=benchmark_names)
    
    # Compute z-scores per benchmark
    df_z = df_scores.copy()
    for col in benchmark_names:
        col_data = df_scores[col]
        valid = col_data.dropna()
        if len(valid) > 1 and valid.std() > 0:
            df_z[col] = (col_data - valid.mean()) / valid.std()
        else:
            df_z[col] = np.nan
    
    return df_scores, df_z, model_names, benchmark_names


def prepare_long_data(
    df_z: pd.DataFrame,
    model_names: List[str],
    benchmark_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Convert wide-format standardized scores to long format for PyMC.
    
    Missing values are removed from the output.
    
    Args:
        df_z: DataFrame with z-score standardized values (models x benchmarks).
        model_names: List of model names.
        benchmark_names: List of benchmark field names.
    
    Returns:
        z_obs: 1D array of observed z-scores.
        idx_model: 1D array of model indices.
        idx_bench: 1D array of benchmark indices.
        n_models: Number of models.
        n_benchmarks: Number of benchmarks.
    """
    n_models = len(model_names)
    n_benchmarks = len(benchmark_names)
    
    z_values = df_z.to_numpy()
    idx_model_list = []
    idx_bench_list = []
    z_obs_list = []
    
    for i_model in range(n_models):
        for j_bench in range(n_benchmarks):
            val = z_values[i_model, j_bench]
            if not np.isnan(val):
                idx_model_list.append(i_model)
                idx_bench_list.append(j_bench)
                z_obs_list.append(val)
    
    z_obs = np.array(z_obs_list, dtype=float)
    idx_model = np.array(idx_model_list, dtype=int)
    idx_bench = np.array(idx_bench_list, dtype=int)
    
    return z_obs, idx_model, idx_bench, n_models, n_benchmarks


def fit_latent_factor_model(
    z_obs: np.ndarray,
    idx_model: np.ndarray,
    idx_bench: np.ndarray,
    n_models: int,
    n_benchmarks: int,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
    progressbar: bool = True,
) -> Any:
    """
    Fit a 1-factor Gaussian latent variable model using PyMC.
    
    Model specification:
        theta_i ~ Normal(0, 1)           # Latent factor per model
        alpha_b ~ Normal(0, 1)           # Benchmark intercept
        lambda_b ~ HalfNormal(0.7)       # Benchmark loading (positive)
        sigma_b ~ HalfNormal(1.0)        # Benchmark noise (positive)
        z_{i,b} ~ Normal(alpha_b + lambda_b * theta_i, sigma_b)
    
    Args:
        z_obs: 1D array of observed z-scores.
        idx_model: 1D array of model indices.
        idx_bench: 1D array of benchmark indices.
        n_models: Number of models.
        n_benchmarks: Number of benchmarks.
        draws: Number of posterior draws per chain.
        tune: Number of tuning steps.
        chains: Number of MCMC chains.
        target_accept: Target acceptance rate for NUTS sampler.
        random_seed: Random seed for reproducibility.
        progressbar: Whether to show progress bar.
    
    Returns:
        idata: ArviZ InferenceData with posterior samples.
    
    Raises:
        ImportError: If pymc is not installed.
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "PyMC is required for Bayesian inference. "
            "Install with: pip install pymc arviz"
        )
    
    coords = {
        "model": np.arange(n_models),
        "benchmark": np.arange(n_benchmarks),
        "obs_id": np.arange(len(z_obs)),
    }
    
    with pm.Model(coords=coords) as model:
        # Index variables
        model_idx = pm.Data("model_idx", idx_model, dims="obs_id")
        bench_idx = pm.Data("bench_idx", idx_bench, dims="obs_id")
        
        # Latent factor per model
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="model")
        
        # Benchmark-specific intercepts
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0, dims="benchmark")
        
        # Benchmark-specific loadings (positive for identifiability)
        lambda_ = pm.HalfNormal("lambda", sigma=0.7, dims="benchmark")
        
        # Benchmark-specific noise (positive)
        sigma = pm.HalfNormal("sigma", sigma=1.0, dims="benchmark")
        
        # Expected mean for each observed (model, benchmark) pair
        mu = alpha[bench_idx] + lambda_[bench_idx] * theta[model_idx]
        
        # Likelihood
        z = pm.Normal("z", mu=mu, sigma=sigma[bench_idx],
                      observed=z_obs, dims="obs_id")
        
        # Sample posterior
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
        )
    
    return idata


def summarize_latent_scores(
    idata: Any,
    model_names: List[str],
    score_name: str = "score",
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """
    Extract posterior summary of the latent factor (theta) per model.
    
    Args:
        idata: ArviZ InferenceData from fit_latent_factor_model.
        model_names: List of model names.
        score_name: Name prefix for the score columns.
        hdi_prob: Probability for highest density interval.
    
    Returns:
        DataFrame with columns:
            - model: Model name
            - {score_name}_mean: Posterior mean
            - {score_name}_sd: Posterior standard deviation
            - {score_name}_hdi_low: Lower HDI bound
            - {score_name}_hdi_high: Upper HDI bound
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError(
            "ArviZ is required for summarizing results. "
            "Install with: pip install arviz"
        )
    
    theta_post = idata.posterior["theta"]
    summary = az.summary(theta_post, hdi_prob=hdi_prob)
    
    hdi_low_col = f"hdi_{(1-hdi_prob)/2*100:.1f}%"
    hdi_high_col = f"hdi_{(1+hdi_prob)/2*100:.1f}%"
    
    df = pd.DataFrame({
        "model": model_names,
        f"{score_name}_mean": summary["mean"].values,
        f"{score_name}_sd": summary["sd"].values,
        f"{score_name}_hdi_low": summary[hdi_low_col].values,
        f"{score_name}_hdi_high": summary[hdi_high_col].values,
    })
    
    return df


def get_benchmark_diagnostics(
    idata: Any,
    benchmark_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Extract diagnostic information for benchmark parameters.
    
    Args:
        idata: ArviZ InferenceData from fit_latent_factor_model.
        benchmark_names: List of benchmark names.
    
    Returns:
        Dict with keys 'lambda', 'sigma', 'alpha', each containing
        a DataFrame with posterior summaries.
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError("ArviZ is required. Install with: pip install arviz")
    
    results = {}
    
    for param in ['lambda', 'sigma', 'alpha']:
        summary = az.summary(idata.posterior[param], hdi_prob=0.95)
        summary.index = benchmark_names
        results[param] = summary
    
    return results


def compute_weighted_zscore(
    df_z: pd.DataFrame,
    model_names: List[str],
    weights: Dict[str, float],
    score_name: str = "score",
    min_benchmarks: int = 2,
) -> pd.DataFrame:
    """
    Compute composite score using weighted average of z-scores.
    
    This is a simpler, faster alternative to the Bayesian approach.
    
    Args:
        df_z: DataFrame with z-score standardized values.
        model_names: List of model names.
        weights: Dict mapping benchmark names to weights.
        score_name: Name prefix for the score columns.
        min_benchmarks: Minimum benchmarks required for valid score.
    
    Returns:
        DataFrame with columns:
            - model: Model name
            - {score_name}_mean: Weighted average of z-scores
            - {score_name}_sd: Standard error estimate
            - {score_name}_hdi_low: Approximate 95% CI lower bound
            - {score_name}_hdi_high: Approximate 95% CI upper bound
            - n_benchmarks: Number of available benchmarks
    """
    results = []
    
    for i, model_name in enumerate(model_names):
        row = df_z.iloc[i]
        
        total_weight = 0
        weighted_sum = 0
        available = 0
        z_scores = []
        
        for bench, w in weights.items():
            if bench in df_z.columns:
                z_val = row[bench]
                if not pd.isna(z_val):
                    weighted_sum += w * z_val
                    total_weight += w
                    available += 1
                    z_scores.append(z_val)
        
        if total_weight > 0 and available >= min_benchmarks:
            score_mean = weighted_sum / total_weight
            score_sd = np.std(z_scores) / np.sqrt(available) if len(z_scores) > 1 else 0.5
            hdi_low = score_mean - 1.96 * score_sd
            hdi_high = score_mean + 1.96 * score_sd
        else:
            score_mean = np.nan
            score_sd = np.nan
            hdi_low = np.nan
            hdi_high = np.nan
        
        results.append({
            'model': model_name,
            f'{score_name}_mean': score_mean,
            f'{score_name}_sd': score_sd,
            f'{score_name}_hdi_low': hdi_low,
            f'{score_name}_hdi_high': hdi_high,
            'n_benchmarks': available,
        })
    
    return pd.DataFrame(results)


def transform_to_0_100(
    df_scores: pd.DataFrame,
    mean_col: str,
    output_col: Optional[str] = None,
    hdi_low_col: Optional[str] = None,
    hdi_high_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transform z-scores to 0-100 scale using min-max normalization.
    
    Args:
        df_scores: DataFrame with score columns.
        mean_col: Name of the mean score column.
        output_col: Name for the 0-100 transformed column (default: {mean_col}_100).
        hdi_low_col: Name of HDI low column to transform (optional).
        hdi_high_col: Name of HDI high column to transform (optional).
    
    Returns:
        DataFrame with added 0-100 scaled column(s).
    """
    df_out = df_scores.copy()
    
    valid_means = df_out[mean_col].dropna()
    if len(valid_means) == 0:
        return df_out
    
    min_val = valid_means.min()
    max_val = valid_means.max()
    
    if output_col is None:
        output_col = f"{mean_col.replace('_mean', '')}_100"
    
    if max_val - min_val < 1e-9:
        df_out[output_col] = 50.0
    else:
        df_out[output_col] = ((df_out[mean_col] - min_val) / (max_val - min_val)) * 100
        
        # Transform HDI bounds if provided
        if hdi_low_col and hdi_low_col in df_out.columns:
            df_out[f"{output_col}_hdi_low"] = ((df_out[hdi_low_col] - min_val) / (max_val - min_val)) * 100
        if hdi_high_col and hdi_high_col in df_out.columns:
            df_out[f"{output_col}_hdi_high"] = ((df_out[hdi_high_col] - min_val) / (max_val - min_val)) * 100
    
    return df_out


def update_models_cache(
    cache_data: dict,
    df_scores: pd.DataFrame,
    score_prefix: str,
    method: str = "bayesian",
    model_name_field: str = "name",
) -> Tuple[dict, int]:
    """
    Update models in cache with computed scores.
    
    Args:
        cache_data: The cache dictionary to update.
        df_scores: DataFrame with score columns.
        score_prefix: Prefix for score fields (e.g., 'crs', 'ccs').
        method: Method used ('bayesian' or 'weighted_zscore').
        model_name_field: Field to use for matching models.
    
    Returns:
        Tuple of (updated cache_data, count of updated models).
    """
    # Build lookup
    score_lookup = dict(zip(df_scores['model'], df_scores.to_dict('records')))
    
    models = cache_data.get('models', cache_data) if isinstance(cache_data, dict) else cache_data
    
    updated_count = 0
    for model in models:
        model_name = model.get(model_name_field, model.get('openrouter_id', ''))
        if model_name in score_lookup:
            record = score_lookup[model_name]
            
            mean_col = f"{score_prefix}_mean"
            sd_col = f"{score_prefix}_sd"
            score_100_col = f"{score_prefix}_100"
            
            if not pd.isna(record.get(mean_col)):
                model[score_prefix] = round(record[mean_col], 4)
                model[f"{score_prefix}_sd"] = round(record.get(sd_col, 0), 4)
                if score_100_col in record and not pd.isna(record[score_100_col]):
                    model[f"{score_prefix}_100"] = round(record[score_100_col], 1)
                model[f"{score_prefix}_method"] = method
                updated_count += 1
    
    return cache_data, updated_count
