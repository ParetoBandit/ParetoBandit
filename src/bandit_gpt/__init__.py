from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("banditgpt")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .router import (
    BanditRouter, ExplorationRate, RouterConfig,
    MissingCostError, NoEligibleModelsError,
    infer_model_family, tetrachoric_corr, compute_correlation_families,
)
from .feature_service import FeatureService
from .calibration import train_pca, generate_warmup_priors
from .providers import (
    LLMClient, OpenRouterClient, OpenAIClient,
    AnthropicClient, GeminiClient, OllamaClient,
    MultiProviderClient,
)

__all__ = [
    "__version__",
    "BanditRouter", "ExplorationRate", "RouterConfig",
    "MissingCostError", "NoEligibleModelsError",
    "FeatureService",
    "infer_model_family",
    "tetrachoric_corr", "compute_correlation_families",
    "train_pca", "generate_warmup_priors",
    "LLMClient", "OpenRouterClient", "OpenAIClient",
    "AnthropicClient", "GeminiClient", "OllamaClient",
    "MultiProviderClient",
]
