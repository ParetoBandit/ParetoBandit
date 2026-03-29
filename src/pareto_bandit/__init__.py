from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("paretobandit")
except PackageNotFoundError:
    __version__ = "0.1.1"

from .calibration import generate_warmup_priors, train_pca
from .exceptions import MissingCostError, NoEligibleModelsError, NoModelScoredError
from .family import compute_correlation_families, infer_model_family, tetrachoric_corr
from .feature_service import FeatureService
from .policy import DisjointLinUCBPolicy, calibrate_priors
from .router import BanditRouter
from .storage import EphemeralContextStore, SqliteContextStore
from .types import ExplorationRate, RegistrationConfig, RouterConfig, RoutingLog

__all__ = [
    "__version__",
    "BanditRouter", "ExplorationRate", "RouterConfig",
    "RegistrationConfig", "RoutingLog",
    "DisjointLinUCBPolicy", "calibrate_priors",
    "MissingCostError", "NoEligibleModelsError", "NoModelScoredError",
    "FeatureService",
    "SqliteContextStore", "EphemeralContextStore",
    "infer_model_family",
    "tetrachoric_corr", "compute_correlation_families",
    "train_pca", "generate_warmup_priors",
]
