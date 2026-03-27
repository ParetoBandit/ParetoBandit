from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("paretobandit")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .policy import DisjointLinUCBPolicy, calibrate_priors
from .types import RouterConfig, ExplorationRate, RegistrationConfig, RoutingLog
from .router import BanditRouter
from .exceptions import MissingCostError, NoEligibleModelsError, NoModelScoredError
from .family import infer_model_family, tetrachoric_corr, compute_correlation_families
from .feature_service import FeatureService
from .storage import SqliteContextStore, EphemeralContextStore
from .calibration import train_pca, generate_warmup_priors
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
