# Production imports
from .router import BanditRouter, ExplorationRate, RouterConfig
from .feature_service import FeatureService

# Note: RouterConfig is the production @dataclass from router.py
# Legacy Pydantic config is in config_legacy.py (deprecated)

try:
    from .cluster_detector import ClusterDetector
    __all__ = ["BanditRouter", "ExplorationRate", "RouterConfig", "FeatureService", "ClusterDetector"]
except ImportError:
    __all__ = ["BanditRouter", "ExplorationRate", "RouterConfig", "FeatureService"]


# DEPRECATED: Legacy BanditGPT support for backward compatibility
def __getattr__(name):
    """
    Provide deprecated access to legacy BanditGPT class.
    
    This maintains backward compatibility while warning users to migrate.
    Will be removed in v2.0.
    """
    if name == "BanditGPT":
        import warnings
        warnings.warn(
            "BanditGPT from bandit_gpt.core is deprecated and will be removed in v2.0. "
            "Please migrate to BanditRouter:\n\n"
            "  OLD: from bandit_gpt.core import BanditGPT\n"
            "       router = BanditGPT()\n\n"
            "  NEW: from bandit_gpt import BanditRouter\n"
            "       router = BanditRouter.create(priors='warmup')\n"
            "       model, log = router.route(prompt, profile='auto')\n\n"
            "See migration guide at: https://github.com/atabernermiller/banditGPT/wiki/Migration-v2",
            DeprecationWarning,
            stacklevel=2
        )
        from .core import BanditGPT
        return BanditGPT
    raise AttributeError(f"module 'bandit_gpt' has no attribute '{name}'")
