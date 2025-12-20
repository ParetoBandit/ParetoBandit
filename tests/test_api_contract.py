import importlib


def test_banditgpt_public_api_contract():
    mod = importlib.import_module("banditgpt")
    expected = {
        "__version__",
        "QualityCostPredictor",
        "TieredGrader",
        "OpenRouterTeacherVerifier",
        "HardPromptHeuristics",
        "UnsafePythonSubprocessVerifier",
        "PriorManager",
        "Settings",
        "load_settings",
        "configure_logging",
        "run_demo",
        "BanditRouter",
        "DisjointLinUCBPolicy",
        "RoutingLog",
    }
    assert set(mod.__all__) == expected


def test_banditgpt_core_public_api_contract():
    core = importlib.import_module("banditgpt.core")
    expected = {
        # Judge abstraction
        "Judge",
        "JudgeWithComplexity",
        "PriorConfig",
        "PriorManager",
        "create_soft_judge",
        "create_tiered_judge",
        "create_custom_judge",
        # Priors + manifest
        "ensure_priors",
        "PriorDownloadError",
        "PriorsManifest",
        "PriorIntegrityError",
        "PriorFileInfo",
        "load_priors_manifest",
        # Settings
        "Settings",
        "load_settings",
        "configure_logging",
        # Graders
        # Settings is exported; ensure presence
        "TieredGrader",
        "OpenRouterTeacherVerifier",
        "HardPromptHeuristics",
        "UnsafePythonSubprocessVerifier",
        "QualityCostPredictor",
        "RunningZScoreNormalizer",
        "LogitReward",
        "get_device",
        # Router + policies
        "BanditRouter",
        "DisjointLinUCBPolicy",
        "ExplorationRate",
        "OptimizationProfile",
        "SharedCovarianceLinUCBPolicy",
        "RoutingLog",
        "build_registry_from_models_cache",
        "build_cost_proportional_priors",
    }
    assert set(core.__all__) == expected
