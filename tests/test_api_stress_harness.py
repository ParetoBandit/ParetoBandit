from __future__ import annotations

import importlib

import numpy as np
import pytest

from pareto_bandit import BanditRouter, ExplorationRate, FeatureService

from tests.stress_utils import (
    DEFAULT_DIMENSION,
    make_context,
    precomputed_feature_service,
    synthetic_registry,
)


@pytest.mark.stress
def test_public_exports_resolve_and_are_callable():
    module = importlib.import_module("pareto_bandit")
    expected = {
        "BanditRouter",
        "ExplorationRate",
        "RouterConfig",
        "FeatureService",
        "infer_model_family",
        "tetrachoric_corr",
        "compute_correlation_families",
    }
    assert expected.issubset(set(module.__all__))
    for symbol in expected:
        assert hasattr(module, symbol)


@pytest.mark.stress
def test_router_factory_and_core_methods_with_precomputed_features():
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )
    from pareto_bandit.config import BEST_K3_HPARAMS
    assert pytest.approx(BEST_K3_HPARAMS["alpha"]) == router.bandit.alpha

    context = make_context(seed=7)
    model, log = router.route(
        context,
        max_cost=0.01,
        max_latency=0.6,
        quality_floor={"hle": 0.4},
        input_tokens=256,
        output_tokens=128,
    )
    assert model in router.registry
    assert log.selected_model == model
    assert log.context_vector is not None

    # Request-id based feedback loop
    router.process_feedback(log.request_id, reward=0.9)

    # Direct update should also accept precomputed vector
    before = router.bandit.b[model].copy()
    router.update(model, context, reward=0.8, weight=1.5)
    assert not np.allclose(before, router.bandit.b[model])


@pytest.mark.stress
def test_user_adjusted_constraints_are_applied_consistently():
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )
    context = make_context(seed=11)

    # Tight cost cap should force cheapest tier.
    model_low_cost, _ = router.route(context, max_cost=0.0015, input_tokens=200, output_tokens=150)
    chosen_cost = router.registry[model_low_cost]["input_cost_per_m"]
    assert chosen_cost <= 0.12

    # Tight latency cap should exclude slow models.
    model_fast, _ = router.route(context, max_latency=0.25)
    chosen_latency = router.registry[model_fast]["time_to_first_token_seconds"]
    assert chosen_latency <= 0.25

    # Quality floor should avoid low-HLE options.
    model_quality, _ = router.route(context, quality_floor={"hle": 0.75})
    chosen_hle = router.registry[model_quality].get("hle", 0.0)
    assert chosen_hle >= 0.75


@pytest.mark.stress
def test_feature_service_precomputed_paths():
    service = FeatureService.for_precomputed(dimension=DEFAULT_DIMENSION)
    vector = make_context(seed=2)

    extracted = service.extract_features(vector)
    assert extracted.shape == (DEFAULT_DIMENSION,)

    assert service.get_dimension() == DEFAULT_DIMENSION
    names = service.get_feature_names()
    assert names[-1] == "bias"


@pytest.mark.stress
def test_exploration_rate_presets_and_numeric_passthrough():
    assert ExplorationRate.get("static") == 0.0
    assert ExplorationRate.get("safe") == 0.1
    assert ExplorationRate.get("balanced") == 1.0
    assert ExplorationRate.get("aggressive") == 2.0
    assert ExplorationRate.get(0.123) == pytest.approx(0.123)
