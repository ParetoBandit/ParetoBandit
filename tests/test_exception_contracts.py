from __future__ import annotations

import numpy as np
import pytest

from bandit_gpt import BanditRouter, ExplorationRate, FeatureService
from tests.stress_utils import (
    DEFAULT_DIMENSION,
    make_context,
    precomputed_feature_service,
    synthetic_registry,
)


@pytest.mark.stress
def test_feature_service_exception_contracts():
    service = FeatureService.for_precomputed(DEFAULT_DIMENSION)

    with pytest.raises(ValueError, match="dimension"):
        service.extract_features(np.ones(DEFAULT_DIMENSION - 1))

    with pytest.raises(TypeError, match="Expected str or np.ndarray"):
        service.extract_features(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty or whitespace"):
        service.extract_features("   ")

    with pytest.raises(TypeError, match="not a string"):
        service.extract_features_batch(["ok", 5])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="empty or whitespace"):
        service.extract_features_batch(["ok", " "])


@pytest.mark.stress
def test_exploration_rate_invalid_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown exploration"):
        ExplorationRate.get("does-not-exist")


@pytest.mark.stress
def test_route_raises_value_error_when_constraints_eliminate_all_candidates():
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )
    with pytest.raises(ValueError, match="No candidates available"):
        router.bandit.select_arm(make_context(seed=9), candidates=["missing/model"])


@pytest.mark.stress
def test_explain_decision_unknown_model_raises_value_error():
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )
    with pytest.raises(ValueError, match="not found"):
        router.explain_decision("missing/model", make_context(seed=2))


@pytest.mark.stress
def test_load_state_dimension_mismatch_raises_value_error(tmp_path):
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )
    bad_state = tmp_path / "state_bad_dim.npz"
    np.savez_compressed(bad_state, _metadata_dim=999)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        router.bandit.load_state(bad_state)
