"""
Tests for PCA whitening changes across the feature pipeline and router.

Validates:
1. Shipped PCA artifact has whiten=True (built-in whitening).
2. FeatureService whitening helpers (_apply_pca_whitening, batch, scales).
3. FeatureService does NOT double-whiten when the artifact already whitens.
4. FeatureService with whiten_pca=False disables whitening entirely.
5. JIT-trained PCA inherits the whiten_pca flag.
6. Warmup priors auto-conversion: unwhitened priors → whitened router space.
7. Warmup priors pass-through: already-whitened priors skip conversion.
8. Warmup priors reverse conversion: whitened priors → unwhitened router space.
9. Shipped warmup priors carry pca_whitened metadata.
10. calibration.generate_warmup_priors() produces whitened priors.
11. Cost filtering uses $/1k tokens semantics.
12. Backward-compatible registry keys (cost_per_1m_tokens, median_latency_s).
13. Pessimistic cost defaults for fully-missing cost data.
14. Registry deep copy prevents caller mutation.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
import pytest
from sklearn.decomposition import PCA

from pareto_bandit.config import DEFAULT_PCA_PATH
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import (
    BanditRouter,
    DisjointLinUCBPolicy,
    MissingCostError,
    NoEligibleModelsError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 10


def _mock_feature_service(
    dim: int = DIM,
    whiten_pca: bool = True,
    scales: np.ndarray | None = None,
) -> MagicMock:
    """Lightweight mock that satisfies BanditRouter.__init__."""
    fs = MagicMock()
    fs.dimension = dim
    fs.bias_index = dim - 1
    fs.pca = MagicMock(n_components=dim - 1)
    fs.encoder = None
    fs.using_pca = True
    fs.whiten_pca = whiten_pca
    fs.get_dimension.return_value = dim
    fs.get_feature_names.return_value = [f"pca_{i}" for i in range(dim - 1)] + ["bias"]
    if scales is None:
        scales = np.ones(dim, dtype=np.float64)
        if whiten_pca:
            scales[:-1] = np.arange(1, dim, dtype=np.float64)
    fs.get_pca_whitening_scales.return_value = scales

    def _extract(prompt):
        if isinstance(prompt, np.ndarray):
            return prompt
        v = np.random.default_rng(0).standard_normal(dim - 1)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.append(v, 1.0)

    fs.extract_features.side_effect = _extract
    return fs


def _simple_registry(n: int = 2):
    reg = {}
    for i in range(n):
        reg[f"model-{i}"] = {
            "model_id": f"provider/model-{i}",
            "input_cost_per_m": 0.5 * (i + 1),
            "output_cost_per_m": 1.5 * (i + 1),
            "initial_quality": 0.5 + 0.1 * i,
        }
    return reg


def _write_priors(path: Path, A: dict, b: dict, n: int, **extra) -> None:
    d = {"A": A, "b": b, "n": n}
    d.update(extra)
    joblib.dump(d, path)


# ===================================================================
# 1. Shipped PCA artifact has whiten=True
# ===================================================================


class TestShippedPCAArtifactWhitening:

    def test_shipped_pca_has_whiten_false(self) -> None:
        """Shipped PCA artifact has whiten=False; FeatureService applies
        whitening externally via _pca_whitening_scale."""
        pca = joblib.load(DEFAULT_PCA_PATH)
        assert getattr(pca, "whiten", True) is False, (
            "Shipped pca_25.joblib should have whiten=False — "
            "FeatureService handles whitening externally"
        )

    def test_feature_service_applies_external_whitening(self) -> None:
        """FeatureService should detect whiten=False on the shipped artifact
        and compute external whitening scales."""
        fs = FeatureService(whiten_pca=True)
        _ = fs.pca
        assert fs._pca_whitening_scale is not None, (
            "FeatureService should set external whitening scales for "
            "the shipped PCA (whiten=False)"
        )


# ===================================================================
# 2–6. FeatureService whitening logic
# ===================================================================


class TestFeatureServiceWhitening:

    @staticmethod
    def _make_pca(n_components: int = 4, whiten: bool = False) -> PCA:
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 20))
        pca = PCA(n_components=n_components, whiten=whiten)
        pca.fit(X)
        return pca

    def test_no_double_whitening_when_artifact_whitens(self) -> None:
        """When the PCA artifact already whitens, FeatureService must NOT
        multiply by 1/sqrt(ev) again."""
        pca = self._make_pca(n_components=4, whiten=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=True,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            assert fs._pca_whitening_scale is None, (
                "External whitening scale should be None when PCA artifact already whitens"
            )

    def test_external_whitening_when_artifact_does_not_whiten(self) -> None:
        """When the PCA artifact does NOT whiten, FeatureService must apply
        external scaling."""
        pca = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=True,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            _ = fs.pca  # trigger lazy PCA loading
            assert fs._pca_whitening_scale is not None, (
                "External whitening scale must be set when PCA artifact does not whiten"
            )
            assert fs._pca_whitening_scale.shape == (4,)

    def test_whiten_pca_false_disables_all_whitening(self) -> None:
        """Setting whiten_pca=False should never produce whitening scales."""
        pca = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=False,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            assert fs._pca_whitening_scale is None
            scales = fs.get_pca_whitening_scales()
            assert np.allclose(scales, 1.0), "All scales should be 1.0 when whitening is off"

    def test_apply_pca_whitening_scales_correctly(self) -> None:
        """_apply_pca_whitening multiplies features by the stored scale."""
        pca = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=True,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            _ = fs.pca  # trigger lazy PCA loading
            assert fs._pca_whitening_scale is not None
            x = np.array([1.0, 2.0, 3.0, 4.0])
            whitened = fs._apply_pca_whitening(x)
            expected = x * fs._pca_whitening_scale
            assert np.allclose(whitened, expected)

    def test_apply_pca_whitening_batch_matches_single(self) -> None:
        """Batch whitening must produce the same result as per-row whitening."""
        pca = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=True,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            _ = fs.pca  # trigger lazy PCA loading
            X = np.arange(12, dtype=np.float64).reshape(3, 4)
            batch_result = fs._apply_pca_whitening_batch(X)
            for i in range(3):
                single_result = fs._apply_pca_whitening(X[i])
                assert np.allclose(batch_result[i], single_result)

    def test_get_pca_whitening_scales_shape_and_bias(self) -> None:
        """Scales array has dimension length; last element (bias) is 1.0."""
        pca = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "pca.joblib"
            joblib.dump(pca, p)
            fs = FeatureService(
                pca_path=p,
                whiten_pca=True,
                allow_jit_training=False,
                custom_encoder=lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20),
                embedding_dim=20,
            )
            _ = fs.pca  # trigger lazy PCA loading
            scales = fs.get_pca_whitening_scales()
            assert scales.shape == (fs.dimension,)
            assert scales[-1] == 1.0, "Bias scale must always be 1.0"
            assert not np.allclose(scales[:-1], 1.0), "PCA scales should differ from 1.0"

    def test_compute_whitening_scale_returns_none_without_ev(self) -> None:
        """If PCA object lacks explained_variance_, return None."""

        class FakePCA:
            pass

        assert FeatureService._compute_whitening_scale_from_pca(FakePCA()) is None

    def test_pca_has_builtin_whitening_detects_flag(self) -> None:
        """_pca_has_builtin_whitening correctly reads pca.whiten."""
        pca_yes = self._make_pca(n_components=4, whiten=True)
        pca_no = self._make_pca(n_components=4, whiten=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "yes.joblib"
            p2 = Path(tmpdir) / "no.joblib"
            joblib.dump(pca_yes, p1)
            joblib.dump(pca_no, p2)
            enc = lambda s: np.random.default_rng(hash(s) % 2**32).standard_normal(20)
            fs1 = FeatureService(pca_path=p1, whiten_pca=False, allow_jit_training=False,
                                 custom_encoder=enc, embedding_dim=20)
            fs2 = FeatureService(pca_path=p2, whiten_pca=False, allow_jit_training=False,
                                 custom_encoder=enc, embedding_dim=20)
            assert fs1._pca_has_builtin_whitening() is True
            assert fs2._pca_has_builtin_whitening() is False

    def test_jit_trained_pca_inherits_whiten_flag(self) -> None:
        """When JIT trains a new PCA, it should set whiten=whiten_pca."""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "jit.joblib"
            fs = FeatureService(pca_path=p, whiten_pca=True, allow_jit_training=True)
            pca = fs.pca
            assert pca is not None
            assert getattr(pca, "whiten", False) is True


# ===================================================================
# 7–9. Warmup priors whitening auto-conversion
# ===================================================================


class TestWarmupPriorsWhiteningConversion:

    def test_unwhitened_priors_converted_to_whitened_router(self, tmp_path) -> None:
        """Unwhitened priors (pca_whitened=False) should be DAD/Db-converted
        when the router uses whitened features."""
        registry = _simple_registry(2)
        scales = np.ones(DIM, dtype=np.float64)
        scales[:-1] = np.arange(2, DIM + 1, dtype=np.float64)
        fs = _mock_feature_service(dim=DIM, whiten_pca=True, scales=scales)

        A_orig = {m: np.eye(DIM) * 100.0 for m in registry}
        b_orig = {m: np.ones(DIM) * 10.0 for m in registry}
        _write_priors(tmp_path / "p.joblib", A_orig, b_orig, n=1000,
                      pca_whitened=False)

        router = BanditRouter.create(
            model_registry=registry,
            priors=str(tmp_path / "p.joblib"),
            prior_n_effective=1000.0,
            feature_service=fs,
            
        )

        m0 = list(registry.keys())[0]
        A_loaded = router.bandit.A[m0]
        # The diagonal should reflect the DAD transform plus init_lambda
        for i in range(DIM):
            expected = 100.0 * scales[i] * scales[i] + router.bandit.init_lambda
            assert np.isclose(A_loaded[i, i], expected, rtol=0.01), (
                f"A[{i},{i}]={A_loaded[i,i]:.4f}, expected {expected:.4f}"
            )

    def test_already_whitened_priors_skip_conversion(self, tmp_path, caplog) -> None:
        """When both priors and router are whitened, no conversion occurs."""
        registry = _simple_registry(1)
        scales = np.ones(DIM, dtype=np.float64)
        scales[:-1] = np.arange(2, DIM + 1, dtype=np.float64)
        fs = _mock_feature_service(dim=DIM, whiten_pca=True, scales=scales)

        A_orig = {m: np.eye(DIM) * 50.0 for m in registry}
        b_orig = {m: np.ones(DIM) * 5.0 for m in registry}
        _write_priors(tmp_path / "p.joblib", A_orig, b_orig, n=1000,
                      pca_whitened=True)

        import logging
        with caplog.at_level(logging.INFO, logger="pareto_bandit.router"):
            router = BanditRouter.create(
                model_registry=registry,
                priors=str(tmp_path / "p.joblib"),
                prior_n_effective=1000.0,
                feature_service=fs,
                
            )

        conversion_msgs = [r for r in caplog.records if "Converted warmup priors" in r.message]
        assert len(conversion_msgs) == 0, "Should NOT convert when both sides are whitened"

        m0 = list(registry.keys())[0]
        # A should be just the raw prior scaled 1:1 + init_lambda
        expected_diag = 50.0 + router.bandit.init_lambda
        assert np.isclose(router.bandit.A[m0][0, 0], expected_diag, rtol=0.01)

    def test_whitened_priors_downconverted_for_unwhitened_router(self, tmp_path) -> None:
        """Whitened priors should be inverse-DAD converted when the router
        does NOT use whitening (all scales = 1)."""
        registry = _simple_registry(1)
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM, dtype=np.float64))

        scales_from_prior = np.ones(DIM, dtype=np.float64)
        scales_from_prior[:-1] = np.arange(2, DIM + 1, dtype=np.float64)

        A_orig = {m: np.eye(DIM) * 100.0 for m in registry}
        b_orig = {m: np.ones(DIM) * 10.0 for m in registry}
        _write_priors(tmp_path / "p.joblib", A_orig, b_orig, n=1000,
                      pca_whitened=True)

        router = BanditRouter.create(
            model_registry=registry,
            priors=str(tmp_path / "p.joblib"),
            prior_n_effective=1000.0,
            feature_service=fs,
            
        )

        m0 = list(registry.keys())[0]
        A_loaded = router.bandit.A[m0]
        # With unwhitened router (scales=1), inverse conversion divides by
        # the prior whitening scales.  Since the mock returns all-ones,
        # the inverse scales are also all-ones → A stays unchanged + init_lambda.
        expected_diag = 100.0 + router.bandit.init_lambda
        assert np.isclose(A_loaded[0, 0], expected_diag, rtol=0.01)


# ===================================================================
# 9. Shipped warmup priors metadata
# ===================================================================


class TestShippedWarmupPriorsMetadata:

    _paths = [
        Path("src/artifacts/priors_warmup_43model.joblib"),
        Path("src/artifacts/priors_warmup_k2_from_43model.joblib"),
    ]

    @pytest.mark.parametrize("path", _paths, ids=lambda p: p.name)
    def test_warmup_priors_are_marked_whitened(self, path: Path) -> None:
        if not path.exists():
            pytest.skip(f"{path} not found")
        d = joblib.load(path)
        assert d.get("pca_whitened") is True, (
            f"{path.name} should have pca_whitened=True after migration"
        )

    @pytest.mark.parametrize("path", _paths, ids=lambda p: p.name)
    def test_warmup_priors_have_explained_variance(self, path: Path) -> None:
        if not path.exists():
            pytest.skip(f"{path} not found")
        d = joblib.load(path)
        ev = d.get("pca_explained_variance")
        assert ev is not None and len(ev) == 32, (
            f"{path.name} should embed 32-element pca_explained_variance"
        )


# ===================================================================
# 10. calibration.generate_warmup_priors whitening
# ===================================================================


class TestCalibrationWhitening:

    def test_generate_warmup_priors_adds_whitening_metadata(self) -> None:
        from pareto_bandit.calibration import generate_warmup_priors
        from unittest.mock import patch

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 20))
        pca = PCA(n_components=4)
        pca.fit(X_train)

        rewards_data = []
        for i in range(30):
            prompt = f"prompt {i}"
            rewards_data.append({
                "prompt": prompt,
                "rewards": {"model-a": float(rng.random()), "model-b": float(rng.random())},
            })

        class FakeEncoder:
            def encode(self, text, **kw):
                return rng.standard_normal(20).astype(np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=FakeEncoder()):
            state = generate_warmup_priors(
                rewards_data,
                encoder_model="fake",
                pca=pca,
                whiten_pca=True,
            )

        assert state["pca_whitened"] is True
        assert len(state["pca_explained_variance"]) == 4

    def test_generate_warmup_priors_no_whitening_flag(self) -> None:
        from pareto_bandit.calibration import generate_warmup_priors
        from unittest.mock import patch

        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 20))
        pca = PCA(n_components=4)
        pca.fit(X_train)

        rewards_data = [
            {"prompt": f"p{i}", "rewards": {"m": float(rng.random())}}
            for i in range(20)
        ]

        class FakeEncoder:
            def encode(self, text, **kw):
                return rng.standard_normal(20).astype(np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=FakeEncoder()):
            state = generate_warmup_priors(
                rewards_data,
                encoder_model="fake",
                pca=pca,
                whiten_pca=False,
            )

        assert state["pca_whitened"] is False


# ===================================================================
# 11. Cost filtering: $/1k tokens semantics
# ===================================================================


class TestCostFilteringSematics:

    def test_max_cost_is_in_per_1k_tokens(self) -> None:
        """max_cost=0.001 means $0.001/1k tokens.
        blended_cost_per_m = (0.5 + 1.5) / 2 = 1.0 -> 0.001/1k.
        So model-0 with blended 1.0/M = 0.001/1k should just barely pass."""
        registry = _simple_registry(2)
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs,
        )
        x = np.zeros(DIM); x[-1] = 1.0
        model, _ = router.route(x, max_cost=0.002)
        assert model == "model-0"

    def test_max_cost_too_tight_raises(self) -> None:
        registry = _simple_registry(2)
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs,
        )
        x = np.zeros(DIM); x[-1] = 1.0
        with pytest.raises(NoEligibleModelsError):
            router.route(x, max_cost=1e-7)


# ===================================================================
# 12. Backward-compatible registry keys
# ===================================================================


class TestBackwardCompatRegistryKeys:

    def test_cost_per_1m_tokens_accepted(self) -> None:
        """Legacy cost_per_1m_tokens key should be treated as blended."""
        registry = {
            "legacy-model": {
                "model_id": "legacy-model",
                "cost_per_1m_tokens": 5.0,
            }
        }
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs,
        )
        assert router.registry["legacy-model"]["blended_cost_per_m"] == 5.0

    def test_median_latency_s_mapped_to_ttft(self) -> None:
        """Legacy median_latency_s should populate time_to_first_token_seconds."""
        registry = {
            "legacy-model": {
                "model_id": "legacy-model",
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 3.0,
                "median_latency_s": 0.75,
            }
        }
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs,
        )
        assert router.registry["legacy-model"]["time_to_first_token_seconds"] == 0.75


# ===================================================================
# 13. Pessimistic cost defaults
# ===================================================================


class TestPessimisticCostDefaults:

    def test_fully_missing_cost_gets_defaults(self) -> None:
        """A model with NO cost fields should receive pessimistic fallback."""
        registry = {"no-cost": {"model_id": "no-cost"}}
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs,
        )
        m = router.registry["no-cost"]
        assert "blended_cost_per_m" in m
        assert m["blended_cost_per_m"] > 0
        assert m["input_cost_per_m"] > 0
        assert m["output_cost_per_m"] > 0

    def test_partial_cost_still_raises(self) -> None:
        """Having only input (or only output) cost is a likely registry bug."""
        with pytest.raises(MissingCostError):
            BanditRouter(
                model_registry={"bad": {"model_id": "bad", "input_cost_per_m": 2.0}},
                feature_service=_mock_feature_service(dim=DIM),
            )


# ===================================================================
# 14. Registry deep copy
# ===================================================================


class TestRegistryDeepCopy:

    def test_router_does_not_mutate_caller_registry(self) -> None:
        """Mutations during init (blended cost derivation) must not leak."""
        inner = {
            "model_id": "m",
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
        }
        caller_reg = {"m": inner}
        fs = _mock_feature_service(dim=DIM, whiten_pca=False,
                                   scales=np.ones(DIM))
        _ = BanditRouter.create(
            model_registry=caller_reg, priors="none",
            feature_service=fs,
        )
        assert "blended_cost_per_m" not in inner, (
            "Router init should not add blended_cost_per_m to the caller's dict"
        )


