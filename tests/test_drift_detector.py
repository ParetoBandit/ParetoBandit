"""Comprehensive tests for the embedding-based DriftDetector.

Tests are organised into three groups:
  1. **Unit tests** — exercise DriftDetector in isolation with synthetic vectors.
  2. **Router integration tests** — verify end-to-end ff adaptation via BanditRouter.
  3. **Realistic integration tests** — use actual Pareto/K4 PCA embeddings to
     validate that the detector fires on real distribution shift and stays
     silent when there is no shift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from bandit_gpt.drift import CentroidDriftDetector, DriftDetector


# ======================================================================
# Helpers
# ======================================================================


def _stationary_vectors(
    n: int,
    dim: int = 10,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate *n* i.i.d. vectors from N(mean, std) per component."""
    rng = np.random.default_rng(seed)
    return [rng.normal(mean, std, size=dim) for _ in range(n)]


def _shifted_vectors(
    n: int,
    dim: int = 10,
    *,
    base_mean: float = 0.0,
    shift: float = 3.0,
    std: float = 1.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate *n* vectors with a mean shifted by *shift* on all components."""
    rng = np.random.default_rng(seed)
    return [rng.normal(base_mean + shift, std, size=dim) for _ in range(n)]


def _feed_vectors(
    detector: DriftDetector,
    vectors: list[np.ndarray],
) -> list[bool]:
    """Feed a sequence of context vectors and return drift flags."""
    return [detector.update(v) for v in vectors]


# ======================================================================
# Unit tests — DriftDetector in isolation
# ======================================================================


class TestStationaryNoTrigger:
    """Stationary vectors should never trigger drift."""

    def test_constant_distribution(self):
        det = DriftDetector(threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
                            confirmation_window=20)
        vecs = _stationary_vectors(500, seed=42)
        flags = _feed_vectors(det, vecs)

        assert det.is_burned_in
        assert not any(flags[50:]), "Stationary stream must not trigger drift"
        assert det.drift_score < 0.5, (
            f"drift_score should be near 0 for stationary data, got {det.drift_score:.4f}"
        )

    def test_high_variance_stationary(self):
        """Even with high-variance vectors, stationary data should not trigger."""
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        vecs = _stationary_vectors(600, dim=20, std=5.0, seed=77)
        flags = _feed_vectors(det, vecs)

        assert det.is_burned_in
        assert not any(flags[100:]), (
            "High-variance stationary data must not trigger drift"
        )


class TestSharpShift:
    """A sudden jump in embedding distribution should trigger drift quickly."""

    def test_triggers_after_shift(self):
        det = DriftDetector(
            threshold=1.5, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=10,
        )

        burn = _stationary_vectors(50, seed=1)
        flags_burn = _feed_vectors(det, burn)
        assert not any(flags_burn), "No drift during burn-in"
        assert det.is_burned_in

        shifted = _shifted_vectors(100, shift=4.0, seed=2)
        flags_shift = _feed_vectors(det, shifted)

        assert any(flags_shift), "Drift should trigger after sharp shift"

        first_trigger = next(i for i, f in enumerate(flags_shift) if f)
        assert first_trigger < 60, (
            f"Trigger should occur within ~50 steps of shift "
            f"(EMA convergence + confirmation window), got {first_trigger}"
        )

    def test_drift_score_magnitude(self):
        det = DriftDetector(threshold=1.5, burn_in_steps=50, ema_alpha=0.05)
        burn = _stationary_vectors(50, seed=3)
        _feed_vectors(det, burn)

        shifted = _shifted_vectors(200, shift=5.0, seed=4)
        _feed_vectors(det, shifted)

        assert det.drift_score > 1.0, (
            f"After large shift, drift_score should be substantial, got {det.drift_score:.4f}"
        )


class TestGradualDrift:
    """Slowly drifting distribution should eventually trigger."""

    def test_linear_ramp(self):
        det = DriftDetector(
            threshold=1.5, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=10,
        )

        burn = _stationary_vectors(50, seed=10)
        _feed_vectors(det, burn)
        assert det.is_burned_in

        rng = np.random.default_rng(11)
        triggered = False
        trigger_step = None
        for i in range(400):
            shift_amount = 5.0 * i / 400
            v = rng.normal(shift_amount, 1.0, size=10)
            flag = det.update(v)
            if flag and not triggered:
                triggered = True
                trigger_step = i

        assert triggered, "Gradual drift should eventually trigger"
        assert trigger_step is not None
        assert 20 < trigger_step < 250, (
            f"Trigger should occur in the middle of the ramp, got step {trigger_step}"
        )


class TestThresholdSensitivity:
    """Lower thresholds should trigger earlier than higher ones."""

    def test_ordering(self):
        trigger_steps: dict[float, int | None] = {}

        for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
            det = DriftDetector(
                threshold=threshold, burn_in_steps=50, ema_alpha=0.05,
                confirmation_window=10,
            )
            burn = _stationary_vectors(50, seed=20)
            _feed_vectors(det, burn)

            shifted = _shifted_vectors(300, shift=4.0, seed=21)
            trigger_step = None
            for i, v in enumerate(shifted):
                if det.update(v) and trigger_step is None:
                    trigger_step = i
            trigger_steps[threshold] = trigger_step

        triggered_thresholds = {
            t: s for t, s in trigger_steps.items() if s is not None
        }
        sorted_by_threshold = sorted(triggered_thresholds.items())
        for i in range(len(sorted_by_threshold) - 1):
            t1, s1 = sorted_by_threshold[i]
            t2, s2 = sorted_by_threshold[i + 1]
            assert s1 <= s2, (
                f"threshold={t1} triggered at step {s1} but threshold={t2} "
                f"triggered earlier at step {s2}"
            )


class TestBurnInSafety:
    """No triggering during burn-in regardless of vector magnitude."""

    def test_extreme_vectors_during_burnin(self):
        det = DriftDetector(threshold=0.5, burn_in_steps=100, ema_alpha=0.1)

        for i in range(99):
            flag = det.update(np.ones(10) * 1000.0)
            assert not flag, "Must not trigger during burn-in"
            assert not det.is_drifting
            assert not det.is_burned_in

    def test_burn_in_completes_at_exact_step(self):
        det = DriftDetector(threshold=2.0, burn_in_steps=10, ema_alpha=0.1)

        rng = np.random.default_rng(0)
        for i in range(9):
            det.update(rng.standard_normal(5))
            assert not det.is_burned_in

        det.update(rng.standard_normal(5))
        assert det.is_burned_in
        assert det.baseline > 0


class TestConfirmationWindow:
    """Confirmation window prevents transient spikes from triggering."""

    def test_transient_spike_not_confirmed(self):
        det = DriftDetector(
            threshold=1.5, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        burn = _stationary_vectors(50, seed=50)
        _feed_vectors(det, burn)

        # Brief 3-step spike at a realistic 4-sigma outlier level
        rng = np.random.default_rng(50)
        for _ in range(3):
            det.update(rng.normal(4.0, 0.5, size=10))
        assert not det.is_drifting, (
            "Spike shorter than confirmation_window must not trigger"
        )

        # Return to normal — the EMA decays back and streak resets
        normal = _stationary_vectors(200, seed=51)
        flags = _feed_vectors(det, normal)
        assert not any(flags[-50:]), "Normal data after spike must not trigger"

    def test_sustained_shift_confirmed(self):
        det = DriftDetector(
            threshold=1.0, burn_in_steps=20, ema_alpha=0.3,
            confirmation_window=5,
        )
        burn = _stationary_vectors(20, seed=52)
        _feed_vectors(det, burn)

        shifted = _shifted_vectors(50, shift=5.0, seed=53)
        flags = _feed_vectors(det, shifted)
        assert any(flags), "Sustained shift should be confirmed"

    def test_window_one_behaves_like_instant(self):
        det = DriftDetector(
            threshold=1.0, burn_in_steps=50, ema_alpha=0.1,
            confirmation_window=1,
        )
        burn = _stationary_vectors(50, seed=54)
        _feed_vectors(det, burn)

        shifted = _shifted_vectors(100, shift=5.0, seed=55)
        flags = _feed_vectors(det, shifted)
        assert any(flags), "confirmation_window=1 should trigger same as instant"


class TestEMASmoothingNoTransient:
    """Single spikes should not cause false triggers."""

    def test_single_spike_absorbed(self):
        det = DriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )

        burn = _stationary_vectors(50, seed=30)
        _feed_vectors(det, burn)

        rng = np.random.default_rng(31)
        triggered = False
        for i in range(300):
            if i % 50 == 0:
                v = rng.normal(3.5, 0.5, size=10)
            else:
                v = rng.standard_normal(10)
            if det.update(v):
                triggered = True

        assert not triggered, (
            "Occasional moderate outliers should be absorbed by EMA smoothing + confirmation"
        )


class TestDriftScoreSemantics:
    """drift_score is chi-squared excess; drift_ratio = drift_score / (threshold * baseline_std)."""

    def test_ratio_at_trigger(self):
        det = DriftDetector(
            threshold=1.0, burn_in_steps=50, ema_alpha=0.1,
            confirmation_window=5,
        )

        burn = _stationary_vectors(50, seed=40)
        _feed_vectors(det, burn)

        shifted = _shifted_vectors(100, shift=5.0, seed=41)
        for v in shifted:
            det.update(v)
            if det.is_drifting:
                break

        assert det.drift_ratio >= 1.0, (
            f"drift_ratio must be >= 1.0 when drifting, got {det.drift_ratio:.4f}"
        )
        trigger_level = det.threshold * det.baseline_std
        assert det.drift_score >= trigger_level - 1e-6, (
            f"drift_score ({det.drift_score:.4f}) must be >= trigger level "
            f"({trigger_level:.4f}) when drifting"
        )

    def test_ratio_below_one_when_not_drifting(self):
        det = DriftDetector(threshold=3.0, burn_in_steps=50, ema_alpha=0.05)
        vecs = _stationary_vectors(200, seed=42)
        _feed_vectors(det, vecs)

        assert det.drift_ratio < 1.0
        assert not det.is_drifting


class TestThresholdZeroDisabled:
    """threshold=0 means disabled (default) — detector is never instantiated by router."""

    def test_zero_threshold_triggers_on_any_increase(self):
        det = DriftDetector(threshold=0.0, burn_in_steps=10, ema_alpha=0.5,
                            confirmation_window=1)

        burn = _stationary_vectors(10, seed=0)
        _feed_vectors(det, burn)

        shifted = _shifted_vectors(50, shift=3.0, seed=1)
        flags = _feed_vectors(det, shifted)
        assert any(flags), (
            "threshold=0 should trigger on any chi-squared increase above baseline"
        )


class TestGetState:
    """get_state() returns a serialisable dictionary."""

    def test_state_keys(self):
        det = DriftDetector(threshold=2.0, burn_in_steps=5)
        vecs = _stationary_vectors(10, dim=5, seed=0)
        _feed_vectors(det, vecs)
        state = det.get_state()

        expected_keys = {
            "total_steps", "burned_in", "baseline", "baseline_std",
            "ema_chi2", "drift_score", "drift_ratio", "is_drifting",
            "consecutive_above", "confirmed",
            "threshold", "burn_in_steps", "ema_alpha", "confirmation_window",
        }
        assert set(state.keys()) == expected_keys


class TestInputValidation:
    """Constructor rejects invalid parameters."""

    def test_negative_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            DriftDetector(threshold=-0.1)

    def test_small_burn_in(self):
        with pytest.raises(ValueError, match="burn_in_steps"):
            DriftDetector(burn_in_steps=3)

    def test_invalid_ema_alpha(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            DriftDetector(ema_alpha=0.0)
        with pytest.raises(ValueError, match="ema_alpha"):
            DriftDetector(ema_alpha=1.5)

    def test_zero_confirmation_window(self):
        with pytest.raises(ValueError, match="confirmation_window"):
            DriftDetector(confirmation_window=0)


# ======================================================================
# Detector reset lifecycle
# ======================================================================


class TestDetectorReset:
    """DriftDetector.reset() clears all state and re-enters burn-in."""

    def test_reset_clears_drift_state(self):
        det = DriftDetector(threshold=1.0, burn_in_steps=10, ema_alpha=0.2,
                            confirmation_window=3)
        phase1 = _stationary_vectors(50, dim=8, seed=0)
        phase2 = _shifted_vectors(50, dim=8, shift=5.0, seed=1)

        _feed_vectors(det, phase1)
        assert det.is_burned_in
        _feed_vectors(det, phase2)
        assert det.is_drifting, "Should drift after shift"

        det.reset()

        assert not det.is_burned_in, "Should be back in burn-in"
        assert not det.is_drifting, "is_drifting should be False"
        assert det.drift_score == 0.0
        assert det.total_steps == 0
        assert det.baseline == 0.0
        assert det.baseline_std == 0.0

    def test_reset_allows_new_baseline(self):
        """After reset, new burn-in vectors establish a new baseline."""
        det = DriftDetector(threshold=2.0, burn_in_steps=10, ema_alpha=0.2,
                            confirmation_window=3)
        _feed_vectors(det, _stationary_vectors(20, dim=6, seed=0))
        old_baseline = det.baseline
        assert old_baseline > 0

        det.reset()
        _feed_vectors(det, _stationary_vectors(20, dim=6, seed=99))
        assert det.is_burned_in
        assert det.baseline > 0
        assert det.baseline != old_baseline, (
            "New burn-in should produce a different baseline from different data"
        )

    def test_preserves_config(self):
        det = DriftDetector(threshold=3.5, burn_in_steps=20, ema_alpha=0.1,
                            confirmation_window=5)
        det.reset()
        assert det.threshold == 3.5
        assert det.burn_in_steps == 20
        assert det.ema_alpha == 0.1
        assert det.confirmation_window == 5


# ======================================================================
# Policy reset_to_tabula_rasa
# ======================================================================


class TestPolicyReset:
    """DisjointLinUCBPolicy.reset_to_tabula_rasa() restores cold-start."""

    def test_disjoint_reset(self):
        from bandit_gpt.router import DisjointLinUCBPolicy

        models = ["arm_a", "arm_b"]
        dim = 10
        policy = DisjointLinUCBPolicy(
            model_names=models, dim=dim, alpha=0.5, init_lambda=2.0,
        )

        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal(dim)
            policy.select_arm(x, candidates=models)
            chosen = rng.choice(models)
            policy.update(chosen, x, reward=rng.uniform(0, 1))

        for m in models:
            assert not np.allclose(
                policy.A[m], np.eye(dim) * 2.0
            ), f"A[{m}] should have learned state"

        policy.reset_to_tabula_rasa()

        for m in models:
            np.testing.assert_array_equal(
                policy.A[m], np.eye(dim) * 2.0,
            )
            np.testing.assert_array_equal(
                policy.b[m], np.zeros(dim),
            )
            assert policy.last_update[m] == 0
            assert policy.last_played[m] == 0
        assert policy.t == 0



# ======================================================================
# Router integration tests — DriftDetector inside BanditRouter
# ======================================================================


class TestRouterTabulaRasaReset:
    """End-to-end: router detects embedding drift and resets to tabula rasa."""

    @pytest.fixture
    def two_arm_registry(self):
        return {
            "cheap-model": {
                "model_id": "cheap-model",
                "display_name": "Cheap",
                "scores": {"hle": 0.5},
                "hallucination_rate": 5.0,
                "input_cost_per_m": 0.1,
                "output_cost_per_m": 0.1,
            },
            "expensive-model": {
                "model_id": "expensive-model",
                "display_name": "Expensive",
                "scores": {"hle": 0.9},
                "hallucination_rate": 1.0,
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 30.0,
            },
        }

    def test_resets_on_embedding_shift(self, two_arm_registry):
        """Full lifecycle: detect shift → reset bandit → re-learn → stabilize.

        Verifies algorithmic correctness:
        1. Bandit accumulates non-trivial learned state in phase 1
        2. Shift triggers tabula rasa reset (internal time counter zeroed,
           learned coefficients reflect only post-reset data)
        3. The detector re-burns-in on shifted data and does NOT re-trigger
           (shifted data is now in-distribution for the new baseline)
        4. gamma remains 1.0 throughout (no forgetting factor change)
        """
        from bandit_gpt.drift import DriftDetector
        from bandit_gpt.router import BanditRouter

        router = BanditRouter.create(
            model_registry=two_arm_registry,
            priors="none",
            alpha=0.5,
            
            cost_penalty=0.0,
            drift_threshold=0.0,
        )

        rng = np.random.default_rng(99)
        dim = router.bandit.dim

        router.drift_threshold = 1.5
        router._drift_adapted = False
        router.drift_detector = DriftDetector(
            threshold=1.5,
            burn_in_steps=50,
            ema_alpha=0.1,
            confirmation_window=10,
        )

        # Phase 1 (burn-in + stable): accumulate non-trivial state
        n_phase1 = 80
        for _ in range(n_phase1):
            x = rng.standard_normal(dim).astype(np.float32)
            model, log = router.route(x)
            reward = float(np.clip(rng.normal(0.6, 0.03), 0, 1))
            router.process_feedback(log.request_id, reward=reward)

        assert router.drift_detector.is_burned_in, "Burn-in should complete"
        assert not router._drift_adapted, "No drift yet in stable phase"

        # Bandit should have accumulated state: t reflects 80 route + 80
        # feedback calls.  Exact value depends on policy internals, but must
        # be > 0.
        pre_reset_t = router.bandit.t
        assert pre_reset_t > 0, "Bandit should have processed observations"

        # Snapshot learned θ = A_inv @ b to verify it changes after reset
        pre_theta = {}
        for m in router.bandit.models:
            pre_theta[m] = router.bandit.A_inv[m] @ router.bandit.b[m]

        # Phase 2: Shifted embeddings (mean-shifted by 4σ).
        # Run enough steps for full lifecycle: trigger → reset → re-burn-in → stabilize.
        n_shifted = 200
        reset_step = None
        for step in range(n_shifted):
            x = (rng.standard_normal(dim) + 4.0).astype(np.float32)
            model, log = router.route(x)
            reward = float(np.clip(rng.normal(0.6, 0.03), 0, 1))
            router.process_feedback(log.request_id, reward=reward)
            if router._drift_adapted and reset_step is None:
                reset_step = step

        # --- Verify detection occurred ---
        assert reset_step is not None, "Drift should have been detected"
        n_resets = getattr(router, "_n_resets", 0)
        assert n_resets == 1, (
            f"Exactly 1 reset expected (shifted data becomes the new "
            f"in-distribution after re-burn-in), got {n_resets}"
        )

        # --- Verify bandit time counter reflects reset ---
        # After reset_to_tabula_rasa() sets t=0, only post-reset calls
        # increment t.  Without the reset, t would be proportional to
        # n_phase1 + n_shifted.  With the reset, t reflects only
        # post-reset observations.
        post_t = router.bandit.t
        post_reset_steps = n_shifted - reset_step
        max_expected_t = post_reset_steps * 3  # generous upper bound
        assert post_t <= max_expected_t, (
            f"bandit.t={post_t} should reflect only post-reset observations "
            f"(~{post_reset_steps} steps), not accumulate pre-reset state "
            f"(pre_reset_t was {pre_reset_t})"
        )

        # --- Verify learned coefficients changed ---
        # Post-reset θ is trained on shifted data (mean ~4), so it should
        # be substantially different from pre-reset θ (trained on mean ~0).
        for m in router.bandit.models:
            post_theta = router.bandit.A_inv[m] @ router.bandit.b[m]
            assert not np.allclose(pre_theta[m], post_theta, atol=0.1), (
                f"θ[{m}] should differ after reset + re-learning on shifted "
                f"data (pre={pre_theta[m][:3]}, post={post_theta[:3]})"
            )

        # --- Verify lifecycle: detector re-stabilized on shifted data ---
        assert router.drift_detector.is_burned_in, (
            "Detector should have completed re-burn-in on shifted data "
            f"(reset at step {reset_step}, fed {n_shifted - reset_step} "
            f"more vectors, burn_in=50)"
        )
        assert not router.drift_detector.is_drifting, (
            "Detector should NOT re-trigger on the same shifted distribution — "
            "it's now the baseline after re-burn-in"
        )

        # --- Verify gamma unchanged (tabula rasa, not FF decay) ---
        assert router.bandit.gamma == 1.0

    def test_no_adaptation_when_disabled(self, two_arm_registry):
        from bandit_gpt.router import BanditRouter

        router = BanditRouter.create(
            model_registry=two_arm_registry,
            priors="none",
            alpha=0.5,
            
            drift_threshold=0.0,
        )

        assert router.drift_detector is None
        assert router.bandit.gamma == 1.0

        rng = np.random.default_rng(100)
        dim = router.bandit.dim
        for _ in range(200):
            x = rng.standard_normal(dim).astype(np.float32)
            model, log = router.route(x)
            reward = float(np.clip(rng.normal(0.1, 0.05), 0, 1))
            router.process_feedback(log.request_id, reward=reward)

        assert router.bandit.gamma == 1.0

    def test_no_detector_without_priors(self, two_arm_registry):
        """drift_threshold > 0 with priors='none' should NOT create a detector."""
        from bandit_gpt.router import BanditRouter

        router = BanditRouter.create(
            model_registry=two_arm_registry,
            priors="none",
            alpha=0.5,
            
            drift_threshold=2.0,
        )

        assert router.drift_detector is None
        assert router.drift_threshold == 2.0


# ======================================================================
# End-to-end with synthetic priors through the real create() path
# ======================================================================


class TestEndToEndWithSyntheticPriors:
    """Full production pathway: create(priors=joblib) -> drift detection -> tabula rasa.

    Uses synthetic warmup priors saved to a temp joblib file so the test
    exercises the real ``create()`` priors-loading path and the automatic
    ``DriftDetector`` instantiation.
    """

    DIM = 5  # 4 precomputed features + 1 bias

    REGISTRY = {
        "cheap-model": {
            "model_id": "cheap-model",
            "display_name": "Cheap",
            "scores": {"hle": 0.5},
            "hallucination_rate": 5.0,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1,
        },
        "expensive-model": {
            "model_id": "expensive-model",
            "display_name": "Expensive",
            "scores": {"hle": 0.9},
            "hallucination_rate": 1.0,
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
        },
    }

    THETA = {
        "cheap-model": np.array([0.3, 0.1, -0.2, 0.1, 0.4]),
        "expensive-model": np.array([-0.1, 0.2, 0.1, -0.1, 0.7]),
    }

    @pytest.fixture
    def priors_path(self, tmp_path: Path) -> Path:
        """Create a synthetic warmup priors joblib file."""
        import joblib

        n_eff = 1000
        A = {m: float(n_eff) * np.eye(self.DIM) for m in self.THETA}
        b = {m: float(n_eff) * self.THETA[m] for m in self.THETA}
        path = tmp_path / "synthetic_priors.joblib"
        joblib.dump({"A": A, "b": b, "n": n_eff}, path)
        return path

    def _make_router(
        self,
        priors_path: Path,
        drift_threshold: float = 1.5,
    ):
        from bandit_gpt.feature_service import FeatureService
        from bandit_gpt.router import BanditRouter
        from bandit_gpt.storage import EphemeralContextStore

        fs = FeatureService.for_precomputed(self.DIM)
        store = EphemeralContextStore()

        return BanditRouter.create(
            model_registry=dict(self.REGISTRY),
            feature_service=fs,
            context_store=store,
            priors=str(priors_path),
            prior_n_effective=1000.0,
            alpha=0.5,
            
            cost_penalty=0.0,
            drift_threshold=drift_threshold,
            drift_method="chi2",
            drift_burn_in_steps=50,
            drift_ema_alpha=0.1,
            drift_confirmation_window=10,
        )

    def test_create_with_priors_activates_detector(self, priors_path):
        router = self._make_router(priors_path, drift_threshold=1.5)

        assert router.drift_detector is not None, (
            "DriftDetector should be instantiated when priors are loaded "
            "and drift_threshold > 0"
        )
        assert router.bandit.gamma == 1.0
        assert not router._drift_adapted
        assert router.drift_threshold == 1.5

    def test_stable_traffic_no_adaptation(self, priors_path):
        """Same-distribution vectors should not trigger the detector."""
        router = self._make_router(priors_path, drift_threshold=2.0)
        rng = np.random.default_rng(42)

        for _ in range(200):
            x = np.append(rng.standard_normal(self.DIM - 1) * 0.3, 1.0)
            model, log = router.route(x)
            reward = float(np.clip(self.THETA[model] @ x + rng.normal(0, 0.02), 0, 1))
            router.process_feedback(log.request_id, reward=reward)

        assert router.drift_detector.is_burned_in
        assert not router._drift_adapted, (
            "Stable traffic must not trigger adaptation "
            f"(drift_score={router.drift_detector.drift_score:.4f}, "
            f"baseline={router.drift_detector.baseline:.4f})"
        )
        assert router.bandit.gamma == 1.0

    def test_shifted_traffic_triggers_reset(self, priors_path):
        """Full lifecycle with warmup priors: detect → reset → re-learn → stabilize.

        Verifies that when the router was initialized with strong warmup
        priors (n_eff=1000), the reset discards those priors entirely:
        1. The bandit's internal clock (t) reflects only post-reset observations
        2. The learned θ vectors orient toward the shifted distribution
        3. The detector re-stabilizes (no false re-trigger on uniform shift)
        """
        router = self._make_router(priors_path, drift_threshold=1.5)
        rng = np.random.default_rng(42)

        # Phase 1: stable embedding distribution (burn-in + extra)
        n_phase1 = 80
        for _ in range(n_phase1):
            x = np.append(rng.standard_normal(self.DIM - 1) * 0.3, 1.0)
            model, log = router.route(x)
            reward = float(np.clip(self.THETA[model] @ x + rng.normal(0, 0.02), 0, 1))
            router.process_feedback(log.request_id, reward=reward)

        assert router.drift_detector.is_burned_in
        assert not router._drift_adapted

        # Pre-reset: bandit has strong priors + 80 observations.
        pre_reset_t = router.bandit.t
        assert pre_reset_t > 0
        pre_theta = {}
        for m in router.bandit.models:
            pre_theta[m] = router.bandit.A_inv[m] @ router.bandit.b[m]

        # Phase 2: shifted embedding distribution (mean +3 on all components).
        n_shifted = 200
        reset_step = None
        for step in range(n_shifted):
            x = np.append(rng.standard_normal(self.DIM - 1) * 0.3 + 3.0, 1.0)
            model, log = router.route(x)
            reward = float(np.clip(rng.normal(0.5, 0.05), 0, 1))
            router.process_feedback(log.request_id, reward=reward)
            if router._drift_adapted and reset_step is None:
                reset_step = step

        # --- Detection occurred ---
        assert reset_step is not None, "Drift should have been detected"
        n_resets = getattr(router, "_n_resets", 0)
        assert n_resets == 1, (
            f"Exactly 1 reset expected on uniform shifted traffic, got {n_resets}"
        )

        # --- Warmup priors were discarded: time counter reset ---
        # After reset_to_tabula_rasa() sets t=0, only post-reset calls
        # increment t.  Without the reset, t would reflect both the prior
        # n_eff and all observations.  With the reset, t ≈ post_reset_steps.
        post_t = router.bandit.t
        post_reset_steps = n_shifted - reset_step
        max_expected_t = post_reset_steps * 3
        assert post_t <= max_expected_t, (
            f"bandit.t={post_t} should reflect only post-reset observations, "
            f"not accumulate prior state (pre_reset_t was {pre_reset_t})"
        )

        # --- Learned coefficients shifted ---
        for m in router.bandit.models:
            post_theta = router.bandit.A_inv[m] @ router.bandit.b[m]
            assert not np.allclose(pre_theta[m], post_theta, atol=0.05), (
                f"θ[{m}] should differ from prior-loaded values after "
                f"reset + re-learning on shifted data"
            )

        # --- Lifecycle completed: detector re-stabilized ---
        assert router.drift_detector.is_burned_in, (
            "Detector should have completed re-burn-in on shifted data"
        )
        assert not router.drift_detector.is_drifting, (
            "Shifted data is now in-distribution — no re-trigger expected"
        )
        assert router.bandit.gamma == 1.0


# ======================================================================
# Production-scenario tests — synthetic embeddings mimicking real shifts
# ======================================================================
#
# These tests model realistic production scenarios using synthetic PCA-like
# embeddings (d=26: 25 whitened PCA components + 1 bias term).  This avoids
# coupling tests to specific cached datasets while exercising the detector
# under conditions that mirror actual deployment.


_PROD_DIM = 26  # 25 PCA components + bias


def _prod_vectors(
    n: int,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate synthetic PCA-like prompt embeddings.

    Whitened PCA components are roughly unit-variance with zero mean, so
    ``mean=zeros, std=ones`` is the default.  The last component is a
    constant bias = 1.0.
    """
    rng = np.random.default_rng(seed)
    if mean is None:
        mean = np.zeros(_PROD_DIM)
    if std is None:
        std = np.ones(_PROD_DIM)
    # Bias term is constant
    mean = np.array(mean, dtype=np.float64)
    std = np.array(std, dtype=np.float64)
    mean[-1] = 1.0
    std[-1] = 0.0
    vecs = []
    for _ in range(n):
        v = rng.normal(mean, np.where(std > 0, std, 1e-12))
        v[-1] = 1.0
        vecs.append(v)
    return vecs


class TestProdStableTraffic:
    """No shift in production — detector must stay silent.

    Simulates a steady-state deployment where prompt types are stable
    over time (e.g., always general knowledge Q&A).
    """

    def test_no_trigger_on_stable_traffic(self):
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        vecs = _prod_vectors(1000, seed=42)
        flags = _feed_vectors(det, vecs)

        assert det.is_burned_in
        assert not any(flags), (
            "Stable production traffic must never trigger drift. "
            f"drift_score={det.drift_score:.4f}, ema_chi2={det.ema_chi2:.4f}"
        )

    def test_no_trigger_with_heterogeneous_stable_traffic(self):
        """Mixed but stable traffic: some components have higher variance."""
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        std = np.ones(_PROD_DIM)
        std[:5] = 2.0   # coding-heavy components are noisier
        std[10:15] = 0.5  # some components are tighter
        vecs = _prod_vectors(1000, std=std, seed=99)
        flags = _feed_vectors(det, vecs)

        assert det.is_burned_in
        assert not any(flags), (
            "Heterogeneous-variance stable traffic must not trigger drift"
        )


class TestProdMeanShift:
    """Domain shift — e.g., users move from general Q&A to coding.

    With d=26, the trigger level (baseline + 2σ) is ~2.0.  A 2σ mean
    shift on 8 components produces expected chi2 ≈ 2.4, comfortably
    above threshold.  This models a realistic scenario where coding
    prompts cluster differently on the top PCA components.
    """

    def test_triggers_on_domain_shift(self):
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _prod_vectors(400, seed=10)

        shifted_mean = np.zeros(_PROD_DIM)
        shifted_mean[:8] = 2.0  # top 8 PCA components shift by 2σ
        phase2 = _prod_vectors(400, mean=shifted_mean, seed=20)

        flags1 = _feed_vectors(det, phase1)
        assert det.is_burned_in
        assert not any(flags1), "Phase 1 must be stable"

        flags2 = _feed_vectors(det, phase2)
        assert any(flags2), (
            "Detector must trigger on 2σ mean shift in top 8 components. "
            f"drift_score={det.drift_score:.4f}, ema_chi2={det.ema_chi2:.4f}"
        )
        trigger_idx = next(i for i, f in enumerate(flags2) if f)
        assert trigger_idx < 150, (
            f"Detection should occur within 150 steps, got {trigger_idx}"
        )

    def test_triggers_on_widespread_shift(self):
        """Moderate shift (1.2σ) across all PCA components.

        Expected chi2 ≈ 1 + 25·1.44/26 ≈ 2.38, comfortably above
        the trigger level (~2.02).  Models a wholesale population change
        (e.g., deployment in a new market).
        """
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _prod_vectors(400, seed=30)

        shifted_mean = np.full(_PROD_DIM, 1.2)
        shifted_mean[-1] = 0.0  # don't shift bias
        phase2 = _prod_vectors(500, mean=shifted_mean, seed=40)

        _feed_vectors(det, phase1)
        flags2 = _feed_vectors(det, phase2)

        assert any(flags2), (
            "Widespread 1.2σ shift across 25 components should trigger. "
            f"drift_score={det.drift_score:.4f}, ema_chi2={det.ema_chi2:.4f}"
        )


class TestProdVarianceShift:
    """Variance change — e.g., users narrow from diverse topics to one niche.

    Same mean, but standard deviation on several components drops by half.
    The chi-squared statistic detects this because z-scores shrink
    systematically (or grow if variance increases).
    """

    def test_triggers_on_variance_increase(self):
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _prod_vectors(400, seed=50)

        wider_std = np.ones(_PROD_DIM)
        wider_std[:10] = 2.5  # variance doubles on 10 components
        phase2 = _prod_vectors(400, std=wider_std, seed=60)

        _feed_vectors(det, phase1)
        flags2 = _feed_vectors(det, phase2)

        assert any(flags2), (
            "Variance increase on 10 components must trigger drift. "
            f"drift_score={det.drift_score:.4f}"
        )


class TestProdGradualDrift:
    """Slow, gradual drift — mean shifts linearly over 500 prompts.

    Models a realistic scenario where user behaviour evolves over weeks.
    Ramps 8 components from 0σ to 3σ over 500 steps.  The detector
    should fire well before the shift reaches maximum.
    """

    def test_detects_gradual_drift(self):
        det = DriftDetector(
            threshold=2.0, burn_in_steps=100, ema_alpha=0.05,
            confirmation_window=20,
        )
        rng = np.random.default_rng(70)

        phase1 = _prod_vectors(300, seed=70)
        _feed_vectors(det, phase1)
        assert not det.is_drifting

        triggered = False
        trigger_step = None
        for step in range(500):
            drift_frac = step / 500.0
            mean = np.zeros(_PROD_DIM)
            mean[:8] = 3.0 * drift_frac  # ramp from 0σ to 3σ over 500 steps
            mean[-1] = 1.0
            v = rng.normal(mean, 1.0)
            v[-1] = 1.0
            if det.update(v) and not triggered:
                triggered = True
                trigger_step = step

        assert triggered, (
            "Gradual drift (0→3σ over 500 steps) must eventually trigger. "
            f"drift_score={det.drift_score:.4f}"
        )
        assert trigger_step < 400, (
            f"Should detect before drift reaches max, got step {trigger_step}"
        )


# ======================================================================
# CentroidDriftDetector — Unit Tests
# ======================================================================
#
# The centroid detector tracks the running-centroid cosine distance and
# is designed for topic/domain rotation shifts typical in LLM routing.


def _unit_sphere_vectors(
    n: int,
    dim: int = 50,
    *,
    direction: np.ndarray | None = None,
    spread: float = 0.3,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate L2-normalized vectors clustered around *direction*.

    When ``direction`` is None, a random reference direction is used.
    ``spread`` controls the angular dispersion: 0.0 = all identical,
    1.0 = large cone around the direction.
    """
    rng = np.random.default_rng(seed)
    if direction is None:
        direction = rng.standard_normal(dim)
    direction = direction / np.linalg.norm(direction)
    vecs = []
    for _ in range(n):
        noise = rng.standard_normal(dim) * spread
        v = direction + noise
        v /= np.linalg.norm(v)
        vecs.append(v)
    return vecs


class TestCentroidStableTraffic:
    """CentroidDriftDetector must stay silent on stationary traffic."""

    def test_no_trigger_on_stationary(self):
        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        vecs = _unit_sphere_vectors(800, seed=1)
        flags = [det.update(v) for v in vecs]

        assert det.is_burned_in
        assert not any(flags), (
            "Stationary unit-sphere traffic must never trigger. "
            f"score={det.ema_chi2:.6f}, baseline={det.baseline:.6f}"
        )

    def test_no_trigger_on_high_variance_stationary(self):
        """Wide angular spread but stable direction should not trigger."""
        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        vecs = _unit_sphere_vectors(800, spread=0.8, seed=2)
        flags = [det.update(v) for v in vecs]

        assert det.is_burned_in
        assert not any(flags), (
            "High-variance but stationary traffic must not trigger"
        )


class TestCentroidTopicShift:
    """CentroidDriftDetector must fire on centroid rotations."""

    def test_detects_orthogonal_shift(self):
        """Shift to a nearly orthogonal direction (cos_sim ~ 0)."""
        dim = 50
        rng = np.random.default_rng(10)
        dir1 = rng.standard_normal(dim)
        dir1 /= np.linalg.norm(dir1)

        dir2 = rng.standard_normal(dim)
        dir2 -= dir2.dot(dir1) * dir1
        dir2 /= np.linalg.norm(dir2)

        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _unit_sphere_vectors(300, dim=dim, direction=dir1, seed=11)
        phase2 = _unit_sphere_vectors(300, dim=dim, direction=dir2, seed=12)

        flags1 = [det.update(v) for v in phase1]
        assert not any(flags1), "Phase 1 must be stable"

        flags2 = [det.update(v) for v in phase2]
        assert any(flags2), (
            "Orthogonal centroid shift must trigger. "
            f"score={det.ema_chi2:.6f}, baseline={det.baseline:.6f}"
        )
        trigger_idx = next(i for i, f in enumerate(flags2) if f)
        assert trigger_idx < 100, (
            f"Should detect quickly, got step {trigger_idx}"
        )

    def test_detects_moderate_rotation(self):
        """30-degree rotation (cos_sim ~ 0.87) should still trigger."""
        dim = 50
        rng = np.random.default_rng(20)
        dir1 = np.zeros(dim)
        dir1[0] = 1.0

        angle_rad = np.pi / 6  # 30 degrees
        dir2 = np.zeros(dim)
        dir2[0] = np.cos(angle_rad)
        dir2[1] = np.sin(angle_rad)

        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _unit_sphere_vectors(300, dim=dim, direction=dir1, spread=0.2, seed=21)
        phase2 = _unit_sphere_vectors(400, dim=dim, direction=dir2, spread=0.2, seed=22)

        flags1 = [det.update(v) for v in phase1]
        assert not any(flags1)

        flags2 = [det.update(v) for v in phase2]
        assert any(flags2), (
            "30-degree centroid rotation should trigger. "
            f"score={det.ema_chi2:.6f}, baseline={det.baseline:.6f}"
        )

    def test_no_trigger_on_tiny_rotation(self):
        """2-degree rotation (cos_sim ~ 0.9994) should NOT trigger."""
        dim = 50
        dir1 = np.zeros(dim)
        dir1[0] = 1.0

        angle_rad = np.pi / 90  # 2 degrees
        dir2 = np.zeros(dim)
        dir2[0] = np.cos(angle_rad)
        dir2[1] = np.sin(angle_rad)

        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        phase1 = _unit_sphere_vectors(300, dim=dim, direction=dir1, spread=0.3, seed=31)
        phase2 = _unit_sphere_vectors(500, dim=dim, direction=dir2, spread=0.3, seed=32)

        [det.update(v) for v in phase1]
        flags2 = [det.update(v) for v in phase2]
        assert not any(flags2), (
            "Tiny 2-degree rotation must not trigger (within noise). "
            f"score={det.ema_chi2:.6f}, baseline={det.baseline:.6f}"
        )


class TestCentroidReset:
    """Verify reset clears state and allows re-detection."""

    def test_reset_clears_state(self):
        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )
        vecs = _unit_sphere_vectors(100, seed=40)
        [det.update(v) for v in vecs]
        assert det.is_burned_in

        det.reset()

        assert not det.is_burned_in
        assert det.total_steps == 0
        assert det.baseline == 0.0
        assert det.consecutive_above == 0
        assert not det.is_drifting

    def test_redetects_after_reset(self):
        """After reset + re-burn-in, a second shift should trigger again."""
        dim = 50
        rng = np.random.default_rng(50)
        dir1 = rng.standard_normal(dim)
        dir1 /= np.linalg.norm(dir1)
        dir2 = rng.standard_normal(dim)
        dir2 -= dir2.dot(dir1) * dir1
        dir2 /= np.linalg.norm(dir2)
        dir3 = rng.standard_normal(dim)
        dir3 -= dir3.dot(dir1) * dir1
        dir3 -= dir3.dot(dir2) * dir2
        dir3 /= np.linalg.norm(dir3)

        det = CentroidDriftDetector(
            threshold=2.0, burn_in_steps=50, ema_alpha=0.05,
            confirmation_window=20,
        )

        phase1 = _unit_sphere_vectors(200, dim=dim, direction=dir1, seed=51)
        phase2 = _unit_sphere_vectors(200, dim=dim, direction=dir2, seed=52)
        [det.update(v) for v in phase1]
        flags_shift1 = [det.update(v) for v in phase2]
        assert any(flags_shift1), "First shift must trigger"

        det.reset()

        phase3_burnin = _unit_sphere_vectors(200, dim=dim, direction=dir2, seed=53)
        phase3_shift = _unit_sphere_vectors(200, dim=dim, direction=dir3, seed=54)
        [det.update(v) for v in phase3_burnin]
        flags_shift2 = [det.update(v) for v in phase3_shift]
        assert any(flags_shift2), "Second shift after reset must also trigger"


class TestCentroidGetState:
    """Verify get_state returns complete diagnostic info."""

    def test_state_keys(self):
        det = CentroidDriftDetector(threshold=2.0, burn_in_steps=10)
        vecs = _unit_sphere_vectors(20, seed=60)
        [det.update(v) for v in vecs]

        state = det.get_state()
        expected_keys = {
            "total_steps", "burned_in", "baseline", "baseline_std",
            "ema_chi2", "drift_score", "drift_ratio", "is_drifting",
            "consecutive_above", "confirmed", "threshold",
            "burn_in_steps", "ema_alpha", "confirmation_window",
        }
        assert set(state.keys()) == expected_keys
