"""
Tests for Constraint-Aware Routing Architecture.

Tests the 3-Stage Routing Funnel:
    Phase 1: Hard Filtering (min_quality, max_cost, max_latency)
    Phase 2: Bandit Selection
    Phase 3: Cascade Decision (cascade_rate / λ)

Also tests the Unified Architecture where Standard Mode = λ=0.
"""

import pytest
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def model_registry():
    """Create a test model registry with known benchmark scores.
    
    Note: The 'cost' field is per-request (not per-1k). The router multiplies
    by 1000 to get $/1k. So cost=0.0001 means $0.10/1k queries.
    """
    return {
        "cheap-weak": {
            "cost": 0.0001,  # $0.10/1k queries
            "latency_s": 0.5,
            "scores": {"math": 30, "code": 25, "reasoning": 35, "mmlu": 40, "avg": 32.5},
        },
        "cheap-good": {
            "cost": 0.0005,  # $0.50/1k queries
            "latency_s": 1.0,
            "scores": {"math": 70, "code": 75, "reasoning": 72, "mmlu": 68, "avg": 71.25},
        },
        "mid-excellent": {
            "cost": 0.002,  # $2.00/1k queries
            "latency_s": 1.5,
            "scores": {"math": 85, "code": 90, "reasoning": 88, "mmlu": 82, "avg": 86.25},
        },
        "expensive-best": {
            "cost": 0.010,  # $10.00/1k queries
            "latency_s": 2.5,
            "scores": {"math": 95, "code": 92, "reasoning": 94, "mmlu": 91, "avg": 93.0},
        },
        "fast-medium": {
            "cost": 0.001,  # $1.00/1k queries
            "latency_s": 0.3,
            "scores": {"math": 60, "code": 65, "reasoning": 58, "mmlu": 62, "avg": 61.25},
        },
    }


@pytest.fixture
def hybrid_router(model_registry):
    """Create a HybridRouter for testing."""
    from banditgpt.core.bandit_router import HybridRouter
    
    return HybridRouter.create(
        model_registry=model_registry,
        fallback_model="expensive-best",
        cascade_rate=0.0,  # Standard Mode by default
        priors="none",  # Cold start for predictable behavior
        exploration="static",  # No exploration noise
    )


# =============================================================================
# Phase 1: Hard Filtering Tests
# =============================================================================

class TestPhase1HardFiltering:
    """Test that hard constraints filter the candidate pool correctly."""
    
    def test_min_quality_filters_weak_models(self, hybrid_router):
        """min_quality should exclude models with low benchmark scores."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            min_quality=70,  # Should exclude cheap-weak (32.5%) and fast-medium (61.25%)
        )
        
        # Should not select models below quality floor
        assert model not in ["cheap-weak", "fast-medium"]
    
    def test_max_cost_filters_expensive_models(self, hybrid_router):
        """max_cost should exclude models above budget."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            max_cost=1.00,  # Should exclude mid-excellent ($2) and expensive-best ($10)
        )
        
        # Should not select models above cost limit
        assert model not in ["mid-excellent", "expensive-best"]
    
    def test_max_latency_filters_slow_models(self, hybrid_router):
        """max_latency should exclude slow models."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            max_latency=1.0,  # Should exclude mid-excellent (1.5s) and expensive-best (2.5s)
        )
        
        # Should not select slow models
        assert model not in ["mid-excellent", "expensive-best"]
    
    def test_combined_constraints(self, hybrid_router):
        """Multiple constraints should be AND-ed together."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            min_quality=50,  # Excludes cheap-weak (32.5%)
            max_cost=5.00,   # Excludes expensive-best ($10)
        )
        
        # Only cheap-good, mid-excellent, fast-medium should remain
        assert model in ["cheap-good", "mid-excellent", "fast-medium"]
    
    def test_quality_floor_dict_format(self, hybrid_router):
        """quality_floor dict should filter by specific benchmark."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            quality_floor={"code": 80},  # Only mid-excellent (90%) and expensive-best (92%)
        )
        
        assert model in ["mid-excellent", "expensive-best"]


# =============================================================================
# Phase 3: Cascade Decision Tests
# =============================================================================

class TestPhase3CascadeDecision:
    """Test that cascade_rate (λ) controls verification correctly."""
    
    def test_cascade_rate_zero_always_single_shot(self, hybrid_router):
        """λ=0 should always return single_shot mode."""
        hybrid_router.cascade_rate = 0.0
        
        # Run multiple times to ensure consistency
        for _ in range(10):
            model, log, mode = hybrid_router.route("Test prompt")
            assert mode == "single_shot"
    
    def test_cascade_rate_one_always_cascade(self, hybrid_router):
        """λ=1 should always return cascade mode."""
        hybrid_router.cascade_rate = 1.0
        
        # Run multiple times to ensure consistency
        for _ in range(10):
            model, log, mode = hybrid_router.route("Test prompt")
            assert mode == "cascade"
    
    def test_cascade_rate_affects_probability(self, hybrid_router):
        """λ=0.5 should cascade roughly half the time."""
        hybrid_router.cascade_rate = 0.5
        
        cascade_count = 0
        total = 100
        for _ in range(total):
            model, log, mode = hybrid_router.route("Test prompt")
            if mode == "cascade":
                cascade_count += 1
        
        # Should be roughly 50% ± some variance
        # Using wide bounds due to probabilistic nature
        assert 20 < cascade_count < 80


# =============================================================================
# Unified Architecture Tests
# =============================================================================

class TestUnifiedArchitecture:
    """Test that Standard Mode = λ=0 with all constraints still applied."""
    
    def test_standard_mode_is_lambda_zero(self, model_registry):
        """Creating with cascade_rate=0 should be Standard Mode."""
        from banditgpt.core.bandit_router import HybridRouter
        
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            cascade_rate=0.0,
            priors="none",
        )
        
        assert router.cascade_rate == 0.0
        
        model, log, mode = router.route("Test")
        assert mode == "single_shot"
    
    def test_standard_mode_still_applies_constraints(self, hybrid_router):
        """Standard mode (λ=0) should still enforce hard constraints."""
        hybrid_router.cascade_rate = 0.0  # Standard Mode
        
        model, log, mode = hybrid_router.route(
            "Test prompt",
            min_quality=50,   # Excludes cheap-weak (32.5%)
            max_cost=5.00,    # Excludes expensive-best ($10)
        )
        
        # Should be single_shot (Standard Mode)
        assert mode == "single_shot"
        
        # Constraints should still be applied
        assert model not in ["cheap-weak"]  # Below quality floor
        assert model not in ["expensive-best"]  # Above cost limit
    
    def test_hybrid_mode_applies_same_constraints(self, hybrid_router):
        """Hybrid mode (λ>0) should apply the same constraints as Standard."""
        hybrid_router.cascade_rate = 1.0  # Always cascade
        
        model, log, mode = hybrid_router.route(
            "Test prompt",
            min_quality=50,   # Excludes cheap-weak (32.5%)
            max_cost=5.00,    # Excludes expensive-best ($10)
        )
        
        # Should be cascade (Hybrid Mode)
        assert mode == "cascade"
        
        # Same constraints should be applied
        assert model not in ["cheap-weak"]  # Below quality floor
        assert model not in ["expensive-best"]  # Above cost limit


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Test that deprecated parameters still work."""
    
    def test_verification_threshold_alias(self, model_registry):
        """verification_threshold should work as alias for cascade_rate."""
        from banditgpt.core.bandit_router import HybridRouter
        
        # Create with deprecated parameter
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            verification_threshold=0.5,  # Deprecated
            priors="none",
        )
        
        # Should set cascade_rate
        assert router.cascade_rate == 0.5
    
    def test_verification_threshold_in_route(self, hybrid_router):
        """verification_threshold in route() should work."""
        hybrid_router.cascade_rate = 0.0  # Default to standard
        
        # Use deprecated parameter in route call
        model, log, mode = hybrid_router.route(
            "Test prompt",
            verification_threshold=1.0,  # Should override to always cascade
        )
        
        assert mode == "cascade"
    
    def test_verification_presets_alias(self, model_registry):
        """VERIFICATION_PRESETS should alias CASCADE_PRESETS."""
        from banditgpt.core.bandit_router import HybridRouter
        
        assert HybridRouter.VERIFICATION_PRESETS == HybridRouter.CASCADE_PRESETS


# =============================================================================
# Named Preset Tests
# =============================================================================

class TestCascadePresets:
    """Test named presets for cascade_rate."""
    
    def test_preset_cost_optimal(self, model_registry):
        """cost_optimal preset should set λ=0."""
        from banditgpt.core.bandit_router import HybridRouter
        
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            mode="cost_optimal",
            priors="none",
        )
        
        assert router.cascade_rate == 0.0
    
    def test_preset_balanced(self, model_registry):
        """balanced preset should set λ=0.5."""
        from banditgpt.core.bandit_router import HybridRouter
        
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            mode="balanced",
            priors="none",
        )
        
        assert router.cascade_rate == 0.5
    
    def test_preset_max_accuracy(self, model_registry):
        """max_accuracy preset should set λ=1.0."""
        from banditgpt.core.bandit_router import HybridRouter
        
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            mode="max_accuracy",
            priors="none",
        )
        
        assert router.cascade_rate == 1.0
    
    def test_cascade_rate_overrides_preset(self, model_registry):
        """Explicit cascade_rate should override mode preset."""
        from banditgpt.core.bandit_router import HybridRouter
        
        router = HybridRouter.create(
            model_registry=model_registry,
            fallback_model="expensive-best",
            mode="max_accuracy",  # Would set λ=1.0
            cascade_rate=0.3,     # But this should override
            priors="none",
        )
        
        assert router.cascade_rate == 0.3


# =============================================================================
# min_quality Shorthand Tests
# =============================================================================

class TestMinQualityShorthand:
    """Test that min_quality is shorthand for quality_floor={'avg': X}."""
    
    def test_min_quality_filters_correctly(self, hybrid_router):
        """min_quality=70 should filter models with avg < 70."""
        model, log, mode = hybrid_router.route(
            "Test prompt",
            min_quality=70,
        )
        
        # cheap-weak (32.5%), fast-medium (61.25%) should be filtered
        assert model not in ["cheap-weak", "fast-medium"]
    
    def test_min_quality_equivalent_to_quality_floor(self, hybrid_router):
        """min_quality should be equivalent to quality_floor={'avg': X}."""
        # Both should produce the same filtering
        model1, _, _ = hybrid_router.route("Test", min_quality=85)
        model2, _, _ = hybrid_router.route("Test", quality_floor={"avg": 85})
        
        # Both should only allow mid-excellent (86.25%) and expensive-best (93%)
        assert model1 in ["mid-excellent", "expensive-best"]
        assert model2 in ["mid-excellent", "expensive-best"]
