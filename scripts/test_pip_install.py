#!/usr/bin/env python3
"""
Smoke-test for bandit_gpt installed via pip.

Exercises every public API surface using *only* synthetic data and
pre-computed context vectors so no model downloads or API keys are needed.

Usage:
    pip install banditgpt          # or: pip install -e .
    python scripts/test_pip_install.py
"""

from __future__ import annotations

import sys
import time
import numpy as np

DIM = 33  # 32 PCA components + 1 bias (matches default pca_32.joblib)
N_MODELS = 5
RNG = np.random.default_rng(42)

PASS = 0
FAIL = 0


def status(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(seed: int | None = None) -> np.ndarray:
    """Random unit context vector with bias term = 1.0."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0)


def synthetic_registry() -> dict:
    """Minimal model registry matching the structure BanditRouter expects."""
    tiers = [
        ("cheap-fast/model-a",   0.10,  60.0, 0.20, 0.55),
        ("cheap-fast/model-b",   0.15,  55.0, 0.25, 0.60),
        ("mid-tier/model-c",     2.00,  30.0, 0.80, 0.75),
        ("premium/model-d",      10.0,  15.0, 1.50, 0.88),
        ("premium/model-e",      15.0,  10.0, 2.00, 0.92),
    ]
    registry = {}
    for name, cost, tps, ttft, quality in tiers:
        registry[name] = {
            "name": name,
            "model_id": name,
            "input_cost_per_m": cost,
            "output_cost_per_m": cost * 1.5,
            "output_tokens_per_second": tps,
            "time_to_first_token_seconds": ttft,
            "initial_quality": quality,
        }
    return registry


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------

def test_imports() -> None:
    print("\n1. Import check")
    try:
        from bandit_gpt import BanditRouter, ExplorationRate, RouterConfig, FeatureService
        status("Public API imports", True)
    except ImportError as e:
        status("Public API imports", False, str(e))
        sys.exit(1)

    try:
        from bandit_gpt.router import RoutingLog
        status("RoutingLog importable", True)
    except ImportError as e:
        status("RoutingLog importable", False, str(e))


# ---------------------------------------------------------------------------
# 2. Router construction
# ---------------------------------------------------------------------------

def test_construction() -> dict:
    print("\n2. Router construction")
    from bandit_gpt import BanditRouter, RouterConfig

    registry = synthetic_registry()
    router = BanditRouter(
        model_registry=registry,
        feature_service=_make_passthrough_feature_service(),
        alpha=0.1,
        embedding_dim=DIM,
        init_lambda=1.0,
        use_corralling=False,
    )
    status("BanditRouter.__init__", True, f"{len(registry)} models, dim={DIM}")

    ok = set(router.bandit.models) == set(registry)
    status("All models registered", ok,
           f"expected {sorted(registry)}, got {sorted(router.bandit.models)}")

    return {"router": router, "registry": registry}


def _make_passthrough_feature_service():
    """A lightweight FeatureService stand-in that returns pre-computed vectors as-is."""
    from bandit_gpt import FeatureService

    class _Passthrough(FeatureService):
        """Lightweight stand-in that skips model downloads entirely."""

        def __init__(self):
            self._dimension = DIM
            self._encoder = None

            class _FakePCA:
                n_components = DIM - 1  # 32 PCA components (bias is separate)

            self._pca = _FakePCA()

        @property
        def dimension(self) -> int:
            return self._dimension

        @property
        def encoder(self):
            return self._encoder

        @property
        def pca(self):
            return self._pca

        @property
        def using_pca(self) -> bool:
            return True

        def get_dimension(self) -> int:
            return self._dimension

        def extract_features(self, prompt):
            if isinstance(prompt, np.ndarray):
                return prompt
            return make_context()

        def extract_features_batch(self, prompts):
            return np.stack([self.extract_features(p) for p in prompts])

        def get_feature_names(self):
            return [f"pca_{i}" for i in range(DIM - 1)] + ["bias"]

    return _Passthrough()


# ---------------------------------------------------------------------------
# 3. Basic routing
# ---------------------------------------------------------------------------

def test_basic_routing(router, registry: dict) -> None:
    print("\n3. Basic routing")
    x = make_context(seed=7)
    model_id, log = router.route(x)

    status("route() returns model id", model_id in registry,
           f"got '{model_id}'")
    status("route() returns RoutingLog", hasattr(log, "selected_model"),
           f"type={type(log).__name__}")
    status("log.selected_model matches", log.selected_model == model_id)
    status("log.context_vector is set", log.context_vector is not None)
    status("log.cost_usd is numeric", isinstance(log.cost_usd, (int, float)))

    # Seed slight differentiation so UCB scores aren't perfectly tied
    for i, m in enumerate(registry):
        router.update(m, make_context(seed=i), reward=0.5 + 0.05 * i)
    models_seen = set()
    for i in range(50):
        mid, _ = router.route(make_context(seed=i + 100))
        models_seen.add(mid)
    status("Exploration covers multiple models", len(models_seen) > 1,
           f"saw {len(models_seen)} distinct models")


# ---------------------------------------------------------------------------
# 4. Online learning
# ---------------------------------------------------------------------------

def test_learning(router, registry: dict) -> None:
    print("\n4. Online learning (reward signal)")
    good_model = "premium/model-e"
    bad_model = "cheap-fast/model-a"

    before_theta_good = router.bandit.A_inv[good_model] @ router.bandit.b[good_model]
    before_theta_bad = router.bandit.A_inv[bad_model] @ router.bandit.b[bad_model]

    ctx = make_context(seed=0)
    for _ in range(200):
        router.update(good_model, ctx, reward=0.95)
        router.update(bad_model, ctx, reward=0.15)

    after_theta_good = router.bandit.A_inv[good_model] @ router.bandit.b[good_model]
    after_theta_bad = router.bandit.A_inv[bad_model] @ router.bandit.b[bad_model]

    good_improved = np.linalg.norm(after_theta_good) > np.linalg.norm(before_theta_good)
    bad_shrank = (after_theta_good @ ctx) > (after_theta_bad @ ctx)

    status("Positive rewards increase theta norm", good_improved)
    status("Good model preferred over bad after training", bad_shrank,
           f"UCB(good)={after_theta_good @ ctx:.4f}, UCB(bad)={after_theta_bad @ ctx:.4f}")

    selections = {}
    for _ in range(100):
        mid, _ = router.route(ctx)
        selections[mid] = selections.get(mid, 0) + 1
    top = max(selections, key=selections.get)
    status("Learned model is top selection", top == good_model,
           f"top={top} ({selections.get(top, 0)}/100)")


# ---------------------------------------------------------------------------
# 5. Cost / latency constraints
# ---------------------------------------------------------------------------

def test_constraints(router, registry: dict) -> None:
    print("\n5. Cost and latency constraints")
    x = make_context(seed=55)

    model_id, _ = router.route(x, max_cost=0.001)
    model_cost = registry[model_id].get("input_cost_per_m", 0)
    status("max_cost filters expensive models", model_cost <= 1.0,
           f"selected {model_id} (cost={model_cost})")

    model_id2, _ = router.route(x, max_latency=0.5)
    model_ttft = registry[model_id2].get("time_to_first_token_seconds", 0)
    status("max_latency filters slow models", model_ttft <= 1.0,
           f"selected {model_id2} (ttft={model_ttft})")


# ---------------------------------------------------------------------------
# 6. Model registration (progressive)
# ---------------------------------------------------------------------------

def test_registration(router) -> None:
    print("\n6. Progressive model registration")
    new_id = "new-vendor/model-x"
    router.register_model(
        model_id=new_id,
        speed="fast",
        cost_usd=0.05,
        latency_s=0.15,
    )
    status("register_model succeeds", new_id in router.registry,
           f"registry has {len(router.registry)} models")

    mid, _ = router.route(make_context(seed=99))
    all_models = set(router.bandit.models)
    status("New model is routable", new_id in all_models)


# ---------------------------------------------------------------------------
# 7. Explainability
# ---------------------------------------------------------------------------

def test_explainability(router) -> None:
    print("\n7. Decision explainability")
    x = make_context(seed=42)
    mid, log = router.route(x)

    try:
        contribs = router.explain_decision(mid, x)
        has_entries = len(contribs) > 0
        status("explain_decision returns contributions", has_entries,
               f"{len(contribs)} features")
    except Exception as e:
        status("explain_decision", False, str(e))

    try:
        probs = router.get_probabilities(x)
        sums_to_one = abs(sum(probs.values()) - 1.0) < 0.01
        status("get_probabilities sums to ~1", sums_to_one,
               f"sum={sum(probs.values()):.4f}")
    except Exception as e:
        status("get_probabilities", False, str(e))


# ---------------------------------------------------------------------------
# 8. ExplorationRate presets
# ---------------------------------------------------------------------------

def test_exploration_presets() -> None:
    print("\n8. ExplorationRate presets")
    from bandit_gpt import ExplorationRate

    for name, expected in [("static", 0.0), ("safe", 0.1), ("balanced", 1.0), ("aggressive", 2.0)]:
        val = ExplorationRate.get(name)
        status(f"ExplorationRate.get('{name}')", val == expected, f"{val}")

    val = ExplorationRate.get(0.42)
    status("ExplorationRate.get(float) passthrough", val == 0.42)


# ---------------------------------------------------------------------------
# 9. RouterConfig
# ---------------------------------------------------------------------------

def test_config() -> None:
    print("\n9. RouterConfig")
    from bandit_gpt import RouterConfig

    cfg = RouterConfig()
    status("Default config instantiates", True)
    status("init_lambda default is 1.0", cfg.init_lambda == 1.0)
    status("max_log_size default is 10000", cfg.max_log_size == 10_000)

    custom = RouterConfig(init_lambda=2.0)
    status("Custom config accepted", custom.init_lambda == 2.0)


# ---------------------------------------------------------------------------
# 10. CLI entry point
# ---------------------------------------------------------------------------

def test_cli_entrypoint() -> None:
    print("\n10. CLI entry point")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "bandit_gpt.cli", "--version"],
        capture_output=True, text=True, timeout=30,
    )
    ok = result.returncode == 0 and "v0.1.0" in result.stdout
    status("python -m bandit_gpt.cli --version", ok,
           result.stdout.strip() or result.stderr.strip())


# ---------------------------------------------------------------------------
# 11. Convergence under specialization
# ---------------------------------------------------------------------------

def test_specialization_convergence() -> None:
    """
    Two synthetic 'domains' (orthogonal context vectors), each with a
    designated best model.  After training, the router should route each
    domain to the correct specialist.
    """
    print("\n11. Specialization convergence (2-domain synthetic)")
    from bandit_gpt import BanditRouter

    registry = synthetic_registry()
    router = BanditRouter(
        model_registry=registry,
        feature_service=_make_passthrough_feature_service(),
        alpha=0.05,
        embedding_dim=DIM,
        use_corralling=False,
    )

    domain_a = np.zeros(DIM)
    domain_a[0] = 1.0
    domain_a[-1] = 1.0  # bias

    domain_b = np.zeros(DIM)
    domain_b[1] = 1.0
    domain_b[-1] = 1.0

    specialist_a = "cheap-fast/model-a"
    specialist_b = "premium/model-d"

    for _ in range(300):
        router.update(specialist_a, domain_a, reward=0.90)
        router.update(specialist_b, domain_b, reward=0.90)
        for other in registry:
            if other != specialist_a:
                router.update(other, domain_a, reward=0.30)
            if other != specialist_b:
                router.update(other, domain_b, reward=0.30)

    hits_a, hits_b = 0, 0
    trials = 50
    for _ in range(trials):
        mid_a, _ = router.route(domain_a)
        mid_b, _ = router.route(domain_b)
        if mid_a == specialist_a:
            hits_a += 1
        if mid_b == specialist_b:
            hits_b += 1

    status(f"Domain A -> {specialist_a}", hits_a > trials * 0.7,
           f"{hits_a}/{trials}")
    status(f"Domain B -> {specialist_b}", hits_b > trials * 0.7,
           f"{hits_b}/{trials}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    logging.disable(logging.WARNING)

    print("=" * 60)
    print("  bandit_gpt  pip-install smoke test")
    print("=" * 60)

    test_imports()
    test_exploration_presets()
    test_config()

    ctx = test_construction()
    router, registry = ctx["router"], ctx["registry"]

    test_basic_routing(router, registry)
    test_learning(router, registry)
    test_constraints(router, registry)
    test_registration(router)
    test_explainability(router)
    test_cli_entrypoint()
    test_specialization_convergence()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
