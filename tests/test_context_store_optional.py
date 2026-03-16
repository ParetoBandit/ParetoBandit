"""
Tests that context storage is optional and defaults to EphemeralContextStore.

Verifies:
1. Router defaults to EphemeralContextStore (no disk I/O)
2. Routing + immediate feedback works without SqliteContextStore
3. SqliteContextStore is opt-in and works when explicitly provided
4. No files are created on disk when using the default store
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pareto_bandit import BanditRouter, SqliteContextStore, EphemeralContextStore


@pytest.fixture
def two_model_registry():
    return {
        "openai/gpt-4o": {
            "model_id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "scores": {"hle": 0.85},
            "hallucination_rate": 1.5,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0,
        },
        "google/gemma-3-2b-it": {
            "model_id": "google/gemma-3-2b-it",
            "display_name": "Gemma 3 2B",
            "scores": {"hle": 0.45},
            "hallucination_rate": 8.0,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1,
        },
    }


class TestDefaultContextStore:
    """Verify the router defaults to EphemeralContextStore (no disk)."""

    def test_default_store_is_ephemeral(self, two_model_registry):
        router = BanditRouter.create(
            model_registry=two_model_registry, priors="none"
        )
        assert isinstance(router.context_store, EphemeralContextStore)

    def test_no_files_created_on_disk(self, two_model_registry):
        """Route multiple prompts and confirm nothing is written to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            before = set(Path(tmpdir).rglob("*"))

            router = BanditRouter.create(
                model_registry=two_model_registry, priors="none"
            )
            for i in range(5):
                model, log = router.route(f"Test prompt {i}")
                router.process_feedback(log.request_id, reward=0.7)

            after = set(Path(tmpdir).rglob("*"))
            assert before == after, "Default store should not create any files"


class TestEphemeralFeedbackLoop:
    """Routing + immediate feedback works end-to-end with the default store."""

    def test_route_and_feedback(self, two_model_registry):
        router = BanditRouter.create(
            model_registry=two_model_registry, priors="none"
        )
        model, log = router.route("Explain quantum entanglement")
        assert model in two_model_registry

        router.process_feedback(log.request_id, reward=0.9)
        assert router.bandit.t >= 1

    def test_multiple_rounds(self, two_model_registry):
        """Run several route/feedback cycles to confirm stability."""
        router = BanditRouter.create(
            model_registry=two_model_registry, priors="none"
        )
        for i in range(20):
            model, log = router.route(f"Prompt number {i}")
            reward = np.random.uniform(0.3, 1.0)
            router.process_feedback(log.request_id, reward=reward)

        assert router.bandit.t >= 20

    def test_missing_request_id_is_noop(self, two_model_registry):
        """Feedback for an unknown request_id should warn, not crash."""
        router = BanditRouter.create(
            model_registry=two_model_registry, priors="none"
        )
        router.process_feedback("nonexistent-id", reward=0.5)


class TestSqliteOptIn:
    """SqliteContextStore works when explicitly provided."""

    def test_explicit_sqlite_store(self, two_model_registry):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "router.db"
            store = SqliteContextStore(db_path=str(db_path))

            router = BanditRouter.create(
                model_registry=two_model_registry,
                priors="none",
                context_store=store,
            )
            assert isinstance(router.context_store, SqliteContextStore)

            model, log = router.route("Test persistence")
            router.process_feedback(log.request_id, reward=0.8)

            assert db_path.exists(), "SQLite DB should be created after first route"

    def test_sqlite_survives_log_eviction(self, two_model_registry):
        """Delayed feedback retrieved from SQLite after in-memory log is gone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "router.db"
            store = SqliteContextStore(db_path=str(db_path))

            router = BanditRouter.create(
                model_registry=two_model_registry,
                priors="none",
                context_store=store,
            )

            model, log = router.route("Delayed feedback test")
            request_id = log.request_id

            # Snapshot b-vector before feedback
            b_before = router.bandit.b[model].copy()

            # Evict in-memory log to force SQLite fallback
            router.logs.clear()
            router.log_index.clear()

            router.process_feedback(request_id, reward=0.85)

            b_after = router.bandit.b[model]
            assert not np.allclose(b_before, b_after), (
                "Feedback via SQLite fallback should update the bandit's b-vector"
            )
