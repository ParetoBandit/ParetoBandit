"""
Live integration tests for multi-provider LLM adapters.

These tests make REAL API calls and require valid keys in a .env file.
They are skipped by default — run them explicitly with:

    pytest tests/test_providers_live.py -m live

Every test sends a trivial arithmetic prompt ("What is 7 * 8?") and
asserts the response contains "56", which is model-agnostic and
deterministic at temperature 0.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CALL_KW = dict(max_tokens=32, temperature=0.0)
PROMPT = "What is 7 * 8? Reply with ONLY the number."
MESSAGES = [{"role": "user", "content": PROMPT}]


def _has_key(var: str) -> bool:
    return bool(os.environ.get(var, ""))


# ---------------------------------------------------------------------------
# Individual provider tests
# ---------------------------------------------------------------------------

@pytest.mark.live
@pytest.mark.skipif(not _has_key("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIClient:
    def test_complete(self):
        from pareto_bandit import OpenAIClient

        client = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.complete("openai/gpt-4o-mini", MESSAGES, **CALL_KW)
        assert "56" in resp.strip()

    def test_custom_base_url_together(self):
        """OpenAIClient with custom base_url (Together API)."""
        if not _has_key("TOGETHER_API_KEY"):
            pytest.skip("TOGETHER_API_KEY not set")
        from pareto_bandit import OpenAIClient

        client = OpenAIClient(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        resp = client.complete(
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo", MESSAGES, **CALL_KW,
        )
        assert "56" in resp.strip()

    def test_custom_base_url_xai(self):
        """OpenAIClient with custom base_url (xAI / Grok)."""
        if not _has_key("XAI_API_KEY"):
            pytest.skip("XAI_API_KEY not set")
        from pareto_bandit import OpenAIClient

        client = OpenAIClient(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
        )
        resp = client.complete("xai/grok-3-mini-fast", MESSAGES, **CALL_KW)
        assert "56" in resp.strip()


@pytest.mark.live
@pytest.mark.skipif(not _has_key("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicClient:
    def test_complete(self):
        from pareto_bandit import AnthropicClient

        client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.complete("anthropic/claude-sonnet-4-20250514", MESSAGES, **CALL_KW)
        assert "56" in resp.strip()


@pytest.mark.live
@pytest.mark.skipif(not _has_key("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
class TestOpenRouterClient:
    def test_complete(self):
        from pareto_bandit import OpenRouterClient

        client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"])
        resp = client.complete("openai/gpt-4o-mini", MESSAGES, **CALL_KW)
        assert "56" in resp.strip()


@pytest.mark.live
@pytest.mark.skipif(
    not (_has_key("GEMINI_API_KEY") or _has_key("GOOGLE_API_KEY")),
    reason="GEMINI_API_KEY / GOOGLE_API_KEY not set",
)
class TestGeminiClient:
    def test_complete(self):
        from pareto_bandit import GeminiClient

        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        client = GeminiClient(api_key=key)
        resp = client.complete("google/gemini-2.0-flash", MESSAGES, **CALL_KW)
        assert "56" in resp.strip()


# ---------------------------------------------------------------------------
# MultiProviderClient integration
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestMultiProviderClient:
    """End-to-end test: register models from multiple providers, route, call."""

    @pytest.fixture()
    def multi_client(self):
        from pareto_bandit import (
            MultiProviderClient, OpenAIClient, AnthropicClient, GeminiClient,
        )

        providers: dict[str, object] = {}
        if _has_key("OPENAI_API_KEY"):
            providers["openai"] = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
        if _has_key("ANTHROPIC_API_KEY"):
            providers["anthropic"] = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
        if _has_key("GEMINI_API_KEY") or _has_key("GOOGLE_API_KEY"):
            key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
            providers["google"] = GeminiClient(api_key=key)
        if _has_key("XAI_API_KEY"):
            providers["xai"] = OpenAIClient(
                api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1",
            )
        if len(providers) < 2:
            pytest.skip("Need at least 2 provider API keys for multi-provider test")

        return MultiProviderClient(providers), list(providers.keys())

    def test_dispatch_routes_to_correct_provider(self, multi_client):
        client, prefixes = multi_client
        model_map = {
            "openai": "openai/gpt-4o-mini",
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "google": "google/gemini-2.0-flash",
            "xai": "xai/grok-3-mini-fast",
        }
        for prefix in prefixes:
            mid = model_map.get(prefix)
            if mid is None:
                continue
            resp = client.complete(mid, MESSAGES, **CALL_KW)
            assert "56" in resp.strip(), f"{mid} returned unexpected: {resp!r}"

    def test_route_and_call(self, multi_client):
        from pareto_bandit import BanditRouter

        client, prefixes = multi_client
        model_map = {
            "openai": "openai/gpt-4o-mini",
            "anthropic": "anthropic/claude-sonnet-4-20250514",
            "google": "google/gemini-2.0-flash",
            "xai": "xai/grok-3-mini-fast",
        }
        registry = {}
        for prefix in prefixes:
            mid = model_map.get(prefix)
            if mid is None:
                continue
            registry[mid] = {"model_id": mid, "input_cost_per_m": 1.0, "output_cost_per_m": 1.0}

        router = BanditRouter.create(registry, priors="none", alpha=0.5)

        for _ in range(3):
            model_id, response, log = router.route_and_call(
                PROMPT, client, max_tokens=32, temperature=0.0,
            )
            assert model_id in registry, f"Routed to unknown model: {model_id}"
            assert "56" in response.strip(), f"{model_id} returned: {response!r}"
            router.process_feedback(log.request_id, reward=1.0)
