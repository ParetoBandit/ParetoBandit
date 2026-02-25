from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import LLMClient


class MultiProviderClient:
    """Dispatch ``complete()`` calls to the right provider client based on model ID.

    Model IDs in banditGPT use the ``provider/model-name`` convention
    (e.g. ``openai/gpt-4o``, ``anthropic/claude-3.5-sonnet``).
    ``MultiProviderClient`` extracts the provider prefix and forwards the
    call to the ``LLMClient`` registered for that prefix.

    This class itself satisfies the ``LLMClient`` protocol, so it works
    as a drop-in for ``route_and_call()``.

    Example::

        from bandit_gpt import (
            MultiProviderClient, OpenAIClient, AnthropicClient, OllamaClient,
        )

        client = MultiProviderClient({
            "openai":    OpenAIClient(api_key="sk-..."),
            "anthropic": AnthropicClient(api_key="sk-ant-..."),
            "meta-llama": OllamaClient(),
        })

        # The router selects "openai/gpt-4o" → dispatched to OpenAIClient
        # The router selects "anthropic/claude-3.5-sonnet" → dispatched to AnthropicClient
        model_id, response, log = router.route_and_call(prompt, client)
    """

    def __init__(
        self,
        providers: dict[str, LLMClient],
        *,
        default: LLMClient | None = None,
    ):
        """
        Parameters:
            providers: Mapping of provider prefix to ``LLMClient`` instance.
                       The prefix is the part before the ``/`` in a model ID
                       (e.g. ``"openai"``, ``"anthropic"``, ``"meta-llama"``).
            default:   Fallback client used when a model ID has no ``/`` or
                       its prefix isn't in *providers*.  If ``None``, a
                       ``KeyError`` is raised for unknown prefixes.
        """
        self._providers = dict(providers)
        self._default = default

    def register(self, prefix: str, client: LLMClient) -> None:
        """Add or replace a provider mapping at runtime."""
        self._providers[prefix] = client

    def _resolve(self, model_id: str) -> LLMClient:
        if "/" in model_id:
            prefix = model_id.split("/", 1)[0]
            client = self._providers.get(prefix)
            if client is not None:
                return client
        if self._default is not None:
            return self._default
        raise KeyError(
            f"No provider registered for model '{model_id}'. "
            f"Known prefixes: {sorted(self._providers)}. "
            f"Either register a client for this prefix or set a default client."
        )

    def complete(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        client = self._resolve(model_id)
        return client.complete(
            model_id, messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
