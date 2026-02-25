from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every provider adapter must satisfy.

    Implementations only need to provide ``complete``; the router never
    imports or instantiates a client directly, so any object that
    structurally matches this protocol works — no inheritance required.
    """

    def complete(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Send a chat-completion request and return the text response."""
        ...
