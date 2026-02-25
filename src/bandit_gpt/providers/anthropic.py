from __future__ import annotations


class AnthropicClient:
    """Adapter for the Anthropic Messages API.

    Model IDs in the canonical registry use the ``anthropic/model-name``
    format; this adapter strips the prefix automatically.
    """

    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The anthropic package is required for AnthropicClient. "
                "Install it with:  pip install banditgpt[anthropic]"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _to_native_id(model_id: str) -> str:
        return model_id.split("/", 1)[-1] if "/" in model_id else model_id

    def complete(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        resp = self._client.messages.create(
            model=self._to_native_id(model_id),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp.content[0].text
