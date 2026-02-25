from __future__ import annotations


class OpenAIClient:
    """Adapter for the OpenAI API (and any OpenAI-compatible endpoint).

    Pass a custom ``base_url`` to target DeepSeek, Grok, Together, or
    any other provider that exposes an OpenAI-compatible chat endpoint.
    """

    def __init__(self, api_key: str, *, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for OpenAIClient. "
                "Install it with:  pip install banditgpt[openai]"
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _to_native_id(model_id: str) -> str:
        """Strip the ``provider/`` prefix used by the canonical registry."""
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
        resp = self._client.chat.completions.create(
            model=self._to_native_id(model_id),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
