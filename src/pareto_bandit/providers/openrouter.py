from __future__ import annotations


class OpenRouterClient:
    """Adapter for the OpenRouter unified API (https://openrouter.ai).

    Uses the ``openai`` SDK pointed at the OpenRouter base URL.
    Model IDs follow the ``provider/model-name`` convention
    (e.g. ``openai/gpt-4o``, ``anthropic/claude-3.5-sonnet``).
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, *, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for OpenRouterClient. "
                "Install it with:  pip install paretobandit[openrouter]"
            ) from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
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
        resp = self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
