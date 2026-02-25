from __future__ import annotations


class OllamaClient:
    """Adapter for a local Ollama instance (no API key required).

    Communicates with the Ollama HTTP API (default: ``http://localhost:11434``).
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, *, base_url: str | None = None):
        try:
            from ollama import Client
        except ImportError as exc:
            raise ImportError(
                "The ollama package is required for OllamaClient. "
                "Install it with:  pip install banditgpt[ollama]"
            ) from exc

        self._client = Client(host=base_url or self.DEFAULT_BASE_URL)

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
        resp = self._client.chat(
            model=self._to_native_id(model_id),
            messages=messages,
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        return resp["message"]["content"]
