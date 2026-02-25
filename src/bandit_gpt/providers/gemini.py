from __future__ import annotations


class GeminiClient:
    """Adapter for the Google Gemini API via the ``google-genai`` SDK.

    Model IDs in the canonical registry use the ``google/model-name``
    format; this adapter strips the prefix automatically.
    """

    def __init__(self, api_key: str):
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The google-genai package is required for GeminiClient. "
                "Install it with:  pip install banditgpt[gemini]"
            ) from exc

        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_native_id(model_id: str) -> str:
        return model_id.split("/", 1)[-1] if "/" in model_id else model_id

    @staticmethod
    def _messages_to_contents(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-style messages to Gemini content parts."""
        contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    def complete(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = self._client.models.generate_content(
            model=self._to_native_id(model_id),
            contents=self._messages_to_contents(messages),
            config=config,
        )
        return resp.text or ""
