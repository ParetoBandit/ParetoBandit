#!/usr/bin/env python3
"""
Small debug tool to inspect OpenRouter responses for reasoning models.

Example:
  python research/kdd/debug_openrouter_response.py --model openai/gpt-5
"""

import argparse
import json
import os
import sys
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def flatten_message(message: Any) -> str:
    """Extract text from message.content/reasoning_content."""
    def flatten_parts(parts: Any) -> List[str]:
        out: List[str] = []
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict):
                    text_val = part.get("text") or part.get("content") or ""
                else:
                    text_val = str(part)
                if text_val:
                    out.append(text_val)
        elif isinstance(parts, str):
            out.append(parts)
        return out

    content = getattr(message, "content", None)
    parts: List[str] = flatten_parts(content)

    # Some reasoning models return reasoning_content instead of content
    reasoning = getattr(message, "reasoning_content", None)
    parts += flatten_parts(reasoning)

    return "\n".join([p for p in parts if p]).strip()


def _normalize_model_id(model: str, direct_openai: bool) -> str:
    """If calling OpenAI directly, strip provider prefix like openai/."""
    if direct_openai and "/" in model:
        return model.split("/", 1)[1]
    return model


def call_model(model: str, prompt: str, use_max_completion: bool, direct_openai: bool) -> Dict[str, Any]:
    from openai import OpenAI

    model_id = _normalize_model_id(model, direct_openai)

    if direct_openai:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        provider = "openai"
    else:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        provider = "openrouter"

    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_max_completion:
        kwargs["max_completion_tokens"] = 100
    else:
        kwargs["max_tokens"] = 100

    try:
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message
        extracted = flatten_message(message)

        # Build a lightweight debug dict
        debug = {
            "model": model,
            "model_sent": model_id,
            "provider": provider,
            "use_max_completion": use_max_completion,
            "finish_reason": getattr(choice, "finish_reason", None),
            "raw_message": {
                "content": getattr(message, "content", None),
                "reasoning_content": getattr(message, "reasoning_content", None),
                "role": getattr(message, "role", None),
            },
            "extracted_text": extracted,
        }
        return debug
    except Exception as e:
        # Return the error so the caller can see the failure without crashing
        return {
            "model": model,
            "model_sent": model_id,
            "provider": provider,
            "use_max_completion": use_max_completion,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Debug OpenRouter response content")
    parser.add_argument("--model", required=True, help="Model id, e.g., openai/gpt-5")
    parser.add_argument(
        "--prompt",
        default="Answer only yes or no: Is the sky blue?",
        help="Prompt to send",
    )
    parser.add_argument(
        "--direct-openai",
        action="store_true",
        help="Call OpenAI directly using OPENAI_API_KEY (not via OpenRouter)",
    )
    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    print("1) Using max_completion_tokens")
    debug_completion = call_model(
        args.model, args.prompt, use_max_completion=True, direct_openai=args.direct_openai
    )
    print(json.dumps(debug_completion, indent=2, ensure_ascii=False))

    print("\n2) Using max_tokens")
    debug_tokens = call_model(
        args.model, args.prompt, use_max_completion=False, direct_openai=args.direct_openai
    )
    print(json.dumps(debug_tokens, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

