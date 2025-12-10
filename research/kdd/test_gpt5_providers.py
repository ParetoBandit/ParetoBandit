#!/usr/bin/env python3
"""
Compare GPT-5 responses from OpenAI direct vs OpenRouter.

This script calls GPT-5 through both providers to understand:
1. What parameter (max_tokens vs max_completion_tokens) works for each
2. How the response structure differs
3. Where the actual text content is located

Usage:
    python research/kdd/test_gpt5_providers.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


def load_gpt5_models() -> Dict[str, Any]:
    """Load GPT-5 model configurations from models_cache.json."""
    cache_path = DATA_PATH / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    
    models_list = data.get("models", data)
    gpt5_models = {}
    
    for model in models_list:
        name = model.get("name", "")
        openrouter_id = model.get("openrouter_id", "")
        
        # Find GPT-5 variants
        if "gpt-5" in name.lower() and openrouter_id:
            gpt5_models[name] = {
                "name": name,
                "slug": model.get("slug", ""),
                "openrouter_id": openrouter_id,
                "provider_model": openrouter_id.replace("openai/", ""),
            }
    
    return gpt5_models


def extract_content(message: Any) -> Dict[str, Any]:
    """Extract all content fields from a message object."""
    result = {
        "content": None,
        "reasoning_content": None,
        "role": None,
        "text_extracted": ""
    }
    
    # Get raw fields
    result["content"] = getattr(message, "content", None)
    result["reasoning_content"] = getattr(message, "reasoning_content", None)
    result["role"] = getattr(message, "role", None)
    
    # Try to extract text from content
    content = result["content"]
    if isinstance(content, str):
        result["text_extracted"] = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        result["text_extracted"] = "\n".join(parts)
    
    # If no text from content, try reasoning_content
    if not result["text_extracted"] and result["reasoning_content"]:
        reasoning = result["reasoning_content"]
        if isinstance(reasoning, str):
            result["text_extracted"] = reasoning
        elif isinstance(reasoning, list):
            parts = []
            for item in reasoning:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            result["text_extracted"] = "\n".join(parts)
    
    return result


def call_openai_direct(model_id: str, prompt: str, tokens: int = 500) -> Dict[str, Any]:
    """Call GPT-5 directly via OpenAI API."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not found in environment"}
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=tokens,
            temperature=0,
        )
        
        choice = response.choices[0]
        content_info = extract_content(choice.message)
        
        return {
            "provider": "openai_direct",
            "model_sent": model_id,
            "success": True,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "content_info": content_info,
        }
    except Exception as e:
        return {
            "provider": "openai_direct",
            "model_sent": model_id,
            "success": False,
            "error": str(e),
        }


def call_openrouter(model_id: str, prompt: str, tokens: int = 500) -> Dict[str, Any]:
    """Call GPT-5 via OpenRouter API."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not found in environment"}
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=tokens,
            temperature=0,
        )
        
        choice = response.choices[0]
        content_info = extract_content(choice.message)
        
        return {
            "provider": "openrouter",
            "model_sent": model_id,
            "success": True,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "content_info": content_info,
        }
    except Exception as e:
        return {
            "provider": "openrouter",
            "model_sent": model_id,
            "success": False,
            "error": str(e),
        }


def main():
    print("="*80)
    print("GPT-5 Provider Comparison Test")
    print("="*80)
    
    # Load GPT-5 models from cache
    gpt5_models = load_gpt5_models()
    
    if not gpt5_models:
        print("❌ No GPT-5 models found in models_cache.json")
        return
    
    print(f"\nFound {len(gpt5_models)} GPT-5 variant(s):")
    for name, info in gpt5_models.items():
        print(f"  • {name}")
        print(f"    OpenRouter ID: {info['openrouter_id']}")
        print(f"    Direct OpenAI ID: {info['provider_model']}")
    
    # Test prompt
    prompt = "Is the sky blue? Answer only 'Yes' or 'No'."
    print(f"\n📝 Test prompt: {prompt}")
    
    # Test each GPT-5 variant
    for name, info in gpt5_models.items():
        print("\n" + "="*80)
        print(f"Testing: {name}")
        print("="*80)
        
        # Test OpenAI Direct
        print("\n1️⃣  OpenAI Direct API")
        print("-"*80)
        openai_result = call_openai_direct(info['provider_model'], prompt)
        print(json.dumps(openai_result, indent=2, ensure_ascii=False))
        
        if openai_result.get("success"):
            text = openai_result.get("content_info", {}).get("text_extracted", "")
            print(f"\n✅ Extracted text: '{text}'")
        else:
            print(f"\n❌ Failed: {openai_result.get('error')}")
        
        # Test OpenRouter
        print("\n2️⃣  OpenRouter API")
        print("-"*80)
        openrouter_result = call_openrouter(info['openrouter_id'], prompt)
        print(json.dumps(openrouter_result, indent=2, ensure_ascii=False))
        
        if openrouter_result.get("success"):
            text = openrouter_result.get("content_info", {}).get("text_extracted", "")
            print(f"\n✅ Extracted text: '{text}'")
        else:
            print(f"\n❌ Failed: {openrouter_result.get('error')}")
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        openai_success = openai_result.get("success", False)
        openrouter_success = openrouter_result.get("success", False)
        
        openai_text = openai_result.get("content_info", {}).get("text_extracted", "") if openai_success else ""
        openrouter_text = openrouter_result.get("content_info", {}).get("text_extracted", "") if openrouter_success else ""
        
        print(f"OpenAI Direct:  {'✅ Success' if openai_success else '❌ Failed'}")
        print(f"OpenRouter:     {'✅ Success' if openrouter_success else '❌ Failed'}")
        
        if openai_success and openrouter_success:
            if openai_text == openrouter_text:
                print(f"✅ Both returned identical text: '{openai_text}'")
            else:
                print(f"⚠️  Different responses:")
                print(f"   OpenAI:     '{openai_text[:100]}'")
                print(f"   OpenRouter: '{openrouter_text[:100]}'")
        
        print()
    
    print("\n" + "="*80)
    print("✅ Test Complete")
    print("="*80)


if __name__ == "__main__":
    main()
