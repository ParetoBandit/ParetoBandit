#!/usr/bin/env python3
"""
Debug GPT-5 responses specifically for SummEdits prompts.
This mimics the exact behavior of run_summedits.py to diagnose the empty response issue.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from openai import OpenAI

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
SUMMEDITS_PATH = PROJECT_ROOT / "factualNLG" / "data" / "summedits"
SUMMEDITS_PROMPTS = PROJECT_ROOT / "factualNLG" / "prompts" / "summedits"

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


def load_prompt_template() -> str:
    """Load the standard zero-shot prompt template."""
    prompt_path = SUMMEDITS_PROMPTS / "standard_zs_prompt.txt"
    with open(prompt_path) as f:
        return f.read()


def format_prompt(doc: str, summary: str, template: str) -> str:
    """Format a document-summary pair as a prompt."""
    prompt = template.replace("[ARTICLE]", doc)
    prompt = prompt.replace("[SUMMARY_SENTENCES]", summary)
    return prompt


def load_sample() -> dict:
    """Load one sample from SummEdits news dataset."""
    file_path = SUMMEDITS_PATH / "summedits_news.json"
    with open(file_path) as f:
        data = json.load(f)
    
    # Get first evaluation sample
    for item in data:
        if item.get("split") == "evaluation":
            return item
    
    return {}


def extract_content(message: Any) -> dict:
    """Extract all content from a message."""
    return {
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "role": getattr(message, "role", None),
    }


def call_with_debug(model_id: str, prompt: str, tokens: int) -> dict:
    """Call OpenRouter with full debugging."""
    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
    
    print(f"Calling {model_id} with max_completion_tokens={tokens}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt preview: {prompt[:200]}...")
    print()
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=tokens,
        )
        
        choice = response.choices[0]
        message = choice.message
        
        # Get raw response
        raw_content = getattr(message, "content", None)
        
        result = {
            "success": True,
            "finish_reason": choice.finish_reason,
            "raw_content": raw_content,
            "raw_content_type": type(raw_content).__name__,
            "message_attrs": dir(message),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
        }
        
        # Check if content is a list
        if isinstance(raw_content, list):
            result["content_list_length"] = len(raw_content)
            result["content_list_items"] = [
                {
                    "type": type(item).__name__,
                    "value": str(item)[:200] if not isinstance(item, dict) else item
                }
                for item in raw_content
            ]
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    print("="*80)
    print("GPT-5 SummEdits Debug")
    print("="*80)
    
    # Load actual SummEdits data
    print("\n1. Loading SummEdits sample...")
    sample = load_sample()
    if not sample:
        print("❌ Failed to load sample")
        return
    
    print(f"✅ Loaded sample")
    print(f"   Doc length: {len(sample.get('doc', ''))} chars")
    print(f"   Summary length: {len(sample.get('summary', ''))} chars")
    print(f"   Label: {sample.get('label')}")
    
    # Load prompt template
    print("\n2. Loading prompt template...")
    template = load_prompt_template()
    print(f"✅ Template loaded: {len(template)} chars")
    
    # Format prompt
    print("\n3. Formatting prompt...")
    prompt = format_prompt(sample['doc'], sample['summary'], template)
    print(f"✅ Final prompt: {len(prompt)} chars")
    
    # Test with different token limits
    model_id = "openai/gpt-5"
    
    for tokens in [500, 1000, 2000]:
        print("\n" + "="*80)
        print(f"Testing with {tokens} tokens")
        print("="*80)
        
        result = call_with_debug(model_id, prompt, tokens)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        
        if result.get("success"):
            raw_content = result.get("raw_content")
            print(f"\n📊 Analysis:")
            print(f"   Finish reason: {result['finish_reason']}")
            print(f"   Content type: {result['raw_content_type']}")
            print(f"   Content is empty: {not raw_content}")
            print(f"   Content is empty string: {raw_content == ''}")
            print(f"   Content value: {repr(raw_content)}")
            
            if result['finish_reason'] == 'length':
                print(f"   ⚠️  Response was truncated (finish_reason=length)")
            
            if not raw_content:
                print(f"   ❌ EMPTY RESPONSE - THIS IS THE BUG")
            else:
                print(f"   ✅ Got content: {raw_content[:100]}")
        else:
            print(f"\n❌ Error: {result.get('error')}")
        
        print()


if __name__ == "__main__":
    main()
