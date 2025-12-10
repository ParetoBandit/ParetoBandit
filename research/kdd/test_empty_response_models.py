#!/usr/bin/env python3
"""
Test MiniMax-M2 and Gemini 3 Pro Preview with different parameters to find what works.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


def test_model_params(model_id: str, prompt: str = "Answer only yes or no: Is the sky blue?") -> Dict[str, Any]:
    """Test different parameter combinations for a model."""
    
    client = OpenAI(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
    
    results = {}
    
    # Test 1: max_tokens with temperature=0 (standard non-reasoning)
    print(f"\n1️⃣  Testing max_tokens=100, temperature=0")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = response.choices[0].message.content
        results['max_tokens_temp0'] = {
            'success': True,
            'content': content,
            'empty': not content or not content.strip(),
            'finish_reason': response.choices[0].finish_reason,
        }
        print(f"   ✅ Success: {repr(content[:100])}")
    except Exception as e:
        results['max_tokens_temp0'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 2: max_tokens with higher limit, temperature=0
    print(f"\n2️⃣  Testing max_tokens=500, temperature=0")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        content = response.choices[0].message.content
        results['max_tokens_500_temp0'] = {
            'success': True,
            'content': content,
            'empty': not content or not content.strip(),
            'finish_reason': response.choices[0].finish_reason,
        }
        print(f"   ✅ Success: {repr(content[:100])}")
    except Exception as e:
        results['max_tokens_500_temp0'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 3: max_completion_tokens (reasoning model style)
    print(f"\n3️⃣  Testing max_completion_tokens=500")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
        )
        content = response.choices[0].message.content
        results['max_completion_tokens'] = {
            'success': True,
            'content': content,
            'empty': not content or not content.strip(),
            'finish_reason': response.choices[0].finish_reason,
        }
        print(f"   ✅ Success: {repr(content[:100])}")
    except Exception as e:
        results['max_completion_tokens'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # Test 4: No temperature parameter
    print(f"\n4️⃣  Testing max_tokens=500, no temperature")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        results['max_tokens_no_temp'] = {
            'success': True,
            'content': content,
            'empty': not content or not content.strip(),
            'finish_reason': response.choices[0].finish_reason,
        }
        print(f"   ✅ Success: {repr(content[:100])}")
    except Exception as e:
        results['max_tokens_no_temp'] = {'success': False, 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    return results


def main():
    print("="*80)
    print("Testing Models with Empty Response Issues")
    print("="*80)
    
    models_to_test = [
        ("minimax/minimax-m2", "MiniMax-M2"),
        ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview"),
    ]
    
    for model_id, name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"Model ID: {model_id}")
        print(f"{'='*80}")
        
        results = test_model_params(model_id)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY for {name}")
        print(f"{'='*80}")
        
        for test_name, result in results.items():
            if result.get('success'):
                status = "❌ EMPTY" if result.get('empty') else "✅ GOT CONTENT"
                print(f"{test_name:30} {status}")
            else:
                print(f"{test_name:30} ❌ ERROR: {result.get('error', '')[:50]}")
        
        print()


if __name__ == "__main__":
    main()
