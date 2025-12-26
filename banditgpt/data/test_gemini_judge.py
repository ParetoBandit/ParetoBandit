"""
Test script for Gemini-3-Flash judge to verify max_tokens requirements.
"""

import os
import requests
import json
from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    # Look for .env in repository root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
    else:
        print(f".env not found at {env_path}")
except ImportError:
    print("python-dotenv not installed, using system environment only")
except Exception as e:
    print(f"Could not load .env: {e}")

def test_gemini_flash_judge(max_tokens=10):
    """Test Gemini-3-Flash as a judge with different max_tokens."""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        return
    
    base_url = "https://openrouter.ai/api/v1"
    model = "google/gemini-3-flash-preview"
    
    # Test prompt and response
    test_prompt = "What is 2+2?"
    test_response = "The answer is 4. This is basic arithmetic."
    
    system_prompt = (
        "You are an impartial judge. Rate the quality of the response to the prompt.\n"
        "Output ONLY a single float number between 0.0 and 1.0.\n"
        "0.0 = Completely wrong, harmful, or unhelpful.\n"
        "0.5 = Partially correct but missing key details.\n"
        "1.0 = Perfectly correct, helpful, and comprehensive.\n"
        "Do not output any other text."
    )
    
    user_content = f"PROMPT: {test_prompt}\n\nRESPONSE: {test_response}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/banditgpt/llm-jury",
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {model}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "unknown")
        
        print(f"✅ Success!")
        print(f"Response: {content}")
        print(f"Finish reason: {finish_reason}")
        
        # Check if response was truncated
        if finish_reason == "length":
            print(f"⚠️  WARNING: Response truncated due to max_tokens limit")
            return False
        
        # Try to parse score
        import re
        match = re.search(r"(\d+(\.\d+)?)", content)
        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
            print(f"Parsed score: {score}")
            return True
        else:
            print(f"⚠️  WARNING: Could not parse score from: {content}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'response' in locals():
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Response text: {response.text}")
        return False

def main():
    print("\nGemini-3-Flash Judge Test")
    print("="*60)
    
    # Test with different max_tokens values
    test_configs = [
        ("Small (10 tokens)", 10),
        ("Medium (50 tokens)", 50),
        ("Large (100 tokens)", 100),
        ("Very Large (500 tokens)", 500),
    ]
    
    results = {}
    for name, max_tokens in test_configs:
        success = test_gemini_flash_judge(max_tokens)
        results[name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}\n")
    
    # Find minimum successful max_tokens
    successful = [tokens for (name, tokens), success in zip(test_configs, results.values()) if success]
    if successful:
        min_tokens = min(successful)
        print(f"Minimum max_tokens that works: {min_tokens}")
        print(f"Recommended max_tokens: {min_tokens * 2} (with safety margin)")
    else:
        print("No configuration worked! Check API key and model availability.")

if __name__ == "__main__":
    main()
