#!/usr/bin/env python3
"""
Test script to debug empty responses from Mistral 7B Instruct.
"""

import os
import sys

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Look for .env in current dir, parent dirs, or repo root
    for env_path in ['.env', '../.env', '../../.env', '../../../.env']:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
            break
except ImportError:
    pass

from openai import OpenAI
from datasets import load_dataset


def test_mistral_response():
    """Test Mistral 7B with HumanEval prompt #3."""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Load HumanEval Plus and get prompt 3
    print("Loading HumanEval Plus dataset...")
    dataset = load_dataset("evalplus/humanevalplus")
    
    # Get the 3rd problem (index 2)
    problems = list(dataset['test'])
    if len(problems) < 3:
        print("Not enough problems in dataset")
        return
    
    problem = problems[2]  # 0-indexed, so index 2 is problem 3
    prompt = problem['prompt']
    task_id = problem.get('task_id', 'unknown')
    entry_point = problem.get('entry_point', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"PROBLEM: {task_id} (entry point: {entry_point})")
    print(f"{'='*60}")
    print(prompt[:500])
    print("..." if len(prompt) > 500 else "")
    
    # Test different configurations
    # NOTE: Validation script uses max_tokens=8000 for Mistral 7B (non-reasoning, non-gemini3)
    test_configs = [
        {
            "name": "EXACT MATCH: Validation config (8000 tokens, no system)",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,  # Matches validation script exactly
        },
        {
            "name": "Lower tokens (1000)",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        },
        {
            "name": "With system prompt (8000 tokens)",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant. Write clean, working Python code."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8000,
        },
    ]
    
    model_id = "mistralai/mistral-7b-instruct"
    
    for config in test_configs:
        print(f"\n{'-'*60}")
        print(f"TEST: {config['name']}")
        print(f"{'-'*60}")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                max_tokens=config["max_tokens"],
                messages=config["messages"]
            )
            
            # Detailed response analysis
            print(f"  choices count: {len(response.choices)}")
            
            if response.choices:
                choice = response.choices[0]
                message = choice.message
                
                print(f"  finish_reason: {choice.finish_reason}")
                print(f"  content type: {type(message.content)}")
                print(f"  content is None: {message.content is None}")
                print(f"  content is empty string: {message.content == ''}")
                
                if hasattr(message, 'refusal') and message.refusal:
                    print(f"  ⚠️ REFUSAL: {message.refusal}")
                
                if message.content:
                    content = message.content.strip()
                    print(f"  content length: {len(content)} chars")
                    print(f"\n  RESPONSE PREVIEW (first 500 chars):")
                    print(f"  {'-'*40}")
                    print(f"  {content[:500]}")
                    if len(content) > 500:
                        print("  ...")
                else:
                    print(f"  ❌ NO CONTENT RECEIVED")
                    print(f"  raw content repr: {repr(message.content)}")
            else:
                print("  ❌ NO CHOICES IN RESPONSE")
                
            # Check usage if available
            if hasattr(response, 'usage') and response.usage:
                print(f"\n  Token usage:")
                print(f"    prompt_tokens: {response.usage.prompt_tokens}")
                print(f"    completion_tokens: {response.usage.completion_tokens}")
                print(f"    total_tokens: {response.usage.total_tokens}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
    
    # Also test with a simpler prompt
    print(f"\n{'='*60}")
    print("SIMPLE TEST: Basic hello world")
    print(f"{'='*60}")
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=200,
            messages=[{"role": "user", "content": "Write a Python function that prints hello world. Just the function code."}]
        )
        
        if response.choices and response.choices[0].message.content:
            print(f"✅ Response: {response.choices[0].message.content[:200]}")
        else:
            print(f"❌ Empty response")
            print(f"   finish_reason: {response.choices[0].finish_reason if response.choices else 'no choices'}")
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_mistral_response()
