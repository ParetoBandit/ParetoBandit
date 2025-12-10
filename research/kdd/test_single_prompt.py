#!/usr/bin/env python3
"""Test a single model with one SummEdits prompt."""

import os
import json
import argparse
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='Test a single model with one SummEdits prompt')
    parser.add_argument('--model', type=str, default='minimax/minimax-m2', help='Model ID')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature')
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    # Load one sample
    data_path = Path(__file__).parent.parent.parent / 'factualNLG/data/summedits/summedits_news.json'
    with open(data_path) as f:
        data = json.load(f)

    sample = [d for d in data if d.get('split') == 'evaluation'][0]

    # Load prompt template
    prompt_path = Path(__file__).parent.parent.parent / 'factualNLG/prompts/summedits/standard_zs_prompt.txt'
    with open(prompt_path) as f:
        template = f.read()

    prompt = template.replace("[ARTICLE]", sample['doc']).replace("[SUMMARY_SENTENCES]", sample['summary'])

    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Expected answer: {'Yes' if sample.get('label') == 1 else 'No'}")
    print("=" * 60)
    print(f"Prompt preview:\n{prompt[:500]}...")
    print("=" * 60)

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    print(f"\nResponse: {repr(content)}")
    print(f"Finish reason: {finish_reason}")
    print(f"Content length: {len(content) if content else 0}")

if __name__ == '__main__':
    main()

