#!/usr/bin/env python3
"""
Fetch evaluation logs for open-source proxy models.

This script downloads instance-level evaluation logs (prompt + pass/fail) from:
- OpenCompass (for Llama, DeepSeek, Qwen)
- EvalPlus (for coding benchmarks)
- AgentBench/ToolBench (for agentic tasks)

These logs enable training a Logistic Regression predictor on open models,
which can then be transferred to predict closed-model performance using their
Artificial Analysis benchmark scores.

Recommended Proxy Models:
- Reasoning: DeepSeek R1, Llama 3.1 70B
- Coding: Qwen 2.5 72B
- Agentic: Mixtral 8x22B
- RAG: Llama 3.3 70B
- General: Phi-4 (small model anchor)

Usage:
    python fetch_open_source_proxy_logs.py --intent reasoning --output logs/
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess


# Proxy model configuration
PROXY_MODELS = {
    "reasoning": {
        "models": [
            {
                "name": "DeepSeek R1",
                "slug": "deepseek-r1",
                "source": "opencompass",
                "hf_repo": "opencompass/llm-leaderboard",
                "benchmarks": ["gpqa", "mmlu_pro", "hle", "math"],
                "why": "Massive public logs on HuggingFace. Covers high reasoning spectrum."
            },
            {
                "name": "Llama 3.1 70B",
                "slug": "llama-3-1-instruct-70b",
                "source": "opencompass",
                "hf_repo": "meta-llama/Llama-3.1-70B-Instruct-evals",
                "benchmarks": ["gpqa", "mmlu_pro", "math"],
                "why": "Meta publishes extensive evaluation logs. Standard baseline."
            }
        ],
        "target_closed": ["o1", "o3", "GPT-5.1", "Claude Opus 4.5"],
        "shared_trait": "High Math Index & Hard Logic scores"
    },
    "coding": {
        "models": [
            {
                "name": "Qwen 2.5 72B",
                "slug": "qwen2-5-72b-instruct",
                "source": "evalplus",
                "hf_repo": "evalplus/evalplus",
                "benchmarks": ["humaneval", "mbpp", "livecodebench"],
                "why": "King of Open Code. EvalPlus publishes full results.json."
            }
        ],
        "target_closed": ["Claude 4.5 Sonnet", "GPT-5.1", "o3"],
        "shared_trait": "High Coding Index"
    },
    "agentic": {
        "models": [
            {
                "name": "Mixtral 8x22B",
                "slug": "mistral-8x22b-instruct",
                "source": "agentbench",
                "hf_repo": "THUDM/AgentBench",
                "benchmarks": ["gaia", "toolbench"],
                "why": "Widely tested on AgentBench/ToolBench. Open tool-use logs."
            }
        ],
        "target_closed": ["GPT-4o", "Claude 4 Opus"],
        "shared_trait": "High tool-use & instruction-following"
    },
    "rag": {
        "models": [
            {
                "name": "Llama 3.3 70B",
                "slug": "llama-3-3-instruct-70b",
                "source": "rgb_leaderboard",
                "hf_repo": "meta-llama/Llama-3.3-70B-Instruct-evals",
                "benchmarks": ["natural_questions", "hotpotqa"],
                "why": "Standard RAG baseline. Abundant retrieval-QA logs on RGB."
            }
        ],
        "target_closed": ["GPT-5.1", "Gemini 2.5 Pro"],
        "shared_trait": "High retrieval precision"
    },
    "general": {
        "models": [
            {
                "name": "Phi-4",
                "slug": "phi-4",
                "source": "microsoft",
                "hf_repo": "microsoft/phi-4",
                "benchmarks": ["mmlu", "arc", "hellaswag"],
                "why": "Small model anchor. High intelligence, low capacity."
            }
        ],
        "target_closed": [],
        "shared_trait": "Efficiency baseline"
    }
}


class LogFetcher:
    """Fetch evaluation logs from various sources."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_opencompass_logs(self, model_info: Dict) -> Path:
        """
        Fetch logs from OpenCompass.
        
        OpenCompass logs are typically hosted on HuggingFace or GitHub.
        """
        print(f"\n📊 Fetching OpenCompass logs for {model_info['name']}...")
        
        model_dir = self.output_dir / model_info['slug']
        model_dir.mkdir(exist_ok=True)
        
        # Try HuggingFace datasets API
        hf_repo = model_info.get('hf_repo')
        if hf_repo:
            print(f"  Checking HuggingFace repo: {hf_repo}")
            
            # Download using datasets library
            try:
                from datasets import load_dataset
                
                for benchmark in model_info['benchmarks']:
                    try:
                        print(f"    Loading {benchmark}...")
                        
                        # Try loading as dataset
                        dataset = load_dataset(
                            hf_repo,
                            benchmark,
                            split='test',
                            trust_remote_code=True
                        )
                        
                        # Save as JSON
                        output_file = model_dir / f"{benchmark}_results.json"
                        
                        results = []
                        for item in dataset:
                            results.append(item)
                        
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        print(f"    ✓ Saved {len(results)} results to {output_file}")
                        
                    except Exception as e:
                        print(f"    ⚠ Could not load {benchmark}: {e}")
                        
            except ImportError:
                print("  ⚠ datasets library not installed. Run: pip install datasets")
        
        # Alternative: Direct download from known URLs
        opencompass_urls = {
            "deepseek-r1": {
                "gpqa": "https://huggingface.co/datasets/opencompass/deepseek-r1-evals/resolve/main/gpqa.json",
                "math": "https://huggingface.co/datasets/opencompass/deepseek-r1-evals/resolve/main/math.json"
            },
            "llama-3-1-instruct-70b": {
                "mmlu_pro": "https://huggingface.co/datasets/meta-llama/Llama-3.1-70B-Instruct-evals/resolve/main/mmlu_pro.json"
            }
        }
        
        slug = model_info['slug']
        if slug in opencompass_urls:
            print(f"  Trying direct download URLs...")
            for benchmark, url in opencompass_urls[slug].items():
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        output_file = model_dir / f"{benchmark}_results.json"
                        with open(output_file, 'w') as f:
                            f.write(response.text)
                        print(f"    ✓ Downloaded {benchmark} to {output_file}")
                    else:
                        print(f"    ⚠ {benchmark}: HTTP {response.status_code}")
                except Exception as e:
                    print(f"    ⚠ {benchmark}: {e}")
        
        return model_dir
    
    def fetch_evalplus_logs(self, model_info: Dict) -> Path:
        """
        Fetch logs from EvalPlus.
        
        EvalPlus provides detailed coding evaluation logs.
        """
        print(f"\n💻 Fetching EvalPlus logs for {model_info['name']}...")
        
        model_dir = self.output_dir / model_info['slug']
        model_dir.mkdir(exist_ok=True)
        
        # EvalPlus GitHub API
        evalplus_base = "https://github.com/evalplus/evalplus/raw/master/data"
        
        for benchmark in model_info['benchmarks']:
            try:
                if benchmark in ['humaneval', 'mbpp']:
                    # Download test cases and results
                    url = f"{evalplus_base}/{benchmark}_results.json"
                    print(f"  Downloading {benchmark}...")
                    
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        output_file = model_dir / f"{benchmark}_results.json"
                        with open(output_file, 'w') as f:
                            f.write(response.text)
                        print(f"    ✓ Saved to {output_file}")
                    else:
                        print(f"    ⚠ HTTP {response.status_code}")
                        
            except Exception as e:
                print(f"    ⚠ {benchmark}: {e}")
        
        return model_dir
    
    def fetch_agentbench_logs(self, model_info: Dict) -> Path:
        """
        Fetch logs from AgentBench.
        
        AgentBench evaluates tool use and agentic capabilities.
        """
        print(f"\n🤖 Fetching AgentBench logs for {model_info['name']}...")
        
        model_dir = self.output_dir / model_info['slug']
        model_dir.mkdir(exist_ok=True)
        
        # AgentBench HuggingFace repo
        print("  Note: AgentBench logs may require manual download from:")
        print("  https://github.com/THUDM/AgentBench")
        print("  https://huggingface.co/datasets/THUDM/AgentBench")
        
        return model_dir
    
    def fetch_logs(self, intent: str, model_info: Dict) -> Path:
        """Fetch logs based on source type."""
        source = model_info.get('source', 'opencompass')
        
        if source == 'opencompass':
            return self.fetch_opencompass_logs(model_info)
        elif source == 'evalplus':
            return self.fetch_evalplus_logs(model_info)
        elif source == 'agentbench':
            return self.fetch_agentbench_logs(model_info)
        elif source in ['rgb_leaderboard', 'microsoft']:
            return self.fetch_opencompass_logs(model_info)
        else:
            print(f"⚠ Unknown source: {source}")
            return self.output_dir / model_info['slug']


def generate_transfer_learning_table() -> str:
    """Generate the proxy transfer learning table for the paper."""
    
    table = """
## Proxy Transfer Learning Strategy

| Training Proxy (Open) | → | Target Model (Closed) | Shared AA Trait |
|-----------------------|---|----------------------|-----------------|
"""
    
    for intent, config in PROXY_MODELS.items():
        if config['target_closed']:
            for model in config['models']:
                targets = ", ".join(config['target_closed'][:2])  # First 2
                table += f"| {model['name']:<21} | → | {targets:<20} | {config['shared_trait']} |\n"
    
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Fetch evaluation logs for open-source proxy models"
    )
    parser.add_argument(
        "--intent",
        type=str,
        choices=["reasoning", "coding", "agentic", "rag", "general", "all"],
        default="all",
        help="Intent category to fetch logs for"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="proxy_logs",
        help="Output directory for logs"
    )
    parser.add_argument(
        "--show-strategy",
        action="store_true",
        help="Show the proxy transfer learning strategy"
    )
    
    args = parser.parse_args()
    
    if args.show_strategy:
        print("="*80)
        print("PROXY TRANSFER LEARNING STRATEGY")
        print("="*80)
        print(generate_transfer_learning_table())
        
        print("\n" + "="*80)
        print("METHODOLOGY STATEMENT")
        print("="*80)
        print("""
We trained our Performance Predictor P exclusively on the open-weights subset
(Llama 3.3, Qwen 2.5, DeepSeek R1, Mixtral 8x22B, Phi-4), leveraging their 
public inference traces from OpenCompass, EvalPlus, and AgentBench.

We then applied this predictor zero-shot to proprietary models (GPT-5.1, o3,
Gemini 3 Pro, Claude Opus 4.5) by projecting their Artificial Analysis (AA)
benchmark scores into the learned coefficient space.

This "proxy transfer" approach enables training on verifiable open-source data
while predicting performance for closed models without requiring inference access.
        """)
        return 0
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fetcher = LogFetcher(output_dir)
    
    print("="*80)
    print("OPEN-SOURCE PROXY LOG FETCHER")
    print("="*80)
    
    # Determine which intents to fetch
    intents_to_fetch = [args.intent] if args.intent != "all" else list(PROXY_MODELS.keys())
    
    for intent in intents_to_fetch:
        config = PROXY_MODELS[intent]
        
        print(f"\n{'='*80}")
        print(f"Intent: {intent.upper()}")
        print(f"{'='*80}")
        print(f"Target Closed Models: {', '.join(config['target_closed']) if config['target_closed'] else 'N/A'}")
        print(f"Shared Trait: {config['shared_trait']}")
        
        for model_info in config['models']:
            print(f"\n{'-'*80}")
            print(f"Model: {model_info['name']}")
            print(f"Why: {model_info['why']}")
            print(f"Source: {model_info['source']}")
            print(f"Benchmarks: {', '.join(model_info['benchmarks'])}")
            
            model_dir = fetcher.fetch_logs(intent, model_info)
            
            # Check what was downloaded
            downloaded_files = list(model_dir.glob("*.json"))
            if downloaded_files:
                print(f"\n✓ Downloaded {len(downloaded_files)} file(s) to {model_dir}")
                for f in downloaded_files:
                    size_kb = f.stat().st_size / 1024
                    print(f"  • {f.name} ({size_kb:.1f} KB)")
            else:
                print(f"\n⚠ No files downloaded. Manual download may be required.")
                print(f"  Check: {model_info.get('hf_repo', 'N/A')}")
    
    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_models = sum(len(config['models']) for config in PROXY_MODELS.values())
    print(f"Total proxy models configured: {total_models}")
    print(f"Logs saved to: {output_dir.absolute()}")
    
    print("\nNext Steps:")
    print("1. Review downloaded logs in each model directory")
    print("2. Extract instance-level labels (prompt + pass/fail)")
    print("3. Train Logistic Regression on open-source logs")
    print("4. Apply predictor to closed models using their AA scores")
    
    return 0


if __name__ == "__main__":
    exit(main())
