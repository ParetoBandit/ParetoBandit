"""
Build Intent Classification Training Dataset.

Target Distribution (reflects production reality):
- REASONING (15%): GSM8k-style math problems
- CODING (15%): MBPP-style code problems  
- FACTUAL_QA (15%): Natural Questions-style
- AGENTIC (15%): Tool/API use requests
- GENERAL (40%): Chitchat, creative writing, opinions (LMSYS filtered)

Data Sources:
- REASONING: GSM8k, MATH
- CODING: MBPP, CodeAlpaca
- FACTUAL_QA: Natural Questions, TriviaQA
- AGENTIC: Glaive, ToolBench
- GENERAL: LMSYS-Chat-1M (filtered), UltraChat, DailyDialog

The "Subtraction Method" for GENERAL:
1. Remove code blocks (```)
2. Remove math/LaTeX ($, \frac, numbers)
3. Remove tool triggers (weather, price, schedule, book)
4. Remove fact-seeking patterns (When was, Define, What is the)
5. Remove long prompts (>50 words)
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import Counter

try:
    from datasets import load_dataset
    from tqdm import tqdm
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets not available. Install with: pip install datasets")


# =============================================================================
# Filtering Functions for GENERAL (Subtraction Method)
# =============================================================================

def has_code_blocks(text: str) -> bool:
    """Check if text contains code blocks or code keywords."""
    # Markdown code blocks
    if '```' in text:
        return True
    # Common code patterns
    code_patterns = [
        r'\bdef\s+\w+\s*\(',      # Python function
        r'\bfunction\s+\w+\s*\(',  # JS function
        r'\bclass\s+\w+',          # Class definition
        r'\bimport\s+\w+',         # Import statement
        r'\bfrom\s+\w+\s+import',  # Python import
        r'\breturn\s+',            # Return statement
        r'^\s*#include',           # C/C++ include
    ]
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    # Code keywords
    code_keywords = ['python', 'javascript', 'java', 'c++', 'rust', 'golang', 
                     'typescript', 'sql', 'html', 'css', 'algorithm', 'debug',
                     'compile', 'syntax error', 'runtime error']
    text_lower = text.lower()
    return any(kw in text_lower for kw in code_keywords)


def has_math_symbols(text: str) -> bool:
    """Check if text contains math/LaTeX symbols or extensive numbers."""
    # LaTeX patterns
    if '$' in text or '\\frac' in text or '\\int' in text or '\\sum' in text:
        return True
    # Math symbols
    math_symbols = ['∫', '∑', '∏', '√', '≠', '≤', '≥', '±', '∞', '∂', '∇']
    if any(s in text for s in math_symbols):
        return True
    # Equations with operators
    if re.search(r'\d+\s*[\+\-\*\/\=]\s*\d+', text):
        return True
    # Math keywords
    math_keywords = ['equation', 'solve for x', 'calculate', 'integral', 
                     'derivative', 'theorem', 'proof', 'probability']
    text_lower = text.lower()
    return any(kw in text_lower for kw in math_keywords)


def has_tool_triggers(text: str) -> bool:
    """Check if text contains tool/API trigger keywords."""
    tool_keywords = [
        'weather', 'price of', 'stock price', 'schedule', 'book a', 
        'reserve', 'send email', 'send message', 'call api', 'fetch',
        'search the web', 'browse', 'download', 'upload', 'execute',
        'run this command', 'terminal', 'shell'
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in tool_keywords)


def has_fact_seeking_pattern(text: str) -> bool:
    """Check if text starts with fact-seeking patterns."""
    fact_patterns = [
        r'^when was\b',
        r'^when did\b', 
        r'^who (is|was|are|were)\b',
        r'^what is the (capital|population|name|date)\b',
        r'^define\b',
        r'^how many\b',
        r'^how much\b',
        r'^what year\b',
        r'^in what year\b',
    ]
    text_lower = text.lower().strip()
    return any(re.match(p, text_lower) for p in fact_patterns)


def is_too_long(text: str, max_words: int = 50) -> bool:
    """Check if text exceeds word limit."""
    return len(text.split()) > max_words


def is_general_prompt(text: str) -> bool:
    """
    Apply subtraction method to identify GENERAL prompts.
    Returns True if the prompt is GENERAL (doesn't match other categories).
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Apply subtraction filters
    if has_code_blocks(text):
        return False
    if has_math_symbols(text):
        return False
    if has_tool_triggers(text):
        return False
    if has_fact_seeking_pattern(text):
        return False
    if is_too_long(text, max_words=50):
        return False
    
    return True


# =============================================================================
# Data Fetching Functions
# =============================================================================

def fetch_reasoning_prompts(limit: int = 500) -> List[Dict]:
    """Fetch REASONING prompts from GSM8k."""
    print(f"\n📊 Fetching REASONING prompts (target: {limit})...")
    prompts = []
    
    try:
        # GSM8k - math word problems
        ds = load_dataset("gsm8k", "main", split="train")
        for item in tqdm(ds, desc="GSM8k"):
            prompts.append({
                "prompt": item["question"],
                "label": "reasoning",
                "source": "gsm8k"
            })
            if len(prompts) >= limit:
                break
    except Exception as e:
        print(f"  Error loading GSM8k: {e}")
    
    print(f"  ✓ Got {len(prompts)} REASONING prompts")
    return prompts[:limit]


def fetch_coding_prompts(limit: int = 500) -> List[Dict]:
    """Fetch CODING prompts from MBPP."""
    print(f"\n💻 Fetching CODING prompts (target: {limit})...")
    prompts = []
    
    try:
        # MBPP - code problems
        ds = load_dataset("mbpp", split="train")
        for item in tqdm(ds, desc="MBPP"):
            prompts.append({
                "prompt": item["text"],
                "label": "coding",
                "source": "mbpp"
            })
            if len(prompts) >= limit:
                break
    except Exception as e:
        print(f"  Error loading MBPP: {e}")
    
    print(f"  ✓ Got {len(prompts)} CODING prompts")
    return prompts[:limit]


def fetch_factual_qa_prompts(limit: int = 500) -> List[Dict]:
    """Fetch FACTUAL_QA prompts from Natural Questions or TriviaQA."""
    print(f"\n📚 Fetching FACTUAL_QA prompts (target: {limit})...")
    prompts = []
    
    try:
        # TriviaQA - factual questions
        ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=True)
        for item in tqdm(ds, desc="TriviaQA"):
            prompts.append({
                "prompt": item["question"],
                "label": "factual_qa",
                "source": "trivia_qa"
            })
            if len(prompts) >= limit:
                break
    except Exception as e:
        print(f"  Error loading TriviaQA: {e}")
        
        # Fallback to Natural Questions
        try:
            ds = load_dataset("natural_questions", split="train")
            for item in tqdm(ds, desc="NQ"):
                q = item.get("question", {}).get("text", "")
                if q:
                    prompts.append({
                        "prompt": q,
                        "label": "factual_qa", 
                        "source": "natural_questions"
                    })
                if len(prompts) >= limit:
                    break
        except Exception as e2:
            print(f"  Error loading Natural Questions: {e2}")
    
    print(f"  ✓ Got {len(prompts)} FACTUAL_QA prompts")
    return prompts[:limit]


def fetch_agentic_prompts(limit: int = 500) -> List[Dict]:
    """Fetch AGENTIC prompts from Glaive or ToolBench."""
    print(f"\n🔧 Fetching AGENTIC prompts (target: {limit})...")
    prompts = []
    
    try:
        # Glaive function calling dataset
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        for item in tqdm(ds, desc="Glaive"):
            # Extract user message
            chat = item.get("chat", "")
            if "USER:" in chat:
                user_msg = chat.split("USER:")[1].split("ASSISTANT:")[0].strip()
                if user_msg and len(user_msg) > 10:
                    prompts.append({
                        "prompt": user_msg,
                        "label": "agentic",
                        "source": "glaive"
                    })
            if len(prompts) >= limit:
                break
    except Exception as e:
        print(f"  Error loading Glaive: {e}")
    
    # Supplement with ToolBench if needed
    if len(prompts) < limit:
        try:
            ds = load_dataset("Yhyu13/ToolBench_toolllama_G123_dfs", split="train")
            for item in tqdm(ds, desc="ToolBench"):
                convs = item.get("conversations", [])
                if len(convs) > 1:
                    user_msg = convs[1].get("value", "")
                    if user_msg and len(user_msg) > 10:
                        prompts.append({
                            "prompt": user_msg[:500],  # Truncate long prompts
                            "label": "agentic",
                            "source": "toolbench"
                        })
                if len(prompts) >= limit:
                    break
        except Exception as e:
            print(f"  Error loading ToolBench: {e}")
    
    print(f"  ✓ Got {len(prompts)} AGENTIC prompts")
    return prompts[:limit]


def fetch_general_prompts(limit: int = 1000) -> List[Dict]:
    """
    Fetch GENERAL prompts using the Subtraction Method.
    Sources: UltraChat, DailyDialog, OpenAssistant
    """
    print(f"\n💬 Fetching GENERAL prompts (target: {limit})...")
    prompts = []
    seen = set()
    
    # Source 1: DailyDialog (pure chitchat)
    try:
        ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)
        for item in tqdm(ds, desc="DailyDialog"):
            for dialog in item.get("dialog", []):
                if dialog and dialog not in seen and is_general_prompt(dialog):
                    seen.add(dialog)
                    prompts.append({
                        "prompt": dialog,
                        "label": "general",
                        "source": "daily_dialog"
                    })
                if len(prompts) >= limit // 3:
                    break
            if len(prompts) >= limit // 3:
                break
    except Exception as e:
        print(f"  Error loading DailyDialog: {e}")
    
    # Source 2: UltraChat (creative/general)
    try:
        ds = load_dataset("stingning/ultrachat", split="train", trust_remote_code=True)
        for item in tqdm(ds, desc="UltraChat"):
            data = item.get("data", [])
            if data and len(data) > 0:
                user_msg = data[0] if isinstance(data[0], str) else ""
                if user_msg and user_msg not in seen and is_general_prompt(user_msg):
                    seen.add(user_msg)
                    prompts.append({
                        "prompt": user_msg,
                        "label": "general",
                        "source": "ultrachat"
                    })
            if len(prompts) >= 2 * limit // 3:
                break
    except Exception as e:
        print(f"  Error loading UltraChat: {e}")
    
    # Source 3: Alpaca (diverse instructions) - fallback
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)
        for item in tqdm(ds, desc="Alpaca"):
            text = item.get("instruction", "")
            if text and text not in seen and is_general_prompt(text):
                seen.add(text)
                prompts.append({
                    "prompt": text,
                    "label": "general",
                    "source": "alpaca"
                })
            if len(prompts) >= limit:
                break
    except Exception as e:
        print(f"  Error loading Alpaca: {e}")
    
    print(f"  ✓ Got {len(prompts)} GENERAL prompts")
    return prompts[:limit]


# =============================================================================
# Dataset Building
# =============================================================================

def build_dataset(
    total_samples: int = 2000,
    distribution: Dict[str, float] = None,
    output_path: str = "data/intent_training_dataset.json"
) -> Dict:
    """
    Build the complete training dataset with specified distribution.
    
    Default distribution (reflects production reality):
    - REASONING: 15%
    - CODING: 15%
    - FACTUAL_QA: 15%
    - AGENTIC: 15%
    - GENERAL: 40%
    """
    if distribution is None:
        distribution = {
            "reasoning": 0.15,
            "coding": 0.15,
            "factual_qa": 0.15,
            "agentic": 0.15,
            "general": 0.40,
        }
    
    # Calculate target counts
    targets = {k: int(total_samples * v) for k, v in distribution.items()}
    
    print("=" * 60)
    print("Building Intent Classification Dataset")
    print("=" * 60)
    print(f"\nTarget distribution:")
    for label, count in targets.items():
        print(f"  {label}: {count} ({distribution[label]*100:.0f}%)")
    
    # Fetch data for each category
    all_prompts = []
    
    reasoning = fetch_reasoning_prompts(targets["reasoning"])
    all_prompts.extend(reasoning)
    
    coding = fetch_coding_prompts(targets["coding"])
    all_prompts.extend(coding)
    
    factual = fetch_factual_qa_prompts(targets["factual_qa"])
    all_prompts.extend(factual)
    
    agentic = fetch_agentic_prompts(targets["agentic"])
    all_prompts.extend(agentic)
    
    general = fetch_general_prompts(targets["general"])
    all_prompts.extend(general)
    
    # Shuffle
    random.shuffle(all_prompts)
    
    # Final distribution
    final_dist = Counter(p["label"] for p in all_prompts)
    
    print("\n" + "=" * 60)
    print("Final Dataset")
    print("=" * 60)
    print(f"Total samples: {len(all_prompts)}")
    print("\nActual distribution:")
    for label, count in sorted(final_dist.items()):
        pct = count / len(all_prompts) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "metadata": {
            "total_samples": len(all_prompts),
            "distribution": dict(final_dist),
            "sources": list(set(p["source"] for p in all_prompts)),
        },
        "samples": all_prompts
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n💾 Dataset saved to: {output_path}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build intent classification training dataset"
    )
    parser.add_argument(
        '--total',
        type=int,
        default=2000,
        help='Total number of samples (default: 2000)'
    )
    parser.add_argument(
        '--output',
        default='data/intent_training_dataset.json',
        help='Output path'
    )
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Use balanced 20%% distribution instead of realistic 15/15/15/15/40'
    )
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("Error: datasets library required")
        print("Install with: pip install datasets tqdm")
        return
    
    if args.balanced:
        distribution = {
            "reasoning": 0.20,
            "coding": 0.20,
            "factual_qa": 0.20,
            "agentic": 0.20,
            "general": 0.20,
        }
    else:
        distribution = None  # Use default realistic distribution
    
    build_dataset(
        total_samples=args.total,
        distribution=distribution,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

