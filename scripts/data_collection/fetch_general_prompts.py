#!/usr/bin/env python3
"""
Fetch "General" category prompts using the Subtraction Strategy on LMSYS-Chat-1M.

The "General" class acts as a "None-of-the-Above" bucket capturing:
- Greetings, chitchat
- Simple requests  
- Roleplay
- Opinion-seeking
- Creative writing

Strategy: Define "General" by what it is NOT.
Filter out: CODING, REASONING, AGENTIC, FACTUAL_QA
Keep: Everything else = GENERAL
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# FILTER PATTERNS (The Subtraction Method)
# =============================================================================

# CODING indicators - remove these
CODING_PATTERNS = [
    r'```',                          # Markdown code blocks
    r'\bdef\s+\w+\s*\(',            # Python function definitions
    r'\bimport\s+\w+',              # Python imports
    r'\bclass\s+\w+',               # Class definitions
    r'\bfunction\s+\w+',            # JavaScript functions
    r'\bconst\s+\w+\s*=',           # JavaScript const
    r'\blet\s+\w+\s*=',             # JavaScript let
    r'\bvar\s+\w+\s*=',             # JavaScript var
    r'public\s+(static\s+)?void',   # Java methods
    r'#include\s*<',                # C/C++ includes
    r'\bprint\s*\(',                # Print statements
    r'\breturn\s+',                 # Return statements
    r'for\s*\(.+;.+;',              # C-style for loops
    r'while\s*\(.+\)',              # While loops
    r'\bif\s*\(.+\)\s*{',           # If statements with braces
    r'\.py\b',                      # .py file references
    r'\.js\b',                      # .js file references
    r'SQL|SELECT|INSERT|UPDATE',    # SQL keywords
    r'API|endpoint|REST|JSON',      # API-related
]

CODING_KEYWORDS = [
    'write a function', 'write a script', 'write code', 'code this',
    'implement a', 'create a function', 'debug this', 'fix this code',
    'programming', 'algorithm', 'syntax error', 'compile',
    'python', 'javascript', 'java ', 'c++', 'typescript', 'rust',
    'html', 'css', 'react', 'node.js', 'django', 'flask',
]

# REASONING/MATH indicators - remove these
MATH_PATTERNS = [
    r'\$.*\$',                      # LaTeX inline
    r'\\\[.*\\\]',                  # LaTeX display
    r'\\frac\{',                    # LaTeX fractions
    r'\\sqrt\{',                    # LaTeX square root
    r'\d+\s*[\+\-\*\/]\s*\d+',     # Arithmetic expressions
    r'=\s*\d+',                     # Equations
    r'solve for [xyz]',             # Algebra
    r'\d+%',                        # Percentages in calculations
    r'calculate|compute|evaluate',  # Math verbs
]

MATH_KEYWORDS = [
    'solve', 'equation', 'calculate', 'derivative', 'integral',
    'probability', 'statistics', 'prove that', 'proof',
    'theorem', 'formula', 'mathematics', 'algebra', 'calculus',
    'geometry', 'trigonometry', 'logarithm', 'exponent',
]

# AGENTIC/TOOL indicators - remove these  
AGENTIC_PATTERNS = [
    r'weather\s+(in|for|at)',       # Weather queries
    r'price\s+of',                  # Price lookups
    r'stock\s+price',               # Stock queries
    r'schedule\s+(a|my)',           # Scheduling
    r'book\s+(a|my)',               # Booking
    r'send\s+(an?\s+)?email',       # Email sending
    r'search\s+(for|the)',          # Search requests
    r'find\s+(me|the)',             # Find requests
    r'look\s+up',                   # Lookup requests
    r'check\s+(if|the|my)',         # Check requests
    r'download|upload',             # File operations
    r'open\s+(the|a|my)',           # Open app/file
    r'set\s+(a|an)\s+(alarm|timer|reminder)',  # Reminders
]

AGENTIC_KEYWORDS = [
    'weather', 'forecast', 'temperature',
    'stock price', 'market', 'trading',
    'book a', 'reserve', 'schedule',
    'send email', 'send message',
    'search for', 'look up', 'find me',
    'download', 'upload', 'install',
    'navigate to', 'directions to',
    'call ', 'dial ', 'text ',
    'set alarm', 'set timer', 'remind me',
    'play music', 'play song',
]

# FACTUAL_QA indicators - remove these
FACTUAL_PATTERNS = [
    r'^(when|where|who|what|which)\s+(was|is|are|were|did)\b',  # Question starters
    r'^define\s+',                  # Define requests
    r'^explain\s+(what|how|why)',   # Explain requests
    r'^what\s+is\s+(the|a|an)\s+\w+\?',  # "What is X?" questions
    r'^how\s+does\s+\w+\s+work',    # How does X work
    r'^who\s+(invented|discovered|created|founded)',  # Who questions
    r'^when\s+was\s+\w+\s+(born|founded|built)',  # When questions
    r'capital\s+of',                # Capital city questions
    r'population\s+of',             # Population questions
]

FACTUAL_KEYWORDS = [
    'what is the capital',
    'when was', 'where is', 'who invented', 'who discovered',
    'define ', 'meaning of', 'definition of',
    'history of', 'origin of',
    'how many', 'how much does',
    'explain the concept', 'explain how',
    'scientific', 'fact about',
]


def matches_patterns(text: str, patterns: List[str]) -> bool:
    """Check if text matches any regex patterns."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any keywords."""
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True
    return False


def is_coding(text: str) -> bool:
    """Check if prompt is CODING related."""
    return matches_patterns(text, CODING_PATTERNS) or contains_keywords(text, CODING_KEYWORDS)


def is_reasoning(text: str) -> bool:
    """Check if prompt is REASONING/MATH related."""
    return matches_patterns(text, MATH_PATTERNS) or contains_keywords(text, MATH_KEYWORDS)


def is_agentic(text: str) -> bool:
    """Check if prompt is AGENTIC related."""
    return matches_patterns(text, AGENTIC_PATTERNS) or contains_keywords(text, AGENTIC_KEYWORDS)


def is_factual_qa(text: str) -> bool:
    """Check if prompt is FACTUAL_QA related."""
    return matches_patterns(text, FACTUAL_PATTERNS) or contains_keywords(text, FACTUAL_KEYWORDS)


def is_general(text: str) -> bool:
    """
    A prompt is GENERAL if it's NOT any of the specialized categories.
    This is the "Subtraction Strategy".
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # Filter out specialized categories
    if is_coding(text):
        return False
    if is_reasoning(text):
        return False
    if is_agentic(text):
        return False
    if is_factual_qa(text):
        return False
    
    return True


def extract_user_prompts_from_lmsys(dataset) -> List[str]:
    """Extract first user message from each conversation."""
    prompts = []
    
    for item in tqdm(dataset, desc="Extracting prompts"):
        conversation = item.get('conversation', [])
        
        # Get first user message
        for msg in conversation:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if content and isinstance(content, str):
                    # Basic cleaning
                    content = content.strip()
                    # Filter by length (not too short, not too long)
                    if 15 < len(content) < 1000:
                        prompts.append(content)
                break
    
    return prompts


def filter_for_general(prompts: List[str]) -> List[str]:
    """Apply subtraction strategy to keep only GENERAL prompts."""
    general = []
    stats = {'coding': 0, 'reasoning': 0, 'agentic': 0, 'factual_qa': 0, 'general': 0}
    
    for prompt in tqdm(prompts, desc="Filtering for GENERAL"):
        if is_coding(prompt):
            stats['coding'] += 1
        elif is_reasoning(prompt):
            stats['reasoning'] += 1
        elif is_agentic(prompt):
            stats['agentic'] += 1
        elif is_factual_qa(prompt):
            stats['factual_qa'] += 1
        else:
            stats['general'] += 1
            general.append(prompt)
    
    print("\nFiltering Statistics:")
    total = sum(stats.values())
    for cat, count in stats.items():
        pct = count / total * 100 if total > 0 else 0
        marker = "✓" if cat == 'general' else "✗"
        print(f"  {marker} {cat:<15}: {count:>6} ({pct:.1f}%)")
    
    return general


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch GENERAL prompts using Subtraction Strategy")
    parser.add_argument('--limit', type=int, default=50000, help='Max prompts to process from LMSYS')
    parser.add_argument('--output-limit', type=int, default=200, help='Max GENERAL prompts to save')
    parser.add_argument('--output', default='data/general_prompts.json', help='Output file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FETCH GENERAL PROMPTS (Subtraction Strategy)")
    print("=" * 60)
    print("\nStrategy: Keep prompts that are NOT:")
    print("  - CODING (code blocks, programming keywords)")
    print("  - REASONING (math, equations, proofs)")
    print("  - AGENTIC (tool use, scheduling, lookups)")
    print("  - FACTUAL_QA (fact questions, definitions)")
    print("\nWhat remains = GENERAL (chitchat, creative, opinions, roleplay)")
    
    # Load LMSYS dataset
    print(f"\n{'='*60}")
    print("Loading LMSYS-Chat-1M...")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        # Load a subset to avoid memory issues
        ds = load_dataset('lmsys/lmsys-chat-1m', split=f'train[:{args.limit}]')
        print(f"Loaded {len(ds)} conversations")
        
    except Exception as e:
        print(f"Error loading LMSYS dataset: {e}")
        print("\nTrying alternative: OpenAssistant/oasst2...")
        
        try:
            ds = load_dataset('OpenAssistant/oasst2', split='train')
            # Filter for initial prompts only
            prompts = []
            for item in tqdm(ds, desc="Extracting"):
                if item.get('role') == 'prompter' and item.get('parent_id') is None:
                    text = item.get('text', '')
                    if text and 15 < len(text) < 1000:
                        prompts.append(text)
                if len(prompts) >= args.limit:
                    break
            print(f"Extracted {len(prompts)} prompts from OpenAssistant")
            
        except Exception as e2:
            print(f"Error: {e2}")
            return
    else:
        # Extract prompts from LMSYS
        prompts = extract_user_prompts_from_lmsys(ds)
        print(f"Extracted {len(prompts)} user prompts")
    
    # Apply subtraction strategy
    print(f"\n{'='*60}")
    print("Applying Subtraction Strategy...")
    print("=" * 60)
    
    general_prompts = filter_for_general(prompts)
    print(f"\nFound {len(general_prompts)} GENERAL prompts")
    
    # Sample if too many
    if len(general_prompts) > args.output_limit:
        random.seed(42)
        general_prompts = random.sample(general_prompts, args.output_limit)
        print(f"Sampled {args.output_limit} prompts")
    
    # Format for labeling pipeline
    output_data = []
    for prompt in general_prompts:
        output_data.append({
            'text': prompt,
            'source': 'lmsys_chat_filtered',
            'expected_label': 'general',
        })
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SAMPLE GENERAL PROMPTS")
    print("=" * 60)
    
    for i, p in enumerate(general_prompts[:15], 1):
        display = p[:100] + '...' if len(p) > 100 else p
        print(f"\n{i}. {display}")
    
    print(f"\n{'='*60}")
    print(f"✅ Saved {len(output_data)} GENERAL prompts to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

