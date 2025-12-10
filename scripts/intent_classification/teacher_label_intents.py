"""
Teacher Labeling for Intent Classification.

Uses a strong model (GPT-4, Claude, or Gemini 3 Pro) as an oracle to label collected prompts.
This is the gold standard methodology cited in RouteLLM and similar papers.

The teacher model classifies prompts into one of 5 categories and we use
these high-quality labels to train our smaller, faster router.

Supported Providers:
- OpenAI (gpt-4o)
- Anthropic (claude-3-5-sonnet)
- OpenRouter (google/gemini-3-pro-preview, and others)
"""

import json
import os
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment only")
    print("   Install with: pip install python-dotenv")


# Teacher labeling prompt template with few-shot examples
def make_labeling_prompt(prompt_text: str) -> str:
    """Create labeling prompt with few-shot examples for better accuracy."""
    return f"""You are an expert Data Labeler for an Intelligent Routing System.

Your goal is to classify the User's Intent into exactly ONE of these 5 categories:

**Categories:**

1. **REASONING** - Mathematical problems, logical reasoning, analytical tasks, proofs
   
2. **CODING** - Programming tasks, code generation, debugging (user wants SOURCE CODE they will run later)

3. **FACTUAL_QA** - Knowledge questions, factual information retrieval, educational explanations

4. **AGENTIC_EXECUTION** - User wants YOU to DO a task NOW using tools/APIs (side-effects, external actions)

5. **GENERAL** - General conversation, greetings, opinions, chitchat, creative writing

---

**THE TRICKY DISTINCTION (CODING vs AGENTIC):**

- Label as **CODING** if the user wants you to WRITE code that they will run later.
  (e.g., "Write a python script to ping a server.")
  
- Label as **AGENTIC_EXECUTION** if the user wants YOU to DO the task right now using a tool/API.
  (e.g., "Ping this server for me.")

---

**FEW-SHOT EXAMPLES:**

Input: "Write a python function to calculate the fibonacci sequence."
Label: CODING
Reasoning: User is asking for source code generation.

Input: "Check if the server at 192.168.1.1 is active."
Label: AGENTIC_EXECUTION
Reasoning: User implies immediate execution using a 'ping' tool.

Input: "I need to sort this list of names: [Alice, Bob, Charlie]."
Label: CODING
Reasoning: This is a data manipulation task best solved by generating a snippet or direct answer.

Input: "Book a flight from NY to London for next Tuesday."
Label: AGENTIC_EXECUTION
Reasoning: Requires side-effect (booking) via an external flight API.

Input: "Find all files larger than 10MB in /var/log and delete them."
Label: AGENTIC_EXECUTION
Reasoning: Requires interacting with the OS/Terminal environment.

Input: "Explain how a binary search tree works."
Label: FACTUAL_QA
Reasoning: Educational concept explanation. No tools or code needed.

Input: "Solve for x: 2x + 5 = 15"
Label: REASONING
Reasoning: Mathematical equation solving requires logical reasoning.

Input: "What's the weather like in Tokyo?"
Label: AGENTIC_EXECUTION
Reasoning: Requires calling a weather API to get current conditions.

Input: "Hello! How are you today?"
Label: GENERAL
Reasoning: Casual greeting/chitchat.

Input: "Write me a poem about the ocean."
Label: GENERAL
Reasoning: Creative writing task, not code or factual.

Input: "Translate 'Hello, how are you?' to Spanish."
Label: GENERAL
Reasoning: Simple translation task - linguistic/language help, not factual knowledge retrieval.

Input: "Help me write a polite email to my landlord about the leak."
Label: GENERAL
Reasoning: Creative drafting/writing assistance, not code or facts.

---

**Instructions:**
- Choose the PRIMARY intent category
- If ambiguous, choose the most prominent intent
- Respond in this exact format:

**Category:** [category name]
**Confidence:** [0.0-1.0]
**Reasoning:** [brief explanation]

Valid category names: REASONING, CODING, FACTUAL_QA, AGENTIC_EXECUTION, GENERAL

---

**NOW CLASSIFY THIS PROMPT:**
{prompt_text}

**Your classification:**"""


def call_openai_teacher(prompt: str, api_key: str, model: str = "gpt-4o") -> Dict:
    """Use OpenAI API as teacher."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": make_labeling_prompt(prompt)}
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},  # Force JSON output
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON with better error handling
        try:
            # Remove any markdown formatting
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Remove any leading/trailing whitespace and newlines
            content = content.strip()
            
            # Try to extract JSON if surrounded by other text
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            result = json.loads(content)
            
            # Normalize category name
            if 'category' in result:
                category = result['category'].upper().strip()
                # Map variations
                category_map = {
                    'REASONING': 'reasoning',
                    'CODING': 'coding',
                    'FACTUAL_QA': 'factual_qa',
                    'AGENTIC_EXECUTION': 'agentic_execution',
                    'AGENTIC': 'agentic_execution',
                    'GENERAL': 'general',
                }
                result['category'] = category_map.get(category, category.lower())
                return result
            else:
                print(f"  ⚠️  Missing 'category' in response: {content[:100]}")
                return None
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error: {e}")
            print(f"  Content received: {repr(content[:300])}")
            return None
        
    except Exception as e:
        print(f"  ❌ Error calling OpenAI: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def call_anthropic_teacher(prompt: str, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> Dict:
    """Use Anthropic API as teacher."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=300,
            temperature=0.0,
            messages=[
                {"role": "user", "content": make_labeling_prompt(prompt)}
            ]
        )
        
        content = response.content[0].text.strip()
        
        # Parse JSON with better error handling
        try:
            # Remove markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Remove any leading/trailing whitespace
            content = content.strip()
            
            # Try to extract JSON if surrounded by other text
            json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            
            result = json.loads(content)
            
            # Normalize category name
            if 'category' in result:
                category = result['category'].upper().strip()
                # Map variations
                category_map = {
                    'REASONING': 'reasoning',
                    'CODING': 'coding',
                    'FACTUAL_QA': 'factual_qa',
                    'AGENTIC_EXECUTION': 'agentic_execution',
                    'AGENTIC': 'agentic_execution',
                    'GENERAL': 'general',
                }
                result['category'] = category_map.get(category, category.lower())
                return result
            else:
                print(f"  ⚠️  Missing 'category' in response: {content[:100]}")
                return None
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error: {e}")
            print(f"  Content received: {repr(content[:300])}")
            return None
        
    except Exception as e:
        print(f"  ❌ Error calling Anthropic: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def call_openrouter_teacher(prompt: str, api_key: str, model: str = "google/gemini-3-pro-preview") -> Dict:
    """Use OpenRouter API as teacher (supports Gemini models with markdown output)."""
    import openai
    
    # OpenRouter uses OpenAI-compatible API
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Thinking models (like Gemini 3 Pro) need more tokens for reasoning + output
    is_thinking_model = 'gemini-3' in model.lower() or 'thinking' in model.lower()
    max_tokens = 2000 if is_thinking_model else 500
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": make_labeling_prompt(prompt)}
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {type(e).__name__}: {e}")
    
    # Check for response content
    if not response.choices or len(response.choices) == 0:
        raise Exception("No response choices returned from API")
    
    content = response.choices[0].message.content
    
    # Handle empty responses
    if not content or content.strip() == "":
        raise Exception("Empty response from API - no content returned. Try increasing max_tokens.")
    
    content = content.strip()
    
    # Parse markdown format
    category = None
    confidence = None
    reasoning = None
    
    # Extract fields from markdown
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('**Category:**'):
            category = line.split('**Category:**')[1].strip()
        elif line.startswith('**Confidence:**'):
            conf_str = line.split('**Confidence:**')[1].strip()
            try:
                confidence = float(conf_str)
            except:
                confidence = 0.9
        elif line.startswith('**Reasoning:**'):
            reasoning = line.split('**Reasoning:**')[1].strip()
    
    if not category:
        # Try fallback JSON parsing
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if 'category' in result:
                    category = result['category']
                    confidence = result.get('confidence', 0.9)
                    reasoning = result.get('reasoning', '')
            except json.JSONDecodeError:
                pass
    
    if not category:
        raise Exception(f"Could not parse category from response. Content: {content[:200]}")
    
    # Normalize category name
    category = category.upper().strip()
    category_map = {
        'REASONING': 'reasoning',
        'CODING': 'coding',
        'FACTUAL_QA': 'factual_qa',
        'AGENTIC_EXECUTION': 'agentic_execution',
        'AGENTIC': 'agentic_execution',
        'GENERAL': 'general',
    }
    
    normalized_category = category_map.get(category, category.lower())
    
    # Validate category
    valid_categories = ['reasoning', 'coding', 'factual_qa', 'agentic_execution', 'general']
    if normalized_category not in valid_categories:
        raise Exception(f"Invalid category '{normalized_category}'. Must be one of: {valid_categories}")
    
    return {
        'category': normalized_category,
        'confidence': confidence or 0.9,
        'reasoning': reasoning or ''
    }


def label_prompts(
    prompts: List[Dict],
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 100,
    save_interval: int = 50,
    output_path: str = "data/real_intent_labeled.json",
) -> List[Dict]:
    """
    Label prompts using teacher model.
    
    Args:
        prompts: List of prompt dicts with 'prompt' field
        provider: 'openai' or 'anthropic'
        api_key: API key for provider
        model: Model name to use
        batch_size: Process this many at a time
        save_interval: Save progress every N prompts
        output_path: Where to save labeled data
        
    Returns:
        List of labeled prompts
    """
    # Get API key
    if api_key is None:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError(f"API key required for {provider}")
    
    # Choose teacher function
    if provider == "openai":
        teacher_fn = call_openai_teacher
        default_model = model or "gpt-4o"
    elif provider == "anthropic":
        teacher_fn = call_anthropic_teacher
        default_model = model or "claude-3-5-sonnet-20241022"
    elif provider == "openrouter":
        teacher_fn = call_openrouter_teacher
        default_model = model or "google/gemini-3-pro-preview"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    print(f"\n🎓 Using {default_model} as teacher")
    print(f"📊 Labeling {len(prompts)} prompts...")
    
    labeled = []
    errors = []
    
    for i, sample in enumerate(tqdm(prompts, desc="Labeling")):
        prompt_text = sample['prompt']
        
        try:
            # Call teacher
            result = teacher_fn(prompt_text, api_key, default_model)
            
            if result and 'category' in result:
                category = result['category'].lower()
                
                # Normalize category names
                if category == 'agentic':
                    category = 'agentic_execution'
                
                labeled.append({
                    **sample,  # Keep original fields
                    'label': category,
                    'teacher_confidence': result.get('confidence', 1.0),
                    'teacher_reasoning': result.get('reasoning', ''),
                    'teacher_model': default_model,
                })
            else:
                errors.append({
                    'prompt': prompt_text[:200],
                    'error': 'Invalid response format',
                    'response': str(result)[:200] if result else None,
                })
        
        except Exception as e:
            error_msg = str(e)
            errors.append({
                'prompt': prompt_text[:200],
                'error': error_msg[:500],
                'error_type': type(e).__name__,
            })
            # Print first few errors for debugging
            if len(errors) <= 3:
                print(f"\n  ⚠️  Error on sample {i+1}: {error_msg[:100]}")
        
        # Rate limiting
        time.sleep(0.1)  # Avoid rate limits
        
        # Save progress
        if (i + 1) % save_interval == 0:
            save_labeled_data(labeled, output_path)
            success_rate = len(labeled) / (i + 1) * 100
            print(f"\n  💾 Saved progress: {len(labeled)}/{i+1} labeled ({success_rate:.1f}% success)")
    
    # Final save
    save_labeled_data(labeled, output_path)
    
    if errors:
        error_path = Path(output_path).with_suffix('.errors.json')
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n  ⚠️  {len(errors)} errors saved to: {error_path}")
    
    return labeled


def save_labeled_data(labeled: List[Dict], output_path: str):
    """Save labeled data to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count by category
    from collections import Counter
    category_counts = Counter(s['label'] for s in labeled)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_samples': len(labeled),
                'by_category': dict(category_counts),
                'note': 'Teacher-labeled using strong model oracle',
            },
            'samples': labeled,
        }, f, indent=2)


def load_toolbench_prompts(limit: int = None) -> List[Dict]:
    """Load ToolBench agentic prompts."""
    toolbench_path = Path("data/toolbench_agentic_prompts.json")
    if not toolbench_path.exists():
        print(f"  ⚠️  ToolBench data not found. Run: python scripts/fetch_toolbench.py")
        return []
    
    with open(toolbench_path) as f:
        data = json.load(f)
    
    # Convert to labeling format
    prompts = []
    for item in data:
        prompts.append({
            'prompt': item['text'],
            'source': 'toolbench',
            'expected_label': 'agentic_execution',
        })
    
    if limit:
        prompts = prompts[:limit]
    
    return prompts


def load_coding_prompts(limit: int = None) -> List[Dict]:
    """Load coding prompts from various sources."""
    coding_sources = [
        "data/real_intent_prompts_raw.json",  # May have coding samples
    ]
    
    prompts = []
    for source_path in coding_sources:
        if Path(source_path).exists():
            with open(source_path) as f:
                data = json.load(f)
            
            samples = data.get('samples', data)
            for item in samples:
                # Filter for likely coding prompts based on source
                source = item.get('source', '')
                if 'code' in source.lower() or 'programming' in source.lower():
                    prompts.append({
                        'prompt': item.get('prompt', item.get('text', '')),
                        'source': source,
                        'expected_label': 'coding',
                    })
    
    if limit:
        prompts = prompts[:limit]
    
    return prompts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Teacher labeling for intent classification"
    )
    parser.add_argument(
        '--input',
        default='data/real_intent_prompts_raw.json',
        help='Input file with raw prompts'
    )
    parser.add_argument(
        '--output',
        default='data/real_intent_labeled.json',
        help='Output file for labeled prompts'
    )
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'openrouter'],
        default='openai',
        help='Teacher model provider'
    )
    parser.add_argument(
        '--model',
        help='Specific model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022, google/gemini-3-pro-preview)'
    )
    parser.add_argument(
        '--api-key',
        help='API key (or use env var)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of prompts to label (for testing)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='Save progress every N prompts'
    )
    # New options for focused labeling
    parser.add_argument(
        '--include-toolbench',
        action='store_true',
        help='Include ToolBench agentic prompts for labeling'
    )
    parser.add_argument(
        '--toolbench-only',
        action='store_true',
        help='Only label ToolBench agentic prompts'
    )
    parser.add_argument(
        '--toolbench-limit',
        type=int,
        default=1000,
        help='Limit number of ToolBench prompts (default: 1000)'
    )
    parser.add_argument(
        '--filter-source',
        help='Only label prompts from specific source (e.g., "sharegpt", "wildchat")'
    )
    parser.add_argument(
        '--skip-labeled',
        action='store_true',
        help='Skip prompts that already have labels in output file'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Teacher Labeling for Intent Classification")
    print("="*60)
    print("Using improved few-shot prompt for CODING vs AGENTIC distinction")
    
    prompts = []
    
    # Option 1: ToolBench only
    if args.toolbench_only:
        print(f"\n📂 Loading ToolBench agentic prompts...")
        prompts = load_toolbench_prompts(limit=args.toolbench_limit)
        if not prompts:
            print("❌ No ToolBench prompts found. Run: python scripts/fetch_toolbench.py")
            return
        print(f"  ✓ Loaded {len(prompts)} ToolBench prompts")
    
    # Option 2: Load from input file
    else:
        print(f"\n📂 Loading prompts from: {args.input}")
        if Path(args.input).exists():
            with open(args.input, 'r') as f:
                data = json.load(f)
            prompts = data.get('samples', data if isinstance(data, list) else [])
            print(f"  ✓ Loaded {len(prompts)} prompts from input file")
        else:
            print(f"  ⚠️  Input file not found: {args.input}")
        
        # Add ToolBench if requested
        if args.include_toolbench:
            print(f"\n📂 Adding ToolBench agentic prompts...")
            toolbench = load_toolbench_prompts(limit=args.toolbench_limit)
            if toolbench:
                prompts.extend(toolbench)
                print(f"  ✓ Added {len(toolbench)} ToolBench prompts")
    
    # Filter by source if requested
    if args.filter_source:
        original_count = len(prompts)
        prompts = [p for p in prompts if args.filter_source.lower() in p.get('source', '').lower()]
        print(f"  ✓ Filtered to {len(prompts)}/{original_count} prompts from source '{args.filter_source}'")
    
    # Skip already labeled if requested
    if args.skip_labeled and Path(args.output).exists():
        with open(args.output) as f:
            existing = json.load(f)
        existing_prompts = set(s.get('prompt', '')[:100] for s in existing.get('samples', []))
        original_count = len(prompts)
        prompts = [p for p in prompts if p.get('prompt', '')[:100] not in existing_prompts]
        print(f"  ✓ Skipping {original_count - len(prompts)} already labeled prompts")
    
    if args.limit:
        prompts = prompts[:args.limit]
        print(f"  ⚠️  Limited to {args.limit} prompts for testing")
    
    if not prompts:
        print("\n❌ No prompts to label!")
        return
    
    print(f"\n  📊 Total prompts to label: {len(prompts)}")
    
    # Check API key (try .env first, then environment, then argument)
    if args.provider == 'openai':
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("\n❌ Error: OPENAI_API_KEY not set")
            print("\n  Option 1: Add to .env file:")
            print("    echo 'OPENAI_API_KEY=your-key-here' >> .env")
            print("\n  Option 2: Set environment variable:")
            print("    export OPENAI_API_KEY=your-key-here")
            print("\n  Option 3: Pass as argument:")
            print("    --api-key your-key-here")
            return
        else:
            # Show first/last 4 chars of key for verification
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print(f"  ✓ Using OpenAI API key: {masked_key}")
            
    elif args.provider == 'anthropic':
        api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("\n❌ Error: ANTHROPIC_API_KEY not set")
            print("\n  Option 1: Add to .env file:")
            print("    echo 'ANTHROPIC_API_KEY=your-key-here' >> .env")
            print("\n  Option 2: Set environment variable:")
            print("    export ANTHROPIC_API_KEY=your-key-here")
            print("\n  Option 3: Pass as argument:")
            print("    --api-key your-key-here")
            return
        else:
            # Show first/last 4 chars of key for verification
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print(f"  ✓ Using Anthropic API key: {masked_key}")
    
    elif args.provider == 'openrouter':
        api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("\n❌ Error: OPENROUTER_API_KEY not set")
            print("\n  Option 1: Add to .env file:")
            print("    echo 'OPENROUTER_API_KEY=your-key-here' >> .env")
            print("\n  Option 2: Set environment variable:")
            print("    export OPENROUTER_API_KEY=your-key-here")
            print("\n  Option 3: Pass as argument:")
            print("    --api-key your-key-here")
            return
        else:
            # Show first/last 4 chars of key for verification
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print(f"  ✓ Using OpenRouter API key: {masked_key}")
    
    # Label
    labeled = label_prompts(
        prompts,
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        save_interval=args.save_interval,
        output_path=args.output,
    )
    
    # Summary
    print("\n" + "="*60)
    print("LABELING SUMMARY")
    print("="*60)
    
    from collections import Counter
    category_counts = Counter(s['label'] for s in labeled)
    
    print(f"\nTotal labeled: {len(labeled)}/{len(prompts)}")
    print(f"Success rate: {len(labeled)/len(prompts)*100:.1f}%")
    
    print("\nBy Category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:<20} {count:>6} samples")
    
    print(f"\n💾 Saved to: {args.output}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Split into train/val/test:")
    print("   python scripts/split_labeled_data.py")
    print("\n2. Train XGBoost:")
    print("   python scripts/train_xgboost_intent.py --dataset data/real_intent_labeled_split.json")
    
    print("\n✅ Teacher labeling complete!")


if __name__ == '__main__':
    main()

