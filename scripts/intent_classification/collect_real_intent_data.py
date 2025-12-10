"""
Collect Real Intent Classification Data from HuggingFace Datasets.

Following the methodology from RouteLLM and KDD best practices:
1. Sample prompts from established benchmarks
2. Use teacher labeling (GPT-4/Claude) as oracle
3. Create high-quality labeled dataset for training

Datasets:
- REASONING: GSM8k (math), MATH (competition math)
- CODING: MBPP (python problems)
- FACTUAL_QA: Natural Questions (Google search)
- AGENTIC_EXECUTION: Glaive Function Calling v2
- GENERAL: LMSYS-Chat-1M (filtered)
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm


def load_gsm8k_reasoning(n_samples: int = 2000) -> List[Dict]:
    """
    Load GSM8k dataset for REASONING class.
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n📚 Loading GSM8k (Grade School Math)...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        
        samples = []
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            question = item['question']
            samples.append({
                'prompt': question,
                'source': 'gsm8k',
                'category_hint': 'reasoning',
                'metadata': {
                    'answer': item.get('answer', ''),
                    'idx': i,
                }
            })
        
        print(f"  ✓ Collected {len(samples)} reasoning prompts from GSM8k")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading GSM8k: {e}")
        return []


def load_mbpp_coding(n_samples: int = 2000) -> List[Dict]:
    """
    Load MBPP dataset for CODING class.
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n💻 Loading MBPP (Mostly Basic Python Problems)...")
        
        # Load without feature types that might cause issues
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", trust_remote_code=True)
        
        samples = []
        # Only get up to n_samples (MBPP has ~120 total)
        max_samples = min(n_samples, len(dataset))
        
        for i in range(max_samples):
            item = dataset[i]
            
            # MBPP has natural language task descriptions in 'prompt' field
            prompt = item['prompt']
            samples.append({
                'prompt': prompt,
                'source': 'mbpp',
                'category_hint': 'coding',
                'metadata': {
                    'test_list': str(item.get('test_list', [])),
                    'code': item.get('code', ''),
                    'task_id': item.get('task_id', i),
                }
            })
        
        print(f"  ✓ Collected {len(samples)} coding prompts from MBPP")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading MBPP: {e}")
        print(f"  Trying alternative HumanEval dataset...")
        return load_humaneval_fallback(n_samples)


def load_natural_questions_qa(n_samples: int = 2000) -> List[Dict]:
    """
    Load Natural Questions for FACTUAL_QA class.
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n❓ Loading Natural Questions (Google Search)...")
        dataset = load_dataset("google-research-datasets/natural_questions", 
                              split="train", streaming=True)
        
        samples = []
        question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            question = item['question']['text']
            question_lower = question.lower().strip()
            
            # Filter for questions starting with question words
            if any(question_lower.startswith(qw) for qw in question_words):
                samples.append({
                    'prompt': question,
                    'source': 'natural_questions',
                    'category_hint': 'factual_qa',
                    'metadata': {
                        'idx': i,
                    }
                })
        
        print(f"  ✓ Collected {len(samples)} QA prompts from Natural Questions")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading Natural Questions: {e}")
        print("  Note: This dataset is very large. Using alternative...")
        return load_trivia_qa_fallback(n_samples)


def load_humaneval_fallback(n_samples: int = 200) -> List[Dict]:
    """Fallback to HumanEval if MBPP fails."""
    try:
        from datasets import load_dataset
        print("  📖 Trying HumanEval as fallback...")
        dataset = load_dataset("openai/openai_humaneval", split="test")
        
        samples = []
        max_samples = min(n_samples, len(dataset))
        
        for i in range(max_samples):
            item = dataset[i]
            prompt = item.get('prompt', '').strip()
            if prompt:
                samples.append({
                    'prompt': f"Complete this Python function:\n{prompt}",
                    'source': 'humaneval',
                    'category_hint': 'coding',
                    'metadata': {
                        'task_id': item.get('task_id', ''),
                        'entry_point': item.get('entry_point', ''),
                    }
                })
        
        print(f"  ✓ Collected {len(samples)} coding prompts from HumanEval")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading HumanEval: {e}")
        return []


def load_trivia_qa_fallback(n_samples: int = 2000) -> List[Dict]:
    """Fallback to TriviaQA if Natural Questions fails."""
    try:
        from datasets import load_dataset
        print("\n  📖 Trying TriviaQA as fallback...")
        dataset = load_dataset("trivia_qa", "unfiltered", split="train")
        
        samples = []
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            question = item['question']
            samples.append({
                'prompt': question,
                'source': 'trivia_qa',
                'category_hint': 'factual_qa',
                'metadata': {
                    'answer': item.get('answer', {}),
                    'idx': i,
                }
            })
        
        print(f"  ✓ Collected {len(samples)} QA prompts from TriviaQA")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading TriviaQA: {e}")
        return []


def load_glaive_agentic(n_samples: int = 2000) -> List[Dict]:
    """
    Load Glaive Function Calling for AGENTIC_EXECUTION class.
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n🤖 Loading Glaive Function Calling v2...")
        dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        
        samples = []
        
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            # Extract user prompts from conversation
            chat = item.get('chat', '[]')
            try:
                messages = json.loads(chat) if isinstance(chat, str) else chat
                
                if not isinstance(messages, list):
                    continue
                
                # Find user messages that mention tools/functions or multi-step tasks
                for j, msg in enumerate(messages):
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        content = msg.get('content', '')
                        if content and len(content) > 20:  # Skip too short
                            samples.append({
                                'prompt': content,
                                'source': 'glaive_function_calling',
                                'category_hint': 'agentic_execution',
                                'metadata': {
                                    'conversation_idx': i,
                                    'message_idx': j,
                                }
                            })
                            break  # One prompt per conversation
            except Exception as parse_error:
                continue
        
        print(f"  ✓ Collected {len(samples)} agentic prompts from Glaive")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading Glaive: {e}")
        print(f"  ⚠️  SYNTHETIC FALLBACK REMOVED - Returning empty list")
        print(f"  Please ensure Glaive dataset is available or use real data")
        return []  # Return empty list instead of synthetic data


# REMOVED: generate_synthetic_agentic() function (December 10, 2025)
# Synthetic fallback removed - all data must be real
# See KDD/data/DATA_AUTHENTICITY_VERIFICATION.md for details


def load_lmsys_general(n_samples: int = 2000) -> List[Dict]:
    """
    Load LMSYS Chat 1M for GENERAL class with negative filtering.
    
    Filters OUT:
    - Code blocks (```)
    - Math symbols ($, LaTeX)
    - Long messages (>50 words)
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n💬 Loading LMSYS-Chat-1M (with filtering)...")
        print("  Note: This is a gated dataset. If it fails, using ShareGPT as alternative...")
        
        try:
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        except:
            print("  LMSYS gated, trying ShareGPT...")
            dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        
        samples = []
        code_pattern = re.compile(r'```')
        math_pattern = re.compile(r'[$\\]|\\[a-zA-Z]+\{')  # LaTeX
        
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            # Get first user message
            conversation = item.get('conversation', [])
            if not conversation:
                continue
            
            first_msg = conversation[0]
            if first_msg.get('role') != 'user':
                continue
            
            content = first_msg.get('content', '')
            
            # Apply negative filters
            if not content or len(content) < 5:
                continue
            
            # Remove code blocks
            if code_pattern.search(content):
                continue
            
            # Remove math
            if math_pattern.search(content):
                continue
            
            # Remove long messages
            word_count = len(content.split())
            if word_count > 50:
                continue
            
            # Remove coding-related keywords
            coding_keywords = ['python', 'javascript', 'function', 'code', 'implement', 'class']
            if any(kw in content.lower() for kw in coding_keywords):
                continue
            
            samples.append({
                'prompt': content,
                'source': 'lmsys_chat',
                'category_hint': 'general',
                'metadata': {
                    'conversation_id': item.get('conversation_id', ''),
                    'model': item.get('model', ''),
                    'idx': i,
                }
            })
        
        print(f"  ✓ Collected {len(samples)} general prompts from LMSYS (filtered)")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading LMSYS: {e}")
        return []


def save_raw_prompts(samples: List[Dict], output_path: str):
    """Save raw collected prompts before teacher labeling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_samples': len(samples),
                'sources': list(set(s['source'] for s in samples)),
                'note': 'Raw prompts before teacher labeling',
            },
            'samples': samples,
        }, f, indent=2)
    
    print(f"\n💾 Saved {len(samples)} raw prompts to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect real intent classification data from HuggingFace"
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=2000,
        help='Number of samples to collect per class'
    )
    parser.add_argument(
        '--output',
        default='data/real_intent_prompts_raw.json',
        help='Output path for raw prompts'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*60)
    print("Real Intent Data Collection")
    print("="*60)
    print(f"\nTarget: {args.samples_per_class} samples per class")
    print(f"Total: {args.samples_per_class * 5} prompts")
    
    # Check dependencies
    try:
        import datasets
        print("\n✓ HuggingFace datasets available")
    except ImportError:
        print("\n✗ Error: datasets library not installed")
        print("Install with: pip install datasets")
        return
    
    # Collect from each source
    all_samples = []
    
    # 1. REASONING - GSM8k
    reasoning_samples = load_gsm8k_reasoning(args.samples_per_class)
    all_samples.extend(reasoning_samples)
    
    # 2. CODING - MBPP
    coding_samples = load_mbpp_coding(args.samples_per_class)
    all_samples.extend(coding_samples)
    
    # 3. FACTUAL_QA - Natural Questions
    qa_samples = load_natural_questions_qa(args.samples_per_class)
    all_samples.extend(qa_samples)
    
    # 4. AGENTIC_EXECUTION - Glaive Function Calling
    agentic_samples = load_glaive_agentic(args.samples_per_class)
    all_samples.extend(agentic_samples)
    
    # 5. GENERAL - LMSYS Chat (filtered)
    general_samples = load_lmsys_general(args.samples_per_class)
    all_samples.extend(general_samples)
    
    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    
    by_category = defaultdict(int)
    by_source = defaultdict(int)
    
    for sample in all_samples:
        by_category[sample['category_hint']] += 1
        by_source[sample['source']] += 1
    
    print("\nBy Category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:<20} {count:>6} samples")
    
    print("\nBy Source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src:<30} {count:>6} samples")
    
    print(f"\nTotal collected: {len(all_samples)} prompts")
    
    # Save
    save_raw_prompts(all_samples, args.output)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Run teacher labeling with GPT-4/Claude:")
    print("   python scripts/teacher_label_intents.py")
    print("\n2. Create train/val/test splits:")
    print("   python scripts/split_labeled_data.py")
    print("\n3. Train XGBoost on real data:")
    print("   python scripts/train_xgboost_intent.py --dataset data/real_intent_labeled.json")
    
    print("\n✅ Data collection complete!")


if __name__ == '__main__':
    main()

