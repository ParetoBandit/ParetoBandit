"""
Collect Real Intent Classification Data from HuggingFace Datasets.

Methodology:
1. Sample prompts from domain-specific established benchmarks
2. Dataset source defines the ground-truth intent label (no teacher labeling needed)
3. Create high-quality labeled dataset for training classifiers

Ground Truth Labels (6 Classes):
- CODING: MBPP, HumanEval → prompts are definitively coding tasks
- REASONING: GSM8k, MATH → prompts are definitively reasoning tasks
- FACTUAL_QA: Natural Questions, TriviaQA → prompts are definitively factual questions
- SUMMARIZATION: CNN/DailyMail, XSum → prompts are definitively summarization requests
- AGENTIC_EXECUTION: Glaive Function Calling v2 → prompts are definitively agentic tasks
- GENERAL: LMSYS-Chat-1M (filtered) → prompts are definitively general conversation

The dataset source is the source of truth for intent labels.
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
                'intent_label': 'reasoning',  # Ground truth from dataset source
                'source': 'gsm8k',
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
    Load coding datasets from multiple sources to reach target samples.
    
    Sources (in order):
    1. MBPP (120 samples)
    2. HumanEval (164 samples) 
    3. CodeAlpaca (20k samples)
    4. APPS (10k samples)
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    from datasets import load_dataset
    
    all_samples = []
    
    # 1. MBPP (~120 samples)
    try:
        print("\n💻 Loading MBPP (Mostly Basic Python Problems)...")
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        
        for i in range(min(n_samples - len(all_samples), len(dataset))):
            item = dataset[i]
            all_samples.append({
                'prompt': item['prompt'],
                'intent_label': 'coding',
                'source': 'mbpp',
                'metadata': {
                    'test_list': str(item.get('test_list', [])),
                    'task_id': item.get('task_id', i),
                }
            })
        print(f"  ✓ Collected {len(all_samples)} from MBPP")
    except Exception as e:
        print(f"  ✗ MBPP failed: {e}")
    
    # 2. HumanEval (~164 samples)
    if len(all_samples) < n_samples:
        try:
            print("  Loading HumanEval...")
            dataset = load_dataset("openai/openai_humaneval", split="test")
            
            for i in range(min(n_samples - len(all_samples), len(dataset))):
                item = dataset[i]
                prompt = item.get('prompt', '').strip()
                if prompt:
                    # Extract just the docstring/description
                    all_samples.append({
                        'prompt': f"Complete this Python function:\n{prompt}",
                        'intent_label': 'coding',
                        'source': 'humaneval',
                        'metadata': {
                            'task_id': item.get('task_id', ''),
                            'entry_point': item.get('entry_point', ''),
                        }
                    })
            print(f"  ✓ Collected {len(all_samples)} total (added {len(all_samples) - len([s for s in all_samples if s['source'] == 'mbpp'])} from HumanEval)")
        except Exception as e:
            print(f"  ✗ HumanEval failed: {e}")
    
    # 3. CodeAlpaca (20k instruction-following code tasks)
    if len(all_samples) < n_samples:
        try:
            print("  Loading CodeAlpaca...")
            dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
            
            count = 0
            for item in dataset:
                if len(all_samples) >= n_samples:
                    break
                
                instruction = item.get('instruction', '').strip()
                if instruction and len(instruction) > 20:
                    # Filter for Python-specific tasks
                    if any(kw in instruction.lower() for kw in ['python', 'function', 'class', 'code', 'implement']):
                        all_samples.append({
                            'prompt': instruction,
                            'intent_label': 'coding',
                            'source': 'code_alpaca',
                            'metadata': {'idx': count}
                        })
                        count += 1
            
            print(f"  ✓ Collected {len(all_samples)} total (added {count} from CodeAlpaca)")
        except Exception as e:
            print(f"  ✗ CodeAlpaca failed: {e}")
    
    # 4. APPS (programming problems)
    if len(all_samples) < n_samples:
        try:
            print("  Loading APPS...")
            dataset = load_dataset("codeparrot/apps", split="train", streaming=True)
            
            count = 0
            for item in dataset:
                if len(all_samples) >= n_samples:
                    break
                
                problem = item.get('question', '').strip()
                if problem and len(problem) > 50 and len(problem) < 1000:
                    all_samples.append({
                        'prompt': problem,
                        'intent_label': 'coding',
                        'source': 'apps',
                        'metadata': {'difficulty': item.get('difficulty', 'unknown')}
                    })
                    count += 1
            
            print(f"  ✓ Collected {len(all_samples)} total (added {count} from APPS)")
        except Exception as e:
            print(f"  ✗ APPS failed: {e}")
    
    print(f"  ✓ Total coding samples: {len(all_samples)}")
    return all_samples


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
                    'intent_label': 'factual_qa',  # Ground truth from dataset source
                    'source': 'natural_questions',
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
                'intent_label': 'factual_qa',  # Ground truth from dataset source
                'source': 'trivia_qa',
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


def load_summarization_data(n_samples: int = 2000) -> List[Dict]:
    """
    Load summarization datasets for SUMMARIZATION class.
    
    Sources: CNN/DailyMail, XSum
    
    Args:
        n_samples: Number of samples to extract
        
    Returns:
        List of prompts with metadata
    """
    try:
        from datasets import load_dataset
        print("\n📄 Loading Summarization Datasets (CNN/DailyMail, XSum)...")
        
        samples = []
        target_per_source = n_samples // 2
        
        # 1. CNN/DailyMail
        try:
            print("  Loading CNN/DailyMail...")
            cnn_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
            
            for i, item in enumerate(cnn_dataset):
                if len([s for s in samples if s['source'] == 'cnn_dailymail']) >= target_per_source:
                    break
                
                article = item.get('article', '').strip()
                if len(article) > 100:  # Ensure substantial content
                    # Create natural summarization prompt
                    prompt = f"Summarize this article:\n\n{article[:1000]}"  # Limit to 1000 chars
                    samples.append({
                        'prompt': prompt,
                        'intent_label': 'summarization',  # Ground truth from dataset source
                        'source': 'cnn_dailymail',
                        'metadata': {
                            'idx': i,
                            'highlights': item.get('highlights', ''),
                        }
                    })
            
            print(f"    ✓ Collected {len([s for s in samples if s['source'] == 'cnn_dailymail'])} from CNN/DailyMail")
        except Exception as e:
            print(f"    ✗ Error loading CNN/DailyMail: {e}")
        
        # 2. Continue with CNN/DailyMail if we need more
        if len(samples) < n_samples:
            print(f"  Collecting remaining {n_samples - len(samples)} from CNN/DailyMail...")
            try:
                cnn_dataset2 = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation", streaming=True)
                
                for i, item in enumerate(cnn_dataset2):
                    if len(samples) >= n_samples:
                        break
                    
                    article = item.get('article', '').strip()
                    if len(article) > 100:
                        prompt = f"Summarize this article:\n\n{article[:1000]}"
                        samples.append({
                            'prompt': prompt,
                            'intent_label': 'summarization',  # Ground truth from dataset source
                            'source': 'cnn_dailymail',
                            'metadata': {
                                'idx': f"val_{i}",
                                'highlights': item.get('highlights', ''),
                            }
                        })
                
                print(f"    ✓ Total CNN/DailyMail: {len([s for s in samples if s['source'] == 'cnn_dailymail'])}")
            except Exception as e:
                print(f"    ✗ Error loading CNN/DailyMail validation: {e}")
        
        print(f"  ✓ Total summarization prompts: {len(samples)}")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading summarization datasets: {e}")
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
        # Load non-streaming to fix collection issues
        dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:5000]")
        
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
                                'intent_label': 'agentic_execution',  # Ground truth from dataset source
                                'source': 'glaive_function_calling',
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
        print("\n💬 Loading General Conversation Dataset...")
        
        # Try multiple datasets in order
        dataset = None
        try:
            print("  Trying WildChat...")
            dataset = load_dataset("allenai/WildChat", split="train", streaming=True)
            dataset_name = "wildchat"
        except Exception as e1:
            print(f"    ✗ WildChat failed: {e1}")
            try:
                print("  Trying LMSYS-Chat-1M...")
                dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
                dataset_name = "lmsys_chat"
            except Exception as e2:
                print(f"    ✗ LMSYS failed: {e2}")
                try:
                    print("  Trying OpenOrca...")
                    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
                    dataset_name = "openorca"
                except Exception as e3:
                    print(f"    ✗ OpenOrca failed: {e3}")
                    print("  ✗ All general conversation datasets failed")
                    return []
        
        if dataset is None:
            return []
        
        samples = []
        code_pattern = re.compile(r'```')
        math_pattern = re.compile(r'[$\\]|\\[a-zA-Z]+\{')  # LaTeX
        
        for i, item in enumerate(dataset):
            if len(samples) >= n_samples:
                break
            
            # Extract content based on dataset structure
            content = None
            if dataset_name == "wildchat":
                conversation = item.get('conversation', [])
                if conversation and len(conversation) > 0:
                    first_msg = conversation[0]
                    if isinstance(first_msg, dict) and first_msg.get('role') == 'user':
                        content = first_msg.get('content', '')
            elif dataset_name == "lmsys_chat":
                conversation = item.get('conversation', [])
                if conversation and len(conversation) > 0:
                    first_msg = conversation[0]
                    if isinstance(first_msg, dict) and first_msg.get('role') == 'user':
                        content = first_msg.get('content', '')
            elif dataset_name == "openorca":
                # OpenOrca has 'question' field
                content = item.get('question', '')
            
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
                'intent_label': 'general',  # Ground truth from dataset source
                'source': dataset_name,
                'metadata': {
                    'idx': i,
                }
            })
        
        print(f"  ✓ Collected {len(samples)} general prompts from {dataset_name} (filtered)")
        return samples
        
    except Exception as e:
        print(f"  ✗ Error loading LMSYS: {e}")
        return []


def save_raw_prompts(samples: List[Dict], output_path: str):
    """Save collected prompts with ground-truth labels from dataset sources."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count labels
    label_counts = defaultdict(int)
    for s in samples:
        label_counts[s['intent_label']] += 1
    
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'total_samples': len(samples),
                'intent_classes': 6,
                'label_counts': dict(label_counts),
                'sources': list(set(s['source'] for s in samples)),
                'note': 'Ground-truth labels from dataset sources (no teacher labeling needed)',
                'methodology': 'Dataset source defines intent label',
            },
            'samples': samples,
        }, f, indent=2)
    
    print(f"\n💾 Saved {len(samples)} labeled prompts to: {output_path}")


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
    print("Real Intent Data Collection (6 Classes)")
    print("="*60)
    print(f"\nTarget: {args.samples_per_class} samples per class")
    print(f"Total: {args.samples_per_class * 6} prompts")
    print("\nIntent classes:")
    print("  1. CODING → CCS (Composite Coding Score)")
    print("  2. REASONING → CRS (Composite Reasoning Score)")
    print("  3. FACTUAL_QA → CFS (Composite Factual Score)")
    print("  4. SUMMARIZATION → CSS (Composite Summarization Score)")
    print("  5. AGENTIC_EXECUTION → CAE (Composite Agentic Execution Score)")
    print("  6. GENERAL → Arena rankings (catch-all)")
    
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
    
    # 1. CODING - MBPP
    coding_samples = load_mbpp_coding(args.samples_per_class)
    all_samples.extend(coding_samples)
    
    # 2. REASONING - GSM8k
    reasoning_samples = load_gsm8k_reasoning(args.samples_per_class)
    all_samples.extend(reasoning_samples)
    
    # 3. FACTUAL_QA - Natural Questions
    qa_samples = load_natural_questions_qa(args.samples_per_class)
    all_samples.extend(qa_samples)
    
    # 4. SUMMARIZATION - CNN/DailyMail, XSum
    summarization_samples = load_summarization_data(args.samples_per_class)
    all_samples.extend(summarization_samples)
    
    # 5. AGENTIC_EXECUTION - Glaive Function Calling
    agentic_samples = load_glaive_agentic(args.samples_per_class)
    all_samples.extend(agentic_samples)
    
    # 6. GENERAL - LMSYS Chat (filtered)
    general_samples = load_lmsys_general(args.samples_per_class)
    all_samples.extend(general_samples)
    
    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    
    by_label = defaultdict(int)
    by_source = defaultdict(int)
    
    for sample in all_samples:
        by_label[sample['intent_label']] += 1
        by_source[sample['source']] += 1
    
    print("\nBy Intent Label (Ground Truth):")
    for label, count in sorted(by_label.items()):
        print(f"  {label:<20} {count:>6} samples")
    
    print("\nBy Source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src:<30} {count:>6} samples")
    
    print(f"\nTotal collected: {len(all_samples)} prompts")
    
    # Save
    save_raw_prompts(all_samples, args.output)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Create train/val/test splits:")
    print("   python scripts/intent_classification/split_intent_data.py")
    print("\n2. Train intent classifiers and compare methods:")
    print("   - Embedding-based (e.g., SentenceTransformers + XGBoost)")
    print("   - Fine-tuned transformer (e.g., BERT, RoBERTa)")
    print("   - Few-shot LLM (e.g., GPT-4, Claude)")
    print("\n3. Evaluate which method best predicts ground-truth labels")
    print("   python scripts/intent_classification/evaluate_classifiers.py")
    
    print("\n✅ Data collection complete with ground-truth labels!")


if __name__ == '__main__':
    main()

