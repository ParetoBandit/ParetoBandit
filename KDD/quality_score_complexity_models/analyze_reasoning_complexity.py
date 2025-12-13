#!/usr/bin/env python3
"""
Reasoning Prompt Complexity Analysis

This script loads all available reasoning datasets, classifies them using
NVIDIA's complexity classifier, and shows the distribution of complex vs
non-complex prompts by source.

Datasets analyzed:
- ARC-Easy: Simple science questions
- ARC-Challenge: Harder science reasoning
- GPQA-Diamond: Graduate-level scientific reasoning
- GSM8K: Grade school math word problems
- MMLU (select subjects): Multiple choice reasoning
- HellaSwag: Commonsense reasoning
- PIQA: Physical intuition reasoning
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_arc_easy(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load ARC-Easy prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading ARC-Easy...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    
    samples = list(dataset)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    
    prompts = []
    for item in samples:
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompts.append({
            'prompt_text': prompt,
            'source': 'ARC-Easy',
            'task_type': 'Science QA',
            'difficulty': 'Easy',
        })
    
    print(f"      ✓ Loaded {len(prompts)} prompts")
    return prompts


def load_arc_challenge(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load ARC-Challenge prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading ARC-Challenge...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    samples = list(dataset)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    
    prompts = []
    for item in samples:
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompts.append({
            'prompt_text': prompt,
            'source': 'ARC-Challenge',
            'task_type': 'Science QA',
            'difficulty': 'Challenge',
        })
    
    print(f"      ✓ Loaded {len(prompts)} prompts")
    return prompts


def load_gpqa(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load GPQA-Diamond prompts (graduate-level science)."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading GPQA-Diamond...")
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        
        samples = list(dataset)
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)
        
        prompts = []
        for item in samples:
            question = item.get('Question', item.get('question', ''))
            if not question:
                continue
                
            # Build multiple choice prompt
            options = []
            for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
                if item.get(key):
                    options.append(item[key])
            
            if len(options) >= 4:
                random.shuffle(options)
                prompt = f"{question}\n\nOptions:\n"
                for i, opt in enumerate(options[:4]):
                    prompt += f"{chr(65+i)}. {opt}\n"
            else:
                prompt = question
            
            prompts.append({
                'prompt_text': prompt,
                'source': 'GPQA-Diamond',
                'task_type': 'Graduate Science',
                'difficulty': 'Expert',
            })
        
        print(f"      ✓ Loaded {len(prompts)} prompts")
        return prompts
        
    except Exception as e:
        print(f"      ⚠️ GPQA not available: {e}")
        return []


def load_gsm8k(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load GSM8K math reasoning prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    samples = list(dataset)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    
    prompts = []
    for item in samples:
        prompt = f"Solve this math problem step by step:\n\n{item['question']}"
        prompts.append({
            'prompt_text': prompt,
            'source': 'GSM8K',
            'task_type': 'Math Reasoning',
            'difficulty': 'Grade School',
        })
    
    print(f"      ✓ Loaded {len(prompts)} prompts")
    return prompts


def load_mmlu(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load MMLU prompts from various subjects."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading MMLU...")
    
    # Select diverse subjects requiring reasoning
    subjects = [
        'abstract_algebra',
        'formal_logic',
        'logical_fallacies', 
        'philosophy',
        'college_physics',
        'high_school_physics',
        'conceptual_physics',
    ]
    
    all_prompts = []
    samples_per_subject = max(n_samples // len(subjects), 10)
    
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            samples = list(dataset)[:samples_per_subject]
            
            for item in samples:
                prompt = f"{item['question']}\n\nOptions:\n"
                for i, choice in enumerate(item['choices']):
                    prompt += f"{chr(65+i)}. {choice}\n"
                
                all_prompts.append({
                    'prompt_text': prompt,
                    'source': f'MMLU-{subject}',
                    'task_type': 'Academic Knowledge',
                    'difficulty': 'College',
                })
        except Exception as e:
            print(f"      ⚠️ MMLU-{subject} failed: {e}")
    
    # Limit total samples
    if len(all_prompts) > n_samples:
        all_prompts = random.sample(all_prompts, n_samples)
    
    print(f"      ✓ Loaded {len(all_prompts)} prompts")
    return all_prompts


def load_hellaswag(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load HellaSwag commonsense reasoning prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    samples = list(dataset)
    if len(samples) > n_samples:
        samples = random.sample(samples, n_samples)
    
    prompts = []
    for item in samples:
        context = item.get('ctx', item.get('context', ''))
        endings = item.get('endings', [])
        
        if context and endings:
            prompt = f"Complete the following:\n\n{context}\n\nOptions:\n"
            for i, ending in enumerate(endings):
                prompt += f"{chr(65+i)}. {ending}\n"
            
            prompts.append({
                'prompt_text': prompt,
                'source': 'HellaSwag',
                'task_type': 'Commonsense',
                'difficulty': 'Medium',
            })
    
    print(f"      ✓ Loaded {len(prompts)} prompts")
    return prompts


def load_piqa(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load PIQA physical intuition prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading PIQA...")
    try:
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
        
        samples = list(dataset)
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)
        
        prompts = []
        for item in samples:
            goal = item.get('goal', '')
            sol1 = item.get('sol1', '')
            sol2 = item.get('sol2', '')
            
            if goal and sol1 and sol2:
                prompt = f"Goal: {goal}\n\nWhich solution is better?\n\nA. {sol1}\nB. {sol2}\n"
                prompts.append({
                    'prompt_text': prompt,
                    'source': 'PIQA',
                    'task_type': 'Physical Intuition',
                    'difficulty': 'Medium',
                })
        
        print(f"      ✓ Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"      ⚠️ PIQA not available: {e}")
        return []


def load_winogrande(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """Load Winogrande commonsense reasoning prompts."""
    from datasets import load_dataset
    random.seed(seed)
    
    print(f"   Loading Winogrande...")
    try:
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        
        samples = list(dataset)
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)
        
        prompts = []
        for item in samples:
            sentence = item.get('sentence', '')
            opt1 = item.get('option1', '')
            opt2 = item.get('option2', '')
            
            if sentence and opt1 and opt2:
                prompt = f"Fill in the blank with the correct option:\n\n{sentence}\n\nOptions:\nA. {opt1}\nB. {opt2}\n"
                prompts.append({
                    'prompt_text': prompt,
                    'source': 'Winogrande',
                    'task_type': 'Commonsense',
                    'difficulty': 'Medium',
                })
        
        print(f"      ✓ Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"      ⚠️ Winogrande not available: {e}")
        return []


def classify_with_nvidia(prompts: List[Dict]) -> pd.DataFrame:
    """Classify all prompts with NVIDIA complexity classifier."""
    from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
    
    print(f"\n🤖 Classifying {len(prompts)} prompts with NVIDIA...")
    
    classifier = NvidiaComplexityClassifier()
    texts = [p['prompt_text'][:2000] for p in prompts]  # Truncate long prompts
    results = classifier.classify_batch(texts)
    
    records = []
    for prompt, result in zip(prompts, results):
        records.append({
            'source': prompt['source'],
            'task_type': prompt['task_type'],
            'difficulty': prompt['difficulty'],
            'is_complex': int(result.is_complex),
            'prompt_complexity_score': result.prompt_complexity_score,
            'creativity_scope': result.creativity_scope,
            'reasoning_score': result.reasoning,
            'constraint_ct': result.constraint_ct,
            'domain_knowledge': result.domain_knowledge,
            'complexity_level': result.complexity_level,
            'task_type_nvidia': result.task_type_1,
        })
    
    return pd.DataFrame(records)


def show_distribution(df: pd.DataFrame):
    """Show complexity distribution by source."""
    
    print(f"\n{'='*80}")
    print("COMPLEXITY DISTRIBUTION BY SOURCE")
    print(f"{'='*80}")
    
    # Group by source
    summary = df.groupby('source').agg({
        'is_complex': ['sum', 'count', 'mean'],
        'prompt_complexity_score': ['mean', 'std', 'min', 'max'],
        'creativity_scope': 'mean',
        'reasoning_score': 'mean',
    }).round(3)
    
    summary.columns = ['n_complex', 'total', 'pct_complex', 'avg_score', 'std_score', 'min_score', 'max_score', 'avg_creativity', 'avg_reasoning']
    summary['pct_complex'] = (summary['pct_complex'] * 100).round(1)
    summary = summary.sort_values('avg_score', ascending=False)
    
    print(f"\n{'Source':<25} {'Total':>8} {'Complex':>8} {'% Complex':>10} {'Avg Score':>10} {'Creativity':>10} {'Reasoning':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for source, row in summary.iterrows():
        print(f"{source:<25} {int(row['total']):>8} {int(row['n_complex']):>8} {row['pct_complex']:>9.1f}% {row['avg_score']:>10.3f} {row['avg_creativity']:>10.3f} {row['avg_reasoning']:>10.3f}")
    
    # Overall summary
    total = len(df)
    total_complex = df['is_complex'].sum()
    print(f"\n{'-'*93}")
    print(f"{'TOTAL':<25} {total:>8} {total_complex:>8} {total_complex/total*100:>9.1f}% {df['prompt_complexity_score'].mean():>10.3f} {df['creativity_scope'].mean():>10.3f} {df['reasoning_score'].mean():>10.3f}")
    
    # By complexity level
    print(f"\n\n{'='*80}")
    print("COMPLEXITY LEVEL DISTRIBUTION")
    print(f"{'='*80}")
    
    level_counts = df.groupby(['source', 'complexity_level']).size().unstack(fill_value=0)
    level_order = ['trivial', 'simple', 'moderate', 'complex', 'expert']
    level_counts = level_counts.reindex(columns=[l for l in level_order if l in level_counts.columns])
    
    print(f"\n{'Source':<25} " + " ".join([f"{level:>10}" for level in level_counts.columns]))
    print(f"{'-'*25} " + " ".join(["-"*10 for _ in level_counts.columns]))
    
    for source, row in level_counts.iterrows():
        print(f"{source:<25} " + " ".join([f"{int(row[level]):>10}" for level in level_counts.columns]))
    
    # Task type from NVIDIA
    print(f"\n\n{'='*80}")
    print("NVIDIA DETECTED TASK TYPES")
    print(f"{'='*80}")
    
    task_counts = df['task_type_nvidia'].value_counts()
    print(f"\n{'Task Type':<30} {'Count':>10} {'Percentage':>12}")
    print(f"{'-'*30} {'-'*10} {'-'*12}")
    for task, count in task_counts.items():
        print(f"{task:<30} {count:>10} {count/len(df)*100:>11.1f}%")
    
    return summary


def create_visualization(df: pd.DataFrame, output_dir: Path):
    """Create visualization of complexity distribution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Complexity score distribution by source
    ax1 = axes[0, 0]
    sources = df['source'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))
    
    for source, color in zip(sorted(sources), colors):
        subset = df[df['source'] == source]['prompt_complexity_score']
        ax1.hist(subset, bins=20, alpha=0.5, label=source, color=color)
    
    ax1.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='is_complex threshold')
    ax1.set_xlabel('Complexity Score', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Complexity Score Distribution by Source', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: % Complex by source
    ax2 = axes[0, 1]
    source_stats = df.groupby('source')['is_complex'].mean() * 100
    source_stats = source_stats.sort_values(ascending=True)
    
    bars = ax2.barh(source_stats.index, source_stats.values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('% Complex (score >= 0.4)', fontweight='bold')
    ax2.set_title('Percentage of Complex Prompts by Source', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, source_stats.values):
        ax2.annotate(f'{val:.1f}%', xy=(val + 1, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=9)
    
    # Plot 3: Creativity vs Reasoning by source
    ax3 = axes[1, 0]
    source_means = df.groupby('source').agg({
        'creativity_scope': 'mean',
        'reasoning_score': 'mean',
        'is_complex': 'mean',
    })
    
    scatter = ax3.scatter(source_means['creativity_scope'], source_means['reasoning_score'],
                         c=source_means['is_complex'], cmap='RdYlGn', s=100, alpha=0.7)
    
    for source in source_means.index:
        ax3.annotate(source, xy=(source_means.loc[source, 'creativity_scope'], 
                                  source_means.loc[source, 'reasoning_score']),
                    fontsize=8, ha='center', va='bottom')
    
    ax3.set_xlabel('Average Creativity Score', fontweight='bold')
    ax3.set_ylabel('Average Reasoning Score', fontweight='bold')
    ax3.set_title('Creativity vs Reasoning by Source', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='% Complex')
    
    # Plot 4: Complexity level stacked bar
    ax4 = axes[1, 1]
    level_counts = df.groupby(['source', 'complexity_level']).size().unstack(fill_value=0)
    level_order = ['trivial', 'simple', 'moderate', 'complex', 'expert']
    level_counts = level_counts.reindex(columns=[l for l in level_order if l in level_counts.columns])
    level_pcts = level_counts.div(level_counts.sum(axis=1), axis=0) * 100
    
    level_pcts.plot(kind='barh', stacked=True, ax=ax4, 
                   color=['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850'])
    ax4.set_xlabel('Percentage', fontweight='bold')
    ax4.set_title('Complexity Level Distribution by Source', fontweight='bold')
    ax4.legend(title='Level', loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    plot_path = output_dir / "reasoning_complexity_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print("REASONING PROMPT COMPLEXITY ANALYSIS")
    print("Classifying reasoning datasets with NVIDIA complexity classifier")
    print("="*80)
    
    output_dir = Path(__file__).parent
    
    # Load all datasets
    print(f"\n📚 Loading reasoning datasets...")
    
    all_prompts = []
    
    # Load each dataset
    all_prompts.extend(load_arc_easy(n_samples=100))
    all_prompts.extend(load_arc_challenge(n_samples=100))
    all_prompts.extend(load_gpqa(n_samples=100))
    all_prompts.extend(load_gsm8k(n_samples=100))
    all_prompts.extend(load_mmlu(n_samples=100))
    all_prompts.extend(load_hellaswag(n_samples=100))
    all_prompts.extend(load_piqa(n_samples=100))
    all_prompts.extend(load_winogrande(n_samples=100))
    
    print(f"\n✓ Total prompts loaded: {len(all_prompts)}")
    
    # Show source counts
    source_counts = {}
    for p in all_prompts:
        source_counts[p['source']] = source_counts.get(p['source'], 0) + 1
    
    print(f"\n📊 Prompts by source:")
    for source, count in sorted(source_counts.items()):
        print(f"   • {source}: {count}")
    
    # Classify with NVIDIA
    df = classify_with_nvidia(all_prompts)
    
    # Show distribution
    summary = show_distribution(df)
    
    # Create visualization
    create_visualization(df, output_dir)
    
    # Save results
    results_path = output_dir / "reasoning_complexity_results.json"
    
    # Build by_source dict with proper keys
    by_source = {}
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        by_source[source] = {
            'n_complex': int(subset['is_complex'].sum()),
            'total': len(subset),
            'pct_complex': float(subset['is_complex'].mean() * 100),
            'avg_complexity_score': float(subset['prompt_complexity_score'].mean()),
            'avg_creativity': float(subset['creativity_scope'].mean()),
            'avg_reasoning': float(subset['reasoning_score'].mean()),
        }
    
    results = {
        'total_prompts': len(df),
        'total_complex': int(df['is_complex'].sum()),
        'pct_complex': float(df['is_complex'].mean() * 100),
        'sources': list(df['source'].unique()),
        'by_source': by_source,
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Save detailed CSV
    csv_path = output_dir / "reasoning_complexity_detailed.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed CSV saved: {csv_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"""
    Total prompts analyzed: {len(df)}
    Sources: {len(df['source'].unique())}
    
    Overall complexity:
    • Complex (score >= 0.4): {df['is_complex'].sum()} ({df['is_complex'].mean()*100:.1f}%)
    • Average score: {df['prompt_complexity_score'].mean():.3f}
    
    Key Finding:
    • NVIDIA's is_complex is driven primarily by CREATIVITY (35% weight)
    • Most reasoning benchmarks score LOW on creativity
    • Even "hard" reasoning tasks may not be "complex" by NVIDIA's definition
    
    This is important for routing: NVIDIA's complexity captures OPEN-ENDED/CREATIVE
    tasks rather than DIFFICULT/REASONING tasks.
    """)


if __name__ == "__main__":
    main()
