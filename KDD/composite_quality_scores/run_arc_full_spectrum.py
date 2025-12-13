#!/usr/bin/env python3
"""
Execute ARC Challenge validation across full CRS spectrum and create visualizations.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from existing validation script
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment
for env_path in ['.env', '../.env', '../../.env']:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        break


@dataclass
class ModelForTest:
    """Model to test."""
    name: str
    openrouter_id: str
    crs_score: float
    crs_rank: int
    crs_tier: str


@dataclass
class TestResult:
    """Results for a single model."""
    name: str
    openrouter_id: str
    crs_score: float
    crs_rank: int
    crs_tier: str
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    responses: List[Dict] = field(default_factory=list)


def load_model_selection() -> List[ModelForTest]:
    """Load the saved model selection."""
    selection_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_full_spectrum_model_selection.json"
    
    if not selection_path.exists():
        print(f"❌ Model selection not found. Run arc_full_spectrum_validation.py first.")
        sys.exit(1)
    
    with open(selection_path, 'r') as f:
        data = json.load(f)
    
    models = []
    for tier_name in ['high', 'mid', 'low']:
        for m in data['tiers'][tier_name]:
            models.append(ModelForTest(
                name=m['name'],
                openrouter_id=m['openrouter_id'],
                crs_score=m['crs_score'],
                crs_rank=m['crs_rank'],
                crs_tier=tier_name
            ))
    
    return models


def load_arc_challenge_problems(n_samples: int = 50, seed: int = 42) -> List[Dict]:
    """Load ARC-Challenge problems."""
    import numpy as np
    
    print(f"\n📚 Loading ARC-Challenge dataset...")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    # Sample
    np.random.seed(seed)
    indices = np.random.choice(len(arc), size=min(n_samples, len(arc)), replace=False)
    
    problems = []
    for idx in indices:
        item = arc[int(idx)]
        
        # Format prompt
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D)."
        
        problems.append({
            'id': f"ARC-CHALLENGE/{item['id']}",
            'question': item['question'],
            'prompt': prompt,
            'correct_letter': item['answerKey'],
        })
    
    print(f"   ✓ Loaded {len(problems)} problems")
    return problems


def call_model(openrouter_id: str, prompt: str, max_retries: int = 3) -> str:
    """Call OpenRouter API."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Token limits
    model_lower = openrouter_id.lower()
    is_reasoning = any(x in model_lower for x in ['reasoning', 'thinking', 'r1', 'o1', 'o3'])
    token_limits = [16000, 32000] if is_reasoning else [4000, 8000]
    
    for attempt in range(max_retries):
        try:
            tokens = token_limits[min(attempt, len(token_limits) - 1)]
            
            response = client.chat.completions.create(
                model=openrouter_id,
                max_tokens=tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content
            
            if isinstance(content, str) and content.strip():
                return content
            
            if isinstance(content, list):
                parts = [str(p.get("text", p.get("content", p))) if isinstance(p, dict) else str(p) for p in content]
                if parts:
                    return "\n".join(parts)
            
            if attempt < max_retries - 1:
                continue
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
    
    return ""


def extract_letter_answer(response: str) -> Optional[str]:
    """Extract multiple choice letter from response."""
    if not response:
        return None
    
    response_upper = response.upper()
    
    patterns = [
        r'ANSWER:\s*\(?([A-D])\)?',
        r'FINAL ANSWER:\s*\(?([A-D])\)?',
        r'THE ANSWER IS\s*\(?([A-D])\)?',
        r'CORRECT ANSWER IS\s*\(?([A-D])\)?',
        r'I (?:CHOOSE|SELECT|PICK)\s*\(?([A-D])\)?',
        r'OPTION\s*([A-D])\s*IS (?:CORRECT|THE ANSWER)',
        r'\*\*([A-D])\*\*\s*$',
        r'THEREFORE[,\s]+([A-D])\b',
        r'SO[,\s]+THE ANSWER IS\s*([A-D])\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match:
            return match.group(1)
    
    # Fallback
    last_300 = response_upper[-300:]
    match = re.search(r'\b([A-D])\b[.\s]*$', last_300)
    if match:
        return match.group(1)
    
    match = re.search(r'\b([A-D])\b', last_300)
    if match:
        return match.group(1)
    
    return None


def run_tests(models: List[ModelForTest], problems: List[Dict]) -> List[TestResult]:
    """Run tests on all models."""
    
    results = []
    total_calls = len(models) * len(problems)
    completed = 0
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"RUNNING ARC-CHALLENGE TESTS")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Problems: {len(problems)}")
    print(f"Total API calls: {total_calls}")
    print(f"{'='*80}\n")
    
    for model_idx, model in enumerate(models):
        print(f"\n{'─'*80}")
        print(f"Model {model_idx+1}/{len(models)}: {model.name}")
        print(f"CRS: {model.crs_score:.2f} (Rank {model.crs_rank}, {model.crs_tier.upper()} tier)")
        print(f"{'─'*80}")
        
        result = TestResult(
            name=model.name,
            openrouter_id=model.openrouter_id,
            crs_score=model.crs_score,
            crs_rank=model.crs_rank,
            crs_tier=model.crs_tier
        )
        
        for prob_idx, problem in enumerate(problems):
            # Progress
            elapsed = time.time() - start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining = (total_calls - completed) * avg_time
                eta_str = f"{int(remaining//3600)}:{int((remaining%3600)//60):02d}:{int(remaining%60):02d}"
            else:
                eta_str = "calculating..."
            
            pct = (completed / total_calls) * 100
            
            if prob_idx % 10 == 0:
                print(f"  Progress: {prob_idx}/{len(problems)} ({pct:.1f}% total, ETA: {eta_str})")
            
            try:
                response = call_model(model.openrouter_id, problem['prompt'])
                extracted = extract_letter_answer(response)
                is_correct = extracted == problem['correct_letter']
                
                result.total += 1
                if is_correct:
                    result.correct += 1
                
                result.responses.append({
                    'problem_id': problem['id'],
                    'correct_letter': problem['correct_letter'],
                    'extracted': extracted,
                    'is_correct': is_correct,
                })
                
                symbol = "✓" if is_correct else "✗"
                if prob_idx % 10 == 0 or prob_idx == len(problems) - 1:
                    print(f"  [{prob_idx+1:2d}] {symbol} Correct: {result.correct}/{result.total} ({result.correct/result.total*100:.1f}%)")
                
            except Exception as e:
                print(f"  [{prob_idx+1:2d}] ❌ Error: {str(e)[:50]}")
                result.total += 1
                result.responses.append({
                    'problem_id': problem['id'],
                    'correct_letter': problem['correct_letter'],
                    'extracted': None,
                    'is_correct': False,
                    'error': str(e)
                })
            
            completed += 1
        
        result.accuracy = (result.correct / result.total * 100) if result.total > 0 else 0
        results.append(result)
        
        print(f"\n  FINAL: {result.correct}/{result.total} = {result.accuracy:.1f}% accuracy")
    
    return results


def save_results(results: List[TestResult], output_dir: Path):
    """Save results to JSON."""
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_models': len(results),
            'n_problems': results[0].total if results else 0,
        },
        'models': [
            {
                'name': r.name,
                'openrouter_id': r.openrouter_id,
                'crs_score': r.crs_score,
                'crs_rank': r.crs_rank,
                'crs_tier': r.crs_tier,
                'correct': r.correct,
                'total': r.total,
                'accuracy': r.accuracy,
                'responses': r.responses,
            }
            for r in results
        ]
    }
    
    output_path = output_dir / "arc_full_spectrum_results.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return output_path


def create_visualizations(results: List[TestResult], output_dir: Path):
    """Create comprehensive visualizations."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr, pearsonr
    
    print(f"\n📊 Creating visualizations...")
    
    # Prepare data
    df = pd.DataFrame([
        {
            'name': r.name,
            'crs_score': r.crs_score,
            'crs_rank': r.crs_rank,
            'crs_tier': r.crs_tier,
            'accuracy': r.accuracy,
        }
        for r in results
    ])
    
    # Load existing results for comparison
    existing_path = output_dir / "arc_easy_vs_challenge_results.json"
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            existing = json.load(f)
        
        for m in existing['models']:
            df = pd.concat([df, pd.DataFrame([{
                'name': m['name'],
                'crs_score': m['crs_score'],
                'crs_rank': m['crs_rank'],
                'crs_tier': 'existing',
                'accuracy': m['challenge_accuracy'],
            }])], ignore_index=True)
    
    # Calculate correlations
    spearman_r, spearman_p = spearmanr(df['crs_score'], df['accuracy'])
    pearson_r, pearson_p = pearsonr(df['crs_score'], df['accuracy'])
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full spectrum scatter (main plot)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Color by tier
    tier_colors = {'high': 'forestgreen', 'mid': 'gold', 'low': 'coral', 'existing': 'lightgray'}
    
    for tier in ['existing', 'low', 'mid', 'high']:
        tier_data = df[df['crs_tier'] == tier]
        if len(tier_data) > 0:
            ax1.scatter(tier_data['crs_score'], tier_data['accuracy'], 
                       s=100, alpha=0.7, c=tier_colors[tier], 
                       label=f"{tier.upper()} ({len(tier_data)})", 
                       edgecolors='black', linewidth=1)
    
    # Trend line
    z = np.polyfit(df['crs_score'], df['accuracy'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['crs_score'].min(), df['crs_score'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Linear fit')
    
    # Annotations
    ax1.text(0.05, 0.95, 
             f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})\n" +
             f"Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})\n" +
             f"n = {len(df)} models",
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('CRS Score (Composite Reasoning)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy on ARC-Challenge (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Full CRS Spectrum: Model Capability vs Reasoning Accuracy', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution by tier (box plot)
    ax2 = fig.add_subplot(gs[0, 2])
    
    tier_order = ['low', 'mid', 'high']
    tier_data = df[df['crs_tier'].isin(tier_order)]
    
    sns.boxplot(data=tier_data, y='crs_tier', x='accuracy', 
                order=tier_order, palette=[tier_colors[t] for t in tier_order], ax=ax2)
    sns.stripplot(data=tier_data, y='crs_tier', x='accuracy', 
                  order=tier_order, color='black', alpha=0.3, ax=ax2)
    
    ax2.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('CRS Tier', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy Distribution\nby CRS Tier', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Tier statistics
    ax3 = fig.add_subplot(gs[1, 2])
    
    tier_stats = tier_data.groupby('crs_tier')['accuracy'].agg(['mean', 'std', 'min', 'max'])
    tier_stats = tier_stats.reindex(tier_order)
    
    ax3.axis('off')
    table_data = []
    for tier in tier_order:
        stats = tier_stats.loc[tier]
        table_data.append([
            tier.upper(),
            f"{stats['mean']:.1f}%",
            f"±{stats['std']:.1f}%",
            f"{stats['min']:.1f}%-{stats['max']:.1f}%"
        ])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Tier', 'Mean', 'Std Dev', 'Range'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax3.set_title('Accuracy Statistics by Tier', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 4: Residuals (over/under performers)
    ax4 = fig.add_subplot(gs[2, :2])
    
    from sklearn.linear_model import LinearRegression
    X = df['crs_score'].values.reshape(-1, 1)
    y = df['accuracy'].values
    model = LinearRegression()
    model.fit(X, y)
    df['predicted'] = model.predict(X)
    df['residual'] = df['accuracy'] - df['predicted']
    
    colors_resid = [tier_colors[t] for t in df['crs_tier']]
    ax4.scatter(df['crs_score'], df['residual'], s=100, alpha=0.7, c=colors_resid, edgecolors='black')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Label outliers
    outliers = df[abs(df['residual']) > 10]
    for _, row in outliers.iterrows():
        ax4.annotate(row['name'][:20], (row['crs_score'], row['residual']),
                    fontsize=8, alpha=0.7)
    
    ax4.set_xlabel('CRS Score', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residual (Actual - Predicted, %)', fontsize=11, fontweight='bold')
    ax4.set_title('Over/Under Performers Relative to CRS', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Summary statistics
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    summary_text = f"""
KEY FINDINGS

Total Models: {len(df)}
• New: {len(df[df['crs_tier'] != 'existing'])}
• Existing: {len(df[df['crs_tier'] == 'existing'])}

Correlation:
• Spearman ρ = {spearman_r:.3f}
• {'Strong' if abs(spearman_r) > 0.7 else 'Moderate' if abs(spearman_r) > 0.4 else 'Weak'}

Tier Performance:
• High CRS: {tier_stats.loc['high', 'mean']:.1f}%
• Mid CRS: {tier_stats.loc['mid', 'mean']:.1f}%
• Low CRS: {tier_stats.loc['low', 'mean']:.1f}%

Range: {df['accuracy'].min():.1f}% - {df['accuracy'].max():.1f}%
"""
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('ARC-Challenge: Full CRS Spectrum Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plot_path = output_dir / "arc_full_spectrum_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {plot_path}")
    
    plt.close()
    
    return df, spearman_r, pearson_r


def print_summary(results: List[TestResult], spearman_r: float, pearson_r: float):
    """Print summary statistics."""
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Overall
    print(f"\nOverall:")
    print(f"   Models tested: {len(results)}")
    accuracies = [r.accuracy for r in results]
    print(f"   Mean accuracy: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")
    print(f"   Range: {min(accuracies):.1f}% - {max(accuracies):.1f}%")
    
    # By tier
    print(f"\nBy CRS Tier:")
    for tier in ['high', 'mid', 'low']:
        tier_results = [r for r in results if r.crs_tier == tier]
        if tier_results:
            tier_accs = [r.accuracy for r in tier_results]
            print(f"   {tier.upper():<8} Mean: {np.mean(tier_accs):.1f}%, Range: {min(tier_accs):.1f}%-{max(tier_accs):.1f}%")
    
    # Correlation
    print(f"\nCorrelation (CRS vs Accuracy):")
    print(f"   Spearman ρ: {spearman_r:+.3f}")
    print(f"   Pearson r:  {pearson_r:+.3f}")
    
    # Top/Bottom performers
    sorted_results = sorted(results, key=lambda x: x.accuracy, reverse=True)
    print(f"\nTop 5 Performers:")
    for r in sorted_results[:5]:
        print(f"   {r.name[:40]:<42} CRS: {r.crs_score:+.2f}, Accuracy: {r.accuracy:.1f}%")
    
    print(f"\nBottom 5 Performers:")
    for r in sorted_results[-5:]:
        print(f"   {r.name[:40]:<42} CRS: {r.crs_score:+.2f}, Accuracy: {r.accuracy:.1f}%")


def main():
    print("="*80)
    print("ARC FULL SPECTRUM: EXECUTION")
    print("="*80)
    
    # Load models
    models = load_model_selection()
    print(f"\n✓ Loaded {len(models)} models")
    
    # Load problems
    problems = load_arc_challenge_problems(n_samples=50, seed=42)
    
    # Run tests
    results = run_tests(models, problems)
    
    # Save results
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    save_results(results, output_dir)
    
    # Create visualizations
    df, spearman_r, pearson_r = create_visualizations(results, output_dir)
    
    # Print summary
    print_summary(results, spearman_r, pearson_r)
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
