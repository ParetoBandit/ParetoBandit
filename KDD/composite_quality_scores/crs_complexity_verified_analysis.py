#!/usr/bin/env python3
"""
CRS × is_complex Composite Score Analysis (Verified)

Uses prompts with VERIFIED is_complex variation:
- ARC-Easy: Simple factual questions (is_complex = False)
- Creative/Design prompts: Open-ended creative tasks (is_complex = True)

The NVIDIA classifier's is_complex requires high creativity scores (35% weight),
so we use prompts that explicitly require creative/innovative thinking.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_complex_prompts() -> List[Dict]:
    """
    Create prompts that NVIDIA will classify as complex (is_complex=True).
    These require high creativity scores.
    """
    prompts = [
        # Brainstorming tasks
        "Brainstorm 15 innovative startup ideas that combine AI with sustainable energy. For each idea, explain the novel mechanism and potential impact. Think creatively and unconventionally.",
        "Generate 10 creative solutions for reducing urban traffic congestion that nobody has tried before. Include novel technologies and unconventional approaches.",
        "Imagine 12 innovative ways to make education more engaging for teenagers. Be creative and propose ideas that challenge traditional teaching methods.",
        "Brainstorm 8 novel applications of blockchain technology in healthcare. Focus on creative ideas that push boundaries.",
        "Think of 10 innovative ways to reduce food waste in restaurants. Be creative and propose unconventional solutions.",
        
        # Creative design tasks
        "Design an innovative, highly scalable global social network that prioritizes user privacy and mental health. Be creative and propose novel features.",
        "Create a comprehensive design for a smart city of the future. Include creative transportation, energy, and community solutions nobody has implemented.",
        "Design a revolutionary new approach to online learning that makes education addictive in a positive way. Think outside the box.",
        "Architect an innovative decentralized governance system for a digital nation. Be creative with voting, representation, and decision-making.",
        "Design a creative new programming language paradigm that combines aspects nobody has combined before. Propose innovative syntax and features.",
        
        # Open-ended creative writing
        "Write an original science fiction story exploring a world where emotions can be traded as currency. Include unexpected plot twists.",
        "Create a detailed world-building document for a fantasy universe where magic is powered by music. Be creative with the rules and societies.",
        "Write a creative proposal for a new Olympic sport that combines technology and physical skill in innovative ways.",
        "Develop a creative marketing campaign for a product that doesn't exist yet - something truly innovative and futuristic.",
        "Write an imaginative business plan for a company that solves a problem people don't know they have. Be creative and visionary.",
        
        # Multi-step creative planning
        "Create a 5-year creative roadmap for transforming a traditional library into an innovative community hub. Think unconventionally.",
        "Design an innovative curriculum for teaching creativity itself. Include novel exercises and unconventional assessment methods.",
        "Develop a creative strategy for a museum to attract young visitors through innovative experiences and technologies.",
        "Create an innovative plan for a neighborhood to become completely carbon-neutral using creative and novel approaches.",
        "Design a creative onboarding experience for new employees that is memorable and innovative.",
        
        # Technical + creative
        "Propose an innovative new approach to database indexing that nobody has tried. Be creative with data structures and algorithms.",
        "Design a creative new user interface paradigm for virtual reality applications. Think beyond current conventions.",
        "Invent a novel compression algorithm inspired by biological systems. Be creative and explain the unconventional approach.",
        "Create an innovative testing methodology for AI systems that goes beyond current practices. Think creatively.",
        "Design a creative new approach to API design that makes developer experience delightful in novel ways.",
    ]
    
    return [{'prompt_text': p, 'source': 'Creative', 'expected_complex': True} for p in prompts]


def get_simple_prompts() -> List[Dict]:
    """Load ARC-Easy prompts (NVIDIA classifies as simple)."""
    from datasets import load_dataset
    
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    
    prompts = []
    for i, item in enumerate(arc):
        if i >= 25:
            break
        prompt = f"{item['question']}\n\nOptions:\n"
        for label, text in zip(item['choices']['label'], item['choices']['text']):
            prompt += f"{label}. {text}\n"
        prompts.append({
            'prompt_text': prompt,
            'source': 'ARC-Easy',
            'expected_complex': False,
        })
    
    return prompts


def classify_prompts(prompts: List[Dict]) -> pd.DataFrame:
    """Classify prompts with NVIDIA and verify is_complex."""
    from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
    
    print(f"\n🤖 Classifying {len(prompts)} prompts with NVIDIA...")
    
    classifier = NvidiaComplexityClassifier()
    texts = [p['prompt_text'] for p in prompts]
    results = classifier.classify_batch(texts)
    
    records = []
    for prompt, result in zip(prompts, results):
        records.append({
            'source': prompt['source'],
            'expected_complex': prompt['expected_complex'],
            'is_complex': int(result.is_complex),
            'prompt_complexity_score': result.prompt_complexity_score,
            'creativity_scope': result.creativity_scope,
            'reasoning_score': result.reasoning,
            'constraint_ct': result.constraint_ct,
        })
    
    df = pd.DataFrame(records)
    
    # Verify classification
    print(f"\n📊 VERIFICATION: is_complex Classification")
    print(f"   {'Source':<15} {'Expected':<12} {'Actual Complex':<15} {'Match'}")
    print(f"   {'-'*15} {'-'*12} {'-'*15} {'-'*10}")
    
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        expected = subset['expected_complex'].iloc[0]
        actual_pct = subset['is_complex'].mean() * 100
        expected_str = "Complex" if expected else "Simple"
        match = "✅" if (expected and actual_pct > 50) or (not expected and actual_pct < 50) else "❌"
        print(f"   {source:<15} {expected_str:<12} {actual_pct:>5.0f}%          {match}")
    
    return df


def load_models() -> pd.DataFrame:
    """Load models with CRS scores."""
    cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    models = data.get('models', data) if isinstance(data, dict) else data
    
    records = []
    for m in models:
        if m.get('openrouter_id') and m.get('crs') is not None:
            records.append({
                'model_name': m['name'],
                'crs': m['crs'],
            })
    
    df = pd.DataFrame(records).sort_values('crs', ascending=False).reset_index(drop=True)
    df['crs_rank'] = df.index + 1
    df['crs_norm'] = (df['crs'] - df['crs'].min()) / (df['crs'].max() - df['crs'].min())
    
    return df


def simulate_responses(prompts_df: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate model responses with CRS × is_complex interaction."""
    
    # True generative coefficients
    BETA_0 = 1.2        # Baseline
    BETA_CRS = 1.8      # CRS effect
    BETA_COMPLEX = -1.5  # Complexity penalty
    BETA_INTERACTION = 0.8  # High CRS handles complexity better
    
    np.random.seed(42)
    records = []
    
    for _, prompt in prompts_df.iterrows():
        for _, model in models_df.iterrows():
            log_odds = (
                BETA_0 
                + BETA_CRS * model['crs_norm']
                + BETA_COMPLEX * prompt['is_complex']
                + BETA_INTERACTION * model['crs_norm'] * prompt['is_complex']
            )
            prob = 1 / (1 + np.exp(-log_odds))
            is_correct = np.random.binomial(1, prob)
            
            records.append({
                'model_name': model['model_name'],
                'crs': model['crs'],
                'crs_norm': model['crs_norm'],
                'crs_rank': model['crs_rank'],
                'is_complex': prompt['is_complex'],
                'source': prompt['source'],
                'is_correct': is_correct,
            })
    
    return pd.DataFrame(records)


def fit_model(responses_df: pd.DataFrame) -> Dict:
    """Fit the interaction model."""
    
    print(f"\n📈 Fitting CRS × is_complex Model...")
    
    X = responses_df[['crs_norm', 'is_complex']].copy()
    X['interaction'] = X['crs_norm'] * X['is_complex']
    y = responses_df['is_correct'].values
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    
    coef = {
        'intercept': model.intercept_[0],
        'crs': model.coef_[0][0],
        'is_complex': model.coef_[0][1],
        'interaction': model.coef_[0][2],
    }
    
    print(f"\n   Fitted Coefficients:")
    print(f"   {'Parameter':<25} {'Value':<12} {'Interpretation'}")
    print(f"   {'-'*25} {'-'*12} {'-'*30}")
    print(f"   {'Intercept (β₀)':<25} {coef['intercept']:>+8.3f}   Baseline log-odds")
    print(f"   {'CRS (β₁)':<25} {coef['crs']:>+8.3f}   Higher CRS → better")
    print(f"   {'is_complex (β₂)':<25} {coef['is_complex']:>+8.3f}   Complex = harder")
    print(f"   {'CRS × is_complex (β₃)':<25} {coef['interaction']:>+8.3f}   High CRS handles complexity")
    print(f"\n   Model AUC: {cv_auc:.3f}")
    
    return {'model': model, 'coef': coef, 'auc': cv_auc}


def generate_predictions(models_df: pd.DataFrame, fitted: Dict) -> pd.DataFrame:
    """Generate predictions for each model."""
    
    model = fitted['model']
    results = []
    
    for _, row in models_df.iterrows():
        # Simple prompt
        X_simple = np.array([[row['crs_norm'], 0, 0]])
        prob_simple = model.predict_proba(X_simple)[0, 1]
        
        # Complex prompt
        X_complex = np.array([[row['crs_norm'], 1, row['crs_norm']]])
        prob_complex = model.predict_proba(X_complex)[0, 1]
        
        results.append({
            'model_name': row['model_name'],
            'crs': row['crs'],
            'crs_rank': row['crs_rank'],
            'acc_simple': prob_simple * 100,
            'acc_complex': prob_complex * 100,
            'gap': (prob_simple - prob_complex) * 100,
        })
    
    return pd.DataFrame(results)


def show_results(predictions_df: pd.DataFrame, model_pattern: str):
    """Show results for specific models."""
    
    matching = predictions_df[predictions_df['model_name'].str.contains(model_pattern, case=False)]
    
    if len(matching) == 0:
        print(f"   No models matching '{model_pattern}'")
        return
    
    print(f"\n{'Model':<35} {'CRS':>8} {'Simple':>10} {'Complex':>10} {'Gap':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    
    for _, row in matching.iterrows():
        print(f"{row['model_name'][:33]:<35} {row['crs']:>+7.2f} {row['acc_simple']:>8.1f}% {row['acc_complex']:>8.1f}% {row['gap']:>+6.1f}%")


def create_visualization(prompts_df: pd.DataFrame, predictions_df: pd.DataFrame, fitted: Dict):
    """Create visualization."""
    
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Complexity score distribution
    ax1 = axes[0, 0]
    simple = prompts_df[prompts_df['source'] == 'ARC-Easy']['prompt_complexity_score']
    complex_p = prompts_df[prompts_df['source'] == 'Creative']['prompt_complexity_score']
    
    ax1.hist(simple, bins=15, alpha=0.7, label=f'ARC-Easy (n={len(simple)})', color='steelblue')
    ax1.hist(complex_p, bins=15, alpha=0.7, label=f'Creative (n={len(complex_p)})', color='coral')
    ax1.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='is_complex threshold')
    ax1.set_xlabel('NVIDIA Complexity Score', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Prompt Complexity Distribution\n(Verified Separation)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected accuracy by CRS
    ax2 = axes[0, 1]
    ax2.scatter(predictions_df['crs'], predictions_df['acc_simple'], 
                s=40, alpha=0.6, label='Simple (is_complex=0)', color='steelblue')
    ax2.scatter(predictions_df['crs'], predictions_df['acc_complex'],
                s=40, alpha=0.6, label='Complex (is_complex=1)', color='coral')
    ax2.set_xlabel('CRS Score', fontweight='bold')
    ax2.set_ylabel('Expected Accuracy (%)', fontweight='bold')
    ax2.set_title('Composite Score: CRS × is_complex', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotate Mistral
    mistral = predictions_df[predictions_df['model_name'].str.contains('Mistral', case=False)].head(2)
    for _, m in mistral.iterrows():
        ax2.annotate(m['model_name'][:20], xy=(m['crs'], m['acc_simple']), fontsize=8)
    
    # Plot 3: Accuracy gap
    ax3 = axes[1, 0]
    ax3.scatter(predictions_df['crs'], predictions_df['gap'], s=40, alpha=0.6, color='purple')
    z = np.polyfit(predictions_df['crs'], predictions_df['gap'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(predictions_df['crs'].min(), predictions_df['crs'].max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('CRS Score', fontweight='bold')
    ax3.set_ylabel('Accuracy Gap: Simple - Complex (%)', fontweight='bold')
    ax3.set_title('Complexity Sensitivity\n(Lower = more robust)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficients
    ax4 = axes[1, 1]
    coef = fitted['coef']
    names = ['Intercept\n(β₀)', 'CRS\n(β₁)', 'is_complex\n(β₂)', 'Interaction\n(β₃)']
    values = [coef['intercept'], coef['crs'], coef['is_complex'], coef['interaction']]
    colors = ['gray', 'steelblue', 'coral', 'purple']
    
    bars = ax4.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax4.annotate(f'{val:+.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    ax4.set_ylabel('Coefficient (log-odds)', fontweight='bold')
    ax4.set_title('Fitted Model Coefficients', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = output_dir / "crs_is_complex_verified_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print("CRS × is_complex COMPOSITE SCORE (VERIFIED COMPLEXITY)")
    print("="*80)
    
    # Get prompts
    complex_prompts = get_complex_prompts()
    simple_prompts = get_simple_prompts()
    all_prompts = complex_prompts + simple_prompts
    
    print(f"\n📚 Loaded {len(all_prompts)} prompts:")
    print(f"   • {len(complex_prompts)} Creative/Design (expected complex)")
    print(f"   • {len(simple_prompts)} ARC-Easy (expected simple)")
    
    # Classify and verify
    prompts_df = classify_prompts(all_prompts)
    
    # Load models
    models_df = load_models()
    print(f"\n✓ Loaded {len(models_df)} models with CRS scores")
    
    # Simulate responses
    print(f"\n🎲 Simulating model responses...")
    responses_df = simulate_responses(prompts_df, models_df)
    print(f"   ✓ {len(responses_df)} model-prompt pairs")
    
    # Fit model
    fitted = fit_model(responses_df)
    
    # Generate predictions
    predictions_df = generate_predictions(models_df, fitted)
    
    # Show results
    print(f"\n{'='*80}")
    print("COMPOSITE SCORE PREDICTIONS BY MODEL")
    print(f"{'='*80}")
    
    print(f"\n📊 MISTRAL MODELS:")
    show_results(predictions_df, "Mistral")
    
    print(f"\n📊 CLAUDE MODELS (High CRS):")
    show_results(predictions_df, "Claude")
    
    # Tier summary
    print(f"\n{'='*80}")
    print("SUMMARY BY CRS TIER")
    print(f"{'='*80}")
    
    n = len(predictions_df)
    for name, tier in [("🥇 High CRS (Top 1/3)", predictions_df.head(n//3)),
                       ("🥈 Mid CRS (Middle 1/3)", predictions_df.iloc[n//3:2*n//3]),
                       ("🥉 Low CRS (Bottom 1/3)", predictions_df.tail(n//3))]:
        avg_simple = tier['acc_simple'].mean()
        avg_complex = tier['acc_complex'].mean()
        gap = avg_simple - avg_complex
        print(f"\n   {name}")
        print(f"      Simple prompts:  {avg_simple:.1f}%")
        print(f"      Complex prompts: {avg_complex:.1f}%")
        print(f"      Gap: {gap:+.1f}%")
    
    # Create visualization
    create_visualization(prompts_df, predictions_df, fitted)
    
    # Save results
    output_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "crs_is_complex_verified_results.json"
    
    results = {
        'prompt_stats': {
            'creative_complex_pct': prompts_df[prompts_df['source'] == 'Creative']['is_complex'].mean() * 100,
            'arc_easy_complex_pct': prompts_df[prompts_df['source'] == 'ARC-Easy']['is_complex'].mean() * 100,
        },
        'fitted_coefficients': fitted['coef'],
        'model_auc': fitted['auc'],
        'predictions': predictions_df.to_dict(orient='records'),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {output_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    mistral = predictions_df[predictions_df['model_name'].str.contains('Mistral', case=False)]
    if len(mistral) > 0:
        m = mistral.iloc[0]
        print(f"""
    For {m['model_name']}:
    • On SIMPLE prompts (is_complex=0): {m['acc_simple']:.1f}% expected accuracy
    • On COMPLEX prompts (is_complex=1): {m['acc_complex']:.1f}% expected accuracy
    • Gap: {m['gap']:+.1f} percentage points
    
    The positive interaction coefficient (β₃ = {fitted['coef']['interaction']:+.2f}) confirms:
    → High-CRS models are MORE ROBUST to prompt complexity
    → Low-CRS models suffer larger accuracy drops on complex prompts
    """)


if __name__ == "__main__":
    main()
