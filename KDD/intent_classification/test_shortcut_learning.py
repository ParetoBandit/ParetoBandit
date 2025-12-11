#!/usr/bin/env python3
"""
Test for Shortcut Learning from Negative Filtering

Tests whether the model learned shortcuts like:
  contains "function" → CODING (wrong!)
  contains "class" → CODING (wrong!)
  
These words appear in non-coding contexts but were filtered from GENERAL.
"""

import pickle
import json
from sentence_transformers import SentenceTransformer

# Load model
print("Loading model...")
with open('../../results/intent_classification/xgboost_intent_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
labels = ['coding', 'factual_qa', 'general', 'reasoning', 'summarization']

def predict(prompt):
    """Predict intent with confidence."""
    embedding = embedder.encode([prompt], convert_to_numpy=True)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0]
    
    predicted_label = labels[int(pred)]
    confidence = float(prob[int(pred)])
    
    return {
        'prompt': prompt,
        'predicted': predicted_label,
        'confidence': confidence,
        'all_probs': {labels[i]: float(prob[i]) for i in range(len(labels))}
    }

# ============================================================================
# Test Cases: Non-coding contexts with filtered keywords
# ============================================================================

test_cases = {
    "KEYWORD: 'function' (biology/medical)": [
        "What is the primary function of the mitochondria?",
        "Describe the function of the liver in the human body",
        "What function does the heart serve?",
        "Explain the function of chloroplasts in plants",
    ],
    
    "KEYWORD: 'class' (education/social)": [
        "What time does your class start?",
        "I'm taking a history class this semester",
        "She's in a different social class than me",
        "The upper class owns most of the wealth",
    ],
    
    "KEYWORD: 'python' (animal)": [
        "How long can a python snake grow?",
        "Are python snakes venomous?",
        "Tell me about the Burmese python",
        "What do pythons eat in the wild?",
    ],
    
    "KEYWORD: 'variable' (statistics/science)": [
        "What is an independent variable in an experiment?",
        "Explain the difference between dependent and independent variables",
        "How do I control for confounding variables?",
        "Temperature is a variable in this experiment",
    ],
    
    "KEYWORD: 'loop' (everyday usage)": [
        "I'm stuck in a loop of negative thoughts",
        "The highway forms a loop around the city",
        "Let me loop you in on the conversation",
        "We're out of the loop on that decision",
    ],
    
    "KEYWORD: 'array' (military/general)": [
        "The troops were arranged in a defensive array",
        "They displayed an impressive array of skills",
        "An array of options is available",
        "The peacocks displayed their array of feathers",
    ],
}

# ============================================================================
# Run Tests
# ============================================================================

print("\n" + "="*80)
print("SHORTCUT LEARNING TEST")
print("="*80)
print("\nTesting if model learned: 'contains KEYWORD' → CODING (incorrect shortcut)\n")

all_results = []
shortcut_failures = []
correct_classifications = []

for category, prompts in test_cases.items():
    print(f"\n{category}")
    print("-" * 60)
    
    for prompt in prompts:
        result = predict(prompt)
        all_results.append({
            'category': category,
            'expected': 'NOT coding',
            **result
        })
        
        # Classify as failure if predicted CODING
        if result['predicted'] == 'coding':
            shortcut_failures.append(result)
            indicator = "❌ SHORTCUT"
        else:
            correct_classifications.append(result)
            indicator = "✅"
        
        print(f"  {indicator} [{result['predicted']:>15}] {result['confidence']:>5.1%} | {prompt[:55]}...")

# ============================================================================
# Analysis
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

total = len(all_results)
failures = len(shortcut_failures)
success_rate = (total - failures) / total * 100

print(f"\nTotal Test Cases: {total}")
print(f"Shortcut Failures (predicted CODING): {failures} ({failures/total*100:.1f}%)")
print(f"Correct (NOT coding): {total - failures} ({success_rate:.1f}%)")

if failures > 0:
    print(f"\n⚠️  SHORTCUT LEARNING DETECTED")
    print(f"   Model incorrectly classified {failures} non-coding prompts as CODING")
    print(f"   Likely learned: 'contains keyword' → CODING")
    
    print(f"\n   Failed Examples:")
    for f in shortcut_failures[:5]:
        print(f"   • '{f['prompt'][:60]}...'")
        print(f"     → Predicted: {f['predicted']} ({f['confidence']:.1%})")
else:
    print(f"\n✅ NO SHORTCUT LEARNING DETECTED")
    print(f"   Model correctly distinguished keyword context")

# Breakdown by keyword
print("\n" + "="*80)
print("BREAKDOWN BY KEYWORD")
print("="*80)

from collections import defaultdict
by_keyword = defaultdict(lambda: {'total': 0, 'coding': 0, 'predictions': defaultdict(int)})

for result in all_results:
    keyword = result['category'].split("'")[1] if "'" in result['category'] else 'unknown'
    by_keyword[keyword]['total'] += 1
    by_keyword[keyword]['predictions'][result['predicted']] += 1
    if result['predicted'] == 'coding':
        by_keyword[keyword]['coding'] += 1

for keyword, stats in sorted(by_keyword.items()):
    coding_rate = stats['coding'] / stats['total'] * 100
    print(f"\n'{keyword}': {stats['coding']}/{stats['total']} → CODING ({coding_rate:.0f}%)")
    for pred, count in sorted(stats['predictions'].items(), key=lambda x: -x[1]):
        print(f"  {pred:>15}: {count} ({count/stats['total']*100:.0f}%)")

# Save results
output = {
    'summary': {
        'total_tests': total,
        'shortcut_failures': failures,
        'success_rate': success_rate,
        'verdict': 'SHORTCUT_DETECTED' if failures > total * 0.2 else 'ROBUST'
    },
    'by_keyword': dict(by_keyword),
    'all_results': all_results,
    'failures': shortcut_failures
}

with open('shortcut_learning_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*80)
print("RECOMMENDATION FOR PAPER")
print("="*80)

if failures > total * 0.2:  # >20% failure rate
    print("""
⚠️  ACKNOWLEDGE AS LIMITATION (Section 6.2):

"The GENERAL class was filtered using negative heuristics (excluding prompts 
containing coding keywords like 'function', 'class', 'variable'). This creates
a risk of shortcut learning where the model associates these keywords with CODING
even in non-technical contexts. Our testing reveals [X]% misclassification rate
on prompts like 'What is the function of the mitochondria?' (biology) being 
classified as CODING. This is a known limitation of heuristic-based filtering
and would ideally be addressed by including GENERAL examples with these keywords
in non-technical contexts, which is difficult to automate."

Severity: MEDIUM - Acknowledge honestly
""")
else:
    print(f"""
✅ REPORT AS EVIDENCE OF ROBUSTNESS:

Despite negative filtering of keywords from GENERAL class, the model shows
{success_rate:.1f}% accuracy on non-coding contexts containing these keywords.

Examples:
  • "What is the function of the mitochondria?" → {[r for r in all_results if 'mitochondria' in r['prompt']][0]['predicted']}
  • "What time does your class start?" → {[r for r in all_results if 'class start' in r['prompt']][0]['predicted']}

This suggests semantic embeddings successfully capture context, mitigating the
risk of lexical shortcut learning.

Severity: LOW - Mention as validation of semantic approach
""")

print(f"\n💾 Saved results to: shortcut_learning_results.json")
