"""Show example restricted queries from the dataset"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from final_release.kdd_paper.table_3.router_performance_comparison import load_battle_dataset
from final_release.high_risk_prompt_classifier import HighRiskPromptClassifier

# Load data
df = load_battle_dataset(1000)

# Classify
clf = HighRiskPromptClassifier(threshold=5.0)

print("=" * 80)
print("RESTRICTED QUERIES (Policy Violations - Medical/Legal/Financial)")
print("=" * 80)

restricted_queries = []

for idx, row in df.iterrows():
    question = row['question']
    result = clf.classify(question)
    
    if result.label == "high":
        restricted_queries.append({
            'question': question,
            'score': result.score,
            'rules': result.matched_rules,
            'text': result.matched_text
        })

print(f"\nFound {len(restricted_queries)} restricted queries out of 1000 ({100*len(restricted_queries)/1000:.1f}%)\n")

for i, item in enumerate(restricted_queries, 1):
    print(f"\n{'-' * 80}")
    print(f"RESTRICTED QUERY #{i}")
    print(f"{'-' * 80}")
    print(f"Score: {item['score']:.1f} (threshold: 5.0)")
    print(f"Matched Rules: {', '.join(item['rules'][:5])}{'...' if len(item['rules']) > 5 else ''}")
    print(f"Matched Text: {', '.join(item['text'][:5])}{'...' if len(item['text']) > 5 else ''}")
    print(f"\nQuestion:")
    print(f"  {item['question'][:300]}{'...' if len(item['question']) > 300 else ''}")
