#!/bin/bash
# Merge rejudged holdout GPT-4-Turbo data

set -e

echo "========================================================================"
echo "MERGING REJUDGED HOLDOUT GPT-4-TURBO DATA"
echo "========================================================================"

cd /Users/annette/repostitories/banditGPT

# Check if rejudged file exists
if [ ! -f "data/holdout_rewards_gpt4turbo_rejudged.jsonl" ]; then
    echo "❌ Error: data/holdout_rewards_gpt4turbo_rejudged.jsonl not found"
    echo "   Rejudging may still be in progress"
    exit 1
fi

echo ""
echo "1️⃣  Analyzing rejudged data..."
python3 << 'PYTHON_SCRIPT'
import json
from collections import Counter

entries = []
with open('data/holdout_rewards_gpt4turbo_rejudged.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('ok', False):
            entries.append(entry)

print(f"  Total rejudged entries: {len(entries)}")

scores = [e['raw_score'] for e in entries]
score_dist = Counter(scores)
print(f"  Score distribution:")
for score in sorted(score_dist.keys()):
    count = score_dist[score]
    pct = count / len(scores) * 100
    print(f"    {score:.2f}: {count:4d} ({pct:5.1f}%)")
PYTHON_SCRIPT

echo ""
echo "2️⃣  Merging with holdout set..."
python3 << 'PYTHON_SCRIPT'
import json
import gzip
from collections import Counter

# Load holdout without GPT-4-Turbo
holdout_entries = []
with gzip.open('src/bandit_gpt/data/offline_dataset/holdout_rewards_complete_NO_GPT4TURBO.jsonl.gz', 'rt') as f:
    for line in f:
        holdout_entries.append(json.loads(line))

print(f"  Loaded {len(holdout_entries)} entries (Mixtral + GPT-4o)")

# Load rejudged GPT-4-Turbo
gpt4turbo_entries = []
with open('data/holdout_rewards_gpt4turbo_rejudged.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        if entry.get('ok', False):
            gpt4turbo_entries.append(entry)

print(f"  Loaded {len(gpt4turbo_entries)} rejudged GPT-4-Turbo entries")

# Get proper holdout prompts
holdout_prompts = set()
for entry in holdout_entries:
    if entry.get('ok', True):
        holdout_prompts.add(entry['prompt'])

print(f"  Holdout prompts: {len(holdout_prompts)}")

# Filter GPT-4-Turbo to only holdout prompts
filtered_gpt4turbo = [e for e in gpt4turbo_entries if e['prompt'] in holdout_prompts]
print(f"  Filtered GPT-4-Turbo to holdout prompts: {len(filtered_gpt4turbo)}")

# Merge
all_entries = holdout_entries + filtered_gpt4turbo

model_counts = Counter(e['model_id'] for e in all_entries if e.get('ok', True))
print(f"\n  Final entries by model:")
for model_id, count in sorted(model_counts.items()):
    print(f"    {model_id}: {count}")

# Save
with gzip.open('src/bandit_gpt/data/offline_dataset/holdout_rewards_complete_FINAL.jsonl.gz', 'wt') as f:
    for entry in all_entries:
        f.write(json.dumps(entry) + '\n')

print(f"\n  ✅ Saved: holdout_rewards_complete_FINAL.jsonl.gz")
PYTHON_SCRIPT

echo ""
echo "3️⃣  Replacing holdout file..."
cd src/bandit_gpt/data/offline_dataset
mv holdout_rewards_complete.jsonl.gz holdout_rewards_complete_BEFORE_FINAL.jsonl.gz
mv holdout_rewards_complete_FINAL.jsonl.gz holdout_rewards_complete.jsonl.gz

echo "  ✅ Replaced holdout_rewards_complete.jsonl.gz"

echo ""
echo "========================================================================"
echo "✅ MERGE COMPLETE!"
echo "========================================================================"

