#!/usr/bin/env python3
"""Inspect High PC1 prompts to identify the dominant pattern."""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from collections import Counter

# Paths from the original script
project_root = Path("/Users/annette/repostitories/banditGPT")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)

def load_lmsys_holdout_with_gaps(dev_file, holdout_file):
    prompt_rewards = {}
    for file_path in [dev_file, holdout_file]:
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    prompt = entry.get('prompt', '').strip()
                    model_id = entry.get('model_id', '')
                    raw_score = entry.get('raw_score', None)
                    
                    if not prompt or raw_score is None: continue
                    
                    if prompt not in prompt_rewards: prompt_rewards[prompt] = {}
                    
                    if 'mixtral' in model_id.lower():
                        prompt_rewards[prompt]['mixtral'] = raw_score
                    elif 'gpt-4-turbo' in model_id.lower() or 'gpt-4' in model_id.lower():
                        prompt_rewards[prompt]['gpt4'] = raw_score
                except: continue
                
    prompts = []
    reward_gaps = []
    for prompt, rewards in prompt_rewards.items():
        if 'mixtral' in rewards and 'gpt4' in rewards:
            gap = rewards['gpt4'] - rewards['mixtral']
            prompts.append(prompt)
            reward_gaps.append(gap)
            
    return prompts, np.array(reward_gaps)

# Load Data
prompts, reward_gaps = load_lmsys_holdout_with_gaps(CANONICAL_DEV_DATA_PATH, CANONICAL_HOLDOUT_DATA_PATH)
print(f"Loaded {len(prompts)} prompts.")

# Load PCA
pca = joblib.load(DEFAULT_PCA_PATH)
encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
embeddings = encoder.encode(prompts, normalize_embeddings=True)
X_2d = pca.transform(embeddings)[:, :2]

# Analyze High PC1 cluster
pc1 = X_2d[:, 0]
high_mask = pc1 >= 0.3
high_pc1_indices = np.where(high_mask)[0]

print(f"\n{'='*80}")
print(f"HIGH PC1 CLUSTER ANALYSIS (N={len(high_pc1_indices)})")
print(f"{'='*80}")

# Identify template patterns
template_patterns = {
    "text_completion": "You are the text completion model",
    "code_completion": "complete the code",
    "system_instruction": "You are a",
    "strict_format": "only send",
    "no_explanation": "don't explain",
}

pattern_counts = {k: 0 for k in template_patterns}
pattern_examples = {k: [] for k in template_patterns}

for idx in high_pc1_indices:
    prompt = prompts[idx]
    prompt_lower = prompt.lower()
    
    for pattern_name, pattern_str in template_patterns.items():
        if pattern_str.lower() in prompt_lower:
            pattern_counts[pattern_name] += 1
            if len(pattern_examples[pattern_name]) < 3:
                pattern_examples[pattern_name].append((prompt[:200], reward_gaps[idx]))

print(f"\nPattern Distribution:")
print(f"{'-'*80}")
for pattern_name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / len(high_pc1_indices)
    print(f"{pattern_name:20s}: {count:4d} ({pct:5.1f}%)")

# Show dominant pattern examples
dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
print(f"\n{'='*80}")
print(f"DOMINANT PATTERN: {dominant_pattern} ({pattern_counts[dominant_pattern]}/{len(high_pc1_indices)} = {100*pattern_counts[dominant_pattern]/len(high_pc1_indices):.1f}%)")
print(f"{'='*80}")

for i, (prompt, gap) in enumerate(pattern_examples[dominant_pattern], 1):
    print(f"\nExample {i}:")
    print(f"Prompt: {prompt}...")
    print(f"Gap: {gap:.4f} (Mixtral wins)")

# Analyze diversity
print(f"\n{'='*80}")
print(f"DIVERSITY ANALYSIS")
print(f"{'='*80}")
unique_starts = Counter([p[:50] for i, p in enumerate(prompts) if high_mask[i]])
print(f"\nTop 5 Most Common Prompt Starts:")
for start, count in unique_starts.most_common(5):
    print(f"  [{count:3d}x] {start[:80]}...")

print(f"\nTotal unique 50-char prefixes: {len(unique_starts)}")
print(f"Entropy: {len(unique_starts) / len(high_pc1_indices):.2f} (1.0 = all unique, 0.0 = all same)")

