
import sys
from pathlib import Path
import json
import gzip
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

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

# Analyze Clusters
pc1 = X_2d[:, 0]
low_mask = pc1 < 0.3
high_mask = pc1 >= 0.3

low_gaps = reward_gaps[low_mask]
high_gaps = reward_gaps[high_mask]

print(f"\n--- CLUSTER ANALYSIS ---")
print(f"Low PC1 (< 0.3) Count: {len(low_gaps)}")
print(f"High PC1 (>= 0.3) Count: {len(high_gaps)}")

print(f"\nMean Reward Gap (GPT4 - Mixtral):")
print(f"  Low PC1: {np.mean(low_gaps):.4f}")
print(f"  High PC1: {np.mean(high_gaps):.4f}")

print(f"\nMedian Reward Gap:")
print(f"  Low PC1: {np.median(low_gaps):.4f}")
print(f"  High PC1: {np.median(high_gaps):.4f}")

print(f"\n% where GPT4 is significantly better (Gap > 0.1):")
print(f"  Low PC1: {np.mean(low_gaps > 0.1):.1%}")
print(f"  High PC1: {np.mean(high_gaps > 0.1):.1%}")

print(f"\n--- EXAMPLES FROM HIGH PC1 (Gap < 0, Mixtral wins) ---")
high_pc1_indices = np.where(high_mask)[0]
# Sort by gap (most negative first)
sorted_indices = high_pc1_indices[np.argsort(reward_gaps[high_pc1_indices])]

for i in range(5):
    idx = sorted_indices[i]
    print(f"\nPrompt: {prompts[idx][:100]}...")
    print(f"  Gap: {reward_gaps[idx]:.4f} (Mixtral > GPT4)")
    
print(f"\n--- EXAMPLES FROM LOW PC1 (Gap > 0, GPT4 wins) ---")
low_pc1_indices = np.where(low_mask)[0]
# Sort by gap (most positive first)
sorted_indices_low = low_pc1_indices[np.argsort(reward_gaps[low_pc1_indices])[::-1]]

for i in range(5):
    idx = sorted_indices_low[i]
    print(f"\nPrompt: {prompts[idx][:100]}...")
    print(f"  Gap: {reward_gaps[idx]:.4f} (GPT4 > Mixtral)")

