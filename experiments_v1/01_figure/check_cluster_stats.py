
import sys
from pathlib import Path
import json
import gzip
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from scipy.stats import mannwhitneyu, ttest_ind
from scipy import stats

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

# Statistical Significance Tests
print(f"\n--- STATISTICAL SIGNIFICANCE TESTS ---")

# Test 1: Mann-Whitney U test (non-parametric, robust to outliers)
statistic_mw, p_value_mw = mannwhitneyu(low_gaps, high_gaps, alternative='two-sided')
print(f"\nMann-Whitney U Test:")
print(f"  Statistic: {statistic_mw:.2f}")
print(f"  P-value: {p_value_mw:.2e}")
print(f"  Significant at α=0.05: {'YES' if p_value_mw < 0.05 else 'NO'}")
print(f"  Significant at α=0.001: {'YES' if p_value_mw < 0.001 else 'NO'}")

# Test 2: Independent t-test (parametric, assumes normality)
statistic_t, p_value_t = ttest_ind(low_gaps, high_gaps, equal_var=False)
print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {statistic_t:.3f}")
print(f"  P-value: {p_value_t:.2e}")
print(f"  Significant at α=0.05: {'YES' if p_value_t < 0.05 else 'NO'}")
print(f"  Significant at α=0.001: {'YES' if p_value_t < 0.001 else 'NO'}")

# Effect Size: Cohen's d
pooled_std = np.sqrt(((len(low_gaps) - 1) * np.var(low_gaps, ddof=1) + 
                       (len(high_gaps) - 1) * np.var(high_gaps, ddof=1)) / 
                      (len(low_gaps) + len(high_gaps) - 2))
cohens_d = (np.mean(low_gaps) - np.mean(high_gaps)) / pooled_std
print(f"\nEffect Size (Cohen's d):")
print(f"  d = {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Interpretation: {effect_interpretation} effect")

# Confidence Intervals (95%)
from scipy.stats import t as t_dist
ci_level = 0.95
alpha = 1 - ci_level

# Low PC1 CI
low_mean = np.mean(low_gaps)
low_se = stats.sem(low_gaps)
low_ci = t_dist.interval(ci_level, len(low_gaps)-1, loc=low_mean, scale=low_se)

# High PC1 CI
high_mean = np.mean(high_gaps)
high_se = stats.sem(high_gaps)
high_ci = t_dist.interval(ci_level, len(high_gaps)-1, loc=high_mean, scale=high_se)

print(f"\n95% Confidence Intervals:")
print(f"  Low PC1:  [{low_ci[0]:+.4f}, {low_ci[1]:+.4f}]")
print(f"  High PC1: [{high_ci[0]:+.4f}, {high_ci[1]:+.4f}]")
print(f"  CIs overlap: {'YES (weak evidence)' if low_ci[0] <= high_ci[1] and high_ci[0] <= low_ci[1] else 'NO (strong evidence)'}")

# Distribution checks
print(f"\n--- DISTRIBUTION STATISTICS ---")
print(f"Low PC1 Cluster:")
print(f"  Skewness: {stats.skew(low_gaps):.3f}")
print(f"  Kurtosis: {stats.kurtosis(low_gaps):.3f}")
print(f"  Std Dev: {np.std(low_gaps, ddof=1):.4f}")

print(f"\nHigh PC1 Cluster:")
print(f"  Skewness: {stats.skew(high_gaps):.3f}")
print(f"  Kurtosis: {stats.kurtosis(high_gaps):.3f}")
print(f"  Std Dev: {np.std(high_gaps, ddof=1):.4f}")

# Normality tests
_, p_shapiro_low = stats.shapiro(low_gaps[:min(5000, len(low_gaps))])  # Shapiro limited to 5000
_, p_shapiro_high = stats.shapiro(high_gaps[:min(5000, len(high_gaps))])
print(f"\nShapiro-Wilk Normality Test:")
print(f"  Low PC1 p-value: {p_shapiro_low:.4f} ({'Normal' if p_shapiro_low > 0.05 else 'Non-normal'})")
print(f"  High PC1 p-value: {p_shapiro_high:.4f} ({'Normal' if p_shapiro_high > 0.05 else 'Non-normal'})")
print(f"  Note: Mann-Whitney U is preferred for non-normal distributions")

print(f"\n--- DIVERSE EXAMPLES FROM HIGH PC1 (Alignment Tax Zone) ---")
print(f"Note: Using stratified sampling to show variety (not cherry-picked)")

high_pc1_indices = np.where(high_mask)[0]
X_high = X_2d[high_pc1_indices]

# Get diverse examples using farthest-first traversal
def get_diverse_indices(embeddings, n_samples):
    """Select diverse samples using farthest-first traversal."""
    if len(embeddings) <= n_samples:
        return list(range(len(embeddings)))
    
    # Start with centroid
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    selected = [np.argmin(distances)]
    
    for _ in range(n_samples - 1):
        # Find farthest from all selected
        min_dists = []
        for i in range(len(embeddings)):
            if i in selected:
                min_dists.append(-1)
            else:
                dists = [np.linalg.norm(embeddings[i] - embeddings[j]) for j in selected]
                min_dists.append(min(dists))
        next_idx = np.argmax(min_dists)
        if min_dists[next_idx] > 0:
            selected.append(next_idx)
    
    return selected

diverse_indices_relative = get_diverse_indices(X_high, 5)
diverse_indices = [high_pc1_indices[i] for i in diverse_indices_relative]

for i, idx in enumerate(diverse_indices, 1):
    print(f"\n{i}. Gap: {reward_gaps[idx]:+.4f} ({'Mixtral wins' if reward_gaps[idx] < 0 else 'GPT4 wins'})")
    print(f"   Prompt: {prompts[idx][:150]}...")
    
print(f"\n--- DIVERSE EXAMPLES FROM LOW PC1 (Natural Language Zone) ---")
print(f"Note: Using stratified sampling to show variety (not cherry-picked)")

low_pc1_indices = np.where(low_mask)[0]
X_low = X_2d[low_pc1_indices]

diverse_indices_relative_low = get_diverse_indices(X_low, 5)
diverse_indices_low = [low_pc1_indices[i] for i in diverse_indices_relative_low]

for i, idx in enumerate(diverse_indices_low, 1):
    print(f"\n{i}. Gap: {reward_gaps[idx]:+.4f} ({'Mixtral wins' if reward_gaps[idx] < 0 else 'GPT4 wins'})")
    print(f"   Prompt: {prompts[idx][:150]}...")

