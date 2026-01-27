"""
Check ground truth rewards: Is GPT-5.1 actually better?
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import DEV_DATA_PATH_ALL_MODELS

MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo", "openai/gpt-5.1"]

print("="*60)
print("GROUND TRUTH REWARD ANALYSIS")
print("="*60)

# Load data
evaluator = AlignedEvaluator.from_jsonl_gz(
    DEV_DATA_PATH_ALL_MODELS,
    required_models=MODELS
)
data = [item for item in evaluator if all(m in item.rewards for m in MODELS)]

print(f"\n✅ Loaded {len(data)} samples with all 3 models\n")

# Calculate average ground truth rewards
rewards_by_model = {m: [] for m in MODELS}

for item in data:
    for model in MODELS:
        r = item.get_reward(model, default=None)
        if r is not None:
            rewards_by_model[model].append(r)

print("📊 Average Ground Truth Rewards:")
print("-" * 60)
for model in MODELS:
    avg = np.mean(rewards_by_model[model])
    std = np.std(rewards_by_model[model])
    print(f"{model:40s}: {avg:.3f} ± {std:.3f}")

print("\n" + "="*60)
print("KEY QUESTION: Is GPT-5.1 actually better?")
print("="*60)

gpt5_avg = np.mean(rewards_by_model["openai/gpt-5.1"])
gpt4_avg = np.mean(rewards_by_model["openai/gpt-4-turbo"])
mixtral_avg = np.mean(rewards_by_model["mistralai/mixtral-8x7b-instruct"])

print(f"\nGPT-5.1 vs GPT-4-turbo: {gpt5_avg - gpt4_avg:+.3f}")
print(f"GPT-5.1 vs Mixtral:     {gpt5_avg - mixtral_avg:+.3f}")
print(f"GPT-4-turbo vs Mixtral: {gpt4_avg - mixtral_avg:+.3f}")

if gpt5_avg > gpt4_avg:
    print(f"\n✅ GPT-5.1 IS better (+{gpt5_avg - gpt4_avg:.3f})")
    print("   → Semantic transfer SHOULD help")
else:
    print(f"\n❌ GPT-5.1 is WORSE ({gpt5_avg - gpt4_avg:.3f})")
    print("   → This explains why transfer doesn't help!")

print("\n" + "="*60)
print("OPTIMAL STRATEGY")
print("="*60)

best_model = max(MODELS, key=lambda m: np.mean(rewards_by_model[m]))
print(f"\nBest model overall: {best_model}")
print(f"Average reward: {np.mean(rewards_by_model[best_model]):.3f}")

# Check per-sample best
best_per_sample = []
for item in data:
    rewards = {m: item.get_reward(m, default=0) for m in MODELS}
    best = max(rewards, key=rewards.get)
    best_per_sample.append(best)

from collections import Counter
counts = Counter(best_per_sample)
print(f"\nHow often each model is best:")
for model, count in counts.most_common():
    pct = 100 * count / len(best_per_sample)
    print(f"  {model:40s}: {count:4d} ({pct:5.1f}%)")

print("\n" + "="*60)

