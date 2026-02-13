"""
Test: Can We Get Statistical Power with More Samples?
======================================================

Let's test the hypothesis: If we run 10,000 steps (10x longer) with realistic
LMSYS distributions, does Corralling successfully decommission?

Theory: We need ~1,174 samples per expert. With 10,000 steps:
- Each expert gets ~5,000 samples
- Each expert tests each model ~2,500 times
- This is 2.1x the needed samples → should work!
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

# Reuse classes from realistic experiment
class StubbornExpert:
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, cost=0.0):
        pass

class SmartExpert:
    def __init__(self, name: str, best_model: str):
        self.name = name
        self.best_model = best_model
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        if np.random.random() < 0.05:
            return "openai/gpt-4-turbo"
        return self.best_model
    
    def update(self, context, model, reward, cost=0.0):
        pass

class RealisticEnvironment:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.phase1_params = {
            "mistralai/mixtral-8x7b-instruct": (0.82, 0.09),
            "openai/gpt-4-turbo": (0.81, 0.10),
        }
        self.phase2_params = {
            "mistralai/mixtral-8x7b-instruct": (0.823, 0.09),
            "openai/gpt-4-turbo": (0.812, 0.10),
        }
        self.shift_step = 100
        self.t = 0
    
    def get_reward(self, model: str) -> float:
        self.t += 1
        if self.t < self.shift_step:
            params = self.phase1_params
        else:
            params = self.phase2_params
        mean, std = params.get(model, (0.5, 0.1))
        reward = self.rng.normal(mean, std)
        return np.clip(reward, 0.0, 1.0)

print("="*70)
print("HYPOTHESIS TEST: More Samples → Statistical Power?")
print("="*70)
print("\nRunning realistic scenario with 10,000 steps (10x longer)...")
print("Theory: Need ~1,174 samples per expert, will get ~5,000")
print("Expected: 100% success rate (vs 25% with 1,000 steps)\n")

# Run multiple seeds
n_seeds = 10
n_steps = 10000
results = []

for seed in range(n_seeds):
    np.random.seed(seed)
    
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    warmup = StubbornExpert("Warmup", "openai/gpt-4-turbo")
    tabula = SmartExpert("Tabula Rasa", "mistralai/mixtral-8x7b-instruct")
    router = CorrallingRouter(experts=[warmup, tabula], models=models,
                              learning_rate=0.1, gamma=0.05)
    env = RealisticEnvironment(seed=seed)
    
    weights_history = []
    
    for t in range(n_steps):
        context = np.random.randn(10)
        selected_model = router.select_model(context)
        reward = env.get_reward(selected_model)
        router.update(context, selected_model, reward)
        weights_history.append(router.weights.copy())
    
    weights = np.array(weights_history)
    shift_step = 100
    decom_idx = np.where((np.arange(len(weights)) >= shift_step) & (weights[:, 0] < 0.1))[0]
    
    decommissioned = len(decom_idx) > 0
    decom_step = decom_idx[0] if decommissioned else None
    reaction_time = (decom_idx[0] - shift_step) if decommissioned else None
    
    results.append({
        "seed": seed,
        "decommissioned": decommissioned,
        "decom_step": decom_step,
        "reaction_time": reaction_time,
        "final_warmup": weights[-1, 0],
        "final_tr": weights[-1, 1],
    })
    
    print(f"  Seed {seed}: {'✅ Decommissioned' if decommissioned else '❌ Failed'} "
          f"(t={decom_step if decom_step else 'N/A'}, "
          f"final: {weights[-1, 0]:.3f} vs {weights[-1, 1]:.3f})")

# Summary
n_success = sum(r["decommissioned"] for r in results)
success_rate = n_success / n_seeds

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n📊 Success Rate: {n_success}/{n_seeds} ({success_rate*100:.0f}%)")

if n_success > 0:
    successful = [r for r in results if r["decommissioned"]]
    decom_times = [r["decom_step"] for r in successful]
    reaction_times = [r["reaction_time"] for r in successful]
    
    print(f"\n📊 Decommissioning Time (successful trials):")
    print(f"   Mean: {np.mean(decom_times):.1f} ± {np.std(decom_times):.1f} steps")
    print(f"   Range: [{np.min(decom_times):.0f}, {np.max(decom_times):.0f}]")
    
    print(f"\n📊 Reaction Time:")
    print(f"   Mean: {np.mean(reaction_times):.1f} ± {np.std(reaction_times):.1f} steps")

final_warmup = [r["final_warmup"] for r in results]
final_tr = [r["final_tr"] for r in results]
print(f"\n📊 Final Weights (all trials):")
print(f"   Warmup: {np.mean(final_warmup):.3f} ± {np.std(final_warmup):.3f}")
print(f"   Tabula Rasa: {np.mean(final_tr):.3f} ± {np.std(final_tr):.3f}")

print(f"\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"\n   1,000 steps:  25% success (5/20 trials)")
print(f"  10,000 steps:  {success_rate*100:.0f}% success ({n_success}/{n_seeds} trials)")
print(f"\n⚡ {success_rate/0.25:.1f}x improvement with 10x more samples!")

if success_rate > 0.8:
    print(f"\n✅ HYPOTHESIS CONFIRMED: More samples → statistical power!")
    print(f"   With 10,000 steps, we have sufficient power to detect d=0.12")
else:
    print(f"\n⚠️  Still not enough! Even 10,000 steps only achieves {success_rate*100:.0f}% success")
    print(f"   May need 20,000+ steps or different hyperparameters")

print(f"\n💡 Key Insight: In simulations, we CAN get more samples.")
print(f"   The constraint is PRODUCTION, not experiments!")
