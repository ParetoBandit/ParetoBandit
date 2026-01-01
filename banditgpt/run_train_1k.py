from pathlib import Path
from rejudge_cot import CoTRewardGenerator

if __name__ == "__main__":
    base = Path(__file__).parent
    
    # Use existing partial data as cache
    cache_file = base / "data/train_rewards_pareto.jsonl"
    
    gen = CoTRewardGenerator(max_workers=64)
    
    print("=" * 60)
    print("TRAINING SET EVALUATION (1K STRATIFIED SAMPLE)")
    print("=" * 60)
    print("Strategy:")
    print("  - Reuse 346 complete prompts from existing data")
    print("  - Generate 654 new prompts (cluster-stratified)")
    print("  - Total: 1,000 prompts × 36 models = 36,000 tasks")
    print("=" * 60)
    
    gen.run(
        prompts_file=base / "data/train_prompts_sampled_1k.jsonl",
        models_file=base / "models.json",
        output_file=base / "data/train_rewards_1k.jsonl",
        cache_file=cache_file
    )
    
    print("\n" + "=" * 60)
    print("✓ Training evaluation complete!")
    print("Next steps:")
    print("  1. Run: python fix_data_leakage.py")
    print("  2. Run: python ../final_release/kdd_paper/figure_1/plot_regret.py")
    print("=" * 60)
