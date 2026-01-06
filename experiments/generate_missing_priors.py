import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.data.scripts.pca_manager import train_pca_pipeline

def main():
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / "src" / "bandit_gpt" / "data"
    offline_dir = data_dir / "offline_dataset"
    
    source_prompts = data_dir / "lmsys_all_prompts_clustered.jsonl"
    train_rewards = offline_dir / "train_rewards_hle_models.jsonl"
    test_rewards = offline_dir / "test_rewards_hle_models.jsonl"
    
    print("Generating priors for offline dataset...")
    train_pca_pipeline(
        source_prompts_path=source_prompts,
        exclusion_paths=[train_rewards, test_rewards],
        output_dir=offline_dir,
        n_components=32,
        max_prompts=25000
    )

if __name__ == "__main__":
    main()
