
from pathlib import Path
from pca_manager_copy import train_pca_pipeline

def main():
    print("🚀 Training 8-dim PCA...")
    source = Path("/Users/annette/repostitories/banditGPT/src/bandit_gpt/data/offline_dataset/lmsys_train_final_rewards_1k_clean.jsonl.gz")
    output = Path("artifacts")
    
    # We don't need to exclude anything since we're using the training set itself 
    # to learn the manifold, and we aren't doing rigorous "unseen" PCA evaluation here.
    # The user just wants a working PCA-8 artifact.
    exclusions = [] 
    
    train_pca_pipeline(
        source_prompts_path=source,
        exclusion_paths=exclusions,
        output_dir=output,
        n_components=8,
        max_prompts=1000  # Use all available in the file
    )

if __name__ == "__main__":
    main()
