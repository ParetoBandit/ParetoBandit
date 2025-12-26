"""
Find the "golden prompt" for each cluster - the prompt closest to the centroid.

This reduces evaluation cost from 5,000 prompts to 500 (one per cluster).
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    base_dir = Path(__file__).parent
    
    # Load all clustered prompts
    all_prompts_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    print(f"Loading prompts from {all_prompts_path}...")
    
    prompts_by_cluster = defaultdict(list)
    
    with open(all_prompts_path) as f:
        for line in f:
            data = json.loads(line)
            cluster_id = data["cluster_id"]
            prompts_by_cluster[cluster_id].append(data)
    
    print(f"Loaded {len(prompts_by_cluster)} clusters")
    
    # Find golden prompt for each cluster (highest similarity to centroid)
    golden_prompts = []
    
    for cluster_id in sorted(prompts_by_cluster.keys()):
        cluster_prompts = prompts_by_cluster[cluster_id]
        
        # Find prompt with highest similarity (closest to centroid)
        golden = max(cluster_prompts, key=lambda p: p["similarity"])
        
        golden_prompts.append({
            "cluster_id": cluster_id,
            "prompt": golden["prompt"],
            "similarity": golden["similarity"],
            "cluster_size": len(cluster_prompts),
            "is_golden": True
        })
    
    # Save golden prompts
    output_path = base_dir / "golden_prompts.jsonl"
    print(f"\nSaving {len(golden_prompts)} golden prompts to {output_path}...")
    
    with open(output_path, 'w') as f:
        for prompt_data in golden_prompts:
            f.write(json.dumps(prompt_data) + "\n")
    
    # Statistics
    print(f"\nGolden Prompts Summary:")
    print(f"  Total clusters: {len(golden_prompts)}")
    print(f"  Average similarity to centroid: {sum(p['similarity'] for p in golden_prompts) / len(golden_prompts):.4f}")
    print(f"  Min similarity: {min(p['similarity'] for p in golden_prompts):.4f}")
    print(f"  Max similarity: {max(p['similarity'] for p in golden_prompts):.4f}")
    
    # Show examples
    print(f"\nExample Golden Prompts:")
    for i in range(min(5, len(golden_prompts))):
        gp = golden_prompts[i]
        print(f"\nCluster {gp['cluster_id']} (size {gp['cluster_size']}, similarity {gp['similarity']:.4f}):")
        print(f"  \"{gp['prompt'][:80]}...\"" if len(gp['prompt']) > 80 else f"  \"{gp['prompt']}\"")
    
    # Comparison: Cost savings
    total_prompts = sum(len(prompts) for prompts in prompts_by_cluster.values())
    print(f"\n{'='*60}")
    print("Cost Savings Analysis:")
    print(f"{'='*60}")
    print(f"Original approach: {total_prompts:,} prompts")
    print(f"Golden prompts approach: {len(golden_prompts):,} prompts")
    print(f"Reduction factor: {total_prompts / len(golden_prompts):.1f}x")
    print(f"\nWith 50 models:")
    print(f"  Original: {total_prompts * 50:,} evaluations")
    print(f"  Golden: {len(golden_prompts) * 50:,} evaluations")
    print(f"  Savings: {(total_prompts * 50) - (len(golden_prompts) * 50):,} evaluations")

if __name__ == "__main__":
    main()
