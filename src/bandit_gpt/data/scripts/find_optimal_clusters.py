"""
Find optimal number of clusters using multiple methods:
1. Elbow Method (Knee detection)
2. Silhouette Score
3. Calinski-Harabasz Index

This helps determine if k=500 is optimal or if we should adjust.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def find_elbow(k_values, inertias):
    """Find elbow using second derivative method."""
    # Normalize
    k_norm = np.array(k_values, dtype=float)
    k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min())
    
    inertia_norm = np.array(inertias, dtype=float)
    inertia_norm = (inertia_norm - inertia_norm.min()) / (inertia_norm.max() - inertia_norm.min())
    
    # Compute second derivative
    second_deriv = np.diff(np.diff(inertia_norm))
    
    # Find index of maximum curvature (elbow)
    # Add 1 because diff reduces array size by 1 twice
    elbow_idx = np.argmax(second_deriv) + 1
    
    return k_values[elbow_idx]

def find_optimal_clusters():
    base_dir = Path(__file__).parent
    
    # Load all unique prompts
    all_prompts_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    print(f"Loading prompts from {all_prompts_path}...")
    
    prompts = []
    with open(all_prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Embed prompts
    print("Embedding prompts...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        prompts, 
        normalize_embeddings=True, 
        show_progress_bar=True,
        batch_size=128
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test different k values
    # Test a range that makes sense for our data size
    k_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
    
    print(f"\nTesting k values: {k_values}")
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
        labels = kmeans.fit_predict(embeddings)
        
        # Metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        calinski = calinski_harabasz_score(embeddings, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        
        print(f"  Inertia: {inertia:.2f}")
        print(f"  Silhouette: {silhouette:.4f}")
        print(f"  Calinski-Harabasz: {calinski:.2f}")
    
    # Find knee/elbow
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # 1. Elbow Method with Knee Detection
    knee_k = find_elbow(k_values, inertias)
    
    print(f"\n1. Elbow Method (Second Derivative):")
    print(f"   Optimal k = {knee_k}")
    
    # 2. Silhouette Score (higher is better)
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_silhouette_k = k_values[best_silhouette_idx]
    
    print(f"\n2. Silhouette Score (higher is better):")
    print(f"   Optimal k = {best_silhouette_k} (score: {silhouette_scores[best_silhouette_idx]:.4f})")
    
    # 3. Calinski-Harabasz Index (higher is better)
    best_calinski_idx = np.argmax(calinski_scores)
    best_calinski_k = k_values[best_calinski_idx]
    
    print(f"\n3. Calinski-Harabasz Index (higher is better):")
    print(f"   Optimal k = {best_calinski_k} (score: {calinski_scores[best_calinski_idx]:.2f})")
    
    # Recommendation
    print(f"\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    # Current k=500
    current_k = 500
    current_idx = k_values.index(current_k) if current_k in k_values else None
    
    if current_idx is not None:
        print(f"\nCurrent k=500:")
        print(f"  Silhouette: {silhouette_scores[current_idx]:.4f}")
        print(f"  Calinski-Harabasz: {calinski_scores[current_idx]:.2f}")
    
    # Weighted recommendation
    recommendations = [knee_k, best_silhouette_k, best_calinski_k]
    
    avg_recommendation = int(np.median(recommendations))
    
    print(f"\nRecommendations from each method: {recommendations}")
    print(f"Median recommendation: k={avg_recommendation}")
    
    if avg_recommendation < 400:
        print(f"\n⚠️  Consider using k={avg_recommendation} instead of k=500")
        print(f"   This would reduce API costs by {500/avg_recommendation:.1f}x")
    elif avg_recommendation > 600:
        print(f"\n⚠️  Consider using k={avg_recommendation} instead of k=500")
        print(f"   This would increase granularity at {avg_recommendation/500:.1f}x cost")
    else:
        print(f"\n✅ k=500 is near optimal!")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Elbow
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=knee_k, color='r', linestyle='--', linewidth=2, label=f'Elbow at k={knee_k}')
    axes[0].axvline(x=500, color='g', linestyle=':', linewidth=2, label='Current k=500')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Silhouette
    axes[1].plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=best_silhouette_k, color='r', linestyle='--', linewidth=2, label=f'Max at k={best_silhouette_k}')
    axes[1].axvline(x=500, color='b', linestyle=':', linewidth=2, label='Current k=500')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Calinski-Harabasz
    axes[2].plot(k_values, calinski_scores, 'mo-', linewidth=2, markersize=8)
    axes[2].axvline(x=best_calinski_k, color='r', linestyle='--', linewidth=2, label=f'Max at k={best_calinski_k}')
    axes[2].axvline(x=500, color='b', linestyle=':', linewidth=2, label='Current k=500')
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[2].set_ylabel('Calinski-Harabasz Index', fontsize=12)
    axes[2].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = base_dir / "optimal_clusters_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save results to file
    results_path = base_dir / "optimal_clusters_results.json"
    results = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "calinski_scores": calinski_scores,
        "recommendations": {
            "elbow_method": knee_k,
            "silhouette_optimal": best_silhouette_k,
            "calinski_optimal": best_calinski_k,
            "median_recommendation": avg_recommendation
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    find_optimal_clusters()
