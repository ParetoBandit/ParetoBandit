#!/usr/bin/env python3
"""
Resolution Gap Analysis: Visualizing HLE "Blur" vs CSR "Precision"

This experiment proves WHY HLE priors fail by showing that:
- HLE: Points to the average (flat across clusters) - "The Blur"
- CSR: Points to specific clusters (spikes on specialist areas) - "The Precision"

Method:
- Load HLE and CSR prior belief vectors (b) for specialist models
- Load cluster centroids from golden prompts
- Compute cosine similarity between b-vector and each cluster centroid
- Visualize the difference in resolution
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from banditgpt.bandit import BanditRouter

# Cluster mapping for interpretation (semantic labels for readability)
CLUSTER_NAMES = {
    # Core anchor clusters
    0: "Math & Logic",
    1: "Coding",
    2: "Creative Writing", 
    3: "Jokes & Humor",
    4: "Explain & Reason",
    # High-value specialist clusters (discovered from data)
    5: "Algorithm Design",
    6: "Data Structures",
    11: "Python Coding",
    17: "Code Review",
    21: "Software Arch",
    24: "API Design",
    31: "Backend Dev",
    32: "System Design",
    42: "Web Dev",
    55: "Database Query",
    56: "ML/AI Code",
    57: "DevOps",
    60: "Frontend JS",
    61: "React/UI",
    63: "Testing/QA",
    65: "Security Code",
    68: "Performance Opt",
    87: "Code Refactor",
    88: "Debug/Fix",
    91: "Cloud/Infra",
    92: "Microservices",
    95: "Distributed Sys",
}

def load_cluster_centroids(pca_model):
    """Load and compute cluster centroids in PCA space"""
    golden_path = Path(__file__).parent.parent.parent / "priors" / "golden_prompts.jsonl"
    
    # Load golden prompts
    prompts_by_cluster = {}
    with open(golden_path) as f:
        for line in f:
            data = json.loads(line)
            cluster_id = data['cluster_id']
            if cluster_id not in prompts_by_cluster:
                prompts_by_cluster[cluster_id] = []
            prompts_by_cluster[cluster_id].append(data['prompt'])
    
    # Encode with SBERT
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    centroids = {}
    for cluster_id, prompts in prompts_by_cluster.items():
        if cluster_id >= 100:  # Only use anchor clusters
            continue
        # Encode prompts
        embeddings = sbert.encode(prompts, show_progress_bar=False)
        # Apply PCA
        pca_embeddings = pca_model.transform(embeddings)
        # Compute centroid
        centroid = np.mean(pca_embeddings, axis=0)
        centroids[cluster_id] = centroid
    
    return centroids

def get_prior_beliefs(router, model_id):
    """Extract prior belief vector (b) for a specific model"""
    # The b vector in the bandit represents prior beliefs
    # It's in the full feature space (PCA + explicit + cluster + bias)
    b_vector = router.bandit.b[model_id]
    
    # Extract just the PCA components (first 32 dimensions)
    b_pca = b_vector[:32]
    
    return b_pca

def compute_cluster_alignment(b_vector, centroids):
    """Compute cosine similarity between b-vector and cluster centroids"""
    similarities = {}
    
    # Normalize b_vector
    b_norm = b_vector / (np.linalg.norm(b_vector) + 1e-10)
    
    for cluster_id, centroid in centroids.items():
        # Normalize centroid
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        # Cosine similarity
        similarity = float(np.dot(b_norm, c_norm))
        similarities[cluster_id] = similarity
    
    return similarities

def visualize_resolution_gap(specialist_model, hle_sims, csr_sims, output_path):
    """Create a high-quality, professional KDD-style visualization of the Resolution Gap."""
    
    # Prepare data
    hle_values = np.array(list(hle_sims.values()))
    csr_values = np.array(list(csr_sims.values()))
    
    # Get top CSR clusters for highlighting (the specialization spikes)
    num_top = 8
    top_csr_items = sorted(csr_sims.items(), key=lambda x: x[1], reverse=True)[:num_top]
    top_ids = [item[0] for item in top_csr_items]
    top_labels = [CLUSTER_NAMES.get(cid, f"Cluster {cid}") for cid in top_ids]
    top_csr_vals = [csr_sims[cid] for cid in top_ids]
    top_hle_vals = [hle_sims[cid] for cid in top_ids]

    # Calculate statistics
    hle_std = np.std(hle_values)
    csr_std = np.std(csr_values)
    
    # Setup professional styles
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    colors = {'HLE': '#D62728', 'CSR': '#1F77B4'} # Professional Red and Blue
    
    # Create figure with 2 panels side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})
    
    # ============ PANEL A: Information Density (Violin Plot) ============
    # Proves the "Blur" hypothesis via spread differences
    parts = ax1.violinplot([hle_values, csr_values], showmeans=False, showmedians=True, widths=0.7)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(colors.values())[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        parts[partname].set_edgecolor('black')
        parts[partname].set_linewidth(1)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['HLE\n(Benchmarks)', 'CSR\n(Task-Specific)'], fontweight='bold')
    ax1.set_ylabel('Cosine Similarity with Task Centroids', fontsize=12)
    ax1.set_title(fr'A) Information Density ($\sigma_{{CSR}} \approx {csr_std/hle_std:.1f}\times \sigma_{{HLE}}$)', 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Add clear Sigma annotations (fixed position, no overlap)
    ax1.text(1, np.max(hle_values) + 0.05, fr'$\sigma={hle_std:.3f}$', ha='center', fontweight='bold', color=colors['HLE'])
    ax1.text(2, np.max(csr_values) + 0.05, fr'$\sigma={csr_std:.3f}$', ha='center', fontweight='bold', color=colors['CSR'])

    # ============ PANEL B: Specialist Misinformation (Bar Chart) ============
    # Shows how HLE actively discourages experts
    x = np.arange(len(top_labels))
    width = 0.35
    
    ax2.bar(x - width/2, top_hle_vals, width, label='HLE (Blurry)', color=colors['HLE'], alpha=0.8, edgecolor='black')
    ax2.bar(x + width/2, top_csr_vals, width, label='CSR (Precise)', color=colors['CSR'], alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('B) Misinformation Gap on Specialist Clusters', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_labels, rotation=40, ha='right', fontsize=10)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.axhline(0, color='black', linewidth=0.8)
    
    # Set shared y-limits for comparability
    all_vals = np.concatenate([hle_values, csr_values])
    ymin = min(np.min(all_vals), -0.3) - 0.1
    ymax = max(np.max(all_vals), 0.75) + 0.1
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # Highlight "Misinformation" (Negative HLE on high CSR)
    for i, (h, c) in enumerate(zip(top_hle_vals, top_csr_vals)):
        if h < 0:
            # Add a small highlight circle or arrow to indicate the harm
            ax2.annotate('!', xy=(i - width/2, h - 0.05), xytext=(0, -15), 
                         textcoords='offset points', ha='center', color=colors['HLE'], 
                         arrowprops=dict(arrowstyle="->", color=colors['HLE']), fontsize=12, fontweight='bold')

    # Overall Figure Decoration
    model_name = specialist_model.split("/")[-1].replace("-", " ").title()
    plt.suptitle(f'Resolution Gap Analysis: {model_name}\nHLE "Blur" vs CSR "Precision" Alignment Breakdown', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94]) # Leave space for suptitle
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"✓ Professional KDD plot saved to: {output_path}")

def main():
    """Main execution"""
    print("=" * 70)
    print("RESOLUTION GAP ANALYSIS: HLE Blur vs CSR Precision")
    print("=" * 70)
    
    # Choose specialist model to analyze
    specialist_model = "deepseek/deepseek-chat-v3-0324"  # DeepSeek coding specialist
    
    print(f"\n[1/5] Analyzing specialist model: {specialist_model}")
    
    # Load models registry
    print("\n[2/5] Loading model registry...")
    models_path = Path(__file__).parent.parent.parent / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    if specialist_model not in registry:
        print(f"ERROR: Model {specialist_model} not in registry")
        print(f"Available models: {list(registry.keys())[:5]}...")
        return
    
    # Create routers with HLE and CSR priors
    print("\n[3/5] Creating routers...")
    
    print("  Loading HLE router...")
    hle_router = BanditRouter.create(
        model_registry=registry,
        priors="hle",
        context_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("  Loading CSR router...")
    csr_router = BanditRouter.create(
        model_registry=registry,
        priors="csr",
        context_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load cluster centroids
    print("\n[4/5] Computing cluster centroids in PCA space...")
    centroids = load_cluster_centroids(csr_router.pca)
    print(f"  Loaded {len(centroids)} cluster centroids")
    
    # Get prior belief vectors
    print("\n[5/5] Analyzing prior beliefs...")
    
    hle_b = get_prior_beliefs(hle_router, specialist_model)
    csr_b = get_prior_beliefs(csr_router, specialist_model)
    
    print(f"  HLE b-vector (PCA): shape={hle_b.shape}, norm={np.linalg.norm(hle_b):.2f}")
    print(f"  CSR b-vector (PCA): shape={csr_b.shape}, norm={np.linalg.norm(csr_b):.2f}")
    
    # Compute alignments
    hle_similarities = compute_cluster_alignment(hle_b, centroids)
    csr_similarities = compute_cluster_alignment(csr_b, centroids)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Cluster Alignment (Cosine Similarity)")
    print("=" * 70)
    
    print(f"\n{'Cluster':<20} {'HLE':>10} {'CSR':>10} {'Δ':>10}")
    print("-" * 55)
    
    for cluster_id in sorted(centroids.keys()):
        name = CLUSTER_NAMES.get(cluster_id, f"C{cluster_id}")
        hle_sim = hle_similarities[cluster_id]
        csr_sim = csr_similarities[cluster_id]
        delta = csr_sim - hle_sim
        
        print(f"{name:<20} {hle_sim:>10.4f} {csr_sim:>10.4f} {delta:>+10.4f}")
    
    # Summary statistics
    print("\n" + "-" * 55)
    hle_std = np.std(list(hle_similarities.values()))
    csr_std = np.std(list(csr_similarities.values()))
    
    print(f"{'Std Dev (σ)':<20} {hle_std:>10.4f} {csr_std:>10.4f}")
    print(f"{'Resolution':<20} {'Blurry':>10} {'Precise':>10}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if csr_std > hle_std * 1.5:
        print(f"\n✓ CSR shows {csr_std/hle_std:.1f}x higher variance than HLE")
        print("  → CSR is PRECISE: Strong alignment with specific clusters")
        print("  → HLE is BLURRY: Flat distribution across all clusters")
    else:
        print(f"\n⚠ CSR and HLE show similar variance (ratio: {csr_std/hle_std:.2f})")
        print("  → Resolution gap may be smaller than expected")
    
    # Find CSR's peak cluster
    max_cluster = max(csr_similarities.items(), key=lambda x: x[1])
    print(f"\nCSR peak alignment: {CLUSTER_NAMES.get(max_cluster[0])} ({max_cluster[1]:.4f})")
    
    # Visualize
    output_path = Path(__file__).parent / "resolution_gap_analysis.png"
    visualize_resolution_gap(specialist_model, hle_similarities, csr_similarities, output_path)
    
    print("\n✅ RESOLUTION GAP ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()
