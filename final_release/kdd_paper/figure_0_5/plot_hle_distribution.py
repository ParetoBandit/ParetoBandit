import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_hle_distribution():
    root_dir = Path(__file__).parent.parent.parent
    models_path = root_dir / "models.json"
    
    with open(models_path, "r") as f:
        data = json.load(f)
    
    hle_scores = [m.get("hle") for m in data["models"] if m.get("hle") is not None]
    
    if not hle_scores:
        print("No HLE scores found in models.json")
        return

    plt.figure(figsize=(10, 6))
    
    # Use a nice color and style
    n, bins, patches = plt.hist(hle_scores, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    
    plt.title("Distribution of HLE Scores across Model Registry", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("HLE Score (Humanity's Last Exam)", fontsize=12)
    plt.ylabel("Number of Models", fontsize=12)
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add vertical lines for percentiles
    median_hle = np.percentile(hle_scores, 50)
    p95_hle = np.percentile(hle_scores, 95)
    
    plt.axvline(median_hle, color='#e74c3c', linestyle='dashed', linewidth=2, label=f'Median (P50): {median_hle:.3f}')
    plt.axvline(p95_hle, color='#2ecc71', linestyle='dotted', linewidth=2, label=f'P95: {p95_hle:.3f}')
    plt.legend()

    plt.tight_layout()
    
    output_path = Path(__file__).parent / "figure0_5_hle_dist.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved HLE distribution plot to {output_path}")

if __name__ == "__main__":
    plot_hle_distribution()
