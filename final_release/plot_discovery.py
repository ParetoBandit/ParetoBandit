import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_discovery():
    root_dir = Path(__file__).parent
    
    # 1. Load Results
    with open(root_dir / "discovery_results.json") as f:
        data = json.load(f)
    
    models = [data["teacher_pet"]["name"], data["target_model"]["name"]]
    priors = [data["teacher_pet"]["prior"], data["target_model"]["prior"]]
    posteriors = [data["teacher_pet"]["posterior"], data["target_model"]["posterior"]]
    
    # 2. Plot
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, priors, width, label='Prior Probability (HLE)', color='#bdc3c7', alpha=0.8)
    rects2 = ax.bar(x + width/2, posteriors, width, label='Posterior Probability (Learned)', color='#3498db', alpha=0.9)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Specialist Probability (Relative Confidence)', fontsize=12)
    ax.set_title('Figure 4: Specialist Discovery (TimescaleDB IoT)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)  # Set limit to 1.1 for label space
    ax.legend(fontsize=10)
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    # Add an arrow and text to highlight the discovery
    target_idx = 1
    ax.annotate('Specialist Discovery!', 
                xy=(target_idx + width/2 + 0.05, posteriors[target_idx] - 0.02), 
                xytext=(target_idx + 0.6, posteriors[target_idx] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=12, fontweight='bold', color='#e74c3c')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    output_path = root_dir / "figure4_discovery.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_discovery()
