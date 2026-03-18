import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import os

def create_cognitive_radar_figure(save_path: str = "cognitive_radar_bandit.png"):
    """
    Generates a publication-quality schematic of Contextual Bandits applied to 
    Cognitive Radar (Adaptive Waveform Selection).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Colors suitable for a scientific paper
    color_agent = '#e6f2ff'
    color_env = '#fff2e6'
    color_radar = '#e6ffe6'
    edge_col = '#333333'
    
    # 1. Agent Box (Contextual Bandit)
    agent_box = FancyBboxPatch((4.5, 4.5), 5, 2.5, boxstyle="round,pad=0.2", 
                               fc=color_agent, ec=edge_col, lw=1.5)
    ax.add_patch(agent_box)
    ax.text(7, 6.5, "Contextual Bandit Agent\n(e.g., LinUCB, Meta-Thompson Sampling)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=edge_col)
    
    ax.text(7, 5.3, "Policy: $\pi(a_t | x_t)$", ha='center', va='center', fontsize=11, style='italic')
    ax.text(7, 4.8, "Update Model: $P(r_t | x_t, a_t)$", ha='center', va='center', fontsize=11, style='italic')

    # 2. Radar System Box (Action execution)
    radar_box = FancyBboxPatch((8.5, 1.0), 4.5, 2, boxstyle="round,pad=0.2", 
                               fc=color_radar, ec=edge_col, lw=1.5)
    ax.add_patch(radar_box)
    ax.text(10.75, 2.5, "Radar System", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=edge_col)
    ax.text(10.75, 1.8, "• Waveform Transmission\n• Signal Reception\n• Signal Processing", 
            ha='center', va='center', fontsize=11)

    # 3. Environment Box
    env_box = FancyBboxPatch((1.0, 1.0), 4.5, 2, boxstyle="round,pad=0.2", 
                             fc=color_env, ec=edge_col, lw=1.5)
    ax.add_patch(env_box)
    ax.text(3.25, 2.5, "Operational Environment", 
            ha='center', va='center', fontsize=12, fontweight='bold', color=edge_col)
    ax.text(3.25, 1.8, "• Target Dynamics\n• Clutter & Interference\n• Electronic Warfare (Jamming)", 
            ha='center', va='center', fontsize=11)

    # Arrows and Labels
    # Context x_t: Environment -> Agent
    arrow_xt = FancyArrowPatch((3.25, 3.2), (4.5, 5.75), connectionstyle="arc3,rad=0.2",
                               arrowstyle='-|>', mutation_scale=20, lw=2, color=edge_col)
    ax.add_patch(arrow_xt)
    ax.text(2.6, 4.8, "Context ($x_t$)\n(Spectrum state,\nPrior tracking est.)", 
            ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Action a_t: Agent -> Radar System
    arrow_at = FancyArrowPatch((9.5, 5.75), (10.75, 3.2), connectionstyle="arc3,rad=0.2",
                               arrowstyle='-|>', mutation_scale=20, lw=2, color=edge_col)
    ax.add_patch(arrow_at)
    ax.text(11.8, 4.8, "Action ($a_t$)\n(Bandwidth, PRF,\nPulse Width)", 
            ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Interaction: Radar -> Environment
    arrow_radar_env = FancyArrowPatch((8.3, 1.5), (5.7, 1.5), connectionstyle="arc3,rad=0.0",
                               arrowstyle='<|-', mutation_scale=20, lw=2, color=edge_col)
    ax.add_patch(arrow_radar_env)
    ax.text(7, 1.1, "Transmitted Signal", 
            ha='center', va='center', fontsize=11)

    arrow_env_radar = FancyArrowPatch((5.7, 2.3), (8.3, 2.3), connectionstyle="arc3,rad=0.0",
                               arrowstyle='-|>', mutation_scale=20, lw=2, color=edge_col)
    ax.add_patch(arrow_env_radar)
    ax.text(7, 2.7, "Radar Return (Echo)", 
            ha='center', va='center', fontsize=11)

    # Reward r_t: Radar System -> Agent
    arrow_rt = FancyArrowPatch((10.75, 3.2), (7.0, 4.5), connectionstyle="arc3,rad=-0.1",
                               arrowstyle='-|>', mutation_scale=20, lw=2, color=edge_col, ls='--')
    ax.add_patch(arrow_rt)
    ax.text(9.2, 3.8, "Reward ($r_t$)\n(Tracking Accuracy,\nSINR, Detection Prob)", 
            ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Title
    plt.title("Adaptive Waveform Selection in Cognitive Radar via Contextual Bandits", 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs("blog", exist_ok=True)
    out_path = os.path.join("blog", "cognitive_radar_bandit.png")
    create_cognitive_radar_figure(out_path)
    print(f"Successfully created figure at {out_path}")
