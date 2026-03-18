import os
import matplotlib
matplotlib.use('Agg')  # Ensure we use a non-interactive backend for rendering
import matplotlib.pyplot as plt

def append_text(fig, x, y, text, space=0.008, **kwargs):
    """Draws text and returns the next horizontal starting position."""
    t = fig.text(x, y, text, **kwargs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = t.get_window_extent(renderer)
    inv = fig.transFigure.inverted()
    return inv.transform(bbox.p1)[0] + space

def create_slide():
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    
    # Title
    fig.text(0.05, 0.85, "Incrementality from ParetoBandit", 
             fontsize=44, fontweight='bold', color='#111827')
    
    # Bullet 1
    fig.text(0.05, 0.65, "●", color='#E50914', fontsize=24)
    fig.text(0.08, 0.65, "Goal: Maximize LLM response quality.", 
             fontsize=30, color='#111827', va='center')
    
    # Bullet 2
    fig.text(0.05, 0.45, "●", color='#E50914', fontsize=24)
    
    y_start = 0.46
    # Line 1: "ParetoBandit: The increase in quality"
    nx = append_text(fig, 0.08, y_start, "ParetoBandit:", fontweight='bold', fontsize=30, color='#111827', va='center')
    nx = append_text(fig, nx, y_start, "The", fontsize=30, color='#111827', va='center')
    nx = append_text(fig, nx, y_start, "increase in quality", fontweight='bold', fontsize=30, color='#E50914', va='center')
    
    # Line 2: "because the prompt was optimally"
    nx = append_text(fig, 0.08, y_start - 0.07, "because the prompt was optimally", fontweight='bold', fontsize=30, color='#E50914', va='center')
    
    # Line 3: "routed; the causal effect of the"
    nx = append_text(fig, 0.08, y_start - 0.14, "routed;", fontweight='bold', fontsize=30, color='#E50914', va='center')
    nx = append_text(fig, nx, y_start - 0.14, "the causal effect of the", fontsize=30, color='#111827', va='center')
    
    # Line 4: "contextual bandit."
    nx = append_text(fig, 0.08, y_start - 0.21, "contextual bandit.", fontsize=30, color='#111827', va='center')
    
    # Right-hand Chart
    ax = fig.add_axes([0.55, 0.25, 0.35, 0.5])
    
    # Draw bars
    ax.bar(0, 1.0, width=0.55, color='#D1D5DB', edgecolor='#9CA3AF', linewidth=2)
    ax.bar(1, 1.0, width=0.55, color='#D1D5DB', edgecolor='#9CA3AF', linewidth=2)
    ax.bar(1, 0.3, bottom=1.0, width=0.55, color='#E50914', edgecolor='#B91C1C', linewidth=2)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('#111827')
    
    # Ticks
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control\n(Static Mix)", "Treatment\n(ParetoBandit)"], fontsize=24, color='#111827')
    ax.tick_params(axis='x', length=0, pad=15)
    
    # Labels inside bars
    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3)
    ax.text(0, 0.5, "Other\nRouters", ha='center', va='center', fontsize=22, color='#111827', bbox=bbox_props)
    ax.text(1, 0.5, "Baseline\nQuality", ha='center', va='center', fontsize=22, color='#111827', bbox=bbox_props)
    
    # Dashed lines
    ax.plot([0.275, 1.5], [1.0, 1.0], color='#6B7280', linestyle='--', lw=1.5)
    ax.plot([1.275, 1.5], [1.3, 1.3], color='#6B7280', linestyle='--', lw=1.5)
    
    # Vertical Arrow
    ax.annotate("", xy=(1.45, 1.3), xytext=(1.45, 1.0),
                arrowprops=dict(arrowstyle="<|-|>", color="#4B5563", lw=2, mutation_scale=20))
    
    # Incremental Text
    ax.text(1.52, 1.15, "Incremental\nQuality", fontsize=24, va='center', color='#111827')
    
    # Y-axis label
    fig.text(0.50, 0.5, "Response Quality", rotation=90, va='center', fontsize=26, color='#111827')
    
    # Bottom label
    ax.text(0.5, -0.32, "Fixed Budget Constraint*", ha='center', va='center', fontsize=24, color='#111827')
    
    # Footnote
    fig.text(0.05, 0.05, "*Both groups enforce the exact same $/request budget. The bandit dynamically finds the Pareto-optimal allocation.",
             fontsize=14, color='#6B7280')
             
    output_path = "blog/paretobandit_netflix_style_slide.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    print(f"Saved: {create_slide()}")
