import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    
    # ---------------------------------------------------------
    # LEFT COLUMN: Netflix-style Typography
    # ---------------------------------------------------------
    
    # Title
    fig.text(0.05, 0.85, "Routing on the Pareto Frontier", 
             fontsize=44, fontweight='bold', color='#111827')
    
    # Bullet 1
    fig.text(0.05, 0.65, "●", color='#E50914', fontsize=24)
    fig.text(0.08, 0.65, "Goal: Maximize quality under a strict budget.", 
             fontsize=30, color='#111827', va='center')
    
    # Bullet 2
    fig.text(0.05, 0.45, "●", color='#E50914', fontsize=24)
    
    y_start = 0.46
    nx = append_text(fig, 0.08, y_start, "ParetoBandit:", fontweight='bold', fontsize=30, color='#111827', va='center')
    nx = append_text(fig, nx, y_start, "Dynamically routes", fontsize=30, color='#111827', va='center')
    
    nx = append_text(fig, 0.08, y_start - 0.07, "each prompt to the optimal model,", fontsize=30, color='#111827', va='center')
    
    nx = append_text(fig, 0.08, y_start - 0.14, "pushing performance to the", fontsize=30, color='#111827', va='center')
    nx = append_text(fig, nx, y_start - 0.14, "absolute limit", fontweight='bold', fontsize=30, color='#E50914', va='center')
    
    nx = append_text(fig, 0.08, y_start - 0.21, "of your", fontsize=30, color='#111827', va='center')
    nx = append_text(fig, nx, y_start - 0.21, "$/request constraint.", fontweight='bold', fontsize=30, color='#E50914', va='center')
    
    # ---------------------------------------------------------
    # RIGHT COLUMN: Stylized Pareto Graphic
    # ---------------------------------------------------------
    ax = fig.add_axes([0.55, 0.20, 0.4, 0.55])
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#111827')
    ax.spines['bottom'].set_color('#111827')
    
    # Define the curve (diminishing returns)
    x = np.linspace(0.1, 1.0, 100)
    y_optimal = 1.0 - np.exp(-5 * x)
    
    # Draw the optimal frontier
    ax.plot(x, y_optimal, color='#111827', lw=3, label="Optimal Frontier")
    
    # Shade the "impossible" region above the curve
    ax.fill_between(x, y_optimal, 1.1, color='#F3F4F6', alpha=0.8, zorder=0)
    ax.text(0.2, 0.95, "Mathematically\nImpossible Region", color='#9CA3AF', fontsize=16, ha='left', va='center')
    
    # Budget constraint line
    budget_x = 0.55
    budget_y = 1.0 - np.exp(-5 * budget_x)
    ax.axvline(x=budget_x, color='#E50914', linestyle='--', lw=2.5, zorder=1)
    
    # Points
    subopt_y = budget_y - 0.25
    ax.plot(budget_x, subopt_y, 'o', color='#9CA3AF', markersize=14, zorder=3)
    ax.plot(budget_x, budget_y, 'o', color='#E50914', markersize=16, zorder=3)
    
    # Arrow showing the lift
    ax.annotate("", 
                xy=(budget_x, budget_y - 0.03), 
                xytext=(budget_x, subopt_y + 0.03),
                arrowprops=dict(arrowstyle="-|>", color="#E50914", lw=3, mutation_scale=20))
    
    # Labels for the points
    bbox_props = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3)
    ax.text(budget_x + 0.03, subopt_y, "Static\nRouting", 
            fontsize=18, color='#6B7280', va='center', bbox=bbox_props)
    
    ax.text(budget_x - 0.03, budget_y, "ParetoBandit", 
            fontsize=22, fontweight='bold', color='#E50914', ha='right', va='center', bbox=bbox_props)
            
    # Constraint Label
    ax.text(budget_x, 0.05, "Target Budget\nConstraint", 
            fontsize=16, fontweight='bold', color='#E50914', ha='center', va='bottom', 
            bbox=dict(facecolor='white', edgecolor='none'))
            
    # Incremental Quality Text inside the arrow gap
    ax.text(budget_x - 0.03, (subopt_y + budget_y)/2, "Incremental\nQuality", 
            fontsize=18, color='#E50914', fontweight='bold', ha='right', va='center')
    
    # Axis Limits and Ticks
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Custom Axis Labels
    ax.text(0.55, -0.05, "Cost ($/request) →", transform=ax.transAxes, 
            fontsize=20, ha='center', va='top', color='#111827')
    ax.text(-0.05, 0.5, "Expected Quality →", transform=ax.transAxes, 
            fontsize=20, ha='right', va='center', rotation=90, color='#111827')
    
    # Footnote
    fig.text(0.05, 0.05, "*Static routing relies on fixed percentages of cheap/expensive models. ParetoBandit allocates dynamically per prompt to hit the optimal frontier.",
             fontsize=14, color='#6B7280')
             
    output_path = "blog/paretobandit_netflix_style_pareto.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    print(f"Saved: {create_slide()}")
