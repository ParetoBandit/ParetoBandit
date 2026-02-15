"""
Conference-style plotting utilities for BanditGPT experiments.

Provides consistent, publication-ready formatting across all figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# conference formatting constants
Conference_STYLE = {
    "figure.figsize": (7, 3.5),  # Double-column width
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "purple": "#CA9161",
    "gray": "#949494",
}


def apply_kdd_style():
    """Apply conference formatting to matplotlib globally."""
    mpl.rcParams.update(Paper_STYLE)


def save_kdd_style_plot(fig, filename, output_dir="results"):
    """
    Save figure with conference formatting.
    
    Args:
        fig: matplotlib figure object
        filename: output filename (e.g., "fig1_regret.pdf")
        output_dir: directory to save in (created if doesn't exist)
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=300, bbox_inches="tight", format="pdf")
    
    # Also save PNG for quick preview
    png_path = filepath.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", format="png")
    
    print(f"✓ Saved: {filepath}")
    print(f"✓ Preview: {png_path}")
    
    return filepath


def plot_with_ci(ax, x, y, yerr=None, label=None, color=None, **kwargs):
    """
    Plot line with confidence interval shading.
    
    Args:
        ax: matplotlib axis
        x: x values
        y: y values (mean)
        yerr: error (std or half-width of CI)
        label: legend label
        color: line color
        **kwargs: additional plotting kwargs
    """
    if color is None:
        color = COLORS["blue"]
    
    # Plot line
    line = ax.plot(x, y, label=label, color=color, linewidth=2, **kwargs)
    
    # Add confidence interval
    if yerr is not None:
        import numpy as np
        y = np.array(y)
        yerr = np.array(yerr)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)
    
    return line


# Placeholder for future implementations
def create_regret_plot(data, output_file="fig1_regret.pdf"):
    """
    [PLACEHOLDER] Generate cumulative regret comparison plot.
    
    Args:
        data: dict with keys=['banditgpt', 'linucb', 'random', 'epsilon_greedy']
              each containing {'timesteps': [...], 'regret': [...], 'ci': [...]}
        output_file: output filename
    """
    apply_kdd_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # TODO: Implement actual plotting logic
    # For now, just a placeholder
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret Comparison")
    ax.legend()
    
    save_kdd_style_plot(fig, output_file)
    plt.close()


if __name__ == "__main__":
    # Test the plotting utilities
    import numpy as np
    
    apply_kdd_style()
    fig, ax = plt.subplots()
    
    x = np.linspace(0, 100, 50)
    y = np.exp(-x / 20)
    yerr = 0.1 * y
    
    plot_with_ci(ax, x, y, yerr=yerr, label="Test", color=COLORS["blue"])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Regret")
    ax.legend()
    
    save_kdd_style_plot(fig, "test_plot.pdf", output_dir=".")
    print("✓ Plot utilities working correctly!")
