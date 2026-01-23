"""
Shared utilities for BanditGPT experiments.

This module provides common functionality used across all experiments:
- KDD-style plotting
- Metric calculation
- Data loading

Usage:
    from experiments.utils.plotting import save_kdd_style_plot
    from experiments.utils.metrics import calculate_cumulative_regret
    from experiments.utils.data_loader import load_test_prompts
"""

__all__ = ["plotting", "metrics", "data_loader"]
