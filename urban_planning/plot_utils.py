"""
Shared plotting utilities for urban planning visualizations.

Ensures consistent styling across all plots.
"""

import matplotlib.pyplot as plt


def setup_plot_style(fontsize=23):
    """
    Configure matplotlib rcParams for consistent plot styling.

    Args:
        fontsize: Base font size for all text elements (default: 23)

    Usage:
        from plot_utils import setup_plot_style
        setup_plot_style()  # Call at start of visualization script
    """
    plt.rcParams.update({
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'savefig.dpi': 75,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'legend.fontsize': fontsize * 0.9,
        'legend.labelspacing': .3,
        'legend.columnspacing': .3,
        'legend.handletextpad': .1,
        'text.usetex': False,                 # Use mathtext, no LaTeX needed
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    })


def setup_high_res_plot_style(fontsize=23, dpi=300):
    """
    Configure matplotlib rcParams for high-resolution publication-quality plots.

    Args:
        fontsize: Base font size for all text elements (default: 23)
        dpi: Resolution for saved figures (default: 300 for publication)
    """
    plt.rcParams.update({
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'savefig.dpi': dpi,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'legend.fontsize': fontsize * 0.9,
        'legend.labelspacing': .3,
        'legend.columnspacing': .3,
        'legend.handletextpad': .1,
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    })


# Consistent color palette
COLORS = {
    'simultaneous': '#e74c3c',      # Red
    'staged': '#3498db',            # Blue
    'contraflow': '#27ae60',        # Green
    'baseline': '#95a5a6',          # Gray

    # Traffic states
    'free_flow': '#2ecc71',         # Light green
    'light_congestion': '#f1c40f',  # Yellow
    'moderate_congestion': '#f39c12',  # Orange
    'severe_congestion': '#e74c3c',    # Red

    # Infrastructure
    'road_network': '#95a5a6',      # Gray
    'origin': '#e67e22',            # Orange
    'safe_zone': '#27ae60',         # Green
    'danger_zone': '#e74c3c',       # Red
}


# Line styles
LINE_STYLES = {
    'simultaneous': '-',
    'staged': '--',
    'contraflow': '-',
    'baseline': '-',
}
