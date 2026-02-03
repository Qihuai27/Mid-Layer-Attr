"""
Visualization functions for cluster statistics.

Plots S_a, S_o, and S_w statistics across transformer layers:
- S_a (aggregation): Information flow from previous tokens TO cluster positions
- S_o (output): Information flow from cluster positions TO final position
- S_w (within): Information flow between other (non-cluster, non-final) positions

Designed for academic papers: compact size (1/4 page in single column),
bold axes, large fonts, external legend, no grid lines.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Soft academic color palette
COLORS = {
    "S_a": "#6A9FD1",  # Soft blue
    "S_o": "#E8927C",  # Soft coral
    "S_w": "#82C9A5",  # Soft green
}

LABELS = {
    "S_a": r"$S_a$",
    "S_o": r"$S_o$",
    "S_w": r"$S_w$",
}


def load_results(path: str) -> Dict:
    """
    Load experiment results from pickle file.

    Args:
        path: Path to the pickle file

    Returns:
        Dictionary containing experiment results
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle both formats: direct results or wrapped in 'results' key
    if "results" in data and "metadata" in data:
        return data["results"]
    return data


def plot_cluster_statistics(
    results: Dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (3.5, 2.5),
    show_std: bool = True,
    dpi: int = 300,
    show_title: bool = False,
) -> None:
    """
    Plot S_a, S_o, and S_w statistics across layers.

    Designed for academic papers (1/4 page in single column of double-column layout).

    Args:
        results: Dictionary containing 'statistics' with mean and std arrays
        output_path: Path to save the figure (if None, displays interactively)
        title: Plot title (ignored if show_title=False)
        figsize: Figure size in inches (default: 3.5x2.5 for 1/4 page)
        show_std: Whether to show standard deviation as shaded region
        dpi: Resolution for saved figure (300 for print)
        show_title: Whether to show title (default: False for papers)
    """
    stats = results["statistics"]
    config = results.get("config", {})

    # Extract mean values (support both new and legacy naming)
    mean_S_a = stats.get("mean_S_a", stats.get("mean_S_wp"))
    mean_S_o = stats.get("mean_S_o", stats.get("mean_S_pq"))
    mean_S_w = stats.get("mean_S_w", stats.get("mean_S_ww"))

    num_layers = len(mean_S_a)
    layers = np.arange(num_layers)

    # Set up matplotlib for academic style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines without markers
    ax.plot(
        layers,
        mean_S_a,
        color=COLORS["S_a"],
        linewidth=2,
        label=LABELS["S_a"],
    )
    ax.plot(
        layers,
        mean_S_o,
        color=COLORS["S_o"],
        linewidth=2,
        label=LABELS["S_o"],
    )
    ax.plot(
        layers,
        mean_S_w,
        color=COLORS["S_w"],
        linewidth=2,
        label=LABELS["S_w"],
    )

    # Add shaded std regions if available and requested
    if show_std:
        std_S_a = stats.get("std_S_a", stats.get("std_S_wp"))
        std_S_o = stats.get("std_S_o", stats.get("std_S_pq"))
        std_S_w = stats.get("std_S_w", stats.get("std_S_ww"))

        if std_S_a is not None:
            ax.fill_between(
                layers,
                mean_S_a - std_S_a,
                mean_S_a + std_S_a,
                color=COLORS["S_a"],
                alpha=0.15,
                linewidth=0,
            )
            ax.fill_between(
                layers,
                mean_S_o - std_S_o,
                mean_S_o + std_S_o,
                color=COLORS["S_o"],
                alpha=0.15,
                linewidth=0,
            )
            ax.fill_between(
                layers,
                mean_S_w - std_S_w,
                mean_S_w + std_S_w,
                color=COLORS["S_w"],
                alpha=0.15,
                linewidth=0,
            )

    # Axis labels (bold style via fontweight)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Information Flow", fontweight='bold')

    # Title (optional, default off for papers)
    if show_title and title:
        ax.set_title(title, fontweight='bold')

    # Axis settings - no grid, clean look
    ax.set_xlim(-0.5, num_layers - 0.5)

    # Sparse x-ticks for readability
    if num_layers <= 12:
        ax.set_xticks(layers[::3])  # Every 3rd layer
    elif num_layers <= 24:
        ax.set_xticks(layers[::4])  # Every 4th layer
    elif num_layers <= 48:
        ax.set_xticks(layers[::8])  # Every 8th layer
    else:
        ax.set_xticks(layers[::10])  # Every 10th layer

    # Remove top and right spines, keep bottom and left bold
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    # No grid
    ax.grid(False)

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Legend outside the plot (below)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    plt.tight_layout()

    # Adjust for external legend
    plt.subplots_adjust(bottom=0.28)

    # Save or show
    if output_path:
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor='none',
            pad_inches=0.02,
        )
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# Backward compatibility alias
plot_attribution_statistics = plot_cluster_statistics
