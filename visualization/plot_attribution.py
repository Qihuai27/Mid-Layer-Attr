"""
Visualization script for attention attribution statistics.

Plots S_wp, S_pq, and S_ww statistics across transformer layers.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for pickle to resolve module references
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Light color palette
COLORS = {
    "S_wp": "#7FB3D5",  # Light blue
    "S_pq": "#F5B7B1",  # Light coral/pink
    "S_ww": "#A9DFBF",  # Light green
}

LABELS = {
    "S_wp": r"$S_{wp}$ (Anchor $\rightarrow$ Previous)",
    "S_pq": r"$S_{pq}$ (Final $\rightarrow$ Anchor)",
    "S_ww": r"$S_{ww}$ (Other Positions)",
}


def load_results(path: str) -> Dict:
    """Load experiment results from pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle both formats: direct results or wrapped in 'results' key
    if "results" in data and "metadata" in data:
        return data["results"]
    return data


def plot_attribution_statistics(
    results: Dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    show_std: bool = True,
    dpi: int = 150,
):
    """
    Plot S_wp, S_pq, and S_ww statistics across layers.

    Args:
        results: Dictionary containing 'statistics' with mean and std arrays
        output_path: Path to save the figure (if None, displays interactively)
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches
        show_std: Whether to show standard deviation as shaded region
        dpi: Resolution for saved figure
    """
    stats = results["statistics"]
    config = results.get("config", {})

    # Extract mean values
    mean_S_wp = stats["mean_S_wp"]
    mean_S_pq = stats["mean_S_pq"]
    mean_S_ww = stats["mean_S_ww"]

    num_layers = len(mean_S_wp)
    layers = np.arange(num_layers)

    # Create figure with light style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines
    ax.plot(
        layers,
        mean_S_wp,
        color=COLORS["S_wp"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=LABELS["S_wp"],
    )
    ax.plot(
        layers,
        mean_S_pq,
        color=COLORS["S_pq"],
        linewidth=2.5,
        marker="s",
        markersize=5,
        label=LABELS["S_pq"],
    )
    ax.plot(
        layers,
        mean_S_ww,
        color=COLORS["S_ww"],
        linewidth=2.5,
        marker="^",
        markersize=5,
        label=LABELS["S_ww"],
    )

    # Add shaded std regions if available and requested
    if show_std and "std_S_wp" in stats:
        std_S_wp = stats["std_S_wp"]
        std_S_pq = stats["std_S_pq"]
        std_S_ww = stats["std_S_ww"]

        ax.fill_between(
            layers,
            mean_S_wp - std_S_wp,
            mean_S_wp + std_S_wp,
            color=COLORS["S_wp"],
            alpha=0.2,
        )
        ax.fill_between(
            layers,
            mean_S_pq - std_S_pq,
            mean_S_pq + std_S_pq,
            color=COLORS["S_pq"],
            alpha=0.2,
        )
        ax.fill_between(
            layers,
            mean_S_ww - std_S_ww,
            mean_S_ww + std_S_ww,
            color=COLORS["S_ww"],
            alpha=0.2,
        )

    # Labels and title
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalized Attention Score", fontsize=12)

    if title is None:
        task = config.get("task_name", "unknown")
        model = config.get("model_name", "unknown")
        shot = config.get("demonstration_shot", "?")
        title = f"Attention Attribution Statistics\n{model} on {task.upper()} ({shot}-shot)"

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    # Axis settings
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_xticks(layers[::2])  # Show every other layer
    ax.tick_params(axis="both", labelsize=10)

    # Light grid
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot attention attribution statistics"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for figure (default: display interactively)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Hide standard deviation shading",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saving",
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Plot
    plot_attribution_statistics(
        results,
        output_path=args.output,
        title=args.title,
        figsize=tuple(args.figsize),
        show_std=not args.no_std,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
