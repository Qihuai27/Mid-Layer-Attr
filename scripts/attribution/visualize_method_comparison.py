#!/usr/bin/env python3
"""
Multi-method comparison visualization for a single sample.

Shows token heatmaps and AUC curves for multiple attribution methods side-by-side,
making it easy to compare their behavior on the same input.

Usage:
    # Compare all available methods on sample 0 from ioi dataset
    python scripts/attribution/visualize_method_comparison.py \
        --model Qwen3-4B \
        --dataset ioi \
        --sample-idx 0

    # Compare specific methods
    python scripts/attribution/visualize_method_comparison.py \
        --model Qwen3-4B \
        --dataset ioi \
        --sample-idx 0 \
        --methods depass attention_rollout grad_input

    # Custom output
    python scripts/attribution/visualize_method_comparison.py \
        --model Qwen3-4B \
        --dataset ioi \
        --sample-idx 0 \
        --output results/attribution/figures/comparison_sample0.png
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Color Schemes
# ============================================================================

METHOD_COLORS = {
    "attention_rollout": "#1f77b4",
    "depass": "#ff7f0e",
    "integrated_gradients": "#2ca02c",
    "grad_input": "#d62728",
    "grad_sam": "#9467bd",
    "shapley": "#8c564b",
    "midlayer": "#e377c2",
    "midlayer_v2": "#7f7f7f",
    "input_causal": "#bcbd22",
    "greedy_optimal": "#17becf",
}

METHOD_LABELS = {
    "attention_rollout": "Attention Rollout",
    "depass": "DePass",
    "integrated_gradients": "Integrated Gradients",
    "grad_input": "Grad×Input",
    "grad_sam": "GradSAM",
    "shapley": "TokenShapley",
    "midlayer": "MidLayer",
    "midlayer_v2": "MidLayer-V2",
    "input_causal": "Input Causal",
    "greedy_optimal": "Greedy Optimal",
}

# Token heatmap colormap
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "attribution",
    ["#FFFFFF", "#E8F4FD", "#B3D9F7", "#5BA3D9", "#1A5F9E", "#0D3B66"],
    N=256
)


# ============================================================================
# Data Loading
# ============================================================================

def load_sample_from_method(
    output_dir: Path,
    model_name: str,
    method: str,
    dataset: str,
    sample_idx: int
) -> Optional[Dict]:
    """Load a specific sample from method results."""
    model_safe = model_name.replace("/", "_")
    result_path = output_dir / "detailed" / model_safe / f"{method}_{dataset}.pkl"

    if not result_path.exists():
        print(f"Warning: {result_path} not found")
        return None

    with open(result_path, "rb") as f:
        results = pickle.load(f)

    if sample_idx >= len(results):
        print(f"Warning: Sample {sample_idx} not found in {method} results")
        return None

    return results[sample_idx]


def discover_available_methods(
    output_dir: Path,
    model_name: str,
    dataset: str
) -> List[str]:
    """Discover which methods have results available."""
    model_safe = model_name.replace("/", "_")
    detailed_dir = output_dir / "detailed" / model_safe

    if not detailed_dir.exists():
        return []

    methods = []
    for path in detailed_dir.glob(f"*_{dataset}.pkl"):
        method = path.stem.replace(f"_{dataset}", "")
        methods.append(method)

    return sorted(methods)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_token_heatmap_compact(
    ax: plt.Axes,
    tokens: List[str],
    scores: np.ndarray,
    title: str,
    max_tokens_per_row: int = 15,
    highlight_top_k: Optional[int] = 3,
):
    """Plot compact token heatmap on given axes."""
    scores = np.asarray(scores).flatten()

    # Normalize scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s > 1e-10:
        norm_scores = (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(scores)

    # Get top-k positions
    top_k_positions = set()
    if highlight_top_k:
        top_k_indices = np.argsort(-scores)[:highlight_top_k]
        top_k_positions = set(top_k_indices)

    # Calculate layout
    token_widths = [max(len(t.replace("Ġ", "").replace("▁", "")), 1) * 0.045 + 0.08
                    for t in tokens]

    row_width = 0.95
    current_x = 0.025
    current_row = 0
    row_starts = [0]

    for i, width in enumerate(token_widths):
        if current_x + width > row_width and i > row_starts[-1]:
            current_row += 1
            row_starts.append(i)
            current_x = 0.025
        current_x += width

    n_rows = current_row + 1
    box_height = 0.7 / (n_rows + 0.5)

    # Draw tokens
    row_idx = 0
    x_pos = 0.025

    for i, (token, score, width) in enumerate(zip(tokens, norm_scores, token_widths)):
        # Check for new row
        if row_idx < len(row_starts) - 1 and i == row_starts[row_idx + 1]:
            row_idx += 1
            x_pos = 0.025

        y_pos = 0.95 - (row_idx + 1) * (0.85 / (n_rows + 0.5))

        # Color
        color = HEATMAP_CMAP(score)

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x_pos, y_pos),
            width - 0.003,
            box_height,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            facecolor=color,
            edgecolor="#CCCCCC",
            linewidth=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Highlight top-k
        if i in top_k_positions:
            highlight_rect = mpatches.FancyBboxPatch(
                (x_pos, y_pos),
                width - 0.003,
                box_height,
                boxstyle="round,pad=0.005,rounding_size=0.015",
                facecolor="none",
                edgecolor="#FFD700",
                linewidth=2,
                transform=ax.transAxes,
            )
            ax.add_patch(highlight_rect)

        # Clean token
        display_token = token.replace("Ġ", " ").replace("▁", " ")
        if display_token.startswith(" "):
            display_token = "·" + display_token[1:]

        # Text color
        text_color = "white" if score > 0.6 else "black"

        # Add text
        ax.text(
            x_pos + (width - 0.003) / 2,
            y_pos + box_height / 2,
            display_token,
            ha="center",
            va="center",
            fontsize=12,
            color=text_color,
            fontfamily="monospace",
            transform=ax.transAxes,
        )

        x_pos += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=5)


def plot_auc_curve_compact(
    ax: plt.Axes,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    auc_value: float,
    curve_type: str,
    color: str,
):
    """Plot compact AUC curve."""
    is_noise = curve_type == "perturbation"

    # Fill under curve
    ax.fill_between(curve_x, curve_y, alpha=0.25, color=color)

    # Plot curve
    ax.plot(
        curve_x, curve_y,
        color=color,
        linewidth=2,
        marker="o" if is_noise else "s",
        markersize=3,
        label=f"AUC={auc_value:.3f}"
    )

    # Labels
    if is_noise:
        ax.set_ylabel("Cosine Sim.", fontsize=13)
        title = "P-AUC (↓)"
    else:
        ax.set_ylabel("Recovery", fontsize=13)
        title = "R-AUC (↑)"

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Fraction", fontsize=12)
    ax.legend(loc="best", fontsize=11, framealpha=0.8)
    ax.set_xlim(-0.02, 1.02)
    y_min = min(curve_y) - 0.05 if is_noise else -0.05
    ax.set_ylim(y_min, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_facecolor("#FAFAFA")


def create_method_comparison_figure(
    samples_data: Dict[str, Dict],
    methods: List[str],
    output_path: Optional[str] = None,
    figsize_per_method: tuple = (4, 6),
    dpi: int = 150,
):
    """
    Create multi-method comparison figure.

    Args:
        samples_data: Dict mapping method name to sample data
        methods: List of method names to display
        output_path: Path to save figure
        figsize_per_method: Size per method column (width, height)
        dpi: Resolution
    """
    n_methods = len(methods)

    # Calculate figure size
    fig_width = figsize_per_method[0] * n_methods
    fig_height = figsize_per_method[1] + 5

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid: each method gets 3 rows (heatmap, P-AUC, R-AUC)
    gs = gridspec.GridSpec(
        3, n_methods,
        height_ratios=[2, 1, 1],
        hspace=0.4,
        wspace=0.3
    )

    # Get sample info from first available method
    first_sample = list(samples_data.values())[0]
    sample_id = first_sample.get("case_id", "Unknown")
    prompt = first_sample.get("prompt", "")

    # Main title
    fig.suptitle(
        f"Multi-Method Comparison (Sample {sample_id})",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # Prompt subtitle
    if prompt:
        prompt_display = prompt if len(prompt) <= 100 else prompt[:97] + "..."
        fig.text(
            0.5, 0.94,
            f'Prompt: "{prompt_display}"',
            ha="center",
            fontsize=14,
            style="italic",
            color="#555555"
        )

    # Plot each method
    for col_idx, method in enumerate(methods):
        if method not in samples_data:
            continue

        sample = samples_data[method]
        method_label = METHOD_LABELS.get(method, method)
        method_color = METHOD_COLORS.get(method, "#333333")

        # Heatmap (top row)
        ax_heat = fig.add_subplot(gs[0, col_idx])
        plot_token_heatmap_compact(
            ax=ax_heat,
            tokens=sample["tokens"],
            scores=sample["attribution_scores"],
            title=method_label,
        )

        # P-AUC curve (middle row)
        ax_p = fig.add_subplot(gs[1, col_idx])
        if sample.get("perturbation_curve_x") is not None:
            plot_auc_curve_compact(
                ax=ax_p,
                curve_x=sample["perturbation_curve_x"],
                curve_y=sample["perturbation_curve_y"],
                auc_value=sample["perturbation_auc"],
                curve_type="perturbation",
                color=method_color,
            )
        else:
            ax_p.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_p.transAxes)
            ax_p.axis("off")

        # R-AUC curve (bottom row)
        ax_r = fig.add_subplot(gs[2, col_idx])
        if sample.get("recovery_curve_x") is not None:
            plot_auc_curve_compact(
                ax=ax_r,
                curve_x=sample["recovery_curve_x"],
                curve_y=sample["recovery_curve_y"],
                auc_value=sample["recovery_auc"],
                curve_type="recovery",
                color=method_color,
            )
        else:
            ax_r.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax_r.transAxes)
            ax_r.axis("off")

    # Add colorbar for heatmaps
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.55, 0.01, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Importance", fontsize=13)
    cbar.ax.tick_params(labelsize=7)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.90)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple attribution methods on a single sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True,
                       help="Model name")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index to visualize")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                       help="Methods to compare (default: all available)")
    parser.add_argument("--input-dir", type=str, default="results/attribution",
                       help="Input directory with detailed results")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--dpi", type=int, default=200,
                       help="Figure resolution")

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)

    # Discover available methods
    if args.methods is None:
        methods = discover_available_methods(input_dir, args.model, args.dataset)
        if not methods:
            print(f"No methods found for {args.model}/{args.dataset}")
            sys.exit(1)
        print(f"Found {len(methods)} methods: {', '.join(methods)}")
    else:
        methods = args.methods

    # Load sample data from all methods
    samples_data = {}
    for method in methods:
        sample = load_sample_from_method(
            input_dir, args.model, method, args.dataset, args.sample_idx
        )
        if sample:
            samples_data[method] = sample

    if not samples_data:
        print(f"No data found for sample {args.sample_idx}")
        sys.exit(1)

    print(f"Loaded {len(samples_data)} methods for sample {args.sample_idx}")

    # Generate output path
    if args.output is None:
        output_dir = Path("results/attribution/figures/comparisons")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_safe = args.model.replace("/", "_")
        args.output = output_dir / f"{model_safe}_{args.dataset}_sample{args.sample_idx}_comparison.png"

    # Create visualization
    create_method_comparison_figure(
        samples_data=samples_data,
        methods=list(samples_data.keys()),
        output_path=args.output,
        dpi=args.dpi,
    )

    print(f"\n✓ Comparison visualization complete!")
    print(f"  Sample: {args.sample_idx}")
    print(f"  Methods: {len(samples_data)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
