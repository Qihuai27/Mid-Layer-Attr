#!/usr/bin/env python3
"""
Plot AUC curves from attribution evaluation results.

This script loads detailed results from the attribution pipeline and generates
publication-quality plots for:
1. Individual sample curves (Perturbation and Recovery)
2. Aggregated mean curves with std bands across methods
3. Comparison bar charts for AUC values

Usage:
    # Plot mean curves for all methods on a dataset
    python scripts/attribution/plot_auc_curves.py --model Qwen3-4B --dataset ioi

    # Plot individual sample curves
    python scripts/attribution/plot_auc_curves.py --model Qwen3-4B --dataset ioi --sample-idx 0

    # Compare all methods in a bar chart
    python scripts/attribution/plot_auc_curves.py --model Qwen3-4B --dataset ioi --mode bar

    # Plot mean curves for specific methods
    python scripts/attribution/plot_auc_curves.py --model Qwen3-4B --dataset ioi \\
        --methods attention_rollout depass integrated_gradients
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Color Palette and Styles (Academic-friendly)
# ============================================================================

# Colorblind-friendly palette
METHOD_COLORS = {
    "attention_rollout": "#1f77b4",     # Blue
    "depass": "#ff7f0e",                 # Orange
    "integrated_gradients": "#2ca02c",  # Green
    "grad_input": "#d62728",             # Red
    "grad_sam": "#9467bd",               # Purple
    "shapley": "#8c564b",                # Brown
    "midlayer": "#e377c2",               # Pink
    "midlayer_v2": "#7f7f7f",            # Gray
    "input_causal": "#bcbd22",           # Yellow-green
    "greedy_optimal": "#17becf",         # Cyan
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


def setup_academic_style():
    """Set up matplotlib for academic-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# ============================================================================
# Data Loading
# ============================================================================

def get_detailed_results_path(output_dir: Path, model_name: str, method: str, dataset: str) -> Path:
    """Get path for detailed results file."""
    model_safe = model_name.replace("/", "_")
    return output_dir / "detailed" / model_safe / f"{method}_{dataset}.pkl"


def load_detailed_results(results_path: Path) -> List[Dict]:
    """Load detailed results from pickle file."""
    if not results_path.exists():
        return []
    with open(results_path, "rb") as f:
        return pickle.load(f)


def discover_available_methods(output_dir: Path, model_name: str, dataset: str) -> List[str]:
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
# Curve Aggregation
# ============================================================================

def aggregate_curves(
    results: List[Dict],
    curve_type: str = "perturbation",
) -> Dict[str, np.ndarray]:
    """
    Aggregate curves across samples to compute mean and std.

    Args:
        results: List of per-sample results
        curve_type: "perturbation" or "recovery"

    Returns:
        Dict with keys: "x", "y_mean", "y_std", "auc_mean", "auc_std"
    """
    x_key = f"{curve_type}_curve_x"
    y_key = f"{curve_type}_curve_y"
    auc_key = f"{curve_type}_auc"

    # Filter samples with valid curves
    valid_results = [
        r for r in results
        if r.get(x_key) is not None and r.get(y_key) is not None
    ]

    if not valid_results:
        return None

    # Extract curves (assume all have same x values)
    x = np.array(valid_results[0][x_key])
    ys = np.array([r[y_key] for r in valid_results])
    aucs = np.array([r[auc_key] for r in valid_results if r.get(auc_key) is not None])

    return {
        "x": x,
        "y_mean": np.mean(ys, axis=0),
        "y_std": np.std(ys, axis=0),
        "auc_mean": np.mean(aucs) if len(aucs) > 0 else None,
        "auc_std": np.std(aucs) if len(aucs) > 0 else None,
        "n_samples": len(valid_results),
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_single_sample_curves(
    results: List[Dict],
    sample_idx: int,
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 3.5),
):
    """
    Plot Perturbation and Recovery curves for a single sample.
    """
    if sample_idx >= len(results):
        print(f"Sample index {sample_idx} out of range (max: {len(results)-1})")
        return

    sample = results[sample_idx]
    setup_academic_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Perturbation curve (left)
    ax = axes[0]
    if sample.get("perturbation_curve_x") is not None:
        x = sample["perturbation_curve_x"]
        y = sample["perturbation_curve_y"]
        auc = sample.get("perturbation_auc", 0)
        ax.plot(x, y, color="#1f77b4", linewidth=2)
        ax.fill_between(x, y, alpha=0.2, color="#1f77b4")
        ax.set_title(f"Perturbation (AUC={auc:.4f})", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Perturbation", fontweight="bold")

    ax.set_xlabel("Fraction of tokens corrupted", fontweight="bold")
    ax.set_ylabel("Cosine similarity", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Recovery curve (right)
    ax = axes[1]
    if sample.get("recovery_curve_x") is not None:
        x = sample["recovery_curve_x"]
        y = sample["recovery_curve_y"]
        auc = sample.get("recovery_auc", 0)
        ax.plot(x, y, color="#2ca02c", linewidth=2)
        ax.fill_between(x, y, alpha=0.2, color="#2ca02c")
        ax.set_title(f"Recovery (AUC={auc:.4f})", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recovery", fontweight="bold")

    ax.set_xlabel("Fraction of tokens restored", fontweight="bold")
    ax.set_ylabel("Hidden state recovery", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add sample info as suptitle
    case_id = sample.get("case_id", sample_idx)
    fig.suptitle(f"Sample {case_id}", fontweight="bold", y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_mean_curves_comparison(
    output_dir: Path,
    model_name: str,
    dataset: str,
    methods: Optional[List[str]] = None,
    curve_type: str = "perturbation",
    output_path: Optional[Path] = None,
    figsize: tuple = (4.5, 3.5),
    show_std: bool = True,
):
    """
    Plot mean curves for multiple methods on the same plot.
    """
    setup_academic_style()

    if methods is None:
        methods = discover_available_methods(output_dir, model_name, dataset)

    if not methods:
        print("No methods found with results.")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        results_path = get_detailed_results_path(output_dir, model_name, method, dataset)
        results = load_detailed_results(results_path)

        if not results:
            continue

        agg = aggregate_curves(results, curve_type)
        if agg is None:
            continue

        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)

        # Plot mean curve
        ax.plot(
            agg["x"], agg["y_mean"],
            color=color,
            linewidth=2,
            label=f"{label} ({agg['auc_mean']:.3f})",
        )

        # Plot std band
        if show_std:
            ax.fill_between(
                agg["x"],
                agg["y_mean"] - agg["y_std"],
                agg["y_mean"] + agg["y_std"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    # Styling
    if curve_type == "perturbation":
        ax.set_xlabel("Fraction of tokens corrupted", fontweight="bold")
        ax.set_ylabel("Cosine similarity", fontweight="bold")
        ax.set_title(f"Perturbation Curves ({dataset})", fontweight="bold")
    else:
        ax.set_xlabel("Fraction of tokens restored", fontweight="bold")
        ax.set_ylabel("Hidden state recovery", fontweight="bold")
        ax.set_title(f"Recovery Curves ({dataset})", fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, len(methods)),
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_auc_bar_comparison(
    output_dir: Path,
    model_name: str,
    dataset: str,
    methods: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 3.5),
):
    """
    Plot bar chart comparing AUC values across methods.
    """
    setup_academic_style()

    if methods is None:
        methods = discover_available_methods(output_dir, model_name, dataset)

    if not methods:
        print("No methods found with results.")
        return

    # Collect data
    method_names = []
    p_aucs = []
    p_stds = []
    r_aucs = []
    r_stds = []

    for method in methods:
        results_path = get_detailed_results_path(output_dir, model_name, method, dataset)
        results = load_detailed_results(results_path)

        if not results:
            continue

        p_agg = aggregate_curves(results, "perturbation")
        r_agg = aggregate_curves(results, "recovery")

        if p_agg is None or r_agg is None:
            continue

        method_names.append(METHOD_LABELS.get(method, method))
        p_aucs.append(p_agg["auc_mean"])
        p_stds.append(p_agg["auc_std"])
        r_aucs.append(r_agg["auc_mean"])
        r_stds.append(r_agg["auc_std"])

    if not method_names:
        print("No valid data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(method_names))
    width = 0.6

    # Perturbation AUC (lower is better)
    ax = axes[0]
    colors = [METHOD_COLORS.get(m, "#333333") for m in methods[:len(method_names)]]
    bars = ax.bar(x, p_aucs, width, yerr=p_stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("P-AUC (lower is better)", fontweight="bold")
    ax.set_title("Perturbation AUC", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Recovery AUC (higher is better)
    ax = axes[1]
    bars = ax.bar(x, r_aucs, width, yerr=r_stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("R-AUC (higher is better)", fontweight="bold")
    ax.set_title("Recovery AUC", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(f"Attribution Method Comparison ({dataset})", fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_all_methods_grid(
    output_dir: Path,
    model_name: str,
    dataset: str,
    methods: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
):
    """
    Plot a grid of Perturbation and Recovery curves for all methods.
    """
    setup_academic_style()

    if methods is None:
        methods = discover_available_methods(output_dir, model_name, dataset)

    if not methods:
        print("No methods found with results.")
        return

    n_methods = len(methods)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, method in enumerate(methods):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        results_path = get_detailed_results_path(output_dir, model_name, method, dataset)
        results = load_detailed_results(results_path)

        if not results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(METHOD_LABELS.get(method, method), fontweight="bold", fontsize=9)
            continue

        # Plot both curves
        p_agg = aggregate_curves(results, "perturbation")
        r_agg = aggregate_curves(results, "recovery")

        color = METHOD_COLORS.get(method, "#333333")

        if p_agg:
            ax.plot(p_agg["x"], p_agg["y_mean"], color="#1f77b4", linewidth=1.5, label="P-AUC")
            ax.fill_between(p_agg["x"], p_agg["y_mean"] - p_agg["y_std"],
                          p_agg["y_mean"] + p_agg["y_std"], alpha=0.15, color="#1f77b4")

        if r_agg:
            ax.plot(r_agg["x"], r_agg["y_mean"], color="#2ca02c", linewidth=1.5, label="R-AUC")
            ax.fill_between(r_agg["x"], r_agg["y_mean"] - r_agg["y_std"],
                          r_agg["y_mean"] + r_agg["y_std"], alpha=0.15, color="#2ca02c")

        # Title with AUC values
        title = METHOD_LABELS.get(method, method)
        if p_agg and r_agg:
            title += f"\nP:{p_agg['auc_mean']:.3f} R:{r_agg['auc_mean']:.3f}"
        ax.set_title(title, fontsize=8, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if row == n_rows - 1:
            ax.set_xlabel("Fraction", fontsize=8)
        if col == 0:
            ax.set_ylabel("Score", fontsize=8)

    # Hide empty subplots
    for idx in range(len(methods), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
              ncol=2, frameon=False, fontsize=9)

    fig.suptitle(f"Attribution Methods Comparison ({model_name} / {dataset})",
                fontweight="bold", fontsize=11, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.92, hspace=0.4, wspace=0.25)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot AUC curves from attribution evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                       help="Methods to plot (default: all available)")
    parser.add_argument("--input-dir", type=str, default="results/attribution",
                       help="Input directory with detailed results")
    parser.add_argument("--output-dir", type=str, default="results/attribution/figures",
                       help="Output directory for figures")
    parser.add_argument("--mode", type=str, default="curves",
                       choices=["curves", "bar", "grid", "sample"],
                       help="Plot mode: curves (mean comparison), bar (AUC bars), grid (all methods), sample (single sample)")
    parser.add_argument("--curve-type", type=str, default="both",
                       choices=["perturbation", "recovery", "both"],
                       help="Curve type to plot (for curves mode)")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index to plot (for sample mode)")
    parser.add_argument("--no-show", action="store_true", help="Don't display, only save")

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_safe = args.model.replace("/", "_")

    if args.mode == "sample":
        # Plot single sample
        method = args.methods[0] if args.methods else discover_available_methods(input_dir, args.model, args.dataset)[0]
        results_path = get_detailed_results_path(input_dir, args.model, method, args.dataset)
        results = load_detailed_results(results_path)

        output_path = output_dir / f"{model_safe}_{args.dataset}_{method}_sample{args.sample_idx}.png"
        plot_single_sample_curves(results, args.sample_idx, output_path if not args.no_show else output_path)

    elif args.mode == "curves":
        # Plot mean curves comparison
        if args.curve_type in ["perturbation", "both"]:
            output_path = output_dir / f"{model_safe}_{args.dataset}_perturbation_curves.png"
            plot_mean_curves_comparison(
                input_dir, args.model, args.dataset, args.methods,
                "perturbation", output_path
            )

        if args.curve_type in ["recovery", "both"]:
            output_path = output_dir / f"{model_safe}_{args.dataset}_recovery_curves.png"
            plot_mean_curves_comparison(
                input_dir, args.model, args.dataset, args.methods,
                "recovery", output_path
            )

    elif args.mode == "bar":
        # Plot bar chart comparison
        output_path = output_dir / f"{model_safe}_{args.dataset}_auc_comparison.png"
        plot_auc_bar_comparison(input_dir, args.model, args.dataset, args.methods, output_path)

    elif args.mode == "grid":
        # Plot grid of all methods
        output_path = output_dir / f"{model_safe}_{args.dataset}_methods_grid.png"
        plot_all_methods_grid(input_dir, args.model, args.dataset, args.methods, output_path)


if __name__ == "__main__":
    main()
