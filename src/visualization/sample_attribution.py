"""
Single sample attribution visualization.

Provides visualization for token-level importance scores and AUC curves.
Designed for research paper quality figures with consistent scientific styling.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec


# =============================================================================
# Scientific Color Palette
# =============================================================================

# Primary scientific colormap: white -> light blue -> deep blue
# Inspired by sequential colormaps commonly used in NLP/ML papers
ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "attribution",
    [
        "#FFFFFF",  # White (low importance)
        "#E8F4FD",  # Very light blue
        "#B3D9F7",  # Light blue
        "#5BA3D9",  # Medium blue
        "#1A5F9E",  # Deep blue
        "#0D3B66",  # Dark blue (high importance)
    ],
    N=256
)

# Alternative: diverging colormap for signed importance (negative -> neutral -> positive)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "attribution_diverging",
    [
        "#B22222",  # Dark red (negative)
        "#F5A6A6",  # Light red
        "#FFFFFF",  # White (neutral)
        "#A6D5F5",  # Light blue
        "#1A5F9E",  # Deep blue (positive)
    ],
    N=256
)

# Scientific color palette for curves
CURVE_COLORS = {
    "noise_insertion": "#E74C3C",       # Vermillion red
    "representation_insertion": "#3498DB",  # Scientific blue
    "random_baseline": "#95A5A6",       # Gray
    "optimal": "#27AE60",               # Green
}

# Style settings
FONT_FAMILY = "Arial"
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9

# Apply global font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
})


# =============================================================================
# Token Heatmap Visualization
# =============================================================================

def plot_token_heatmap(
    tokens: List[str],
    scores: np.ndarray,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (14, 2),
    cmap: Optional[LinearSegmentedColormap] = None,
    normalize: bool = True,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    max_tokens_per_row: int = 25,
    highlight_positions: Optional[List[int]] = None,
    highlight_color: str = "#FFD700",
    font_size: int = 9,
    box_height: float = 0.6,
    box_spacing: float = 0.05,
    text_color_threshold: float = 0.6,
) -> plt.Figure:
    """
    Plot token-level attribution heatmap.

    Args:
        tokens: List of token strings
        scores: Importance scores for each token
        ax: Matplotlib axes (creates new figure if None)
        figsize: Figure size (width, height per row)
        cmap: Colormap for importance scores
        normalize: Whether to normalize scores to [0, 1]
        show_colorbar: Whether to show colorbar
        title: Plot title
        max_tokens_per_row: Maximum tokens per row before wrapping
        highlight_positions: Token positions to highlight with border
        highlight_color: Border color for highlighted tokens
        font_size: Font size for tokens
        box_height: Height of each token box
        box_spacing: Spacing between token boxes
        text_color_threshold: Threshold for switching text to white

    Returns:
        Matplotlib figure
    """
    if cmap is None:
        cmap = ATTRIBUTION_CMAP

    scores = np.asarray(scores).flatten()
    n_tokens = len(tokens)

    if len(scores) != n_tokens:
        raise ValueError(f"Mismatch: {n_tokens} tokens, {len(scores)} scores")

    # Normalize scores
    if normalize:
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s > 1e-10:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores)
    else:
        norm_scores = np.clip(scores, 0, 1)

    # Calculate layout
    n_rows = int(np.ceil(n_tokens / max_tokens_per_row))

    # Create figure if needed
    if ax is None:
        fig_height = figsize[1] * n_rows + (0.5 if show_colorbar else 0)
        fig, ax = plt.subplots(figsize=(figsize[0], fig_height))
    else:
        fig = ax.figure

    # Set up normalization for colormap
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Calculate box widths based on token lengths
    # Use monospace-like width calculation
    token_widths = [max(len(t.replace("Ġ", "").replace("▁", "")), 1) * 0.08 + 0.15
                    for t in tokens]

    # Draw tokens row by row
    y_offset = n_rows - 1
    x_offset = 0
    row_start_idx = 0

    for i, (token, score, width) in enumerate(zip(tokens, norm_scores, token_widths)):
        # Check if we need to wrap to next row
        if i > row_start_idx and x_offset + width > figsize[0] - 0.5:
            y_offset -= 1
            x_offset = 0
            row_start_idx = i

        # Get color
        color = cmap(score)

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x_offset, y_offset * (box_height + 0.3)),
            width - box_spacing,
            box_height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color,
            edgecolor="#CCCCCC",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Highlight border if needed
        if highlight_positions and i in highlight_positions:
            highlight_rect = mpatches.FancyBboxPatch(
                (x_offset, y_offset * (box_height + 0.3)),
                width - box_spacing,
                box_height,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor="none",
                edgecolor=highlight_color,
                linewidth=2,
            )
            ax.add_patch(highlight_rect)

        # Clean token for display
        display_token = token.replace("Ġ", " ").replace("▁", " ")
        if display_token.startswith(" "):
            display_token = "·" + display_token[1:]

        # Choose text color based on background brightness
        text_color = "white" if score > text_color_threshold else "black"

        # Add text
        ax.text(
            x_offset + (width - box_spacing) / 2,
            y_offset * (box_height + 0.3) + box_height / 2,
            display_token,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            fontfamily="monospace",
        )

        x_offset += width

    # Set axis properties
    ax.set_xlim(-0.1, figsize[0])
    ax.set_ylim(-0.2, n_rows * (box_height + 0.3) + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)

    # Add colorbar
    if show_colorbar:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Importance Score", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout(rect=[0, 0, 0.9, 1] if show_colorbar else None)

    return fig


# =============================================================================
# AUC Curve Visualization
# =============================================================================

def plot_auc_curves(
    noise_curve_x: np.ndarray,
    noise_curve_y: np.ndarray,
    repr_curve_x: np.ndarray,
    repr_curve_y: np.ndarray,
    noise_auc: float,
    repr_auc: float,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot both AUC curves side by side.

    Args:
        noise_curve_x: X values for noise insertion curve
        noise_curve_y: Y values for noise insertion curve
        repr_curve_x: X values for representation insertion curve
        repr_curve_y: Y values for representation insertion curve
        noise_auc: Noise insertion AUC value
        repr_auc: Representation insertion AUC value
        ax: Matplotlib axes (creates new figure with 2 subplots if None)
        figsize: Figure size
        title: Overall figure title

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax.figure
        axes = [ax, ax]  # Single axis mode not recommended

    # Common style
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Left: Noise Insertion AUC ---
    ax1 = axes[0]

    # Fill area under curve
    ax1.fill_between(
        noise_curve_x, noise_curve_y,
        alpha=0.3,
        color=CURVE_COLORS["noise_insertion"]
    )

    # Plot curve
    ax1.plot(
        noise_curve_x, noise_curve_y,
        color=CURVE_COLORS["noise_insertion"],
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=f"AUC = {noise_auc:.3f}"
    )

    ax1.set_xlabel("Fraction of Tokens Corrupted", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Cosine Similarity", fontsize=LABEL_SIZE)
    ax1.set_title("Noise Insertion (Lower AUC = Better)", fontsize=TITLE_SIZE, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(min(noise_curve_y) - 0.05, 1.05)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Right: Representation Insertion AUC ---
    ax2 = axes[1]

    # Fill area under curve
    ax2.fill_between(
        repr_curve_x, repr_curve_y,
        alpha=0.3,
        color=CURVE_COLORS["representation_insertion"]
    )

    # Plot curve
    ax2.plot(
        repr_curve_x, repr_curve_y,
        color=CURVE_COLORS["representation_insertion"],
        linewidth=2.5,
        marker="s",
        markersize=5,
        label=f"AUC = {repr_auc:.3f}"
    )

    ax2.set_xlabel("Fraction of Tokens Restored", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Hidden State Recovery", fontsize=LABEL_SIZE)
    ax2.set_title("Representation Insertion (Higher AUC = Better)", fontsize=TITLE_SIZE, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=LEGEND_SIZE, framealpha=0.9)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Overall title
    if title:
        fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig


# =============================================================================
# Combined Visualization
# =============================================================================

@dataclass
class SampleVisualizationData:
    """Container for sample visualization data."""
    tokens: List[str]
    importance_scores: np.ndarray
    noise_curve_x: np.ndarray
    noise_curve_y: np.ndarray
    repr_curve_x: np.ndarray
    repr_curve_y: np.ndarray
    noise_auc: float
    repr_auc: float
    text: Optional[str] = None
    method_name: Optional[str] = None
    target_pos: Optional[int] = None

    @classmethod
    def from_eval_results(
        cls,
        tokens: List[str],
        importance_scores: np.ndarray,
        noise_result: Any,  # TokenAUCResult
        repr_result: Any,   # TokenAUCResult
        text: Optional[str] = None,
        method_name: Optional[str] = None,
    ) -> "SampleVisualizationData":
        """Create from evaluation results."""
        return cls(
            tokens=tokens,
            importance_scores=np.asarray(importance_scores),
            noise_curve_x=noise_result.curve_x,
            noise_curve_y=noise_result.curve_y,
            repr_curve_x=repr_result.curve_x,
            repr_curve_y=repr_result.curve_y,
            noise_auc=noise_result.auc,
            repr_auc=repr_result.auc,
            text=text,
            method_name=method_name,
        )


def plot_sample_attribution(
    data: Union[SampleVisualizationData, Dict[str, Any]],
    figsize: Tuple[float, float] = (14, 8),
    output_path: Optional[str] = None,
    dpi: int = 150,
    show_colorbar: bool = True,
    max_tokens_per_row: int = 25,
    highlight_top_k: Optional[int] = 5,
) -> plt.Figure:
    """
    Create complete visualization for a single sample.

    Combines token heatmap and dual AUC curves in a single figure.

    Args:
        data: Visualization data (SampleVisualizationData or dict)
        figsize: Figure size
        output_path: Path to save figure (if None, returns figure)
        dpi: Resolution for saved figure
        show_colorbar: Whether to show colorbar
        max_tokens_per_row: Max tokens per row in heatmap
        highlight_top_k: Highlight top-k important tokens

    Returns:
        Matplotlib figure
    """
    # Convert dict to dataclass if needed
    if isinstance(data, dict):
        data = SampleVisualizationData(**data)

    # Calculate heatmap height based on token count
    n_rows = int(np.ceil(len(data.tokens) / max_tokens_per_row))
    heatmap_height_ratio = max(1.5, n_rows * 0.8)

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[heatmap_height_ratio, 2],
        hspace=0.35
    )

    # --- Top: Token Heatmap ---
    ax_heatmap = fig.add_subplot(gs[0])

    # Get positions of top-k tokens to highlight
    highlight_positions = None
    if highlight_top_k:
        top_k_indices = np.argsort(-data.importance_scores)[:highlight_top_k]
        highlight_positions = list(top_k_indices)

    # Create title
    title = "Token Attribution Heatmap"
    if data.method_name:
        title += f" ({data.method_name})"

    # Plot heatmap on the axis
    plot_token_heatmap_on_ax(
        ax=ax_heatmap,
        tokens=data.tokens,
        scores=data.importance_scores,
        highlight_positions=highlight_positions,
        title=title,
        max_tokens_per_row=max_tokens_per_row,
    )

    # --- Bottom: AUC Curves ---
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3)
    ax_noise = fig.add_subplot(gs_bottom[0])
    ax_repr = fig.add_subplot(gs_bottom[1])

    plot_single_auc_curve(
        ax=ax_noise,
        curve_x=data.noise_curve_x,
        curve_y=data.noise_curve_y,
        auc_value=data.noise_auc,
        curve_type="noise_insertion",
    )

    plot_single_auc_curve(
        ax=ax_repr,
        curve_x=data.repr_curve_x,
        curve_y=data.repr_curve_y,
        auc_value=data.repr_auc,
        curve_type="representation_insertion",
    )

    # Add colorbar for heatmap
    if show_colorbar:
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=ATTRIBUTION_CMAP, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Importance", fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Adjust layout
    plt.subplots_adjust(right=0.9)

    # Save or return
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {output_path}")
        plt.close(fig)
        return None

    return fig


def plot_token_heatmap_on_ax(
    ax: plt.Axes,
    tokens: List[str],
    scores: np.ndarray,
    highlight_positions: Optional[List[int]] = None,
    title: Optional[str] = None,
    max_tokens_per_row: int = 25,
    font_size: int = 9,
    box_height: float = 0.5,
    text_color_threshold: float = 0.6,
) -> None:
    """
    Plot token heatmap on given axes.

    Internal helper function for combined visualization.
    """
    cmap = ATTRIBUTION_CMAP
    scores = np.asarray(scores).flatten()
    n_tokens = len(tokens)

    # Normalize scores
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s > 1e-10:
        norm_scores = (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(scores)

    # Calculate token widths
    token_widths = [max(len(t.replace("Ġ", "").replace("▁", "")), 1) * 0.06 + 0.12
                    for t in tokens]

    # Calculate layout
    row_width = 0.95  # Normalized width
    current_x = 0.02
    current_row = 0
    row_starts = [0]

    for i, width in enumerate(token_widths):
        if current_x + width > row_width and i > row_starts[-1]:
            current_row += 1
            row_starts.append(i)
            current_x = 0.02
        current_x += width

    n_rows = current_row + 1

    # Draw tokens
    row_idx = 0
    x_pos = 0.02

    for i, (token, score, width) in enumerate(zip(tokens, norm_scores, token_widths)):
        # Check for new row
        if row_idx < len(row_starts) - 1 and i == row_starts[row_idx + 1]:
            row_idx += 1
            x_pos = 0.02

        y_pos = 1 - (row_idx + 1) / (n_rows + 0.5)

        # Get color
        color = cmap(score)

        # Draw box
        rect = mpatches.FancyBboxPatch(
            (x_pos, y_pos),
            width - 0.005,
            box_height / n_rows,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="#CCCCCC",
            linewidth=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Highlight border
        if highlight_positions and i in highlight_positions:
            highlight_rect = mpatches.FancyBboxPatch(
                (x_pos, y_pos),
                width - 0.005,
                box_height / n_rows,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="none",
                edgecolor="#FFD700",
                linewidth=2,
                transform=ax.transAxes,
            )
            ax.add_patch(highlight_rect)

        # Clean token for display
        display_token = token.replace("Ġ", " ").replace("▁", " ")
        if display_token.startswith(" "):
            display_token = "·" + display_token[1:]

        # Text color
        text_color = "white" if score > text_color_threshold else "black"

        # Add text
        ax.text(
            x_pos + (width - 0.005) / 2,
            y_pos + box_height / n_rows / 2,
            display_token,
            ha="center",
            va="center",
            fontsize=font_size,
            color=text_color,
            fontfamily="monospace",
            transform=ax.transAxes,
        )

        x_pos += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)


def plot_single_auc_curve(
    ax: plt.Axes,
    curve_x: np.ndarray,
    curve_y: np.ndarray,
    auc_value: float,
    curve_type: str = "noise_insertion",
) -> None:
    """
    Plot single AUC curve on given axes.

    Args:
        ax: Matplotlib axes
        curve_x: X values
        curve_y: Y values
        auc_value: AUC value to display
        curve_type: "noise_insertion" or "representation_insertion"
    """
    is_noise = curve_type == "noise_insertion"
    color = CURVE_COLORS["noise_insertion" if is_noise else "representation_insertion"]

    # Fill under curve
    ax.fill_between(curve_x, curve_y, alpha=0.3, color=color)

    # Plot curve
    ax.plot(
        curve_x, curve_y,
        color=color,
        linewidth=2.5,
        marker="o" if is_noise else "s",
        markersize=5,
        label=f"AUC = {auc_value:.3f}"
    )

    # Labels
    if is_noise:
        ax.set_xlabel("Fraction Corrupted", fontsize=LABEL_SIZE)
        ax.set_ylabel("Cosine Similarity", fontsize=LABEL_SIZE)
        ax.set_title("Noise Insertion AUC (↓)", fontsize=TITLE_SIZE, fontweight="bold")
    else:
        ax.set_xlabel("Fraction Restored", fontsize=LABEL_SIZE)
        ax.set_ylabel("Hidden State Recovery", fontsize=LABEL_SIZE)
        ax.set_title("Representation Insertion AUC (↑)", fontsize=TITLE_SIZE, fontweight="bold")

    ax.legend(loc="upper right" if is_noise else "lower right", fontsize=LEGEND_SIZE)
    ax.set_xlim(-0.02, 1.02)
    y_min = min(curve_y) - 0.05 if is_noise else -0.05
    ax.set_ylim(y_min, 1.05)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")


# =============================================================================
# Convenience Function for Direct Use
# =============================================================================

def visualize_attribution(
    model,
    tokenizer,
    input_ids,
    importance_scores: np.ndarray,
    target_pos: int = -1,
    target_token_id: Optional[int] = None,
    method_name: str = "Attribution",
    output_path: Optional[str] = None,
    steps: int = 10,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 150,
) -> Optional[plt.Figure]:
    """
    Complete attribution visualization pipeline.

    Computes AUC metrics and creates visualization.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_ids: Input token IDs (tensor or array)
        importance_scores: Attribution scores per token
        target_pos: Target position for evaluation
        target_token_id: Target token ID (auto-detected if None)
        method_name: Name of attribution method
        output_path: Path to save figure
        steps: Number of steps for AUC computation
        figsize: Figure size
        dpi: Resolution

    Returns:
        Matplotlib figure if output_path is None
    """
    import torch
    from ..eval import noise_insertion_auc_token, representation_insertion_auc_token

    # Ensure tensor
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    # Auto-detect target token if needed
    if target_token_id is None:
        with torch.no_grad():
            logits = model(input_ids).logits
            target_token_id = logits[0, target_pos].argmax().item()

    # Compute AUC metrics
    noise_result = noise_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=importance_scores,
        target_pos=target_pos,
        target_token_id=target_token_id,
        steps=steps,
        seed=42,  # Fix random seed for reproducibility
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    repr_result = representation_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=importance_scores,
        base_token_id=pad_token_id,
        steps=steps,
    )

    # Create visualization data
    vis_data = SampleVisualizationData.from_eval_results(
        tokens=tokens,
        importance_scores=importance_scores,
        noise_result=noise_result,
        repr_result=repr_result,
        method_name=method_name,
    )

    # Create visualization
    return plot_sample_attribution(
        data=vis_data,
        figsize=figsize,
        output_path=output_path,
        dpi=dpi,
    )
