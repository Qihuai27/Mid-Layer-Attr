#!/usr/bin/env python3
"""
Top-K Token Comparison Experiment: Comparing Attribution Methods with Greedy Optimal Baseline.

This script compares the top-k important tokens identified by various attribution methods
against the greedy optimal baseline (which provides the theoretical upper bound).

Experiment Design:
- Compare top-2 tokens from each method vs greedy optimal (ground truth)
- Evaluate on IOI dataset with 100 samples
- Metrics:
  - Exact Match Rate: Both top-2 tokens match in order
  - Set Match Rate: Both top-2 tokens match (order-agnostic)
  - Top-1 Match Rate: The most important token matches
  - Partial Match Rate: At least one of top-2 matches
- Visualize 3 representative samples

Usage:
    python scripts/attribution/run_topk_comparison.py --model phi-2
    python scripts/attribution/run_topk_comparison.py --model Qwen3-4B --max-samples 50
    python scripts/attribution/run_topk_comparison.py --model phi-2 --visualize-samples 0 1 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.base import load_model_and_tokenizer
from src.attribution import (
    AttentionRollout,
    DePass,
    IntegratedGradients,
    GradInput,
    GradSAM,
    TokenShapley,
    GreedyOptimalAttribution,
    InputCausalAttribution,
)


# =============================================================================
# Configuration
# =============================================================================

# Attribution methods to compare (excluding midlayer, midlayer_v2)
METHODS_CONFIG = {
    "attention_rollout": {
        "class": AttentionRollout,
        "params": {"residual_weight": 0.5},
        "display_name": "Attention Rollout",
        "color": "#3498DB",
    },
    "depass": {
        "class": DePass,
        "params": {"mlp_softmax_temp": 0.1},
        "display_name": "DePass",
        "color": "#E74C3C",
    },
    "integrated_gradients": {
        "class": IntegratedGradients,
        "params": {"steps": 20},
        "display_name": "Integrated Gradients",
        "color": "#2ECC71",
    },
    "grad_input": {
        "class": GradInput,
        "params": {},
        "display_name": "Grad × Input",
        "color": "#9B59B6",
    },
    "grad_sam": {
        "class": GradSAM,
        "params": {},
        "display_name": "GradSAM",
        "color": "#F39C12",
    },
    "shapley": {
        "class": TokenShapley,
        "params": {"n_samples": 50},
        "display_name": "Token Shapley",
        "color": "#1ABC9C",
    },
    "input_causal": {
        "class": InputCausalAttribution,
        "params": {"baseline_mode": "mean"},
        "display_name": "Input Causal",
        "color": "#E91E63",
    },
}

# Greedy optimal as baseline
BASELINE_CONFIG = {
    "class": GreedyOptimalAttribution,
    "params": {"baseline_mode": "mean", "verbose": False},
    "display_name": "Greedy Optimal",
    "color": "#27AE60",
}

# Scientific colormap
ATTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "attribution",
    ["#FFFFFF", "#E8F4FD", "#B3D9F7", "#5BA3D9", "#1A5F9E", "#0D3B66"],
    N=256
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_idx: int
    prompt: str
    answer: str
    tokens: List[str]
    target_pos: int
    target_token_id: int
    greedy_top2: List[int]
    greedy_scores: List[float]
    method_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_method_result(
        self,
        method_name: str,
        top2: List[int],
        scores: np.ndarray,
        exact_match: bool,
        set_match: bool,
        top1_match: bool,
        partial_match: bool,
    ):
        """Add result for a specific method."""
        self.method_results[method_name] = {
            "top2": top2,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "exact_match": exact_match,
            "set_match": set_match,
            "top1_match": top1_match,
            "partial_match": partial_match,
        }


@dataclass
class ExperimentResults:
    """Aggregated experiment results."""
    model_name: str
    dataset: str
    num_samples: int
    top_k: int
    timestamp: str
    sample_results: List[SampleResult] = field(default_factory=list)

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute match rate statistics for each method."""
        stats = {}

        for method_name in METHODS_CONFIG.keys():
            exact_matches = []
            set_matches = []
            top1_matches = []
            partial_matches = []

            for sample in self.sample_results:
                if method_name in sample.method_results:
                    result = sample.method_results[method_name]
                    exact_matches.append(result["exact_match"])
                    set_matches.append(result["set_match"])
                    top1_matches.append(result["top1_match"])
                    partial_matches.append(result["partial_match"])

            if exact_matches:
                stats[method_name] = {
                    "exact_match_rate": np.mean(exact_matches) * 100,
                    "set_match_rate": np.mean(set_matches) * 100,
                    "top1_match_rate": np.mean(top1_matches) * 100,
                    "partial_match_rate": np.mean(partial_matches) * 100,
                    "num_samples": len(exact_matches),
                }

        return stats


# =============================================================================
# Core Functions
# =============================================================================

def load_ioi_dataset(max_samples: int = 100) -> List[Dict]:
    """Load IOI dataset."""
    dataset_path = project_root / "datasets" / "ioi_data.json"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    return data[:max_samples]


def get_top_k_indices(scores: np.ndarray, k: int, target_pos: int) -> List[int]:
    """Get indices of top-k tokens by importance score.

    Args:
        scores: Importance scores for each token
        k: Number of top tokens to return
        target_pos: Target position (tokens after this are excluded)

    Returns:
        List of indices for top-k tokens
    """
    # Only consider tokens up to target_pos
    valid_scores = scores[:target_pos + 1].copy()

    # Get top-k indices
    top_k = np.argsort(-valid_scores)[:k]

    return top_k.tolist()


def compare_top_k(
    method_top_k: List[int],
    greedy_top_k: List[int],
) -> Tuple[bool, bool, bool, bool]:
    """Compare method's top-k with greedy optimal's top-k.

    Returns:
        (exact_match, set_match, top1_match, partial_match)
    """
    # Exact match: same tokens in same order
    exact_match = method_top_k == greedy_top_k

    # Set match: same tokens, order doesn't matter
    set_match = set(method_top_k) == set(greedy_top_k)

    # Top-1 match: first token matches
    top1_match = method_top_k[0] == greedy_top_k[0] if method_top_k and greedy_top_k else False

    # Partial match: at least one token overlaps
    partial_match = len(set(method_top_k) & set(greedy_top_k)) > 0

    return exact_match, set_match, top1_match, partial_match


def run_attribution_method(
    method,
    model,
    input_ids: torch.Tensor,
    target_pos: int,
    target_token_id: int,
) -> np.ndarray:
    """Run attribution method and return scores."""
    try:
        result = method.attribute(
            model=model,
            input_ids=input_ids,
            target_pos=target_pos,
            target_token_id=target_token_id,
        )
        return result.scores
    except Exception as e:
        print(f"Warning: Method {method.name} failed: {e}")
        return None


# =============================================================================
# Visualization
# =============================================================================

def visualize_sample_comparison(
    sample_result: SampleResult,
    output_dir: Path,
    sample_idx: int,
    tokenizer,
) -> None:
    """Create detailed visualization for a single sample.

    Shows:
    - Token heatmaps for greedy optimal and each method
    - Comparison of top-2 tokens
    - Match statistics
    """
    n_methods = len(sample_result.method_results) + 1  # +1 for greedy

    fig = plt.figure(figsize=(16, 4 + 2 * n_methods))

    # Title
    prompt_display = sample_result.prompt[:80] + "..." if len(sample_result.prompt) > 80 else sample_result.prompt
    fig.suptitle(
        f"Sample {sample_idx}: Top-2 Token Comparison\n\"{prompt_display}\"",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(n_methods + 1, 2, height_ratios=[1] + [1] * n_methods, width_ratios=[3, 1])

    tokens = sample_result.tokens
    n_tokens = len(tokens)
    greedy_scores = np.array(sample_result.greedy_scores)
    greedy_top2 = sample_result.greedy_top2

    # Normalize greedy scores
    if greedy_scores.max() > greedy_scores.min():
        norm_greedy = (greedy_scores - greedy_scores.min()) / (greedy_scores.max() - greedy_scores.min())
    else:
        norm_greedy = np.zeros_like(greedy_scores)

    # Row 0: Greedy Optimal (baseline)
    ax_greedy = fig.add_subplot(gs[0, 0])
    _plot_token_bar(ax_greedy, tokens, norm_greedy, greedy_top2,
                   "Greedy Optimal (Ground Truth)", BASELINE_CONFIG["color"])

    ax_greedy_info = fig.add_subplot(gs[0, 1])
    ax_greedy_info.axis("off")
    top2_tokens = [tokens[i] for i in greedy_top2]
    ax_greedy_info.text(0.1, 0.5, f"Top-2: {top2_tokens}", fontsize=10, va="center",
                       fontfamily="monospace", wrap=True)

    # Rows 1+: Each method
    row = 1
    for method_name, config in METHODS_CONFIG.items():
        if method_name not in sample_result.method_results:
            continue

        result = sample_result.method_results[method_name]
        scores = np.array(result["scores"])
        method_top2 = result["top2"]

        # Normalize scores
        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.zeros_like(scores)

        ax_method = fig.add_subplot(gs[row, 0])
        _plot_token_bar(ax_method, tokens, norm_scores, method_top2,
                       config["display_name"], config["color"])

        # Info panel
        ax_info = fig.add_subplot(gs[row, 1])
        ax_info.axis("off")

        method_top2_tokens = [tokens[i] for i in method_top2]
        match_symbols = []
        if result["exact_match"]:
            match_symbols.append("✓ Exact")
        elif result["set_match"]:
            match_symbols.append("≈ Set")
        elif result["top1_match"]:
            match_symbols.append("① Top-1")
        elif result["partial_match"]:
            match_symbols.append("½ Partial")
        else:
            match_symbols.append("✗ None")

        info_text = f"Top-2: {method_top2_tokens}\n{' '.join(match_symbols)}"
        color = "#27AE60" if result["exact_match"] else ("#F39C12" if result["set_match"] else "#E74C3C")
        ax_info.text(0.1, 0.5, info_text, fontsize=9, va="center",
                    fontfamily="monospace", color=color, wrap=True)

        row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / f"sample_{sample_idx}_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved visualization: {output_path}")


def _plot_token_bar(
    ax: plt.Axes,
    tokens: List[str],
    scores: np.ndarray,
    top2_indices: List[int],
    title: str,
    color: str,
) -> None:
    """Plot horizontal bar chart for token scores with top-2 highlighted."""
    n_tokens = min(len(tokens), len(scores))

    # Limit display for readability
    max_display = 30
    if n_tokens > max_display:
        # Show tokens around important ones
        important_range = set()
        for idx in top2_indices:
            for j in range(max(0, idx - 2), min(n_tokens, idx + 3)):
                important_range.add(j)

        # Add start and end tokens
        for j in range(min(5, n_tokens)):
            important_range.add(j)
        for j in range(max(0, n_tokens - 3), n_tokens):
            important_range.add(j)

        display_indices = sorted(important_range)[:max_display]
    else:
        display_indices = list(range(n_tokens))

    display_tokens = [tokens[i].replace("Ġ", "·").replace("▁", "·")[:8] for i in display_indices]
    display_scores = [scores[i] for i in display_indices]

    colors = []
    for i in display_indices:
        if i == top2_indices[0]:
            colors.append("#E74C3C")  # Top-1: red
        elif len(top2_indices) > 1 and i == top2_indices[1]:
            colors.append("#F39C12")  # Top-2: orange
        else:
            colors.append(color)

    y_pos = np.arange(len(display_tokens))
    ax.barh(y_pos, display_scores, color=colors, alpha=0.8, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_tokens, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def plot_summary_statistics(
    stats: Dict[str, Dict[str, float]],
    output_dir: Path,
    model_name: str,
) -> None:
    """Plot summary bar chart of match rates."""
    methods = list(stats.keys())
    n_methods = len(methods)

    # Prepare data
    exact_rates = [stats[m]["exact_match_rate"] for m in methods]
    set_rates = [stats[m]["set_match_rate"] for m in methods]
    top1_rates = [stats[m]["top1_match_rate"] for m in methods]
    partial_rates = [stats[m]["partial_match_rate"] for m in methods]

    # Display names
    display_names = [METHODS_CONFIG[m]["display_name"] for m in methods]

    x = np.arange(n_methods)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - 1.5*width, exact_rates, width, label="Exact Match", color="#27AE60", alpha=0.9)
    bars2 = ax.bar(x - 0.5*width, set_rates, width, label="Set Match", color="#3498DB", alpha=0.9)
    bars3 = ax.bar(x + 0.5*width, top1_rates, width, label="Top-1 Match", color="#F39C12", alpha=0.9)
    bars4 = ax.bar(x + 1.5*width, partial_rates, width, label="Partial Match", color="#9B59B6", alpha=0.9)

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel("Match Rate (%)", fontsize=11)
    ax.set_xlabel("Attribution Method", fontsize=11)
    ax.set_title(
        f"Top-2 Token Match Rates vs Greedy Optimal Baseline\n(Model: {model_name}, IOI Dataset)",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()

    output_path = output_dir / f"summary_match_rates_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")


def plot_detailed_comparison_table(
    stats: Dict[str, Dict[str, float]],
    output_dir: Path,
    model_name: str,
) -> None:
    """Create a detailed comparison table as figure."""
    methods = list(stats.keys())
    display_names = [METHODS_CONFIG[m]["display_name"] for m in methods]

    # Data for table
    columns = ["Method", "Exact\n(%)", "Set\n(%)", "Top-1\n(%)", "Partial\n(%)"]
    cell_data = []

    for m in methods:
        s = stats[m]
        cell_data.append([
            METHODS_CONFIG[m]["display_name"],
            f"{s['exact_match_rate']:.1f}",
            f"{s['set_match_rate']:.1f}",
            f"{s['top1_match_rate']:.1f}",
            f"{s['partial_match_rate']:.1f}",
        ])

    # Sort by exact match rate (descending)
    cell_data.sort(key=lambda x: float(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(methods)))
    ax.axis('off')

    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#3498DB'] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color cells based on values
    for i, row in enumerate(cell_data):
        for j in range(1, 5):
            val = float(row[j])
            if val >= 50:
                table[(i + 1, j)].set_facecolor('#D5F5E3')
            elif val >= 25:
                table[(i + 1, j)].set_facecolor('#FCF3CF')
            else:
                table[(i + 1, j)].set_facecolor('#FADBD8')

    ax.set_title(
        f"Top-2 Token Match Rates (%) - {model_name}\nCompared with Greedy Optimal Baseline",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    output_path = output_dir / f"comparison_table_{model_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison table: {output_path}")


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    model_name: str,
    max_samples: int = 100,
    top_k: int = 2,
    device: str = "cuda:0",
    visualize_samples: Optional[List[int]] = None,
    skip_slow_methods: bool = False,
) -> ExperimentResults:
    """Run the complete top-k comparison experiment."""

    print("=" * 70)
    print(f"Top-{top_k} Token Comparison Experiment")
    print(f"  Model: {model_name}")
    print(f"  Dataset: IOI ({max_samples} samples)")
    print(f"  Baseline: Greedy Optimal")
    print("=" * 70)

    # Setup output directory
    output_dir = project_root / "results" / "1-order"
    viz_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/4] Loading IOI dataset...")
    dataset = load_ioi_dataset(max_samples)
    print(f"  Loaded {len(dataset)} samples")

    # Load model
    print(f"\n[2/4] Loading model: {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    model.eval()
    print(f"  Model loaded on {device}")

    # Initialize methods
    print("\n[3/4] Initializing attribution methods...")

    baseline_method = BASELINE_CONFIG["class"](**BASELINE_CONFIG["params"])
    print(f"  Baseline: {BASELINE_CONFIG['display_name']}")

    methods = {}
    for name, config in METHODS_CONFIG.items():
        if skip_slow_methods and name in ["shapley"]:
            print(f"  Skipping slow method: {name}")
            continue
        try:
            methods[name] = config["class"](**config["params"])
            print(f"  Loaded: {config['display_name']}")
        except Exception as e:
            print(f"  Warning: Failed to load {name}: {e}")

    # Initialize results
    results = ExperimentResults(
        model_name=model_name,
        dataset="ioi",
        num_samples=len(dataset),
        top_k=top_k,
        timestamp=datetime.now().isoformat(),
    )

    # Default visualization samples
    if visualize_samples is None:
        visualize_samples = [0, len(dataset) // 2, len(dataset) - 1]

    # Run experiment
    print(f"\n[4/4] Running experiment on {len(dataset)} samples...")

    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        prompt = sample["prompt"]
        answer = sample["answer"]

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        T = input_ids.shape[1]
        target_pos = T - 1  # Last position

        # Get target token ID (the answer token)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        target_token_id = answer_ids[0] if answer_ids else None

        if target_token_id is None:
            # Fallback to model prediction
            with torch.no_grad():
                logits = model(input_ids).logits
                target_token_id = logits[0, target_pos].argmax().item()

        # Run greedy optimal baseline
        try:
            greedy_result = baseline_method.attribute(
                model=model,
                input_ids=input_ids,
                target_pos=target_pos,
                target_token_id=target_token_id,
            )
            greedy_scores = greedy_result.scores
            greedy_top2 = get_top_k_indices(greedy_scores, top_k, target_pos)
        except Exception as e:
            print(f"\n  Warning: Greedy optimal failed for sample {idx}: {e}")
            continue

        # Create sample result
        sample_result = SampleResult(
            sample_idx=idx,
            prompt=prompt,
            answer=answer,
            tokens=tokens,
            target_pos=target_pos,
            target_token_id=target_token_id,
            greedy_top2=greedy_top2,
            greedy_scores=greedy_scores.tolist(),
        )

        # Run each method
        for method_name, method in methods.items():
            try:
                scores = run_attribution_method(
                    method=method,
                    model=model,
                    input_ids=input_ids,
                    target_pos=target_pos,
                    target_token_id=target_token_id,
                )

                if scores is None:
                    continue

                method_top2 = get_top_k_indices(scores, top_k, target_pos)
                exact, set_m, top1, partial = compare_top_k(method_top2, greedy_top2)

                sample_result.add_method_result(
                    method_name=method_name,
                    top2=method_top2,
                    scores=scores,
                    exact_match=exact,
                    set_match=set_m,
                    top1_match=top1,
                    partial_match=partial,
                )
            except Exception as e:
                print(f"\n  Warning: {method_name} failed for sample {idx}: {e}")
                continue

        results.sample_results.append(sample_result)

        # Visualize if requested
        if idx in visualize_samples:
            print(f"\n  Creating visualization for sample {idx}...")
            visualize_sample_comparison(sample_result, viz_dir, idx, tokenizer)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare attribution methods' top-k tokens with greedy optimal baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi-2",
        help="Model name (e.g., phi-2, Qwen3-4B)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top tokens to compare",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        nargs="+",
        default=None,
        help="Sample indices to visualize (default: first, middle, last)",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow methods like Shapley",
    )
    args = parser.parse_args()

    # Run experiment
    results = run_experiment(
        model_name=args.model,
        max_samples=args.max_samples,
        top_k=args.top_k,
        device=args.device,
        visualize_samples=args.visualize_samples,
        skip_slow_methods=args.skip_slow,
    )

    # Compute and display statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    stats = results.compute_statistics()

    print(f"\nTop-{args.top_k} Token Match Rates (vs Greedy Optimal):\n")
    print(f"{'Method':<25} {'Exact':>10} {'Set':>10} {'Top-1':>10} {'Partial':>10}")
    print("-" * 65)

    # Sort by exact match rate
    sorted_methods = sorted(stats.keys(), key=lambda m: stats[m]["exact_match_rate"], reverse=True)

    for method_name in sorted_methods:
        s = stats[method_name]
        display_name = METHODS_CONFIG[method_name]["display_name"]
        print(f"{display_name:<25} {s['exact_match_rate']:>9.1f}% {s['set_match_rate']:>9.1f}% "
              f"{s['top1_match_rate']:>9.1f}% {s['partial_match_rate']:>9.1f}%")

    # Save results
    output_dir = project_root / "results" / "1-order"

    # Save JSON results
    results_dict = {
        "model_name": results.model_name,
        "dataset": results.dataset,
        "num_samples": results.num_samples,
        "top_k": results.top_k,
        "timestamp": results.timestamp,
        "statistics": stats,
        "samples": [
            {
                "sample_idx": s.sample_idx,
                "prompt": s.prompt,
                "answer": s.answer,
                "tokens": s.tokens,
                "target_pos": s.target_pos,
                "target_token_id": s.target_token_id,
                "greedy_top2": s.greedy_top2,
                "method_results": {
                    k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                    for k, v in s.method_results.items()
                },
            }
            for s in results.sample_results
        ],
    }

    results_path = output_dir / f"topk_comparison_{args.model}.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate summary plots
    print("\nGenerating summary plots...")
    plot_summary_statistics(stats, output_dir / "visualizations", args.model)
    plot_detailed_comparison_table(stats, output_dir / "visualizations", args.model)

    print("\n" + "=" * 70)
    print("Experiment completed!")
    print(f"Results directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
