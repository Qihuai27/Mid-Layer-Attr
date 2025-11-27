#!/usr/bin/env python
"""
Run attribution analysis experiment.

Usage:
    # Default (gradient saliency from paper)
    python scripts/run_attribution.py --task sst2 --model gpt2-xl --shot 1

    # Using attention sum
    python scripts/run_attribution.py --task sst2 --model gpt2-xl --flow-metric attention_sum

    # Using attention rollout
    python scripts/run_attribution.py --task sst2 --model gpt2-xl --flow-metric attention_rollout

Available flow metrics:
    - attention_sum: Direct attention weights summed across heads
    - gradient_saliency: |A ⊙ ∂L/∂A| (default, from paper)
    - attention_rollout: Cumulative attention across layers
    - attention_value_weighted: Attention × value vector norm
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.attribution.run import run_attribution_experiment
from experiments.attribution.statistics import FlowMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attribution analysis experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (gradient saliency)
  python scripts/run_attribution.py --task sst2 --model gpt2-xl

  # Using attention sum (fast, no gradients needed)
  python scripts/run_attribution.py --task sst2 --model gpt2-xl --flow-metric attention_sum

  # Using attention rollout (multi-layer cumulative)
  python scripts/run_attribution.py --task sst2 --model gpt2-xl --flow-metric attention_rollout
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        choices=["sst2", "agnews", "trec", "emo"],
        help="Task name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        help="Model name",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=1,
        help="Number of demonstrations per class",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of test samples",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    parser.add_argument(
        "--flow-metric",
        type=str,
        default="gradient_saliency",
        choices=[
            "attention_sum",
            "gradient_saliency",
            "attention_rollout",
            "attention_value_weighted",
            "causal_ablation",
            "attention_mask_ablation",
        ],
        help="Information flow metric to use (default: gradient_saliency)",
    )
    parser.add_argument(
        "--no-gradients",
        action="store_true",
        help="[Deprecated] Use --flow-metric attention_sum instead",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/attribution",
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle deprecated --no-gradients flag
    flow_metric = args.flow_metric
    if args.no_gradients and flow_metric == "gradient_saliency":
        flow_metric = "attention_sum"
        print("Warning: --no-gradients is deprecated. Using --flow-metric attention_sum")

    print(f"Running attribution analysis:")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Shot: {args.shot}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Flow metric: {flow_metric}")
    print()

    results = run_attribution_experiment(
        task_name=args.task,
        model_name=args.model,
        demonstration_shot=args.shot,
        sample_size=args.sample_size,
        seeds=args.seeds,
        device=args.device,
        flow_metric=flow_metric,
        save_dir=args.save_dir,
    )

    print("\nResults summary:")
    stats = results["statistics"]
    print(f"  Flow metric: {flow_metric}")
    print(f"  Mean S_wp across layers: {stats['mean_S_wp'].mean():.4f}")
    print(f"  Mean S_pq across layers: {stats['mean_S_pq'].mean():.4f}")
    print(f"  Mean S_ww across layers: {stats['mean_S_ww'].mean():.4f}")


if __name__ == "__main__":
    main()
