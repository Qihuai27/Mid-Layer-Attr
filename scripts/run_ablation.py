#!/usr/bin/env python
"""
Run causal ablation experiment.

Usage:
    python scripts/run_ablation.py --task sst2 --model gpt2-xl --mask-layers 5 --mask-pos first
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.ablation.run import run_ablation_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run causal ablation experiment")

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
        "--mask-layers",
        type=int,
        default=5,
        help="Number of layers to mask",
    )
    parser.add_argument(
        "--mask-pos",
        type=str,
        default="first",
        choices=["first", "last"],
        help="Position of layers to mask",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default="anchor",
        choices=["anchor", "non_anchor"],
        help="Type of intervention",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/ablation",
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running ablation experiment:")
    print(f"  Task: {args.task}")
    print(f"  Model: {args.model}")
    print(f"  Shot: {args.shot}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Mask layers: {args.mask_layers} ({args.mask_pos})")
    print(f"  Intervention: {args.intervention}")
    print()

    results = run_ablation_experiment(
        task_name=args.task,
        model_name=args.model,
        demonstration_shot=args.shot,
        sample_size=args.sample_size,
        seeds=args.seeds,
        device=args.device,
        mask_layer_num=args.mask_layers,
        mask_layer_pos=args.mask_pos,
        intervention_type=args.intervention,
        save_dir=args.save_dir,
    )

    print("\nResults summary:")
    agg = results["aggregated"]
    print(f"  Mean baseline accuracy: {agg['mean_baseline_accuracy']:.4f}")
    print(f"  Mean intervened accuracy: {agg['mean_intervened_accuracy']:.4f}")
    print(f"  Mean accuracy drop: {agg['mean_accuracy_drop']:.4f} (+/- {agg['std_accuracy_drop']:.4f})")


if __name__ == "__main__":
    main()
