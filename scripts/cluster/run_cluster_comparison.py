#!/usr/bin/env python
"""
Run cluster comparison experiments.

Compares pattern-based cluster detection vs random cluster selection
using attention rollout as the information flow metric.

Usage:
    python scripts/cluster/run_cluster_comparison.py
    python scripts/cluster/run_cluster_comparison.py --models phi-2 Qwen3-4B --sample-size 50
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def run_single_experiment(
    model_name: str,
    task_name: str = "agnews",
    flow_metric: str = "attention_rollout",
    use_random_cluster: bool = False,
    demonstration_shot: int = 1,
    sample_size: int = 100,
    seeds: List[int] = None,
    device: str = "cuda:0",
    save_dir: str = "results/cluster",
    num_clusters_per_label: int = 2,
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    from experiments.cluster.run import run_cluster_experiment

    if seeds is None:
        seeds = [42]  # Use single seed for consistency across models

    cluster_type = "random" if use_random_cluster else "pattern"
    print(f"\n{'='*60}")
    print(f"Running: {model_name} | {task_name} | {flow_metric} | {cluster_type} cluster")
    print(f"{'='*60}")

    results = run_cluster_experiment(
        task_name=task_name,
        model_name=model_name,
        demonstration_shot=demonstration_shot,
        sample_size=sample_size,
        seeds=seeds,
        device=device,
        flow_metric=flow_metric,
        save_dir=save_dir,
        use_random_cluster=use_random_cluster,
        num_clusters_per_label=num_clusters_per_label,
    )

    return results


def run_all_experiments(
    models: List[str],
    task_name: str = "agnews",
    flow_metric: str = "attention_rollout",
    demonstration_shot: int = 1,
    sample_size: int = 100,
    seed: int = 42,
    device: str = "cuda:0",
    save_dir: str = "results/cluster",
) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments on all models with both pattern and random clusters.

    Returns:
        Dictionary mapping (model, cluster_type) to results
    """
    all_results = {}
    seeds = [seed]

    for model_name in models:
        # Run with pattern cluster
        try:
            pattern_results = run_single_experiment(
                model_name=model_name,
                task_name=task_name,
                flow_metric=flow_metric,
                use_random_cluster=False,
                demonstration_shot=demonstration_shot,
                sample_size=sample_size,
                seeds=seeds,
                device=device,
                save_dir=save_dir,
            )
            all_results[(model_name, "pattern")] = pattern_results
        except Exception as e:
            print(f"Error running pattern cluster experiment for {model_name}: {e}")
            all_results[(model_name, "pattern")] = {"error": str(e)}

        # Run with random cluster
        try:
            random_results = run_single_experiment(
                model_name=model_name,
                task_name=task_name,
                flow_metric=flow_metric,
                use_random_cluster=True,
                demonstration_shot=demonstration_shot,
                sample_size=sample_size,
                seeds=seeds,
                device=device,
                save_dir=save_dir,
            )
            all_results[(model_name, "random")] = random_results
        except Exception as e:
            print(f"Error running random cluster experiment for {model_name}: {e}")
            all_results[(model_name, "random")] = {"error": str(e)}

    return all_results


def print_summary(all_results: Dict[str, Dict[str, Any]], models: List[str]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Cluster':<10} {'Mean S_a':<12} {'Mean S_o':<12} {'Mean S_w':<12}")
    print("-"*80)

    for model_name in models:
        for cluster_type in ["pattern", "random"]:
            key = (model_name, cluster_type)
            if key in all_results:
                result = all_results[key]
                if "error" in result:
                    print(f"{model_name:<20} {cluster_type:<10} ERROR: {result['error'][:40]}")
                else:
                    stats = result["statistics"]
                    mean_sa = np.mean(stats["mean_S_a"])
                    mean_so = np.mean(stats["mean_S_o"])
                    mean_sw = np.mean(stats["mean_S_w"])
                    print(f"{model_name:<20} {cluster_type:<10} {mean_sa:<12.4f} {mean_so:<12.4f} {mean_sw:<12.4f}")

    print("="*80)


def save_combined_results(
    all_results: Dict[str, Dict[str, Any]],
    save_path: str,
    models: List[str],
    config: Dict[str, Any],
):
    """Save combined results to a single file."""
    # Convert tuple keys to string keys for JSON compatibility
    serializable_results = {}
    for (model, cluster_type), result in all_results.items():
        key = f"{model}_{cluster_type}"
        if "error" in result:
            serializable_results[key] = {"error": result["error"]}
        else:
            serializable_results[key] = {
                "statistics": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in result["statistics"].items()
                },
                "config": result.get("config", {}),
            }

    output = {
        "results": serializable_results,
        "models": models,
        "config": config,
    }

    # Save as pickle for full fidelity
    pickle_path = save_path.replace(".json", ".pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({"results": all_results, "models": models, "config": config}, f)
    print(f"Results saved to: {pickle_path}")

    # Also save summary as JSON
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Summary saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cluster comparison experiments"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to run experiments on (default: all available)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="agnews",
        help="Task name (default: agnews)",
    )
    parser.add_argument(
        "--flow-metric",
        type=str,
        default="attention_rollout",
        help="Flow metric to use (default: attention_rollout)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (single seed for fair comparison)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/cluster",
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Default models (all available)
    default_models = [
        "phi-2",
        "Qwen3-4B",
        "llama-3.2-1b",
        "tinyllama-1.1b",
        "llama2-7b",
        "llama3-8b-instruct",
    ]

    models = args.models if args.models else default_models

    # Check which models exist
    model_dir = project_root / "model"
    available_models = []
    for model in models:
        if (model_dir / model).exists():
            available_models.append(model)
        else:
            print(f"Warning: Model {model} not found in {model_dir}, skipping...")

    if not available_models:
        print("Error: No valid models found!")
        sys.exit(1)

    print(f"Running experiments on {len(available_models)} models: {available_models}")
    print(f"Task: {args.task}")
    print(f"Flow metric: {args.flow_metric}")
    print(f"Shot: {args.shot}")
    print(f"Sample size: {args.sample_size}")
    print(f"Seed: {args.seed}")

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Run all experiments
    all_results = run_all_experiments(
        models=available_models,
        task_name=args.task,
        flow_metric=args.flow_metric,
        demonstration_shot=args.shot,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
    )

    # Print summary
    print_summary(all_results, available_models)

    # Save combined results
    config = {
        "task": args.task,
        "flow_metric": args.flow_metric,
        "shot": args.shot,
        "sample_size": args.sample_size,
        "seed": args.seed,
    }
    save_combined_results(
        all_results,
        f"{args.save_dir}/cluster_comparison_{args.task}_{args.flow_metric}.json",
        available_models,
        config,
    )


if __name__ == "__main__":
    main()
