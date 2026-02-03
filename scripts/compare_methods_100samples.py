#!/usr/bin/env python3
"""
Compare all attribution methods (including token-shapley) on first 100 samples.

Usage:
    python scripts/compare_methods_100samples.py --model Qwen3-4B
    python scripts/compare_methods_100samples.py --model Llama-3.2-3B-Instruct
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer
from src.eval import (
    noise_insertion_auc_token,
    representation_insertion_auc_token,
)


def load_attribution_results(result_path: str, max_samples: int = 100) -> List[Dict]:
    """Load attribution results from JSON file."""
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:max_samples]


def evaluate_single_sample(
    model,
    tokenizer,
    sample: Dict,
    device: str = "cuda",
    steps: int = 10,
) -> Dict[str, float]:
    """Evaluate a single sample with both AUC metrics."""
    prompt = sample["prompt"]
    scores = np.array(sample["attribution_scores"])
    target_pos = sample["target_pos"]
    target_token_id = sample["target_token_id"]

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Verify length matches
    if input_ids.shape[1] != len(scores):
        return None

    # Compute Noise Insertion AUC (lower is better)
    try:
        ni_result = noise_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            target_pos=target_pos,
            target_token_id=target_token_id,
            steps=steps,
            baseline_embed_mode="mean",
        )
        noise_auc = ni_result.auc
    except Exception as e:
        noise_auc = None

    # Compute Representation Insertion AUC (higher is better)
    try:
        ri_result = representation_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            base_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            steps=steps,
            layer_index=-1,
            distance_mode="cosine",
            position_weighting="linear",
        )
        repr_auc = ri_result.auc
    except Exception as e:
        repr_auc = None

    return {
        "case_id": sample["case_id"],
        "noise_insertion_auc": noise_auc,
        "representation_insertion_auc": repr_auc,
    }


def evaluate_method(
    model,
    tokenizer,
    result_path: str,
    device: str = "cuda",
    max_samples: int = 100,
    steps: int = 10,
) -> Dict[str, Any]:
    """Evaluate all samples for a method."""
    results = load_attribution_results(result_path, max_samples)

    noise_aucs = []
    repr_aucs = []

    for sample in tqdm(results, desc=f"Evaluating {Path(result_path).stem}", leave=False):
        result = evaluate_single_sample(model, tokenizer, sample, device, steps)
        if result:
            if result["noise_insertion_auc"] is not None:
                noise_aucs.append(result["noise_insertion_auc"])
            if result["representation_insertion_auc"] is not None:
                repr_aucs.append(result["representation_insertion_auc"])

    return {
        "num_samples": len(results),
        "p_auc_mean": float(np.mean(noise_aucs)) if noise_aucs else None,
        "p_auc_std": float(np.std(noise_aucs)) if noise_aucs else None,
        "r_auc_mean": float(np.mean(repr_aucs)) if repr_aucs else None,
        "r_auc_std": float(np.std(repr_aucs)) if repr_aucs else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare attribution methods on first 100 samples")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., Qwen3-4B, Llama-3.2-3B-Instruct)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate per dataset")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for AUC computation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path")

    args = parser.parse_args()

    # Setup paths
    scores_dir = project_root / "results" / "attribution" / "scores" / args.model

    if not scores_dir.exists():
        print(f"Error: Scores directory not found: {scores_dir}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model_path = project_root / "model" / args.model
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, local_path=str(model_path))

    # Find all method-dataset combinations
    result_files = sorted(scores_dir.glob("*.json"))

    # Exclude non-attribution files
    exclude = ["eval_results", "record"]
    result_files = [f for f in result_files if not any(e in f.stem for e in exclude)]

    # Collect results
    all_results = []

    for result_file in result_files:
        # Parse method and dataset from filename
        stem = result_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"Skipping invalid filename: {result_file}")
            continue

        method, dataset = parts

        print(f"\nEvaluating {method} on {dataset}...")

        eval_result = evaluate_method(
            model=model,
            tokenizer=tokenizer,
            result_path=str(result_file),
            device=args.device,
            max_samples=args.max_samples,
            steps=args.steps,
        )

        all_results.append({
            "model": args.model,
            "dataset": dataset,
            "method": method,
            "samples": eval_result["num_samples"],
            "p_auc_mean": eval_result["p_auc_mean"],
            "p_auc_std": eval_result["p_auc_std"],
            "r_auc_mean": eval_result["r_auc_mean"],
            "r_auc_std": eval_result["r_auc_std"],
        })

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print table
    print("\n" + "="*100)
    print(f"COMPARISON TABLE FOR {args.model} (First {args.max_samples} samples)")
    print("="*100)

    for dataset in df["dataset"].unique():
        df_ds = df[df["dataset"] == dataset].copy()
        df_ds = df_ds.sort_values("p_auc_mean")

        print(f"\n{dataset.upper()}:")
        print("-"*80)
        print(f"{'Method':<25} {'P-AUC (↓)':<20} {'R-AUC (↑)':<20} {'Samples'}")
        print("-"*80)

        for _, row in df_ds.iterrows():
            p_str = f"{row['p_auc_mean']:.4f} ± {row['p_auc_std']:.4f}" if row['p_auc_mean'] else "N/A"
            r_str = f"{row['r_auc_mean']:.4f} ± {row['r_auc_std']:.4f}" if row['r_auc_mean'] else "N/A"
            print(f"{row['method']:<25} {p_str:<20} {r_str:<20} {row['samples']}")

    print("\n" + "="*100)
    print("P-AUC = Perturbation AUC (lower is better)")
    print("R-AUC = Recovery AUC (higher is better)")
    print("="*100)

    # Save to CSV
    if args.output:
        output_path = args.output
    else:
        output_path = project_root / "results" / "attribution" / f"comparison_{args.model}_100samples.csv"

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
