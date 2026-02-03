#!/usr/bin/env python3
"""
Evaluate only shapley method and merge with existing results.
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

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    if input_ids.shape[1] != len(scores):
        return None

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


def evaluate_shapley(
    model,
    tokenizer,
    scores_dir: Path,
    datasets: List[str],
    device: str = "cuda",
    max_samples: int = 100,
    steps: int = 10,
) -> Dict[str, Dict]:
    """Evaluate shapley method on specified datasets."""
    results = {}

    for dataset in datasets:
        result_file = scores_dir / f"shapley_{dataset}.json"
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping")
            continue

        print(f"\nEvaluating shapley on {dataset}...")
        samples = load_attribution_results(str(result_file), max_samples)

        noise_aucs = []
        repr_aucs = []

        for sample in tqdm(samples, desc=f"shapley_{dataset}"):
            result = evaluate_single_sample(model, tokenizer, sample, device, steps)
            if result:
                if result["noise_insertion_auc"] is not None:
                    noise_aucs.append(result["noise_insertion_auc"])
                if result["representation_insertion_auc"] is not None:
                    repr_aucs.append(result["representation_insertion_auc"])

        results[dataset] = {
            "num_samples": len(samples),
            "p_auc_mean": float(np.mean(noise_aucs)) if noise_aucs else None,
            "p_auc_std": float(np.std(noise_aucs)) if noise_aucs else None,
            "r_auc_mean": float(np.mean(repr_aucs)) if repr_aucs else None,
            "r_auc_std": float(np.std(repr_aucs)) if repr_aucs else None,
        }

        print(f"  P-AUC: {results[dataset]['p_auc_mean']:.4f} ± {results[dataset]['p_auc_std']:.4f}")
        print(f"  R-AUC: {results[dataset]['r_auc_mean']:.4f} ± {results[dataset]['r_auc_std']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate shapley method only")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--datasets", type=str, nargs="+", default=["ioi", "counterfact"])

    args = parser.parse_args()

    scores_dir = project_root / "results" / "attribution" / "scores" / args.model

    if not scores_dir.exists():
        print(f"Error: Scores directory not found: {scores_dir}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model_path = project_root / "model" / args.model
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, local_path=str(model_path))

    results = evaluate_shapley(
        model=model,
        tokenizer=tokenizer,
        scores_dir=scores_dir,
        datasets=args.datasets,
        device=args.device,
        max_samples=args.max_samples,
        steps=args.steps,
    )

    # Save results
    output_file = project_root / "results" / "attribution" / "eval" / f"shapley_{args.model}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Also output CSV format
    rows = []
    for dataset, data in results.items():
        rows.append({
            "model": args.model,
            "dataset": dataset,
            "method": "shapley",
            "samples": data["num_samples"],
            "p_auc_mean": data["p_auc_mean"],
            "p_auc_std": data["p_auc_std"],
            "r_auc_mean": data["r_auc_mean"],
            "r_auc_std": data["r_auc_std"],
        })

    df = pd.DataFrame(rows)
    csv_file = project_root / "results" / "attribution" / "eval" / f"shapley_{args.model}.csv"
    df.to_csv(csv_file, index=False)
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
