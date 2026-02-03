#!/usr/bin/env python3
"""
Evaluate attribution results for multiple models and datasets, comparing all methods.

Usage:
    python scripts/attribution/eval_comparison_table.py --max-samples 100
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer
from src.eval import (
    noise_insertion_auc_token,
    representation_insertion_auc_token,
)


def load_attribution_results(result_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load attribution results from JSON file."""
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    if max_samples:
        results = results[:max_samples]
    return results


def evaluate_single_sample(
    model,
    tokenizer,
    sample: Dict,
    device: str = "cuda",
    steps: int = 10,
    debug: bool = False,
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
        if debug:
            print(f"  Length mismatch: {input_ids.shape[1]} vs {len(scores)}")
        return None

    # Compute Noise Insertion AUC (lower is better) - Perturbation AUC
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
        if debug:
            print(f"  Error in noise_insertion: {e}")
        noise_auc = None

    # Compute Representation Insertion AUC (higher is better) - Recovery AUC
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
        if debug:
            print(f"  Error in representation_insertion: {e}")
        repr_auc = None

    return {
        "case_id": sample["case_id"],
        "perturbation_auc": noise_auc,
        "recovery_auc": repr_auc,
    }


def evaluate_method_file(
    model,
    tokenizer,
    result_path: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    steps: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:
    """Evaluate all samples in an attribution result file."""
    results = load_attribution_results(result_path, max_samples)

    eval_results = []
    p_aucs = []  # Perturbation AUC
    r_aucs = []  # Recovery AUC
    errors = []

    for i, sample in enumerate(tqdm(results, desc=f"Evaluating {Path(result_path).name}", leave=False)):
        result = evaluate_single_sample(model, tokenizer, sample, device, steps, debug=debug and i < 3)
        if result:
            eval_results.append(result)
            if result["perturbation_auc"] is not None:
                p_aucs.append(result["perturbation_auc"])
            if result["recovery_auc"] is not None:
                r_aucs.append(result["recovery_auc"])
        else:
            errors.append(i)

    if debug and errors:
        print(f"    Failed samples: {len(errors)}/{len(results)}")

    # Compute statistics
    return {
        "num_samples": len(eval_results),
        "num_failed": len(errors),
        "perturbation_auc": {
            "mean": float(np.mean(p_aucs)) if p_aucs else None,
            "std": float(np.std(p_aucs)) if p_aucs else None,
        },
        "recovery_auc": {
            "mean": float(np.mean(r_aucs)) if r_aucs else None,
            "std": float(np.std(r_aucs)) if r_aucs else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate attribution results for multiple models and datasets")
    parser.add_argument("--models", nargs="+", default=["Qwen3-4B", "Llama-3.2-3B-Instruct"],
                        help="Model names to evaluate")
    parser.add_argument("--datasets", nargs="+", default=["ioi", "counterfact"],
                        help="Datasets to evaluate")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate per file")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for AUC computation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--output-dir", type=str, default="results/attribution/eval",
                        help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Attribution methods to evaluate
    methods = [
        "attention_rollout",
        "depass",
        "grad_input",
        "grad_sam",
        "input_causal",
        "integrated_gradients",
        "midlayer",
        "midlayer_v2",
        "shapley",
    ]

    all_results = {}

    # Evaluate each model
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Loading model: {model_name}")
        print(f"{'='*80}")

        model_path = project_root / "model" / model_name
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device=args.device,
            local_path=str(model_path)
        )

        model_results = {}

        # Evaluate each dataset
        for dataset in args.datasets:
            print(f"\n--- Dataset: {dataset} ---")

            dataset_results = {}

            # Evaluate each method
            for method in methods:
                result_file = project_root / "results" / "attribution" / "scores" / model_name / f"{method}_{dataset}.json"

                if not result_file.exists():
                    print(f"  [SKIP] {method}: File not found")
                    continue

                print(f"  Evaluating {method}...", end=" ", flush=True)

                try:
                    eval_result = evaluate_method_file(
                        model=model,
                        tokenizer=tokenizer,
                        result_path=str(result_file),
                        device=args.device,
                        max_samples=args.max_samples,
                        steps=args.steps,
                        debug=args.debug,
                    )

                    dataset_results[method] = eval_result

                    p_auc = eval_result["perturbation_auc"]["mean"]
                    r_auc = eval_result["recovery_auc"]["mean"]
                    p_auc_str = f"{p_auc:.4f}" if p_auc is not None else "N/A"
                    r_auc_str = f"{r_auc:.4f}" if r_auc is not None else "N/A"
                    success_rate = f"{eval_result['num_samples']}/{eval_result['num_samples'] + eval_result['num_failed']}"
                    print(f"✓ {success_rate} | P-AUC: {p_auc_str}, R-AUC: {r_auc_str}")

                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

            model_results[dataset] = dataset_results

        all_results[model_name] = model_results

        # Clear GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Save full results
    output_path = Path(args.output_dir) / "comparison_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # Generate comparison tables
    print("\n" + "="*100)
    print("COMPARISON TABLE - Perturbation AUC (P-AUC, lower is better)")
    print("="*100)

    # Build DataFrame for P-AUC
    p_auc_data = []
    for model_name in args.models:
        for dataset in args.datasets:
            row = {"Model": model_name, "Dataset": dataset}
            if dataset in all_results[model_name]:
                for method in methods:
                    if method in all_results[model_name][dataset]:
                        p_auc = all_results[model_name][dataset][method]["perturbation_auc"]["mean"]
                        row[method] = f"{p_auc:.4f}" if p_auc else "N/A"
            p_auc_data.append(row)

    df_p_auc = pd.DataFrame(p_auc_data)
    print(df_p_auc.to_string(index=False))

    print("\n" + "="*100)
    print("COMPARISON TABLE - Recovery AUC (R-AUC, higher is better)")
    print("="*100)

    # Build DataFrame for R-AUC
    r_auc_data = []
    for model_name in args.models:
        for dataset in args.datasets:
            row = {"Model": model_name, "Dataset": dataset}
            if dataset in all_results[model_name]:
                for method in methods:
                    if method in all_results[model_name][dataset]:
                        r_auc = all_results[model_name][dataset][method]["recovery_auc"]["mean"]
                        row[method] = f"{r_auc:.4f}" if r_auc else "N/A"
            r_auc_data.append(row)

    df_r_auc = pd.DataFrame(r_auc_data)
    print(df_r_auc.to_string(index=False))

    # Save tables to CSV
    p_auc_csv = Path(args.output_dir) / "perturbation_auc_comparison.csv"
    r_auc_csv = Path(args.output_dir) / "recovery_auc_comparison.csv"
    df_p_auc.to_csv(p_auc_csv, index=False)
    df_r_auc.to_csv(r_auc_csv, index=False)

    print(f"\n{'='*100}")
    print(f"Tables saved to:")
    print(f"  - {p_auc_csv}")
    print(f"  - {r_auc_csv}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
