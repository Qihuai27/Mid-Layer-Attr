#!/usr/bin/env python3
"""
Evaluate attribution results using AUC metrics.

Usage:
    python scripts/eval_attribution.py --results-dir results/attribution_scores --model Qwen3-4B
    python scripts/eval_attribution.py --result-file results/attribution_scores/depass_ioi.json --model Qwen3-4B
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer
from src.eval import (
    noise_insertion_auc_token,
    representation_insertion_auc_token,
)


def load_attribution_results(result_path: str) -> List[Dict]:
    """Load attribution results from JSON file."""
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        print(f"Warning: Length mismatch for case {sample['case_id']}: {input_ids.shape[1]} vs {len(scores)}")
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
        print(f"Error in noise insertion for case {sample['case_id']}: {e}")
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
        print(f"Error in representation insertion for case {sample['case_id']}: {e}")
        repr_auc = None
    
    return {
        "case_id": sample["case_id"],
        "noise_insertion_auc": noise_auc,
        "representation_insertion_auc": repr_auc,
    }


def evaluate_attribution_file(
    model,
    tokenizer,
    result_path: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    steps: int = 10,
) -> Dict[str, Any]:
    """Evaluate all samples in an attribution result file."""
    results = load_attribution_results(result_path)
    
    if max_samples:
        results = results[:max_samples]
    
    eval_results = []
    noise_aucs = []
    repr_aucs = []
    
    for sample in tqdm(results, desc=f"Evaluating {Path(result_path).stem}"):
        result = evaluate_single_sample(model, tokenizer, sample, device, steps)
        if result:
            eval_results.append(result)
            if result["noise_insertion_auc"] is not None:
                noise_aucs.append(result["noise_insertion_auc"])
            if result["representation_insertion_auc"] is not None:
                repr_aucs.append(result["representation_insertion_auc"])
    
    # Compute statistics
    return {
        "file": result_path,
        "num_samples": len(eval_results),
        "perturbation_auc": {
            "mean": float(np.mean(noise_aucs)) if noise_aucs else None,
            "std": float(np.std(noise_aucs)) if noise_aucs else None,
            "median": float(np.median(noise_aucs)) if noise_aucs else None,
        },
        "recovery_auc": {
            "mean": float(np.mean(repr_aucs)) if repr_aucs else None,
            "std": float(np.std(repr_aucs)) if repr_aucs else None,
            "median": float(np.median(repr_aucs)) if repr_aucs else None,
        },
        "per_sample_results": eval_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate attribution results with AUC metrics")
    parser.add_argument("--model", type=str, default="Qwen3-4B",
                        help="Model name (in model/ directory)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing attribution results")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Single result file to evaluate")
    parser.add_argument("--output-dir", type=str, default="results/eval_scores",
                        help="Output directory for evaluation results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate per file")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of steps for AUC computation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    if args.results_dir is None and args.result_file is None:
        print("Error: Must specify either --results-dir or --result-file")
        sys.exit(1)
    
    # Load model
    model_path = project_root / "model" / args.model
    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, local_path=str(model_path))
    
    # Get result files to evaluate
    if args.result_file:
        result_files = [args.result_file]
    else:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("*.json"))
    
    # Evaluate each file
    all_eval_results = {}
    
    for result_file in result_files:
        print(f"\n{'='*60}")
        print(f"Evaluating: {result_file}")
        print(f"{'='*60}")
        
        eval_result = evaluate_attribution_file(
            model=model,
            tokenizer=tokenizer,
            result_path=str(result_file),
            device=args.device,
            max_samples=args.max_samples,
            steps=args.steps,
        )
        
        file_key = Path(result_file).stem
        all_eval_results[file_key] = eval_result
        
        # Print summary
        print(f"\nResults for {file_key}:")
        print(f"  Samples: {eval_result['num_samples']}")
        p_auc = eval_result['perturbation_auc']
        r_auc = eval_result['recovery_auc']
        if p_auc['mean'] is not None:
            print(f"  Perturbation AUC (lower is better): {p_auc['mean']:.4f} +/- {p_auc['std']:.4f}")
        if r_auc['mean'] is not None:
            print(f"  Recovery AUC (higher is better): {r_auc['mean']:.4f} +/- {r_auc['std']:.4f}")
    
    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / f"eval_results_{args.model}.json"
    
    # Save summary (without per-sample results for compact file)
    summary = {}
    for key, val in all_eval_results.items():
        summary[key] = {
            "file": val["file"],
            "num_samples": val["num_samples"],
            "perturbation_auc": val["perturbation_auc"],
            "recovery_auc": val["recovery_auc"],
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Summary saved to {output_path}")
    print(f"{'='*60}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Method-Dataset':<40} {'P-AUC (↓)':<15} {'R-AUC (↑)':<15}")
    print("-"*80)
    for key in sorted(all_eval_results.keys()):
        val = all_eval_results[key]
        p_mean = val['perturbation_auc']['mean']
        r_mean = val['recovery_auc']['mean']
        p_str = f"{p_mean:.4f}" if p_mean else "N/A"
        r_str = f"{r_mean:.4f}" if r_mean else "N/A"
        print(f"{key:<40} {p_str:<15} {r_str:<15}")
    print("="*80)
    print("\nP-AUC = Perturbation AUC (lower is better)")
    print("R-AUC = Recovery AUC (higher is better)")


if __name__ == "__main__":
    main()
