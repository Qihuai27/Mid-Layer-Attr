#!/usr/bin/env python
"""
Compare different attribution methods on evaluation datasets.

This script runs multiple attribution methods (DePass, AttentionRollout,
IntegratedGradients, GradInput, GradSAM, TokenShapley, MidLayer, etc.) and evaluates
them using noise insertion and representation insertion AUC metrics.

Usage:
    python scripts/attribution/compare_methods.py --model Qwen3-4B
    python scripts/attribution/compare_methods.py --model phi-2 --datasets ioi counterfact
    python scripts/attribution/compare_methods.py --model Qwen3-4B --methods attention_rollout depass grad_input --max-samples 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer
from src.attribution import (
    AttentionRollout,
    DePass,
    IntegratedGradients,
    MidLayerAttribution,
    MidLayerAttributionV2,
    InputCausalAttribution,
    GreedyOptimalAttribution,
    GradInput,
    GradSAM,
    TokenShapley,
)
from src.eval import noise_insertion_auc_token, representation_insertion_auc_token


# Dataset file paths
DATASET_FILES = {
    "ioi": "datasets/ioi_data.json",
    "counterfact": "datasets/counterfact_data.json",
    "LongRA": "datasets/LongRA.json",
}

# Default methods (practical runtime)
DEFAULT_METHODS = [
    "attention_rollout",
    "depass",
    "integrated_gradients",
    "grad_input",
    "grad_sam",
    "midlayer",
    "midlayer_v2",
    "input_causal",
]

# Slow methods (require many forward passes)
SLOW_METHODS = ["greedy_optimal", "shapley"]

# All available methods including slow ones
ALL_METHODS = DEFAULT_METHODS + SLOW_METHODS


def get_attribution_method(method_name: str, model_name: str = "", tokenizer=None):
    """Get attribution method instance by name."""
    # Some methods are architecture-specific
    model_type = model_name.lower()
    is_compatible = any(t in model_type for t in ["llama", "qwen", "phi"])

    method_map = {
        "attention_rollout": lambda: AttentionRollout(),
        "depass": lambda: DePass(mlp_softmax_temp=0.1) if is_compatible else None,
        "integrated_gradients": lambda: IntegratedGradients(n_steps=30),
        "grad_input": lambda: GradInput(absolute=True),
        "grad_sam": lambda: GradSAM(layer_aggregation="mean", use_relu=True),
        "shapley": lambda: TokenShapley(n_samples=200, mask_type="zero_embed"),
        "midlayer": lambda: MidLayerAttribution() if is_compatible else None,
        "midlayer_v2": lambda: MidLayerAttributionV2() if is_compatible else None,
        "input_causal": lambda: InputCausalAttribution(baseline_mode="mean"),
        "greedy_optimal": lambda: GreedyOptimalAttribution(baseline_mode="mean", verbose=True),
    }

    if method_name not in method_map:
        return None

    return method_map[method_name]()


def load_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from JSON file."""
    dataset_path = project_root / DATASET_FILES.get(dataset_name, "")
    if not dataset_path.exists():
        print(f"Warning: Dataset file not found: {dataset_path}")
        return []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_target_token_id(model, tokenizer, sample: Dict) -> Optional[int]:
    """Get target token ID from sample."""
    answer = sample.get("answer") or sample.get("target_token")
    if answer is None:
        return None
    # Add space prefix for tokenization consistency
    tokens = tokenizer.encode(" " + str(answer), add_special_tokens=False)
    if len(tokens) > 0:
        return tokens[0]
    return None


def run_attribution_and_eval(
    model,
    tokenizer,
    method,
    sample: Dict,
    device: str = "cuda",
    eval_steps: int = 10,
) -> Optional[Dict[str, Any]]:
    """Run attribution on a single sample and evaluate."""
    prompt = sample.get("prompt", "")
    if not prompt:
        return None

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    target_pos = seq_len - 1

    # Get target token
    target_token_id = get_target_token_id(model, tokenizer, sample)
    if target_token_id is None:
        with torch.no_grad():
            outputs = model(input_ids)
            target_token_id = outputs.logits[0, target_pos].argmax().item()

    # Run attribution
    try:
        result = method.attribute(
            model=model,
            input_ids=input_ids,
            target_pos=target_pos,
            target_token_id=target_token_id,
        )
        scores = result.scores
    except Exception as e:
        print(f"Attribution error: {e}")
        return None

    # Evaluate with Noise Insertion AUC
    noise_auc = None
    try:
        ni_result = noise_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            target_pos=target_pos,
            target_token_id=target_token_id,
            steps=eval_steps,
            baseline_embed_mode="mean",
        )
        noise_auc = ni_result.auc
    except Exception as e:
        pass

    # Evaluate with Representation Insertion AUC
    repr_auc = None
    try:
        ri_result = representation_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            base_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            steps=eval_steps,
            layer_index=-1,
            distance_mode="cosine",
            position_weighting="linear",
        )
        repr_auc = ri_result.auc
    except Exception as e:
        pass

    return {
        "case_id": sample.get("case_id") or sample.get("id", 0),
        "noise_insertion_auc": noise_auc,
        "representation_insertion_auc": repr_auc,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare attribution methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen3-4B", help="Model name")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        choices=ALL_METHODS,
        help="Methods to compare. Default excludes greedy_optimal and shapley (slow)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ioi", "counterfact", "LongRA"],
        help="Evaluation datasets",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/attribution/eval",
        help="Save directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per dataset (default: all samples)"
    )
    parser.add_argument(
        "--eval-steps", type=int, default=10, help="Steps for AUC computation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Comparing attribution methods on {args.model}")
    print(f"Methods: {args.methods}")
    print(f"Datasets: {args.datasets}")
    print(f"Max samples: {args.max_samples}")

    # Load model
    model_path = project_root / "model" / args.model
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, local_path=str(model_path)
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for method_name in args.methods:
        method = get_attribution_method(method_name, args.model)
        if method is None:
            print(f"Skipping {method_name} (incompatible with {args.model})")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        method_results = {}

        for dataset_name in args.datasets:
            print(f"\n  Dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)

            if not dataset:
                print(f"  Skipping {dataset_name} (no data)")
                continue

            # Limit samples
            samples = dataset[:args.max_samples] if args.max_samples else dataset

            noise_aucs = []
            repr_aucs = []

            for sample in tqdm(samples, desc=f"  {method_name}/{dataset_name}"):
                result = run_attribution_and_eval(
                    model=model,
                    tokenizer=tokenizer,
                    method=method,
                    sample=sample,
                    device=args.device,
                    eval_steps=args.eval_steps,
                )

                if result:
                    if result["noise_insertion_auc"] is not None:
                        noise_aucs.append(result["noise_insertion_auc"])
                    if result["representation_insertion_auc"] is not None:
                        repr_aucs.append(result["representation_insertion_auc"])

            method_results[dataset_name] = {
                "num_samples": len(noise_aucs),
                "perturbation_auc": {
                    "mean": float(np.mean(noise_aucs)) if noise_aucs else None,
                    "std": float(np.std(noise_aucs)) if noise_aucs else None,
                },
                "recovery_auc": {
                    "mean": float(np.mean(repr_aucs)) if repr_aucs else None,
                    "std": float(np.std(repr_aucs)) if repr_aucs else None,
                },
            }

            # Print interim results
            p_auc = method_results[dataset_name]["perturbation_auc"]
            r_auc = method_results[dataset_name]["recovery_auc"]
            if p_auc["mean"] is not None:
                print(f"    P-AUC: {p_auc['mean']:.4f} +/- {p_auc['std']:.4f}")
            if r_auc["mean"] is not None:
                print(f"    R-AUC: {r_auc['mean']:.4f} +/- {r_auc['std']:.4f}")

        results[method_name] = method_results

    # Save results
    model_name_safe = args.model.replace("/", "_")
    output_path = save_dir / f"comparison_model_{model_name_safe}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison table
    print(f"\n{'='*90}")
    print("COMPARISON TABLE")
    print(f"{'='*90}")
    print(f"{'Method':<25} {'Dataset':<15} {'P-AUC (↓)':<15} {'R-AUC (↑)':<15}")
    print("-"*90)

    for method_name, method_results in results.items():
        for dataset_name, metrics in method_results.items():
            p_auc = metrics["perturbation_auc"]
            r_auc = metrics["recovery_auc"]
            p_str = f"{p_auc['mean']:.4f}" if p_auc["mean"] else "N/A"
            r_str = f"{r_auc['mean']:.4f}" if r_auc["mean"] else "N/A"
            print(f"{method_name:<25} {dataset_name:<15} {p_str:<15} {r_str:<15}")

    print(f"{'='*90}")
    print("\nP-AUC = Perturbation AUC (lower is better)")
    print("R-AUC = Recovery AUC (higher is better)")


if __name__ == "__main__":
    main()
