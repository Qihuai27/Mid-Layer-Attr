#!/usr/bin/env python3
"""
Unified Attribution Pipeline: Score + Evaluate in one run.

This script provides a one-stop solution for:
1. Running attribution methods on datasets
2. Saving attribution scores
3. Evaluating with AUC metrics (Perturbation AUC + Recovery AUC)
4. Generating comparison tables

Metrics:
- Perturbation AUC (P-AUC, lower is better): measures prediction degradation when corrupting important tokens
- Recovery AUC (R-AUC, higher is better): measures hidden state recovery when restoring tokens

Features:
- Supports multiple methods and datasets in one run
- Real-time saving of per-sample results with full curve data
- Checkpoint/resume support: can resume from last evaluated sample
- Saves both detailed scores and evaluation summaries
- Memory-efficient: unloads attribution method between runs

Usage:
    # Run single method on single dataset
    python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B --method attention_rollout --dataset ioi

    # Run all methods on all datasets
    python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B

    # Resume from checkpoint (auto-detects progress)
    python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B --resume

    # Quick test with few samples
    python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B --max-samples 10 --eval-samples 10

Examples:
    # Full evaluation on phi-2
    python scripts/attribution/run_attribution_pipeline.py --model phi-2

    # Compare specific methods
    python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B \\
        --methods attention_rollout depass integrated_gradients \\
        --datasets ioi counterfact
"""

import argparse
import gc
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

import numpy as np
import torch
from tqdm import tqdm

# Add project root
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


# ============================================================================
# Configuration
# ============================================================================

# Default methods (practical, reasonable runtime)
ALL_METHODS = [
    "attention_rollout",
    "depass",
    "integrated_gradients",
    "grad_input",
    "grad_sam",
    "midlayer",
    "midlayer_v2",
    "input_causal",
]

# Slow methods (theoretical baselines, O(T²) complexity or many forward passes)
SLOW_METHODS = ["greedy_optimal", "shapley"]

# All available methods
AVAILABLE_METHODS = ALL_METHODS + SLOW_METHODS

ALL_DATASETS = ["ioi", "counterfact", "LongRA"]

DATASET_FILES = {
    "ioi": "datasets/ioi_data.json",
    "counterfact": "datasets/counterfact_data.json",
    "LongRA": "datasets/LongRA.json",
}

# Methods that require specific architectures
ARCH_SPECIFIC_METHODS = {"depass", "midlayer", "midlayer_v2"}
COMPATIBLE_ARCHS = ["llama", "qwen", "phi"]


@dataclass
class PipelineConfig:
    """Configuration for the attribution pipeline."""

    model_name: str = "Qwen3-4B"
    methods: List[str] = field(default_factory=lambda: ALL_METHODS)
    datasets: List[str] = field(default_factory=lambda: ALL_DATASETS)
    device: str = "cuda:0"
    max_samples: Optional[int] = None  # None = all samples
    eval_samples: Optional[int] = None  # If None, use all scored samples
    eval_steps: int = 10
    output_dir: str = "results/attribution"
    resume: bool = False  # Resume from checkpoint

    def __post_init__(self):
        # Default eval_samples to max_samples if not specified
        if self.eval_samples is None:
            self.eval_samples = self.max_samples


# ============================================================================
# Attribution Methods Factory
# ============================================================================

def get_attribution_method(method_name: str, model_name: str = ""):
    """Create attribution method instance."""
    model_type = model_name.lower()
    is_compatible = any(arch in model_type for arch in COMPATIBLE_ARCHS)

    factories = {
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

    if method_name not in factories:
        return None

    return factories[method_name]()


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset from JSON file."""
    dataset_path = project_root / DATASET_FILES.get(dataset_name, "")
    if not dataset_path.exists():
        print(f"Warning: Dataset not found: {dataset_path}")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_token_id(model, tokenizer, sample: Dict, device: str) -> int:
    """Get target token ID from sample or model prediction."""
    answer = sample.get("answer") or sample.get("target_token")
    if answer is not None:
        tokens = tokenizer.encode(" " + str(answer), add_special_tokens=False)
        if tokens:
            return tokens[0]

    # Fall back to model prediction
    prompt = sample.get("prompt", "")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        return outputs.logits[0, -1].argmax().item()


# ============================================================================
# Checkpoint Management
# ============================================================================

def get_checkpoint_path(output_dir: Path, model_name: str, method: str, dataset: str) -> Path:
    """Get path for checkpoint file."""
    model_safe = model_name.replace("/", "_")
    return output_dir / "checkpoints" / model_safe / f"{method}_{dataset}.pkl"


def get_detailed_results_path(output_dir: Path, model_name: str, method: str, dataset: str) -> Path:
    """Get path for detailed results file (with curve data)."""
    model_safe = model_name.replace("/", "_")
    return output_dir / "detailed" / model_safe / f"{method}_{dataset}.pkl"


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return {"completed_indices": set(), "results": []}


def save_checkpoint(checkpoint_path: Path, checkpoint_data: Dict[str, Any]):
    """Save checkpoint to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)


def save_detailed_results(results_path: Path, results: List[Dict]):
    """Save detailed results with curve data."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)


def load_detailed_results(results_path: Path) -> List[Dict]:
    """Load detailed results."""
    if results_path.exists():
        with open(results_path, "rb") as f:
            return pickle.load(f)
    return []


# ============================================================================
# Attribution Scoring
# ============================================================================

def run_attribution_on_sample(
    model,
    tokenizer,
    method,
    sample: Dict,
    device: str,
) -> Optional[Dict[str, Any]]:
    """Run attribution on a single sample."""
    prompt = sample.get("prompt", "")
    if not prompt:
        return None

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    target_pos = seq_len - 1
    target_token_id = get_target_token_id(model, tokenizer, sample, device)

    try:
        result = method.attribute(
            model=model,
            input_ids=input_ids,
            target_pos=target_pos,
            target_token_id=target_token_id,
        )
        scores = result.scores.tolist() if hasattr(result.scores, "tolist") else list(result.scores)
    except Exception as e:
        print(f"Attribution error for case {sample.get('case_id', 'unknown')}: {e}")
        return None

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return {
        "case_id": sample.get("case_id") or sample.get("id", 0),
        "prompt": prompt,
        "answer": sample.get("answer") or sample.get("target_token", ""),
        "target_token_id": target_token_id,
        "target_pos": target_pos,
        "tokens": tokens,
        "attribution_scores": scores,
    }


# ============================================================================
# Evaluation with Curve Data
# ============================================================================

def evaluate_sample_with_curves(
    model,
    tokenizer,
    sample: Dict,
    device: str,
    steps: int,
) -> Dict[str, Any]:
    """
    Evaluate attribution scores for a single sample.
    Returns full curve data for later plotting.
    """
    prompt = sample["prompt"]
    scores = np.array(sample["attribution_scores"])
    target_pos = sample["target_pos"]
    target_token_id = sample["target_token_id"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    result = {
        "case_id": sample["case_id"],
        "prompt": prompt,
        "target_pos": target_pos,
        "target_token_id": target_token_id,
        "tokens": sample.get("tokens", []),
        "attribution_scores": sample["attribution_scores"],
        "seq_len": input_ids.shape[1],
    }

    # Verify length matches
    if input_ids.shape[1] != len(scores):
        result["error"] = f"Length mismatch: input_ids={input_ids.shape[1]}, scores={len(scores)}"
        result["perturbation_auc"] = None
        result["recovery_auc"] = None
        return result

    # Perturbation AUC (lower is better) - tests necessity
    try:
        p_result = noise_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            target_pos=target_pos,
            target_token_id=target_token_id,
            steps=steps,
            baseline_embed_mode="mean",
        )
        result["perturbation_auc"] = p_result.auc
        result["perturbation_curve_x"] = p_result.curve_x.tolist()
        result["perturbation_curve_y"] = p_result.curve_y.tolist()
        result["perturbation_phi_max"] = p_result.phi_max
        result["perturbation_phi_min"] = p_result.phi_min
    except Exception as e:
        result["perturbation_auc"] = None
        result["perturbation_error"] = str(e)

    # Recovery AUC (higher is better) - tests sufficiency
    try:
        r_result = representation_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            base_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            steps=steps,
            layer_index=-1,
            distance_mode="cosine",
            position_weighting="linear",
        )
        result["recovery_auc"] = r_result.auc
        result["recovery_curve_x"] = r_result.curve_x.tolist()
        result["recovery_curve_y"] = r_result.curve_y.tolist()
        result["recovery_phi_max"] = r_result.phi_max
        result["recovery_phi_min"] = r_result.phi_min
    except Exception as e:
        result["recovery_auc"] = None
        result["recovery_error"] = str(e)

    return result


def run_evaluation_with_checkpoint(
    model,
    tokenizer,
    attr_results: List[Dict],
    checkpoint_path: Path,
    detailed_path: Path,
    max_samples: Optional[int],
    steps: int,
    device: str,
    resume: bool = False,
) -> List[Dict]:
    """
    Run evaluation with checkpoint support.
    Saves results after each sample for fault tolerance.
    """
    # Load existing checkpoint if resuming
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        completed_indices = checkpoint["completed_indices"]
        detailed_results = checkpoint["results"]
        print(f"  Resuming from checkpoint: {len(completed_indices)} samples already completed")
    else:
        completed_indices = set()
        detailed_results = []

    # Determine samples to evaluate
    samples = attr_results[:max_samples] if max_samples else attr_results

    # Find samples that need evaluation
    samples_to_eval = [
        (idx, sample) for idx, sample in enumerate(samples)
        if idx not in completed_indices
    ]

    if not samples_to_eval:
        print("  All samples already evaluated!")
        return detailed_results

    print(f"  Evaluating {len(samples_to_eval)} remaining samples (of {len(samples)} total)")

    for idx, sample in tqdm(samples_to_eval, desc="  Evaluating"):
        eval_result = evaluate_sample_with_curves(
            model, tokenizer, sample, device, steps
        )
        eval_result["sample_index"] = idx

        detailed_results.append(eval_result)
        completed_indices.add(idx)

        # Save checkpoint after each sample
        checkpoint = {
            "completed_indices": completed_indices,
            "results": detailed_results,
        }
        save_checkpoint(checkpoint_path, checkpoint)

    # Save final detailed results
    save_detailed_results(detailed_path, detailed_results)

    return detailed_results


def compute_summary_stats(detailed_results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from detailed results."""
    perturbation_aucs = [
        r["perturbation_auc"] for r in detailed_results
        if r.get("perturbation_auc") is not None
    ]
    recovery_aucs = [
        r["recovery_auc"] for r in detailed_results
        if r.get("recovery_auc") is not None
    ]

    return {
        "num_samples": len(detailed_results),
        "num_evaluated_perturbation": len(perturbation_aucs),
        "num_evaluated_recovery": len(recovery_aucs),
        "perturbation_auc": {
            "mean": float(np.mean(perturbation_aucs)) if perturbation_aucs else None,
            "std": float(np.std(perturbation_aucs)) if perturbation_aucs else None,
            "median": float(np.median(perturbation_aucs)) if perturbation_aucs else None,
        },
        "recovery_auc": {
            "mean": float(np.mean(recovery_aucs)) if recovery_aucs else None,
            "std": float(np.std(recovery_aucs)) if recovery_aucs else None,
            "median": float(np.median(recovery_aucs)) if recovery_aucs else None,
        },
    }


# ============================================================================
# File I/O
# ============================================================================

def get_scores_path(output_dir: Path, model_name: str, method: str, dataset: str) -> Path:
    """Get path for scores file."""
    model_safe = model_name.replace("/", "_")
    return output_dir / "scores" / model_safe / f"{method}_{dataset}.json"


def get_eval_path(output_dir: Path, model_name: str) -> Path:
    """Get path for evaluation results."""
    model_safe = model_name.replace("/", "_")
    return output_dir / "eval" / f"pipeline_{model_safe}.json"


def save_scores(results: List[Dict], path: Path):
    """Save attribution scores to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_scores(path: Path) -> List[Dict]:
    """Load attribution scores from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_eval_results(results: Dict, path: Path):
    """Save evaluation results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(config: PipelineConfig):
    """Run the full attribution pipeline."""
    print("=" * 80)
    print("ATTRIBUTION PIPELINE")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Methods: {config.methods}")
    print(f"Datasets: {config.datasets}")
    print(f"Max samples: {config.max_samples or 'all'}")
    print(f"Eval samples: {config.eval_samples or 'all'}")
    print(f"Output dir: {config.output_dir}")
    print(f"Resume: {config.resume}")
    print("=" * 80)

    output_dir = Path(config.output_dir)

    # Load model
    print("\nLoading model...")
    model_path = project_root / "model" / config.model_name
    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        device=config.device,
        local_path=str(model_path) if model_path.exists() else None,
    )

    # Store all evaluation results
    all_eval_results = {
        "model": config.model_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": config.max_samples,
            "eval_samples": config.eval_samples,
            "eval_steps": config.eval_steps,
        },
        "results": {},
    }

    # Process each method
    for method_name in config.methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        method = get_attribution_method(method_name, config.model_name)
        if method is None:
            print(f"Skipping {method_name} (incompatible with {config.model_name})")
            continue

        method_results = {}

        for dataset_name in config.datasets:
            print(f"\n  Dataset: {dataset_name}")
            scores_path = get_scores_path(output_dir, config.model_name, method_name, dataset_name)
            checkpoint_path = get_checkpoint_path(output_dir, config.model_name, method_name, dataset_name)
            detailed_path = get_detailed_results_path(output_dir, config.model_name, method_name, dataset_name)

            # Step 1: Get or compute scores
            if scores_path.exists():
                print(f"  Loading existing scores from {scores_path}")
                attr_results = load_scores(scores_path)
            else:
                # Load dataset
                dataset = load_dataset(dataset_name)
                if not dataset:
                    print(f"  Skipping {dataset_name} (no data)")
                    continue

                # Run attribution
                samples = dataset[:config.max_samples] if config.max_samples else dataset
                attr_results = []
                for sample in tqdm(samples, desc="  Scoring"):
                    result = run_attribution_on_sample(model, tokenizer, method, sample, config.device)
                    if result:
                        attr_results.append(result)

                # Save scores
                save_scores(attr_results, scores_path)
                print(f"  Saved {len(attr_results)} scores to {scores_path}")

            # Step 2: Evaluate with checkpoint support
            print(f"  Evaluating with checkpoint support...")
            detailed_results = run_evaluation_with_checkpoint(
                model=model,
                tokenizer=tokenizer,
                attr_results=attr_results,
                checkpoint_path=checkpoint_path,
                detailed_path=detailed_path,
                max_samples=config.eval_samples,
                steps=config.eval_steps,
                device=config.device,
                resume=config.resume,
            )

            # Compute summary stats
            summary = compute_summary_stats(detailed_results)
            method_results[dataset_name] = summary

            # Print results
            p_auc = summary["perturbation_auc"]
            r_auc = summary["recovery_auc"]
            print(f"  Results ({summary['num_evaluated_perturbation']} samples):")
            if p_auc["mean"] is not None:
                print(f"    P-AUC (↓): {p_auc['mean']:.4f} ± {p_auc['std']:.4f}")
            if r_auc["mean"] is not None:
                print(f"    R-AUC (↑): {r_auc['mean']:.4f} ± {r_auc['std']:.4f}")

            print(f"  Detailed results saved to: {detailed_path}")

        all_eval_results["results"][method_name] = method_results

        # Clear method to free memory
        del method
        gc.collect()
        torch.cuda.empty_cache()

    # Save all evaluation results
    eval_path = get_eval_path(output_dir, config.model_name)
    save_eval_results(all_eval_results, eval_path)
    print(f"\nEvaluation summary saved to: {eval_path}")

    # Print comparison table
    print_comparison_table(all_eval_results)

    return all_eval_results


def print_comparison_table(results: Dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Method':<25} {'Dataset':<15} {'Samples':<10} {'P-AUC (↓)':<20} {'R-AUC (↑)':<20}")
    print("-" * 100)

    for method_name, method_results in results["results"].items():
        for dataset_name, metrics in method_results.items():
            p_auc = metrics["perturbation_auc"]
            r_auc = metrics["recovery_auc"]
            n_samples = metrics["num_evaluated_perturbation"]
            p_str = f"{p_auc['mean']:.4f} ± {p_auc['std']:.4f}" if p_auc["mean"] else "N/A"
            r_str = f"{r_auc['mean']:.4f} ± {r_auc['std']:.4f}" if r_auc["mean"] else "N/A"
            print(f"{method_name:<25} {dataset_name:<15} {n_samples:<10} {p_str:<20} {r_str:<20}")

    print("=" * 100)
    print("\nP-AUC = Perturbation AUC (lower is better): measures necessity")
    print("R-AUC = Recovery AUC (higher is better): measures sufficiency")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Attribution Pipeline: Score + Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  P-AUC (Perturbation AUC): Lower is better - measures how much prediction degrades
                           when corrupting important tokens (necessity test)
  R-AUC (Recovery AUC):    Higher is better - measures how well hidden states recover
                           when restoring important tokens (sufficiency test)

Methods:
  Default: attention_rollout, depass, integrated_gradients, grad_input, grad_sam,
           midlayer, midlayer_v2, input_causal
  Slow:    greedy_optimal (O(T²)), shapley (many forward passes)

Checkpoint/Resume:
  The pipeline saves progress after each sample. Use --resume to continue
  from the last checkpoint if the process was interrupted.

Output Structure:
  results/attribution/
  ├── scores/{model}/           # Attribution scores (JSON)
  ├── detailed/{model}/         # Full results with curves (pickle)
  ├── checkpoints/{model}/      # Checkpoint files (pickle)
  └── eval/                     # Summary statistics (JSON)

Examples:
  # Run all default methods on all datasets (full dataset)
  python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B

  # Run specific methods
  python scripts/attribution/run_attribution_pipeline.py --model phi-2 \\
      --methods attention_rollout depass --datasets ioi counterfact

  # Resume from checkpoint after interruption
  python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B --resume

  # Include shapley (slow!) as theoretical baseline
  python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B \\
      --methods attention_rollout shapley --max-samples 20

  # Quick test with limited samples
  python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B \\
      --max-samples 50 --eval-samples 50
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B",
        help="Model name (in model/ directory)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=ALL_METHODS,
        choices=AVAILABLE_METHODS,
        help="Attribution methods to run. Default excludes greedy_optimal and shapley (slow)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for attribution (per dataset). Default: all samples",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Max samples for evaluation (default: same as --max-samples)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Steps for AUC computation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/attribution",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(
        model_name=args.model,
        methods=args.methods,
        datasets=args.datasets,
        device=args.device,
        max_samples=args.max_samples,
        eval_samples=args.eval_samples,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
