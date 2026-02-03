#!/usr/bin/env python3
"""
Run all attribution methods on all local models and all datasets, then evaluate.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --max-samples 100
    python scripts/run_all_experiments.py --models Qwen3-4B llama2-7b --datasets ioi counterfact
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer

# All available models in model/ directory
ALL_MODELS = [
    "Qwen3-4B",
    "llama2-7b",
    "llama-3.2-1b",
    "llama3-8b-instruct",
    "phi-2",
    "tinyllama-1.1b",
]

# All datasets
ALL_DATASETS = ["ioi", "counterfact", "LongRA"]

# Default attribution methods (practical runtime)
ALL_METHODS = [
    "attention_rollout",
    "depass",
    "integrated_gradients",
    "midlayer",
    "midlayer_v2",
    "input_causal",
]

# Slow methods (O(T²) - theoretical baselines)
SLOW_METHODS = ["greedy_optimal"]

# All available methods
AVAILABLE_METHODS = ALL_METHODS + SLOW_METHODS

DATASET_FILES = {
    "ioi": "datasets/ioi_data.json",
    "counterfact": "datasets/counterfact_data.json",
    "LongRA": "datasets/LongRA.json",
}


def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def unload_model(model, tokenizer):
    """Unload model to free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def get_attribution_method(method_name: str):
    from src.attribution import (
        AttentionRollout,
        DePass,
        IntegratedGradients,
        MidLayerAttribution,
        MidLayerAttributionV2,
        InputCausalAttribution,
        GreedyOptimalAttribution,
    )

    method_map = {
        "attention_rollout": AttentionRollout(),
        "depass": DePass(mlp_softmax_temp=0.1),
        "integrated_gradients": IntegratedGradients(n_steps=30),
        "midlayer": MidLayerAttribution(),
        "midlayer_v2": MidLayerAttributionV2(),
        "input_causal": InputCausalAttribution(baseline_mode="mean"),
        "greedy_optimal": GreedyOptimalAttribution(baseline_mode="mean", verbose=True),
    }

    return method_map.get(method_name)


def get_target_token_id(model, tokenizer, sample: Dict) -> Optional[int]:
    answer = sample.get("answer") or sample.get("target_token")
    if answer is None:
        return None
    tokens = tokenizer.encode(" " + answer, add_special_tokens=False)
    if len(tokens) > 0:
        return tokens[0]
    return None


def run_attribution_on_sample(model, tokenizer, method, sample: Dict, device: str = "cuda") -> Dict[str, Any]:
    prompt = sample.get("prompt", "")
    if not prompt:
        return None
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    target_pos = seq_len - 1
    target_token_id = get_target_token_id(model, tokenizer, sample)
    
    if target_token_id is None:
        with torch.no_grad():
            outputs = model(input_ids)
            target_token_id = outputs.logits[0, target_pos].argmax().item()
    
    try:
        result = method.attribute(
            model=model,
            input_ids=input_ids,
            target_pos=target_pos,
            target_token_id=target_token_id,
        )
        scores = result.scores.tolist()
    except Exception as e:
        print(f"Error in attribution: {e}")
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
        "method": method.name,
    }


def run_batch_attribution(model, tokenizer, method, dataset: List[Dict], max_samples: Optional[int] = None, device: str = "cuda") -> List[Dict]:
    results = []
    samples = dataset[:max_samples] if max_samples else dataset
    
    for sample in tqdm(samples, desc=f"Running {method.name}"):
        result = run_attribution_on_sample(model, tokenizer, method, sample, device)
        if result is not None:
            results.append(result)
    
    return results


def evaluate_attribution_results(model, tokenizer, results: List[Dict], max_samples: Optional[int] = None, steps: int = 10, device: str = "cuda") -> Dict[str, Any]:
    from src.eval import noise_insertion_auc_token, representation_insertion_auc_token
    
    if max_samples:
        results = results[:max_samples]
    
    noise_aucs = []
    repr_aucs = []
    
    for sample in tqdm(results, desc="Evaluating"):
        prompt = sample["prompt"]
        scores = np.array(sample["attribution_scores"])
        target_pos = sample["target_pos"]
        target_token_id = sample["target_token_id"]
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        if input_ids.shape[1] != len(scores):
            continue
        
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
            noise_aucs.append(ni_result.auc)
        except Exception as e:
            pass
        
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
            repr_aucs.append(ri_result.auc)
        except Exception as e:
            pass
    
    return {
        "num_samples": len(results),
        "perturbation_auc": {
            "mean": float(np.mean(noise_aucs)) if noise_aucs else None,
            "std": float(np.std(noise_aucs)) if noise_aucs else None,
        },
        "recovery_auc": {
            "mean": float(np.mean(repr_aucs)) if repr_aucs else None,
            "std": float(np.std(repr_aucs)) if repr_aucs else None,
        },
    }


def save_results(results: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run all attribution experiments")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run (default: all)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to run (default: all)")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset (default: all samples)")
    parser.add_argument("--eval-samples", type=int, default=None, help="Max samples for evaluation (default: all)")
    parser.add_argument("--steps", type=int, default=10, help="AUC computation steps")
    parser.add_argument("--output-dir", type=str, default="results/full_experiments", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if results exist")
    
    args = parser.parse_args()
    
    models = args.models or ALL_MODELS
    datasets = args.datasets or ALL_DATASETS
    methods = args.methods or ALL_METHODS
    
    # Filter models that exist
    available_models = []
    for model_name in models:
        model_path = project_root / "model" / model_name
        if model_path.exists():
            available_models.append(model_name)
        else:
            print(f"Model not found, skipping: {model_name}")
    
    if not available_models:
        print("No available models found!")
        sys.exit(1)
    
    print(f"Models: {available_models}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Max samples: {args.max_samples}")
    print(f"Eval samples: {args.eval_samples}")
    
    # Store all evaluation results
    all_eval_results = {}
    
    for model_name in available_models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        model_path = project_root / "model" / model_name
        model, tokenizer = load_model_and_tokenizer(model_name, device=args.device, local_path=str(model_path))
        
        model_results = {}
        
        for method_name in methods:
            # Check if method is compatible with model
            if method_name in ["depass", "midlayer", "midlayer_v2"]:
                model_type = getattr(model.config, "model_type", "").lower()
                if not any(t in model_type for t in ["llama", "qwen"]):
                    print(f"Skipping {method_name} for {model_name} (incompatible architecture)")
                    continue
            
            method = get_attribution_method(method_name)
            if method is None:
                continue
            
            for ds_name in datasets:
                result_key = f"{method_name}_{ds_name}"
                output_path = project_root / args.output_dir / model_name / f"{result_key}.json"
                
                if args.skip_existing and output_path.exists():
                    print(f"Skipping {result_key} (exists)")
                    # Load existing for evaluation
                    with open(output_path, 'r') as f:
                        attr_results = json.load(f)
                else:
                    print(f"\n--- {method_name} on {ds_name} ---")
                    
                    ds_path = project_root / DATASET_FILES[ds_name]
                    dataset = load_dataset(str(ds_path))
                    
                    attr_results = run_batch_attribution(
                        model=model,
                        tokenizer=tokenizer,
                        method=method,
                        dataset=dataset,
                        max_samples=args.max_samples,
                        device=args.device,
                    )
                    
                    save_results(attr_results, str(output_path))
                
                # Evaluate
                print(f"Evaluating {result_key}...")
                eval_result = evaluate_attribution_results(
                    model=model,
                    tokenizer=tokenizer,
                    results=attr_results,
                    max_samples=args.eval_samples,
                    steps=args.steps,
                    device=args.device,
                )
                
                model_results[result_key] = eval_result

                p_auc = eval_result['perturbation_auc']
                r_auc = eval_result['recovery_auc']
                if p_auc['mean'] is not None:
                    print(f"  P-AUC: {p_auc['mean']:.4f} +/- {p_auc['std']:.4f}")
                if r_auc['mean'] is not None:
                    print(f"  R-AUC: {r_auc['mean']:.4f} +/- {r_auc['std']:.4f}")
        
        all_eval_results[model_name] = model_results
        
        # Save model evaluation results
        eval_output_path = project_root / args.output_dir / model_name / "eval_results.json"
        os.makedirs(eval_output_path.parent, exist_ok=True)
        with open(eval_output_path, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        # Unload model
        unload_model(model, tokenizer)
    
    # Save combined results
    combined_output_path = project_root / args.output_dir / "all_eval_results.json"
    with open(combined_output_path, 'w') as f:
        json.dump(all_eval_results, f, indent=2)
    
    # Print final comparison table
    print(f"\n{'='*100}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Method-Dataset':<35} {'P-AUC (↓)':<15} {'R-AUC (↑)':<15}")
    print("-"*100)

    for model_name, model_results in all_eval_results.items():
        for result_key, eval_result in model_results.items():
            p_auc = eval_result['perturbation_auc']
            r_auc = eval_result['recovery_auc']
            p_str = f"{p_auc['mean']:.4f}" if p_auc['mean'] else "N/A"
            r_str = f"{r_auc['mean']:.4f}" if r_auc['mean'] else "N/A"
            print(f"{model_name:<20} {result_key:<35} {p_str:<15} {r_str:<15}")

    print(f"{'='*100}")
    print("\nP-AUC = Perturbation AUC (lower is better)")
    print("R-AUC = Recovery AUC (higher is better)")
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
