#!/usr/bin/env python3
"""
Run token_shapley attribution on first 100 samples for comparison.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import load_model_and_tokenizer
from src.attribution import TokenShapley


def load_dataset(dataset_name: str) -> List[Dict]:
    """Load dataset."""
    if dataset_name == "LongRA":
        file_path = "datasets/LongRA.json"
    else:
        file_path = f"datasets/{dataset_name}_data.json"

    with open(file_path) as f:
        data = json.load(f)
    return data


def run_attribution(
    model_name: str,
    dataset_name: str,
    max_samples: int = 100,
    n_shapley_samples: int = 200,
):
    """Run token_shapley attribution."""
    print(f"\n{'='*80}")
    print(f"Running TokenShapley Attribution")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {max_samples}")
    print(f"Shapley samples per token: {n_shapley_samples}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device="cuda:0",
    )
    model.eval()

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    dataset = dataset[:max_samples]
    print(f"Processing {len(dataset)} samples\n")

    # Initialize attribution method
    method = TokenShapley(n_samples=n_shapley_samples)

    # Run attribution
    results = []

    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, desc="Attribution")):
            prompt = sample["prompt"]
            answer = sample["answer"]

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            input_ids = inputs["input_ids"]

            # Get target position (last token)
            target_pos = input_ids.shape[1] - 1

            # Get target token
            full_text = prompt + answer
            full_inputs = tokenizer(full_text, return_tensors="pt").to("cuda:0")
            target_token_id = full_inputs["input_ids"][0, target_pos].item()

            # Get tokens for result
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            # Run attribution
            result = method.attribute(
                model=model,
                input_ids=input_ids,
                target_pos=target_pos,
                target_token_id=target_token_id,
                tokenizer=tokenizer,
            )

            # Save result
            results.append({
                "case_id": i,
                "prompt": prompt,
                "answer": answer,
                "target_token_id": target_token_id,
                "target_pos": target_pos,
                "tokens": tokens,
                "attribution_scores": result.scores.tolist(),
                "method": "TokenShapley",
            })

            # Save intermediate results every 10 samples
            if (i + 1) % 10 == 0:
                output_dir = Path(f"results/attribution/scores/{model_name}")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"shapley_{dataset_name}.json"
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved intermediate results: {len(results)} samples")

    # Save final results
    output_dir = Path(f"results/attribution/scores/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"shapley_{dataset_name}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to: {output_file}")
    print(f"  Total samples: {len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--n-shapley-samples", type=int, default=200)
    args = parser.parse_args()

    run_attribution(
        model_name=args.model,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        n_shapley_samples=args.n_shapley_samples,
    )


if __name__ == "__main__":
    main()
