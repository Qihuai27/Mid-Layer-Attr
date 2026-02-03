#!/usr/bin/env python3
"""
Visualize attribution results for a single sample.

Usage:
    # With model (computes AUC metrics)
    python scripts/visualize_sample.py --model gpt2-xl --text "This movie is great"

    # From saved attribution results
    python scripts/visualize_sample.py --result-file results/sample_attribution.json

    # Demo mode (synthetic data)
    python scripts/visualize_sample.py --demo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_visualization(output_path: Optional[str] = None):
    """Generate demo visualization with synthetic data."""
    from src.visualization import (
        plot_sample_attribution,
        SampleVisualizationData,
    )

    # Synthetic sample
    tokens = [
        "The", "Ġmovie", "Ġwas", "Ġabsolutely", "Ġfantastic", "!",
        "ĠThe", "Ġacting", "Ġand", "Ġdirection", "Ġwere", "Ġsuperb", "."
    ]

    # Synthetic importance scores (higher for sentiment words)
    importance_scores = np.array([
        0.1, 0.3, 0.15, 0.6, 0.95, 0.4,  # "fantastic" is most important
        0.2, 0.5, 0.1, 0.4, 0.2, 0.85, 0.1  # "superb" is also important
    ])

    # Synthetic AUC curves
    steps = 10
    x = np.linspace(0, 1, steps + 1)

    # Noise insertion: good attribution drops quickly
    noise_y = 1 - 0.7 * (1 - np.exp(-3 * x))
    noise_auc = np.trapz(noise_y, x)

    # Representation insertion: good attribution recovers quickly
    repr_y = 1 - np.exp(-4 * x)
    repr_auc = np.trapz(repr_y, x)

    # Create visualization data
    data = SampleVisualizationData(
        tokens=tokens,
        importance_scores=importance_scores,
        noise_curve_x=x,
        noise_curve_y=noise_y,
        repr_curve_x=x,
        repr_curve_y=repr_y,
        noise_auc=noise_auc,
        repr_auc=repr_auc,
        method_name="Demo Attribution",
    )

    # Create visualization
    output = output_path or "demo_attribution.png"
    fig = plot_sample_attribution(
        data=data,
        figsize=(14, 8),
        output_path=output,
        dpi=150,
        highlight_top_k=3,
    )

    if fig is not None:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"Demo visualization saved to: {output}")


def visualize_with_model(
    model_name: str,
    text: str,
    method: str = "attention_rollout",
    output_path: Optional[str] = None,
    device: str = "cuda",
):
    """Visualize attribution for text using model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.attribution import attention_rollout, attention_sum
    from src.visualization import visualize_attribution

    print(f"Loading model: {model_name}...")

    # Load model
    model_path = project_root / "model" / model_name
    if not model_path.exists():
        model_path = model_name  # Try HuggingFace hub

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if "phi" not in model_name.lower() else torch.float32,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    print(f"Input: {text}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")

    # Compute attribution
    print(f"Computing attribution with {method}...")
    if method == "attention_rollout":
        scores = attention_rollout(model, input_ids, target_pos=-1)
    elif method == "attention_sum":
        scores = attention_sum(model, input_ids, target_pos=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create visualization
    output = output_path or f"attribution_{model_name}_{method}.png"
    fig = visualize_attribution(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        importance_scores=scores,
        target_pos=-1,
        method_name=method.replace("_", " ").title(),
        output_path=output,
        dpi=150,
    )

    if fig is not None:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"Visualization saved to: {output}")


def visualize_from_file(
    result_file: str,
    sample_index: int = 0,
    output_path: Optional[str] = None,
):
    """Visualize from saved attribution results."""
    from src.visualization import plot_sample_attribution, SampleVisualizationData

    print(f"Loading results from: {result_file}")

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if isinstance(results, list):
        if sample_index >= len(results):
            raise ValueError(f"Sample index {sample_index} out of range (max {len(results) - 1})")
        sample = results[sample_index]
    else:
        sample = results

    # Extract data
    tokens = sample.get("tokens", [])
    if not tokens and "prompt" in sample:
        # Fallback: split by space (rough approximation)
        tokens = sample["prompt"].split()

    importance_scores = np.array(sample["attribution_scores"])

    # Check for AUC data
    if "noise_curve_x" in sample and "repr_curve_x" in sample:
        data = SampleVisualizationData(
            tokens=tokens,
            importance_scores=importance_scores,
            noise_curve_x=np.array(sample["noise_curve_x"]),
            noise_curve_y=np.array(sample["noise_curve_y"]),
            repr_curve_x=np.array(sample["repr_curve_x"]),
            repr_curve_y=np.array(sample["repr_curve_y"]),
            noise_auc=sample.get("noise_auc", 0.5),
            repr_auc=sample.get("repr_auc", 0.5),
            method_name=sample.get("method", "Attribution"),
        )

        output = output_path or f"attribution_sample_{sample_index}.png"
        plot_sample_attribution(
            data=data,
            output_path=output,
            dpi=150,
        )
    else:
        # Only plot heatmap if no AUC data
        from src.visualization import plot_token_heatmap
        import matplotlib.pyplot as plt

        fig = plot_token_heatmap(
            tokens=tokens,
            scores=importance_scores,
            title=f"Token Attribution (Sample {sample_index})",
        )

        output = output_path or f"attribution_heatmap_{sample_index}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Visualization saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize token attribution")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--model", type=str, help="Model name for live attribution")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--method", type=str, default="attention_rollout",
                        choices=["attention_rollout", "attention_sum"],
                        help="Attribution method")
    parser.add_argument("--result-file", type=str, help="Path to saved attribution results")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in result file")
    parser.add_argument("--output", type=str, help="Output path for figure")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if args.demo:
        demo_visualization(args.output)
    elif args.result_file:
        visualize_from_file(args.result_file, args.sample_index, args.output)
    elif args.model and args.text:
        visualize_with_model(args.model, args.text, args.method, args.output, args.device)
    else:
        print("Usage:")
        print("  Demo mode: python scripts/visualize_sample.py --demo")
        print("  With model: python scripts/visualize_sample.py --model gpt2-xl --text 'Your text here'")
        print("  From file: python scripts/visualize_sample.py --result-file results/attribution.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
