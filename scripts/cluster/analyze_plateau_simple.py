#!/usr/bin/env python3
"""Analyze plateau layers from cluster experiment results."""

import pickle
import numpy as np
from pathlib import Path

def find_plateau_layer(values, threshold=0.005, window=3):
    """
    Find the layer where values plateau using derivative method.

    Args:
        values: Array of values per layer
        threshold: Relative change threshold (0.5%)
        window: Number of consecutive layers to check

    Returns:
        Layer index where plateau begins, or -1 if no plateau found
    """
    if len(values) < window + 2:
        return -1

    # Calculate derivatives
    derivatives = np.diff(values)

    # Normalize by current value to get relative slope
    rel_derivatives = np.zeros(len(derivatives))
    for i in range(len(derivatives)):
        if values[i] > 0:
            rel_derivatives[i] = derivatives[i] / values[i]

    # Find where derivative stays small
    for i in range(len(rel_derivatives) - window + 1):
        window_derivs = rel_derivatives[i:i+window]
        if np.all(np.abs(window_derivs) < threshold):
            return i

    return -1

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a dummy class for missing modules
            return type(name, (), {})

def analyze_result_file(filepath):
    """Analyze a single result file."""
    with open(filepath, 'rb') as f:
        data = CustomUnpickler(f).load()

    results = data['results']
    metadata = data.get('metadata', {})

    stats = results['statistics']
    config = results['config']

    mean_S_a = stats['mean_S_a']
    mean_S_o = stats['mean_S_o']
    mean_S_w = stats['mean_S_w']

    num_layers = len(mean_S_a)
    task = metadata.get('task_name', config.get('task_name', 'unknown'))
    model = metadata.get('model_name', config.get('model_name', 'unknown'))

    # Find plateau layers
    plateau_S_a = find_plateau_layer(mean_S_a, threshold=0.005, window=3)
    plateau_S_o = find_plateau_layer(mean_S_o, threshold=0.005, window=3)

    return {
        'task': task,
        'model': model,
        'num_layers': num_layers,
        'mean_S_a': mean_S_a,
        'mean_S_o': mean_S_o,
        'mean_S_w': mean_S_w,
        'plateau_S_a': plateau_S_a,
        'plateau_S_o': plateau_S_o,
        'final_S_a': mean_S_a[-1],
        'final_S_o': mean_S_o[-1],
        'final_S_w': mean_S_w[-1],
    }

def main():
    result_files = [
        'results/cluster/sst2_phi-2_shot1_attention_rollout_pattern_seeds42.pkl',
        'results/cluster/agnews_phi-2_shot1_attention_rollout_pattern_seeds42.pkl',
        'results/cluster/sst2_Qwen3-4B_shot1_attention_rollout_pattern_seeds42.pkl',
        'results/cluster/agnews_Qwen3-4B_shot1_attention_rollout_pattern_seeds42.pkl',
    ]

    print("="*100)
    print("Cluster Experiment Plateau Analysis")
    print("="*100)
    print()

    all_results = []
    for filepath in result_files:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue

        result = analyze_result_file(filepath)
        all_results.append(result)

        print(f"File: {filepath.name}")
        print(f"  Task: {result['task']}")
        print(f"  Model: {result['model']}")
        print(f"  Layers: {result['num_layers']}")
        print(f"  S_a plateau layer: {result['plateau_S_a']} (relative change < 0.5% for 3 layers)")
        print(f"  S_o plateau layer: {result['plateau_S_o']} (relative change < 0.5% for 3 layers)")
        print(f"  Final S_a: {result['final_S_a']:.6f}")
        print(f"  Final S_o: {result['final_S_o']:.9f}")  # More precision for S_o
        print(f"  Max S_o: {np.max(result['mean_S_o']):.6f}")  # Show max S_o
        print()

    # Generate summary table
    print("="*100)
    print("SUMMARY TABLE: Attention Rollout Plateau Layers")
    print("="*100)
    print()
    print(f"{'Model':<20} {'Task':<10} {'Layers':<8} {'S_a Plateau':<15} {'S_o Plateau':<15} {'Final S_a':<12} {'Max S_o':<12}")
    print(f"{'-'*20} {'-'*10} {'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")

    for r in all_results:
        s_a_str = str(r['plateau_S_a']) if r['plateau_S_a'] >= 0 else "N/A"
        s_o_str = str(r['plateau_S_o']) if r['plateau_S_o'] >= 0 else "N/A"
        max_s_o = np.max(r['mean_S_o'])

        print(f"{r['model']:<20} {r['task']:<10} {r['num_layers']:<8} {s_a_str:<15} {s_o_str:<15} {r['final_S_a']:<12.6f} {max_s_o:<12.6f}")

    print()
    print("="*100)
    print()
    print("Interpretation:")
    print("  - Plateau layer: The layer where information stops accumulating significantly")
    print("                   (relative change < 0.5% for 3 consecutive layers)")
    print("  - S_a (Aggregation): Information flow from preceding text TO cluster positions")
    print("  - S_o (Output): Information flow from cluster positions TO final prediction")
    print("  - Max S_o shown instead of Final S_o (which approaches zero in deeper layers)")
    print("  - Lower plateau layer = information accumulates faster in early layers")
    print("  - Higher plateau layer = information continues accumulating through deeper layers")
    print()
    print("Observation:")
    print("  - S_a plateaus around layers 6-7 for both models")
    print("  - S_o decreases from small values (~0.003-0.005) to near-zero, showing no clear plateau")
    print("  - This suggests cluster positions aggregate information early, but contribute little to final prediction")
    print()

if __name__ == "__main__":
    main()
