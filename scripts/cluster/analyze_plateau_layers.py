#!/usr/bin/env python3
"""
Analyze attention_rollout cluster results to find plateau layers.
Find where S_a (aggregation) and S_o (output) stop accumulating.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import json
from collections import defaultdict

def find_plateau_layer(values, threshold=0.01, window=3):
    """
    Find the layer where values plateau (stop significantly increasing).

    Args:
        values: Array of values per layer
        threshold: Relative change threshold to consider as plateau
        window: Number of consecutive layers to check

    Returns:
        Layer index where plateau begins, or -1 if no plateau found
    """
    if len(values) < window + 1:
        return -1

    for i in range(len(values) - window):
        # Calculate relative changes in the window
        changes = []
        for j in range(i, i + window):
            if values[j] > 0:
                rel_change = abs(values[j+1] - values[j]) / values[j]
                changes.append(rel_change)

        # If all changes in window are below threshold, we found plateau
        if changes and all(c < threshold for c in changes):
            return i

    return -1

def find_plateau_layer_derivative(values, threshold=0.005, window=3):
    """
    Find plateau using derivative (slope) method.
    Plateau = where derivative approaches zero.
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
    for i in range(len(rel_derivatives) - window):
        window_derivs = rel_derivatives[i:i+window]
        if np.all(np.abs(window_derivs) < threshold):
            return i

    return -1

def find_saturation_layer(values, saturation_ratio=0.95):
    """
    Find layer where value reaches X% of maximum value.
    This is another way to define "plateau".
    """
    if len(values) == 0:
        return -1

    max_val = np.max(values)
    if max_val == 0:
        return -1

    threshold_val = saturation_ratio * max_val

    for i, val in enumerate(values):
        if val >= threshold_val:
            return i

    return -1

def analyze_result_file(filepath):
    """Analyze a single result pickle file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath.name}")
    print(f"{'='*80}")

    # Try to load with custom unpickler to handle missing modules
    try:
        with open(filepath, 'rb') as f:
            import pickle
            import io

            # Try standard pickle first
            try:
                data = pickle.load(f)
            except ModuleNotFoundError as e:
                # If that fails, try with a custom unpickler that ignores missing modules
                f.seek(0)

                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        # Try to find the class normally first
                        try:
                            return super().find_class(module, name)
                        except (ModuleNotFoundError, AttributeError):
                            # If not found, return a dummy class
                            print(f"  Warning: Could not find {module}.{name}, using placeholder")
                            return type(name, (), {})

                data = CustomUnpickler(f).load()
    except Exception as e:
        print(f"  Error loading file: {e}")
        return None

    # Check if data has the new structure (with 'results' key)
    if isinstance(data, dict) and 'results' in data:
        results_data = data['results']
        metadata = data.get('metadata', {})

        # Extract statistics from results
        stats = results_data.get('statistics', {})
        config = results_data.get('config', {})

        # Extract metadata
        task = metadata.get('task_name', 'unknown')
        model = metadata.get('model_name', 'unknown')
        flow_metric = config.get('flow_metric', 'unknown')
    else:
        # Old structure (direct access)
        stats = data.get('statistics', {})
        task = data.get('task', 'unknown')
        model = data.get('model', 'unknown')
        flow_metric = data.get('flow_metric', 'unknown')

    # Extract statistics
    mean_S_a = stats.get('mean_S_a', [])
    mean_S_o = stats.get('mean_S_o', [])
    mean_S_w = stats.get('mean_S_w', [])

    num_layers = len(mean_S_a)

    print(f"\nModel info:")
    print(f"  Layers: {num_layers}")
    print(f"  Task: {task}")
    print(f"  Model: {model}")
    print(f"  Flow metric: {flow_metric}")

    # Print per-layer statistics
    print(f"\nPer-layer statistics:")
    print(f"  {'Layer':<8} {'S_a':<12} {'S_o':<12} {'S_w':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(num_layers):
        print(f"  {i:<8} {mean_S_a[i]:<12.6f} {mean_S_o[i]:<12.6f} {mean_S_w[i]:<12.6f}")

    # Find plateau layers using multiple methods
    print(f"\nPlateau analysis:")

    # Method 1: Relative change threshold
    plateau_S_a_method1 = find_plateau_layer(mean_S_a, threshold=0.01, window=3)
    plateau_S_o_method1 = find_plateau_layer(mean_S_o, threshold=0.01, window=3)

    print(f"  Method 1 (rel change < 1% for 3 layers):")
    print(f"    S_a plateau at layer: {plateau_S_a_method1}")
    print(f"    S_o plateau at layer: {plateau_S_o_method1}")

    # Method 2: Derivative method
    plateau_S_a_method2 = find_plateau_layer_derivative(mean_S_a, threshold=0.005, window=3)
    plateau_S_o_method2 = find_plateau_layer_derivative(mean_S_o, threshold=0.005, window=3)

    print(f"  Method 2 (derivative < 0.5% for 3 layers):")
    print(f"    S_a plateau at layer: {plateau_S_a_method2}")
    print(f"    S_o plateau at layer: {plateau_S_o_method2}")

    # Method 3: Saturation (95% of max)
    plateau_S_a_method3 = find_saturation_layer(mean_S_a, saturation_ratio=0.95)
    plateau_S_o_method3 = find_saturation_layer(mean_S_o, saturation_ratio=0.95)

    print(f"  Method 3 (reaches 95% of max):")
    print(f"    S_a plateau at layer: {plateau_S_a_method3}")
    print(f"    S_o plateau at layer: {plateau_S_o_method3}")

    # Final values
    print(f"\nFinal values:")
    print(f"  S_a[-1]: {mean_S_a[-1]:.6f}")
    print(f"  S_o[-1]: {mean_S_o[-1]:.6f}")
    print(f"  S_w[-1]: {mean_S_w[-1]:.6f}")

    return {
        'filepath': str(filepath),
        'task': task,
        'model': model,
        'num_layers': num_layers,
        'mean_S_a': mean_S_a,
        'mean_S_o': mean_S_o,
        'mean_S_w': mean_S_w,
        'plateau_S_a_method1': plateau_S_a_method1,
        'plateau_S_o_method1': plateau_S_o_method1,
        'plateau_S_a_method2': plateau_S_a_method2,
        'plateau_S_o_method2': plateau_S_o_method2,
        'plateau_S_a_method3': plateau_S_a_method3,
        'plateau_S_o_method3': plateau_S_o_method3,
        'final_S_a': mean_S_a[-1] if len(mean_S_a) > 0 else 0,
        'final_S_o': mean_S_o[-1] if len(mean_S_o) > 0 else 0,
        'final_S_w': mean_S_w[-1] if len(mean_S_w) > 0 else 0,
    }

def generate_summary_table(results):
    """Generate a summary table of all results."""
    print(f"\n{'='*120}")
    print("SUMMARY TABLE: Attention Rollout Plateau Layers")
    print(f"{'='*120}")

    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r['model']].append(r)

    # Print header
    print(f"\n{'Model':<30} {'Task':<10} {'Layers':<8} {'S_a Plateau':<15} {'S_o Plateau':<15} {'Final S_a':<12} {'Final S_o':<12}")
    print(f"{'-'*30} {'-'*10} {'-'*8} {'-'*15} {'-'*15} {'-'*12} {'-'*12}")

    for model, model_results in sorted(by_model.items()):
        for r in model_results:
            # Use method 2 (derivative) as primary, fallback to method 3
            s_a_plateau = r['plateau_S_a_method2']
            if s_a_plateau < 0:
                s_a_plateau = r['plateau_S_a_method3']

            s_o_plateau = r['plateau_S_o_method2']
            if s_o_plateau < 0:
                s_o_plateau = r['plateau_S_o_method3']

            s_a_str = str(s_a_plateau) if s_a_plateau >= 0 else "N/A"
            s_o_str = str(s_o_plateau) if s_o_plateau >= 0 else "N/A"

            print(f"{model:<30} {r['task']:<10} {r['num_layers']:<8} {s_a_str:<15} {s_o_str:<15} {r['final_S_a']:<12.6f} {r['final_S_o']:<12.6f}")

    print(f"{'-'*120}")

    # Print interpretation
    print(f"\nInterpretation:")
    print(f"  - Plateau layer: The layer where information stops accumulating (derivative < 0.5% for 3 consecutive layers)")
    print(f"  - S_a (Aggregation): Information flow from preceding text TO cluster points")
    print(f"  - S_o (Output): Information flow from cluster points TO final prediction position")
    print(f"  - Lower plateau layer = information accumulates faster in early layers")
    print(f"  - Higher plateau layer = information continues to accumulate through deeper layers")

def main():
    results_dir = project_root / "results/cluster/raw"

    # Find all attention_rollout result files
    result_files = list(results_dir.glob("*attention_rollout*.pkl"))

    # Also check in subdirectories
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            result_files.extend(subdir.glob("*attention_rollout*.pkl"))

    if not result_files:
        print("No attention_rollout result files found!")
        return

    print(f"Found {len(result_files)} result files")

    # Analyze each file
    all_results = []
    for filepath in sorted(result_files):
        try:
            result = analyze_result_file(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {filepath.name}: {e}")
            continue

    # Generate summary table
    if all_results:
        generate_summary_table(all_results)

        # Save to JSON
        output_file = project_root / "results/cluster/plateau_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved detailed results to: {output_file}")
    else:
        print("\nNo results to summarize.")

if __name__ == "__main__":
    main()
