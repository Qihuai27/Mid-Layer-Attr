#!/usr/bin/env python3
"""
Batch run cluster experiments with attention_rollout on multiple models and datasets.
"""

import subprocess
import sys
from pathlib import Path

# Define experiments to run
EXPERIMENTS = [
    # Small models
    {"task": "sst2", "model": "phi-2", "shot": 1},
    {"task": "agnews", "model": "phi-2", "shot": 1},

    # Medium models
    {"task": "sst2", "model": "Qwen3-4B", "shot": 1},
    {"task": "agnews", "model": "Qwen3-4B", "shot": 1},

    {"task": "sst2", "model": "llama-3.2-1b", "shot": 1},
    {"task": "agnews", "model": "llama-3.2-1b", "shot": 1},

    {"task": "sst2", "model": "Llama-3.2-3B-Instruct", "shot": 1},
    {"task": "agnews", "model": "Llama-3.2-3B-Instruct", "shot": 1},
]

def run_experiment(task, model, shot):
    """Run a single cluster experiment."""
    cmd = [
        "python", "scripts/cluster/run_cluster.py",
        "--task", task,
        "--model", model,
        "--shot", str(shot),
        "--flow-metric", "attention_rollout",
        "--sample-size", "100",
        "--seeds", "42"
    ]

    print(f"\n{'='*80}")
    print(f"Running: {task} + {model} (shot={shot})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Completed: {task} + {model}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {task} + {model}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {task} + {model}")
        print(f"Exception: {e}")
        return False

def main():
    print(f"Starting batch experiments...")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Processing...")
        success = run_experiment(**exp)
        results.append((exp, success))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for _, success in results if success)
    print(f"Successful: {successful}/{len(EXPERIMENTS)}")
    print(f"Failed: {len(EXPERIMENTS) - successful}/{len(EXPERIMENTS)}")

    print("\nDetailed results:")
    for exp, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {exp['task']:10s} + {exp['model']:30s}")

    return 0 if successful == len(EXPERIMENTS) else 1

if __name__ == "__main__":
    sys.exit(main())
