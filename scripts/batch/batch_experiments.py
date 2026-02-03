#!/usr/bin/env python
"""
Generate batch experiment scripts for running on multiple GPUs.

Usage:
    python scripts/batch_experiments.py --experiment attribution --gpus 0 1 2 3
"""

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Any


def generate_commands(
    experiment: str,
    tasks: List[str],
    models: List[str],
    shots: List[int],
    **kwargs,
) -> List[str]:
    """Generate all experiment commands."""
    commands = []

    if experiment == "attribution":
        script = "scripts/run_attribution.py"
        for task, model, shot in itertools.product(tasks, models, shots):
            cmd = f"python {script} --task {task} --model {model} --shot {shot}"
            commands.append(cmd)

    elif experiment == "ablation":
        script = "scripts/run_ablation.py"
        mask_layers = kwargs.get("mask_layers", [1, 3, 5])
        mask_positions = kwargs.get("mask_positions", ["first", "last"])

        for task, model, shot, layers, pos in itertools.product(
            tasks, models, shots, mask_layers, mask_positions
        ):
            cmd = (
                f"python {script} --task {task} --model {model} --shot {shot} "
                f"--mask-layers {layers} --mask-pos {pos}"
            )
            commands.append(cmd)

    return commands


def distribute_to_gpus(commands: List[str], gpu_list: List[int]) -> Dict[int, List[str]]:
    """Distribute commands across GPUs."""
    gpu_commands = {gpu: [] for gpu in gpu_list}

    for i, cmd in enumerate(commands):
        gpu = gpu_list[i % len(gpu_list)]
        # Add CUDA device to command
        cmd_with_gpu = f"CUDA_VISIBLE_DEVICES={gpu} {cmd} --device cuda:0"
        gpu_commands[gpu].append(cmd_with_gpu)

    return gpu_commands


def write_scripts(gpu_commands: Dict[int, List[str]], output_dir: str = "."):
    """Write shell scripts for each GPU."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write per-GPU scripts
    for gpu, commands in gpu_commands.items():
        script_path = output_path / f"gpu_{gpu}.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# GPU {gpu} experiments\n\n")
            for cmd in commands:
                f.write(f"{cmd}\n")
        script_path.chmod(0o755)

    # Write master script
    master_path = output_path / "run_all.sh"
    with open(master_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Run all experiments in parallel\n\n")
        for gpu in gpu_commands:
            f.write(f"bash {output_path}/gpu_{gpu}.sh &\n")
        f.write("\nwait\necho 'All experiments completed!'\n")
    master_path.chmod(0o755)

    print(f"Generated scripts in {output_path}:")
    print(f"  - run_all.sh (master script)")
    for gpu in gpu_commands:
        print(f"  - gpu_{gpu}.sh ({len(gpu_commands[gpu])} commands)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate batch experiment scripts")

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["attribution", "ablation", "both"],
        help="Type of experiment",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["sst2", "agnews", "trec", "emo"],
        help="Tasks to run",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt2-xl"],
        help="Models to use",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1],
        help="Shot numbers",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="GPU indices to use",
    )
    parser.add_argument(
        "--mask-layers",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Mask layer numbers (for ablation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/batch",
        help="Output directory for scripts",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    experiments = [args.experiment] if args.experiment != "both" else ["attribution", "ablation"]

    all_commands = []
    for exp in experiments:
        commands = generate_commands(
            experiment=exp,
            tasks=args.tasks,
            models=args.models,
            shots=args.shots,
            mask_layers=args.mask_layers,
        )
        all_commands.extend(commands)

    print(f"Generated {len(all_commands)} experiment commands")

    gpu_commands = distribute_to_gpus(all_commands, args.gpus)
    write_scripts(gpu_commands, args.output_dir)


if __name__ == "__main__":
    main()
