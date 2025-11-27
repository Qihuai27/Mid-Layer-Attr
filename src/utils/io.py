"""
I/O utilities for saving and loading results.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def ensure_dir(path: Union[str, Path]):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_results(
    results: Any,
    path: Union[str, Path],
    format: str = "pickle",
    metadata: Optional[Dict] = None,
):
    """
    Save experiment results.

    Args:
        results: Results to save
        path: Output path
        format: 'pickle', 'numpy', or 'json'
        metadata: Optional metadata to save alongside results
    """
    path = Path(path)
    ensure_dir(path.parent)

    if format == "pickle":
        data = {"results": results, "metadata": metadata}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    elif format == "numpy":
        if isinstance(results, dict):
            np.savez(path, **results, metadata=metadata)
        else:
            np.save(path, results)

    elif format == "json":
        data = {"results": results, "metadata": metadata}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    path: Union[str, Path],
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load experiment results.

    Args:
        path: Path to results file
        format: File format (auto-detected if None)

    Returns:
        Dict with 'results' and optional 'metadata' keys
    """
    path = Path(path)

    if format is None:
        format = path.suffix.lstrip(".")
        if format in ("pkl",):
            format = "pickle"
        elif format in ("npy", "npz"):
            format = "numpy"

    if format == "pickle":
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "results" in data:
            return data
        return {"results": data, "metadata": None}

    elif format == "numpy":
        data = np.load(path, allow_pickle=True)
        if hasattr(data, "files"):  # npz file
            return dict(data)
        return {"results": data, "metadata": None}

    elif format == "json":
        with open(path, "r") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_experiment_name(
    task_name: str,
    model_name: str,
    shot: int,
    seeds: list,
    **kwargs,
) -> str:
    """
    Generate a standardized experiment name.

    Args:
        task_name: Name of the task
        model_name: Name of the model
        shot: Number of shots per class
        seeds: List of random seeds
        **kwargs: Additional parameters to include

    Returns:
        Experiment name string
    """
    parts = [
        task_name,
        model_name.replace("/", "-"),
        f"shot{shot}",
        f"seeds{'_'.join(map(str, seeds))}",
    ]

    for key, value in sorted(kwargs.items()):
        if value is not None:
            parts.append(f"{key}{value}")

    return "_".join(parts)
