"""
Random seed utilities.
"""

import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def temp_seed(seed: int):
    """
    Context manager for temporary random seed.

    Args:
        seed: Temporary seed value

    Usage:
        with temp_seed(42):
            # Random operations with seed 42
        # Original random state restored
    """
    # Save state
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]

    # Set temporary seed
    set_seed(seed)

    try:
        yield
    finally:
        # Restore state
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            for i, state in enumerate(cuda_states):
                torch.cuda.set_rng_state(state, i)
