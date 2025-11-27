"""
Anchor word detection and definitions module.
"""

from .detector import AnchorDetector, AnchorPositions
from .definitions import TaskConfig, load_task_config

__all__ = ["AnchorDetector", "AnchorPositions", "TaskConfig", "load_task_config"]
