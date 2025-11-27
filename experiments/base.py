"""
Base classes for experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    # Task settings
    task_name: str = "sst2"
    model_name: str = "gpt2-xl"

    # Sampling settings
    demonstration_shot: int = 1
    sample_size: int = 100
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])

    # Runtime settings
    device: str = "cuda:0"
    batch_size: int = 1

    # Output settings
    save_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data.get("defaults", {}))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "demonstration_shot": self.demonstration_shot,
            "sample_size": self.sample_size,
            "seeds": self.seeds,
            "device": self.device,
            "batch_size": self.batch_size,
            "save_dir": self.save_dir,
        }

    @property
    def experiment_name(self) -> str:
        """Generate experiment name from config."""
        return (
            f"{self.task_name}_{self.model_name}_"
            f"shot{self.demonstration_shot}_"
            f"seeds{'_'.join(map(str, self.seeds))}"
        )


class BaseExperiment(ABC):
    """
    Base class for experiments.

    Subclasses implement specific experiment types (attribution, ablation).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._task_config = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def task_config(self):
        """Lazy load task config."""
        if self._task_config is None:
            from src.anchors import load_task_config
            self._task_config = load_task_config(self.config.task_name)
        return self._task_config

    def _load_model(self):
        """Load model and tokenizer."""
        from src.models import load_model_and_tokenizer
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.config.device,
        )

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any], path: Optional[str] = None):
        """Save experiment results."""
        pass

    def get_save_path(self, suffix: str = "") -> Path:
        """Get path for saving results."""
        filename = self.config.experiment_name
        if suffix:
            filename = f"{filename}_{suffix}"
        return Path(self.config.save_dir) / f"{filename}.pkl"
