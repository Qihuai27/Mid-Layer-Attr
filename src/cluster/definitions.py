"""
Task configuration and cluster point (anchor word) definitions.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import yaml


@dataclass
class TaskConfig:
    """Configuration for a classification task with cluster point definitions."""

    name: str
    description: str
    dataset_path: str
    text_field: str
    label_field: str
    labels: Dict[int, str]  # {0: " Negative", 1: " Positive"}
    prompt_template: str  # "Review: {text}\nSentiment:{label}"
    cluster_prefix: List[str]  # ["Sentiment", ":"] - tokens before label word

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def label_words(self) -> List[str]:
        """Return label words in order."""
        return [self.labels[i] for i in range(self.num_classes)]

    def format_example(self, text: str, label: Optional[int] = None) -> str:
        """Format a single example using the prompt template."""
        label_text = self.labels[label] if label is not None else ""
        return self.prompt_template.format(text=text, label=label_text)


def load_task_config(task_name: str, config_dir: Optional[str] = None) -> TaskConfig:
    """
    Load task configuration from YAML file.

    Args:
        task_name: Name of the task (e.g., 'sst2', 'agnews')
        config_dir: Directory containing task config files.
                   Defaults to configs/tasks/ relative to project root.

    Returns:
        TaskConfig object
    """
    if config_dir is None:
        # Default to configs/tasks/ relative to this file
        config_dir = Path(__file__).parent.parent.parent / "configs" / "tasks"

    config_path = Path(config_dir) / f"{task_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return TaskConfig(
        name=config['name'],
        description=config.get('description', ''),
        dataset_path=config['dataset']['path'],
        text_field=config['dataset']['text_field'],
        label_field=config['dataset']['label_field'],
        labels={int(k): v for k, v in config['labels'].items()},
        prompt_template=config['prompt_template'],
        cluster_prefix=config['anchor_pattern']['prefix'],  # YAML still uses anchor_pattern for compatibility
    )


def get_available_tasks(config_dir: Optional[str] = None) -> List[str]:
    """Get list of available task names from config directory."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs" / "tasks"

    config_dir = Path(config_dir)
    if not config_dir.exists():
        return []

    return [f.stem for f in config_dir.glob("*.yaml")]
