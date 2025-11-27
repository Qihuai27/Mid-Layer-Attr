"""
Demonstration sampling strategies for ICL.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from ..anchors.definitions import TaskConfig


class DemonstrationSampler:
    """
    Sampler for selecting demonstrations for in-context learning.

    Supports stratified sampling to ensure balanced class representation.
    """

    def __init__(
        self,
        dataset: Dataset,
        task_config: TaskConfig,
        seed: int = 42,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: Dataset to sample from (typically training set)
            task_config: Task configuration
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.task_config = task_config
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Index examples by class
        self.label_field = task_config.label_field
        self.class_indices = self._build_class_indices()

    def _build_class_indices(self) -> Dict[int, np.ndarray]:
        """Build index of examples per class."""
        labels = np.array(self.dataset[self.label_field])
        class_indices = {}

        for class_id in range(self.task_config.num_classes):
            indices = np.where(labels == class_id)[0]
            class_indices[class_id] = indices

        return class_indices

    def sample(
        self,
        shot_per_class: int,
        exclude_indices: Optional[List[int]] = None,
    ) -> Dataset:
        """
        Sample demonstrations with stratified sampling.

        Args:
            shot_per_class: Number of examples per class
            exclude_indices: Indices to exclude from sampling

        Returns:
            Dataset containing sampled demonstrations
        """
        exclude_set = set(exclude_indices) if exclude_indices else set()
        sampled_indices = []

        for class_id, indices in self.class_indices.items():
            # Filter out excluded indices
            available = [i for i in indices if i not in exclude_set]

            if len(available) < shot_per_class:
                raise ValueError(
                    f"Not enough examples for class {class_id}: "
                    f"need {shot_per_class}, have {len(available)}"
                )

            # Sample without replacement
            selected = self.rng.choice(available, size=shot_per_class, replace=False)
            sampled_indices.extend(selected)

        # Shuffle the selected examples
        self.rng.shuffle(sampled_indices)

        return self.dataset.select(sampled_indices)

    def sample_with_test(
        self,
        shot_per_class: int,
        test_size: int,
        test_dataset: Optional[Dataset] = None,
    ) -> Tuple[Dataset, Dataset]:
        """
        Sample both demonstrations and test examples.

        Args:
            shot_per_class: Number of demonstration examples per class
            test_size: Number of test examples to sample
            test_dataset: Dataset to sample test examples from (defaults to same dataset)

        Returns:
            Tuple of (demonstrations, test_examples)
        """
        # Sample demonstrations
        demonstrations = self.sample(shot_per_class)
        demo_indices = list(range(len(demonstrations)))

        # Sample test examples
        if test_dataset is None:
            test_dataset = self.dataset
            # Exclude demonstration indices when sampling from same dataset
            available_test = [
                i for i in range(len(test_dataset))
                if i not in set(demonstrations._indices or range(len(demonstrations)))
            ]
        else:
            available_test = list(range(len(test_dataset)))

        if len(available_test) < test_size:
            test_size = len(available_test)

        test_indices = self.rng.choice(available_test, size=test_size, replace=False)
        test_examples = test_dataset.select(test_indices)

        return demonstrations, test_examples


def sample_demonstrations(
    dataset: Dataset,
    task_config: TaskConfig,
    shot_per_class: int,
    seed: int = 42,
) -> Dataset:
    """
    Convenience function to sample demonstrations.

    Args:
        dataset: Dataset to sample from
        task_config: Task configuration
        shot_per_class: Number of examples per class
        seed: Random seed

    Returns:
        Sampled demonstrations
    """
    sampler = DemonstrationSampler(dataset, task_config, seed)
    return sampler.sample(shot_per_class)
