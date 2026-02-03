"""
Dataset loading utilities.
"""

from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset

from ..cluster.definitions import TaskConfig


def load_dataset(
    task_config: TaskConfig,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load a dataset based on task configuration.

    Args:
        task_config: Task configuration containing dataset info
        cache_dir: Optional directory for caching

    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    dataset_name = task_config.dataset_path

    # Handle special dataset names
    if dataset_name == "sst2":
        dataset = hf_load_dataset("glue", "sst2", cache_dir=cache_dir)
        # Rename validation to test for consistency
        if "validation" in dataset:
            dataset["test"] = dataset["validation"]
            del dataset["validation"]

    elif dataset_name == "ag_news":
        dataset = hf_load_dataset("ag_news", cache_dir=cache_dir)

    elif dataset_name == "trec":
        dataset = hf_load_dataset("trec", cache_dir=cache_dir)
        # TREC uses coarse_label by default
        def rename_label(example):
            example["label"] = example["coarse_label"]
            return example
        dataset = dataset.map(rename_label)

    elif dataset_name == "emotion":
        dataset = hf_load_dataset("emotion", cache_dir=cache_dir)

    else:
        # Try loading directly
        dataset = hf_load_dataset(dataset_name, cache_dir=cache_dir)

    # Ensure required fields exist
    required_fields = [task_config.text_field, task_config.label_field]
    for split in dataset:
        for field in required_fields:
            if field not in dataset[split].features:
                raise ValueError(
                    f"Field '{field}' not found in dataset '{dataset_name}' split '{split}'. "
                    f"Available fields: {list(dataset[split].features.keys())}"
                )

    return dataset


def get_train_test_split(
    dataset: DatasetDict,
) -> Tuple[Dataset, Dataset]:
    """
    Extract train and test splits from a dataset.

    Args:
        dataset: DatasetDict

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train = dataset.get("train")
    test = dataset.get("test") or dataset.get("validation")

    if train is None:
        raise ValueError("Dataset does not have a 'train' split")
    if test is None:
        raise ValueError("Dataset does not have a 'test' or 'validation' split")

    return train, test
