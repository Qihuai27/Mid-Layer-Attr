"""
Ablation experiment runner.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.base import BaseExperiment, ExperimentConfig
from src.anchors import AnchorDetector, load_task_config
from src.data import load_dataset, DemonstrationSampler, ICLWrapper
from src.models import load_model_and_tokenizer
from src.models.hooks.gpt2 import GPT2MaskingHookManager
from src.models.hooks.gptj import GPTJMaskingHookManager
from src.utils import set_seed, save_results

from .interventions import (
    AttentionIntervention,
    MaskAnchorAttention,
    MaskNonAnchorAttention,
    InterventionSpec,
)
from .evaluator import AblationEvaluator, AblationResult


@dataclass
class AblationConfig(ExperimentConfig):
    """Configuration for ablation experiments."""

    save_dir: str = "results/ablation"

    # Layer masking settings
    mask_layer_num: int = 5
    mask_layer_pos: str = "first"  # 'first' or 'last'

    # Intervention type
    intervention_type: str = "anchor"  # 'anchor' or 'non_anchor'


class AblationExperiment(BaseExperiment):
    """
    Causal ablation experiment.

    Tests the causal role of anchor words by masking their attention
    and measuring the effect on classification accuracy.
    """

    def __init__(self, config: AblationConfig):
        super().__init__(config)
        self.config: AblationConfig = config

    def _get_hook_manager(self, model):
        """Get appropriate masking hook manager for the model."""
        model_name_lower = self.config.model_name.lower()
        if "gpt2" in model_name_lower:
            return GPT2MaskingHookManager(model)
        elif "gpt-j" in model_name_lower or "gptj" in model_name_lower:
            return GPTJMaskingHookManager(model)
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

    def _get_layer_indices(self, num_layers: int) -> List[int]:
        """Get layer indices to mask based on config."""
        if self.config.mask_layer_pos == "first":
            return list(range(self.config.mask_layer_num))
        else:  # last
            return list(range(num_layers - self.config.mask_layer_num, num_layers))

    def run(self) -> Dict[str, Any]:
        """
        Run the ablation experiment.

        Returns:
            Dict containing:
            - results: List of AblationResult per seed
            - config: Experiment configuration
        """
        # Load task config
        task_config = load_task_config(self.config.task_name)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.config.device,
        )

        # Create anchor detector
        anchor_detector = AnchorDetector(task_config, tokenizer)
        label_token_ids = anchor_detector.get_label_id_dict()

        # Get number of layers
        num_layers = len(model.transformer.h)
        layer_indices = self._get_layer_indices(num_layers)

        # Create intervention
        if self.config.intervention_type == "anchor":
            intervention = MaskAnchorAttention()
        else:
            intervention = MaskNonAnchorAttention()

        intervention_spec = InterventionSpec(
            intervention=intervention,
            layer_indices=layer_indices,
        )

        # Set up masking hook manager
        hook_manager = self._get_hook_manager(model)

        # Create evaluator
        evaluator = AblationEvaluator(
            model=model,
            tokenizer=tokenizer,
            model_name=self.config.model_name,
            anchor_detector=anchor_detector,
            label_token_ids=label_token_ids,
            device=self.config.device,
        )
        # Replace hook manager with masking version
        evaluator.hook_manager = hook_manager

        # Load dataset
        dataset = load_dataset(task_config)

        all_results = []

        for seed in self.config.seeds:
            set_seed(seed)
            print(f"\nRunning with seed {seed}")

            # Sample demonstrations and test examples
            sampler = DemonstrationSampler(dataset["train"], task_config, seed)
            demonstrations = sampler.sample(self.config.demonstration_shot)

            # Get test samples
            test_dataset = dataset["test"]
            if len(test_dataset) > self.config.sample_size:
                test_indices = np.random.choice(
                    len(test_dataset), self.config.sample_size, replace=False
                )
                test_dataset = test_dataset.select(test_indices)

            # Create ICL wrapper
            wrapper = ICLWrapper(task_config, tokenizer)
            wrapped_dataset = wrapper.wrap_dataset(demonstrations, test_dataset)
            tokenized_dataset = wrapper.tokenize_dataset(wrapped_dataset)

            # Convert to torch format
            tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            # Create dataloader
            def collate_fn(batch):
                return {
                    "input_ids": torch.stack([x["input_ids"] for x in batch]),
                    "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
                    "labels": torch.tensor([x["label"] for x in batch]),
                }

            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_fn,
            )

            # Run ablation
            result = evaluator.run_ablation(dataloader, intervention_spec)
            all_results.append(result)

            print(f"  Baseline accuracy: {result.baseline_accuracy:.4f}")
            print(f"  Intervened accuracy: {result.intervened_accuracy:.4f}")
            print(f"  Accuracy drop: {result.accuracy_drop:.4f}")

        # Aggregate results
        avg_baseline = np.mean([r.baseline_accuracy for r in all_results])
        avg_intervened = np.mean([r.intervened_accuracy for r in all_results])
        avg_drop = np.mean([r.accuracy_drop for r in all_results])

        results = {
            "per_seed_results": [r.to_dict() for r in all_results],
            "aggregated": {
                "mean_baseline_accuracy": float(avg_baseline),
                "mean_intervened_accuracy": float(avg_intervened),
                "mean_accuracy_drop": float(avg_drop),
                "std_accuracy_drop": float(np.std([r.accuracy_drop for r in all_results])),
            },
            "intervention": intervention_spec.describe(),
            "config": self.config.to_dict(),
        }

        return results

    def save_results(self, results: Dict[str, Any], path: Optional[str] = None):
        """Save experiment results."""
        if path is None:
            path = self.get_save_path(
                suffix=f"mask{self.config.mask_layer_num}_{self.config.mask_layer_pos}"
            )
        save_results(results, path, format="pickle", metadata=self.config.to_dict())


def run_ablation_experiment(
    task_name: str = "sst2",
    model_name: str = "gpt2-xl",
    demonstration_shot: int = 1,
    sample_size: int = 100,
    seeds: List[int] = None,
    device: str = "cuda:0",
    mask_layer_num: int = 5,
    mask_layer_pos: str = "first",
    intervention_type: str = "anchor",
    save_dir: str = "results/ablation",
) -> Dict[str, Any]:
    """
    Convenience function to run ablation experiment.

    Args:
        task_name: Task to run on
        model_name: Model to use
        demonstration_shot: Number of demonstrations per class
        sample_size: Number of test samples
        seeds: Random seeds
        device: Device to run on
        mask_layer_num: Number of layers to mask
        mask_layer_pos: 'first' or 'last' layers to mask
        intervention_type: 'anchor' or 'non_anchor'
        save_dir: Directory to save results

    Returns:
        Experiment results
    """
    if seeds is None:
        seeds = [42, 43, 44]

    config = AblationConfig(
        task_name=task_name,
        model_name=model_name,
        demonstration_shot=demonstration_shot,
        sample_size=sample_size,
        seeds=seeds,
        device=device,
        mask_layer_num=mask_layer_num,
        mask_layer_pos=mask_layer_pos,
        intervention_type=intervention_type,
        save_dir=save_dir,
    )

    experiment = AblationExperiment(config)
    results = experiment.run()
    experiment.save_results(results)

    return results
