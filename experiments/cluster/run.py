"""
Cluster experiment runner.

Analyzes how attention flows through cluster points (anchor words) by computing
layer-wise statistics on attention patterns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from experiments.base import BaseExperiment, ExperimentConfig
from src.cluster import (
    ClusterDetector,
    RandomClusterDetector,
    load_task_config,
    AttentionExtractor,
    FlowMetric,
    get_flow_computer,
    compute_cluster_statistics,
    compute_all_layer_statistics,
    aggregate_statistics,
)
from src.data import load_dataset, DemonstrationSampler, ICLWrapper
from src.models import load_model_and_tokenizer
from src.utils import set_seed, save_results


@dataclass
class ClusterConfig(ExperimentConfig):
    """Configuration for cluster experiments."""

    save_dir: str = "results/cluster"
    use_gradients: bool = True  # Whether to use gradient-based saliency
    flow_metric: str = "gradient_saliency"  # Information flow metric to use
    use_random_cluster: bool = False  # Use random cluster detection instead of pattern matching
    num_clusters_per_label: int = 2  # Number of random clusters per label (when use_random_cluster=True)

    @property
    def experiment_name(self) -> str:
        """Generate experiment name including flow metric."""
        cluster_type = "random" if self.use_random_cluster else "pattern"
        return (
            f"{self.task_name}_{self.model_name}_"
            f"shot{self.demonstration_shot}_"
            f"{self.flow_metric}_{cluster_type}_"
            f"seeds{'_'.join(map(str, self.seeds))}"
        )


class ClusterExperiment(BaseExperiment):
    """
    Cluster (information aggregation) experiment.

    Analyzes how attention flows through cluster points by computing
    layer-wise statistics on attention patterns.
    """

    def __init__(self, config: ClusterConfig):
        super().__init__(config)
        self.config: ClusterConfig = config

    def run(self) -> Dict[str, Any]:
        """
        Run the cluster experiment.

        Returns:
            Dict containing:
            - statistics: Aggregated statistics across all samples
            - per_sample: Per-sample statistics
            - config: Experiment configuration
        """
        # Load task config
        task_config = load_task_config(self.config.task_name)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.config.device,
        )

        # Set model to eval mode (disables dropout)
        model.eval()

        # Create cluster detector
        if self.config.use_random_cluster:
            cluster_detector = RandomClusterDetector(
                task_config, tokenizer,
                num_clusters_per_label=self.config.num_clusters_per_label,
            )
            print(f"Using RandomClusterDetector with {self.config.num_clusters_per_label} clusters per label")
        else:
            cluster_detector = ClusterDetector(task_config, tokenizer)

        # Create attention extractor
        extractor = AttentionExtractor(model, self.config.model_name)

        # Create flow computer based on config
        flow_metric = FlowMetric(self.config.flow_metric)
        flow_computer = get_flow_computer(flow_metric)
        print(f"Using flow metric: {flow_computer.name}")

        # Check if gradient is needed - only use if flow_computer requires it
        needs_gradient = flow_computer.requires_gradient

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

            # Run cluster analysis
            seed_stats = []

            for idx in tqdm(range(len(tokenized_dataset)), desc=f"Seed {seed}"):
                example = tokenized_dataset[idx]

                # Prepare inputs
                input_ids = torch.tensor([example["input_ids"]], device=self.config.device)
                attention_mask = torch.tensor(
                    [example["attention_mask"]], device=self.config.device
                )
                labels = torch.tensor([example["label"]], device=self.config.device)

                # Detect cluster positions
                cluster_positions = cluster_detector.detect(input_ids)

                # Extract attention (with gradients if needed)
                if needs_gradient:
                    graphs, gradients = extractor.extract_with_gradients(
                        input_ids, labels, attention_mask
                    )
                else:
                    graphs = extractor.extract(input_ids, attention_mask)
                    gradients = None

                # Compute statistics with selected flow metric
                # Handle rollout specially (needs all layer weights)
                if flow_computer.is_multi_layer:
                    flow_computer.reset_cache()
                    all_weights = [g.weights for g in graphs]
                    stats = []
                    for i, graph in enumerate(graphs):
                        grad = gradients[i] if gradients else None
                        stat = compute_cluster_statistics(
                            graph, cluster_positions,
                            flow_computer=flow_computer,
                            gradient=grad,
                            layer_idx=i,
                            all_layer_weights=all_weights,
                        )
                        stats.append(stat)
                else:
                    stats = compute_all_layer_statistics(
                        graphs, cluster_positions,
                        flow_computer=flow_computer,
                        gradients=gradients,
                    )
                seed_stats.append(stats)

                # Clear intermediate tensors to free GPU memory
                del graphs
                if gradients is not None:
                    del gradients
                torch.cuda.empty_cache()

            all_results.append(seed_stats)

        # Aggregate results across all seeds and samples
        flat_stats = [stat for seed_stats in all_results for stat in seed_stats]
        aggregated = aggregate_statistics(flat_stats)

        results = {
            "statistics": aggregated,
            "per_seed": all_results,
            "config": self.config.to_dict(),
        }

        return results

    def save_results(self, results: Dict[str, Any], path: Optional[str] = None):
        """Save experiment results."""
        if path is None:
            path = self.get_save_path()
        save_results(results, path, format="pickle", metadata=self.config.to_dict())


def run_cluster_experiment(
    task_name: str = "sst2",
    model_name: str = "gpt2-xl",
    demonstration_shot: int = 1,
    sample_size: int = 100,
    seeds: List[int] = None,
    device: str = "cuda:0",
    use_gradients: bool = True,
    flow_metric: str = "gradient_saliency",
    save_dir: str = "results/cluster",
    use_random_cluster: bool = False,
    num_clusters_per_label: int = 2,
) -> Dict[str, Any]:
    """
    Convenience function to run cluster experiment.

    Args:
        task_name: Task to run on
        model_name: Model to use
        demonstration_shot: Number of demonstrations per class
        sample_size: Number of test samples
        seeds: Random seeds
        device: Device to run on
        use_gradients: Whether to use gradient-based saliency (deprecated, use flow_metric instead)
        flow_metric: Information flow metric to use. Options:
            - "attention_sum": Direct attention weights
            - "gradient_saliency": |A ⊙ ∂L/∂A| (default, from paper)
            - "attention_rollout": Cumulative attention across layers
            - "attention_value_weighted": Attention × value vector norm
        save_dir: Directory to save results
        use_random_cluster: Use random cluster detection instead of pattern matching
        num_clusters_per_label: Number of random clusters per label (when use_random_cluster=True)

    Returns:
        Experiment results
    """
    if seeds is None:
        seeds = [42, 43, 44]

    config = ClusterConfig(
        task_name=task_name,
        model_name=model_name,
        demonstration_shot=demonstration_shot,
        sample_size=sample_size,
        seeds=seeds,
        device=device,
        use_gradients=use_gradients,
        flow_metric=flow_metric,
        save_dir=save_dir,
        use_random_cluster=use_random_cluster,
        num_clusters_per_label=num_clusters_per_label,
    )

    experiment = ClusterExperiment(config)
    results = experiment.run()
    experiment.save_results(results)

    return results
