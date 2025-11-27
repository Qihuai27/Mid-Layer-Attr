"""
Evaluation utilities for ablation experiments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.anchors import AnchorDetector
from src.models.hooks import HookManager, get_hook_manager
from src.models.hooks.base import HookMode, MaskingHook

from .interventions import AttentionIntervention, InterventionSpec


@dataclass
class AblationResult:
    """Results from an ablation experiment."""

    # Accuracy metrics
    baseline_accuracy: float
    intervened_accuracy: float
    accuracy_drop: float

    # Per-sample predictions
    baseline_predictions: np.ndarray
    intervened_predictions: np.ndarray
    labels: np.ndarray

    # Per-sample probabilities
    baseline_probs: Optional[np.ndarray] = None
    intervened_probs: Optional[np.ndarray] = None

    # Intervention details
    intervention_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "intervened_accuracy": self.intervened_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "baseline_predictions": self.baseline_predictions.tolist(),
            "intervened_predictions": self.intervened_predictions.tolist(),
            "labels": self.labels.tolist(),
            "intervention_description": self.intervention_description,
        }


class AblationEvaluator:
    """
    Evaluator for ablation experiments.

    Compares model performance with and without interventions
    to measure causal effects.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        anchor_detector: AnchorDetector,
        label_token_ids: Dict[int, int],
        device: str = "cuda:0",
    ):
        """
        Args:
            model: The transformer model
            tokenizer: Tokenizer
            model_name: Name of the model
            anchor_detector: Anchor word detector
            label_token_ids: Mapping from label index to token ID
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.anchor_detector = anchor_detector
        self.label_token_ids = label_token_ids
        self.device = device

        # Initialize hook manager
        self.hook_manager = get_hook_manager(model, model_name)

    def evaluate_baseline(
        self,
        dataloader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model without intervention (baseline).

        Returns:
            Tuple of (predictions, probabilities, labels)
        """
        self.hook_manager.disable_all()

        predictions = []
        probabilities = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Baseline evaluation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].cpu().numpy()

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get logits at final position
                if attention_mask is not None:
                    final_pos = attention_mask.sum(dim=-1) - 1
                else:
                    final_pos = torch.full((input_ids.size(0),), input_ids.size(1) - 1)

                batch_idx = torch.arange(input_ids.size(0), device=self.device)
                final_logits = outputs.logits[batch_idx, final_pos]

                # Extract label logits
                label_indices = sorted(self.label_token_ids.keys())
                token_ids = [self.label_token_ids[i] for i in label_indices]
                label_logits = final_logits[:, token_ids]

                # Compute predictions and probabilities
                probs = torch.softmax(label_logits, dim=-1).cpu().numpy()
                preds = label_logits.argmax(dim=-1).cpu().numpy()

                predictions.extend(preds)
                probabilities.extend(probs)
                labels.extend(batch_labels)

        return np.array(predictions), np.array(probabilities), np.array(labels)

    def evaluate_with_intervention(
        self,
        dataloader,
        intervention: InterventionSpec,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model with intervention applied.

        Args:
            dataloader: DataLoader with test examples
            intervention: Intervention specification

        Returns:
            Tuple of (predictions, probabilities, labels)
        """
        # Get mask function from intervention
        mask_fn = intervention.intervention.get_mask_fn(self.anchor_detector)

        # Set up hooks for masking
        layer_indices = intervention.layer_indices
        if layer_indices is None:
            layer_indices = list(range(len(self.hook_manager.hooks)))

        # Configure hooks
        for i, hook in enumerate(self.hook_manager.hooks):
            if i in layer_indices:
                if isinstance(hook, MaskingHook):
                    hook.set_mask_fn(mask_fn)
                    hook.mode = HookMode.INTERVENE
                else:
                    # For non-masking hooks, we need to handle differently
                    hook.mode = HookMode.DISABLED
            else:
                hook.mode = HookMode.DISABLED

        predictions = []
        probabilities = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Intervention evaluation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].cpu().numpy()

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get logits at final position
                if attention_mask is not None:
                    final_pos = attention_mask.sum(dim=-1) - 1
                else:
                    final_pos = torch.full((input_ids.size(0),), input_ids.size(1) - 1)

                batch_idx = torch.arange(input_ids.size(0), device=self.device)
                final_logits = outputs.logits[batch_idx, final_pos]

                # Extract label logits
                label_indices = sorted(self.label_token_ids.keys())
                token_ids = [self.label_token_ids[i] for i in label_indices]
                label_logits = final_logits[:, token_ids]

                # Compute predictions and probabilities
                probs = torch.softmax(label_logits, dim=-1).cpu().numpy()
                preds = label_logits.argmax(dim=-1).cpu().numpy()

                predictions.extend(preds)
                probabilities.extend(probs)
                labels.extend(batch_labels)

        # Disable hooks after evaluation
        self.hook_manager.disable_all()

        return np.array(predictions), np.array(probabilities), np.array(labels)

    def run_ablation(
        self,
        dataloader,
        intervention: InterventionSpec,
    ) -> AblationResult:
        """
        Run complete ablation experiment.

        Args:
            dataloader: DataLoader with test examples
            intervention: Intervention to apply

        Returns:
            AblationResult with baseline and intervened metrics
        """
        # Evaluate baseline
        baseline_preds, baseline_probs, labels = self.evaluate_baseline(dataloader)
        baseline_acc = (baseline_preds == labels).mean()

        # Evaluate with intervention
        interv_preds, interv_probs, _ = self.evaluate_with_intervention(
            dataloader, intervention
        )
        interv_acc = (interv_preds == labels).mean()

        return AblationResult(
            baseline_accuracy=float(baseline_acc),
            intervened_accuracy=float(interv_acc),
            accuracy_drop=float(baseline_acc - interv_acc),
            baseline_predictions=baseline_preds,
            intervened_predictions=interv_preds,
            labels=labels,
            baseline_probs=baseline_probs,
            intervened_probs=interv_probs,
            intervention_description=intervention.describe(),
        )
