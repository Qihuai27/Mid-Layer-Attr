"""
Greedy Optimal Attribution: Input-level greedy search for optimal token importance.

This method finds the optimal token removal order by greedily selecting
the token whose removal causes the largest probability drop at each step.

While computationally expensive (O(T²) forward passes), it provides an
upper bound reference for evaluating other attribution methods.

Note: This is NOT a practical attribution method, but a theoretical baseline.
"""

from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import AttributionMethod, AttributionResult


class GreedyOptimalAttribution(AttributionMethod):
    """
    Greedy optimal attribution via exhaustive search at input embedding level.

    At each step, finds the token whose removal (replacing with baseline)
    causes the maximum probability drop for the target token.

    Complexity: O(T² * forward_pass_cost)

    This serves as an upper bound for Noise Insertion AUC evaluation.
    """

    def __init__(
        self,
        baseline_mode: str = "mean",  # "mean" or "zero"
        verbose: bool = False,
    ):
        """
        Initialize Greedy Optimal Attribution.

        Args:
            baseline_mode: How to compute baseline embedding
                - "mean": Mean of all token embeddings
                - "zero": Zero vector
            verbose: Whether to print progress
        """
        self.baseline_mode = baseline_mode
        self.verbose = verbose

    @property
    def name(self) -> str:
        return f"greedy_optimal_{self.baseline_mode}"

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute greedy optimal attribution scores.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids, shape [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Target token ID (if None, use argmax prediction)

        Returns:
            AttributionResult with importance scores
        """
        # Prepare input
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        T = input_ids.shape[1]

        # Handle negative target_pos
        if target_pos < 0:
            target_pos = T + target_pos
        if target_pos < 0 or target_pos >= T:
            raise ValueError(f"target_pos {target_pos} out of range [0, {T})")

        # Get embeddings
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            base_embeds = embed_layer(input_ids)[0]  # [T, d]

        # Compute baseline embedding
        if self.baseline_mode == "mean":
            baseline_embed = base_embeds.mean(dim=0)
        elif self.baseline_mode == "zero":
            baseline_embed = torch.zeros_like(base_embeds[0])
        else:
            raise ValueError(f"Unknown baseline_mode: {self.baseline_mode}")

        # Get target token if not provided
        if target_token_id is None:
            with torch.no_grad():
                outputs = model(inputs_embeds=base_embeds.unsqueeze(0))
                logits = outputs.logits[0, target_pos]
                target_token_id = logits.argmax().item()

        # Get original probability
        with torch.no_grad():
            outputs = model(inputs_embeds=base_embeds.unsqueeze(0))
            original_prob = torch.softmax(outputs.logits[0, target_pos], dim=-1)[target_token_id].item()

        if self.verbose:
            print(f"Greedy Optimal Attribution (T={T}, target_pos={target_pos})")
            print(f"  Original prob: {original_prob:.4f}")
            print(f"  Baseline mode: {self.baseline_mode}")

        # Greedy search: find optimal removal order
        available = set(range(target_pos + 1))  # Only positions that can affect target
        selected_order = []
        current_embeds = base_embeds.clone()
        probabilities = [original_prob]

        with torch.no_grad():
            for step in range(target_pos + 1):
                if self.verbose and step % 5 == 0:
                    print(f"  Step {step + 1}/{target_pos + 1}...")

                best_idx = None
                best_prob_after = float('inf')

                # Try removing each remaining token
                for idx in available:
                    test_embeds = current_embeds.clone()
                    test_embeds[idx] = baseline_embed

                    outputs = model(inputs_embeds=test_embeds.unsqueeze(0))
                    prob = torch.softmax(outputs.logits[0, target_pos], dim=-1)[target_token_id].item()

                    if prob < best_prob_after:
                        best_prob_after = prob
                        best_idx = idx

                # Select the token that causes maximum drop
                selected_order.append(best_idx)
                available.remove(best_idx)
                current_embeds[best_idx] = baseline_embed
                probabilities.append(best_prob_after)

        # Convert selection order to importance scores
        # Earlier selection = higher importance
        scores = np.zeros(T, dtype=np.float32)
        for rank, idx in enumerate(selected_order):
            # Score = (T - rank) so first selected gets highest score
            scores[idx] = len(selected_order) - rank

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        # Positions after target_pos get 0
        scores[target_pos + 1:] = 0

        # Compute probability drops for metadata
        prob_drops = [probabilities[i] - probabilities[i + 1] for i in range(len(probabilities) - 1)]

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "baseline_mode": self.baseline_mode,
                "original_prob": original_prob,
                "final_prob": probabilities[-1],
                "selection_order": selected_order,
                "probabilities": probabilities,
                "prob_drops": prob_drops,
                "total_forward_passes": len(selected_order) * (len(selected_order) + 1) // 2,
            }
        )


class InputCausalAttribution(AttributionMethod):
    """
    Input-level causal attribution (Input s2).

    For each position, replace its embedding with baseline and measure
    the probability drop for the target token.

    This is a fast approximation that doesn't account for token interactions.
    Complexity: O(T) forward passes.
    """

    def __init__(
        self,
        baseline_mode: str = "mean",
    ):
        """
        Initialize Input Causal Attribution.

        Args:
            baseline_mode: How to compute baseline embedding ("mean" or "zero")
        """
        self.baseline_mode = baseline_mode

    @property
    def name(self) -> str:
        return f"input_causal_{self.baseline_mode}"

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute input-level causal attribution scores.
        """
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        T = input_ids.shape[1]

        if target_pos < 0:
            target_pos = T + target_pos

        # Get embeddings
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            base_embeds = embed_layer(input_ids)[0]

        # Baseline
        if self.baseline_mode == "mean":
            baseline_embed = base_embeds.mean(dim=0)
        else:
            baseline_embed = torch.zeros_like(base_embeds[0])

        # Get target token
        if target_token_id is None:
            with torch.no_grad():
                outputs = model(inputs_embeds=base_embeds.unsqueeze(0))
                target_token_id = outputs.logits[0, target_pos].argmax().item()

        # Original probability
        with torch.no_grad():
            outputs = model(inputs_embeds=base_embeds.unsqueeze(0))
            original_prob = torch.softmax(outputs.logits[0, target_pos], dim=-1)[target_token_id].item()

        # Compute probability drop for each position
        scores = np.zeros(T, dtype=np.float32)

        with torch.no_grad():
            for j in range(target_pos + 1):
                modified_embeds = base_embeds.clone()
                modified_embeds[j] = baseline_embed

                outputs = model(inputs_embeds=modified_embeds.unsqueeze(0))
                modified_prob = torch.softmax(outputs.logits[0, target_pos], dim=-1)[target_token_id].item()

                scores[j] = max(0, original_prob - modified_prob)

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "baseline_mode": self.baseline_mode,
                "original_prob": original_prob,
            }
        )


def greedy_optimal_attribution(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int = -1,
    target_token_id: Optional[int] = None,
    baseline_mode: str = "mean",
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for greedy optimal attribution.

    Warning: This is very slow (O(T²) forward passes).
    """
    method = GreedyOptimalAttribution(
        baseline_mode=baseline_mode,
        verbose=verbose,
    )
    result = method.attribute(model, input_ids, target_pos, target_token_id, **kwargs)
    return result.scores


def input_causal_attribution(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int = -1,
    target_token_id: Optional[int] = None,
    baseline_mode: str = "mean",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for input causal attribution (Input s2).
    """
    method = InputCausalAttribution(baseline_mode=baseline_mode)
    result = method.attribute(model, input_ids, target_pos, target_token_id, **kwargs)
    return result.scores
