"""
Gradient × Input attribution method.

This is one of the simplest gradient-based attribution methods.
It computes the element-wise product of the input embeddings
and their gradients w.r.t. the output.

Reference:
    Simonyan et al., "Deep Inside Convolutional Networks: Visualising
    Image Classification Models and Saliency Maps" (2013)
    https://arxiv.org/abs/1312.6034

    Shrikumar et al., "Not Just a Black Box: Learning Important Features
    Through Propagating Activation Differences" (2016)
    https://arxiv.org/abs/1605.01713

Formula:
    score_i = sum_j(x_i,j * ∂F/∂x_i,j)

Where:
    - x_i,j is the j-th dimension of embedding for token i
    - F is the model output (logit for target token)
    - sum_j is over embedding dimensions

Intuition:
    - Gradient indicates how much output changes per unit input change
    - Multiplying by input value weights this by the actual input magnitude
    - High score = both large gradient AND large input value
"""

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import GradientBasedMethod, AttributionResult


class GradInput(GradientBasedMethod):
    """
    Gradient × Input attribution method.

    Computes token importance by multiplying input embeddings with their
    gradients w.r.t. the target prediction.
    """

    def __init__(
        self,
        absolute: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Initialize Gradient × Input method.

        Args:
            absolute: If True, take absolute value of scores (default True)
                     This makes all attributions positive, useful for ranking
            normalize_embeddings: If True, normalize input embeddings before
                                 computing attribution (L2 normalization per token)
        """
        self.absolute = absolute
        self.normalize_embeddings = normalize_embeddings

    @property
    def name(self) -> str:
        return "grad_input"

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using Gradient × Input.

        Args:
            model: HuggingFace causal LM
            input_ids: Input token ids [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Token id being predicted (if None, use argmax)
            **kwargs: Additional arguments (unused)

        Returns:
            AttributionResult with importance scores
        """
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        # Ensure input_ids is 2D
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        T = input_ids.shape[1]

        # Get input embeddings
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)  # [1, T, d]

        # Optional normalization
        if self.normalize_embeddings:
            input_embeds = input_embeds / (input_embeds.norm(dim=-1, keepdim=True) + 1e-8)

        # Create new tensor for gradient computation
        input_embeds = input_embeds.detach().requires_grad_(True)

        # Forward pass
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits  # [1, T, V]

        # Get target token if not provided
        if target_token_id is None:
            target_token_id = logits[0, target_pos].argmax().item()

        # Get target logit
        target_logit = logits[0, target_pos, target_token_id]

        # Backward pass
        model.zero_grad()
        target_logit.backward()

        # Get gradients
        gradients = input_embeds.grad[0]  # [T, d]

        # Compute Gradient × Input
        input_values = input_embeds[0].detach()  # [T, d]
        grad_input = gradients * input_values  # [T, d]

        # Sum over embedding dimension
        scores = grad_input.sum(dim=-1)  # [T]

        # Optional: take absolute value
        if self.absolute:
            scores = scores.abs()

        scores = scores.detach().cpu().numpy()

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "absolute": self.absolute,
                "normalize_embeddings": self.normalize_embeddings,
                "target_token_id": target_token_id,
            }
        )


def grad_input(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int,
    target_token_id: Optional[int] = None,
    absolute: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Functional interface for Gradient × Input attribution.

    Args:
        model: HuggingFace causal LM
        input_ids: Input token ids
        target_pos: Position to explain
        target_token_id: Token being predicted (optional)
        absolute: Take absolute value of scores

    Returns:
        scores: Attribution scores [T]
    """
    method = GradInput(absolute=absolute)
    result = method.attribute(model, input_ids, target_pos, target_token_id, **kwargs)
    return result.scores
