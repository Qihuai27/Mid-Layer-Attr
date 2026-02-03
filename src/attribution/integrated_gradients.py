"""
Integrated Gradients attribution method.

Integrated Gradients computes feature importance by integrating gradients
along a straight-line path from a baseline to the input.

Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks" (ICML 2017)
    https://arxiv.org/abs/1703.01365

Key idea:
    - Interpolate between baseline (e.g., zero embedding) and input
    - Compute gradients at each interpolation point
    - Integrate (sum) gradients along the path
    - IG satisfies completeness: sum of attributions = output difference

Formula:
    IG_i = (x_i - x'_i) * ∫_{α=0}^{1} ∂F(x' + α(x - x')) / ∂x_i dα

Where:
    - x is the input embedding
    - x' is the baseline embedding
    - F is the model output (logit for target token)
"""

from typing import Optional, Union, Literal, List
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import AttributionMethod, GradientBasedMethod, AttributionResult


class IntegratedGradients(GradientBasedMethod):
    """
    Integrated Gradients for computing token importance scores.

    This method computes attributions by integrating gradients along a path
    from a baseline (e.g., zero or PAD embeddings) to the actual input.
    """

    def __init__(
        self,
        n_steps: int = 50,
        baseline_type: Literal["zero", "pad", "mean"] = "zero",
        internal_batch_size: int = 10,
    ):
        """
        Initialize Integrated Gradients.

        Args:
            n_steps: Number of interpolation steps for Riemann approximation
            baseline_type: Type of baseline embedding
                - "zero": Zero vector
                - "pad": PAD token embedding (requires tokenizer)
                - "mean": Mean of input embeddings
            internal_batch_size: Batch size for gradient computation (memory vs speed)
        """
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.internal_batch_size = internal_batch_size

    @property
    def name(self) -> str:
        return "integrated_gradients"

    def _get_baseline_embedding(
        self,
        model: PreTrainedModel,
        input_embeds: Tensor,
        pad_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Get baseline embedding based on baseline_type.

        Args:
            model: HuggingFace model
            input_embeds: Input embeddings [T, d]
            pad_token_id: PAD token id (for "pad" baseline)

        Returns:
            Baseline embeddings [T, d]
        """
        T, d = input_embeds.shape
        device = input_embeds.device
        dtype = input_embeds.dtype

        if self.baseline_type == "zero":
            return torch.zeros_like(input_embeds)

        elif self.baseline_type == "mean":
            mean_embed = input_embeds.mean(dim=0, keepdim=True)
            return mean_embed.expand(T, -1)

        elif self.baseline_type == "pad":
            if pad_token_id is None:
                # Fallback to zero if no pad token
                return torch.zeros_like(input_embeds)
            embed_layer = model.get_input_embeddings()
            pad_ids = torch.tensor([[pad_token_id]], device=device)
            pad_embed = embed_layer(pad_ids)[0, 0]  # [d]
            return pad_embed.unsqueeze(0).expand(T, -1)

        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def _compute_gradients(
        self,
        model: PreTrainedModel,
        inputs_embeds: Tensor,
        target_pos: int,
        target_token_id: int,
    ) -> Tensor:
        """
        Compute gradients of target logit w.r.t. input embeddings.

        Args:
            model: HuggingFace model
            inputs_embeds: Input embeddings [1, T, d]
            target_pos: Position to compute loss for
            target_token_id: Target token id

        Returns:
            Gradients [T, d]
        """
        # Keep in original dtype but enable gradients
        inputs_embeds = inputs_embeds.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=inputs_embeds)
        logits = outputs.logits  # [1, T, V]

        # Get logit for target token at target position
        target_logit = logits[0, target_pos, target_token_id]

        # Backward pass
        model.zero_grad()
        target_logit.backward()

        gradients = inputs_embeds.grad[0]  # [T, d]
        return gradients.detach()

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using Integrated Gradients.

        Args:
            model: HuggingFace causal LM
            input_ids: Input token ids [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Token id being predicted (if None, use argmax)
            pad_token_id: PAD token id for baseline (optional)
            **kwargs: Additional arguments

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

        # Get input embeddings (detached, we'll create our own gradient path)
        embed_layer = model.get_input_embeddings()
        with torch.no_grad():
            input_embeds = embed_layer(input_ids)[0].clone()  # [T, d]

        # Get target token if not provided
        if target_token_id is None:
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, target_pos]
                target_token_id = logits.argmax().item()

        # Get baseline (also detached)
        with torch.no_grad():
            baseline_embeds = self._get_baseline_embedding(model, input_embeds, pad_token_id).clone()

        # Compute interpolation alphas
        # Skip alpha=0 to avoid numerical issues with zero baseline
        # Use linspace from small epsilon to 1
        alphas = torch.linspace(1e-3, 1, self.n_steps, device=device, dtype=torch.float32)

        # Compute integrated gradients using Riemann sum
        # Use float32 for accumulation to avoid overflow
        integrated_grads = torch.zeros(T, input_embeds.shape[-1], device=device, dtype=torch.float32)

        # Process in batches for memory efficiency
        for batch_start in range(0, len(alphas), self.internal_batch_size):
            batch_end = min(batch_start + self.internal_batch_size, len(alphas))
            batch_alphas = alphas[batch_start:batch_end]

            for alpha in batch_alphas:
                # Interpolated input: x' + α(x - x')
                # Create as leaf tensor with requires_grad
                interp_embeds = (baseline_embeds.float() + alpha * (input_embeds.float() - baseline_embeds.float()))
                interp_embeds = interp_embeds.to(model_dtype).unsqueeze(0)  # [1, T, d]
                interp_embeds = interp_embeds.detach().requires_grad_(True)

                # Forward pass
                outputs = model(inputs_embeds=interp_embeds)
                logits = outputs.logits  # [1, T, V]

                # Get logit for target token at target position
                target_logit = logits[0, target_pos, target_token_id]

                # Backward pass
                model.zero_grad()
                target_logit.backward()

                # Accumulate gradients
                if interp_embeds.grad is not None:
                    integrated_grads += interp_embeds.grad[0].float()

        # Average gradients (Riemann sum approximation)
        integrated_grads = integrated_grads / self.n_steps

        # Multiply by input difference: IG = (x - x') * avg_grads
        input_diff = (input_embeds.float() - baseline_embeds.float())
        attributions = input_diff * integrated_grads  # [T, d]

        # Sum over embedding dimension to get per-token scores
        scores = attributions.sum(dim=-1).detach().cpu().numpy()  # [T]

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "n_steps": self.n_steps,
                "baseline_type": self.baseline_type,
                "target_token_id": target_token_id,
            }
        )


def integrated_gradients(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int,
    target_token_id: Optional[int] = None,
    n_steps: int = 50,
    baseline_type: str = "zero",
    **kwargs,
) -> np.ndarray:
    """
    Functional interface for Integrated Gradients.

    Args:
        model: HuggingFace causal LM
        input_ids: Input token ids
        target_pos: Position to explain
        target_token_id: Token being predicted (optional)
        n_steps: Number of interpolation steps
        baseline_type: Type of baseline ("zero", "pad", "mean")

    Returns:
        scores: Attribution scores [T]
    """
    method = IntegratedGradients(n_steps=n_steps, baseline_type=baseline_type)
    result = method.attribute(model, input_ids, target_pos, target_token_id, **kwargs)
    return result.scores
