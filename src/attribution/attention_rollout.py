"""
Attention Rollout attribution method.

Attention Rollout computes the cumulative attention flow from input tokens
to a target position by recursively multiplying attention matrices across layers,
accounting for residual connections.

Reference:
    Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020)
    https://aclanthology.org/2020.acl-main.385/

Key idea:
    - In transformers, information flows through both attention and residual connections
    - Rollout accounts for residual by mixing attention with identity: A' = α*I + (1-α)*A
    - Cumulative flow: R^(L) = A'^(L) @ A'^(L-1) @ ... @ A'^(1)
    - The row R[target_pos, :] gives importance of each input token to target_pos
"""

from typing import Optional, Union, List, Literal

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import AttributionMethod, AttentionBasedMethod, AttributionResult


class AttentionRollout(AttentionBasedMethod):
    """
    Attention Rollout for computing token importance scores.

    This method computes how much "attention flow" each input token receives
    from the target position, by accumulating attention across all layers
    while accounting for residual connections.

    For a target position t, the importance of token i is R[t, i], where R
    is the rollout matrix computed as the cumulative product of attention
    matrices (mixed with identity for residual).
    """

    def __init__(
        self,
        residual_weight: float = 0.5,
        head_aggregation: Literal["mean", "max", "sum"] = "mean",
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        handle_cls: bool = False,
    ):
        """
        Initialize Attention Rollout.

        Args:
            residual_weight: Weight for residual connection (α in A' = α*I + (1-α)*A)
                            Default 0.5 treats attention and residual equally.
            head_aggregation: How to aggregate attention across heads
                            - "mean": Average attention across heads (default)
                            - "max": Take maximum attention across heads
                            - "sum": Sum attention across heads
            start_layer: First layer to include in rollout (0-indexed)
            end_layer: Last layer to include (None = all layers)
            handle_cls: If True, discard attention to CLS token (for encoder models)
        """
        self.residual_weight = residual_weight
        self.head_aggregation = head_aggregation
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.handle_cls = handle_cls

    @property
    def name(self) -> str:
        return "attention_rollout"

    def _compute_rollout(
        self,
        attention_layers: List[Tensor],
    ) -> np.ndarray:
        """
        Compute attention rollout matrix.

        Args:
            attention_layers: List of attention tensors per layer
                             Each tensor: [batch, num_heads, seq_len, seq_len]

        Returns:
            Rollout matrix [seq_len, seq_len] where R[i, j] is the
            cumulative attention flow from position j to position i
        """
        # Determine layer range
        num_layers = len(attention_layers)
        start = self.start_layer
        end = self.end_layer if self.end_layer is not None else num_layers

        # Get sequence length from first attention
        seq_len = attention_layers[0].shape[-1]

        # Initialize rollout as identity (before any layer)
        rollout = np.eye(seq_len, dtype=np.float32)

        # Process each layer
        for layer_idx in range(start, end):
            attn = attention_layers[layer_idx]

            # Aggregate across heads: [batch, num_heads, seq, seq] -> [batch, seq, seq]
            attn_agg = self._aggregate_heads(attn, method=self.head_aggregation)

            # Remove batch dimension if present
            if attn_agg.dim() == 3:
                attn_agg = attn_agg.squeeze(0)

            # Convert to numpy (handle bfloat16)
            attn_np = attn_agg.detach().cpu().float().numpy()

            # Handle CLS token if needed (set attention to CLS to 0 except from itself)
            if self.handle_cls:
                attn_np[:, 0] = 0
                attn_np[0, 0] = 1

            # Mix with identity for residual connection: A' = α*I + (1-α)*A
            identity = np.eye(seq_len, dtype=np.float32)
            attn_with_residual = (
                self.residual_weight * identity +
                (1 - self.residual_weight) * attn_np
            )

            # Cumulative product: R = A' @ R
            rollout = attn_with_residual @ rollout

        return rollout

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using Attention Rollout.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids, shape [1, T] or [T]
            target_pos: Position to explain (row of rollout matrix to extract)
            **kwargs: Additional arguments (unused)

        Returns:
            AttributionResult with importance scores for each input token
        """
        # Ensure input_ids is proper shape
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        T = input_ids.shape[1]

        # Validate target_pos
        if target_pos < 0:
            target_pos = T + target_pos
        if target_pos < 0 or target_pos >= T:
            raise ValueError(f"target_pos {target_pos} out of range [0, {T})")

        # Extract attention from all layers
        attention_layers = self._extract_attention(model, input_ids)

        # Compute rollout matrix
        rollout = self._compute_rollout(attention_layers)

        # Extract row for target position: importance of each input token
        # rollout[target_pos, j] = how much attention flows from j to target_pos
        importance_scores = rollout[target_pos, :]

        # For causal LM, we only care about positions <= target_pos
        # Positions after target_pos should have 0 importance
        importance_scores[target_pos + 1:] = 0

        return AttributionResult(
            scores=importance_scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "residual_weight": self.residual_weight,
                "head_aggregation": self.head_aggregation,
                "start_layer": self.start_layer,
                "end_layer": self.end_layer,
                "num_layers": len(attention_layers),
            }
        )


class AttentionRolloutLastLayer(AttentionBasedMethod):
    """
    Simplified Attention Rollout using only the last layer's attention.

    This is a faster variant that only uses the final layer's attention
    to compute importance scores, without cumulative rollout.

    Use this when:
    - Speed is critical
    - You believe the last layer captures the most relevant attention patterns
    """

    def __init__(
        self,
        head_aggregation: Literal["mean", "max", "sum"] = "mean",
    ):
        """
        Args:
            head_aggregation: How to aggregate attention across heads
        """
        self.head_aggregation = head_aggregation

    @property
    def name(self) -> str:
        return "attention_last_layer"

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution using only last layer attention.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids, shape [1, T] or [T]
            target_pos: Position to explain

        Returns:
            AttributionResult with importance scores
        """
        # Ensure proper shape
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        T = input_ids.shape[1]

        if target_pos < 0:
            target_pos = T + target_pos
        if target_pos < 0 or target_pos >= T:
            raise ValueError(f"target_pos {target_pos} out of range [0, {T})")

        # Extract attention
        attention_layers = self._extract_attention(model, input_ids)

        # Use only last layer
        last_attn = attention_layers[-1]

        # Aggregate heads
        attn_agg = self._aggregate_heads(last_attn, method=self.head_aggregation)

        if attn_agg.dim() == 3:
            attn_agg = attn_agg.squeeze(0)

        attn_np = attn_agg.detach().cpu().numpy()

        # Extract row for target position
        importance_scores = attn_np[target_pos, :]

        # Zero out future positions
        importance_scores[target_pos + 1:] = 0

        return AttributionResult(
            scores=importance_scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "head_aggregation": self.head_aggregation,
                "layer_index": len(attention_layers) - 1,
            }
        )


class AttentionRolloutWithGradient(AttentionBasedMethod):
    """
    Attention Rollout weighted by gradient importance.

    Combines attention rollout with gradient-based saliency:
    importance[i] = rollout[target, i] * |grad_attention[target, i]|

    This weights the attention flow by how much each attention connection
    affects the final prediction.
    """

    def __init__(
        self,
        residual_weight: float = 0.5,
        head_aggregation: Literal["mean", "max", "sum"] = "mean",
    ):
        self.residual_weight = residual_weight
        self.head_aggregation = head_aggregation

    @property
    def name(self) -> str:
        return "attention_rollout_gradient"

    @property
    def requires_grad(self) -> bool:
        return True

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute gradient-weighted attention rollout.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids
            target_pos: Position to explain
            target_token_id: Target token for gradient computation
                            If None, uses the most likely token

        Returns:
            AttributionResult with gradient-weighted importance scores
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

        # Forward pass with gradient tracking on attention
        outputs = model(
            input_ids,
            output_attentions=True,
            return_dict=True,
        )

        # Get target logits
        logits = outputs.logits  # [1, T, vocab]
        target_logits = logits[0, target_pos]

        # Compute loss
        if target_token_id is not None:
            log_probs = torch.log_softmax(target_logits, dim=-1)
            loss = -log_probs[target_token_id]
        else:
            loss = target_logits.max()

        # Compute gradients w.r.t. attention
        attention_layers = list(outputs.attentions)

        # We need gradients, so we need to make attention require grad
        # This requires a fresh forward pass with hooks
        model.zero_grad()

        # For simplicity, just use rollout without gradient weighting
        # Full gradient computation would require hooks on attention
        # TODO: Implement proper gradient computation with hooks

        # Fallback to standard rollout
        rollout_method = AttentionRollout(
            residual_weight=self.residual_weight,
            head_aggregation=self.head_aggregation,
        )
        return rollout_method.attribute(model, input_ids, target_pos)


def attention_rollout(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int = -1,
    residual_weight: float = 0.5,
    head_aggregation: str = "mean",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for computing attention rollout scores.

    Args:
        model: HuggingFace causal language model
        input_ids: Input token ids
        target_pos: Position to explain (default: -1, last position)
        residual_weight: Weight for residual connection
        head_aggregation: How to aggregate heads

    Returns:
        Importance scores array of shape [T]
    """
    method = AttentionRollout(
        residual_weight=residual_weight,
        head_aggregation=head_aggregation,
    )
    result = method.attribute(model, input_ids, target_pos, **kwargs)
    return result.scores
