"""
Attention intervention strategies for causal ablation.

Interventions modify attention weights to test causal hypotheses about
the role of anchor words in ICL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

import torch
from torch import Tensor

from src.anchors import AnchorDetector, AnchorPositions


class AttentionIntervention(ABC):
    """
    Base class for attention interventions.

    Interventions modify attention weights to test causal effects.
    """

    def __init__(self, mask_value: float = -1e4):
        self.mask_value = mask_value

    @abstractmethod
    def get_mask_fn(
        self,
        anchor_detector: AnchorDetector,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Return a function that generates a mask for attention weights.

        Args:
            anchor_detector: Detector for finding anchor positions

        Returns:
            Function (attn_weights, input_ids) -> boolean mask
            True values indicate positions to mask out
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Return a description of this intervention."""
        pass


class MaskAnchorAttention(AttentionIntervention):
    """
    Mask attention FROM anchor positions to previous tokens.

    This intervention blocks information flow from anchor words,
    testing whether they are necessary for ICL performance.

    When applied to shallow layers, this tests the hypothesis that
    early layers process anchor word semantics.
    """

    def __init__(
        self,
        mask_value: float = -1e4,
        window_size: int = 0,
    ):
        """
        Args:
            mask_value: Value to fill masked positions with
            window_size: How many tokens after anchor to also mask
        """
        super().__init__(mask_value)
        self.window_size = window_size

    def get_mask_fn(
        self,
        anchor_detector: AnchorDetector,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return masking function."""

        def mask_fn(attn_weights: Tensor, input_ids: Tensor) -> Tensor:
            """
            Generate mask for anchor positions.

            Masks attention FROM anchor positions (and window after)
            TO all positions before them.
            """
            batch_size, num_heads, seq_q, seq_k = attn_weights.shape
            device = attn_weights.device

            # Detect anchor positions
            anchor_positions = anchor_detector.detect(input_ids)

            # Create mask (False = keep, True = mask out)
            mask = torch.zeros_like(attn_weights, dtype=torch.bool)

            for label_idx, positions in anchor_positions.label_positions.items():
                for pos in positions.tolist() if positions.numel() > 0 else []:
                    # Mask from anchor position (and window) to all previous positions
                    end_pos = min(pos + self.window_size + 1, seq_q)
                    mask[:, :, pos:end_pos, :pos] = True

            return mask

        return mask_fn

    def describe(self) -> str:
        return f"MaskAnchorAttention(window_size={self.window_size})"


class MaskNonAnchorAttention(AttentionIntervention):
    """
    Mask attention from randomly selected non-anchor positions.

    This serves as a control experiment to verify that the effect
    of masking anchor words is specific and not due to general
    information reduction.
    """

    def __init__(
        self,
        mask_value: float = -1e4,
        num_positions: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            mask_value: Value to fill masked positions with
            num_positions: Number of positions to mask (None = same as anchors)
            seed: Random seed for position selection
        """
        super().__init__(mask_value)
        self.num_positions = num_positions
        self.seed = seed

    def get_mask_fn(
        self,
        anchor_detector: AnchorDetector,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Return masking function."""
        import random
        rng = random.Random(self.seed)

        def mask_fn(attn_weights: Tensor, input_ids: Tensor) -> Tensor:
            """
            Generate mask for random non-anchor positions.
            """
            batch_size, num_heads, seq_q, seq_k = attn_weights.shape
            device = attn_weights.device

            # Detect anchor positions
            anchor_positions = anchor_detector.detect(input_ids)

            # Get non-anchor positions
            non_anchor = anchor_positions.get_non_anchor_positions(exclude_final=True)

            # Determine number to mask
            num_to_mask = self.num_positions
            if num_to_mask is None:
                num_to_mask = anchor_positions.num_anchors

            # Randomly select positions
            if len(non_anchor) >= num_to_mask:
                selected = rng.sample(non_anchor.tolist(), num_to_mask)
            else:
                selected = non_anchor.tolist()

            # Create mask
            mask = torch.zeros_like(attn_weights, dtype=torch.bool)
            for pos in selected:
                mask[:, :, pos, :pos] = True

            return mask

        return mask_fn

    def describe(self) -> str:
        return f"MaskNonAnchorAttention(num_positions={self.num_positions})"


class MaskLayerAttention(AttentionIntervention):
    """
    Mask all attention in specific layers.

    This tests whether certain layers are critical for ICL,
    by completely blocking attention in those layers.
    """

    def __init__(
        self,
        layers: List[int],
        mask_value: float = -1e4,
        position: str = "first",  # 'first' or 'last'
    ):
        """
        Args:
            layers: Layer indices to mask (or number of layers if position is set)
            mask_value: Value to fill masked positions with
            position: 'first' to mask from beginning, 'last' to mask from end
        """
        super().__init__(mask_value)
        self.layers = layers
        self.position = position

    def get_layer_indices(self, num_layers: int) -> List[int]:
        """Get actual layer indices based on position setting."""
        if isinstance(self.layers, int):
            # Treat as number of layers
            if self.position == "first":
                return list(range(self.layers))
            else:  # last
                return list(range(num_layers - self.layers, num_layers))
        return self.layers

    def get_mask_fn(
        self,
        anchor_detector: AnchorDetector,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Return masking function.

        Note: This intervention works differently - it's applied
        at the layer level rather than within attention.
        """

        def mask_fn(attn_weights: Tensor, input_ids: Tensor) -> Tensor:
            """Mask all attention (returns all-True mask)."""
            return torch.ones_like(attn_weights, dtype=torch.bool)

        return mask_fn

    def describe(self) -> str:
        return f"MaskLayerAttention(layers={self.layers}, position={self.position})"


@dataclass
class InterventionSpec:
    """Specification for an intervention to apply."""

    intervention: AttentionIntervention
    layer_indices: Optional[List[int]] = None  # None = all layers

    def describe(self) -> str:
        layers_str = f"layers={self.layer_indices}" if self.layer_indices else "all layers"
        return f"{self.intervention.describe()} @ {layers_str}"
