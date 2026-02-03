"""
Attention weight extraction for cluster analysis.

Extracts attention weights from each layer, constructing a bipartite graph
representation of information flow from input tokens to output tokens.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedModel

from src.models.hooks import HookManager, get_hook_manager
from src.models.hooks.base import HookMode


@dataclass
class AttentionBipartiteGraph:
    """
    Bipartite graph representation of attention weights for a single layer.

    The attention matrix can be viewed as a weighted bipartite graph:
    - Left nodes: input token positions (keys)
    - Right nodes: output token positions (queries)
    - Edge weights: attention weights

    Attributes:
        layer_idx: Index of the transformer layer
        weights: Attention weights [batch, heads, seq_q, seq_k]
        input_ids: Original input token IDs
    """

    layer_idx: int
    weights: Tensor  # [batch, heads, seq_q, seq_k]
    input_ids: Optional[Tensor] = None

    @property
    def num_heads(self) -> int:
        return self.weights.size(1)

    @property
    def seq_length(self) -> int:
        return self.weights.size(-1)

    def aggregate_heads(self, method: str = "sum") -> Tensor:
        """
        Aggregate attention weights across heads.

        Args:
            method: 'sum', 'mean', or 'max'

        Returns:
            Aggregated weights [batch, seq_q, seq_k]
        """
        if method == "sum":
            return self.weights.sum(dim=1)
        elif method == "mean":
            return self.weights.mean(dim=1)
        elif method == "max":
            return self.weights.max(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def get_attention_to_positions(
        self,
        query_positions: Tensor,
        key_positions: Tensor,
    ) -> Tensor:
        """
        Get attention weights between specific positions.

        Args:
            query_positions: Positions of query tokens
            key_positions: Positions of key tokens

        Returns:
            Attention weights [batch, heads, len(query), len(key)]
        """
        # Index into the attention matrix
        return self.weights[:, :, query_positions][:, :, :, key_positions]


class AttentionExtractor:
    """
    Extracts attention weights from transformer layers.

    Supports two modes:
    1. Standard extraction: Just capture attention weights
    2. Gradient extraction: Capture attention weights with gradient information
                           for saliency-based attribution
    """

    def __init__(
        self,
        model: PreTrainedModel,
        model_name: str,
    ):
        """
        Initialize the extractor.

        Args:
            model: Transformer model
            model_name: Name of the model (for selecting appropriate hooks)
        """
        self.model = model
        self.model_name = model_name
        self.hook_manager: Optional[HookManager] = None

    def _ensure_hooks(self, with_gradients: bool = False):
        """Ensure hooks are set up."""
        if self.hook_manager is None:
            self.hook_manager = get_hook_manager(self.model, self.model_name)

    def extract(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> List[AttentionBipartiteGraph]:
        """
        Extract attention weights from all layers.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            List of AttentionBipartiteGraph for each layer
        """
        self._ensure_hooks()

        # Enable observation mode
        self.hook_manager.enable_observation()

        # Forward pass
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Collect captured weights
        graphs = []
        captured = self.hook_manager.get_captured_weights()

        for layer_idx, weights in enumerate(captured):
            if weights is not None:
                graphs.append(AttentionBipartiteGraph(
                    layer_idx=layer_idx,
                    weights=weights,
                    input_ids=input_ids,
                ))

        # Reset hooks
        self.hook_manager.disable_all()
        self.hook_manager.reset()

        return graphs

    def extract_with_gradients(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        loss_fn: Optional[callable] = None,
    ) -> Tuple[List[AttentionBipartiteGraph], List[Tensor]]:
        """
        Extract attention weights with gradient information.

        This enables saliency-based attribution by computing gradients
        of the loss with respect to attention weights.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Ground truth labels for computing loss
            attention_mask: Attention mask [batch, seq_len]
            loss_fn: Loss function (defaults to cross-entropy)

        Returns:
            Tuple of (attention graphs, gradients per layer)
        """
        self._ensure_hooks()

        # Enable observation mode
        self.hook_manager.enable_observation()

        # We need gradients for the forward pass even if model params don't require grad
        # This enables gradient flow through the computation graph
        with torch.enable_grad():
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Compute loss
            if loss_fn is None:
                import torch.nn.functional as F
                # Get logits at final position
                if attention_mask is not None:
                    final_pos = attention_mask.sum(dim=-1) - 1
                else:
                    final_pos = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)

                batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                final_logits = outputs.logits[batch_idx, final_pos]
                loss = F.cross_entropy(final_logits, labels)
            else:
                loss = loss_fn(outputs, labels)

            # Backward pass to capture gradients via hooks
            loss.backward()

        # Collect weights and gradients
        graphs = []
        gradients = []
        captured = self.hook_manager.get_captured_weights()
        grads = self.hook_manager.get_gradients()

        for layer_idx, (weights, grad) in enumerate(zip(captured, grads)):
            if weights is not None:
                graphs.append(AttentionBipartiteGraph(
                    layer_idx=layer_idx,
                    weights=weights,
                    input_ids=input_ids,
                ))
            gradients.append(grad)

        # Reset hooks
        self.hook_manager.disable_all()
        self.hook_manager.reset()

        return graphs, gradients
