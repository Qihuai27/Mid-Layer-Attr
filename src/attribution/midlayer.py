"""
MidLayer Attribution: Two-stage attribution via intermediate layer.

This method computes token attribution by combining:
1. Stage 1: Attention Rollout from input (layer 0) to a mid-layer (default: layer 8)
   - s1(i,j): attention flow from input token i to mid-layer token j
2. Stage 2: Causal intervention from mid-layer to final logits
   - s2(j): cosine distance of logits when zeroing token j's hidden state at mid-layer

Final score: score(i) = Σ_j s2(j) * s1(i,j)

This combines "input→mid-layer" attention flow with "mid-layer→output" causal influence.
"""

from typing import Optional, Union, Literal, List, Tuple

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .base import AttributionMethod, AttentionBasedMethod, AttributionResult


class MidLayerAttribution(AttentionBasedMethod):
    """
    Two-stage attribution method via intermediate layer.

    Stage 1: Compute attention rollout from input to mid-layer, yielding s1(i,j)
    Stage 2: Compute causal influence from mid-layer to logits, yielding s2(j)
    Final: score(i) = Σ_j s2(j) * s1(i,j)
    """

    def __init__(
        self,
        mid_layer: Optional[int] = None,  # If None, auto-select based on model depth
        mid_layer_ratio: float = 0.25,  # Use layer at this ratio of total depth
        residual_weight: float = 0.5,
        head_aggregation: Literal["mean", "max", "sum"] = "mean",
    ):
        """
        Initialize MidLayer Attribution.

        Args:
            mid_layer: The intermediate layer index (0-indexed). If None, auto-select
                      based on mid_layer_ratio.
            mid_layer_ratio: If mid_layer is None, use layer at this ratio of depth.
                            Default 0.25 means layer 8 for 32-layer model, layer 9 for 36-layer.
            residual_weight: Weight for residual connection in attention rollout.
            head_aggregation: How to aggregate attention across heads.
        """
        self.mid_layer = mid_layer
        self.mid_layer_ratio = mid_layer_ratio
        self.residual_weight = residual_weight
        self.head_aggregation = head_aggregation
        self._actual_mid_layer = None  # Set during attribute()

    @property
    def name(self) -> str:
        layer = self._actual_mid_layer if self._actual_mid_layer is not None else self.mid_layer
        if layer is not None:
            return f"midlayer_L{layer + 1}"
        return "midlayer"

    def _get_mid_layer(self, model: PreTrainedModel) -> int:
        """Determine the mid-layer index to use."""
        if self.mid_layer is not None:
            return self.mid_layer

        # Auto-select based on model depth ratio
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        else:
            num_layers = 32

        return int(num_layers * self.mid_layer_ratio)

    def _compute_stage1_rollout(
        self,
        attention_layers: List[Tensor],
        target_pos: int,
    ) -> np.ndarray:
        """
        Stage 1: Compute attention rollout from layer 0 to mid_layer.

        Args:
            attention_layers: List of attention tensors per layer.
                Each tensor: [batch, num_heads, seq_len, seq_len]
            target_pos: Target position (used for causal masking, not for extraction here)

        Returns:
            s1: Rollout matrix [seq_len, seq_len] where s1[i,j] is the
                attention flow from input token i to mid-layer token j
        """
        seq_len = attention_layers[0].shape[-1]

        # Initialize rollout as identity
        rollout = np.eye(seq_len, dtype=np.float32)

        # Process layers 0 to mid_layer (inclusive)
        mid_layer = self._actual_mid_layer if self._actual_mid_layer is not None else self.mid_layer
        for layer_idx in range(mid_layer + 1):
            attn = attention_layers[layer_idx]

            # Aggregate across heads
            attn_agg = self._aggregate_heads(attn, method=self.head_aggregation)

            if attn_agg.dim() == 3:
                attn_agg = attn_agg.squeeze(0)

            attn_np = attn_agg.detach().cpu().float().numpy()

            # Mix with identity for residual: A' = α*I + (1-α)*A
            identity = np.eye(seq_len, dtype=np.float32)
            attn_with_residual = (
                self.residual_weight * identity +
                (1 - self.residual_weight) * attn_np
            )

            # Cumulative product: R = A' @ R
            rollout = attn_with_residual @ rollout

        # rollout[j, i] = flow from input i to mid-layer j
        # We want s1[i, j] = flow from input i to mid-layer j
        # So s1 = rollout.T
        s1 = rollout.T

        # Row normalization: normalize each input token's total contribution to 1
        # This ensures each input token has equal "voting power" when combined with s2
        row_sums = s1.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        s1 = s1 / row_sums

        return s1

    def _compute_stage2_causal(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        target_pos: int,
    ) -> Tuple[np.ndarray, Tensor]:
        """
        Stage 2: Compute causal influence of each mid-layer token on final logits.

        For each position j, zero out its hidden state at mid-layer input,
        run remaining layers, and measure the change in target token probability.

        Args:
            model: HuggingFace model
            input_ids: Input token IDs [1, T]
            target_pos: Position to evaluate logits at

        Returns:
            s2: Causal influence scores [T]
            original_logits: Original logits for reference
        """
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]

        # Use hooks to capture hidden states at mid-layer
        mid_layer_hidden = {}

        def capture_mid_layer_hook(module, inputs, outputs):
            # Capture the output of mid_layer-1 (input to mid_layer)
            if isinstance(outputs, tuple):
                mid_layer_hidden['states'] = outputs[0].clone()
            else:
                mid_layer_hidden['states'] = outputs.clone()

        # Get the layer to hook
        mid_layer = self._actual_mid_layer if self._actual_mid_layer is not None else self.mid_layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if mid_layer > 0:
                hook_layer = layers[mid_layer - 1]
            else:
                hook_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer'):
            layers = model.transformer.h
            if mid_layer > 0:
                hook_layer = layers[mid_layer - 1]
            else:
                hook_layer = model.transformer.wte
        else:
            raise ValueError("Unsupported model architecture")

        # Step 1: Run full forward to get original logits and mid-layer hidden states
        with torch.no_grad():
            handle = hook_layer.register_forward_hook(capture_mid_layer_hook)
            outputs = model(input_ids, output_hidden_states=True)
            handle.remove()

            original_logits = outputs.logits[0, target_pos].clone()  # [V]
            mid_layer_input = mid_layer_hidden['states'].clone()  # [1, T, D]

            # Get target token (most likely prediction)
            target_token_id = original_logits.argmax().item()
            original_prob = torch.softmax(original_logits, dim=-1)[target_token_id].item()

        # Step 2: For each position, zero it out and run forward from mid_layer
        s2 = np.zeros(seq_len, dtype=np.float32)

        # Get components for forward from mid_layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if hasattr(model.model, 'norm'):
                final_norm = model.model.norm
            else:
                final_norm = model.model.final_layernorm
            lm_head = model.lm_head
            rotary_emb = getattr(model.model, 'rotary_emb', None)
        else:
            layers = model.transformer.h
            final_norm = model.transformer.ln_f
            lm_head = model.lm_head
            rotary_emb = None

        # Create position_ids for models that need it
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Get position embeddings if model uses rotary
        position_embeddings = None
        if rotary_emb is not None:
            cos, sin = rotary_emb(mid_layer_input, position_ids)
            position_embeddings = (cos, sin)

        with torch.no_grad():
            for j in range(seq_len):
                # Skip positions after target_pos (they don't influence the target)
                if j > target_pos:
                    s2[j] = 0.0
                    continue

                # Zero out position j at mid-layer input
                modified_input = mid_layer_input.clone()
                modified_input[0, j, :] = 0.0

                # Run through remaining layers
                hidden_states = modified_input
                for layer_idx in range(mid_layer, len(layers)):
                    layer = layers[layer_idx]
                    try:
                        if position_embeddings is not None:
                            layer_outputs = layer(
                                hidden_states,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            layer_outputs = layer(hidden_states, position_ids=position_ids)
                    except TypeError:
                        layer_outputs = layer(hidden_states)

                    if isinstance(layer_outputs, tuple):
                        hidden_states = layer_outputs[0]
                    else:
                        hidden_states = layer_outputs

                hidden_states = final_norm(hidden_states)
                modified_logits = lm_head(hidden_states[0, target_pos])  # [V]

                # Compute probability drop for target token
                modified_prob = torch.softmax(modified_logits, dim=-1)[target_token_id].item()
                prob_drop = max(0, original_prob - modified_prob)  # Only count drops

                s2[j] = prob_drop

        return s2, original_logits

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using two-stage mid-layer method.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids, shape [1, T] or [T]
            target_pos: Position to explain

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

        # Get mid_layer (auto-select if not specified)
        self._actual_mid_layer = self._get_mid_layer(model)

        # Extract attention from all layers
        attention_layers = self._extract_attention(model, input_ids)

        # Validate mid_layer
        num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.transformer.h)
        if self._actual_mid_layer >= num_layers:
            raise ValueError(f"mid_layer {self._actual_mid_layer} >= num_layers {num_layers}")

        # Stage 1: Attention rollout from input to mid-layer
        s1 = self._compute_stage1_rollout(attention_layers, target_pos)
        # s1 shape: [T, T], s1[i, j] = flow from input i to mid-layer j

        # Stage 2: Causal influence from mid-layer to logits
        s2, _ = self._compute_stage2_causal(model, input_ids, target_pos)
        # s2 shape: [T]

        # Final score: score(i) = Σ_j s2(j) * s1(i, j)
        # s1[i, :] @ s2 for each i
        scores = s1 @ s2  # [T]

        # Zero out positions after target_pos (causal constraint)
        scores[target_pos + 1:] = 0

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "mid_layer": self._actual_mid_layer,
                "residual_weight": self.residual_weight,
                "head_aggregation": self.head_aggregation,
                "s1_shape": s1.shape,
                "s2_shape": s2.shape,
                "s2_max": float(s2.max()),
                "s2_mean": float(s2.mean()),
            }
        )


def midlayer_attribution(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int = -1,
    mid_layer: int = 7,
    residual_weight: float = 0.5,
    head_aggregation: str = "mean",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for mid-layer attribution.

    Args:
        model: HuggingFace causal language model
        input_ids: Input token ids
        target_pos: Position to explain (default: -1, last position)
        mid_layer: Intermediate layer index (0-indexed, default: 7 = 8th layer)
        residual_weight: Weight for residual connection in attention rollout
        head_aggregation: How to aggregate attention heads

    Returns:
        Importance scores array of shape [T]
    """
    method = MidLayerAttribution(
        mid_layer=mid_layer,
        residual_weight=residual_weight,
        head_aggregation=head_aggregation,
    )
    result = method.attribute(model, input_ids, target_pos, **kwargs)
    return result.scores
