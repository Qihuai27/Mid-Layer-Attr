"""
Phi specific attention hooks.

Compatible with transformers >= 4.37.0 and Phi models (phi-2).
"""

from typing import List, Optional, Type, Tuple, Callable

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .base import (
    AttentionHook,
    GradientCaptureHook,
    HookManager,
    HookMode,
    MaskingHook,
    create_forward_wrapper,
)


def hooked_eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    attention_hook: Optional[AttentionHook] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Modified eager attention forward with hook support for Phi.
    """
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # >>> Hook point: before softmax (for intervention/masking) <<<
    if attention_hook is not None:
        attn_weights = attention_hook(attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # >>> Hook point: after softmax (for gradient capture) <<<
    if attention_hook is not None and isinstance(attention_hook, GradientCaptureHook):
        attn_weights = attention_hook.capture_post_softmax(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def create_hooked_forward(original_forward, attention_hook: AttentionHook):
    """
    Create a hooked version of PhiAttention.forward that uses our custom attention.
    """
    def hooked_forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        # Apply rotary embeddings
        from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
        cos, sin = position_embeddings

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_ndims],
            query_states[..., self.rotary_ndims :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_ndims],
            key_states[..., self.rotary_ndims :],
        )
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use our hooked attention
        attn_output, attn_weights = hooked_eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            attention_hook=attention_hook,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights

    return hooked_forward


class PhiHookManager(HookManager):
    """
    Hook manager for Phi models (phi-2).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hook_class: Type[AttentionHook] = GradientCaptureHook,
    ):
        self.hook_class = hook_class
        self._original_forwards = []
        super().__init__(model)

    def _get_num_layers(self) -> int:
        return len(self.model.model.layers)

    def _setup_hooks(self):
        """Register hooks with Phi attention layers."""
        num_layers = self._get_num_layers()

        # Create hooks for each layer
        self.hooks = [self.hook_class(i) for i in range(num_layers)]

        # Store original forwards and replace with hooked versions
        self._original_forwards = []
        for i, layer in enumerate(self.model.model.layers):
            self._original_forwards.append(layer.self_attn.forward)
            # Bind the hooked forward to the attention module
            layer.self_attn.forward = create_hooked_forward(
                layer.self_attn.forward,
                self.hooks[i],
            ).__get__(layer.self_attn, type(layer.self_attn))

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)


class PhiMaskingHookManager(PhiHookManager):
    """Phi hook manager specifically for masking experiments."""

    def __init__(self, model: PreTrainedModel, mask_value: float = -1e4):
        self.mask_value = mask_value
        super().__init__(model, hook_class=MaskingHook)

    def _setup_hooks(self):
        """Register masking hooks with Phi attention layers."""
        num_layers = self._get_num_layers()

        # Create masking hooks for each layer
        self.hooks = [MaskingHook(i, self.mask_value) for i in range(num_layers)]

        # Store original forwards and replace with hooked versions
        self._original_forwards = []
        for i, layer in enumerate(self.model.model.layers):
            self._original_forwards.append(layer.self_attn.forward)
            layer.self_attn.forward = create_hooked_forward(
                layer.self_attn.forward,
                self.hooks[i],
            ).__get__(layer.self_attn, type(layer.self_attn))

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)
