"""
Qwen3 specific attention hooks.

Compatible with transformers >= 4.40.0 and Qwen3 models.
"""

from functools import partial
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


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    Repeat KV heads to match query heads (for GQA).
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


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
    Modified eager attention forward with hook support for Qwen3.
    """
    # Repeat KV for GQA
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # >>> Hook point: before softmax (for intervention/masking) <<<
    if attention_hook is not None:
        attn_weights = attention_hook(attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # >>> Hook point: after softmax (for gradient capture) <<<
    if attention_hook is not None and isinstance(attention_hook, GradientCaptureHook):
        attn_weights = attention_hook.capture_post_softmax(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def create_hooked_forward(original_forward, attention_hook: AttentionHook):
    """
    Create a hooked version of Qwen3Attention.forward that uses our custom attention.
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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return hooked_forward


class Qwen3HookManager(HookManager):
    """
    Hook manager for Qwen3 models.
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
        """Register hooks with Qwen3 attention layers."""
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


class Qwen3MaskingHookManager(Qwen3HookManager):
    """Qwen3 hook manager specifically for masking experiments."""

    def __init__(self, model: PreTrainedModel, mask_value: float = -1e4):
        self.mask_value = mask_value
        super().__init__(model, hook_class=MaskingHook)

    def _setup_hooks(self):
        """Register masking hooks with Qwen3 attention layers."""
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
