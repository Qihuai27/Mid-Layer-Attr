"""
GPT-2 specific attention hooks.

Compatible with transformers >= 4.40.0 which uses attention_interface pattern.
"""

from functools import partial
from typing import List, Optional, Type

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
    module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    head_mask: Optional[Tensor] = None,
    attention_hook: Optional[AttentionHook] = None,
    **kwargs,
) -> tuple:
    """
    Modified eager attention forward with hook support.

    This replaces the default eager_attention_forward to allow intercepting
    attention weights for observation and intervention.
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # Causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # >>> Hook point: before softmax (for intervention/masking) <<<
    if attention_hook is not None:
        attn_weights = attention_hook(attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # >>> Hook point: after softmax (for gradient capture) <<<
    if attention_hook is not None and isinstance(attention_hook, GradientCaptureHook):
        attn_weights = attention_hook.capture_post_softmax(attn_weights)

    # Downcast (if necessary) back to V's dtype
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


def create_hooked_forward(original_forward, attention_hook: AttentionHook):
    """
    Create a hooked version of GPT2Attention.forward that uses our custom attention.
    """
    def hooked_forward(
        self,
        hidden_states,
        past_key_value=None,
        cache_position=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined."
                )
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        # Use our hooked attention instead of the default
        attn_output, attn_weights = hooked_eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            head_mask=head_mask,
            attention_hook=attention_hook,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output, None)
        return outputs

    return hooked_forward


class GPT2HookManager(HookManager):
    """
    Hook manager for GPT-2 models.

    Compatible with transformers >= 4.40.0 which uses the attention_interface pattern.
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
        return len(self.model.transformer.h)

    def _setup_hooks(self):
        """Register hooks with GPT-2 attention layers."""
        num_layers = self._get_num_layers()

        # Create hooks for each layer
        self.hooks = [self.hook_class(i) for i in range(num_layers)]

        # Store original forwards and replace with hooked versions
        self._original_forwards = []
        for i, layer in enumerate(self.model.transformer.h):
            self._original_forwards.append(layer.attn.forward)
            # Bind the hooked forward to the attention module
            layer.attn.forward = create_hooked_forward(
                layer.attn.forward,
                self.hooks[i],
            ).__get__(layer.attn, type(layer.attn))

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)


class GPT2MaskingHookManager(GPT2HookManager):
    """GPT-2 hook manager specifically for masking experiments."""

    def __init__(self, model: PreTrainedModel, mask_value: float = -1e4):
        self.mask_value = mask_value
        super().__init__(model, hook_class=MaskingHook)

    def _setup_hooks(self):
        """Register masking hooks with GPT-2 attention layers."""
        num_layers = self._get_num_layers()

        # Create masking hooks for each layer
        self.hooks = [MaskingHook(i, self.mask_value) for i in range(num_layers)]

        # Store original forwards and replace with hooked versions
        self._original_forwards = []
        for i, layer in enumerate(self.model.transformer.h):
            self._original_forwards.append(layer.attn.forward)
            layer.attn.forward = create_hooked_forward(
                layer.attn.forward,
                self.hooks[i],
            ).__get__(layer.attn, type(layer.attn))

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)
