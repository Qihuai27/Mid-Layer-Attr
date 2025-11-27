"""
GPT-J specific attention hooks.
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
    MaskingHook,
    create_forward_wrapper,
)


def gptj_attention_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    attention_hook: Optional[AttentionHook] = None,
) -> tuple:
    """
    Modified GPT-J attention forward pass with hook support.

    This replaces the model's _attn method to allow intercepting attention weights.
    """
    query_length, key_length = query.size(-2), key.size(-2)

    # Get causal mask
    causal_mask = self.bias[:, :, key_length - query_length:key_length, :key_length]

    # Convert to float32 for attention computation
    query = query.to(torch.float32)
    key = key.to(torch.float32)

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # Apply causal mask
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    # Scale
    attn_weights = attn_weights / self.scale_attn

    # Apply attention mask (padding)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # >>> Hook point: before softmax <<<
    if attention_hook is not None:
        attn_weights = attention_hook(attn_weights)

    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Convert back to value dtype and apply dropout
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    # Apply head mask
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # Compute output
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class GPTJHookManager(HookManager):
    """
    Hook manager for GPT-J models.

    Registers hooks with each transformer layer's attention module.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hook_class: Type[AttentionHook] = GradientCaptureHook,
    ):
        self.hook_class = hook_class
        super().__init__(model)

    def _get_num_layers(self) -> int:
        return len(self.model.transformer.h)

    def _setup_hooks(self):
        """Register hooks with GPT-J attention layers."""
        num_layers = self._get_num_layers()

        # Create hooks for each layer
        self.hooks = [self.hook_class(i) for i in range(num_layers)]

        # Replace _attn method in each layer
        for i, layer in enumerate(self.model.transformer.h):
            layer.attn._attn = partial(
                gptj_attention_forward,
                layer.attn,
                attention_hook=self.hooks[i],
            )

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)


class GPTJMaskingHookManager(GPTJHookManager):
    """GPT-J hook manager specifically for masking experiments."""

    def __init__(self, model: PreTrainedModel, mask_value: float = -1e4):
        self.mask_value = mask_value
        super().__init__(model, hook_class=MaskingHook)

    def _setup_hooks(self):
        """Register masking hooks with GPT-J attention layers."""
        num_layers = self._get_num_layers()

        # Create masking hooks for each layer
        self.hooks = [MaskingHook(i, self.mask_value) for i in range(num_layers)]

        # Replace _attn method in each layer
        for i, layer in enumerate(self.model.transformer.h):
            layer.attn._attn = partial(
                gptj_attention_forward,
                layer.attn,
                attention_hook=self.hooks[i],
            )

        # Wrap model forward to register input_ids
        self.model.forward = create_forward_wrapper(self)(self.model.forward)
