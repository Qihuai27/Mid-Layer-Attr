"""
Base classes for attention hooks.

The hook system allows intercepting attention weights during forward passes
for both observation (attribution analysis) and intervention (ablation).
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel


class HookMode(Enum):
    """Mode of operation for attention hooks."""
    DISABLED = "disabled"      # Pass through without modification
    OBSERVE = "observe"        # Capture attention weights without modification
    INTERVENE = "intervene"    # Modify attention weights


class AttentionHook(nn.Module):
    """
    Base class for attention hooks.

    Hooks can operate in three modes:
    - DISABLED: Pass attention weights through unchanged
    - OBSERVE: Capture attention weights for analysis (no modification)
    - INTERVENE: Apply modifications to attention weights

    Subclasses implement specific behaviors by overriding _observe() and _intervene().
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.mode = HookMode.DISABLED
        self._input_ids: Optional[Tensor] = None
        self._captured_weights: Optional[Tensor] = None

    def forward(self, attn_weights: Tensor) -> Tensor:
        """
        Process attention weights based on current mode.

        Args:
            attn_weights: Attention weights [batch, heads, seq_q, seq_k]

        Returns:
            Processed attention weights
        """
        if self.mode == HookMode.DISABLED:
            return attn_weights
        elif self.mode == HookMode.OBSERVE:
            return self._observe(attn_weights)
        elif self.mode == HookMode.INTERVENE:
            return self._intervene(attn_weights)
        else:
            return attn_weights

    def _observe(self, attn_weights: Tensor) -> Tensor:
        """
        Observe (capture) attention weights without modification.

        Override this method for custom observation behavior.
        """
        self._captured_weights = attn_weights.detach().clone()
        return attn_weights

    def _intervene(self, attn_weights: Tensor) -> Tensor:
        """
        Intervene on attention weights.

        Override this method for custom intervention behavior.
        """
        return attn_weights

    def register_input_ids(self, input_ids: Tensor):
        """Register current input IDs for position-aware operations."""
        self._input_ids = input_ids

    @property
    def input_ids(self) -> Optional[Tensor]:
        return self._input_ids

    @property
    def captured_weights(self) -> Optional[Tensor]:
        return self._captured_weights

    def reset(self):
        """Reset captured state."""
        self._captured_weights = None


class GradientCaptureHook(AttentionHook):
    """
    Hook that captures gradients for attribution analysis.

    This hook uses register_hook on attention weights after softmax to capture
    gradients flowing back through the attention weights.
    """

    def __init__(self, layer_idx: int):
        super().__init__(layer_idx)
        self._gradient: Optional[Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _observe(self, attn_weights: Tensor) -> Tensor:
        """Observe with gradient tracking via tensor hook."""
        self._captured_weights = attn_weights.detach().clone()
        return attn_weights

    def capture_post_softmax(self, attn_weights: Tensor) -> Tensor:
        """
        Capture gradients on post-softmax attention weights.

        This should be called after softmax in the attention forward pass.
        Uses register_hook to capture gradients without modifying the computation.
        """
        # Store weights (detached for memory efficiency)
        self._captured_weights = attn_weights.detach().clone()

        # Register gradient hook if in observe mode and tensor requires grad context
        if self.mode == HookMode.OBSERVE and attn_weights.requires_grad:
            def grad_hook(grad: Tensor):
                self._gradient = grad.detach().clone()
            self._hook_handle = attn_weights.register_hook(grad_hook)

        return attn_weights

    @property
    def gradient(self) -> Optional[Tensor]:
        """Get captured gradient."""
        return self._gradient

    def zero_grad(self):
        """Clear gradients."""
        self._gradient = None
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        """Reset all state."""
        super().reset()
        self._gradient = None
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class MaskingHook(AttentionHook):
    """
    Hook that masks (zeros out) specific attention positions.

    Used for causal ablation experiments to block information flow
    from anchor words to other positions.
    """

    def __init__(
        self,
        layer_idx: int,
        mask_value: float = -1e4,
    ):
        super().__init__(layer_idx)
        self.mask_value = mask_value
        self._mask_positions: Optional[Callable[[Tensor], Tensor]] = None

    def set_mask_fn(self, mask_fn: Callable[[Tensor, Tensor], Tensor]):
        """
        Set the masking function.

        Args:
            mask_fn: Function that takes (attn_weights, input_ids) and returns
                    a boolean mask of positions to mask out.
        """
        self._mask_positions = mask_fn

    def _intervene(self, attn_weights: Tensor) -> Tensor:
        """Apply masking to attention weights."""
        if self._mask_positions is None or self._input_ids is None:
            return attn_weights

        mask = self._mask_positions(attn_weights, self._input_ids)
        attn_weights = attn_weights.masked_fill(mask, self.mask_value)

        return attn_weights


class HookManager(ABC):
    """
    Manager for registering and coordinating attention hooks across model layers.

    This is an abstract base class. Concrete implementations (GPT2HookManager, etc.)
    handle model-specific hook registration.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.hooks: List[AttentionHook] = []
        self._original_forward = None
        self._setup_hooks()

    @abstractmethod
    def _setup_hooks(self):
        """Register hooks with model layers. Implemented by subclasses."""
        pass

    @abstractmethod
    def _get_num_layers(self) -> int:
        """Get number of layers in the model."""
        pass

    def register_input_ids(self, input_ids: Tensor):
        """Register input IDs with all hooks."""
        for hook in self.hooks:
            hook.register_input_ids(input_ids)

    def set_mode(
        self,
        mode: HookMode,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Set the mode for hooks.

        Args:
            mode: HookMode to set
            layer_indices: Which layers to affect (None = all layers)
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.hooks)))

        for idx in layer_indices:
            if 0 <= idx < len(self.hooks):
                self.hooks[idx].mode = mode

    def enable_observation(self, layer_indices: Optional[List[int]] = None):
        """Enable observation mode for specified layers."""
        self.set_mode(HookMode.OBSERVE, layer_indices)

    def enable_intervention(self, layer_indices: Optional[List[int]] = None):
        """Enable intervention mode for specified layers."""
        self.set_mode(HookMode.INTERVENE, layer_indices)

    def disable(self, layer_indices: Optional[List[int]] = None):
        """Disable hooks for specified layers."""
        self.set_mode(HookMode.DISABLED, layer_indices)

    def disable_all(self):
        """Disable all hooks."""
        self.set_mode(HookMode.DISABLED)

    def get_captured_weights(self) -> List[Optional[Tensor]]:
        """Get captured attention weights from all layers."""
        return [hook.captured_weights for hook in self.hooks]

    def get_gradients(self) -> List[Optional[Tensor]]:
        """Get captured gradients from all layers (for GradientCaptureHook)."""
        gradients = []
        for hook in self.hooks:
            if isinstance(hook, GradientCaptureHook):
                gradients.append(hook.gradient)
            else:
                gradients.append(None)
        return gradients

    def reset(self):
        """Reset all hooks."""
        for hook in self.hooks:
            hook.reset()

    def zero_grad(self):
        """Zero gradients on all hooks."""
        for hook in self.hooks:
            if isinstance(hook, GradientCaptureHook):
                hook.zero_grad()


def create_forward_wrapper(manager: HookManager):
    """Create a wrapper for model forward that registers input_ids with hooks."""
    def wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            # Extract input_ids from args or kwargs
            input_ids = kwargs.get('input_ids')
            if input_ids is None and args:
                input_ids = args[0]

            if input_ids is not None:
                manager.register_input_ids(input_ids)

            return fn(*args, **kwargs)
        return wrapped
    return wrapper
