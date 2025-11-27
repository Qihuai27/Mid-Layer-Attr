"""
Attention hook system for intercepting and modifying attention weights.
"""

from .base import AttentionHook, HookManager
from .gpt2 import GPT2HookManager
from .gptj import GPTJHookManager


def get_hook_manager(model, model_name: str) -> HookManager:
    """
    Factory function to get the appropriate hook manager for a model.

    Args:
        model: The transformer model
        model_name: Name/type of the model

    Returns:
        HookManager instance for the model
    """
    model_name_lower = model_name.lower()

    if "gpt2" in model_name_lower:
        return GPT2HookManager(model)
    elif "gpt-j" in model_name_lower or "gptj" in model_name_lower:
        return GPTJHookManager(model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


__all__ = [
    "AttentionHook",
    "HookManager",
    "GPT2HookManager",
    "GPTJHookManager",
    "get_hook_manager",
]
