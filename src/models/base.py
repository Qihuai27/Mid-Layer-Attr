"""
Base model wrapper for language models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Project root for finding local models
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda:0",
    torch_dtype: Optional[torch.dtype] = None,
    local_path: Optional[str] = None,
    max_length: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a causal language model and its tokenizer.

    Args:
        model_name: Model identifier (e.g., 'gpt2-xl', 'EleutherAI/gpt-j-6b')
                   If a local model directory exists in model/{model_name}, it will be used.
        device: Device to load the model on
        torch_dtype: Data type for model weights
        local_path: Optional local path to load from
        max_length: Optional max sequence length (useful for models with very large default)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if model exists in local model/ directory
    if local_path:
        model_path = local_path
    else:
        local_model_dir = _PROJECT_ROOT / "model" / model_name
        if local_model_dir.exists():
            model_path = str(local_model_dir)
            print(f"Loading model from local path: {model_path}")
        else:
            model_path = model_name

    # Auto-select dtype based on model
    # phi-2 has numerical stability issues with float16, needs float32
    if torch_dtype is None:
        model_name_lower = model_name.lower() if isinstance(model_name, str) else ""
        if "phi" in model_name_lower or "phi" in model_path.lower():
            torch_dtype = torch.float32  # phi-2 needs float32 for numerical stability
        else:
            torch_dtype = torch.float16  # Default to float16 for other models

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable max_length for models with very large defaults
    if max_length is not None:
        tokenizer.model_max_length = max_length
    elif tokenizer.model_max_length > 512:
        tokenizer.model_max_length = 256  # Reasonable default for ICL tasks

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if "cuda" in device else None,
        attn_implementation="eager",  # Required for output_attentions with custom hooks
    )

    if "cuda" not in device:
        model = model.to(device)

    model.eval()

    return model, tokenizer


def get_model_num_layers(model: PreTrainedModel, model_name: str) -> int:
    """Get the number of transformer layers in a model."""
    model_name_lower = model_name.lower()

    if "gpt2" in model_name_lower:
        return len(model.transformer.h)
    elif "gpt-j" in model_name_lower or "gptj" in model_name_lower:
        return len(model.transformer.h)
    elif "llama" in model_name_lower or "qwen" in model_name_lower:
        return len(model.model.layers)
    else:
        # Try common patterns
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return len(model.transformer.h)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        else:
            raise ValueError(f"Cannot determine number of layers for model: {model_name}")


class LMWrapper(nn.Module):
    """
    Wrapper around a causal language model for ICL experiments.

    Provides a unified interface for forward passes with configurable
    output options (logits, probabilities, attention weights, etc.).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.num_layers = get_model_num_layers(model, model_name)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple

        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_final_logits(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get logits at the final position for each sequence.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Logits at final position [batch_size, vocab_size]
        """
        outputs = self.forward(input_ids, attention_mask)

        # Find final positions (last non-padding token)
        if attention_mask is not None:
            final_positions = attention_mask.sum(dim=-1) - 1
        else:
            final_positions = torch.full(
                (input_ids.size(0),), input_ids.size(1) - 1,
                device=input_ids.device
            )

        # Extract logits at final positions
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        final_logits = outputs.logits[batch_indices, final_positions]

        return final_logits

    def get_label_probs(
        self,
        input_ids: Tensor,
        label_token_ids: Dict[int, int],
        attention_mask: Optional[Tensor] = None,
        normalize: bool = True,
    ) -> Tensor:
        """
        Get probabilities for specific label tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            label_token_ids: Dict mapping label index to token ID
            attention_mask: Attention mask
            normalize: Whether to normalize probabilities to sum to 1

        Returns:
            Label probabilities [batch_size, num_labels]
        """
        logits = self.get_final_logits(input_ids, attention_mask)

        # Extract logits for label tokens
        label_indices = sorted(label_token_ids.keys())
        token_ids = [label_token_ids[i] for i in label_indices]

        label_logits = logits[:, token_ids]  # [batch_size, num_labels]

        if normalize:
            probs = torch.softmax(label_logits, dim=-1)
        else:
            probs = torch.sigmoid(label_logits)

        return probs
