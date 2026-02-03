"""
TokenShapley: Token-level Shapley Value Attribution.

Implements Shapley value computation for token-level attribution using
Monte Carlo sampling. Based on the paper:
    "TokenShapley: Token Level Context Attribution with Shapley Value"

Algorithm:
    1. Sample random permutations of tokens
    2. For each permutation, compute marginal contribution of each token
    3. Shapley value = average marginal contribution across permutations

The marginal contribution of token i in permutation π is:
    v(S_π^i ∪ {i}) - v(S_π^i)
where S_π^i is the set of tokens before i in permutation π.

Shapley values satisfy desirable axiomatic properties:
    - Efficiency: contributions sum to total value
    - Symmetry: tokens with identical effects get equal attribution
    - Null player: irrelevant tokens get zero attribution
    - Linearity: additive over multiple games
"""

from typing import Optional, Union, Literal

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import PerturbationBasedMethod, AttributionResult


class TokenShapley(PerturbationBasedMethod):
    """
    TokenShapley attribution using Monte Carlo sampling.

    Computes approximate Shapley values for each token by averaging
    marginal contributions across random permutations.
    """

    def __init__(
        self,
        n_samples: int = 200,
        mask_type: Literal["zero_embed", "pad", "unk"] = "zero_embed",
        baseline_output: Literal["prob", "logit"] = "prob",
    ):
        """
        Initialize TokenShapley.

        Args:
            n_samples: Number of random permutations to sample
            mask_type: How to mask excluded tokens
                - "zero_embed": Replace embedding with zeros (default, most neutral)
                - "pad": Replace with PAD token
                - "unk": Replace with UNK token
            baseline_output: Output type to measure
                - "prob": Use probability of target token
                - "logit": Use raw logit of target token
        """
        self.n_samples = n_samples
        self.mask_type = mask_type
        self.baseline_output = baseline_output

    @property
    def name(self) -> str:
        return "shapley"

    def _get_mask_token_id(
        self,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
        """Get token ID to use for masking based on mask_type."""
        if self.mask_type == "unk":
            token_id = tokenizer.unk_token_id
            if token_id is None:
                token_id = 0
        elif self.mask_type == "pad":
            token_id = tokenizer.pad_token_id
            if token_id is None:
                token_id = tokenizer.eos_token_id or 0
        else:
            raise ValueError(f"Unknown mask_type for token: {self.mask_type}")
        return token_id

    def _compute_value(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        mask: np.ndarray,
        target_pos: int,
        target_token_id: int,
        mask_token_id: Optional[int] = None,
    ) -> float:
        """
        Compute model output value for a subset of tokens.

        Args:
            model: HuggingFace model
            input_ids: Original input [1, T]
            mask: Binary mask [T] where 1 = include, 0 = exclude
            target_pos: Position to evaluate
            target_token_id: Target token for probability
            mask_token_id: Token ID for masking (if not using zero_embed)

        Returns:
            Output value (probability or logit of target token)
        """
        device = input_ids.device
        T = input_ids.shape[1]

        if self.mask_type == "zero_embed":
            # Zero out embeddings of excluded tokens
            embed_layer = model.get_input_embeddings()
            embeds = embed_layer(input_ids)  # [1, T, d]
            # Match mask dtype to embeddings dtype (handles float16/bfloat16 models)
            mask_tensor = torch.from_numpy(mask).to(device=device, dtype=embeds.dtype)
            embeds = embeds * mask_tensor.view(1, T, 1)
            outputs = model(inputs_embeds=embeds)
        else:
            # Replace excluded tokens with mask token
            masked_ids = input_ids.clone()
            mask_tensor = torch.from_numpy(mask).bool().to(device)
            masked_ids[0, ~mask_tensor] = mask_token_id
            outputs = model(masked_ids)

        logits = outputs.logits[0, target_pos]

        if self.baseline_output == "prob":
            probs = torch.softmax(logits, dim=-1)
            return probs[target_token_id].item()
        else:
            return logits[target_token_id].item()

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        target_token_id: Optional[int] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute Shapley value attribution scores.

        Args:
            model: HuggingFace causal LM
            input_ids: Input token ids [1, T] or [T]
            target_pos: Position to explain
            target_token_id: Token id being predicted (if None, use argmax)
            tokenizer: Tokenizer for masking (required for pad/unk mask_type)
            **kwargs: Additional arguments

        Returns:
            AttributionResult with Shapley value scores
        """
        device = next(model.parameters()).device

        # Ensure input_ids is 2D
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        T = input_ids.shape[1]

        # Get target token if not provided
        if target_token_id is None:
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, target_pos]
                target_token_id = logits.argmax().item()

        # Get mask token ID if needed
        mask_token_id = None
        if self.mask_type != "zero_embed":
            if tokenizer is None:
                mask_token_id = 0
            else:
                mask_token_id = self._get_mask_token_id(tokenizer)

        # Initialize Shapley values
        shapley_values = np.zeros(T, dtype=np.float64)

        # Monte Carlo sampling of permutations
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Random permutation of token indices
                perm = np.random.permutation(T)

                # Track which tokens are in the current coalition
                coalition = np.zeros(T, dtype=np.float32)
                prev_value = self._compute_value(
                    model, input_ids, coalition, target_pos,
                    target_token_id, mask_token_id
                )

                # Add tokens one by one in permutation order
                for idx in perm:
                    coalition[idx] = 1.0
                    curr_value = self._compute_value(
                        model, input_ids, coalition, target_pos,
                        target_token_id, mask_token_id
                    )

                    # Marginal contribution
                    shapley_values[idx] += curr_value - prev_value
                    prev_value = curr_value

        # Average over samples
        shapley_values /= self.n_samples

        return AttributionResult(
            scores=shapley_values.astype(np.float32),
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "n_samples": self.n_samples,
                "mask_type": self.mask_type,
                "baseline_output": self.baseline_output,
                "target_token_id": target_token_id,
            }
        )


def shapley(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int,
    target_token_id: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    n_samples: int = 200,
    mask_type: str = "zero_embed",
    **kwargs,
) -> np.ndarray:
    """
    Functional interface for TokenShapley attribution.

    Args:
        model: HuggingFace causal LM
        input_ids: Input token ids
        target_pos: Position to explain
        target_token_id: Token being predicted (optional)
        tokenizer: Tokenizer for masking
        n_samples: Number of permutation samples
        mask_type: How to mask excluded tokens

    Returns:
        scores: Shapley value attribution scores [T]
    """
    method = TokenShapley(n_samples=n_samples, mask_type=mask_type)
    result = method.attribute(
        model, input_ids, target_pos, target_token_id, tokenizer=tokenizer, **kwargs
    )
    return result.scores
