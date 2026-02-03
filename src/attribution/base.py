"""
Base class for attribution methods.

All attribution methods should inherit from AttributionMethod and implement
the `attribute()` method to compute token-level importance scores.

Design Principles:
1. Input: model, input_ids, target_pos (the position whose prediction we analyze)
2. Output: importance_scores array of length T, where each value indicates
   how important that input token is for the prediction at target_pos
3. Higher scores = more important tokens
4. Output can be directly used with src/eval metrics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class AttributionResult:
    """
    Result container for attribution methods.

    Attributes:
        scores: Token-level importance scores, shape [T] or [T, T] for pairwise
        target_pos: The target position being explained
        method_name: Name of the attribution method
        metadata: Additional method-specific information
    """
    scores: np.ndarray
    target_pos: int
    method_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        """Number of input tokens."""
        return self.scores.shape[0]

    def get_top_k_indices(self, k: int) -> np.ndarray:
        """Get indices of top-k most important tokens."""
        return np.argsort(-self.scores)[:k]

    def get_top_k_scores(self, k: int) -> np.ndarray:
        """Get top-k importance scores."""
        indices = self.get_top_k_indices(k)
        return self.scores[indices]

    def normalize(self, method: str = "minmax") -> "AttributionResult":
        """
        Return a new AttributionResult with normalized scores.

        Args:
            method: Normalization method
                - "minmax": Scale to [0, 1]
                - "sum": Divide by sum (scores sum to 1)
                - "softmax": Apply softmax
                - "zscore": Z-score normalization

        Returns:
            New AttributionResult with normalized scores
        """
        scores = self.scores.copy()

        if method == "minmax":
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val > 1e-10:
                scores = (scores - min_val) / (max_val - min_val)
            else:
                scores = np.zeros_like(scores)
        elif method == "sum":
            total = np.abs(scores).sum()
            if total > 1e-10:
                scores = scores / total
        elif method == "softmax":
            exp_scores = np.exp(scores - scores.max())
            scores = exp_scores / exp_scores.sum()
        elif method == "zscore":
            mean, std = scores.mean(), scores.std()
            if std > 1e-10:
                scores = (scores - mean) / std
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return AttributionResult(
            scores=scores,
            target_pos=self.target_pos,
            method_name=self.method_name,
            metadata={**self.metadata, "normalized": method}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scores": self.scores.tolist(),
            "target_pos": self.target_pos,
            "method_name": self.method_name,
            "metadata": self.metadata,
        }


class AttributionMethod(ABC):
    """
    Abstract base class for attribution methods.

    All attribution methods compute importance scores for input tokens
    with respect to the model's prediction at a target position.

    For autoregressive LMs:
    - target_pos is typically the last token position
    - The model predicts the next token based on positions 0..target_pos
    - importance_scores[i] indicates how important token i is for that prediction
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the attribution method."""
        pass

    @property
    def requires_grad(self) -> bool:
        """Whether this method requires gradient computation."""
        return False

    @property
    def requires_baseline(self) -> bool:
        """Whether this method requires a baseline input."""
        return False

    @abstractmethod
    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores for input tokens.

        Args:
            model: HuggingFace causal language model
            input_ids: Input token ids, shape [1, T] or [T]
            target_pos: Position to explain (typically last token for next-token prediction)
            **kwargs: Method-specific arguments

        Returns:
            AttributionResult containing importance scores and metadata
        """
        pass

    def __call__(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """Alias for attribute()."""
        return self.attribute(model, input_ids, target_pos, **kwargs)


class AttentionBasedMethod(AttributionMethod):
    """
    Base class for attention-based attribution methods.

    These methods extract and process attention weights to compute
    token importance scores.
    """

    def _extract_attention(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
    ) -> List[Tensor]:
        """
        Extract attention weights from all layers.

        Args:
            model: HuggingFace model
            input_ids: Input token ids [1, T]

        Returns:
            List of attention tensors, one per layer
            Each tensor has shape [batch, num_heads, seq_len, seq_len]
        """
        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions is a tuple of tensors, one per layer
        # Each tensor: [batch, num_heads, seq_len, seq_len]
        return list(outputs.attentions)

    def _aggregate_heads(
        self,
        attention: Tensor,
        method: str = "mean"
    ) -> Tensor:
        """
        Aggregate attention across heads.

        Args:
            attention: Attention tensor [batch, num_heads, seq_q, seq_k]
            method: Aggregation method ("mean", "max", "sum")

        Returns:
            Aggregated attention [batch, seq_q, seq_k]
        """
        if method == "mean":
            return attention.mean(dim=1)
        elif method == "max":
            return attention.max(dim=1).values
        elif method == "sum":
            return attention.sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods.

    These methods use gradients of the output with respect to inputs
    to compute importance scores.
    """

    @property
    def requires_grad(self) -> bool:
        return True

    def _get_embeddings(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
    ) -> Tensor:
        """
        Get input embeddings with gradient tracking.

        Args:
            model: HuggingFace model
            input_ids: Input token ids [1, T]

        Returns:
            Embeddings tensor [1, T, d] with requires_grad=True
        """
        embed_layer = model.get_input_embeddings()
        embeddings = embed_layer(input_ids)
        embeddings = embeddings.detach().requires_grad_(True)
        return embeddings

    def _compute_loss(
        self,
        logits: Tensor,
        target_pos: int,
        target_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Compute loss for gradient computation.

        Args:
            logits: Model output logits [1, T, vocab_size]
            target_pos: Position to compute loss for
            target_token_id: If provided, use cross-entropy with this target
                            Otherwise, use max logit

        Returns:
            Scalar loss tensor
        """
        target_logits = logits[0, target_pos]  # [vocab_size]

        if target_token_id is not None:
            # Cross-entropy loss with specific target
            log_probs = torch.log_softmax(target_logits, dim=-1)
            loss = -log_probs[target_token_id]
        else:
            # Use max logit (most likely prediction)
            loss = target_logits.max()

        return loss


class PerturbationBasedMethod(AttributionMethod):
    """
    Base class for perturbation-based attribution methods.

    These methods perturb inputs and measure the effect on outputs
    to compute importance scores.
    """

    @property
    def requires_baseline(self) -> bool:
        return True

    def _create_baseline(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        baseline_type: str = "pad",
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Tensor:
        """
        Create baseline input for perturbation.

        Args:
            model: HuggingFace model
            input_ids: Original input [1, T]
            baseline_type: Type of baseline
                - "pad": All PAD tokens
                - "unk": All UNK tokens
                - "zero_embed": Zero embeddings (returns embeddings, not ids)
                - "mask": All MASK tokens (for MLM models)
            tokenizer: Required for pad/unk/mask baselines

        Returns:
            Baseline input_ids [1, T] or embeddings [1, T, d]
        """
        T = input_ids.shape[1]
        device = input_ids.device

        if baseline_type == "zero_embed":
            embed_layer = model.get_input_embeddings()
            d = embed_layer.embedding_dim
            return torch.zeros(1, T, d, device=device, dtype=embed_layer.weight.dtype)

        if tokenizer is None:
            raise ValueError(f"tokenizer required for baseline_type={baseline_type}")

        if baseline_type == "pad":
            token_id = tokenizer.pad_token_id
            if token_id is None:
                token_id = tokenizer.eos_token_id or 0
        elif baseline_type == "unk":
            token_id = tokenizer.unk_token_id
            if token_id is None:
                token_id = 0
        elif baseline_type == "mask":
            token_id = tokenizer.mask_token_id
            if token_id is None:
                raise ValueError("Tokenizer has no mask token")
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        return torch.full((1, T), token_id, device=device, dtype=input_ids.dtype)
