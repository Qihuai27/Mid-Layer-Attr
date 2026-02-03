"""
Unified cluster point position detection.

This module provides a single, unified implementation for detecting cluster point
(anchor word) positions in tokenized sequences. Cluster points are label words
(e.g., "Positive", "Negative") that aggregate information from preceding text.

The detection uses a bigram matching strategy:
prefix tokens + label token form a unique pattern that identifies cluster positions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from .definitions import TaskConfig


@dataclass
class ClusterPositions:
    """
    Container for detected cluster point positions in a sequence.

    Attributes:
        label_positions: Dict mapping label index to tensor of positions
                        where that label's cluster word appears
        final_position: Position of the final (prediction) token
        sequence_length: Total length of the sequence
    """
    label_positions: Dict[int, Tensor]  # {0: tensor([pos1, pos2]), 1: tensor([pos3, pos4])}
    final_position: Tensor  # tensor([final_pos]) or tensor([fp1, fp2, ...]) for batch
    sequence_length: int

    @property
    def all_cluster_positions(self) -> Tensor:
        """Get all cluster positions as a single tensor."""
        all_pos = []
        for positions in self.label_positions.values():
            all_pos.append(positions)
        if all_pos:
            return torch.cat(all_pos, dim=-1)
        return torch.tensor([], dtype=torch.long)

    @property
    def num_clusters(self) -> int:
        """Total number of cluster positions detected."""
        return sum(pos.numel() for pos in self.label_positions.values())

    def get_non_cluster_positions(self, exclude_final: bool = True) -> Tensor:
        """
        Get positions that are NOT cluster points.

        Args:
            exclude_final: Also exclude the final position

        Returns:
            Tensor of non-cluster positions
        """
        all_positions = set(range(self.sequence_length))
        cluster_positions = set(self.all_cluster_positions.tolist())

        if exclude_final:
            final_pos = self.final_position.item() if self.final_position.numel() == 1 else self.final_position[0].item()
            cluster_positions.add(final_pos)

        non_cluster = sorted(all_positions - cluster_positions)
        return torch.tensor(non_cluster, dtype=torch.long)


class ClusterDetector:
    """
    Unified cluster point position detector.

    This class encapsulates the logic for finding cluster point positions
    in tokenized ICL sequences. It uses a bigram matching strategy:
    prefix tokens + label token form a unique pattern that identifies
    cluster positions.

    Example:
        For SST2 with template "Review: {text}\nSentiment:{label}":
        - prefix = ["Sentiment", ":"]
        - label = " Positive"
        - The detector finds positions where "Sentiment" + ":" + " Positive" appears
    """

    def __init__(
        self,
        task_config: TaskConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize the cluster detector.

        Args:
            task_config: Task configuration containing labels and cluster patterns
            tokenizer: Tokenizer for encoding prefix/label tokens
        """
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        # Encode prefix tokens
        self.prefix_token_ids = self._encode_prefix(task_config.cluster_prefix)

        # Encode label tokens
        self.label_token_ids = {
            label_idx: self._encode_label(label_text)
            for label_idx, label_text in task_config.labels.items()
        }

    def _encode_prefix(self, prefix_parts: List[str]) -> List[int]:
        """Encode prefix parts to token IDs."""
        token_ids = []
        for part in prefix_parts:
            # Handle special encoding (some words need space prefix)
            encoded = self.tokenizer.encode(part, add_special_tokens=False)
            if encoded:
                token_ids.append(encoded[-1])  # Take the last token if multiple
        return token_ids

    def _encode_label(self, label_text: str) -> int:
        """Encode label text to a single token ID."""
        encoded = self.tokenizer.encode(label_text, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Label '{label_text}' encodes to empty sequence")
        return encoded[0]  # Label should be a single token

    def detect(self, input_ids: Tensor) -> ClusterPositions:
        """
        Detect cluster point positions in the input sequence.

        This uses a clever encoding trick to find bigram/trigram patterns:
        We encode position info as: id + prev_id * 100000 + prev_prev_id * 100000^2
        This creates a unique identifier for each position based on its context.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len] or [seq_len]

        Returns:
            ClusterPositions containing detected positions
        """
        # Handle unbatched input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Find final position (last non-padding token)
        if self.pad_token_id is not None:
            final_pos = (input_ids != self.pad_token_id).int().sum(-1) - 1
        else:
            final_pos = torch.full((batch_size,), seq_len - 1, device=device)

        # Detect positions for each label
        label_positions = {}

        for label_idx, label_token_id in self.label_token_ids.items():
            positions = self._find_cluster_pattern(
                input_ids, label_token_id, self.prefix_token_ids, device
            )
            label_positions[label_idx] = positions

        return ClusterPositions(
            label_positions=label_positions,
            final_position=final_pos,
            sequence_length=seq_len,
        )

    def _find_cluster_pattern(
        self,
        input_ids: Tensor,
        label_token_id: int,
        prefix_token_ids: List[int],
        device: torch.device,
    ) -> Tensor:
        """
        Find positions matching the cluster pattern (prefix + label).

        Uses polynomial encoding for efficient bigram/trigram matching:
        encoded[i] = input[i] + input[i-1] * BASE + input[i-2] * BASE^2

        Args:
            input_ids: Shape [batch_size, seq_len]
            label_token_id: Token ID of the label word
            prefix_token_ids: Token IDs of the prefix pattern
            device: Device for tensor operations

        Returns:
            Tensor of positions where the pattern matches
        """
        BASE = 100000  # Large enough to avoid collisions

        batch_size, seq_len = input_ids.shape

        # Build target pattern value
        target = label_token_id
        for offset, prefix_id in enumerate(reversed(prefix_token_ids)):
            target += prefix_id * (BASE ** (offset + 1))

        # Build encoded sequence with context
        encoded = input_ids.detach().clone()

        if len(prefix_token_ids) >= 1:
            encoded[:, 1:] += input_ids[:, :-1] * BASE
        if len(prefix_token_ids) >= 2:
            encoded[:, 2:] += input_ids[:, :-2] * (BASE ** 2)

        # Find matching positions
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        matches = (encoded == target)

        # For batched input, we return positions per batch item
        # For simplicity, flatten if batch_size == 1
        if batch_size == 1:
            positions = position_indices[matches]
        else:
            # Return as list of tensors per batch item
            positions = position_indices[matches].reshape(batch_size, -1)

        return positions

    def get_label_id_dict(self) -> Dict[int, int]:
        """Get mapping from label index to token ID."""
        return self.label_token_ids.copy()


class RandomClusterDetector:
    """
    Random cluster position detector.

    When explicit cluster patterns cannot be detected (e.g., due to tokenizer
    differences), this detector randomly samples token positions as cluster points.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        tokenizer: PreTrainedTokenizer,
        num_clusters_per_label: int = 2,
        seed: Optional[int] = None,
        exclude_special_tokens: bool = True,
    ):
        """
        Initialize the random cluster detector.

        Args:
            task_config: Task configuration (used for num_classes)
            tokenizer: Tokenizer (used to identify special tokens)
            num_clusters_per_label: Number of random clusters to sample per label class
            seed: Random seed for reproducibility
            exclude_special_tokens: Whether to exclude special tokens from sampling
        """
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.num_clusters_per_label = num_clusters_per_label
        self.seed = seed
        self.exclude_special_tokens = exclude_special_tokens
        self.pad_token_id = tokenizer.pad_token_id

        # Get special token IDs to exclude
        self.special_token_ids = set()
        if exclude_special_tokens:
            if tokenizer.pad_token_id is not None:
                self.special_token_ids.add(tokenizer.pad_token_id)
            if tokenizer.eos_token_id is not None:
                self.special_token_ids.add(tokenizer.eos_token_id)
            if tokenizer.bos_token_id is not None:
                self.special_token_ids.add(tokenizer.bos_token_id)

    def detect(self, input_ids: Tensor) -> ClusterPositions:
        """
        Randomly sample cluster positions from the input sequence.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len] or [seq_len]

        Returns:
            ClusterPositions with randomly sampled positions
        """
        # Handle unbatched input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Find final position (last non-padding token)
        if self.pad_token_id is not None:
            final_pos = (input_ids != self.pad_token_id).int().sum(-1) - 1
        else:
            final_pos = torch.full((batch_size,), seq_len - 1, device=device)

        # Get valid positions (excluding special tokens and final position)
        valid_positions = self._get_valid_positions(input_ids, final_pos)

        # Randomly sample clusters for each label
        label_positions = {}
        num_labels = self.task_config.num_classes

        # Set seed if provided
        if self.seed is not None:
            torch.manual_seed(self.seed)

        total_clusters = num_labels * self.num_clusters_per_label
        if len(valid_positions) < total_clusters:
            # If not enough positions, use all valid positions
            sampled = valid_positions
        else:
            # Randomly sample without replacement
            perm = torch.randperm(len(valid_positions))[:total_clusters]
            sampled = valid_positions[perm]

        # Distribute clusters across labels
        for label_idx in range(num_labels):
            start = label_idx * self.num_clusters_per_label
            end = start + self.num_clusters_per_label
            if end <= len(sampled):
                label_positions[label_idx] = sampled[start:end].to(device)
            else:
                # Not enough samples, take what we can
                label_positions[label_idx] = sampled[start:].to(device) if start < len(sampled) else torch.tensor([], dtype=torch.long, device=device)

        return ClusterPositions(
            label_positions=label_positions,
            final_position=final_pos,
            sequence_length=seq_len,
        )

    def _get_valid_positions(self, input_ids: Tensor, final_pos: Tensor) -> Tensor:
        """Get positions that are valid for sampling (not special tokens, not final)."""
        batch_size, seq_len = input_ids.shape

        # For simplicity, handle batch_size=1 case
        if batch_size == 1:
            final_p = final_pos.item()
            valid = []
            for i in range(seq_len):
                token_id = input_ids[0, i].item()
                if token_id not in self.special_token_ids and i != final_p:
                    valid.append(i)
            return torch.tensor(valid, dtype=torch.long)
        else:
            # For batched, just use first item's pattern
            final_p = final_pos[0].item()
            valid = []
            for i in range(seq_len):
                token_id = input_ids[0, i].item()
                if token_id not in self.special_token_ids and i != final_p:
                    valid.append(i)
            return torch.tensor(valid, dtype=torch.long)

    def get_label_id_dict(self) -> Dict[int, int]:
        """Get mapping from label index to token ID (not applicable for random detector)."""
        return {}


# Backward compatibility aliases
AnchorDetector = ClusterDetector
AnchorPositions = ClusterPositions
RandomAnchorDetector = RandomClusterDetector
