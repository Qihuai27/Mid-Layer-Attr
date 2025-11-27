"""
Unified anchor word position detection.

This module provides a single, unified implementation for detecting anchor word
positions in tokenized sequences, replacing the duplicated logic that was
scattered across multiple files in the original codebase.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from .definitions import TaskConfig


@dataclass
class AnchorPositions:
    """
    Container for detected anchor word positions in a sequence.

    Attributes:
        label_positions: Dict mapping label index to tensor of positions
                        where that label's anchor word appears
        final_position: Position of the final (prediction) token
        sequence_length: Total length of the sequence
    """
    label_positions: Dict[int, Tensor]  # {0: tensor([pos1, pos2]), 1: tensor([pos3, pos4])}
    final_position: Tensor  # tensor([final_pos]) or tensor([fp1, fp2, ...]) for batch
    sequence_length: int

    @property
    def all_anchor_positions(self) -> Tensor:
        """Get all anchor positions as a single tensor."""
        all_pos = []
        for positions in self.label_positions.values():
            all_pos.append(positions)
        if all_pos:
            return torch.cat(all_pos, dim=-1)
        return torch.tensor([], dtype=torch.long)

    @property
    def num_anchors(self) -> int:
        """Total number of anchor positions detected."""
        return sum(pos.numel() for pos in self.label_positions.values())

    def get_non_anchor_positions(self, exclude_final: bool = True) -> Tensor:
        """
        Get positions that are NOT anchor words.

        Args:
            exclude_final: Also exclude the final position

        Returns:
            Tensor of non-anchor positions
        """
        all_positions = set(range(self.sequence_length))
        anchor_positions = set(self.all_anchor_positions.tolist())

        if exclude_final:
            final_pos = self.final_position.item() if self.final_position.numel() == 1 else self.final_position[0].item()
            anchor_positions.add(final_pos)

        non_anchor = sorted(all_positions - anchor_positions)
        return torch.tensor(non_anchor, dtype=torch.long)


class AnchorDetector:
    """
    Unified anchor word position detector.

    This class encapsulates the logic for finding anchor word positions
    in tokenized ICL sequences. It uses a bigram matching strategy:
    prefix tokens + label token form a unique pattern that identifies
    anchor positions.

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
        Initialize the anchor detector.

        Args:
            task_config: Task configuration containing labels and anchor patterns
            tokenizer: Tokenizer for encoding prefix/label tokens
        """
        self.task_config = task_config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        # Encode prefix tokens
        self.prefix_token_ids = self._encode_prefix(task_config.anchor_prefix)

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

    def detect(self, input_ids: Tensor) -> AnchorPositions:
        """
        Detect anchor word positions in the input sequence.

        This uses a clever encoding trick to find bigram/trigram patterns:
        We encode position info as: id + prev_id * 100000 + prev_prev_id * 100000^2
        This creates a unique identifier for each position based on its context.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len] or [seq_len]

        Returns:
            AnchorPositions containing detected positions
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
            positions = self._find_anchor_pattern(
                input_ids, label_token_id, self.prefix_token_ids, device
            )
            label_positions[label_idx] = positions

        return AnchorPositions(
            label_positions=label_positions,
            final_position=final_pos,
            sequence_length=seq_len,
        )

    def _find_anchor_pattern(
        self,
        input_ids: Tensor,
        label_token_id: int,
        prefix_token_ids: List[int],
        device: torch.device,
    ) -> Tensor:
        """
        Find positions matching the anchor pattern (prefix + label).

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
