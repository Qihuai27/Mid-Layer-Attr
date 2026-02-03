"""
Text/Token segmentation module for unit-level attribution evaluation.

This module provides:
1. Unit building: Convert tokens into groups (units) for coarser-grained evaluation
2. Score aggregation: Aggregate token-level importance scores to unit-level

Input Convention:
- importance_scores: Array of length T, where each value is the importance score for that token
- Units are groups of consecutive or related token indices
"""

from typing import Optional, Union, List, Callable, Literal
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizer


# =============================================================================
# Type Definitions
# =============================================================================

# A unit is a list of token indices that belong together
Unit = List[int]
Units = List[Unit]


@dataclass
class SegmentationResult:
    """Result container for segmentation."""
    units: Units
    unit_texts: Optional[List[str]] = None  # Decoded text for each unit
    mode: str = "unknown"

    def __len__(self) -> int:
        return len(self.units)

    @property
    def num_units(self) -> int:
        return len(self.units)

    @property
    def num_tokens(self) -> int:
        return sum(len(u) for u in self.units)

    def get_unit_sizes(self) -> List[int]:
        """Get the size (number of tokens) of each unit."""
        return [len(u) for u in self.units]


# =============================================================================
# Unit Building Functions
# =============================================================================

def build_units_token(num_tokens: int) -> Units:
    """
    Build units where each token is its own unit.

    Args:
        num_tokens: Total number of tokens T

    Returns:
        Units: [[0], [1], ..., [T-1]]
    """
    return [[i] for i in range(num_tokens)]


def build_units_fixed_length(num_tokens: int, chunk_size: int) -> Units:
    """
    Build units of fixed size.

    Args:
        num_tokens: Total number of tokens T
        chunk_size: Number of tokens per unit

    Returns:
        Units: [[0, 1, ..., k-1], [k, ..., 2k-1], ...]
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    units = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        units.append(list(range(start, end)))
    return units


def build_units_bpe_word(
    token_ids: Union[List[int], np.ndarray],
    tokenizer: PreTrainedTokenizer,
) -> Units:
    """
    Build units based on BPE word boundaries.

    Groups subword tokens that belong to the same word into one unit.
    Uses the convention that continuation tokens often start with specific
    prefixes (e.g., "##" for BERT, "Ġ" for GPT-2, "▁" for SentencePiece).

    Args:
        token_ids: Token ids array
        tokenizer: HuggingFace tokenizer

    Returns:
        Units where each unit is a word (group of subword tokens)
    """
    token_ids = list(token_ids)
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return []

    # Decode each token to check for word boundary
    units = []
    current_unit = [0]

    for i in range(1, num_tokens):
        # Decode the token
        token_str = tokenizer.decode([token_ids[i]])

        # Check if this is a word start
        # Different tokenizers use different conventions:
        # - GPT-2/RoBERTa: word-start tokens begin with "Ġ" (space)
        # - SentencePiece: word-start tokens begin with "▁"
        # - BERT: continuation tokens start with "##"

        is_word_start = False

        # GPT-2 style: starts with space
        if token_str.startswith(" ") or token_str.startswith("Ġ"):
            is_word_start = True
        # SentencePiece style: starts with ▁
        elif token_str.startswith("▁"):
            is_word_start = True
        # Check if previous token ends a word (heuristic)
        elif i > 0:
            prev_decoded = tokenizer.decode([token_ids[i-1]])
            if prev_decoded.endswith(" ") or prev_decoded.endswith("\n"):
                is_word_start = True

        # Special tokens are always their own unit
        if token_ids[i] in tokenizer.all_special_ids:
            if current_unit:
                units.append(current_unit)
            units.append([i])
            current_unit = []
        elif is_word_start:
            if current_unit:
                units.append(current_unit)
            current_unit = [i]
        else:
            current_unit.append(i)

    # Don't forget the last unit
    if current_unit:
        units.append(current_unit)

    return units


def build_units_sentence(
    token_ids: Union[List[int], np.ndarray],
    tokenizer: PreTrainedTokenizer,
    text: Optional[str] = None,
    sentence_delimiters: str = ".!?\n"
) -> Units:
    """
    Build units based on sentence boundaries.

    Args:
        token_ids: Token ids array
        tokenizer: HuggingFace tokenizer
        text: Original text (optional, for better sentence detection)
        sentence_delimiters: Characters that end sentences

    Returns:
        Units where each unit is a sentence
    """
    token_ids = list(token_ids)
    num_tokens = len(token_ids)

    if num_tokens == 0:
        return []

    # Decode all tokens
    decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    units = []
    current_unit = []

    for i in range(num_tokens):
        current_unit.append(i)

        # Check if this token ends a sentence
        token_text = decoded_tokens[i]
        is_sentence_end = False

        for delim in sentence_delimiters:
            if delim in token_text:
                # Check it's at the end of the token (not middle of abbreviation)
                stripped = token_text.rstrip()
                if stripped.endswith(delim):
                    is_sentence_end = True
                    break

        if is_sentence_end and current_unit:
            units.append(current_unit)
            current_unit = []

    # Don't forget remaining tokens
    if current_unit:
        units.append(current_unit)

    return units


def build_units_ngram(num_tokens: int, n: int, stride: int = 1) -> Units:
    """
    Build overlapping n-gram units.

    Args:
        num_tokens: Total number of tokens
        n: Size of each n-gram
        stride: Step between consecutive n-grams

    Returns:
        Units: Overlapping n-grams
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    units = []
    for start in range(0, num_tokens - n + 1, stride):
        units.append(list(range(start, start + n)))

    # Handle remaining tokens if any
    if units and units[-1][-1] < num_tokens - 1:
        last_start = units[-1][0] + stride
        if last_start < num_tokens:
            units.append(list(range(last_start, num_tokens)))

    return units


def build_units_custom(
    num_tokens: int,
    boundaries: List[int]
) -> Units:
    """
    Build units from custom boundaries.

    Args:
        num_tokens: Total number of tokens
        boundaries: List of unit start indices (must include 0)

    Returns:
        Units defined by the boundaries
    """
    if not boundaries:
        return [list(range(num_tokens))]

    boundaries = sorted(set(boundaries))
    if boundaries[0] != 0:
        boundaries = [0] + boundaries

    units = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else num_tokens
        if start < end:
            units.append(list(range(start, end)))

    return units


def build_units(
    token_ids: Union[List[int], np.ndarray],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    text: Optional[str] = None,
    mode: Literal["token", "bpe_word", "sentence", "fixed_length", "ngram", "custom"] = "token",
    **kwargs
) -> SegmentationResult:
    """
    Main interface for building units from tokens.

    Args:
        token_ids: Token ids array of length T
        tokenizer: HuggingFace tokenizer (required for some modes)
        text: Original text (optional, for sentence mode)
        mode: Segmentation mode
            - "token": Each token is a unit
            - "bpe_word": Group subwords into words
            - "sentence": Group tokens into sentences
            - "fixed_length": Fixed-size chunks (requires chunk_size kwarg)
            - "ngram": Overlapping n-grams (requires n kwarg, optional stride)
            - "custom": Custom boundaries (requires boundaries kwarg)
        **kwargs: Mode-specific arguments
            - chunk_size: For "fixed_length" mode
            - n, stride: For "ngram" mode
            - boundaries: For "custom" mode
            - sentence_delimiters: For "sentence" mode

    Returns:
        SegmentationResult with units and metadata
    """
    token_ids = list(token_ids) if not isinstance(token_ids, list) else token_ids
    num_tokens = len(token_ids)

    if mode == "token":
        units = build_units_token(num_tokens)

    elif mode == "fixed_length":
        chunk_size = kwargs.get("chunk_size", 4)
        units = build_units_fixed_length(num_tokens, chunk_size)

    elif mode == "bpe_word":
        if tokenizer is None:
            raise ValueError("tokenizer required for bpe_word mode")
        units = build_units_bpe_word(token_ids, tokenizer)

    elif mode == "sentence":
        if tokenizer is None:
            raise ValueError("tokenizer required for sentence mode")
        sentence_delimiters = kwargs.get("sentence_delimiters", ".!?\n")
        units = build_units_sentence(token_ids, tokenizer, text, sentence_delimiters)

    elif mode == "ngram":
        n = kwargs.get("n", 3)
        stride = kwargs.get("stride", 1)
        units = build_units_ngram(num_tokens, n, stride)

    elif mode == "custom":
        boundaries = kwargs.get("boundaries", [0])
        units = build_units_custom(num_tokens, boundaries)

    else:
        raise ValueError(f"Unknown segmentation mode: {mode}")

    # Optionally decode unit texts
    unit_texts = None
    if tokenizer is not None:
        unit_texts = []
        for unit in units:
            unit_token_ids = [token_ids[i] for i in unit]
            unit_texts.append(tokenizer.decode(unit_token_ids))

    return SegmentationResult(units=units, unit_texts=unit_texts, mode=mode)


# =============================================================================
# Score Aggregation Functions
# =============================================================================

def aggregate_scores_sum(scores: np.ndarray) -> float:
    """Aggregate by sum."""
    return float(np.sum(scores))


def aggregate_scores_mean(scores: np.ndarray) -> float:
    """Aggregate by mean."""
    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def aggregate_scores_max(scores: np.ndarray) -> float:
    """Aggregate by max."""
    return float(np.max(scores)) if len(scores) > 0 else 0.0


def aggregate_scores_min(scores: np.ndarray) -> float:
    """Aggregate by min."""
    return float(np.min(scores)) if len(scores) > 0 else 0.0


def aggregate_scores_l2(scores: np.ndarray) -> float:
    """Aggregate by L2 norm."""
    return float(np.linalg.norm(scores))


AGGREGATION_FUNCTIONS = {
    "sum": aggregate_scores_sum,
    "mean": aggregate_scores_mean,
    "max": aggregate_scores_max,
    "min": aggregate_scores_min,
    "l2": aggregate_scores_l2,
}


def aggregate_token_scores_to_units(
    token_scores: Union[np.ndarray, List[float]],
    units: Units,
    agg_mode: Literal["sum", "mean", "max", "min", "l2"] = "sum",
    custom_agg_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> np.ndarray:
    """
    Aggregate token-level importance scores to unit-level scores.

    Args:
        token_scores: Importance scores for each token (length T)
        units: List of units, where each unit is a list of token indices
        agg_mode: Aggregation mode
            - "sum": Sum of token scores in unit
            - "mean": Mean of token scores in unit
            - "max": Maximum token score in unit
            - "min": Minimum token score in unit
            - "l2": L2 norm of token scores in unit
        custom_agg_fn: Optional custom aggregation function

    Returns:
        Unit-level scores array (length M = number of units)
    """
    token_scores = np.asarray(token_scores).flatten()

    if custom_agg_fn is not None:
        agg_fn = custom_agg_fn
    elif agg_mode in AGGREGATION_FUNCTIONS:
        agg_fn = AGGREGATION_FUNCTIONS[agg_mode]
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    unit_scores = []
    for unit in units:
        if len(unit) == 0:
            unit_scores.append(0.0)
        else:
            scores_in_unit = token_scores[unit]
            unit_scores.append(agg_fn(scores_in_unit))

    return np.array(unit_scores)


def get_unit_token_mapping(units: Units) -> dict:
    """
    Create a mapping from token index to unit index.

    Args:
        units: List of units

    Returns:
        Dict mapping token_idx -> unit_idx
    """
    token_to_unit = {}
    for unit_idx, unit in enumerate(units):
        for token_idx in unit:
            token_to_unit[token_idx] = unit_idx
    return token_to_unit


def expand_unit_scores_to_tokens(
    unit_scores: np.ndarray,
    units: Units,
    num_tokens: int,
    default_value: float = 0.0
) -> np.ndarray:
    """
    Expand unit-level scores back to token-level.

    Each token gets the score of its unit.

    Args:
        unit_scores: Scores for each unit (length M)
        units: List of units
        num_tokens: Total number of tokens T
        default_value: Score for tokens not in any unit

    Returns:
        Token-level scores (length T)
    """
    token_scores = np.full(num_tokens, default_value)
    for unit_idx, unit in enumerate(units):
        for token_idx in unit:
            if 0 <= token_idx < num_tokens:
                token_scores[token_idx] = unit_scores[unit_idx]
    return token_scores


# =============================================================================
# Utility Functions
# =============================================================================

def validate_units(units: Units, num_tokens: int) -> bool:
    """
    Validate that units cover all tokens exactly once.

    Args:
        units: List of units
        num_tokens: Expected total number of tokens

    Returns:
        True if valid, raises ValueError otherwise
    """
    seen = set()
    for unit in units:
        for idx in unit:
            if idx < 0 or idx >= num_tokens:
                raise ValueError(f"Token index {idx} out of range [0, {num_tokens})")
            if idx in seen:
                raise ValueError(f"Token index {idx} appears in multiple units")
            seen.add(idx)

    if len(seen) != num_tokens:
        missing = set(range(num_tokens)) - seen
        raise ValueError(f"Tokens not covered by any unit: {missing}")

    return True


def merge_small_units(
    units: Units,
    min_size: int,
    merge_with: Literal["previous", "next"] = "next"
) -> Units:
    """
    Merge units smaller than min_size with adjacent units.

    Args:
        units: Original units
        min_size: Minimum unit size
        merge_with: Direction to merge ("previous" or "next")

    Returns:
        New units with small units merged
    """
    if not units:
        return units

    result = []
    buffer = []

    for unit in units:
        buffer.extend(unit)
        if len(buffer) >= min_size:
            result.append(buffer)
            buffer = []

    # Handle remaining buffer
    if buffer:
        if result and merge_with == "previous":
            result[-1].extend(buffer)
        else:
            result.append(buffer)

    return result


def split_large_units(units: Units, max_size: int) -> Units:
    """
    Split units larger than max_size into smaller units.

    Args:
        units: Original units
        max_size: Maximum unit size

    Returns:
        New units with large units split
    """
    result = []
    for unit in units:
        if len(unit) <= max_size:
            result.append(unit)
        else:
            for i in range(0, len(unit), max_size):
                result.append(unit[i:i + max_size])
    return result
