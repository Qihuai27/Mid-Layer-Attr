"""
Evaluation module for text attribution methods.

This module provides AUC-based metrics for evaluating attribution quality
at both token and unit (word/sentence/chunk) granularity.

Metrics:
- Noise Insertion AUC: Measures score degradation when corrupting important tokens/units (necessity)
- Representation Insertion AUC: Measures hidden state recovery when restoring important tokens/units (sufficiency)

Usage:
    from src.eval import (
        # Token-level metrics
        noise_insertion_auc_token,
        representation_insertion_auc_token,
        compute_both_token_aucs,

        # Segmentation
        build_units,
        aggregate_token_scores_to_units,

        # Unit-level metrics
        noise_insertion_auc_units,
        representation_insertion_auc_units,
        compute_both_unit_aucs,
    )

    # Example: Token-level evaluation
    result = noise_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=saliency_scores,  # Token importance from attribution method
        target_pos=target_pos,
        target_token_id=target_token_id,
    )
    print(f"Noise Insertion AUC: {result.auc}")

    # Example: Unit-level evaluation
    seg_result = build_units(token_ids, tokenizer, mode="bpe_word")
    result = noise_insertion_auc_units(
        model=model,
        input_ids=input_ids,
        token_importance_scores=saliency_scores,
        units=seg_result.units,
        target_pos=target_pos,
        target_token_id=target_token_id,
    )
"""

# Token-level metrics
from .metrics_token import (
    # Main metrics
    noise_insertion_auc_token,
    representation_insertion_auc_token,
    compute_both_token_aucs,
    # Result class
    TokenAUCResult,
    # Helper functions (for advanced usage)
    to_device,
    trapezoidal_auc,
    score_from_logits,
    get_embeddings_from_input_ids,
    get_baseline_embedding,
    apply_noise_to_embedding,
)

# Segmentation
from .segmentation import (
    # Main functions
    build_units,
    aggregate_token_scores_to_units,
    # Result class
    SegmentationResult,
    # Individual builders
    build_units_token,
    build_units_fixed_length,
    build_units_bpe_word,
    build_units_sentence,
    build_units_ngram,
    build_units_custom,
    # Utilities
    get_unit_token_mapping,
    expand_unit_scores_to_tokens,
    validate_units,
    merge_small_units,
    split_large_units,
    # Type aliases
    Unit,
    Units,
)

# Unit-level metrics
from .metrics_units import (
    # Main metrics
    noise_insertion_auc_units,
    representation_insertion_auc_units,
    compute_both_unit_aucs,
    compare_segmentations,
    # Result class
    UnitAUCResult,
)

__all__ = [
    # Token-level metrics
    "noise_insertion_auc_token",
    "representation_insertion_auc_token",
    "compute_both_token_aucs",
    "TokenAUCResult",
    # Token helpers
    "to_device",
    "trapezoidal_auc",
    "score_from_logits",
    "get_embeddings_from_input_ids",
    "get_baseline_embedding",
    "apply_noise_to_embedding",
    # Segmentation
    "build_units",
    "aggregate_token_scores_to_units",
    "SegmentationResult",
    "build_units_token",
    "build_units_fixed_length",
    "build_units_bpe_word",
    "build_units_sentence",
    "build_units_ngram",
    "build_units_custom",
    "get_unit_token_mapping",
    "expand_unit_scores_to_tokens",
    "validate_units",
    "merge_small_units",
    "split_large_units",
    "Unit",
    "Units",
    # Unit-level metrics
    "noise_insertion_auc_units",
    "representation_insertion_auc_units",
    "compute_both_unit_aucs",
    "compare_segmentations",
    "UnitAUCResult",
]
