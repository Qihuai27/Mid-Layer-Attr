"""
Unit-level AUC metrics for text attribution evaluation.

This module implements two AUC metrics at unit granularity:
1. noise_insertion_auc_units: Measures score degradation when adding noise to important units
2. representation_insertion_auc_units: Measures hidden state recovery when restoring important units

Units are groups of tokens (e.g., words, sentences, chunks) defined by the segmentation module.

Input Convention:
- token_importance_scores: Array of length T, importance score for each token
- units: List of token index lists from segmentation module
- The metrics aggregate token scores to unit level, then sort units by importance
"""

from typing import Optional, Union, List, Literal
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedModel

from .metrics_token import (
    to_device,
    trapezoidal_auc,
    score_from_logits,
    get_embeddings_from_input_ids,
    get_baseline_embedding,
    apply_noise_to_embedding,
    TokenAUCResult,
)
from .segmentation import (
    Units,
    aggregate_token_scores_to_units,
)


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class UnitAUCResult:
    """Result container for unit-level AUC metrics."""
    auc: float
    curve_x: np.ndarray         # Fraction of tokens corrupted/restored
    curve_y: np.ndarray         # Normalized score at each step
    curve_x_units: np.ndarray   # Fraction of units corrupted/restored
    phi_max: float              # Score with original input
    phi_min: float              # Score with fully corrupted/baseline input
    num_units: int              # Total number of units
    num_tokens: int             # Total number of tokens
    unit_scores: np.ndarray     # Aggregated unit-level importance scores

    def to_dict(self) -> dict:
        return {
            "auc": self.auc,
            "curve_x": self.curve_x.tolist(),
            "curve_y": self.curve_y.tolist(),
            "curve_x_units": self.curve_x_units.tolist(),
            "phi_max": self.phi_max,
            "phi_min": self.phi_min,
            "num_units": self.num_units,
            "num_tokens": self.num_tokens,
            "unit_scores": self.unit_scores.tolist(),
        }


# =============================================================================
# Metric 1: Noise Insertion AUC (Unit Level)
# =============================================================================

def noise_insertion_auc_units(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    token_importance_scores: Union[Tensor, np.ndarray, List[float]],
    units: Units,
    target_pos: int,
    target_token_id: Optional[int] = None,
    agg_mode: Literal["sum", "mean", "max", "min", "l2"] = "sum",
    steps: int = 10,
    baseline_embed_mode: Literal["mean", "zero", "gaussian"] = "mean",
    noise_sigma: float = 0.0,
    score_mode: Literal["target_logit", "target_logprob", "max_logit"] = "target_logprob",
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    exclude_positions: Optional[List[int]] = None,
) -> UnitAUCResult:
    """
    Compute Noise Insertion AUC at unit level.

    This metric measures how the target position's score degrades as we
    progressively add noise to units in order of importance (most important first).
    All tokens in a unit are corrupted together.

    Algorithm:
    1. Aggregate token scores to unit scores
    2. Sort units by importance (descending)
    3. For each step k:
       - Corrupt the top-k most important units (all tokens in each unit)
       - Compute target position score
       - Record normalized score
    4. Compute AUC under the degradation curve

    Lower AUC indicates better attribution.

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids [1, T] or [T]
        token_importance_scores: Importance score for each token (length T)
        units: List of units from segmentation module
        target_pos: Position to evaluate
        target_token_id: Ground-truth token id for target position
        agg_mode: How to aggregate token scores to unit scores
        steps: Number of discrete steps for AUC curve
        baseline_embed_mode: How to compute baseline embedding
        noise_sigma: Gaussian noise std to add
        score_mode: How to compute score from logits
        device: Device to run on
        seed: Random seed for reproducibility
        exclude_positions: Token positions to exclude from corruption

    Returns:
        UnitAUCResult containing AUC and curve data
    """
    # Setup
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Prepare input_ids
    input_ids = to_device(input_ids, device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    T = input_ids.shape[1]

    # Prepare token importance scores
    token_importance_scores = np.asarray(token_importance_scores).flatten()
    if len(token_importance_scores) != T:
        raise ValueError(f"token_importance_scores length {len(token_importance_scores)} != T={T}")

    # Handle excluded positions
    exclude_set = set(exclude_positions) if exclude_positions else set()

    # Filter units to exclude those containing excluded positions
    filtered_units = []
    filtered_unit_indices = []
    for idx, unit in enumerate(units):
        # Check if any token in unit is excluded
        if not any(t in exclude_set for t in unit):
            filtered_units.append(unit)
            filtered_unit_indices.append(idx)

    # Aggregate to unit scores
    unit_scores_all = aggregate_token_scores_to_units(
        token_importance_scores, units, agg_mode=agg_mode
    )
    unit_scores = unit_scores_all[filtered_unit_indices]

    M = len(filtered_units)  # Number of units to process

    # Sort units by importance (descending)
    sorted_unit_indices = np.argsort(-unit_scores)

    # Get embeddings
    with torch.no_grad():
        base_embeds = get_embeddings_from_input_ids(model, input_ids)  # [T, d]
        baseline_embed = get_baseline_embedding(base_embeds, mode=baseline_embed_mode)

    # Compute original score (phi_max)
    with torch.no_grad():
        logits_orig = model(inputs_embeds=base_embeds.unsqueeze(0)).logits
        phi_max = score_from_logits(logits_orig, target_pos, target_token_id, score_mode)

    # Compute fully corrupted score (phi_min)
    with torch.no_grad():
        corrupted_embeds = base_embeds.clone()
        for i in range(T):
            if i not in exclude_set:
                corrupted_embeds[i] = apply_noise_to_embedding(
                    base_embeds[i], baseline_embed,
                    lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
                )
        logits_corrupted = model(inputs_embeds=corrupted_embeds.unsqueeze(0)).logits
        phi_min = score_from_logits(logits_corrupted, target_pos, target_token_id, score_mode)

    # Build AUC curve
    xs = []           # Fraction of tokens corrupted
    xs_units = []     # Fraction of units corrupted
    ys = []

    for k in range(steps + 1):
        # Number of units to corrupt at this step
        q_k = int(round(M * k / steps))

        # Create corrupted embeddings
        with torch.no_grad():
            embeds_k = base_embeds.clone()
            n_corrupted_tokens = 0

            # Corrupt top-q_k units
            for j in range(q_k):
                unit_idx = sorted_unit_indices[j]
                unit = filtered_units[unit_idx]
                for token_idx in unit:
                    embeds_k[token_idx] = apply_noise_to_embedding(
                        base_embeds[token_idx], baseline_embed,
                        lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
                    )
                    n_corrupted_tokens += 1

            # Forward pass
            logits_k = model(inputs_embeds=embeds_k.unsqueeze(0)).logits
            phi_k = score_from_logits(logits_k, target_pos, target_token_id, score_mode)

        # Normalize score
        denom = phi_max - phi_min
        if abs(denom) < 1e-10:
            r_k = 0.0
        else:
            r_k = (phi_k - phi_min) / denom

        # Record
        x_k = n_corrupted_tokens / T if T > 0 else 0.0
        x_k_units = q_k / M if M > 0 else 0.0
        xs.append(x_k)
        xs_units.append(x_k_units)
        ys.append(r_k)

    # Compute AUC (using token fraction as x-axis for consistency)
    auc = trapezoidal_auc(xs, ys)

    return UnitAUCResult(
        auc=auc,
        curve_x=np.array(xs),
        curve_y=np.array(ys),
        curve_x_units=np.array(xs_units),
        phi_max=phi_max,
        phi_min=phi_min,
        num_units=len(units),
        num_tokens=T,
        unit_scores=unit_scores_all,
    )


# =============================================================================
# Metric 2: Representation Insertion AUC (Unit Level)
# =============================================================================

def representation_insertion_auc_units(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    token_importance_scores: Union[Tensor, np.ndarray, List[float]],
    units: Units,
    base_token_id: int,
    agg_mode: Literal["sum", "mean", "max", "min", "l2"] = "sum",
    steps: int = 10,
    layer_index: int = -1,
    distance_mode: Literal["frobenius", "cosine"] = "cosine",
    device: Optional[Union[str, torch.device]] = None,
    exclude_positions: Optional[List[int]] = None,
) -> UnitAUCResult:
    """
    Compute Representation Insertion AUC at unit level.

    This metric measures how well the hidden representation recovers as we
    progressively restore units in order of importance (most important first).
    All tokens in a unit are restored together.

    Algorithm:
    1. Aggregate token scores to unit scores
    2. Sort units by importance (descending)
    3. For each step k:
       - Restore the top-k most important units (all tokens in each unit)
       - Compute hidden state distance from original
       - Record recovery ratio
    4. Compute AUC under the recovery curve

    Higher AUC indicates better attribution.

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids [1, T] or [T]
        token_importance_scores: Importance score for each token (length T)
        units: List of units from segmentation module
        base_token_id: Token id to use as baseline (e.g., PAD token)
        agg_mode: How to aggregate token scores to unit scores
        steps: Number of discrete steps for AUC curve
        layer_index: Which layer's hidden states to use (-1 for last layer)
        distance_mode: Distance metric for comparing hidden states
        device: Device to run on
        exclude_positions: Token positions to always keep as original

    Returns:
        UnitAUCResult containing AUC and curve data
    """
    # Setup
    if device is None:
        device = next(model.parameters()).device

    # Prepare input_ids
    input_ids = to_device(input_ids, device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    T = input_ids.shape[1]

    # Prepare token importance scores
    token_importance_scores = np.asarray(token_importance_scores).flatten()
    if len(token_importance_scores) != T:
        raise ValueError(f"token_importance_scores length {len(token_importance_scores)} != T={T}")

    # Handle excluded positions
    exclude_set = set(exclude_positions) if exclude_positions else set()

    # Filter units
    filtered_units = []
    filtered_unit_indices = []
    for idx, unit in enumerate(units):
        if not any(t in exclude_set for t in unit):
            filtered_units.append(unit)
            filtered_unit_indices.append(idx)

    # Aggregate to unit scores
    unit_scores_all = aggregate_token_scores_to_units(
        token_importance_scores, units, agg_mode=agg_mode
    )
    unit_scores = unit_scores_all[filtered_unit_indices]

    M = len(filtered_units)

    # Sort units by importance (descending)
    sorted_unit_indices = np.argsort(-unit_scores)

    # Compute original hidden states H_full
    with torch.no_grad():
        outputs_full = model(input_ids, output_hidden_states=True)
        H_full = outputs_full.hidden_states[layer_index][0]  # [T, d]

    # Compute baseline hidden states H_base
    base_input_ids = torch.full_like(input_ids, base_token_id)
    # Keep excluded positions as original
    for pos in exclude_set:
        if 0 <= pos < T:
            base_input_ids[0, pos] = input_ids[0, pos]

    with torch.no_grad():
        outputs_base = model(base_input_ids, output_hidden_states=True)
        H_base = outputs_base.hidden_states[layer_index][0]  # [T, d]

    # Compute normalization denominator
    if distance_mode == "frobenius":
        denom = torch.norm(H_full - H_base, p='fro').item()
    elif distance_mode == "cosine":
        cos_sim = torch.nn.functional.cosine_similarity(H_full, H_base, dim=-1)
        denom = (1 - cos_sim.mean()).item()
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode}")

    if abs(denom) < 1e-10:
        denom = 1e-10

    phi_max = 1.0
    phi_min = 0.0

    # Build AUC curve
    xs = []
    xs_units = []
    ys = []

    for k in range(steps + 1):
        # Number of units to restore at this step
        q_k = int(round(M * k / steps))

        with torch.no_grad():
            cur_input_ids = base_input_ids.clone()
            n_restored_tokens = 0

            # Keep excluded positions
            for pos in exclude_set:
                if 0 <= pos < T:
                    cur_input_ids[0, pos] = input_ids[0, pos]

            # Restore top-q_k units
            for j in range(q_k):
                unit_idx = sorted_unit_indices[j]
                unit = filtered_units[unit_idx]
                for token_idx in unit:
                    cur_input_ids[0, token_idx] = input_ids[0, token_idx]
                    n_restored_tokens += 1

            # Forward pass
            outputs_k = model(cur_input_ids, output_hidden_states=True)
            H_k = outputs_k.hidden_states[layer_index][0]

        # Compute recovery
        if distance_mode == "frobenius":
            dist = torch.norm(H_full - H_k, p='fro').item()
        elif distance_mode == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(H_full, H_k, dim=-1)
            dist = (1 - cos_sim.mean()).item()

        R_k = 1 - dist / denom
        R_k = max(0.0, min(1.0, R_k))

        # Record
        x_k = n_restored_tokens / T if T > 0 else 0.0
        x_k_units = q_k / M if M > 0 else 0.0
        xs.append(x_k)
        xs_units.append(x_k_units)
        ys.append(R_k)

    # Compute AUC
    auc = trapezoidal_auc(xs, ys)

    return UnitAUCResult(
        auc=auc,
        curve_x=np.array(xs),
        curve_y=np.array(ys),
        curve_x_units=np.array(xs_units),
        phi_max=phi_max,
        phi_min=phi_min,
        num_units=len(units),
        num_tokens=T,
        unit_scores=unit_scores_all,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_both_unit_aucs(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    token_importance_scores: Union[Tensor, np.ndarray, List[float]],
    units: Units,
    target_pos: int,
    target_token_id: Optional[int] = None,
    base_token_id: Optional[int] = None,
    agg_mode: Literal["sum", "mean", "max", "min", "l2"] = "sum",
    steps: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> tuple:
    """
    Compute both unit-level AUC metrics.

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids
        token_importance_scores: Importance scores for each token
        units: List of units from segmentation
        target_pos: Target position for noise insertion AUC
        target_token_id: Target token id for noise insertion AUC
        base_token_id: Baseline token id for representation insertion AUC
        agg_mode: How to aggregate token scores to unit scores
        steps: Number of steps for both metrics
        device: Device to run on
        **kwargs: Additional arguments

    Returns:
        Tuple of (noise_insertion_result, representation_insertion_result)
    """
    noise_result = noise_insertion_auc_units(
        model=model,
        input_ids=input_ids,
        token_importance_scores=token_importance_scores,
        units=units,
        target_pos=target_pos,
        target_token_id=target_token_id,
        agg_mode=agg_mode,
        steps=steps,
        device=device,
        **{k: v for k, v in kwargs.items() if k in [
            'baseline_embed_mode', 'noise_sigma', 'score_mode', 'seed', 'exclude_positions'
        ]}
    )

    if base_token_id is None:
        raise ValueError("base_token_id required for representation insertion AUC")

    repr_result = representation_insertion_auc_units(
        model=model,
        input_ids=input_ids,
        token_importance_scores=token_importance_scores,
        units=units,
        base_token_id=base_token_id,
        agg_mode=agg_mode,
        steps=steps,
        device=device,
        **{k: v for k, v in kwargs.items() if k in [
            'layer_index', 'distance_mode', 'exclude_positions'
        ]}
    )

    return noise_result, repr_result


def compare_segmentations(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    token_importance_scores: Union[Tensor, np.ndarray, List[float]],
    units_list: List[Units],
    unit_names: List[str],
    target_pos: int,
    target_token_id: Optional[int] = None,
    base_token_id: Optional[int] = None,
    **kwargs
) -> dict:
    """
    Compare different segmentations on the same input.

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids
        token_importance_scores: Token-level importance scores
        units_list: List of different segmentations to compare
        unit_names: Names for each segmentation
        target_pos: Target position
        target_token_id: Target token id
        base_token_id: Baseline token id
        **kwargs: Additional arguments for AUC functions

    Returns:
        Dict with results for each segmentation
    """
    results = {}

    for units, name in zip(units_list, unit_names):
        noise_result, repr_result = compute_both_unit_aucs(
            model=model,
            input_ids=input_ids,
            token_importance_scores=token_importance_scores,
            units=units,
            target_pos=target_pos,
            target_token_id=target_token_id,
            base_token_id=base_token_id,
            **kwargs
        )
        results[name] = {
            "noise_insertion_auc": noise_result.auc,
            "representation_insertion_auc": repr_result.auc,
            "num_units": len(units),
            "noise_result": noise_result,
            "repr_result": repr_result,
        }

    return results
