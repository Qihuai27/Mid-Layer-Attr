"""
Token-level AUC metrics for text attribution evaluation.

This module implements two AUC metrics at token granularity:
1. noise_insertion_auc_token: Measures score degradation when adding noise to important tokens
2. representation_insertion_auc_token: Measures hidden state recovery when restoring important tokens

Design Rationale:
- Noise Insertion: Tests "necessity" - does removing this token's information hurt prediction?
- Representation Insertion: Tests "sufficiency" - does restoring this token recover the hidden states?
- Hidden states capture the full sequence's information aggregation, unlike next-token prediction
  which is dominated by local/positional effects in the fixed-length + PAD design.

Input Convention:
- importance_scores: Array of length T, where each value is the importance score for that token
- Higher scores indicate more important tokens
- The metrics internally sort by descending importance to determine intervention order

Distance Metrics:
- cosine: Cosine distance (1 - cosine_similarity), scale-invariant, default for hidden states
- frobenius: Frobenius norm, sensitive to magnitude
"""

from typing import Optional, Union, Literal, Tuple, List
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer


# =============================================================================
# Helper Functions
# =============================================================================

def to_device(
    tensor_or_array: Union[Tensor, np.ndarray],
    device: Union[str, torch.device]
) -> Tensor:
    """Move input to specified device, converting numpy arrays if needed."""
    if isinstance(tensor_or_array, np.ndarray):
        tensor_or_array = torch.from_numpy(tensor_or_array)
    return tensor_or_array.to(device)


def trapezoidal_auc(xs: Union[List[float], np.ndarray], ys: Union[List[float], np.ndarray]) -> float:
    """
    Compute AUC using trapezoidal rule.

    Args:
        xs: Ascending x-coordinates (e.g., fraction of tokens corrupted)
        ys: Corresponding y-values (e.g., normalized scores)

    Returns:
        AUC value computed as sum of trapezoids
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have same length, got {len(xs)} and {len(ys)}")
    if len(xs) < 2:
        return 0.0

    # Trapezoidal rule: sum of 0.5 * (y_i + y_{i+1}) * (x_{i+1} - x_i)
    auc = 0.0
    for i in range(len(xs) - 1):
        auc += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return float(auc)


def score_from_logits(
    logits: Tensor,
    target_pos: int,
    target_token_id: Optional[int] = None,
    mode: Literal["target_logit", "target_logprob", "max_logit"] = "target_logprob"
) -> float:
    """
    Extract a scalar score from logits at a specific position.

    Args:
        logits: Shape [1, T, V] or [T, V]
        target_pos: Position index to evaluate
        target_token_id: Ground-truth token id (required for target_* modes)
        mode: Scoring mode
            - "target_logit": Raw logit for target token
            - "target_logprob": Log probability for target token
            - "max_logit": Maximum logit at position

    Returns:
        Scalar score value
    """
    if logits.dim() == 3:
        logits = logits[0]  # [T, V]

    pos_logits = logits[target_pos]  # [V]

    if mode == "max_logit":
        return float(pos_logits.max().item())

    if target_token_id is None:
        raise ValueError(f"target_token_id required for mode={mode}")

    if mode == "target_logit":
        return float(pos_logits[target_token_id].item())

    if mode == "target_logprob":
        log_probs = torch.log_softmax(pos_logits, dim=-1)
        return float(log_probs[target_token_id].item())

    raise ValueError(f"Unknown mode: {mode}")


def get_embeddings_from_input_ids(
    model: PreTrainedModel,
    input_ids: Tensor
) -> Tensor:
    """
    Get token embeddings from input_ids.

    Args:
        model: HuggingFace causal LM with get_input_embeddings()
        input_ids: Shape [1, T] or [T]

    Returns:
        Embeddings tensor of shape [T, d]
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids)  # [1, T, d]
    return embeds[0]  # [T, d]


def get_baseline_embedding(
    base_embeds: Tensor,
    mode: Literal["mean", "zero", "gaussian"] = "mean",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model: Optional[PreTrainedModel] = None,
    special_token_id: Optional[int] = None
) -> Tensor:
    """
    Compute baseline embedding for noise injection.

    Args:
        base_embeds: Original embeddings [T, d]
        mode: Baseline mode
            - "mean": Mean of all token embeddings
            - "zero": Zero vector
            - "gaussian": Sample from N(0, 1) scaled by embedding std
        tokenizer: Optional tokenizer for special token mode
        model: Optional model for special token embedding
        special_token_id: Optional specific token id to use as baseline

    Returns:
        Baseline embedding [d]
    """
    d = base_embeds.shape[-1]
    device = base_embeds.device
    dtype = base_embeds.dtype

    if special_token_id is not None and model is not None:
        # Use specific token's embedding
        token_tensor = torch.tensor([[special_token_id]], device=device)
        return get_embeddings_from_input_ids(model, token_tensor)[0]

    if mode == "mean":
        return base_embeds.mean(dim=0)
    elif mode == "zero":
        return torch.zeros(d, device=device, dtype=dtype)
    elif mode == "gaussian":
        std = base_embeds.std()
        return torch.randn(d, device=device, dtype=dtype) * std
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")


def apply_noise_to_embedding(
    original_embed: Tensor,
    baseline_embed: Tensor,
    lambda_k: float,
    noise_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Apply noise/interpolation to a single token embedding.

    Formula: (1 - λ) * original + λ * baseline + σ * ε
    where ε ~ N(0, I)

    Args:
        original_embed: Original embedding [d]
        baseline_embed: Baseline embedding [d]
        lambda_k: Interpolation factor in [0, 1]
        noise_sigma: Gaussian noise standard deviation
        generator: Optional random generator for reproducibility

    Returns:
        Noised embedding [d]
    """
    # Convex combination
    noised = (1 - lambda_k) * original_embed + lambda_k * baseline_embed

    # Add Gaussian noise
    if noise_sigma > 0:
        noise = torch.randn_like(original_embed, generator=generator) * noise_sigma
        noised = noised + noise

    return noised


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class TokenAUCResult:
    """Result container for token-level AUC metrics."""
    auc: float
    curve_x: np.ndarray  # Fraction of tokens corrupted/restored
    curve_y: np.ndarray  # Normalized score at each step
    phi_max: float       # Score with original input
    phi_min: float       # Score with fully corrupted/baseline input

    def to_dict(self) -> dict:
        return {
            "auc": self.auc,
            "curve_x": self.curve_x.tolist(),
            "curve_y": self.curve_y.tolist(),
            "phi_max": self.phi_max,
            "phi_min": self.phi_min
        }


# =============================================================================
# Metric 1: Noise Insertion AUC (Token Level)
# =============================================================================

def _compute_noise_insertion_curve(
    model: PreTrainedModel,
    base_embeds: Tensor,
    baseline_embed: Tensor,
    logits_orig_target: Tensor,
    sorted_indices: List[int],
    steps: int,
    T: int,
    noise_sigma: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[float], List[float]]:
    """
    Helper function to compute noise insertion curve for a given token order.
    
    Returns:
        xs: x-coordinates (fraction of tokens corrupted)
        ys: y-coordinates (cosine similarity scores)
    """
    num_tokens_to_corrupt = len(sorted_indices)
    xs = []
    ys = []
    
    for k in range(steps + 1):
        m_k = int(round(num_tokens_to_corrupt * k / steps))
        
        with torch.no_grad():
            embeds_k = base_embeds.clone()
            for j in range(m_k):
                i = sorted_indices[j]
                embeds_k[i] = apply_noise_to_embedding(
                    base_embeds[i], baseline_embed,
                    lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
                )
            
            logits_k = model(inputs_embeds=embeds_k.unsqueeze(0)).logits
            logits_k_target = logits_k[0, -1]  # Use last position
        
        cos_sim = torch.nn.functional.cosine_similarity(
            logits_orig_target.unsqueeze(0),
            logits_k_target.unsqueeze(0)
        ).item()
        
        x_k = m_k / T if T > 0 else 0.0
        xs.append(x_k)
        ys.append(cos_sim)
    
    return xs, ys


def noise_insertion_auc_token(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    importance_scores: Union[Tensor, np.ndarray, List[float]],
    target_pos: int,
    target_token_id: Optional[int] = None,
    steps: int = 10,
    lambda_schedule: Optional[np.ndarray] = None,
    baseline_embed_mode: Literal["mean", "zero", "gaussian"] = "mean",
    noise_sigma: float = 0.0,
    score_mode: Literal["target_logit", "target_logprob", "max_logit", "logit_shift"] = "logit_shift",
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    exclude_positions: Optional[List[int]] = None,
    normalize_by_random: bool = True,
    n_random_samples: int = 3,
) -> TokenAUCResult:
    """
    Compute Noise Insertion AUC at token level.

    This metric measures how the prediction at target_pos changes as we
    progressively add noise to tokens in order of importance (most important first).

    Algorithm:
    1. Sort tokens by importance (descending)
    2. For each step k:
       - Corrupt the top-k most important tokens by mixing with baseline + noise
       - Compute logit shift (cosine similarity between corrupted and original logits)
       - Record normalized score
    3. Compute AUC under the degradation curve
    4. (Optional) Normalize by random baseline: (AUC - random_AUC) / (1 - random_AUC)

    Lower AUC indicates better attribution (important tokens identified correctly).

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids [1, T] or [T]
        importance_scores: Importance score for each token (length T)
        target_pos: Position to evaluate (usually last context token or first prediction)
        target_token_id: Ground-truth token id (only used if score_mode is target_*)
        steps: Number of discrete steps for AUC curve
        lambda_schedule: Custom interpolation schedule (default: linspace(0, 1, steps+1))
        baseline_embed_mode: How to compute baseline embedding
        noise_sigma: Gaussian noise std to add
        score_mode: How to compute score from logits
            - "logit_shift": Cosine similarity between corrupted and original logits (default)
            - "target_logit": Raw logit for target token
            - "target_logprob": Log probability for target token
            - "max_logit": Maximum logit at position
        device: Device to run on
        seed: Random seed for reproducibility
        exclude_positions: Token positions to exclude from corruption (e.g., special tokens)
        normalize_by_random: If True, normalize AUC relative to random baseline
        n_random_samples: Number of random orderings to average for baseline

    Returns:
        TokenAUCResult containing AUC and curve data
    """
    # Setup
    if device is None:
        device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Prepare input_ids
    input_ids = to_device(input_ids, device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    T = input_ids.shape[1]

    # Prepare importance scores
    importance_scores = np.asarray(importance_scores).flatten()
    if len(importance_scores) != T:
        raise ValueError(f"importance_scores length {len(importance_scores)} != T={T}")

    # Handle excluded positions by setting their importance to -inf
    if exclude_positions:
        importance_scores = importance_scores.copy()
        for pos in exclude_positions:
            if 0 <= pos < T:
                importance_scores[pos] = -np.inf

    # Get token embeddings
    with torch.no_grad():
        base_embeds = get_embeddings_from_input_ids(model, input_ids)  # [T, d]
        baseline_embed = get_baseline_embedding(base_embeds, mode=baseline_embed_mode)

    # Compute original logits
    with torch.no_grad():
        outputs_orig = model(inputs_embeds=base_embeds.unsqueeze(0))
        logits_orig = outputs_orig.logits
        logits_orig_target = logits_orig[0, target_pos]  # [V]

    # For phi_max and phi_min (reference values)
    phi_max = 1.0
    phi_min = 0.0

    # Compute fully corrupted logits for phi_min
    with torch.no_grad():
        corrupted_embeds = base_embeds.clone()
        for i in range(T):
            if exclude_positions and i in exclude_positions:
                continue
            corrupted_embeds[i] = apply_noise_to_embedding(
                base_embeds[i], baseline_embed,
                lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
            )
        logits_corrupted = model(inputs_embeds=corrupted_embeds.unsqueeze(0)).logits
        logits_corrupted_target = logits_corrupted[0, target_pos]
        cos_sim_min = torch.nn.functional.cosine_similarity(
            logits_orig_target.unsqueeze(0),
            logits_corrupted_target.unsqueeze(0)
        ).item()
        phi_min = cos_sim_min

    # Sort tokens by importance (descending)
    sorted_indices = np.argsort(-importance_scores)

    # Filter out excluded positions from sorted indices
    if exclude_positions:
        sorted_indices = [i for i in sorted_indices if i not in exclude_positions]

    num_tokens_to_corrupt = len(sorted_indices)

    # Build AUC curve for attribution-based ordering
    xs = []
    ys = []

    for k in range(steps + 1):
        m_k = int(round(num_tokens_to_corrupt * k / steps))

        with torch.no_grad():
            embeds_k = base_embeds.clone()
            for j in range(m_k):
                i = sorted_indices[j]
                embeds_k[i] = apply_noise_to_embedding(
                    base_embeds[i], baseline_embed,
                    lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
                )

            logits_k = model(inputs_embeds=embeds_k.unsqueeze(0)).logits
            logits_k_target = logits_k[0, target_pos]

        if score_mode == "logit_shift":
            cos_sim = torch.nn.functional.cosine_similarity(
                logits_orig_target.unsqueeze(0),
                logits_k_target.unsqueeze(0)
            ).item()
            r_k = cos_sim
        else:
            phi_k = score_from_logits(logits_k, target_pos, target_token_id, score_mode)
            phi_orig = score_from_logits(logits_orig, target_pos, target_token_id, score_mode)
            P_k = math.exp(phi_k) if phi_k > -700 else 0.0
            P_max = math.exp(phi_orig) if phi_orig > -700 else 0.0
            r_k = (P_k + 1) / (P_max + 1) - 0.5

        x_k = m_k / T if T > 0 else 0.0
        xs.append(x_k)
        ys.append(r_k)

    # Compute raw AUC
    raw_auc = trapezoidal_auc(xs, ys)

    # Compute random baseline AUC if requested
    if normalize_by_random and score_mode == "logit_shift":
        random_aucs = []
        available_indices = list(sorted_indices)  # Indices available for shuffling

        # Use fixed seeds for reproducibility across different attribution methods
        # This ensures fair comparison: same sample gets same random baselines
        random_baseline_seeds = [42, 43, 44]

        for sample_idx in range(n_random_samples):
            # Random permutation with fixed seed
            rng = np.random.RandomState(random_baseline_seeds[sample_idx])
            random_indices = available_indices.copy()
            rng.shuffle(random_indices)
            
            random_ys = []
            for k in range(steps + 1):
                m_k = int(round(num_tokens_to_corrupt * k / steps))
                
                with torch.no_grad():
                    embeds_k = base_embeds.clone()
                    for j in range(m_k):
                        i = random_indices[j]
                        embeds_k[i] = apply_noise_to_embedding(
                            base_embeds[i], baseline_embed,
                            lambda_k=1.0, noise_sigma=noise_sigma, generator=generator
                        )
                    
                    logits_k = model(inputs_embeds=embeds_k.unsqueeze(0)).logits
                    logits_k_target = logits_k[0, target_pos]
                
                cos_sim = torch.nn.functional.cosine_similarity(
                    logits_orig_target.unsqueeze(0),
                    logits_k_target.unsqueeze(0)
                ).item()
                random_ys.append(cos_sim)
            
            random_auc = trapezoidal_auc(xs, random_ys)
            random_aucs.append(random_auc)
        
        avg_random_auc = np.mean(random_aucs)
        
        # Normalize: (raw_auc - random_auc) / (1 - random_auc) + 1
        # Lower is better, so good attribution has raw_auc < random_auc
        # After normalization + 1: values around 1 = random baseline, < 1 = better than random
        if abs(1 - avg_random_auc) > 1e-10:
            normalized_auc = (raw_auc - avg_random_auc) / (1 - avg_random_auc) + 1
        else:
            normalized_auc = 1.0
        
        auc = normalized_auc
    else:
        auc = raw_auc

    return TokenAUCResult(
        auc=auc,
        curve_x=np.array(xs),
        curve_y=np.array(ys),
        phi_max=phi_max,
        phi_min=phi_min
    )


# =============================================================================
# Metric 2: Representation Insertion AUC (Token Level)
# =============================================================================

def representation_insertion_auc_token(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    importance_scores: Union[Tensor, np.ndarray, List[float]],
    base_token_id: int,
    steps: int = 10,
    layer_index: int = -1,
    distance_mode: Literal["frobenius", "cosine"] = "cosine",
    position_weighting: Literal["uniform", "linear"] = "linear",
    device: Optional[Union[str, torch.device]] = None,
    exclude_positions: Optional[List[int]] = None,
) -> TokenAUCResult:
    """
    Compute Representation Insertion AUC at token level.

    This metric measures how well the hidden representation recovers as we
    progressively restore tokens in order of importance (most important first).

    Algorithm:
    1. Compute original hidden states H_full
    2. Compute baseline hidden states H_base (all tokens replaced with base_token_id)
    3. Sort tokens by importance (descending)
    4. For each step k:
       - Restore the top-k most important tokens to original ids
       - Compute hidden states H_k
       - Compute recovery: R_k = 1 - ||H_full - H_k|| / ||H_full - H_base||
    5. Compute AUC under the recovery curve

    Higher AUC indicates better attribution (important tokens identified correctly).

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids [1, T] or [T]
        importance_scores: Importance score for each token (length T)
        base_token_id: Token id to use as baseline (e.g., PAD token)
        steps: Number of discrete steps for AUC curve
        layer_index: Which layer's hidden states to use (-1 for last layer)
        distance_mode: Distance metric for comparing hidden states
        position_weighting: How to weight positions when computing distance
            - "uniform": Equal weight for all positions
            - "linear": Linear increasing weight (later positions matter more for causal LM)
        device: Device to run on
        exclude_positions: Token positions to always keep as original (not replace with baseline)

    Returns:
        TokenAUCResult containing AUC and curve data
    """
    # Setup
    if device is None:
        device = next(model.parameters()).device

    # Prepare input_ids
    input_ids = to_device(input_ids, device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    T = input_ids.shape[1]

    # Prepare importance scores
    importance_scores = np.asarray(importance_scores).flatten()
    if len(importance_scores) != T:
        raise ValueError(f"importance_scores length {len(importance_scores)} != T={T}")

    # Handle excluded positions
    if exclude_positions:
        importance_scores = importance_scores.copy()
        for pos in exclude_positions:
            if 0 <= pos < T:
                importance_scores[pos] = -np.inf

    # Compute original hidden states H_full
    with torch.no_grad():
        outputs_full = model(input_ids, output_hidden_states=True)
        H_full = outputs_full.hidden_states[layer_index][0]  # [T, d]

    # Compute baseline hidden states H_base
    base_input_ids = torch.full_like(input_ids, base_token_id)
    # Keep excluded positions as original
    if exclude_positions:
        for pos in exclude_positions:
            if 0 <= pos < T:
                base_input_ids[0, pos] = input_ids[0, pos]

    with torch.no_grad():
        outputs_base = model(base_input_ids, output_hidden_states=True)
        H_base = outputs_base.hidden_states[layer_index][0]  # [T, d]

    # Compute position weights
    if position_weighting == "uniform":
        pos_weights = torch.ones(T, device=device, dtype=H_full.dtype)
    elif position_weighting == "linear":
        # Linear weights: w_i = (i + 1) / T, so last position has highest weight
        pos_weights = torch.arange(1, T + 1, device=device, dtype=H_full.dtype) / T
    else:
        raise ValueError(f"Unknown position_weighting: {position_weighting}")

    # Normalize weights to sum to 1
    pos_weights = pos_weights / pos_weights.sum()

    # Compute normalization denominator with position weighting
    if distance_mode == "frobenius":
        # Weighted Frobenius: sqrt(sum_i w_i * ||H_full[i] - H_base[i]||^2)
        diff = H_full - H_base  # [T, d]
        weighted_sq_norms = pos_weights * (diff ** 2).sum(dim=-1)  # [T]
        denom = torch.sqrt(weighted_sq_norms.sum()).item()
    elif distance_mode == "cosine":
        # Weighted average cosine distance across positions
        cos_sim = torch.nn.functional.cosine_similarity(H_full, H_base, dim=-1)  # [T]
        denom = (pos_weights * (1 - cos_sim)).sum().item()
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode}")

    if abs(denom) < 1e-10:
        denom = 1e-10

    # Sort tokens by importance (descending)
    sorted_indices = np.argsort(-importance_scores)

    # Filter out excluded positions
    if exclude_positions:
        sorted_indices = [i for i in sorted_indices if i not in exclude_positions]

    num_tokens_to_restore = len(sorted_indices)

    # For reference scores
    phi_max = 1.0  # Fully restored
    phi_min = 0.0  # Baseline

    # Build AUC curve
    xs = []
    ys = []

    for k in range(steps + 1):
        # Number of tokens to restore at this step
        m_k = int(round(num_tokens_to_restore * k / steps))

        # Create partially restored input
        with torch.no_grad():
            cur_input_ids = base_input_ids.clone()

            # Keep excluded positions as original
            if exclude_positions:
                for pos in exclude_positions:
                    if 0 <= pos < T:
                        cur_input_ids[0, pos] = input_ids[0, pos]

            # Restore top-m_k important tokens
            for j in range(m_k):
                i = sorted_indices[j]
                cur_input_ids[0, i] = input_ids[0, i]

            # Forward pass
            outputs_k = model(cur_input_ids, output_hidden_states=True)
            H_k = outputs_k.hidden_states[layer_index][0]

        # Compute recovery with position weighting
        if distance_mode == "frobenius":
            diff = H_full - H_k  # [T, d]
            weighted_sq_norms = pos_weights * (diff ** 2).sum(dim=-1)  # [T]
            dist = torch.sqrt(weighted_sq_norms.sum()).item()
        elif distance_mode == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(H_full, H_k, dim=-1)  # [T]
            dist = (pos_weights * (1 - cos_sim)).sum().item()

        R_k = 1 - dist / denom
        R_k = max(0.0, min(1.0, R_k))  # Clamp to [0, 1]

        # Record
        x_k = m_k / T if T > 0 else 0.0
        xs.append(x_k)
        ys.append(R_k)

    # Compute AUC
    auc = trapezoidal_auc(xs, ys)

    return TokenAUCResult(
        auc=auc,
        curve_x=np.array(xs),
        curve_y=np.array(ys),
        phi_max=phi_max,
        phi_min=phi_min
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_both_token_aucs(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    importance_scores: Union[Tensor, np.ndarray, List[float]],
    target_pos: int,
    target_token_id: Optional[int] = None,
    base_token_id: Optional[int] = None,
    steps: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> Tuple[TokenAUCResult, TokenAUCResult]:
    """
    Compute both token-level AUC metrics.

    Args:
        model: HuggingFace causal LM
        input_ids: Token ids
        importance_scores: Importance scores for each token
        target_pos: Target position for noise insertion AUC
        target_token_id: Target token id for noise insertion AUC
        base_token_id: Baseline token id for representation insertion AUC
        steps: Number of steps for both metrics
        device: Device to run on
        **kwargs: Additional arguments passed to both functions

    Returns:
        Tuple of (noise_insertion_result, representation_insertion_result)
    """
    noise_result = noise_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=importance_scores,
        target_pos=target_pos,
        target_token_id=target_token_id,
        steps=steps,
        device=device,
        **{k: v for k, v in kwargs.items() if k in [
            'lambda_schedule', 'baseline_embed_mode', 'noise_sigma',
            'score_mode', 'seed', 'exclude_positions'
        ]}
    )

    if base_token_id is None:
        raise ValueError("base_token_id required for representation insertion AUC")

    repr_result = representation_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=importance_scores,
        base_token_id=base_token_id,
        steps=steps,
        device=device,
        **{k: v for k, v in kwargs.items() if k in [
            'layer_index', 'distance_mode', 'position_weighting', 'exclude_positions'
        ]}
    )

    return noise_result, repr_result
