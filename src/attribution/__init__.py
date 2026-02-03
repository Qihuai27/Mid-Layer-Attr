"""
Attribution methods for text generation models.

This module provides various attribution methods to compute token-level
importance scores for explaining model predictions.

Available Methods:
- AttentionRollout: Cumulative attention flow across layers
- AttentionRolloutLastLayer: Single-layer attention (faster)
- DePass: Decomposed forward pass attribution
- IntegratedGradients: Path-integrated gradients
- GradInput: Gradient × Input saliency
- GradSAM: Gradient-weighted Saliency Attention Maps
- TokenShapley: Token-level Shapley Value attribution
- MidLayerAttribution: Two-stage mid-layer attribution
- MidLayerAttributionV2: Mid-layer with clustering
- GreedyOptimalAttribution: Greedy optimal attribution
- InputCausalAttribution: Input causal attribution

Usage:
    from src.attribution import AttentionRollout, attention_rollout

    # Object-oriented interface
    method = AttentionRollout(residual_weight=0.5)
    result = method.attribute(model, input_ids, target_pos=-1)
    print(f"Top tokens: {result.get_top_k_indices(5)}")

    # Functional interface
    scores = attention_rollout(model, input_ids, target_pos=-1)

Integration with eval module:
    from src.attribution import attention_rollout
    from src.eval import noise_insertion_auc_token

    # Compute attribution
    scores = attention_rollout(model, input_ids, target_pos=-1)

    # Evaluate attribution quality
    auc_result = noise_insertion_auc_token(
        model=model,
        input_ids=input_ids,
        importance_scores=scores,
        target_pos=len(input_ids[0]) - 1,
        target_token_id=next_token_id,
    )
"""

# Base classes
from .base import (
    AttributionMethod,
    AttributionResult,
    AttentionBasedMethod,
    GradientBasedMethod,
    PerturbationBasedMethod,
)

# Attention-based methods
from .attention_rollout import (
    AttentionRollout,
    AttentionRolloutLastLayer,
    attention_rollout,
)

# Decomposed forward pass methods
from .depass import (
    DePass,
    DePassManager,
    depass,
)

# Gradient-based methods
from .integrated_gradients import (
    IntegratedGradients,
    integrated_gradients,
)

from .grad_input import (
    GradInput,
    grad_input,
)

from .grad_sam import (
    GradSAM,
    grad_sam,
)

# Perturbation-based methods
from .shapley import (
    TokenShapley,
    shapley,
)

# Mid-layer two-stage method
from .midlayer import (
    MidLayerAttribution,
    midlayer_attribution,
)

# Mid-layer V2 with clustering
from .midlayer_v2 import (
    MidLayerAttributionV2,
    midlayer_attribution_v2,
)

# Greedy optimal and input causal
from .greedy_optimal import (
    GreedyOptimalAttribution,
    InputCausalAttribution,
    greedy_optimal_attribution,
    input_causal_attribution,
)

__all__ = [
    # Base classes
    "AttributionMethod",
    "AttributionResult",
    "AttentionBasedMethod",
    "GradientBasedMethod",
    "PerturbationBasedMethod",
    # Attention methods
    "AttentionRollout",
    "AttentionRolloutLastLayer",
    "attention_rollout",
    # DePass methods
    "DePass",
    "DePassManager",
    "depass",
    # Gradient-based methods
    "IntegratedGradients",
    "integrated_gradients",
    "GradInput",
    "grad_input",
    "GradSAM",
    "grad_sam",
    # Perturbation-based methods
    "TokenShapley",
    "shapley",
    # Mid-layer two-stage method
    "MidLayerAttribution",
    "midlayer_attribution",
    # Mid-layer V2 with clustering
    "MidLayerAttributionV2",
    "midlayer_attribution_v2",
    # Greedy optimal and input causal
    "GreedyOptimalAttribution",
    "InputCausalAttribution",
    "greedy_optimal_attribution",
    "input_causal_attribution",
]
