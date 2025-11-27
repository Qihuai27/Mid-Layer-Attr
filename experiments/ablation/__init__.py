"""
Causal ablation experiments.

This module implements causal ablation experiments for validating
the role of anchor words in ICL by masking attention patterns.
"""

from .interventions import (
    AttentionIntervention,
    MaskAnchorAttention,
    MaskNonAnchorAttention,
    MaskLayerAttention,
)
from .evaluator import AblationEvaluator, AblationResult
from .run import AblationExperiment

__all__ = [
    "AttentionIntervention",
    "MaskAnchorAttention",
    "MaskNonAnchorAttention",
    "MaskLayerAttention",
    "AblationEvaluator",
    "AblationResult",
    "AblationExperiment",
]
