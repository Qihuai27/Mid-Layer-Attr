"""
Attribution analysis experiments.

This module implements attention attribution analysis for studying
how anchor words affect information flow in transformers.
"""

from .extractor import AttentionExtractor, AttentionBipartiteGraph
from .statistics import (
    # Core classes
    AnchorStatistics,
    FlowMetric,
    InformationFlowComputer,
    # Basic metrics
    AttentionSumFlow,
    RawAttentionFlow,  # Alias for AttentionSumFlow
    GradientSaliencyFlow,
    # Advanced metrics
    AttentionRolloutFlow,
    AttentionValueWeightedFlow,
    # Ablation-based metrics
    CausalAblationFlow,
    AttentionMaskAblationFlow,
    # Factory and computation functions
    get_flow_computer,
    compute_anchor_statistics,
    compute_all_layer_statistics,
    aggregate_statistics,
)
from .run import AttributionExperiment

__all__ = [
    # Extractors
    "AttentionExtractor",
    "AttentionBipartiteGraph",
    # Statistics
    "AnchorStatistics",
    # Metrics enum and base class
    "FlowMetric",
    "InformationFlowComputer",
    # Basic flow computers
    "AttentionSumFlow",
    "RawAttentionFlow",
    "GradientSaliencyFlow",
    # Advanced flow computers
    "AttentionRolloutFlow",
    "AttentionValueWeightedFlow",
    # Ablation-based flow computers
    "CausalAblationFlow",
    "AttentionMaskAblationFlow",
    # Functions
    "get_flow_computer",
    "compute_anchor_statistics",
    "compute_all_layer_statistics",
    "aggregate_statistics",
    # Experiment runner
    "AttributionExperiment",
]
