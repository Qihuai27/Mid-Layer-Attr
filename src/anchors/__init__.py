"""
Backward compatibility module - deprecated, use src.cluster instead.

This module provides aliases for the old "anchors" terminology:
- AnchorDetector -> ClusterDetector
- AnchorPositions -> ClusterPositions
- AnchorStatistics -> ClusterStatistics
- S_wp -> S_a (aggregation)
- S_pq -> S_o (output)
- S_ww -> S_w (within)
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "src.anchors is deprecated, use src.cluster instead. "
    "Terminology mapping: Anchor -> Cluster, S_wp -> S_a, S_pq -> S_o, S_ww -> S_w",
    DeprecationWarning,
    stacklevel=2
)

# Import from cluster module and provide aliases
from src.cluster import (
    ClusterDetector as AnchorDetector,
    ClusterPositions as AnchorPositions,
    RandomClusterDetector,
    TaskConfig,
    load_task_config,
    get_available_tasks,
    AttentionExtractor,
    AttentionBipartiteGraph,
    FlowMetric,
    InformationFlowComputer,
    AttentionSumFlow,
    GradientSaliencyFlow,
    AttentionRolloutFlow,
    AttentionValueWeightedFlow,
    CausalAblationFlow,
    AttentionMaskAblationFlow,
    get_flow_computer,
    ClusterStatistics as AnchorStatistics,
    compute_cluster_statistics as compute_anchor_statistics,
    compute_all_layer_statistics,
    aggregate_statistics,
)

__all__ = [
    # Detector (with old names)
    "AnchorDetector",
    "AnchorPositions",
    "RandomClusterDetector",
    # Task config
    "TaskConfig",
    "load_task_config",
    "get_available_tasks",
    # Extractor
    "AttentionExtractor",
    "AttentionBipartiteGraph",
    # Flow metrics
    "FlowMetric",
    "InformationFlowComputer",
    "AttentionSumFlow",
    "GradientSaliencyFlow",
    "AttentionRolloutFlow",
    "AttentionValueWeightedFlow",
    "CausalAblationFlow",
    "AttentionMaskAblationFlow",
    "get_flow_computer",
    # Statistics (with old names)
    "AnchorStatistics",
    "compute_anchor_statistics",
    "compute_all_layer_statistics",
    "aggregate_statistics",
]
