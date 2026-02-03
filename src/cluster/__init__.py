"""
Cluster (信息汇聚) module for studying early-layer aggregation in LLMs.

This module investigates the "Label Words are Anchors" phenomenon:
- How cluster words (label words like "Positive"/"Negative") aggregate information
- Information flow patterns: S_a (text→cluster), S_o (cluster→final), S_w (baseline)
- Layer-wise statistics showing aggregation effects

Key statistics:
- S_a (aggregation): Information flow from previous tokens TO cluster positions
- S_o (output): Information flow from cluster positions TO final position
- S_w (within): Information flow between other (non-cluster, non-final) positions

Key components:
- ClusterDetector: Detects cluster point positions using bigram pattern matching
- ClusterPositions: Container for detected cluster positions
- AttentionExtractor: Extracts attention weights for analysis
- FlowMetric & InformationFlowComputer: Various information flow metrics
- ClusterStatistics: Statistics about attention patterns relative to cluster points
"""

from .detector import ClusterDetector, ClusterPositions, RandomClusterDetector
from .definitions import TaskConfig, load_task_config, get_available_tasks
from .extractor import AttentionExtractor, AttentionBipartiteGraph
from .statistics import (
    FlowMetric,
    InformationFlowComputer,
    AttentionSumFlow,
    GradientSaliencyFlow,
    AttentionRolloutFlow,
    AttentionValueWeightedFlow,
    CausalAblationFlow,
    AttentionMaskAblationFlow,
    get_flow_computer,
    ClusterStatistics,
    AnchorStatistics,  # backward compatibility
    compute_cluster_statistics,
    compute_anchor_statistics,  # backward compatibility
    compute_all_layer_statistics,
    aggregate_statistics,
)

__all__ = [
    # Detector
    "ClusterDetector",
    "ClusterPositions",
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
    # Statistics
    "ClusterStatistics",
    "compute_cluster_statistics",
    "compute_all_layer_statistics",
    "aggregate_statistics",
]
