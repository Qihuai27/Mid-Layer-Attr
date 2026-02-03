"""
Cluster experiment module.

This module provides experiment infrastructure for studying
the information aggregation (cluster) phenomenon in LLMs.
"""

from .run import (
    ClusterConfig,
    ClusterExperiment,
    run_cluster_experiment,
)

__all__ = [
    "ClusterConfig",
    "ClusterExperiment",
    "run_cluster_experiment",
]
