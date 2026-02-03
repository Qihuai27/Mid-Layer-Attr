"""
ICL Analysis - A framework for studying In-Context Learning in Large Language Models.

This package provides two main research directions:

1. **Cluster (信息汇聚研究)**: Studies how label words act as information aggregation
   points in ICL. Based on "Label Words are Anchors" phenomenon.

   Key modules:
   - src.cluster: Cluster detection and information flow analysis
   - experiments.cluster: Experiment runners for cluster analysis

2. **Attribution (归因研究)**: Token-level input attribution for autoregressive models,
   with novel evaluation metrics (Noise Insertion AUC, Representation Insertion AUC).

   Key modules:
   - src.attribution: Attribution methods (DePass, AttentionRollout, IntegratedGradients, etc.)
   - src.eval: Evaluation metrics for attribution quality

Shared Infrastructure:
- src.models: Model loading and attention hooks
- src.data: Dataset loading and ICL prompt wrapping
- src.visualization: Plotting utilities
- src.utils: Random seeds, I/O utilities
"""

__version__ = "0.2.0"

# Core modules
from . import cluster
from . import attribution
from . import eval
from . import models
from . import data
from . import visualization
from . import utils

# Backward compatibility - deprecated, use src.cluster instead
from . import anchors

__all__ = [
    "cluster",
    "attribution",
    "eval",
    "models",
    "data",
    "visualization",
    "utils",
    "anchors",  # deprecated
]
