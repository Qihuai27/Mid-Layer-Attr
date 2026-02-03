"""
Visualization module for cluster and attribution analysis.
"""

from .plotting import (
    plot_cluster_statistics,
    plot_attribution_statistics,  # backward compatibility alias
    load_results,
)
from .sample_attribution import (
    plot_token_heatmap,
    plot_auc_curves,
    plot_sample_attribution,
    visualize_attribution,
    SampleVisualizationData,
    ATTRIBUTION_CMAP,
    DIVERGING_CMAP,
    CURVE_COLORS,
)

__all__ = [
    # Layer-level statistics (cluster analysis)
    "plot_cluster_statistics",
    "plot_attribution_statistics",  # backward compatibility alias
    "load_results",
    # Sample-level attribution
    "plot_token_heatmap",
    "plot_auc_curves",
    "plot_sample_attribution",
    "visualize_attribution",
    "SampleVisualizationData",
    # Color schemes
    "ATTRIBUTION_CMAP",
    "DIVERGING_CMAP",
    "CURVE_COLORS",
]
