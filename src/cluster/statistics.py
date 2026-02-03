"""
Statistics computation for cluster point attention analysis.

Computes three key statistics:
- S_a (aggregation): Information flow from previous tokens TO cluster positions
- S_o (output): Information flow from cluster positions TO final position
- S_w (within): Information flow between other (non-cluster, non-final) positions

Supports multiple information flow metrics:
- attention_sum: Direct attention weights summed across heads
- gradient_saliency: |A ⊙ ∂L/∂A| (Gradient × Input)
- attention_rollout: Cumulative attention across layers
- attention_value_weighted: Attention × value vector norm
- causal_ablation: Batched hidden state zeroing
- attention_mask_ablation: Attention mask blocking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from .detector import ClusterPositions
from .extractor import AttentionBipartiteGraph

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class FlowMetric(str, Enum):
    """Available information flow metrics."""
    ATTENTION_SUM = "attention_sum"
    GRADIENT_SALIENCY = "gradient_saliency"
    ATTENTION_ROLLOUT = "attention_rollout"
    ATTENTION_VALUE_WEIGHTED = "attention_value_weighted"
    CAUSAL_ABLATION = "causal_ablation"
    ATTENTION_MASK_ABLATION = "attention_mask_ablation"
    # Legacy alias
    RAW_ATTENTION = "attention_sum"


class InformationFlowComputer(ABC):
    """
    Abstract base class for computing token-to-token information flow.

    The compute() method takes attention weights and optional additional context,
    and returns an information flow matrix I(i,j) where entry (i,j) represents
    the strength of information flow from token j to token i.
    """

    @abstractmethod
    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute information flow matrix I(i,j).

        Args:
            attn_weights: Attention weights [batch, heads, seq_q, seq_k]
            gradient: Optional gradient tensor (same shape as attn_weights)
            **kwargs: Additional context (value_states, hidden_states, model, etc.)

        Returns:
            Information flow matrix [seq_q, seq_k]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @property
    @abstractmethod
    def requires_gradient(self) -> bool:
        """Whether this metric requires gradient information."""
        pass

    @property
    def requires_model(self) -> bool:
        """Whether this metric requires access to the model for additional forward passes."""
        return False

    @property
    def requires_value_states(self) -> bool:
        """Whether this metric requires value states."""
        return False

    @property
    def is_multi_layer(self) -> bool:
        """Whether this metric needs information from multiple layers."""
        return False


# =============================================================================
# Basic Attention-based Metrics
# =============================================================================

class AttentionSumFlow(InformationFlowComputer):
    """
    Use attention weights summed across heads as information flow.

    I_l(i,j) = sum_h A_{h,l}(i,j)

    This is the simplest metric - just aggregates attention scores.
    """

    @property
    def name(self) -> str:
        return "attention_sum"

    @property
    def requires_gradient(self) -> bool:
        return False

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        # Sum over heads: [batch, heads, seq_q, seq_k] -> [batch, seq_q, seq_k]
        flow = attn_weights.sum(dim=1)

        # Squeeze batch dimension if single sample
        if flow.dim() == 3:
            flow = flow.squeeze(0)

        return flow.detach().cpu().numpy()


# Alias for backward compatibility
RawAttentionFlow = AttentionSumFlow


class GradientSaliencyFlow(InformationFlowComputer):
    """
    Gradient-based saliency as information flow (from "Label Words are Anchors").

    I_l(i,j) = sum_h |A_{h,l}(i,j) ⊙ ∂L/∂A_{h,l}(i,j)|

    This measures how much each attention connection contributes to the final prediction
    using first-order Taylor expansion (Gradient × Input).
    """

    @property
    def name(self) -> str:
        return "gradient_saliency"

    @property
    def requires_gradient(self) -> bool:
        return True

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        if gradient is None:
            raise ValueError("GradientSaliencyFlow requires gradient tensor")

        # I_l = sum_h |A_{h,l} ⊙ ∂L/∂A_{h,l}|
        flow = torch.abs(attn_weights * gradient).sum(dim=1)

        # Squeeze batch dimension if single sample
        if flow.dim() == 3:
            flow = flow.squeeze(0)

        return flow.detach().cpu().numpy()


# =============================================================================
# Advanced Attention-based Metrics
# =============================================================================

class AttentionRolloutFlow(InformationFlowComputer):
    """
    Attention Rollout - cumulative attention across layers.

    For layer l:
        Ā^(l) = 0.5 * I + 0.5 * A^(l)  (account for residual connection)
        R^(l) = Ā^(l) @ R^(l-1)        (cumulative product)
        R^(0) = I                       (identity)

    I_l(i,j) = R^(l)(i,j)

    This accounts for multi-layer information propagation through residual connections.
    Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
    """

    def __init__(self, residual_weight: float = 0.5):
        """
        Args:
            residual_weight: Weight for residual connection (default 0.5)
        """
        self.residual_weight = residual_weight
        self._rollout_cache: Dict[int, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "attention_rollout"

    @property
    def requires_gradient(self) -> bool:
        return False

    @property
    def is_multi_layer(self) -> bool:
        return True

    def reset_cache(self):
        """Reset the rollout cache for a new sample."""
        self._rollout_cache = {}

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        layer_idx: int = 0,
        all_layer_weights: Optional[List[Tensor]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute rollout for a specific layer.

        Args:
            attn_weights: Current layer attention [batch, heads, seq_q, seq_k]
            layer_idx: Current layer index
            all_layer_weights: All layer attention weights (for computing rollout from scratch)
        """
        # Mean over heads: [batch, seq_q, seq_k]
        attn = attn_weights.mean(dim=1)
        if attn.dim() == 3:
            attn = attn.squeeze(0)
        attn = attn.detach().cpu().numpy()

        seq_len = attn.shape[0]

        # Account for residual: Ā = α*I + (1-α)*A
        identity = np.eye(seq_len)
        attn_with_residual = self.residual_weight * identity + (1 - self.residual_weight) * attn

        if layer_idx == 0:
            # First layer: rollout = attention with residual
            rollout = attn_with_residual
        else:
            # Get previous rollout
            if layer_idx - 1 in self._rollout_cache:
                prev_rollout = self._rollout_cache[layer_idx - 1]
            elif all_layer_weights is not None and layer_idx > 0:
                # Compute from scratch if we have all layers
                prev_rollout = identity
                for l in range(layer_idx):
                    prev_attn = all_layer_weights[l].mean(dim=1)
                    if prev_attn.dim() == 3:
                        prev_attn = prev_attn.squeeze(0)
                    prev_attn = prev_attn.detach().cpu().numpy()
                    prev_attn_res = self.residual_weight * identity + (1 - self.residual_weight) * prev_attn
                    prev_rollout = prev_attn_res @ prev_rollout
            else:
                # Fallback: just use current layer
                prev_rollout = identity

            # Cumulative product: R^(l) = Ā^(l) @ R^(l-1)
            rollout = attn_with_residual @ prev_rollout

        # Cache for next layer
        self._rollout_cache[layer_idx] = rollout

        return rollout


class AttentionValueWeightedFlow(InformationFlowComputer):
    """
    Attention weighted by value vector norm.

    I_l(i,j) = sum_h A_{h,l}(i,j) * ||V_{h,l}(j)||_2

    This weights attention by the magnitude of information being transferred,
    giving more importance to connections that transfer larger value vectors.
    """

    @property
    def name(self) -> str:
        return "attention_value_weighted"

    @property
    def requires_gradient(self) -> bool:
        return False

    @property
    def requires_value_states(self) -> bool:
        return True

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        value_states: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute attention weighted by value norms.

        Args:
            attn_weights: Attention weights [batch, heads, seq_q, seq_k]
            value_states: Value vectors [batch, heads, seq_k, head_dim]
        """
        if value_states is None:
            # Fallback to simple attention sum if no value states
            return AttentionSumFlow().compute(attn_weights)

        # Compute value norms: [batch, heads, seq_k]
        value_norms = torch.norm(value_states, dim=-1)

        # Weight attention by value norms
        # attn_weights: [batch, heads, seq_q, seq_k]
        # value_norms: [batch, heads, seq_k] -> [batch, heads, 1, seq_k]
        weighted_attn = attn_weights * value_norms.unsqueeze(-2)

        # Sum over heads
        flow = weighted_attn.sum(dim=1)

        if flow.dim() == 3:
            flow = flow.squeeze(0)

        return flow.detach().cpu().numpy()


# =============================================================================
# Ablation-based Metrics
# =============================================================================

class CausalAblationFlow(InformationFlowComputer):
    """
    Causal ablation - measure effect of zeroing input token hidden states.

    For each input position j:
        I_l(i,j) = ||h_i^(l) - h_i^(l, j=0)||_2

    where h_i^(l, j=0) is the output hidden state at position i when
    the input hidden state at position j is zeroed.

    This directly measures causal influence but is computationally expensive.
    Uses batched computation for efficiency.
    """

    def __init__(self, chunk_size: int = 32):
        """
        Args:
            chunk_size: Number of ablations to batch together (for memory efficiency)
        """
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "causal_ablation"

    @property
    def requires_gradient(self) -> bool:
        return False

    @property
    def requires_model(self) -> bool:
        return True

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        layer_module: Optional[Any] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute causal ablation scores.

        Args:
            attn_weights: Attention weights [batch, heads, seq_q, seq_k]
            hidden_states: Input hidden states to this layer [batch, seq, hidden_dim]
            layer_module: The transformer layer module for forward passes
            attention_mask: Attention mask [batch, seq]
        """
        if hidden_states is None or layer_module is None:
            # Fallback to attention sum
            return AttentionSumFlow().compute(attn_weights)

        device = hidden_states.device
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get baseline output (no ablation)
        with torch.no_grad():
            baseline_output = layer_module(hidden_states, attention_mask=attention_mask)
            if isinstance(baseline_output, tuple):
                baseline_output = baseline_output[0]
            baseline_output = baseline_output.squeeze(0)  # [seq, hidden_dim]

        # Compute ablation effects
        flow_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

        # Process in chunks for memory efficiency
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_positions = list(range(chunk_start, chunk_end))
            num_positions = len(chunk_positions)

            # Create batched ablated inputs
            # [num_positions, seq, hidden_dim]
            ablated_inputs = hidden_states.expand(num_positions, -1, -1).clone()

            for i, pos in enumerate(chunk_positions):
                ablated_inputs[i, pos, :] = 0.0

            # Batched forward pass
            with torch.no_grad():
                if attention_mask is not None:
                    ablated_mask = attention_mask.expand(num_positions, -1)
                else:
                    ablated_mask = None

                ablated_outputs = layer_module(ablated_inputs, attention_mask=ablated_mask)
                if isinstance(ablated_outputs, tuple):
                    ablated_outputs = ablated_outputs[0]

            # Compute L2 differences for each ablated position
            for i, pos in enumerate(chunk_positions):
                # Difference: [seq, hidden_dim]
                diff = baseline_output - ablated_outputs[i]
                # L2 norm per output position: [seq]
                l2_norms = torch.norm(diff, dim=-1).cpu().numpy()
                # I(output_pos, ablated_pos) = L2 norm of change
                flow_matrix[:, pos] = l2_norms

        return flow_matrix


class AttentionMaskAblationFlow(InformationFlowComputer):
    """
    Attention mask ablation - measure effect of blocking attention to specific positions.

    For each position j, mask attention to j and measure output change:
        I_l(i,j) = ||h_i^(l) - h_i^(l, attn[:,j]=0)||_2

    This is more efficient than hidden state ablation as it only requires
    modifying the attention mask, not re-computing Q/K/V projections.
    """

    def __init__(self, chunk_size: int = 32):
        """
        Args:
            chunk_size: Number of ablations to batch together
        """
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "attention_mask_ablation"

    @property
    def requires_gradient(self) -> bool:
        return False

    @property
    def requires_model(self) -> bool:
        return True

    def compute(
        self,
        attn_weights: Tensor,
        gradient: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
        layer_module: Optional[Any] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute attention mask ablation scores.

        Args:
            attn_weights: Attention weights [batch, heads, seq_q, seq_k]
            hidden_states: Input hidden states [batch, seq, hidden_dim]
            layer_module: The transformer layer module
            attention_mask: Original attention mask [batch, 1, 1, seq] or [batch, seq]
        """
        if hidden_states is None or layer_module is None:
            return AttentionSumFlow().compute(attn_weights)

        device = hidden_states.device
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get baseline output
        with torch.no_grad():
            baseline_output = layer_module(hidden_states, attention_mask=attention_mask)
            if isinstance(baseline_output, tuple):
                baseline_output = baseline_output[0]
            baseline_output = baseline_output.squeeze(0)

        flow_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)

        # Create base attention mask if not provided
        if attention_mask is None:
            # Causal mask: [1, 1, seq, seq]
            base_mask = torch.zeros(1, 1, seq_len, seq_len, device=device)
        else:
            # Expand to [1, 1, seq, seq] if needed
            if attention_mask.dim() == 2:
                base_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                base_mask = base_mask.expand(-1, -1, seq_len, -1)
            else:
                base_mask = attention_mask

        # Process in chunks
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_positions = list(range(chunk_start, chunk_end))
            num_positions = len(chunk_positions)

            # Create batched masked attention
            # Block attention to each position in the chunk
            ablated_masks = base_mask.expand(num_positions, -1, -1, -1).clone()
            for i, pos in enumerate(chunk_positions):
                # Set attention to position pos to -inf (will become 0 after softmax)
                ablated_masks[i, :, :, pos] = -1e9

            # Batched hidden states
            ablated_inputs = hidden_states.expand(num_positions, -1, -1)

            with torch.no_grad():
                ablated_outputs = layer_module(ablated_inputs, attention_mask=ablated_masks)
                if isinstance(ablated_outputs, tuple):
                    ablated_outputs = ablated_outputs[0]

            # Compute differences
            for i, pos in enumerate(chunk_positions):
                diff = baseline_output - ablated_outputs[i]
                l2_norms = torch.norm(diff, dim=-1).cpu().numpy()
                flow_matrix[:, pos] = l2_norms

        return flow_matrix


# =============================================================================
# Factory Function
# =============================================================================

def get_flow_computer(
    metric: FlowMetric,
    **kwargs,
) -> InformationFlowComputer:
    """
    Factory function to get information flow computer.

    Args:
        metric: FlowMetric enum value
        **kwargs: Additional arguments for specific computers (e.g., chunk_size)

    Returns:
        InformationFlowComputer instance
    """
    if metric == FlowMetric.ATTENTION_SUM or metric == FlowMetric.RAW_ATTENTION:
        return AttentionSumFlow()
    elif metric == FlowMetric.GRADIENT_SALIENCY:
        return GradientSaliencyFlow()
    elif metric == FlowMetric.ATTENTION_ROLLOUT:
        return AttentionRolloutFlow(
            residual_weight=kwargs.get("residual_weight", 0.5)
        )
    elif metric == FlowMetric.ATTENTION_VALUE_WEIGHTED:
        return AttentionValueWeightedFlow()
    elif metric == FlowMetric.CAUSAL_ABLATION:
        return CausalAblationFlow(
            chunk_size=kwargs.get("chunk_size", 32)
        )
    elif metric == FlowMetric.ATTENTION_MASK_ABLATION:
        return AttentionMaskAblationFlow(
            chunk_size=kwargs.get("chunk_size", 32)
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")


@dataclass
class ClusterStatistics:
    """
    Statistics about attention patterns relative to cluster points.

    Attributes:
        S_a: Aggregation - normalized info flow from previous tokens TO cluster positions
        S_o: Output - normalized info flow from cluster positions TO final position
        S_w: Within - normalized info flow between other (non-cluster, non-final) positions
        layer_idx: Layer index these statistics are computed for
        metric: Name of the information flow metric used
    """

    S_a: float  # Aggregation: previous tokens → cluster positions
    S_o: float  # Output: cluster positions → final position
    S_w: float  # Within: other positions → other positions
    layer_idx: int
    metric: str = "gradient_saliency"

    def to_dict(self) -> Dict[str, float]:
        return {
            "S_a": self.S_a,
            "S_o": self.S_o,
            "S_w": self.S_w,
            "layer_idx": self.layer_idx,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ClusterStatistics":
        # Handle legacy keys for backward compatibility
        if "S_wp" in d and "S_a" not in d:
            d = d.copy()
            d["S_a"] = d.pop("S_wp")
            d["S_o"] = d.pop("S_pq")
            d["S_w"] = d.pop("S_ww")
        return cls(**d)

    # Backward compatibility properties
    @property
    def S_wp(self) -> float:
        """Deprecated: Use S_a instead."""
        return self.S_a

    @property
    def S_pq(self) -> float:
        """Deprecated: Use S_o instead."""
        return self.S_o

    @property
    def S_ww(self) -> float:
        """Deprecated: Use S_w instead."""
        return self.S_w


# Backward compatibility alias
AnchorStatistics = ClusterStatistics


def compute_cluster_statistics(
    attn_graph: AttentionBipartiteGraph,
    cluster_positions: ClusterPositions,
    flow_computer: Optional[InformationFlowComputer] = None,
    gradient: Optional[Tensor] = None,
    # Legacy parameters for backward compatibility
    use_gradient: Optional[bool] = None,
    # Additional kwargs for advanced flow computers
    **kwargs,
) -> ClusterStatistics:
    """
    Compute cluster-based attention statistics for a single layer.

    Based on the paper "Label Words are Anchors", Section 2.1:
    - I_l(i,j): Information flow matrix (computed by flow_computer)
    - C_a = {(p_k, j): j < p_k} - text tokens flowing TO cluster positions (aggregation)
    - C_o = {(q, p_k)} - cluster positions flowing TO final position (output)
    - C_w = all other (i,j) pairs where j < i, excluding C_a and C_o (within)
    - S_C = (1/|C|) * sum_{(i,j) in C} I_l(i,j)

    Args:
        attn_graph: Attention bipartite graph for the layer
        cluster_positions: Detected cluster positions
        flow_computer: Information flow computer (default: GradientSaliencyFlow if gradient provided, else RawAttentionFlow)
        gradient: Gradient tensor if using gradient-based metrics
        use_gradient: [Deprecated] Use flow_computer instead
        **kwargs: Additional arguments passed to flow_computer.compute()
            - layer_idx: Layer index (for AttentionRolloutFlow)
            - all_layer_weights: All layer attention weights (for AttentionRolloutFlow)
            - value_states: Value vectors (for AttentionValueWeightedFlow)
            - hidden_states: Input hidden states (for ablation methods)
            - layer_module: Transformer layer module (for ablation methods)
            - attention_mask: Attention mask (for ablation methods)

    Returns:
        ClusterStatistics for the layer
    """
    # Handle legacy use_gradient parameter
    if flow_computer is None:
        if use_gradient is not None and use_gradient and gradient is not None:
            flow_computer = GradientSaliencyFlow()
        elif gradient is not None:
            flow_computer = GradientSaliencyFlow()
        else:
            flow_computer = RawAttentionFlow()

    # Get attention weights [batch, heads, seq_q, seq_k]
    attn_weights = attn_graph.weights

    # Compute information flow matrix I_l(i,j)
    flow_matrix = flow_computer.compute(attn_weights, gradient, **kwargs)
    seq_len = flow_matrix.shape[0]

    # Get positions
    cluster_pos = cluster_positions.all_cluster_positions.cpu().numpy()
    final_pos = cluster_positions.final_position
    if final_pos.dim() > 0:
        final_pos = final_pos[0]
    final_pos = int(final_pos.cpu().item())

    # Build C_a: {(p_k, j): j < p_k} - positions where text flows TO clusters (aggregation)
    # I(i,j) means info flows from j to i, so we want rows=cluster_pos, cols < cluster_pos
    C_a_mask = np.zeros((seq_len, seq_len), dtype=bool)
    for p_k in cluster_pos:
        C_a_mask[p_k, :p_k] = True  # row p_k, columns 0 to p_k-1

    # Build C_o: {(q, p_k)} - positions where clusters flow TO final position (output)
    C_o_mask = np.zeros((seq_len, seq_len), dtype=bool)
    C_o_mask[final_pos, cluster_pos] = True

    # Build C_w: all (i,j) where j < i, excluding C_a and C_o (within)
    # First create lower triangular mask (j < i, causal)
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool), k=-1)
    C_w_mask = causal_mask & ~C_a_mask & ~C_o_mask

    # Compute raw sums: sum_{(i,j) in C} I_l(i,j)
    S_a_raw = flow_matrix[C_a_mask].sum()
    S_o_raw = flow_matrix[C_o_mask].sum()
    S_w_raw = flow_matrix[C_w_mask].sum()

    # Normalize by set cardinality |C|: S_C = (1/|C|) * sum
    C_a_size = C_a_mask.sum()
    C_o_size = C_o_mask.sum()  # Should equal num_cluster
    C_w_size = C_w_mask.sum()

    S_a_norm = S_a_raw / C_a_size if C_a_size > 0 else 0
    S_o_norm = S_o_raw / C_o_size if C_o_size > 0 else 0
    S_w_norm = S_w_raw / C_w_size if C_w_size > 0 else 0

    return ClusterStatistics(
        S_a=float(S_a_norm),
        S_o=float(S_o_norm),
        S_w=float(S_w_norm),
        layer_idx=attn_graph.layer_idx,
        metric=flow_computer.name,
    )


# Backward compatibility alias
compute_anchor_statistics = compute_cluster_statistics


def compute_all_layer_statistics(
    attn_graphs: List[AttentionBipartiteGraph],
    cluster_positions: ClusterPositions,
    flow_computer: Optional[InformationFlowComputer] = None,
    gradients: Optional[List[Tensor]] = None,
) -> List[ClusterStatistics]:
    """
    Compute statistics for all layers.

    Args:
        attn_graphs: List of attention graphs per layer
        cluster_positions: Detected cluster positions
        flow_computer: Information flow computer (default: auto-select based on gradients)
        gradients: Optional gradients per layer

    Returns:
        List of ClusterStatistics per layer
    """
    # Auto-select flow computer if not provided
    if flow_computer is None:
        if gradients is not None:
            flow_computer = GradientSaliencyFlow()
        else:
            flow_computer = RawAttentionFlow()

    stats = []
    for i, graph in enumerate(attn_graphs):
        grad = gradients[i] if gradients is not None else None
        stat = compute_cluster_statistics(
            graph, cluster_positions, flow_computer=flow_computer, gradient=grad
        )
        stats.append(stat)
    return stats


def aggregate_statistics(
    all_stats: List[List[ClusterStatistics]],
) -> Dict[str, np.ndarray]:
    """
    Aggregate statistics across multiple samples.

    Args:
        all_stats: List of per-sample statistics (each is list per layer)

    Returns:
        Dict with aggregated statistics arrays:
        - S_a, S_o, S_w: Raw arrays [num_samples, num_layers]
        - mean_S_a, mean_S_o, mean_S_w: Mean across samples [num_layers]
        - std_S_a, std_S_o, std_S_w: Std across samples [num_layers]
    """
    num_samples = len(all_stats)
    num_layers = len(all_stats[0]) if all_stats else 0

    S_a = np.zeros((num_samples, num_layers))
    S_o = np.zeros((num_samples, num_layers))
    S_w = np.zeros((num_samples, num_layers))

    for sample_idx, sample_stats in enumerate(all_stats):
        for layer_stats in sample_stats:
            layer_idx = layer_stats.layer_idx
            S_a[sample_idx, layer_idx] = layer_stats.S_a
            S_o[sample_idx, layer_idx] = layer_stats.S_o
            S_w[sample_idx, layer_idx] = layer_stats.S_w

    return {
        # New naming
        "S_a": S_a,  # [num_samples, num_layers]
        "S_o": S_o,
        "S_w": S_w,
        "mean_S_a": S_a.mean(axis=0),  # [num_layers]
        "mean_S_o": S_o.mean(axis=0),
        "mean_S_w": S_w.mean(axis=0),
        "std_S_a": S_a.std(axis=0),
        "std_S_o": S_o.std(axis=0),
        "std_S_w": S_w.std(axis=0),
        # Backward compatibility aliases
        "S_wp": S_a,
        "S_pq": S_o,
        "S_ww": S_w,
        "mean_S_wp": S_a.mean(axis=0),
        "mean_S_pq": S_o.mean(axis=0),
        "mean_S_ww": S_w.mean(axis=0),
        "std_S_wp": S_a.std(axis=0),
        "std_S_pq": S_o.std(axis=0),
        "std_S_ww": S_w.std(axis=0),
    }
