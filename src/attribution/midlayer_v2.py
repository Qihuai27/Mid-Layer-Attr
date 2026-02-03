"""
MidLayer Attribution V2: Two-stage attribution with clustering.

Improvement over V1:
- Before Stage 2, cluster mid-layer activations based on similarity
- Tokens with similarity > mean + 2*std are merged into groups
- Causal intervention is performed at group level
- Final score: score(i) = Σ_g max_j∈g(s1(i,j)) * s2(g)

This reduces redundant computations and captures semantic groupings.
"""

from typing import Optional, Union, Literal, List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedModel

from .base import AttributionMethod, AttentionBasedMethod, AttributionResult


@dataclass
class ClusterInfo:
    """Information about a cluster of tokens."""
    cluster_id: int
    member_indices: List[int]  # Token positions in this cluster
    centroid: Optional[np.ndarray] = None  # Cluster centroid (optional)


class MidLayerAttributionV2(AttentionBasedMethod):
    """
    Two-stage attribution with mid-layer clustering.

    Stage 1: Compute attention rollout from input to mid-layer (same as V1)
    Clustering: Group similar mid-layer activations
    Stage 2: Causal intervention at group level
    Final: score(i) = Σ_g max_j∈g(s1(i,j)) * s2(g)
    """

    def __init__(
        self,
        mid_layer: Optional[int] = None,
        mid_layer_ratio: float = 0.25,
        residual_weight: float = 0.5,
        head_aggregation: Literal["mean", "max", "sum"] = "mean",
        cluster_threshold_std: float = 2.0,  # Merge if similarity > mean + threshold_std * std
        similarity_metric: Literal["cosine", "euclidean"] = "cosine",
    ):
        """
        Initialize MidLayer Attribution V2.

        Args:
            mid_layer: The intermediate layer index (0-indexed).
            mid_layer_ratio: If mid_layer is None, use layer at this ratio of depth.
            residual_weight: Weight for residual connection in attention rollout.
            head_aggregation: How to aggregate attention across heads.
            cluster_threshold_std: Number of std above mean for clustering threshold.
            similarity_metric: Metric for computing similarity between activations.
        """
        self.mid_layer = mid_layer
        self.mid_layer_ratio = mid_layer_ratio
        self.residual_weight = residual_weight
        self.head_aggregation = head_aggregation
        self.cluster_threshold_std = cluster_threshold_std
        self.similarity_metric = similarity_metric
        self._actual_mid_layer = None

    @property
    def name(self) -> str:
        layer = self._actual_mid_layer if self._actual_mid_layer is not None else self.mid_layer
        if layer is not None:
            return f"midlayer_v2_L{layer + 1}"
        return "midlayer_v2"

    def _get_mid_layer(self, model: PreTrainedModel) -> int:
        """Determine the mid-layer index to use."""
        if self.mid_layer is not None:
            return self.mid_layer

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        else:
            num_layers = 32

        return int(num_layers * self.mid_layer_ratio)

    def _compute_stage1_rollout(
        self,
        attention_layers: List[Tensor],
        target_pos: int,
    ) -> np.ndarray:
        """
        Stage 1: Compute attention rollout from layer 0 to mid_layer.
        (Same as V1)
        """
        seq_len = attention_layers[0].shape[-1]
        rollout = np.eye(seq_len, dtype=np.float32)

        mid_layer = self._actual_mid_layer if self._actual_mid_layer is not None else self.mid_layer
        for layer_idx in range(mid_layer + 1):
            attn = attention_layers[layer_idx]
            attn_agg = self._aggregate_heads(attn, method=self.head_aggregation)

            if attn_agg.dim() == 3:
                attn_agg = attn_agg.squeeze(0)

            attn_np = attn_agg.detach().cpu().float().numpy()

            identity = np.eye(seq_len, dtype=np.float32)
            attn_with_residual = (
                self.residual_weight * identity +
                (1 - self.residual_weight) * attn_np
            )
            rollout = attn_with_residual @ rollout

        s1 = rollout.T

        # Row normalization
        row_sums = s1.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        s1 = s1 / row_sums

        return s1

    def _compute_similarity_matrix(
        self,
        hidden_states: Tensor,
        target_pos: int,
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix for mid-layer activations.

        Args:
            hidden_states: Mid-layer hidden states [1, T, D]
            target_pos: Only consider positions up to target_pos

        Returns:
            similarity: [T, T] similarity matrix
        """
        H = hidden_states[0, :target_pos + 1, :]  # [T', D]
        T = H.shape[0]

        if self.similarity_metric == "cosine":
            # Normalize for cosine similarity
            H_norm = F.normalize(H, p=2, dim=-1)
            similarity = torch.mm(H_norm, H_norm.t())  # [T', T']
        else:  # euclidean
            # Convert to similarity: sim = 1 / (1 + dist)
            dists = torch.cdist(H.unsqueeze(0), H.unsqueeze(0))[0]  # [T', T']
            similarity = 1.0 / (1.0 + dists)

        sim_np = similarity.cpu().float().numpy()

        # Pad to full sequence length
        full_sim = np.zeros((hidden_states.shape[1], hidden_states.shape[1]), dtype=np.float32)
        full_sim[:T, :T] = sim_np

        return full_sim

    def _cluster_by_similarity(
        self,
        similarity_matrix: np.ndarray,
        target_pos: int,
    ) -> List[ClusterInfo]:
        """
        Cluster tokens based on similarity using threshold: mean + k*std.

        Uses agglomerative approach: iteratively merge most similar pairs
        until no pair exceeds threshold.

        Args:
            similarity_matrix: [T, T] pairwise similarity
            target_pos: Only consider positions up to target_pos

        Returns:
            List of ClusterInfo objects
        """
        T = target_pos + 1
        sim = similarity_matrix[:T, :T].copy()

        # Compute threshold: mean + k * std (excluding diagonal)
        mask = ~np.eye(T, dtype=bool)
        off_diag = sim[mask]
        threshold = off_diag.mean() + self.cluster_threshold_std * off_diag.std()

        # Initialize: each token is its own cluster
        # Use Union-Find for efficient clustering
        parent = list(range(T))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Find all pairs above threshold and merge
        for i in range(T):
            for j in range(i + 1, T):
                if sim[i, j] > threshold:
                    union(i, j)

        # Build cluster groups
        clusters_dict: Dict[int, List[int]] = {}
        for i in range(T):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(i)

        # Convert to ClusterInfo list
        clusters = []
        for cluster_id, (_, members) in enumerate(sorted(clusters_dict.items())):
            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                member_indices=sorted(members),
            ))

        return clusters

    def _compute_stage2_clustered(
        self,
        model: PreTrainedModel,
        input_ids: Tensor,
        target_pos: int,
        clusters: List[ClusterInfo],
        mid_layer_input: Tensor,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Stage 2: Compute causal influence at cluster level.

        For each cluster, zero out all member positions and measure
        the probability drop.

        Args:
            model: HuggingFace model
            input_ids: Input token IDs [1, T]
            target_pos: Position to evaluate
            clusters: List of cluster info
            mid_layer_input: Mid-layer hidden states [1, T, D]

        Returns:
            s2_tokens: Per-token scores (cluster score assigned to all members) [T]
            s2_clusters: Per-cluster scores {cluster_id: score}
        """
        device = next(model.parameters()).device
        seq_len = input_ids.shape[1]

        # Get model components
        mid_layer = self._actual_mid_layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            if hasattr(model.model, 'norm'):
                final_norm = model.model.norm
            else:
                final_norm = model.model.final_layernorm
            lm_head = model.lm_head
            rotary_emb = getattr(model.model, 'rotary_emb', None)
        else:
            layers = model.transformer.h
            final_norm = model.transformer.ln_f
            lm_head = model.lm_head
            rotary_emb = None

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        position_embeddings = None
        if rotary_emb is not None:
            cos, sin = rotary_emb(mid_layer_input, position_ids)
            position_embeddings = (cos, sin)

        # Get original probability
        with torch.no_grad():
            hidden_states = mid_layer_input.clone()
            for layer_idx in range(mid_layer, len(layers)):
                layer = layers[layer_idx]
                try:
                    if position_embeddings is not None:
                        layer_outputs = layer(
                            hidden_states,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        layer_outputs = layer(hidden_states, position_ids=position_ids)
                except TypeError:
                    layer_outputs = layer(hidden_states)

                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            hidden_states = final_norm(hidden_states)
            original_logits = lm_head(hidden_states[0, target_pos])
            target_token_id = original_logits.argmax().item()
            original_prob = torch.softmax(original_logits, dim=-1)[target_token_id].item()

        # Compute s2 for each cluster
        s2_clusters = {}
        s2_tokens = np.zeros(seq_len, dtype=np.float32)

        with torch.no_grad():
            for cluster in clusters:
                # Skip if cluster is entirely after target_pos
                if min(cluster.member_indices) > target_pos:
                    s2_clusters[cluster.cluster_id] = 0.0
                    continue

                # Zero out all positions in this cluster
                modified_input = mid_layer_input.clone()
                for j in cluster.member_indices:
                    if j <= target_pos:
                        modified_input[0, j, :] = 0.0

                # Forward through remaining layers
                hidden_states = modified_input
                for layer_idx in range(mid_layer, len(layers)):
                    layer = layers[layer_idx]
                    try:
                        if position_embeddings is not None:
                            layer_outputs = layer(
                                hidden_states,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                            )
                        else:
                            layer_outputs = layer(hidden_states, position_ids=position_ids)
                    except TypeError:
                        layer_outputs = layer(hidden_states)

                    hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

                hidden_states = final_norm(hidden_states)
                modified_logits = lm_head(hidden_states[0, target_pos])
                modified_prob = torch.softmax(modified_logits, dim=-1)[target_token_id].item()

                prob_drop = max(0, original_prob - modified_prob)
                s2_clusters[cluster.cluster_id] = prob_drop

                # Assign cluster score to all member tokens
                for j in cluster.member_indices:
                    s2_tokens[j] = prob_drop

        return s2_tokens, s2_clusters

    def _compute_final_scores(
        self,
        s1: np.ndarray,
        clusters: List[ClusterInfo],
        s2_clusters: Dict[int, float],
        target_pos: int,
    ) -> np.ndarray:
        """
        Compute final scores using: score(i) = Σ_g max_j∈g(s1(i,j)) * s2(g)

        Args:
            s1: Attention rollout matrix [T, T], s1[i,j] = flow from input i to mid-layer j
            clusters: List of cluster info
            s2_clusters: Per-cluster causal scores

        Returns:
            scores: Final attribution scores [T]
        """
        T = s1.shape[0]
        scores = np.zeros(T, dtype=np.float32)

        for i in range(T):
            if i > target_pos:
                continue

            score_i = 0.0
            for cluster in clusters:
                # Get max s1(i, j) for j in cluster
                member_indices = [j for j in cluster.member_indices if j <= target_pos]
                if not member_indices:
                    continue

                max_s1 = max(s1[i, j] for j in member_indices)
                s2_g = s2_clusters.get(cluster.cluster_id, 0.0)
                score_i += max_s1 * s2_g

            scores[i] = score_i

        return scores

    def attribute(
        self,
        model: PreTrainedModel,
        input_ids: Union[Tensor, np.ndarray],
        target_pos: int,
        **kwargs,
    ) -> AttributionResult:
        """
        Compute attribution scores using clustered mid-layer method.
        """
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        T = input_ids.shape[1]

        if target_pos < 0:
            target_pos = T + target_pos
        if target_pos < 0 or target_pos >= T:
            raise ValueError(f"target_pos {target_pos} out of range [0, {T})")

        self._actual_mid_layer = self._get_mid_layer(model)

        # Extract attention from all layers
        attention_layers = self._extract_attention(model, input_ids)

        num_layers = len(model.model.layers) if hasattr(model, 'model') else len(model.transformer.h)
        if self._actual_mid_layer >= num_layers:
            raise ValueError(f"mid_layer {self._actual_mid_layer} >= num_layers {num_layers}")

        # Stage 1: Attention rollout
        s1 = self._compute_stage1_rollout(attention_layers, target_pos)

        # Get mid-layer hidden states for clustering
        mid_layer_hidden = {}

        def capture_mid_layer_hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                mid_layer_hidden['states'] = outputs[0].clone()
            else:
                mid_layer_hidden['states'] = outputs.clone()

        mid_layer = self._actual_mid_layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            hook_layer = layers[mid_layer - 1] if mid_layer > 0 else model.model.embed_tokens
        else:
            layers = model.transformer.h
            hook_layer = layers[mid_layer - 1] if mid_layer > 0 else model.transformer.wte

        with torch.no_grad():
            handle = hook_layer.register_forward_hook(capture_mid_layer_hook)
            model(input_ids, output_hidden_states=True)
            handle.remove()
            mid_layer_input = mid_layer_hidden['states'].clone()

        # Clustering: Group similar activations
        similarity_matrix = self._compute_similarity_matrix(mid_layer_input, target_pos)
        clusters = self._cluster_by_similarity(similarity_matrix, target_pos)

        # Stage 2: Causal intervention at cluster level
        s2_tokens, s2_clusters = self._compute_stage2_clustered(
            model, input_ids, target_pos, clusters, mid_layer_input
        )

        # Final scores: score(i) = Σ_g max_j∈g(s1(i,j)) * s2(g)
        scores = self._compute_final_scores(s1, clusters, s2_clusters, target_pos)

        # Zero out positions after target_pos
        scores[target_pos + 1:] = 0

        # Compute cluster statistics for metadata
        cluster_sizes = [len(c.member_indices) for c in clusters]

        return AttributionResult(
            scores=scores,
            target_pos=target_pos,
            method_name=self.name,
            metadata={
                "mid_layer": self._actual_mid_layer,
                "residual_weight": self.residual_weight,
                "head_aggregation": self.head_aggregation,
                "cluster_threshold_std": self.cluster_threshold_std,
                "num_clusters": len(clusters),
                "cluster_sizes": cluster_sizes,
                "avg_cluster_size": np.mean(cluster_sizes),
                "s2_clusters": s2_clusters,
            }
        )


def midlayer_attribution_v2(
    model: PreTrainedModel,
    input_ids: Union[Tensor, np.ndarray],
    target_pos: int = -1,
    mid_layer: int = None,
    residual_weight: float = 0.5,
    head_aggregation: str = "mean",
    cluster_threshold_std: float = 2.0,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for mid-layer attribution v2.
    """
    method = MidLayerAttributionV2(
        mid_layer=mid_layer,
        residual_weight=residual_weight,
        head_aggregation=head_aggregation,
        cluster_threshold_std=cluster_threshold_std,
    )
    result = method.attribute(model, input_ids, target_pos, **kwargs)
    return result.scores
