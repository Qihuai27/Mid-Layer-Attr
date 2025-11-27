# Attribution Analysis Module

This module implements attention attribution analysis for studying how anchor words affect information flow in transformer models during in-context learning (ICL).

## Core Concept

The key insight from the paper "Label Words are Anchors" is that **any token-to-token information flow matrix** can be used to compute statistics about anchor word attention patterns. The module decouples:

1. **Information Flow Computation**: How to measure the strength of information flow from token j to token i
2. **Statistics Computation**: How to aggregate flow values into meaningful metrics (S_wp, S_pq, S_ww)

## Information Flow Metrics

The `InformationFlowComputer` abstract class defines the interface for computing token-to-token information flow matrices. Six metrics are currently implemented:

### Overview

| Metric | Class | Formula | Gradient | Model | Speed |
|--------|-------|---------|----------|-------|-------|
| Attention Sum | `AttentionSumFlow` | $\sum_h A_{h,l}(i,j)$ | No | No | Fast |
| Gradient Saliency | `GradientSaliencyFlow` | $\sum_h \|A \odot \nabla A\|$ | Yes | No | Medium |
| Attention Rollout | `AttentionRolloutFlow` | $\prod_{l'=1}^{l} \bar{A}^{(l')}$ | No | No | Fast |
| Attention × Value | `AttentionValueWeightedFlow` | $\sum_h A_{h,l}(i,j) \cdot \|V_h(j)\|$ | No | No | Fast |
| Causal Ablation | `CausalAblationFlow` | $\|h_i - h_i^{\setminus j}\|_2$ | No | Yes | Slow |
| Attention Mask Ablation | `AttentionMaskAblationFlow` | $\|h_i - h_i^{mask_j}\|_2$ | No | Yes | Slow |

### Detailed Descriptions

#### 1. Attention Sum (`AttentionSumFlow`)

Simplest metric - sum attention weights across heads:

$$I_l(i,j) = \sum_h A_{h,l}(i,j)$$

- **Pros**: Simple, fast, no extra requirements
- **Cons**: Doesn't account for downstream impact

#### 2. Gradient Saliency (`GradientSaliencyFlow`)

From "Label Words are Anchors" paper - gradient-weighted attention:

$$I_l(i,j) = \sum_h |A_{h,l}(i,j) \odot \frac{\partial \mathcal{L}}{\partial A_{h,l}(i,j)}|$$

- **Pros**: Measures actual contribution to predictions (Taylor expansion)
- **Cons**: Requires backward pass

#### 3. Attention Rollout (`AttentionRolloutFlow`)

Cumulative attention across layers, accounting for residual connections:

$$\bar{A}^{(l)} = \alpha I + (1-\alpha) A^{(l)}$$
$$R^{(l)} = \bar{A}^{(l)} \cdot R^{(l-1)}$$

Where $\alpha=0.5$ (residual weight) and $R^{(0)}=I$.

- **Pros**: Captures multi-layer information propagation
- **Cons**: Requires attention from all previous layers
- **Reference**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)

#### 4. Attention × Value Weighted (`AttentionValueWeightedFlow`)

Attention weighted by value vector magnitude:

$$I_l(i,j) = \sum_h A_{h,l}(i,j) \cdot \|V_{h,l}(j)\|_2$$

- **Pros**: Weights by information magnitude being transferred
- **Cons**: Requires value states capture

#### 5. Causal Ablation (`CausalAblationFlow`)

Zero each input token's hidden state, measure output change:

$$I_l(i,j) = \|h_i^{(l)} - h_i^{(l, j=0)}\|_2$$

- **Pros**: Direct causal measurement
- **Cons**: Expensive (O(seq_len) forward passes per layer)
- **Optimization**: Batched computation with configurable `chunk_size`

#### 6. Attention Mask Ablation (`AttentionMaskAblationFlow`)

Block attention to each position, measure output change:

$$I_l(i,j) = \|h_i^{(l)} - h_i^{(l, attn[:,j]=0)}\|_2$$

- **Pros**: More efficient than hidden state ablation
- **Cons**: Still requires multiple forward passes
- **Optimization**: Batched computation with configurable `chunk_size`

## Adding Custom Metrics

To add a new information flow metric, inherit from `InformationFlowComputer`:

```python
from experiments.attribution.statistics import InformationFlowComputer

class MyCustomFlow(InformationFlowComputer):
    @property
    def name(self) -> str:
        return "my_custom_flow"

    @property
    def requires_gradient(self) -> bool:
        return False  # Set True if gradient needed

    @property
    def requires_model(self) -> bool:
        return False  # Set True if model forward passes needed

    @property
    def requires_value_states(self) -> bool:
        return False  # Set True if value vectors needed

    @property
    def is_multi_layer(self) -> bool:
        return False  # Set True if needs info from multiple layers

    def compute(
        self,
        attn_weights,      # [batch, heads, seq_q, seq_k]
        gradient=None,     # Same shape, if requires_gradient
        **kwargs,          # value_states, hidden_states, layer_module, etc.
    ):
        # Your custom logic here
        flow = attn_weights.mean(dim=1)  # Example
        if flow.dim() == 3:
            flow = flow.squeeze(0)
        return flow.detach().cpu().numpy()
```

Then register in `get_flow_computer()` and `FlowMetric` enum.

## Statistics Computation

Given an information flow matrix $I_l(i,j)$, the module computes three statistics based on position sets:

### Position Set Definitions

| Set | Definition | Meaning |
|-----|------------|---------|
| $C_{wp}$ | $\{(p_k, j) : j < p_k\}$ | Text tokens flowing TO anchor positions |
| $C_{pq}$ | $\{(q, p_k)\}$ | Anchor positions flowing TO final position |
| $C_{ww}$ | All $(i,j)$ where $j < i$, excluding $C_{wp}$ and $C_{pq}$ | Other causal attention pairs |

Where:
- $p_k$ = anchor word positions (e.g., "Positive", "Negative")
- $q$ = final position (prediction position)

### Statistics Formula

For each set $C$:

$$S_C = \frac{1}{|C|} \sum_{(i,j) \in C} I_l(i,j)$$

### Interpretation

| Statistic | High Value Indicates |
|-----------|---------------------|
| $S_{wp}$ | Anchor words aggregate information from preceding text |
| $S_{pq}$ | Final position attends strongly to anchor words for prediction |
| $S_{ww}$ | Baseline attention strength between other positions |

## Usage Examples

### Basic Usage with Gradient Saliency (Default)

```python
from experiments.attribution import (
    AttentionExtractor,
    compute_all_layer_statistics,
)

# Extract attention with gradients
extractor = AttentionExtractor(model, "gpt2-xl")
graphs, gradients = extractor.extract_with_gradients(input_ids, labels)

# Compute statistics (auto-selects GradientSaliencyFlow when gradients provided)
stats = compute_all_layer_statistics(graphs, anchor_positions, gradients=gradients)
```

### Using Attention Sum

```python
from experiments.attribution import AttentionSumFlow

graphs = extractor.extract(input_ids)
flow_computer = AttentionSumFlow()
stats = compute_all_layer_statistics(
    graphs, anchor_positions,
    flow_computer=flow_computer
)
```

### Using Attention Rollout

```python
from experiments.attribution import AttentionRolloutFlow

# Rollout needs to accumulate across layers
rollout = AttentionRolloutFlow(residual_weight=0.5)
rollout.reset_cache()  # Reset for new sample

# Process layers in order
all_weights = [g.weights for g in graphs]
for i, graph in enumerate(graphs):
    stat = compute_anchor_statistics(
        graph, anchor_positions,
        flow_computer=rollout,
        layer_idx=i,
        all_layer_weights=all_weights,
    )
```

### Using Attention × Value Weighted

```python
from experiments.attribution import AttentionValueWeightedFlow

flow_computer = AttentionValueWeightedFlow()
stat = compute_anchor_statistics(
    graph, anchor_positions,
    flow_computer=flow_computer,
    value_states=value_states,  # [batch, heads, seq, head_dim]
)
```

### Using Causal Ablation (Slow but Accurate)

```python
from experiments.attribution import CausalAblationFlow

# chunk_size controls memory/speed tradeoff
flow_computer = CausalAblationFlow(chunk_size=32)
stat = compute_anchor_statistics(
    graph, anchor_positions,
    flow_computer=flow_computer,
    hidden_states=hidden_states,  # Input to layer
    layer_module=model.transformer.h[layer_idx],  # Layer module
    attention_mask=attention_mask,
)
```

### Using Attention Mask Ablation

```python
from experiments.attribution import AttentionMaskAblationFlow

flow_computer = AttentionMaskAblationFlow(chunk_size=32)
stat = compute_anchor_statistics(
    graph, anchor_positions,
    flow_computer=flow_computer,
    hidden_states=hidden_states,
    layer_module=model.transformer.h[layer_idx],
    attention_mask=attention_mask,
)
```

### Using Factory Function

```python
from experiments.attribution import FlowMetric, get_flow_computer

# Create flow computer from enum
flow_computer = get_flow_computer(
    FlowMetric.ATTENTION_ROLLOUT,
    residual_weight=0.5,
)

# Or for ablation methods
flow_computer = get_flow_computer(
    FlowMetric.CAUSAL_ABLATION,
    chunk_size=64,
)
```

## Module Structure

```
experiments/attribution/
├── __init__.py          # Public API exports
├── README.md            # This file
├── extractor.py         # AttentionExtractor, AttentionBipartiteGraph
├── statistics.py        # InformationFlowComputer implementations, statistics computation
└── run.py               # AttributionExperiment, run_attribution_experiment()
```

## Running Experiments

```bash
# Run attribution analysis with gradient saliency (default)
python scripts/run_attribution.py \
    --task sst2 \
    --model gpt2-xl \
    --shot 1 \
    --sample-size 100 \
    --device cuda:0 \
    --save-dir results/attribution

# Visualize results
python visualization/plot_attribution.py \
    --results results/attribution/sst2_gpt2-xl_shot1_seeds42_43_44.pkl \
    --output visualization/sst2_attribution.png
```

## Performance Considerations

| Metric | Time Complexity | Memory | Recommended Use |
|--------|----------------|--------|-----------------|
| Attention Sum | O(1) | Low | Quick analysis, baseline |
| Gradient Saliency | O(backward) | Medium | Paper-faithful attribution |
| Attention Rollout | O(L) | Low | Multi-layer flow analysis |
| Attention × Value | O(1) | Low | Information magnitude weighting |
| Causal Ablation | O(N × forward) | High | Ground truth, small sequences |
| Attention Mask Ablation | O(N × forward) | High | Attention-specific causality |

Where:
- L = number of layers
- N = sequence length
- forward = single layer forward pass time

## References

- Wang, L., et al. "Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning." EMNLP 2023.
- Abnar, S., & Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020.
