# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository focuses on two research directions:

1. **Cluster (信息汇聚研究)**: Studying the "Label Words are Anchors" phenomenon in In-Context Learning (ICL) - how cluster points (label words like "Positive"/"Negative") aggregate information from preceding text and serve as key positions for prediction.

2. **Attribution (自回归归因)**: Token-level input attribution for autoregressive models, proposing two new evaluation metrics:
   - **Perturbation AUC (P-AUC)** (necessity, lower is better): Measures prediction degradation when corrupting important tokens
   - **Recovery AUC (R-AUC)** (sufficiency, higher is better): Measures hidden state recovery when restoring important tokens

## Common Commands

### Installation
```bash
pip install -r requirements.txt
# Or as editable package
pip install -e .
```

### Running Cluster Experiments
```bash
# Default gradient saliency
python scripts/cluster/run_cluster.py --task sst2 --model gpt2-xl --shot 1

# With specific flow metric
python scripts/cluster/run_cluster.py --task sst2 --model gpt2-xl --flow-metric attention_rollout

# Compare pattern vs random clusters
python scripts/cluster/run_cluster_comparison.py --models phi-2 Qwen3-4B

# Available flow metrics: attention_sum, gradient_saliency, attention_rollout, attention_value_weighted
```

### Running Attribution Evaluation
```bash
# One-stop pipeline (recommended)
python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B

# Compare attribution methods
python scripts/attribution/compare_methods.py --model Qwen3-4B

# Evaluate attribution quality
python scripts/attribution/eval_attribution.py --model phi-2 --method depass
```

### Plotting Results
```bash
python scripts/plot_results.py --results results/cluster/raw/<result_file>.pkl
```

## Architecture

### Core Modules

**src/cluster/** - Cluster (information aggregation) analysis
- `definitions.py`: TaskConfig - Task configurations with labels, prompts, cluster patterns
- `detector.py`: ClusterDetector - Detects cluster point positions using bigram pattern matching
- `extractor.py`: AttentionExtractor - Extracts attention weights with optional gradients
- `statistics.py`: Information flow metrics (S_a, S_o, S_w) and ClusterStatistics
  - S_a (aggregation): Information flow from previous tokens TO cluster positions
  - S_o (output): Information flow from cluster positions TO final position
  - S_w (within): Information flow between other (non-cluster, non-final) positions

**src/attribution/** - Attribution method implementations
- `base.py`: Abstract base classes (`AttributionMethod`, `AttributionResult`) defining the interface
- `depass.py`: DePass (Decomposed Forward Pass) - traces token contributions through transformer layers
- `attention_rollout.py`: Cumulative attention flow across layers
- `integrated_gradients.py`: Path-integrated gradients from baseline to input
- `midlayer.py`, `midlayer_v2.py`: Two-stage mid-layer attribution methods
- `greedy_optimal.py`: Greedy optimal and input causal attribution

**src/eval/** - Novel evaluation metrics for attribution quality
- `metrics_token.py`: Token-level AUC metrics (perturbation AUC, recovery AUC)
- `metrics_units.py`: Unit-level (word/sentence) AUC metrics
- `segmentation.py`: Token-to-unit segmentation utilities

**src/models/hooks/** - Model-specific attention hooks
- Hook managers for extracting attention weights from different model architectures
- Use `get_hook_manager(model, model_name)` factory to get appropriate manager
- Supported: GPT-2, GPT-J, LLaMA, Qwen3, Phi-2

**experiments/** - Experiment infrastructure
- `cluster/run.py`: ClusterExperiment runner for information aggregation analysis
- `ablation/`: Causal ablation experiments

### Directory Structure

```
anchor/
├── src/
│   ├── cluster/           # Cluster (aggregation) analysis
│   ├── attribution/       # Attribution methods
│   ├── eval/              # Evaluation metrics
│   ├── models/hooks/      # Model-specific attention hooks
│   ├── data/              # Data loading and ICL wrapping
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Random seeds, I/O
│
├── experiments/
│   ├── cluster/           # Cluster experiment runners
│   └── ablation/          # Ablation experiments
│
├── scripts/
│   ├── cluster/           # Cluster analysis scripts
│   ├── attribution/       # Attribution evaluation scripts
│   ├── ablation/          # Ablation scripts
│   └── batch/             # Batch experiment scripts
│
├── configs/
│   ├── tasks/             # Task definitions (SST2, AGNews, TREC, EMO)
│   └── experiments/       # Experiment configurations
│
├── results/
│   ├── cluster/           # Cluster experiment results
│   │   ├── raw/           # Raw pickle files
│   │   └── figures/       # Generated plots
│   ├── attribution/       # Attribution results
│   │   ├── scores/        # Attribution scores
│   │   └── eval/          # Evaluation metrics
│   └── ablation/          # Ablation results
│
└── model/                 # Local model storage
```

### Key Patterns

**Cluster analysis follows this workflow:**
```python
from src.cluster import ClusterDetector, AttentionExtractor, compute_cluster_statistics

detector = ClusterDetector(task_config, tokenizer)
extractor = AttentionExtractor(model, model_name)

positions = detector.detect(input_ids)
graphs = extractor.extract(input_ids)
stats = compute_cluster_statistics(graph, positions)
# stats.S_a (aggregation), stats.S_o (output), stats.S_w (within)
```

**Attribution methods follow this interface:**
```python
from src.attribution import DePass

method = DePass(mlp_softmax_temp=0.1)
result = method.attribute(model, input_ids, target_pos=-1)
# result.scores: np.ndarray of token importance scores
# result.get_top_k_indices(k): indices of top-k important tokens
```

**Evaluation integrates directly with attribution:**
```python
from src.attribution import attention_rollout
from src.eval import noise_insertion_auc_token

scores = attention_rollout(model, input_ids, target_pos=-1)
# Perturbation AUC (lower is better)
auc = noise_insertion_auc_token(model, input_ids, scores, target_pos, target_token_id)
```

### Adding New Components

**New attribution methods:** Inherit from `AttributionMethod` (or `AttentionBasedMethod`/`GradientBasedMethod`/`PerturbationBasedMethod`) and implement `attribute()`.

**New hook managers:** Inherit from `HookManager` in `src/models/hooks/base.py` and register in `get_hook_manager()`.

**New flow metrics:** Inherit from `InformationFlowComputer` in `src/cluster/statistics.py`.

## Evaluation Metrics (Novel Contributions)

- **Perturbation AUC (P-AUC)** (lower is better): Tests necessity - does removing token information hurt prediction?
- **Recovery AUC (R-AUC)** (higher is better): Tests sufficiency - does restoring token recover hidden states?
- Use `position_weighting="linear"` for Recovery AUC to account for causal masking
- Design choice: Uses hidden states rather than next-token prediction to capture full sequence information aggregation

## Model Notes

- Phi-2 requires `torch.float32` for numerical stability (auto-handled by `load_model_and_tokenizer`)
- Models require `attn_implementation="eager"` for attention extraction with custom hooks
- Local models should be placed in `model/` directory (e.g., `model/phi-2/`, `model/Qwen3-4B/`)

## Backward Compatibility

The old `src.anchors` module is deprecated but still available for backward compatibility:
- `AnchorDetector` → `ClusterDetector`
- `AnchorPositions` → `ClusterPositions`
- `AnchorStatistics` → `ClusterStatistics`
- `S_wp` → `S_a` (aggregation)
- `S_pq` → `S_o` (output)
- `S_ww` → `S_w` (within)

Using `from src.anchors import ...` will trigger a deprecation warning.
