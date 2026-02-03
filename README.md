# Text Attribution Framework for LLMs

A comprehensive framework for computing and evaluating **token-level attribution** in Large Language Models. This project provides multiple attribution methods and rigorous evaluation metrics to understand which input tokens contribute most to model predictions.

## Features

- **Multiple Attribution Methods**
  - DePass (Decomposed Forward Pass) - traces token contributions through transformer layers
  - Attention Rollout - cumulative attention flow across layers
  - Integrated Gradients - path-integrated gradients from baseline to input
  - MidLayer Attribution - two-stage mid-layer attribution
  - Input Causal - causal intervention-based attribution

- **Rigorous Evaluation Metrics**
  - **Perturbation AUC (P-AUC)** - measures prediction degradation when corrupting important tokens (necessity test, lower is better)
  - **Recovery AUC (R-AUC)** - measures hidden state recovery when restoring tokens (sufficiency test, higher is better)
  - Support for both token-level and unit-level (word/sentence) evaluation

- **Multi-Model Support**
  - LLaMA / LLaMA-2 / LLaMA-3
  - Qwen / Qwen2 / Qwen3
  - Phi-2
  - GPT-2 / GPT-J

## Installation

```bash
git clone <repo-url>
cd anchor

pip install -r requirements.txt
# Or install as package
pip install -e .
```

## Quick Start

### One-Stop Attribution Pipeline

The easiest way to run attribution experiments:

```bash
# Run all methods on all datasets (full dataset)
python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B

# Run specific methods and datasets
python scripts/attribution/run_attribution_pipeline.py --model phi-2 \
    --methods attention_rollout depass --datasets ioi counterfact

# Quick test with limited samples
python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B \
    --max-samples 50 --eval-samples 50

# Resume from existing scores
python scripts/attribution/run_attribution_pipeline.py --model Qwen3-4B --skip-existing
```

### Computing Attribution Scores (Python API)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.attribution import DePass, AttentionRollout, IntegratedGradients

# Load model
model = AutoModelForCausalLM.from_pretrained("path/to/model", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Prepare input
text = "The capital of France is"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
target_pos = input_ids.shape[1] - 1  # Last position

# Compute attribution with DePass
depass = DePass(mlp_softmax_temp=0.1)
result = depass.attribute(model, input_ids, target_pos=target_pos)

print(f"Attribution scores: {result.scores}")
print(f"Top-3 important tokens: {result.get_top_k_indices(3)}")
```

### Evaluating Attribution Quality

```python
from src.eval import noise_insertion_auc_token, representation_insertion_auc_token

# Get next token prediction
with torch.no_grad():
    logits = model(input_ids).logits
    next_token_id = logits[0, -1].argmax().item()

# Evaluate with Perturbation AUC (lower is better)
p_result = noise_insertion_auc_token(
    model=model,
    input_ids=input_ids,
    importance_scores=result.scores,
    target_pos=target_pos,
    target_token_id=next_token_id,
    steps=10,
)
print(f"Perturbation AUC: {p_result.auc:.4f}")  # Lower is better

# Evaluate with Recovery AUC (higher is better)
r_result = representation_insertion_auc_token(
    model=model,
    input_ids=input_ids,
    importance_scores=result.scores,
    base_token_id=tokenizer.pad_token_id,
    steps=10,
    position_weighting="linear",  # Account for causal masking
)
print(f"Recovery AUC: {r_result.auc:.4f}")  # Higher is better
```

## Project Structure

```
anchor/
├── src/
│   ├── attribution/          # Attribution methods
│   │   ├── depass.py         # DePass: Decomposed Forward Pass
│   │   ├── attention_rollout.py  # Attention Rollout
│   │   ├── integrated_gradients.py  # Integrated Gradients
│   │   ├── midlayer.py       # Mid-layer attribution
│   │   └── base.py           # Base classes
│   │
│   ├── eval/                 # Evaluation metrics
│   │   ├── metrics_token.py  # Token-level AUC metrics
│   │   ├── metrics_units.py  # Unit-level AUC metrics
│   │   └── segmentation.py   # Token-to-unit segmentation
│   │
│   ├── models/               # Model utilities
│   │   ├── base.py           # Model loading
│   │   └── hooks/            # Attention hooks (LLaMA, Qwen, Phi, GPT-2, etc.)
│   │
│   ├── cluster/              # Cluster analysis (for ICL research)
│   ├── data/                 # Data loading and ICL wrapping
│   └── visualization/        # Plotting utilities
│
├── scripts/
│   └── attribution/
│       ├── run_attribution_pipeline.py  # One-stop pipeline (recommended)
│       ├── compare_methods.py           # Compare multiple methods
│       └── eval_attribution.py          # Evaluate saved scores
│
├── datasets/                 # Evaluation datasets (IOI, CounterFact, LongRA)
├── configs/                  # Configuration files
└── results/                  # Output directory
```

## Attribution Methods

### DePass (Decomposed Forward Pass)

DePass traces token contributions by decomposing the forward pass through transformer layers:

```python
from src.attribution import DePass

method = DePass(
    mlp_softmax_temp=0.1,       # Temperature for MLP decomposition
    mlp_decomposed_function="softmax",  # or "taylor"
)
result = method.attribute(model, input_ids, target_pos=target_pos)
```

**Key idea**: Initialize attribution state at each token position, then propagate through attention and MLP layers to track how each input token contributes to the final hidden states.

### Attention Rollout

Computes cumulative attention flow across all layers:

```python
from src.attribution import AttentionRollout

method = AttentionRollout(
    residual_weight=0.5,        # Weight for residual connections
    head_aggregation="mean",    # How to aggregate attention heads
)
result = method.attribute(model, input_ids, target_pos=target_pos)
```

### Integrated Gradients

Path-integrated gradients from baseline to input:

```python
from src.attribution import IntegratedGradients

method = IntegratedGradients(
    n_steps=50,                 # Number of interpolation steps
    baseline_type="zero",       # "zero", "pad", or "mean"
)
result = method.attribute(model, input_ids, target_pos=target_pos, target_token_id=next_token_id)
```

## Evaluation Metrics

### Perturbation AUC (P-AUC) - Lower is Better

Measures how prediction degrades when corrupting tokens in order of importance (necessity test):

```python
from src.eval import noise_insertion_auc_token

result = noise_insertion_auc_token(
    model=model,
    input_ids=input_ids,
    importance_scores=scores,      # From attribution method
    target_pos=target_pos,
    target_token_id=target_token_id,
    steps=10,
    baseline_embed_mode="mean",    # "mean", "zero", or "gaussian"
)
```

**Interpretation**: If the attribution correctly identifies important tokens, corrupting them first should rapidly degrade the prediction, resulting in low AUC.

### Recovery AUC (R-AUC) - Higher is Better

Measures how hidden states recover when restoring tokens in order of importance (sufficiency test):

```python
from src.eval import representation_insertion_auc_token

result = representation_insertion_auc_token(
    model=model,
    input_ids=input_ids,
    importance_scores=scores,
    base_token_id=tokenizer.pad_token_id,
    steps=10,
    distance_mode="cosine",        # "cosine" or "frobenius"
    position_weighting="linear",   # "linear" or "uniform"
)
```

**Position Weighting**: Use `"linear"` to weight later positions more heavily, accounting for causal masking where later positions aggregate information from all previous tokens.

### Unit-Level Evaluation

Evaluate at word or sentence granularity:

```python
from src.eval import build_units, noise_insertion_auc_units

# Segment tokens into words
seg_result = build_units(input_ids[0], tokenizer, mode="bpe_word")

# Evaluate at word level
result = noise_insertion_auc_units(
    model=model,
    input_ids=input_ids,
    token_importance_scores=scores,
    units=seg_result.units,
    target_pos=target_pos,
    target_token_id=target_token_id,
)
```

## Benchmark Results

Results on Qwen3-4B with the prompt *"In the heart of Europe, the country of France has a beautiful capital city known for the Eiffel Tower. The name of this famous city is"* (predicting "Paris"):

| Method | Perturbation AUC (↓) | Recovery AUC (↑) |
|--------|:-------------------:|:---------------------------:|
| DePass | **0.2492** | 0.2859 |
| Integrated Gradients | 0.4956 | **0.3946** |
| Attention Rollout | 0.5669 | 0.1820 |
| Random | 0.4799 | 0.2870 |

**Key findings**:
- DePass excels at Perturbation AUC, identifying tokens most critical for prediction
- Integrated Gradients performs best on Recovery AUC
- Attention Rollout performs poorly due to attention sink on BOS token
- Linear position weighting is crucial for meaningful Recovery AUC evaluation

## Supported Models

| Model Family | Tested Models | Notes |
|--------------|---------------|-------|
| LLaMA | LLaMA-3-8B | Full support |
| Qwen | Qwen3-4B | Full support including GQA |
| Phi | Phi-2 | Full support (uses float32 for stability) |
| GPT-2 | gpt2, gpt2-xl | Full support |
| GPT-J | gpt-j-6b | Full support |

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/attribution/run_attribution_pipeline.py` | **Recommended**: One-stop pipeline for scoring + evaluation |
| `scripts/attribution/compare_methods.py` | Compare multiple methods on datasets |
| `scripts/attribution/eval_attribution.py` | Evaluate pre-computed attribution scores |
| `scripts/batch/run_all.py` | Batch experiments across models |

## Citation

```bibtex
@article{depass2025,
  title={DePass: Unified Feature Attributing by Simple Decomposed Forward Pass},
  author={...},
  journal={NeurIPS},
  year={2025}
}
```

## License

MIT License
