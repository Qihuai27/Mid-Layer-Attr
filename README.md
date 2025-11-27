# ICL Anchor Analysis

Analysis of anchor word effects in In-Context Learning (ICL).

## Overview

This repository provides a clean implementation for studying how **anchor words** (label words like "Positive", "Negative") affect information flow in transformers during in-context learning.

### Key Research Questions

1. **Attribution Analysis**: How does attention flow to/from anchor words across layers?
2. **Causal Ablation**: What happens when we block information flow from anchor words?

## Project Structure

```
icl-anchor-analysis/
├── configs/                    # Configuration files
│   ├── tasks/                  # Task definitions (sst2, agnews, etc.)
│   └── experiments/            # Experiment configs
│
├── src/                        # Core library
│   ├── anchors/                # Anchor word detection
│   ├── models/                 # Model wrappers and attention hooks
│   ├── data/                   # Data loading and ICL wrapping
│   └── utils/                  # Utilities
│
├── experiments/                # Experiment implementations
│   ├── attribution/            # Attention attribution analysis
│   └── ablation/               # Causal ablation experiments
│
├── scripts/                    # Run scripts
│   ├── run_attribution.py      # Run attribution analysis
│   ├── run_ablation.py         # Run ablation experiment
│   └── batch_experiments.py    # Generate batch scripts
│
├── notebooks/                  # Analysis notebooks
└── results/                    # Experiment results
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd icl-anchor-analysis

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Attribution Analysis

Analyze how attention flows through anchor words:

```bash
python scripts/run_attribution.py \
    --task sst2 \
    --model gpt2-xl \
    --shot 1 \
    --sample-size 100
```

### Causal Ablation

Test the causal role of anchor words by masking their attention:

```bash
python scripts/run_ablation.py \
    --task sst2 \
    --model gpt2-xl \
    --shot 1 \
    --mask-layers 5 \
    --mask-pos first
```

### Batch Experiments

Generate scripts for running multiple experiments:

```bash
python scripts/batch_experiments.py \
    --experiment both \
    --tasks sst2 agnews trec emo \
    --gpus 0 1 2 3

# Then run all experiments
bash scripts/batch/run_all.sh
```

## Supported Tasks

| Task | Classes | Description |
|------|---------|-------------|
| sst2 | 2 | Sentiment analysis (Positive/Negative) |
| agnews | 4 | News classification (World/Sports/Business/Technology) |
| trec | 6 | Question type classification |
| emo | 4 | Emotion classification (Others/Happy/Sad/Angry) |

## Supported Models

- GPT-2 XL (`gpt2-xl`)
- GPT-J 6B (`EleutherAI/gpt-j-6b`)

## Key Components

### Anchor Detection (`src/anchors/`)

Unified anchor word position detection using bigram matching:
```python
from src.anchors import AnchorDetector, load_task_config

task_config = load_task_config("sst2")
detector = AnchorDetector(task_config, tokenizer)
positions = detector.detect(input_ids)
```

### Attention Hooks (`src/models/hooks/`)

Modular system for intercepting and modifying attention:
```python
from src.models.hooks import get_hook_manager, HookMode

manager = get_hook_manager(model, "gpt2-xl")
manager.enable_observation()  # Capture attention weights
# or
manager.enable_intervention()  # Modify attention weights
```

### Experiments

**Attribution** (`experiments/attribution/`):
- Extract attention weights per layer
- Compute statistics: S_wp (anchor→tokens), S_pq (final→anchor), S_ww (other)

**Ablation** (`experiments/ablation/`):
- Apply interventions (mask anchor/non-anchor attention)
- Measure accuracy drop

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{wang2023label,
  title={Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning},
  author={...},
  journal={...},
  year={2023}
}
```
