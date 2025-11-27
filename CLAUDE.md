# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ICL Anchor Analysis is a research project studying how anchor words (label words like "Positive", "Negative") affect information flow in transformer models during in-context learning (ICL). Two core experiments:
- **Attribution Analysis**: Understand attention flow to/from anchor words across layers
- **Causal Ablation**: Test anchor word importance by blocking their information flow

Supported tasks: SST2 (sentiment), AG News (topic), TREC (question type), Emotion
Supported models: GPT-2 XL, GPT-J 6B

## Common Commands

```bash
# Installation
pip install -e .                    # Basic install
pip install -e ".[dev]"             # With pytest, black, isort
pip install -e ".[viz]"             # With matplotlib, seaborn

# Run attribution analysis
python scripts/run_attribution.py \
    --task sst2 \
    --model gpt2-xl \
    --shot 1 \
    --sample-size 100 \
    --device cuda:0 \
    --save-dir results/attribution

# Run ablation experiment
python scripts/run_ablation.py \
    --task sst2 \
    --model gpt2-xl \
    --shot 1 \
    --mask-layers 5 \
    --mask-pos first \
    --intervention anchor \
    --device cuda:0

# Batch experiments across tasks/GPUs
python scripts/batch_experiments.py \
    --experiment both \
    --tasks sst2 agnews trec emo \
    --gpus 0 1 2 3

# Run generated batch scripts
bash scripts/batch/run_all.sh
```

## Architecture

### Core Components

**Anchor Detection** (`src/anchors/`): Uses polynomial-encoded bigram matching for efficient anchor position detection. `AnchorDetector` finds label word positions in tokenized sequences.

**Hook System** (`src/models/hooks/`): Non-invasive attention interception via registered hooks:
- `HookManager`: Abstract base for model-specific implementations
- Three modes: OBSERVE (capture weights), INTERVENE (modify weights), DISABLED
- `GPT2HookManager` and `GPTJHookManager` for model-specific layer access

**Data Pipeline** (`src/data/`):
- `ICLWrapper`: Constructs in-context learning prompts from demonstrations + test examples
- `DataLoader`: Loads datasets from HuggingFace
- `DemonstrationSampler`: Samples few-shot demonstrations

**Model Wrappers** (`src/models/base.py`): `LMWrapper` provides unified interface for forward passes, logit extraction, and label probability computation.

### Experiment Structure

Both experiments follow: Load config → Load model → Load dataset → Sample demonstrations → Process examples → Run core logic → Save results

**Attribution** (`experiments/attribution/`):
- `AttentionExtractor`: Captures attention weights per layer
- `AnchorStatistics`: Computes attention flow statistics:
  - S_wp: Attention FROM anchor positions TO previous tokens (anchor as query)
  - S_pq: Attention FROM final position TO anchor positions
  - S_ww: Baseline attention for non-anchor, non-final positions

**Ablation** (`experiments/ablation/`):
- `AttentionIntervention` subclasses: `MaskAnchorAttention`, `MaskNonAnchorAttention`, `MaskLayerAttention`
- `AblationEvaluator`: Measures accuracy drop from baseline

### Configuration

Task configs (`configs/tasks/*.yaml`): Define dataset, labels, anchor patterns, prompt templates
Experiment configs (`configs/experiments/*.yaml`): Default hyperparameters, statistics, interventions

## Key Design Patterns

- **Lazy loading**: Models/tokenizers loaded on first access via properties
- **Hook-based interception**: Attention modification through registered hooks, not model changes
- **Polynomial encoding**: `id + prev_id * 100000 + prev_prev_id * 100000^2` for fast pattern matching
- **Factory pattern**: `get_hook_manager()` returns model-specific implementations
- **Context managers**: `temp_seed()` for temporary random state isolation

## Extension Points

**Add new task**: Create `configs/tasks/newtask.yaml` with labels, anchor_prefix, and prompt template

**Add new model**: Create hook manager subclass in `src/models/hooks/`, register in `get_hook_manager()` factory

**Add new intervention**: Subclass `AttentionIntervention`, implement `get_mask_fn()` and `describe()`

## Memory Considerations

- GPT2-XL: ~12GB GPU memory
- GPT-J: ~24GB GPU memory
- Batch size is 1 by default; reduce `--sample-size` if needed
