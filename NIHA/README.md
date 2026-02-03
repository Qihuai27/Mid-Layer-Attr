# NIHA: NIAH Single-Token Dataset

This folder contains a generator for a *Needle-in-a-Haystack* (NIAH) long-context stress test where the secret answer is constrained to be **exactly one tokenizer token in-context**.

Pre-generated dataset in this repo: `NIAH/niah_single_token.jsonl` (tokenizer `model/Qwen3-4B`, `seed=1234`, `target_len=2048`, `num_examples=100`).

## Generate (default)

```bash
python NIHA/generate_niah_single_token.py
```

Outputs: `NIAH/niah_single_token.jsonl` (plus `NIAH/niah_single_token.stats.json`)

## Common options

```bash
python NIHA/generate_niah_single_token.py \
  --tokenizer_name_or_path model/Qwen3-4B \
  --corpus_paths datasets \
  --num_examples 100 \
  --target_len 2048 \
  --seed 1234 \
  --output_path NIAH/niah_single_token.jsonl
```

## Optional: correctness filtering (verification)

If you have a local causal LM checkpoint available, you can keep only examples where the model’s **next-token argmax** matches `answer_token_id`:

```bash
python NIHA/generate_niah_single_token.py \
  --verify_model_name_or_path model/Qwen3-4B \
  --verify_device cpu
```

## Output fields (JSONL)

Each line is one example with:

- `prompt_text` and `prompt_token_ids` (no special tokens)
- exact token-span indices: `needle_start`, `needle_end`, `query_start`
- metadata: `target_len`, `depth`, `answer_text`, `answer_token_id`, etc.
