#!/usr/bin/env python3
"""Simple test to verify evaluation works"""

import sys
sys.path.insert(0, '/mnt/hdd16t/workspace/anchor')

import json
import numpy as np
import torch
from pathlib import Path

from src.models import load_model_and_tokenizer
from src.eval import (
    noise_insertion_auc_token,
    representation_insertion_auc_token,
)

print("="*80)
print("Loading model: Qwen3-4B")
print("="*80)

model, tokenizer = load_model_and_tokenizer(
    'Qwen3-4B',
    device='cuda',
    local_path='model/Qwen3-4B'
)

print("\nModel loaded successfully!")

# Load data
result_path = 'results/attribution/scores/Qwen3-4B/attention_rollout_ioi.json'
print(f"\nLoading: {result_path}")

with open(result_path) as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print("\nEvaluating first 10 samples...")

p_aucs = []
r_aucs = []

for i, sample in enumerate(data[:10]):
    print(f"\nSample {i}:")
    prompt = sample["prompt"]
    scores = np.array(sample["attribution_scores"])
    target_pos = sample["target_pos"]
    target_token_id = sample["target_token_id"]

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    print(f"  Length: {input_ids.shape[1]}, Target pos: {target_pos}")

    # Noise insertion AUC
    try:
        result = noise_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            target_pos=target_pos,
            target_token_id=target_token_id,
            steps=5,
            baseline_embed_mode="mean",
        )
        p_auc = result.auc
        p_aucs.append(p_auc)
        print(f"  P-AUC: {p_auc:.4f}")
    except Exception as e:
        print(f"  P-AUC Error: {e}")

    # Recovery AUC
    try:
        result = representation_insertion_auc_token(
            model=model,
            input_ids=input_ids,
            importance_scores=scores,
            base_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            steps=5,
            layer_index=-1,
            distance_mode="cosine",
            position_weighting="linear",
        )
        r_auc = result.auc
        r_aucs.append(r_auc)
        print(f"  R-AUC: {r_auc:.4f}")
    except Exception as e:
        print(f"  R-AUC Error: {e}")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Perturbation AUC: {np.mean(p_aucs):.4f} +/- {np.std(p_aucs):.4f}")
print(f"Recovery AUC: {np.mean(r_aucs):.4f} +/- {np.std(r_aucs):.4f}")
print("="*80)
