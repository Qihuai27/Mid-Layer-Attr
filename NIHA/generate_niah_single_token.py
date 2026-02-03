#!/usr/bin/env python3
"""
Generate a Needle-in-a-Haystack (NIAH) single-token JSONL dataset.

This follows the procedure described in the user-provided LaTeX:
- sample a single-token answer under a HuggingFace tokenizer *in context*
- sample a contiguous haystack span in tokenizer-token space (length L)
- insert a fixed needle sentence at a controlled depth
- export exact token-level spans for the needle and query
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


SYSTEM_TEXT = "You are a helpful assistant."
QUERY_TEXT = "Question: What is the password? Answer with one word: "
NEEDLE_PREFIX = "IMPORTANT: The password is "
NEEDLE_SUFFIX = "."


_ALNUM_RE = re.compile(r"^[A-Za-z0-9]{2,12}$")


@dataclass(frozen=True)
class NIAHExample:
    id: str
    seed: int
    model_name_or_path: str
    target_len: int
    depth: float
    haystack_len_before: int
    insert_pos_before: int
    answer_text: str
    answer_token_id: int
    prompt_text: str
    prompt_token_ids: list[int]
    needle_start: int
    needle_end: int
    query_start: int


def _has_control_chars(text: str) -> bool:
    for ch in text:
        if ch in {"\n", "\r", "\t"}:
            return True
        if ord(ch) < 32:
            return True
    return False


def _is_visually_meaningful_candidate(text: str) -> bool:
    if not text:
        return False
    if "\uFFFD" in text:
        return False
    # Allow leading whitespace (common in BPE vocab like GPT2/LLaMA),
    # but disallow trailing whitespace or pure-whitespace tokens.
    if text.strip() == "":
        return False
    if text.endswith(" "):
        return False
    if _has_control_chars(text):
        return False
    if not _ALNUM_RE.match(text.lstrip()):
        return False
    return True


def _is_single_token_in_context(tokenizer, answer_text: str, answer_token_id: int) -> bool:
    """
    Enforce the in-context single-token constraint under the fixed needle template.

    Key idea: tokenization must be boundary-stable:
    - prefix and prefix+answer must share the exact prefix tokenization (no cross-boundary merges)
    - the answer must add exactly one token, equal to answer_token_id
    - adding the trailing '.' must not change the answer tokenization
    - the answer token id must appear exactly once in the full needle
    """
    prefix_ids = tokenizer.encode(NEEDLE_PREFIX, add_special_tokens=False)
    prefix_plus_answer_ids = tokenizer.encode(
        NEEDLE_PREFIX + answer_text, add_special_tokens=False
    )
    if prefix_plus_answer_ids[: len(prefix_ids)] != prefix_ids:
        return False
    added = prefix_plus_answer_ids[len(prefix_ids) :]
    if len(added) != 1 or added[0] != answer_token_id:
        return False

    needle_text = NEEDLE_PREFIX + answer_text + NEEDLE_SUFFIX
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    if needle_ids[: len(prefix_ids)] != prefix_ids:
        return False
    if len(needle_ids) <= len(prefix_ids) or needle_ids[len(prefix_ids)] != answer_token_id:
        return False
    if needle_ids.count(answer_token_id) != 1:
        return False
    return True


def _collect_corpus_text(
    corpus_paths: Sequence[Path],
    *,
    include_exts: set[str],
    max_file_bytes: int,
) -> str:
    files: list[Path] = []
    for p in corpus_paths:
        if p.is_file():
            files.append(p)
            continue
        if p.is_dir():
            for fp in p.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in include_exts:
                    continue
                files.append(fp)

    files = sorted(set(files))
    if not files:
        raise ValueError("No corpus files found under: " + ", ".join(map(str, corpus_paths)))

    chunks: list[str] = []
    for fp in files:
        try:
            if fp.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        try:
            chunks.append(fp.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    if not chunks:
        raise ValueError("Corpus files were found, but none could be read (or all exceeded size limit).")
    return "\n".join(chunks)


def _choose_haystack_start_positions(tokenizer) -> list[int]:
    """
    Prefer haystack spans that begin at a token whose decode starts with whitespace,
    so the initial system->haystack boundary is visually separated without adding tokens.
    """
    good: list[int] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if piece and piece[0].isspace():
            good.append(token_id)
    return good


def _sample_answer(tokenizer, rng: random.Random, *, max_tries: int) -> tuple[int, str]:
    for _ in range(max_tries):
        token_id = rng.randrange(len(tokenizer))
        answer_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if not _is_visually_meaningful_candidate(answer_text):
            continue
        if not _is_single_token_in_context(tokenizer, answer_text, token_id):
            continue
        return token_id, answer_text
    raise RuntimeError(f"Failed to sample a valid single-token answer after {max_tries} tries.")


def _sample_haystack_span(
    corpus_token_ids: Sequence[int],
    rng: random.Random,
    *,
    target_len: int,
    preferred_first_token_ids: Sequence[int] | None,
) -> list[int]:
    if len(corpus_token_ids) < target_len:
        raise ValueError(
            f"Corpus too short in token space: {len(corpus_token_ids)} < target_len={target_len}."
        )

    max_start = len(corpus_token_ids) - target_len
    if max_start <= 0:
        return list(corpus_token_ids[:target_len])

    if preferred_first_token_ids:
        # Try a few times to find a span that starts with a visually separated token.
        for _ in range(20):
            start = rng.randint(0, max_start)
            if corpus_token_ids[start] in preferred_first_token_ids:
                return list(corpus_token_ids[start : start + target_len])

    start = rng.randint(0, max_start)
    return list(corpus_token_ids[start : start + target_len])


def _verify_next_token_argmax(
    model,
    *,
    prompt_token_ids: Sequence[int],
    answer_token_id: int,
) -> bool:
    import torch

    input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # [1, seq, vocab]
    pred = int(torch.argmax(logits[0, -1]).item())
    return pred == int(answer_token_id)


def generate_dataset(
    *,
    tokenizer_name_or_path: str,
    corpus_paths: Sequence[Path],
    output_path: Path,
    num_examples: int,
    target_len: int,
    depths: Sequence[float],
    seed: int,
    max_resamples: int,
    max_answer_tries: int,
    include_exts: set[str],
    max_file_bytes: int,
    local_files_only: bool,
    verify_model_name_or_path: str | None,
    verify_device: str,
) -> None:
    stats = Counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    # This generator deliberately works with long token sequences (e.g., tokenized corpora),
    # so avoid spurious "sequence length > model_max_length" warnings from the tokenizer.
    tokenizer.model_max_length = 1_000_000_000

    model = None
    if verify_model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            verify_model_name_or_path,
            local_files_only=local_files_only,
        )
        model.to(verify_device)
        model.eval()

    corpus_text = _collect_corpus_text(
        corpus_paths, include_exts=include_exts, max_file_bytes=max_file_bytes
    )
    corpus_token_ids = tokenizer.encode(corpus_text, add_special_tokens=False)
    if len(corpus_token_ids) < target_len:
        raise ValueError(
            f"Corpus tokenized length ({len(corpus_token_ids)}) is shorter than target_len ({target_len})."
        )

    preferred_first_token_ids = _choose_haystack_start_positions(tokenizer)

    rng = random.Random(seed)
    # Add explicit separators to make prompt boundaries stable/readable.
    system_ids = tokenizer.encode(SYSTEM_TEXT + "\n\n", add_special_tokens=False)
    query_ids = tokenizer.encode("\n\n" + QUERY_TEXT, add_special_tokens=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(num_examples), desc="Generating NIAH examples"):
            example_id = f"niah-{i:06d}"
            for attempt in range(max_resamples):
                stats["total_attempts"] += 1
                depth = rng.choice(list(depths))
                answer_token_id, answer_text = _sample_answer(
                    tokenizer, rng, max_tries=max_answer_tries
                )

                # Prevent trivial collisions: answer token appears in system/query.
                if answer_token_id in system_ids or answer_token_id in query_ids:
                    stats["answer_token_in_system_or_query"] += 1
                    continue

                haystack = _sample_haystack_span(
                    corpus_token_ids,
                    rng,
                    target_len=target_len,
                    preferred_first_token_ids=preferred_first_token_ids,
                )
                if answer_token_id in haystack:
                    stats["answer_token_in_haystack_ids"] += 1
                    continue

                haystack_text = tokenizer.decode(haystack, clean_up_tokenization_spaces=False)
                if answer_text.lower() in haystack_text.lower():
                    stats["answer_in_haystack_text"] += 1
                    continue

                insert_pos_before = int(math.floor(depth * len(haystack)))
                haystack_prefix = haystack[:insert_pos_before]
                haystack_suffix = haystack[insert_pos_before:]

                needle_text = NEEDLE_PREFIX + answer_text + NEEDLE_SUFFIX
                needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)

                prompt_token_ids = (
                    system_ids + haystack_prefix + needle_ids + haystack_suffix + query_ids
                )

                needle_start = len(system_ids) + len(haystack_prefix)
                needle_end = needle_start + len(needle_ids)
                query_start = len(system_ids) + len(haystack_prefix) + len(needle_ids) + len(
                    haystack_suffix
                )

                prompt_text = tokenizer.decode(
                    prompt_token_ids, clean_up_tokenization_spaces=False
                )

                if model is not None:
                    try:
                        if not _verify_next_token_argmax(
                            model, prompt_token_ids=prompt_token_ids, answer_token_id=answer_token_id
                        ):
                            stats["verify_failed"] += 1
                            continue
                    except Exception as e:
                        raise RuntimeError(
                            "Verification failed during model forward pass. "
                            "Try a smaller --target_len or disable verification."
                        ) from e

                record = NIAHExample(
                    id=example_id,
                    seed=seed,
                    model_name_or_path=tokenizer_name_or_path,
                    target_len=target_len,
                    depth=float(depth),
                    haystack_len_before=len(haystack),
                    insert_pos_before=insert_pos_before,
                    answer_text=answer_text,
                    answer_token_id=int(answer_token_id),
                    prompt_text=prompt_text,
                    prompt_token_ids=[int(x) for x in prompt_token_ids],
                    needle_start=int(needle_start),
                    needle_end=int(needle_end),
                    query_start=int(query_start),
                )
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                stats["kept_examples"] += 1
                break
            else:
                stats["max_resample_exceeded"] += 1
                raise RuntimeError(
                    f"Failed to generate example {example_id} after max_resamples={max_resamples}."
                )

    # Write stats next to output jsonl.
    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as sf:
        json.dump(dict(stats), sf, ensure_ascii=False, indent=2)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=os.environ.get("NIAH_TOKENIZER", "model/Qwen3-4B"),
        help="HuggingFace tokenizer name or local path (default: model/Qwen3-4B).",
    )
    parser.add_argument(
        "--corpus_paths",
        type=str,
        nargs="+",
        default=["datasets"],
        help="One or more local files/directories used as the text corpus (default: ./datasets).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="NIAH/niah_single_token.jsonl",
        help="Output JSONL path (default: NIAH/niah_single_token.jsonl).",
    )
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--target_len", type=int, default=2048)
    parser.add_argument(
        "--depths",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated insertion depths (default: 0.1,0.3,0.5,0.7,0.9).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_resamples", type=int, default=500)
    parser.add_argument("--max_answer_tries", type=int, default=5000)
    parser.add_argument(
        "--verify_model_name_or_path",
        type=str,
        default=None,
        help="Optional: causal LM for correctness filtering (single forward pass).",
    )
    parser.add_argument(
        "--verify_device",
        type=str,
        default="cpu",
        help="Device for verification model (default: cpu).",
    )
    parser.add_argument(
        "--include_exts",
        type=str,
        default=".txt,.md,.py,.json,.yaml,.yml,.csv",
        help="Comma-separated file extensions to include for corpus collection.",
    )
    parser.add_argument(
        "--max_file_mb",
        type=int,
        default=20,
        help="Skip corpus files larger than this size (MB).",
    )
    parser.add_argument(
        "--allow_remote",
        action="store_true",
        help="Allow downloading the tokenizer from HuggingFace Hub (may require network approval).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    include_exts = {x.strip().lower() for x in args.include_exts.split(",") if x.strip()}
    corpus_paths = [Path(p) for p in args.corpus_paths]
    generate_dataset(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        corpus_paths=corpus_paths,
        output_path=Path(args.output_path),
        num_examples=int(args.num_examples),
        target_len=int(args.target_len),
        depths=depths,
        seed=int(args.seed),
        max_resamples=int(args.max_resamples),
        max_answer_tries=int(args.max_answer_tries),
        verify_model_name_or_path=args.verify_model_name_or_path,
        verify_device=str(args.verify_device),
        include_exts=include_exts,
        max_file_bytes=int(args.max_file_mb) * 1024 * 1024,
        local_files_only=not bool(args.allow_remote),
    )


if __name__ == "__main__":
    main()
