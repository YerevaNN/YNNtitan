#!/usr/bin/env python3
"""Estimate training-sequence count for a dataset and plan batch/steps for ~1 epoch."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

from torchtitan.tokenizers.tokenizer import build_tokenizer
from torchtitan.utils.dataset_utils import (
    chemlactica_style_data_processing,
    conformer_data_processing,
    pubchem_data_processing,
)

_supported_datasets = {
    "c4_test": "test/assets/c4_test",
    "c4": "allenai/c4",
    "chemlactica_train_mini": "test/assets/chemlactica_train_mini",
    "chemlactica_train": "/mnt/weka/gsimonyan/data/rdkit_computed_rel+form/train_rdkit_computed_rel+form",
    "conformers_train": "/auto/home/menuab/code/3DMolGen/data/pcqm/train",
    "conformers_valid": "/auto/home/menuab/code/3DMolGen/data/pcqm/valid",
    "chemlactica_valid": "/mnt/weka/gsimonyan/data/rdkit_computed_rel+form",
    "chemlactica_valid_mini": "test/assets/chemlactica_valid_mini",
}
_pubchem_dir = os.environ.get("PUBCHEM_DATA_DIR")
if _pubchem_dir:
    _supported_datasets["pubchem_train"] = f"{_pubchem_dir}/train_rdkit_computed_rel+form"
    _supported_datasets["pubchem_valid"] = _pubchem_dir

_supported_data_processing_styles = {
    "chemlactica_style": chemlactica_style_data_processing,
    "conformer_style": conformer_data_processing,
    "pubchem_data_processing": pubchem_data_processing,
}


def count_training_sequences(
    dataset_path: str,
    data_processing_style: str,
    tokenizer,
    seq_len: int,
    representation_type: str,
    seed: int | None = None,
    max_records: int | None = None,
    sample_fraction: float | None = None,
) -> dict:
    """Mirror HuggingFaceDataset tokenization + seq_len chunking."""
    data_processing_fn = _supported_data_processing_styles[data_processing_style]
    rng = np.random.default_rng(seed)

    files = sorted(glob.glob(os.path.join(dataset_path, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files under {dataset_path}")

    max_buffer_token_len = 1 + seq_len
    all_tokens: list[int] = []
    raw_records = 0
    skipped_records = 0
    total_tokens = 0
    training_sequences = 0

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if sample_fraction is not None and rng.random() >= sample_fraction:
                    continue
                if max_records is not None and raw_records >= max_records:
                    break
                if not line.strip():
                    continue

                raw_records += 1
                sample_text = data_processing_fn(line, rng, representation_type)
                if not sample_text:
                    skipped_records += 1
                    continue

                sample_tokens = tokenizer.encode(sample_text, bos=True, eos=True)
                total_tokens += len(sample_tokens)
                all_tokens.extend(sample_tokens)

                while len(all_tokens) >= max_buffer_token_len:
                    all_tokens = all_tokens[max_buffer_token_len:]
                    training_sequences += 1

        if max_records is not None and raw_records >= max_records:
            break

    return {
        "raw_records": raw_records,
        "skipped_records": skipped_records,
        "total_tokens": total_tokens,
        "training_sequences": training_sequences,
        "leftover_tokens": len(all_tokens),
        "jsonl_files": len(files),
    }


def suggest_batch_decomposition(
    global_batch: int, num_gpus: int
) -> list[tuple[int, int, int]]:
    """Return (batch_size, grad_accum, local_batch) triples that hit global_batch."""
    if global_batch % num_gpus != 0:
        return []

    local_batch = global_batch // num_gpus
    suggestions: list[tuple[int, int, int]] = []
    for grad_accum in (1, 2, 4, 8, 9, 10, 16):
        if local_batch % grad_accum == 0:
            batch_size = local_batch // grad_accum
            if batch_size >= 1:
                suggestions.append((batch_size, grad_accum, local_batch))
    return suggestions[:5]


def format_int(n: float) -> str:
    return f"{int(round(n)):,}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how many training sequences a dataset yields and plan "
            "global_batch_size x steps ≈ dataset size."
        )
    )
    parser.add_argument("--dataset", type=str, default="chemlactica_train_mini")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--data-processing-style",
        type=str,
        default="pubchem_data_processing",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="torchtitan/tokenizers/Llama-3.2-chem-1B-v2/",
    )
    parser.add_argument("--tokenizer-type", type=str, default="tiktoken")
    parser.add_argument("--representation-type", type=str, default="SMILES")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=None, help="Fixed training steps")
    parser.add_argument(
        "--global-batch",
        type=int,
        default=None,
        help="Fixed global batch size (batch_size x grad_accum x num_gpus)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Used only for per-GPU batch suggestions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for data processing (training uses an unseeded RNG by default)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after N raw JSONL records (quick test)",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Randomly sample this fraction of records for fast estimation",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not dataset_path:
        if args.dataset not in _supported_datasets:
            print(f"Unknown dataset {args.dataset!r}; pass --dataset-path", file=sys.stderr)
            return 1
        dataset_path = _supported_datasets[args.dataset]

    if not os.path.isdir(dataset_path):
        print(f"Dataset path not found: {dataset_path}", file=sys.stderr)
        return 1

    tokenizer = build_tokenizer(args.tokenizer_type, args.tokenizer_path)
    stats = count_training_sequences(
        dataset_path=dataset_path,
        data_processing_style=args.data_processing_style,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        representation_type=args.representation_type,
        seed=args.seed,
        max_records=args.max_records,
        sample_fraction=args.sample_fraction,
    )

    total_sequences = stats["training_sequences"]
    if args.sample_fraction is not None:
        scale = 1.0 / args.sample_fraction
        total_sequences_est = int(round(total_sequences * scale))
        print("=== Dataset estimate (sampled) ===")
        print(f"  dataset:              {args.dataset}")
        print(f"  path:                 {dataset_path}")
        print(f"  jsonl files:          {stats['jsonl_files']}")
        print(f"  sample fraction:      {args.sample_fraction}")
        print(f"  sampled raw records:  {format_int(stats['raw_records'])}")
        print(f"  sampled sequences:    {format_int(total_sequences)}")
        print(f"  estimated sequences:  {format_int(total_sequences_est)}  (× {scale:.2f})")
        total_sequences = total_sequences_est
    else:
        print("=== Dataset count ===")
        print(f"  dataset:              {args.dataset}")
        print(f"  path:                 {dataset_path}")
        print(f"  jsonl files:          {stats['jsonl_files']}")
        print(f"  raw records:          {format_int(stats['raw_records'])}")
        print(f"  skipped records:      {format_int(stats['skipped_records'])}")
        print(f"  total tokens:         {format_int(stats['total_tokens'])}")
        print(f"  training sequences:   {format_int(total_sequences)}  (seq_len={args.seq_len})")
        print(f"  leftover tokens:      {format_int(stats['leftover_tokens'])}")

    print()
    print("=== Training plan (global_batch × steps ≈ dataset sequences) ===")

    if args.steps is not None:
        optimal_global_batch = max(1, math.ceil(total_sequences / args.steps))
        actual_seen = args.steps * optimal_global_batch
        coverage = 100.0 * actual_seen / total_sequences
        print(f"  fixed steps:              {format_int(args.steps)}")
        print(f"  optimal global batch:     {format_int(optimal_global_batch)}")
        print(f"  sequences seen:           {format_int(actual_seen)}")
        print(f"  coverage:                 {coverage:.2f}%")
        suggestions = suggest_batch_decomposition(optimal_global_batch, args.num_gpus)
        if suggestions:
            print(f"  per-GPU options ({args.num_gpus} GPUs):")
            for batch_size, grad_accum, local_batch in suggestions:
                print(
                    f"    batch_size={batch_size}, "
                    f"gradient_accumulation_steps={grad_accum} "
                    f"(local={local_batch}, global={optimal_global_batch})"
                )
        elif optimal_global_batch % args.num_gpus != 0:
            print(
                f"  note: global batch {optimal_global_batch} is not divisible by "
                f"{args.num_gpus} GPUs; adjust steps or GPU count slightly."
            )

    if args.global_batch is not None:
        optimal_steps = max(1, math.ceil(total_sequences / args.global_batch))
        actual_seen = optimal_steps * args.global_batch
        coverage = 100.0 * actual_seen / total_sequences
        local_batch = args.global_batch / args.num_gpus
        print(f"  fixed global batch:       {format_int(args.global_batch)}")
        print(f"  optimal steps:            {format_int(optimal_steps)}")
        print(f"  sequences seen:           {format_int(actual_seen)}")
        print(f"  coverage:                 {coverage:.2f}%")
        print(f"  local batch per GPU:      {local_batch:g}  ({args.num_gpus} GPUs)")

    if args.steps is None and args.global_batch is None:
        print("  Pass --steps and/or --global-batch to get recommendations.")
        print("  Example:")
        print("    --steps 20000")
        print("    --global-batch 504 --num-gpus 4")

    print()
    print("Notes:")
    print("  - Counts training sequences (seq_len chunks), not raw JSONL lines.")
    print("  - Data processing uses randomness; use the same --seed for reproducibility.")
    print("  - Training dataloader defaults to infinite=True, so fixed steps can re-loop data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
