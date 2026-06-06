#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""
Load a HuggingFace-exported model (e.g. from export_merged_pt_to_hf.py) and generate
from the same CSV / sampling settings as generate_from_csv.py for side-by-side checks.

Outputs may still differ from ``generate_from_csv.py`` for the reasons noted in
that script's docstring. With ``--logits-top-k``, this script replays a forward
pass per generated token to log raw last-position logits for the sequence that
``generate()`` actually sampled.

Example:
  python generate_from_csv_hf.py \\
    --hf-model-dir path/to/hf_export_step20000 \\
    --csv path/to/0_mols.csv \\
    --csv-delimiter ";" \\
    --prompt-column prompt \\
    --num-samples 8 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def _read_prompts_csv(path: Path, column: str, delimiter: str) -> List[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"Column {column!r} not in CSV headers: {reader.fieldnames}"
            )
        prompts = []
        for row in reader:
            v = row.get(column)
            if v is None or not str(v).strip():
                continue
            prompts.append(str(v).strip())
    if not prompts:
        raise ValueError(f"No non-empty values in column {column!r}")
    return prompts


def _top_logits_payload(
    last_logits: torch.Tensor, k: int, chosen_id: int
) -> Dict[str, Any]:
    k = min(k, last_logits.numel())
    vals, idx = torch.topk(last_logits.float(), k=k)
    return {
        "chosen_id": chosen_id,
        "top_logits": [
            {"token_id": int(i), "logit": float(v)} for v, i in zip(vals.tolist(), idx.tolist())
        ],
    }


def _encode_prompt_like_titan(
    tokenizer: AutoTokenizer, prompt: str, use_bos: bool
) -> List[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if use_bos and tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    return ids


@torch.inference_mode()
def _logits_trace_for_sequence(
    model: AutoModelForCausalLM,
    full_ids: List[int],
    prompt_len: int,
    logits_top_k: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Raw last-position logits for each chosen continuation token (replay forwards)."""
    trace: List[Dict[str, Any]] = []
    for step, t in enumerate(range(prompt_len, len(full_ids))):
        prefix = full_ids[:t]
        x = torch.tensor([prefix], dtype=torch.long, device=device)
        out = model(x)
        last = out.logits[0, -1]
        chosen = full_ids[t]
        row = _top_logits_payload(last, logits_top_k, chosen)
        row["step"] = step
        trace.append(row)
    return trace


@torch.inference_mode()
def generate_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    use_bos: bool,
) -> tuple[List[int], int]:
    """Returns (full_token_ids_after_generate, prompt_len)."""
    ids = _encode_prompt_like_titan(tokenizer, prompt, use_bos)
    if len(ids) >= max_seq_len:
        keep = max_seq_len - 1
        logging.warning(
            "Prompt length %d >= max_seq_len %d; truncating from the left.",
            len(ids),
            max_seq_len,
        )
        ids = ids[-keep:]
    prompt_len = len(ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    do_sample = temperature > 0
    gen_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p < 1.0:
            gen_kwargs["top_p"] = top_p

    out = model.generate(input_ids, **gen_kwargs)
    full = out[0].tolist()
    return full, prompt_len


def _decode_new_tokens(
    tokenizer: AutoTokenizer, full_ids: List[int], prompt_len: int
) -> str:
    new_tokens = full_ids[prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model-dir",
        type=str,
        required=True,
        help="Directory from export_merged_pt_to_hf / save_pretrained.",
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column name for prompts (default: prompt).",
    )
    parser.add_argument(
        "--csv-delimiter",
        type=str,
        default=",",
        help="CSV delimiter (use ';' for semicolon-separated).",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Context length cap (match training seq_len; default 2048).",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument(
        "--logits-top-k",
        type=int,
        default=0,
        help="If > 0, add 'logits_trace': top-k raw logits per generated token (replay; slower).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    random.seed(args.seed)
    set_seed(args.seed)

    model_dir = Path(args.hf_model_dir).resolve()
    csv_path = Path(args.csv).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    logging.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    logging.info("Loading model from %s", model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    delim = args.csv_delimiter
    if delim == "\\t":
        delim = "\t"
    prompts = _read_prompts_csv(csv_path, args.prompt_column, delim)
    k = min(args.num_samples, len(prompts))
    chosen = random.sample(prompts, k=k)
    use_bos = not args.no_bos

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "HF generate: %d prompts, max_new_tokens=%d, max_seq_len=%d",
        k,
        args.max_new_tokens,
        args.max_seq_len,
    )

    for i, prompt in enumerate(chosen):
        full_ids, prompt_len = generate_hf(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
            top_p=args.top_p,
            use_bos=use_bos,
        )
        completion = _decode_new_tokens(tokenizer, full_ids, prompt_len)
        record: Dict[str, Any] = {"index": i, "prompt": prompt, "generation": completion}
        if args.logits_top_k > 0:
            record["logits_trace"] = _logits_trace_for_sequence(
                model,
                full_ids,
                prompt_len,
                args.logits_top_k,
                device,
            )
            record["prompt_token_ids"] = full_ids[:prompt_len]
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if out_path:
            with out_path.open("a", encoding="utf-8") as jf:
                jf.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
