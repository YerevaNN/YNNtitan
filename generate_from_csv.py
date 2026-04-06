#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""
Load a torchtitan Llama checkpoint (e.g. merged DCP .pt) and run text generation
on random prompts from a CSV column.

Example:
  python generate_from_csv.py \\
    --job.config_file train_configs/llama3_170m.toml \\
    --checkpoint path/to/step-20000_merged.pt \\
    --csv prompts.csv \\
    --prompt-column prompt \\
    --num-samples 8 \\
    --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.tokenizers.tokenizer import build_tokenizer


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


def _sample_top_p(probs: torch.Tensor, top_p: float) -> int:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    mask[..., 0] = False
    filtered = sorted_probs.masked_fill(mask, 0.0)
    filtered = filtered / filtered.sum()
    choice = torch.multinomial(filtered, num_samples=1).item()
    return int(sorted_idx[choice].item())


def _next_token(
    logits: torch.Tensor, temperature: float, top_p: float
) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        return _sample_top_p(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.inference_mode()
def generate_completion(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    use_bos: bool,
) -> str:
    ids: List[int] = tokenizer.encode(prompt, bos=use_bos, eos=False)
    if len(ids) >= max_seq_len:
        keep = max_seq_len - 1
        logger.warning(
            "Prompt length %d >= max_seq_len %d; truncating from the left.",
            len(ids),
            max_seq_len,
        )
        ids = ids[-keep:]
    prompt_len = len(ids)
    device = next(model.parameters()).device
    eos = tokenizer.eos_id

    for _ in range(max_new_tokens):
        if len(ids) >= max_seq_len:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)
        next_id = _next_token(logits[0, -1], temperature, top_p)
        ids.append(next_id)
        if next_id == eos:
            break

    gen_tokens = ids[prompt_len:]
    return tokenizer.decode(gen_tokens)


def _cast_float_modules(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast parameters and real buffers to dtype; keep RoPE `freqs_cis` complex."""
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.to(dtype=dtype)
        for _, buf in module.named_buffers():
            if buf.is_complex():
                continue
            buf.data = buf.data.to(dtype=dtype)


def _load_titan_model(
    job_config: JobConfig, checkpoint_path: Path, dtype: torch.dtype, device: torch.device
):
    model_name = job_config.model.name
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    model_config = models_config[model_name][job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.padded_n_words
    model_config.max_seq_len = job_config.training.seq_len

    model_cls = model_name_to_cls[model_name]
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    logger.info(f"Loading checkpoint {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(
            f"Expected key 'model' in {checkpoint_path} (merged torchtitan checkpoint)."
        )
    state = ckpt["model"]
    model.to_empty(device=device)
    model.load_state_dict(state, strict=True)
    _cast_float_modules(model, dtype)
    model.eval()
    return model, tokenizer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job.config_file",
        dest="config_file",
        type=str,
        required=True,
        help="Training TOML (model flavor, tokenizer_path, seq_len, norm_type).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Merged .pt from dcp_to_torch (must contain a 'model' state dict).",
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
        help="CSV field delimiter (default: comma). Use ';' for semicolon-separated files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of rows to sample (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature; 0 = greedy.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p (1.0 = disabled).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(TORCH_DTYPE_MAP.keys()),
        help="Model activations dtype on GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cuda:0 / cpu (cpu is slow).",
    )
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not prepend BOS when encoding the prompt.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="If set, append one JSON object per line (prompt + generation).",
    )
    args_ns, job_argv = parser.parse_known_args(argv)
    job_argv = list(job_argv)
    if "--job.config_file" not in job_argv:
        job_argv = ["--job.config_file", args_ns.config_file] + job_argv

    job_config = JobConfig()
    job_config.parse_args(job_argv)
    init_logger(job_config.logging.log_level)

    random.seed(args_ns.seed)
    torch.manual_seed(args_ns.seed)

    csv_path = Path(args_ns.csv).resolve()
    checkpoint_path = Path(args_ns.checkpoint).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(args_ns.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    dtype = TORCH_DTYPE_MAP[args_ns.dtype]
    model, tokenizer = _load_titan_model(
        job_config, checkpoint_path, dtype=dtype, device=device
    )

    delim = args_ns.csv_delimiter
    if delim == "\\t":
        delim = "\t"
    prompts = _read_prompts_csv(csv_path, args_ns.prompt_column, delim)
    k = min(args_ns.num_samples, len(prompts))
    chosen = random.sample(prompts, k=k)
    use_bos = not args_ns.no_bos

    out_path = Path(args_ns.output_jsonl) if args_ns.output_jsonl else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    max_seq = job_config.training.seq_len
    logger.info(
        f"Generating for {k} prompts (max_new_tokens={args_ns.max_new_tokens}, "
        f"max_seq_len={max_seq})"
    )

    for i, prompt in enumerate(chosen):
        completion = generate_completion(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args_ns.max_new_tokens,
            max_seq_len=max_seq,
            temperature=args_ns.temperature,
            top_p=args_ns.top_p,
            use_bos=use_bos,
        )
        record = {"index": i, "prompt": prompt, "generation": completion}
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        if out_path:
            with out_path.open("a", encoding="utf-8") as jf:
                jf.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
