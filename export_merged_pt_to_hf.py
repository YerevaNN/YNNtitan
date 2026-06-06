#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""
Load a merged torchtitan checkpoint (.pt from dcp_to_torch) and export to HuggingFace
`save_pretrained` layout (config + weights).

Requires one CUDA device (logit verification in export matches upstream torchtitan).

Example:
  python export_merged_pt_to_hf.py \\
    --job.config_file train_configs/llama3_170m.toml \\
    --checkpoint path/to/step-20000_merged.pt \\
    --output-dir path/to/hf_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.models.llama.utils import export_llama3_weights
from torchtitan.tokenizers.tokenizer import build_tokenizer


def _resolve_hf_config_dir(hf_config_dir: str | None, output_dir: Path) -> Path:
    if hf_config_dir:
        return Path(hf_config_dir).resolve()
    if (output_dir / "config.json").is_file():
        return output_dir
    raise ValueError(
        "Pass --hf-config-dir to a local HF export folder containing config.json "
        "(for example your 170M hf_export_step20000 directory)."
    )


def _load_titan_from_merged_pt(
    job_config: JobConfig, checkpoint_path: Path, device: torch.device
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

    logger.info(f"Loading merged checkpoint {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise KeyError(
            f"Expected key 'model' in {checkpoint_path} (output of dcp_to_torch)."
        )
    state = ckpt["model"]
    model.to_empty(device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job.config_file",
        dest="config_file",
        type=str,
        required=True,
        help="Training TOML (model name, flavor, tokenizer_path, seq_len, norm_type).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Merged .pt containing a 'model' state dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for HF config + weights (created if missing).",
    )
    parser.add_argument(
        "--hf-config-dir",
        type=str,
        default=None,
        help=(
            "Local HF folder with config.json used as the architecture template. "
            "Defaults to --output-dir when config.json already exists there."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda (default) — required for export verification.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip HF vs torchtitan logit check (not recommended).",
    )
    parser.add_argument(
        "--no-save-tokenizer",
        action="store_true",
        help="Do not write tokenizer files into output-dir.",
    )

    args_ns, job_argv = parser.parse_known_args(argv)
    job_argv = list(job_argv)
    if "--job.config_file" not in job_argv:
        job_argv = ["--job.config_file", args_ns.config_file] + job_argv

    job_config = JobConfig()
    job_config.parse_args(job_argv)
    init_logger(job_config.logging.log_level)

    checkpoint_path = Path(args_ns.checkpoint).resolve()
    output_dir = Path(args_ns.output_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(args_ns.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for export (verification runs on GPU).")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args_ns.hf_config_dir:
        _resolve_hf_config_dir(args_ns.hf_config_dir, output_dir)

    model, tokenizer = _load_titan_from_merged_pt(
        job_config, checkpoint_path, device=device
    )

    export_llama3_weights(
        model,
        str(output_dir),
        tokenizer.model,
        tokenizer.padded_n_words,
        verify=not args_ns.no_verify,
    )

    if not args_ns.no_save_tokenizer:
        tokenizer.model.save_pretrained(str(output_dir))
        logger.info("Saved tokenizer alongside model.")

    logger.info("Done. HF export at %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
