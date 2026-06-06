#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_model_dir(
    model_id_or_path: str, *, token: str | None, revision: str | None
) -> Path:
    p = Path(model_id_or_path).expanduser()
    if p.exists():
        return p.resolve()

    local_dir = snapshot_download(
        repo_id=model_id_or_path,
        repo_type="model",
        token=token,
        revision=revision,
        local_files_only=False,
    )
    return Path(local_dir).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Load a Hugging Face model and tokenizer from local paths or Hub repo ids.\n\n"
            "Auth: export HF_TOKEN=hf_... (required for private repos)."
        )
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help='Local model directory OR Hub id like "tigranfah/Llama-3-380M-e5254a".',
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Local tokenizer directory OR Hub id like "
            '"tigranfah/Llama-3.2-chem-1B-v2". Defaults to --model.'
        ),
    )
    ap.add_argument("--revision", type=str, default=None, help="Hub revision (branch/tag/commit).")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to load on (default: "cuda" if available else "cpu").',
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")

    model_dir = _resolve_model_dir(args.model, token=token, revision=args.revision)
    tok_spec = args.tokenizer or args.model
    tokenizer_dir = _resolve_model_dir(tok_spec, token=token, revision=args.revision)
    print(f"Resolved model dir: {model_dir}")
    print(f"Resolved tokenizer dir: {tokenizer_dir}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        use_fast=True,
        token=token,
        revision=args.revision,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        token=token,
        revision=args.revision,
    )

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: device={device}, dtype={torch_dtype}, parameters={n_params:,}")
    print(f"Tokenizer vocab_size={getattr(tokenizer, 'vocab_size', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
