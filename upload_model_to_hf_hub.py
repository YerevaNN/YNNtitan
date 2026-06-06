#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""
Upload a local HuggingFace-style directory (e.g. from export_merged_pt_to_hf.py:
config.json, model files, tokenizer) to the Hugging Face Hub.

Authentication (pick one):
  - Export HF_TOKEN with a write-capable token from https://huggingface.co/settings/tokens
  - Or pass --token (avoid shell history; prefer env)

The Hub repo is created if missing (same as opening https://huggingface.co/new ).
If you see 403 on create, run ``huggingface-cli whoami`` and use that exact username
in ``--repo-id``, or create the empty model repo on the website and pass
``--skip-create-repo``.

Large files (e.g. ``model.safetensors``) upload via **Git LFS**. A 403 on a URL
containing ``info/lfs/objects/batch`` almost always means the token can reach the
API but is **not allowed to write LFS blobs** — typical with **fine-grained**
tokens missing **Contents: Read and write** on that repo. Fix: create a
**classic** token with role **Write**, or edit the fine-grained token so this
repository has full write (including file/LFS upload).

Example:
  export HF_TOKEN=hf_...
  python upload_model_to_hf_hub.py \\
    --local-dir /path/to/hf_export_step20000 \\
    --repo-id YourUser/llama-170m-step20k \\
    --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Directory produced by save_pretrained / export_merged_pt_to_hf.py.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub model id, e.g. username/my-model-name.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (only if creating the repo here).",
    )
    parser.add_argument(
        "--skip-create-repo",
        action="store_true",
        help="Do not call create_repo (repo must already exist on the Hub).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF access token with write scope. Defaults to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model weights and tokenizer",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    local = Path(args.local_dir).resolve()
    if not local.is_dir():
        print(f"error: not a directory: {local}", file=sys.stderr)
        return 1

    # Prefer explicit token, but fall back to cached login (hf auth login).
    token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    if not args.skip_create_repo:
        try:
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="model",
                private=args.private,
                exist_ok=True,
            )
        except HfHubHTTPError as e:
            if e.response.status_code == 403:
                print(
                    "error: 403 creating repo — your token cannot create under this namespace.\n"
                    "  1) Check your Hub username: huggingface-cli whoami\n"
                    "  2) Use --repo-id <that_username>/Llama-3-170M-dcc444 (case-sensitive)\n"
                    "  3) New token: https://huggingface.co/settings/tokens — use a token with\n"
                    "     write access (classic 'Write', or fine-grained: Repositories → write)\n"
                    "  4) Or create the empty model at https://huggingface.co/new (in browser)\n"
                    "     then re-run with --skip-create-repo",
                    file=sys.stderr,
                )
            raise

    try:
        api.upload_folder(
            folder_path=str(local),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    except HfHubHTTPError as e:
        if e.response.status_code == 403 and "lfs" in str(e.response.url).lower():
            print(
                "error: 403 on Git LFS upload — weights (e.g. model.safetensors) were rejected.\n"
                "  • Prefer a classic token: https://huggingface.co/settings/tokens → New token → Type: Write\n"
                "  • If you use a fine-grained token: open the token → add this repo with\n"
                "    Repository permissions → Contents (or equivalent): Read and write\n"
                "  • Confirm: huggingface-cli whoami  matches the owner in --repo-id\n"
                "  • Ensure your HF account email is verified (Settings → Account)",
                file=sys.stderr,
            )
        raise

    print(f"Uploaded {local} -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
