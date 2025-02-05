# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama.model import ModelArgs, Transformer
from torchtitan.models.llama.utils import download_llama3_weights, export_llama3_weights

__all__ = ["Transformer", "download_llama3_weights", "export_llama3_weights"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "7B": ModelArgs(dim=4096, n_layers=32, n_heads=32),
    "13B": ModelArgs(dim=5120, n_layers=40, n_heads=40),
    "26B": ModelArgs(dim=5120, n_layers=80, n_heads=40),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "125M": ModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        rope_theta=500000,
        share_embeddings=True,
    ),
    "350M": ModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        rope_theta=500000,
        share_embeddings=True,
    ),
    "750M": ModelArgs(
        dim=1536,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        rope_theta=500000,
        share_embeddings=True,
    ),
    "1B": ModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=500000,
        share_embeddings=True,
    ),
    "3B": ModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        rope_theta=500000,
        ffn_dim_multiplier=2 / 3,  # in Llama3.2-3B dim is 3072, but ffn dim is 8192
        share_embeddings=True,
    ),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}
