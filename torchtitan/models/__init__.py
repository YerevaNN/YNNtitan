# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama import llama2_configs, llama3_configs, Transformer, download_llama3_weights, export_llama3_weights
from torchtitan.models.opt import opt_configs, OPT, download_opt_weights, export_opt_weights
from torchtitan.models.qwen3 import qwen3_configs, Qwen3Transformer, download_qwen3_weights, export_qwen3_weights, build_qwen3_model_args_from_pretrained

models_config = {
    "llama2": llama2_configs,
    "llama3": llama3_configs,
    "opt": opt_configs,
    "qwen3": qwen3_configs,
}

model_name_to_cls = {
    "llama2": Transformer,
    "llama3": Transformer,
    "opt": OPT,
    "qwen3": Qwen3Transformer,
}

model_name_to_tokenizer = {
    "llama2": "sentencepiece",
    "llama3": "tiktoken",
    "opt": "tiktoken",
    "qwen3": "tiktoken",
}

model_name_to_weights_download_fns = {
    "opt": download_opt_weights,
    "llama3": download_llama3_weights,
    "qwen3": download_qwen3_weights,
}

model_name_to_weights_export_fns = {
    "opt": export_opt_weights,
    "llama3": export_llama3_weights,
    "qwen3": export_qwen3_weights,
}