# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.logging import logger
from torchtitan.models.llama import Transformer
from transformers import AutoModelForCausalLM


# reverse_permute for sliced rotary
def reverse_permute(w, n_heads, dim1, dim2):
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def get_hf_llama3_state_dict_keys_mapping(
    num_layers: int, include_lm_head: bool = False
):
    """
    Get a mapping between state dict keys of different implementations.

    Args:
        num_layers (int): number of transformer layers (blocks).

    Returns:
        dict: mapping between local implementation state dict keys and hf implementation state dict keys

    """
    keys_mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        # add layer weight mappings here
        "norm.weight": "model.norm.weight",
    }
    if include_lm_head:
        keys_mapping["output.weight"] = "lm_head.weight"
    for layer in range(num_layers):
        keys_mapping.update(
            {
                f"layers.{layer}.attention.wq.weight": f"model.layers.{layer}.self_attn.q_proj.weight",
                f"layers.{layer}.attention.wk.weight": f"model.layers.{layer}.self_attn.k_proj.weight",
                f"layers.{layer}.attention.wv.weight": f"model.layers.{layer}.self_attn.v_proj.weight",
                f"layers.{layer}.attention.wo.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
                f"layers.{layer}.feed_forward.w1.weight": f"model.layers.{layer}.mlp.gate_proj.weight",
                f"layers.{layer}.feed_forward.w3.weight": f"model.layers.{layer}.mlp.up_proj.weight",
                f"layers.{layer}.feed_forward.w2.weight": f"model.layers.{layer}.mlp.down_proj.weight",
                f"layers.{layer}.attention_norm.weight": f"model.layers.{layer}.input_layernorm.weight",
                f"layers.{layer}.ffn_norm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
            }
        )

    return keys_mapping


def verify_logits_matching(
    model: Transformer,
    hf_model,
    tokenizer,
    atol: float,
    prompts=("Hello world", "The capital of France is "),
):
    device = "cuda"
    hf_model.to(device)
    model.eval()
    for prompt in prompts:
        data = tokenizer(prompt, return_tensors="pt").to(device)
        hf_logits = hf_model(**data).logits
        logits = model(data.input_ids)
        assert torch.allclose(hf_logits, logits, atol=atol)


def download_llama3_weights(
    model: Transformer,
    weights_path: str,
    tokenizer,
    source: str,
    token_embedding_size: int,
):
    """
    write docs
    """
    if source == "huggingface":
        hf_model = AutoModelForCausalLM.from_pretrained(weights_path)
        hf_model.resize_token_embeddings(new_num_tokens=token_embedding_size)
        include_lm_head = not model.model_args.share_embeddings
        keys_mapping = get_hf_llama3_state_dict_keys_mapping(
            model.n_layers, include_lm_head
        )
        hf_state_dict = hf_model.state_dict()
        corrected_state_dict = {}
        for key, value in keys_mapping.items():
            assert hf_state_dict[value].shape == model.state_dict()[key].shape
            if "self_attn.q_proj.weight" in value:
                corrected_state_dict[key] = reverse_permute(
                    hf_state_dict[value],
                    model.model_args.n_heads,
                    model.model_args.dim,
                    model.model_args.dim,
                )
            elif "self_attn.k_proj.weight" in value:
                kv_dim = model.model_args.dim // (
                    model.model_args.n_heads // model.model_args.n_kv_heads
                )
                corrected_state_dict[key] = reverse_permute(
                    hf_state_dict[value],
                    model.model_args.n_kv_heads,
                    kv_dim,
                    model.model_args.dim,
                )
            else:
                corrected_state_dict[key] = hf_state_dict[value]

        with torch.device(model.freqs_cis.device):
            corrected_state_dict["freqs_cis"] = model._precompute_freqs_cis()

        model.load_state_dict(corrected_state_dict)
        verify_logits_matching(
            model=model, hf_model=hf_model, tokenizer=tokenizer, atol=1e-1
        )
        logger.info("Successfully loaded Llama 3 model to titan model.")
    else:
        raise NotImplementedError


def map_n_layers_to_model_name(n_layers):
    return {
        16: "meta-llama/Llama-3.2-1B",
    }[n_layers]


def export_llama3_weights(
    model: Transformer, save_dir, tokenizer, token_embedding_size: int
):
    """
    write docs
    """
    weights_path = map_n_layers_to_model_name(model.n_layers)
    hf_model = AutoModelForCausalLM.from_pretrained(weights_path)
    hf_model.resize_token_embeddings(new_num_tokens=token_embedding_size)
    include_lm_head = not model.model_args.share_embeddings
    keys_mapping = get_hf_llama3_state_dict_keys_mapping(
        model.n_layers, include_lm_head
    )
    state_dict = model.state_dict()
    corrected_state_dict = {}
    for key, value in keys_mapping.items():
        assert hf_model.state_dict()[value].shape == state_dict[key].shape
        if "self_attn.q_proj.weight" in value:
            corrected_state_dict[value] = permute(
                state_dict[key],
                model.model_args.n_heads,
                model.model_args.dim,
                model.model_args.dim,
            )
        elif "self_attn.k_proj.weight" in value:
            kv_dim = model.model_args.dim // (
                model.model_args.n_heads // model.model_args.n_kv_heads
            )
            corrected_state_dict[value] = permute(
                state_dict[key],
                model.model_args.n_kv_heads,
                kv_dim,
                model.model_args.dim,
            )
        else:
            corrected_state_dict[value] = state_dict[key]

    if model.model_args.share_embeddings:
        assert hf_model.state_dict()[value].shape == state_dict[key].shape
        corrected_state_dict["lm_head.weight"] = state_dict["tok_embeddings.weight"]

    hf_model.load_state_dict(corrected_state_dict)
    verify_logits_matching(
        model=model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        atol=1e-1,
        prompts=["", "[QED]", "[SAFE]"],
    )
    hf_model.save_pretrained(save_dir)
    logger.info(
        f"Successfully exported Llama 3 model to huggingface model at {save_dir}."
    )
