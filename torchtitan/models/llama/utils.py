# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import torch
from torchtitan.logging import logger
from torchtitan.models.llama.configs import llama3_configs
from torchtitan.models.llama.model import ModelArgs, Transformer

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

# Official Llama 3.2 checkpoints (gated). Sizes match our "1B" / "3B" flavors;
# RoPE theta and other HF fields come from the Hub config.
OFFICIAL_LLAMA32 = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}


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
    prompts=("[SMILES]", "[QED]0.54[/QED]", "[SAS][0.12,0.45][/SAS]"),
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
    verify: bool = True,
):
    """
    write docs
    """
    if source == "huggingface":
        hf_model = AutoModelForCausalLM.from_pretrained(
            weights_path, token=_hf_token()
        )
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
        if verify:
            verify_logits_matching(
                model=model, hf_model=hf_model, tokenizer=tokenizer, atol=0.1
            )
        logger.info("Successfully loaded Llama 3 model to titan model.")
    else:
        raise NotImplementedError


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def is_official_llama32_flavor(flavor: str) -> bool:
    return flavor in OFFICIAL_LLAMA32


def _ffn_args_for_intermediate(dim: int, intermediate_size: int):
    """Invert titan FeedForward sizing so hidden dim matches HF intermediate_size."""
    base = 4 * dim
    if intermediate_size == base:
        return None, 256
    for multiple_of in (256, 1):
        multiplier = intermediate_size / base
        hidden = int(multiplier * base)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        if hidden == intermediate_size:
            return multiplier, multiple_of
    raise ValueError(
        f"Cannot map HF intermediate_size={intermediate_size} with dim={dim} "
        "onto titan FFN (4*dim * multiplier, rounded to multiple_of)."
    )


def _rope_theta_from_hf_config(hf_config) -> float:
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
        return float(rope_parameters["rope_theta"])
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)
    return 500000.0


def model_args_from_hf_config(hf_config) -> ModelArgs:
    dim = int(hf_config.hidden_size)
    n_heads = int(hf_config.num_attention_heads)
    n_kv_heads = int(
        getattr(hf_config, "num_key_value_heads", None) or n_heads
    )
    intermediate_size = int(hf_config.intermediate_size)
    ffn_dim_multiplier, multiple_of = _ffn_args_for_intermediate(
        dim, intermediate_size
    )
    return ModelArgs(
        dim=dim,
        n_layers=int(hf_config.num_hidden_layers),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        rope_theta=_rope_theta_from_hf_config(hf_config),
        norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-5)),
        share_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        ffn_dim_multiplier=ffn_dim_multiplier,
        multiple_of=multiple_of,
    )


def resolve_llama3_model_args(flavor: str) -> ModelArgs:
    """Our configs.py flavor, or official Llama-3.2-1B / Llama-3.2-3B from the Hub."""
    if flavor in llama3_configs:
        return llama3_configs[flavor]
    if not is_official_llama32_flavor(flavor):
        known = sorted(list(llama3_configs) + list(OFFICIAL_LLAMA32))
        raise ValueError(f"Unknown llama3 flavor {flavor!r}. Known: {known}")
    hub_id = OFFICIAL_LLAMA32[flavor]
    token = _hf_token()
    try:
        hf_config = AutoConfig.from_pretrained(hub_id, token=token)
    except OSError as err:
        raise OSError(
            f"Could not load {hub_id}: {err}. "
            "Accept the model license on Hugging Face and set HF_TOKEN."
        ) from err
    model_args = model_args_from_hf_config(hf_config)
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    rope_type = None
    if isinstance(rope_parameters, dict):
        rope_type = rope_parameters.get("rope_type")
    logger.info(
        "Using official %s config from %s (sizes + rope_theta=%s). "
        "Training still uses simple RoPE; HF rope_type=%s is not applied.",
        flavor,
        hub_id,
        model_args.rope_theta,
        rope_type,
    )
    return model_args


def format_hf_rope_parameters(flavor: str) -> str:
    """Pretty-print HF export rope_parameters for a llama3 flavor."""
    if flavor not in llama3_configs:
        raise ValueError(f"Unknown llama3 flavor: {flavor}")
    hf_config = model_args_to_hf_config(llama3_configs[flavor])
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = hf_config.to_dict().get("rope_parameters", {})
    return json.dumps({"rope_parameters": rope_parameters}, indent=2)


def _titan_ffn_hidden_dim(model_args: ModelArgs) -> int:
    """Match FeedForward hidden dim used in training."""
    hidden_dim = 4 * model_args.dim
    if model_args.ffn_dim_multiplier is not None:
        hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
    multiple_of = model_args.multiple_of
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def model_args_to_hf_config(model_args: ModelArgs) -> LlamaConfig:
    """Build a HuggingFace LlamaConfig from the training ModelArgs.

    Does not load a Hub / local Llama-3.2 template. RoPE is the same simple
    (default) rotary used in training: ``precompute_freqs_cis(..., rope_theta)``.
    """
    n_kv_heads = (
        model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
    )
    vocab_size = model_args.vocab_size if model_args.vocab_size > 0 else 32000
    return LlamaConfig(
        hidden_size=model_args.dim,
        num_hidden_layers=model_args.n_layers,
        num_attention_heads=model_args.n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=model_args.dim // model_args.n_heads,
        intermediate_size=_titan_ffn_hidden_dim(model_args),
        rms_norm_eps=model_args.norm_eps,
        rope_theta=model_args.rope_theta,
        max_position_embeddings=model_args.max_seq_len,
        vocab_size=vocab_size,
        tie_word_embeddings=model_args.share_embeddings,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )


def export_llama3_weights(
    model: Transformer,
    save_dir,
    tokenizer,
    token_embedding_size: int,
    verify: bool = True,
):
    """
    Map torchtitan Llama3 weights to HuggingFace `AutoModelForCausalLM` and save via `save_pretrained`.
    """

    model_config = model_args_to_hf_config(model.model_args)
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    # include dtype in hf config
    model_config.torch_dtype = next(model.parameters()).dtype
    hf_model = AutoModelForCausalLM.from_config(model_config)
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
    if verify:
        verify_logits_matching(model=model, hf_model=hf_model, tokenizer=tokenizer, atol=12)
    hf_model.save_pretrained(save_dir)
    logger.info(
        f"Successfully exported Llama 3 model to huggingface model at {save_dir}."
    )
