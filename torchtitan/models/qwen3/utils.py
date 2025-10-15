import torch
from transformers import AutoModelForCausalLM, AutoConfig

from torchtitan.logging import logger
from torchtitan.models.qwen3.model import Qwen3Transformer


# reverse_permute for sliced rotary (same logic as LLaMA)
def _reverse_permute(w, n_heads, dim1, dim2):
	return (
		w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
		.transpose(1, 2)
		.reshape(dim1, dim2)
	)


# permute for sliced rotary (same logic as LLaMA)
def _permute(w, n_heads, dim1, dim2):
	return (
		w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
		.transpose(1, 2)
		.reshape(dim1, dim2)
	)


def _get_hf_qwen3_state_dict_keys_mapping(num_layers: int, include_lm_head: bool = False):
	keys_mapping = {
		"tok_embeddings.weight": "model.embed_tokens.weight",
		"norm.weight": "model.norm.weight",
		# Compiled model variants
		"_orig_mod.norm.weight": "model.norm.weight",
	}
	if include_lm_head:
		keys_mapping["output.weight"] = "lm_head.weight"
		keys_mapping["_orig_mod.output.weight"] = "lm_head.weight"
	for layer in range(num_layers):
		# Standard keys (for uncompiled models)
		layer_keys = {
			f"layers.{layer}.attention.wq.weight": f"model.layers.{layer}.self_attn.q_proj.weight",
			f"layers.{layer}.attention.wk.weight": f"model.layers.{layer}.self_attn.k_proj.weight",
			f"layers.{layer}.attention.wv.weight": f"model.layers.{layer}.self_attn.v_proj.weight",
			f"layers.{layer}.attention.wo.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
			f"layers.{layer}.attention.q_norm.weight": f"model.layers.{layer}.self_attn.q_norm.weight",
			f"layers.{layer}.attention.k_norm.weight": f"model.layers.{layer}.self_attn.k_norm.weight",
			f"layers.{layer}.feed_forward.w1.weight": f"model.layers.{layer}.mlp.gate_proj.weight",
			f"layers.{layer}.feed_forward.w3.weight": f"model.layers.{layer}.mlp.up_proj.weight",
			f"layers.{layer}.feed_forward.w2.weight": f"model.layers.{layer}.mlp.down_proj.weight",
			f"layers.{layer}.attention_norm.weight": f"model.layers.{layer}.input_layernorm.weight",
			f"layers.{layer}.ffn_norm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
		}
		keys_mapping.update(layer_keys)
		
		# Compiled model keys (with _orig_mod prefix from torch.compile)
		# torch.compile adds _orig_mod prefix to all submodules within each layer
		compiled_keys = {}
		for key, value in layer_keys.items():
			if key.startswith(f"layers.{layer}."):
				# For layer-level keys, _orig_mod is inserted after the layer number
				compiled_key = key.replace(f"layers.{layer}.", f"layers.{layer}._orig_mod.")
				compiled_keys[compiled_key] = value
		keys_mapping.update(compiled_keys)
		
	return keys_mapping


def download_qwen3_weights(
	model: Qwen3Transformer,
	weights_path: str,
	tokenizer,
	source: str,
	token_embedding_size: int,
):
	if source != "huggingface":
		raise NotImplementedError
	# Load HF model and align vocab size
	hf_model = AutoModelForCausalLM.from_pretrained(weights_path)
	hf_model.resize_token_embeddings(new_num_tokens=token_embedding_size)
	include_lm_head = not model.model_args.share_embeddings
	keys_mapping = _get_hf_qwen3_state_dict_keys_mapping(model.n_layers, include_lm_head)
	hf_state_dict = hf_model.state_dict()
	corrected_state_dict = {}
	# determine dims with optional head_dim
	head_dim = model.model_args.head_dim if model.model_args.head_dim is not None else model.model_args.dim // model.model_args.n_heads
	q_out = model.model_args.n_heads * head_dim
	kv_out = model.model_args.n_kv_heads * head_dim
	for key, value in keys_mapping.items():
		assert hf_state_dict[value].shape == model.state_dict()[key].shape, f"Shape mismatch for {key} vs {value}: {hf_state_dict[value].shape} != {model.state_dict()[key].shape}"
		# Apply reverse_permute for q/k projections to match Titan sliced-rotary layout
		if "self_attn.q_proj.weight" in value:
			corrected_state_dict[key] = _reverse_permute(
				hf_state_dict[value],
				model.model_args.n_heads,
				q_out,
				model.model_args.dim,
			)
		elif "self_attn.k_proj.weight" in value:
			corrected_state_dict[key] = _reverse_permute(
				hf_state_dict[value],
				model.model_args.n_kv_heads,
				kv_out,
				model.model_args.dim,
			)
		else:
			corrected_state_dict[key] = hf_state_dict[value]
	with torch.device(model.freqs_cis.device):
		corrected_state_dict["freqs_cis"] = model._precompute_freqs_cis()
	model.load_state_dict(corrected_state_dict)
	logger.info("Successfully loaded Qwen3 model to titan model.")


def export_qwen3_weights(model: Qwen3Transformer, save_dir: str, tokenizer, token_embedding_size: int):
	logger.info(f"Starting Qwen3 model export to {save_dir}")
	logger.info(f"Model: {model.n_layers} layers, vocab_size: {model.model_args.vocab_size}")
	
	# Build an HF config that matches our model
	cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
	cfg.hidden_size = model.model_args.dim
	cfg.num_hidden_layers = model.model_args.n_layers
	cfg.num_attention_heads = model.model_args.n_heads
	cfg.num_key_value_heads = model.model_args.n_kv_heads
	cfg.head_dim = (model.model_args.head_dim if model.model_args.head_dim is not None else model.model_args.dim // model.model_args.n_heads)
	cfg.intermediate_size = int(4 * model.model_args.dim * (model.model_args.ffn_dim_multiplier or 1.0))
	cfg.sliding_window = model.model_args.sliding_window
	# Handle layer_types - if None, default all layers to full attention
	if model.model_args.layer_types is not None:
		cfg.layer_types = model.model_args.layer_types
	else:
		# Default all layers to full attention when layer_types is None
		cfg.layer_types = ["full_attention"] * model.model_args.n_layers
	hf_model = AutoModelForCausalLM.from_config(cfg)
	hf_model.resize_token_embeddings(new_num_tokens=token_embedding_size)
	include_lm_head = not model.model_args.share_embeddings
	keys_mapping = _get_hf_qwen3_state_dict_keys_mapping(model.n_layers, include_lm_head)
	state_dict = model.state_dict()
	
	# Filter keys_mapping to only include keys that actually exist in state_dict
	# This handles both compiled and uncompiled models
	filtered_keys_mapping = {}
	for key, value in keys_mapping.items():
		if key in state_dict:
			filtered_keys_mapping[key] = value
	
	logger.info(f"Using {len(filtered_keys_mapping)} keys from state_dict (out of {len(keys_mapping)} possible)")
	
	# Verify we have the essential keys (checking both compiled and uncompiled versions)
	essential_keys_variants = [
		["tok_embeddings.weight"],  # This key should always be present
		["norm.weight", "_orig_mod.norm.weight"]  # Either compiled or uncompiled version
	]
	missing_essential = []
	for key_variants in essential_keys_variants:
		if not any(k in state_dict for k in key_variants):
			missing_essential.append(key_variants[0])  # Report the primary key name
	
	if missing_essential:
		logger.error(f"Missing essential keys: {missing_essential}")
		logger.info(f"Available keys (first 10): {list(state_dict.keys())[:10]}")
		raise KeyError(f"Missing essential keys in model state_dict: {missing_essential}")
	
	keys_mapping = filtered_keys_mapping
	
	corrected_state_dict = {}
	# dims for permute
	head_dim = model.model_args.head_dim if model.model_args.head_dim is not None else model.model_args.dim // model.model_args.n_heads
	q_out = model.model_args.n_heads * head_dim
	kv_out = model.model_args.n_kv_heads * head_dim
	for key, value in keys_mapping.items():
		assert hf_model.state_dict()[value].shape == state_dict[key].shape
		# Apply permute for q/k projections when exporting to HF
		if "self_attn.q_proj.weight" in value:
			corrected_state_dict[value] = _permute(
				state_dict[key],
				model.model_args.n_heads,
				q_out,
				model.model_args.dim,
			)
		elif "self_attn.k_proj.weight" in value:
			corrected_state_dict[value] = _permute(
				state_dict[key],
				model.model_args.n_kv_heads,
				kv_out,
				model.model_args.dim,
			)
		else:
			corrected_state_dict[value] = state_dict[key]
	if model.model_args.share_embeddings:
		corrected_state_dict["lm_head.weight"] = state_dict["tok_embeddings.weight"]
	hf_model.load_state_dict(corrected_state_dict)
	hf_model.save_pretrained(save_dir)
	logger.info(f"Successfully exported Qwen3 model to huggingface model at {save_dir}.") 