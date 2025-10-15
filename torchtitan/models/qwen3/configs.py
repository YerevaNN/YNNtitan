from dataclasses import dataclass
from typing import Dict

from transformers import AutoConfig

from torchtitan.models.qwen3.model import Qwen3ModelArgs

# Static presets for common sizes (can be extended); values match HF Qwen3 defaults where known.
qwen3_configs: Dict[str, Qwen3ModelArgs] = {
	# Will often be overridden dynamically from HF config by build_qwen3_model_args_from_pretrained
	"debugmodel": Qwen3ModelArgs(dim=256, n_layers=8, n_heads=8, n_kv_heads=2, rope_theta=10000.0, max_seq_len=256),
	"0.6B": Qwen3ModelArgs(dim=1024, n_layers=28, n_heads=16, n_kv_heads=8, rope_theta=10000.0, share_embeddings=True),
	"1.7B": Qwen3ModelArgs(dim=1536, n_layers=28, n_heads=12, n_kv_heads=2, rope_theta=10000.0, share_embeddings=True),
	"8B": Qwen3ModelArgs(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, rope_theta=10000.0, share_embeddings=True),
}


def build_qwen3_model_args_from_pretrained(model_dir: str) -> Qwen3ModelArgs:
	"""
	Builds Qwen3ModelArgs by reading HF config.json from a local pretrained directory.
	"""
	cfg = AutoConfig.from_pretrained(model_dir)
	# Map HF Qwen3 config to our Qwen3ModelArgs used by Qwen3Transformer
	dim = cfg.hidden_size
	n_layers = cfg.num_hidden_layers
	n_heads = cfg.num_attention_heads
	n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
	rope_theta = float(getattr(cfg, "rope_theta", 1000000.0))
	share_embeddings = bool(getattr(cfg, "tie_word_embeddings", True))
	# Infer ffn dim multiplier if available
	intermediate_size = getattr(cfg, "intermediate_size", 4 * dim)
	ffn_dim_multiplier = intermediate_size / (4 * dim)  # 4 * dim is the default
	# Optional head dim (some Qwen variants specify it explicitly)
	head_dim = getattr(cfg, "head_dim", None)
	# Handle sliding window and layer types
	sliding_window = getattr(cfg, "sliding_window", None)
	layer_types = getattr(cfg, "layer_types", None)
	
	return Qwen3ModelArgs(
		dim=dim,
		n_layers=n_layers,
		n_heads=n_heads,
		n_kv_heads=n_kv_heads,
		rope_theta=rope_theta,
		share_embeddings=share_embeddings,
		ffn_dim_multiplier=ffn_dim_multiplier,
		sliding_window=sliding_window,
		layer_types=layer_types,
		head_dim=head_dim,
	)