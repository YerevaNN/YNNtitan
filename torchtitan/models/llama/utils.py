from transformers import AutoModelForCausalLM
import torch
from torchtitan.models.llama import Transformer
from torchtitan.logging import logger


# reverse_permute for sliced rotary
def reverse_permute(w, n_heads, dim1, dim2):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2): 
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def get_hf_llama3_state_dict_keys_mapping(num_layers: int):
    """
        Get a mapping between state dict keys of different implementations.

        Args:
            num_layers (int): number of transformer layers (blocks).

        Returns:
            dict: mapping between local implementation state dict keys and hf implementation state dict keys

        """
    keys_mapping = {
        'tok_embeddings.weight': 'model.embed_tokens.weight',
        # add layer weight mappings here
        'norm.weight': 'model.norm.weight',
        "output.weight": 'lm_head.weight',
    }
    for layer in range(num_layers):
        keys_mapping.update({
            f'layers.{layer}.attention.wq.weight': f'model.layers.{layer}.self_attn.q_proj.weight',
            f'layers.{layer}.attention.wk.weight': f'model.layers.{layer}.self_attn.k_proj.weight',
            f'layers.{layer}.attention.wv.weight': f'model.layers.{layer}.self_attn.v_proj.weight',
            f'layers.{layer}.attention.wo.weight': f'model.layers.{layer}.self_attn.o_proj.weight',
            f'layers.{layer}.feed_forward.w1.weight': f'model.layers.{layer}.mlp.gate_proj.weight',
            f'layers.{layer}.feed_forward.w3.weight': f'model.layers.{layer}.mlp.up_proj.weight',
            f'layers.{layer}.feed_forward.w2.weight': f'model.layers.{layer}.mlp.down_proj.weight',
            f'layers.{layer}.attention_norm.weight': f'model.layers.{layer}.input_layernorm.weight',
            f'layers.{layer}.ffn_norm.weight': f'model.layers.{layer}.post_attention_layernorm.weight'
        })

    return keys_mapping


def download_llama3_weights(model: Transformer, weights_path: str, source: str, token_embedding_size: int):
    """
        write docs
    """
    if source == "huggingface":
        hf_model = AutoModelForCausalLM.from_pretrained(weights_path)
        # hf_model.resize_token_embeddings(new_num_tokens=token_embedding_size)
        keys_mapping = get_hf_llama3_state_dict_keys_mapping(model.n_layers)
        hf_state_dict = hf_model.state_dict()
        corrected_state_dict = {}
        for key, value in keys_mapping.items():
            assert hf_state_dict[value].shape == model.state_dict()[key].shape
            if "self_attn.q_proj.weight" in value:
                corrected_state_dict[key] = reverse_permute(
                    hf_state_dict[value], model.model_args.n_heads,
                    model.model_args.dim, model.model_args.dim
                )
            elif "self_attn.k_proj.weight" in value:
                kv_dim = model.model_args.dim // (model.model_args.n_heads // model.model_args.n_kv_heads)
                corrected_state_dict[key] = reverse_permute(
                    hf_state_dict[value], model.model_args.n_kv_heads,
                    kv_dim, model.model_args.dim
                )
            else:
                corrected_state_dict[key] = hf_state_dict[value]
        
        with torch.device(model.freqs_cis.device):
            corrected_state_dict["freqs_cis"] = model._precompute_freqs_cis()

        model.load_state_dict(corrected_state_dict)
        logger.info("Successfully loaded Llama 3 model to the titan model.")

        # from transformers import AutoTokenizer
        # tok = AutoTokenizer.from_pretrained(weights_path)
        # device = "cuda"
        # hf_model.to(device)
        # model.eval()
        # text = "Hello world"
        # data = tok(text, return_tensors="pt").to(device)
        # hf_logits = hf_model(**data).logits
        # logits = model(data.input_ids)
        # print(torch.allclose(hf_logits, logits, atol=1e-4))
    else:
        raise NotImplemented
