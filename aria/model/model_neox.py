from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import nn
import torch
from typing import Optional
from aria.model import TransformerLM, ModelConfig
from aria.model.cache import KVCache
from aria.tokenizer import AbsTokenizer


class GPTNeoXAria(TransformerLM):
    """A wrapper for GPTNeoXForCausalLM."""

    def __init__(self, model_config: ModelConfig, use_cache: bool = False):
        super(TransformerLM, self).__init__()
        if model_config.yarn_config is not None:
            raise NotImplementedError("YaRN is not supported yet.")
        vocab_size = AbsTokenizer().vocab_size
        config = GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=model_config.d_model,
            num_hidden_layers=model_config.n_layers,
            num_attention_heads=model_config.n_heads,
            intermediate_size=model_config.d_model * model_config.ff_mult,
            hidden_act="gelu",
            rotary_pct=1.0,
            rotary_emb_base=10000,
            max_position_embeddings=model_config.max_seq_len,
            use_cache=use_cache,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            rope_scaling=None,
            attention_bias=True,
        )
        self.model = GPTNeoXForCausalLM(config)
        self.model.model_config = model_config
        if model_config.grad_checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[KVCache]] = None,
    ):
        if past_kv is None:
            output = self.model(src, attention_mask=attn_mask)
        else:
            bs = src.size(0)
            hf_past_kv = tuple(
                (kv.k_cache[:bs, : kv.next_pos], kv.v_cache[:bs, : kv.next_pos])
                for i, kv in enumerate(past_kv)
            )
            output: CausalLMOutputWithPast = self.model(
                src, attention_mask=attn_mask, past_key_values=hf_past_kv
            )
            if output.past_key_values is not None:
                for i, kv in enumerate(past_kv):
                    kv.update(
                        output.past_key_values[i][0][:, kv.next_pos :],
                        output.past_key_values[i][1][:, kv.next_pos :],
                    )
        return output.logits
