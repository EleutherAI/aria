"""Includes (PyTorch) transformer model and config classes."""

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F


class ModelConfig:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        drop_p: float,
        max_seq_len: int,
        grad_checkpoint: bool,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.drop_p = drop_p
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint

    def set_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size


# Taken from GPT-NeoX see:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        if device is None: # todo: maybe we don't need this...
            device = "cuda" if torch.cuda.is_available() else None

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        #self.cos_cached = emb.cos().to(dtype)
        #self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, past_len: int = 0):
    """Returns tuple (xq, xk). Expects shape (s_len, b_sz, n_head, d_head)."""
    cos = cos[past_len:past_len + q.size(0), None, None]
    sin = sin[past_len:past_len + q.size(0), None, None]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (
        rotate_half(k) * sin
    )


class FusedEncoderBlock(nn.Module):
    """Transformer block using F.scaled_dot_product_attention().

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Layer norm is applied before attention and feed forward, instead of
            after.
        - Keys arising from padding are masked during attention.
        - GELU activation is used instead of ReLU.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig, use_yarn=False):
        super().__init__()

        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Positional embeddings
        if use_yarn:
            # todo: need more testing on this
            self.rotary_emb = DynamicYaRNScaledRotaryEmbedding(self.d_head,
                                                               max_position_embeddings=8192,
                                                               original_max_position_embeddings=2048,
                                                               beta_fast = 16,
                                                               beta_slow = 2)
        else:
            self.rotary_emb = RotaryEmbedding(self.d_head)

        # Attention
        self.mixed_qkv = nn.Linear(
            in_features=model_config.d_model,
            out_features=3 * model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
        )
        self.resid_dropout = nn.Dropout(model_config.drop_p)

        # FF Layer
        self.ff_dropout = nn.Dropout(model_config.drop_p)
        self.ff_linear_1 = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
        )
        self.ff_linear_2 = nn.Linear(
            in_features=model_config.d_model * model_config.ff_mult,
            out_features=model_config.d_model,
        )
        self.ff_activation = nn.GELU()

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

    def forward(self, x: torch.Tensor, use_cache=False, past_kv=None):
        att, kv = self._att_block(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + att
        x = x + self._ff_block(self.norm2(x))

        return x, kv

    def _att_block(self, x: torch.Tensor, use_cache=False, past_kv=None):
        batch_size, seq_len, _ = x.shape
        mixed_qkv = self.mixed_qkv(x)
        xq, xk, xv = mixed_qkv.chunk(3, -1)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)

        past_len = 0 if past_kv is None else past_kv[0].size(1)
        # apply_rotary_post_emb expects: (s_len, b_sz, n_head, d_head)
        cos, sin = self.rotary_emb(x=xv, seq_len=seq_len + past_len)
        xq, xk = xq.transpose(0, 1), xk.transpose(0, 1)
        xq, xk = apply_rotary_pos_emb(q=xq, k=xk, cos=cos, sin=sin, past_len=past_len)
        xq, xk = xq.transpose(0, 1), xk.transpose(0, 1)
        # xq, xk: (b_sz, s_len, n_head, d_head)
        if past_kv is not None:
            assert len(past_kv) == 2
            xk = torch.concat([past_kv[0], xk], axis=1)
            xv = torch.concat([past_kv[1], xv], axis=1)
        kv = (xk, xv)
        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = 0.1  # Bug?
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        # See - https://shorturl.at/jtI17
        if past_kv is None:
            att = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                dropout_p=att_dropout,
                is_causal=True,
            )
        else:
            assert xq.size(2) == 1
            mask = torch.ones(1, xk.size(2), dtype=bool, device=xk.device)
            att = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                dropout_p=att_dropout,
                is_causal=False,
                attn_mask=mask,
            )

        # Reshape for out: (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out)), kv if use_cache else None

    def _ff_block(self, x: torch.Tensor):
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))

        return self.ff_dropout(x)


class Transformer(nn.Module):
    """Transformer decoder with no language model head.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
        )

        self.out_layer_norm = nn.LayerNorm(model_config.d_model)
        self.encode_layers = nn.ModuleList()
        for _ in range(model_config.n_layers):
            self.encode_layers.append(FusedEncoderBlock(model_config))

    def forward(self, src: torch.Tensor, use_cache=False, past_kv=None):
        """Forward pass of Transformer.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Model outputs with shape (batch_size, seq_len,
                d_model).
        """
        hidden_states = self.tok_embeddings(src)

        assert src.shape[1] <= self.model_config.max_seq_len, "Too long."

        # NOTE: If you want to use gradient checkpointing then you must
        # remove torch.compile from the train script as this is not currently
        # supported.
        # Implements gradient checkpoints on Encoder Layers.
        if self.model_config.grad_checkpoint is True:
            for layer in self.encode_layers:

                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    preserve_rng_state=True,
                    use_reentrant=True,
                )

        else:
            new_past_kv = []
            past_kv = [None] * len(self.encode_layers) if past_kv is None else past_kv
            for layer, _kv in zip(self.encode_layers, past_kv):
                hidden_states, kv = layer(hidden_states, use_cache=use_cache, past_kv=_kv)
                new_past_kv.append(kv)

        return self.out_layer_norm(hidden_states), new_past_kv if use_cache else None


class TransformerLM(nn.Module):
    """Transformer decoder with head for language modelling.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.lm_head = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )

    def forward(self, src: torch.Tensor, use_cache=False, past_kv=None):
        """Forward pass of Transformer decoder with LM head.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Forward pass of src through Transformer and LM head.
                Has shape (batch_size, seq_len, vocab_size).
        """
        hidden, past_kv = self.model(src, use_cache=use_cache, past_kv=past_kv)
        logits = self.lm_head(hidden)

        if use_cache:
            return logits, past_kv
        else:
            return logits