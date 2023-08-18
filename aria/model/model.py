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
# https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/positional_embeddings.py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        """Returns tuple cos, sin"""
        # Comment out bfloat16() specific code for now
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # if self.precision == torch.bfloat16:
            #     emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            # if self.precision == torch.bfloat16:
            #     self.cos_cached = self.cos_cached.bfloat16()
            #     self.sin_cached = self.sin_cached.bfloat16()

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    """Returns tuple xq, xk"""
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (
        rotate_half(k) * sin
    )


class FusedEncoderBlock(nn.Module):
    """Transformer encoder block using F.scaled_dot_product_attention().

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Layer norm is applied before attention and feed forward, instead of
            after.
        - Keys arising from padding are masked during attention.
        - GELU activation is used instead of ReLU.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Positional embeddings
        self.rotary_emb = RotaryEmbedding(self.d_head)

        # Attention
        self.q = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
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

    def forward(self, x: torch.Tensor):
        x = x + self._att_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)

        cos, sin = self.rotary_emb(x=xv, seq_dim=1, seq_len=seq_len)
        xq, xk = apply_rotary_pos_emb(q=xq, k=xk, cos=cos, sin=sin)

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
        att = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            dropout_p=att_dropout,
            is_causal=True,
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

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

    def forward(self, src: torch.Tensor):
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
                )

        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states)

        return self.out_layer_norm(hidden_states)


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

    def forward(self, src: torch.Tensor):
        """Forward pass of Transformer decoder with LM head.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Forward pass of src through Transformer and LM head.
                Has shape (batch_size, seq_len, vocab_size).
        """
        logits = self.lm_head(self.model(src))

        return logits
