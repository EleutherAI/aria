"""Includes (PyTorch) transformer model and config classes."""

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F
from typing import Tuple


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

    def __eq__(self, other):
        try:
            vs_equal = self.vocab_size == other.vocab_size
        except:
            # Default behaviour if one/both is missing
            vs_equal = True

        return (
            vs_equal
            and self.d_model == other.d_model
            and self.n_heads == other.n_heads
            and self.n_layers == other.n_layers
            and self.ff_mult == other.ff_mult
            and self.drop_p == other.drop_p
            and self.max_seq_len == other.max_seq_len
            and self.grad_checkpoint == other.grad_checkpoint
        )


# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


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

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self._att_block(self.norm1(x), freqs_cis)
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        # Reshape for rotary embeddings
        xq = xq.view(batch_size, seq_len, self.n_heads, self.d_head)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.d_head)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

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

        # Used for Rotary Embeddings - see LLaMA
        d_head = model_config.d_model // model_config.n_heads
        max_seq_len = model_config.max_seq_len
        self.register_buffer(
            "freqs_cis",
            self._precompute_freqs_cis(d_head, max_seq_len),
        )

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

        # Slices freqs_cis (pos embeddings) according to src seq_len
        assert src.shape[1] <= self.model_config.max_seq_len, "Too long."
        freqs_cis = torch.view_as_complex(self.freqs_cis)[: src.shape[1]]

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
                    freqs_cis,
                    preserve_rng_state=True,
                )

        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states, freqs_cis)

        return self.out_layer_norm(hidden_states)

    # Taken from facebookresearch/llama/model.py
    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

        # Workaround see - https://github.com/pytorch/pytorch/issues/71613
        return torch.view_as_real(freqs_cis)


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
