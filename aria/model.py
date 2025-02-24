"""Training implementation."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    drop_p: float
    max_seq_len: int
    grad_checkpoint: bool
    vocab_size: Optional[int] = None
    class_size: Optional[int] = None
    tag_to_id: Optional[dict] = None
    emb_size: Optional[dict] = None

    def set_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size


class FusedEncoderBlock(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Attention
        self.mixed_qkv = nn.Linear(
            in_features=model_config.d_model,
            out_features=3 * model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model,
            bias=False,
        )

        # FF Layer
        self.ff_gate_proj = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_up_proj = nn.Linear(
            in_features=model_config.d_model,
            out_features=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_down_proj = nn.Linear(
            in_features=model_config.d_model * model_config.ff_mult,
            out_features=model_config.d_model,
            bias=False,
        )

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self._att_block(self.norm1(x), freqs_cis)
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        mixed_qkv = self.mixed_qkv(x)
        xq, xk, xv = mixed_qkv.chunk(3, -1)

        # Reshape for rotary embeddings
        # Need contiguous for q, k since in-place RoPE cannot be applied on a view
        xq = xq.reshape(
            batch_size, seq_len, self.n_heads, self.d_head
        ).contiguous()
        xk = xk.reshape(
            batch_size, seq_len, self.n_heads, self.d_head
        ).contiguous()
        xv = xv.view(batch_size, seq_len, self.n_heads, self.d_head)

        # apply_rotary_post_emb expects: (b_sz, s_len, n_head, d_head)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # scaled_dot_product_attention expects: (b_sz, n_head, s_len, d_head)
        att = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            is_causal=True,
        )

        # Reshape for out: (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.d_head)

        return self.att_proj_linear(out)

    def _ff_block(self, x: torch.Tensor):

        return self.ff_down_proj(
            F.silu(self.ff_gate_proj(x)) * self.ff_up_proj(x)
        )


class Transformer(nn.Module):
    """Transformer decoder with no language model head.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.freqs_cis = None

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
        )

        self.out_layer_norm = nn.LayerNorm(model_config.d_model)
        self.encode_layers = nn.ModuleList()
        for _ in range(model_config.n_layers):
            self.encode_layers.append(FusedEncoderBlock(model_config))

    def forward(
        self,
        src: torch.Tensor,
    ):
        """Forward pass of Transformer.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).
            attn_mask (Optional[torch.tensor]): Attention mask of shape
                (batch_size, seq_len). Defaults to None.
            past_kv (Optional[list[KVCache]]): a list of kv caches. The list index
                corresponds to the layer index.

        Returns:
            torch.tensor: Model outputs with shape (batch_size, seq_len,
                d_model).
        """
        hidden_states = self.tok_embeddings(src)

        if self.freqs_cis is None:
            self.freqs_cis = precompute_freqs_cis(
                seq_len=self.model_config.max_seq_len,
                n_elem=self.model_config.d_model // self.model_config.n_heads,
                base=500000,
                dtype=hidden_states.dtype,
            ).to(src.device)
        freqs_cis = self.freqs_cis[: src.shape[1]]

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
                    use_reentrant=True,
                )
        else:
            for layer in self.encode_layers:
                hidden_states = layer(hidden_states, freqs_cis=freqs_cis)

        return self.out_layer_norm(hidden_states)


class TransformerLM(nn.Module):
    """Transformer decoder with head for language modelling.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.vocab_size is not None

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.lm_head = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )

    def forward(
        self,
        src: torch.Tensor,
    ):
        """Forward pass of Transformer decoder with LM head.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).
            attn_mask (Optional[torch.tensor]): Attention mask of shape
                (batch_size, seq_len). Defaults to None.
            past_kv (Optional[list[KVCache]]): a list of kv caches. The list index
                corresponds to the layer index.

        Returns:
            torch.tensor: Forward pass of src through Transformer and LM head.
                Has shape (batch_size, seq_len, vocab_size).
        """
        hidden = self.model(src)
        logits = self.lm_head(hidden)

        return logits


class TransformerCL(nn.Module):
    """Transformer decoder with head for classification.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.class_size is not None

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.class_head = nn.Linear(
            model_config.d_model, model_config.class_size, bias=False
        )

    def forward(
        self,
        src: torch.Tensor,
    ):
        """Forward pass of Transformer decoder with CL head.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.tensor: Forward pass of src through Transformer and CL head.
                Has shape (batch_size, seq_len, class_size).
        """
        hidden = self.model(src)
        logits = self.class_head(hidden)

        return logits


class TransformerEMB(nn.Module):
    """Transformer decoder with head for embedding.

    Args:
        model_config (ModelConfig): Model config settings.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        assert model_config.emb_size is not None

        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.emb_head = nn.Linear(
            model_config.d_model, model_config.emb_size, bias=False
        )

    def forward(
        self,
        src: torch.Tensor,
    ):
        """Forward pass of Transformer decoder with EMB head.

        Args:
            src (torch.tensor): Input to encoder block, of shape (batch_size,
                seq_len, d_model).
        Returns:
            torch.tensor: Forward pass of src through Transformer and EMB head.
                Has shape (batch_size, seq_len, emb_size).
        """
        hidden = self.model(src)
        emb = self.emb_head(hidden)

        return emb


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 500000,
    dtype: torch.dtype = torch.bfloat16,
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    return cache.to(dtype=dtype)


@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    In-place RoPE. Credits to Katherine Crowson:
    x shape (b_sz, s_len, n_head, d_head).
    cos, sin shape (s_len, d_head // 2).
    """

    d = x.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(tmp, sin, value=1)
    return x
