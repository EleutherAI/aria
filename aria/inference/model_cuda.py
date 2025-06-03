"""Inference implementation with torch-compiler friendly kv-cache."""

import torch
import torch.nn as nn

from torch.nn import functional as F
from aria.model import ModelConfig


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.dtype = dtype
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class TransformerLM(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)
        self.lm_head = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )
        self.embedding_adapter = nn.Linear(
            model_config.emb_size, model_config.d_model, bias=False
        )

    def forward(
        self,
        idxs: torch.Tensor,
        input_pos: torch.Tensor,
        pad_idxs: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            idxs=idxs,
            input_pos=input_pos,
            pad_idxs=pad_idxs,
        )
        logits = self.lm_head(hidden_states)

        return logits

    def fill_condition_kv(self, cond_emb: torch.Tensor):
        adapted_emb = self.embedding_adapter(cond_emb)
        self.model.fill_condition_kv(emb=adapted_emb)

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len=4096,
        dtype=torch.bfloat16,
    ):
        assert batch_size >= 1
        for b in self.model.encode_layers:
            b.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_seq_len,
                n_heads=self.model_config.n_heads,
                head_dim=self.model_config.d_model // self.model_config.n_heads,
                dtype=dtype,
            ).cuda()

        self.model.freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=self.model_config.d_model // self.model_config.n_heads,
            base=500000,
            dtype=dtype,
        ).cuda()
        self.model.causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        ).cuda()


class Transformer(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.d_model,
        )
        self.encode_layers = nn.ModuleList(
            TransformerBlock(model_config) for _ in range(model_config.n_layers)
        )
        self.out_layer_norm = nn.LayerNorm(model_config.d_model)

        self.freqs_cis = None
        self.causal_mask = None

    def fill_condition_kv(self, emb: torch.Tensor):
        assert self.freqs_cis is not None, "Caches must be initialized first"

        input_pos = torch.tensor([0], device=emb.device)
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        x = emb.unsqueeze(dim=1)

        for layer in self.encode_layers:
            x = layer(x, input_pos, freqs_cis, mask)

    def forward(
        self,
        idxs: torch.Tensor,
        input_pos: torch.Tensor,
        pad_idxs: torch.Tensor | None = None,
    ):
        assert self.freqs_cis is not None, "Caches must be initialized first"

        mask = self.causal_mask[None, None, input_pos]

        if pad_idxs is not None:
            mask = mask & ~(pad_idxs.unsqueeze(1).unsqueeze(1))

        freqs_cis = self.freqs_cis[input_pos]

        x = self.tok_embeddings(idxs)
        for layer in self.encode_layers:
            x = layer(x, input_pos, freqs_cis, mask)

        x = self.out_layer_norm(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()

        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = self.d_model // self.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Att
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

        # FF
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

        self.kv_cache = None

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ):
        assert self.kv_cache is not None, "Cache not initialized"

        x += self._att_block(
            x=self.norm1(x),
            input_pos=input_pos,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        x = x + self._ff_block(self.norm2(x))

        return x

    def get_kv(self, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor):
        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=input_pos)

        return k, v

    def _att_block(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ):

        q, k, v = self.mixed_qkv(x).split(
            [self.d_model, self.d_model, self.d_model], dim=-1
        )

        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k, v = self.get_kv(k, v, input_pos=input_pos)
        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(1, 2).reshape(
            batch_size, seq_len, self.n_heads * self.d_head
        )

        return self.att_proj_linear(wv)

    def _ff_block(self, x: torch.Tensor):
        return self.ff_down_proj(
            F.silu(self.ff_gate_proj(x)) * self.ff_up_proj(x)
        )


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


# TODO: Fix
# @torch.jit.script
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
