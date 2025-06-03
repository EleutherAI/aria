"""Inference implementation for mlx backend"""

import mlx.core as mx
import mlx.nn as nn

from aria.model import ModelConfig


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        self.dtype = dtype
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.k_cache = mx.zeros(cache_shape, dtype=dtype)
        self.v_cache = mx.zeros(cache_shape, dtype=dtype)

    def update(self, input_pos: mx.array, k_val: mx.array, v_val: mx.array):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.d_model = model_config.d_model
        self.n_heads = model_config.n_heads
        self.d_head = self.d_model // self.n_heads
        self.max_seq_len = model_config.max_seq_len
        self.scale = self.d_head**-0.5

        # Att
        self.mixed_qkv = nn.Linear(
            input_dims=model_config.d_model,
            output_dims=3 * model_config.d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            input_dims=model_config.d_model,
            output_dims=model_config.d_model,
            bias=False,
        )

        # FF
        self.ff_gate_proj = nn.Linear(
            input_dims=model_config.d_model,
            output_dims=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_up_proj = nn.Linear(
            input_dims=model_config.d_model,
            output_dims=model_config.d_model * model_config.ff_mult,
            bias=False,
        )
        self.ff_down_proj = nn.Linear(
            input_dims=model_config.d_model * model_config.ff_mult,
            output_dims=model_config.d_model,
            bias=False,
        )

        # Pre layer norms
        self.norm1 = nn.LayerNorm(model_config.d_model)
        self.norm2 = nn.LayerNorm(model_config.d_model)

        self.kv_cache = None

    def __call__(
        self,
        x: mx.array,
        input_pos: mx.array,
        offset: int,
        mask: mx.array,
    ):
        assert self.kv_cache is not None, "Cache not initialized"

        x += self._att_block(
            x=self.norm1(x),
            input_pos=input_pos,
            offset=offset,
            mask=mask,
        )
        x = x + self._ff_block(self.norm2(x))

        return x

    def get_kv(self, k: mx.array, v: mx.array, input_pos: mx.array):
        k, v = self.kv_cache.update(k_val=k, v_val=v, input_pos=input_pos)

        return k, v

    def _att_block(
        self,
        x: mx.array,
        input_pos: mx.array,
        offset: int,
        mask: mx.array,
    ):

        qkv_splits = self.mixed_qkv(x).split(3, axis=2)
        q, k, v = qkv_splits[0], qkv_splits[1], qkv_splits[2]

        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        q = apply_rotary_emb_mlx(q, offset=offset)
        k = apply_rotary_emb_mlx(k, offset=offset)
        q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))

        k, v = self.get_kv(k, v, input_pos=input_pos)
        wv = mx.fast.scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            scale=self.scale,
            mask=mask,
        )

        # (bz, nh, L, dh) -> (bz, L, nh, dh) -> (bz, L, d)
        wv = wv.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.n_heads * self.d_head
        )

        return self.att_proj_linear(wv)

    def _ff_block(self, x: mx.array):
        return self.ff_down_proj(
            nn.silu(self.ff_gate_proj(x)) * self.ff_up_proj(x)
        )


class Transformer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.tok_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            dims=model_config.d_model,
        )
        self.encode_layers = [
            TransformerBlock(model_config) for _ in range(model_config.n_layers)
        ]
        self.out_layer_norm = nn.LayerNorm(model_config.d_model)

    def fill_condition_kv(self, emb: mx.array):
        assert self.causal_mask is not None, "Caches must be initialized first"
        assert self.model_config.emb_size is not None

        input_pos = mx.array([0], dtype=mx.int32)
        mask = self.causal_mask[None, None, input_pos]
        offset = 0

        x = mx.expand_dims(emb, axis=1)

        for layer in self.encode_layers:
            x = layer(x, input_pos, offset, mask)

        self.causal_mask = None

    def __call__(
        self,
        idxs: mx.array,
        input_pos: mx.array,
        offset: int,
        pad_idxs: mx.array | None = None,
    ):
        assert self.causal_mask is not None, "Caches must be initialized first"

        mask = self.causal_mask[None, None, input_pos]

        if pad_idxs is not None:
            pad_mask = mx.expand_dims(mx.expand_dims(pad_idxs, axis=1), axis=1)
            mask = mask & ~pad_mask

        x = self.tok_embeddings(idxs)
        for layer in self.encode_layers:
            x = layer(x, input_pos, offset, mask)

        x = self.out_layer_norm(x)

        return x


class TransformerLM(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.max_seq_len = model_config.max_seq_len
        self.model = Transformer(model_config)  # Implement
        self.lm_head = nn.Linear(
            model_config.d_model, model_config.vocab_size, bias=False
        )

        if model_config.emb_size is not None:
            self.embedding_adapter = nn.Linear(
                model_config.emb_size, model_config.d_model, bias=False
            )

    def __call__(
        self,
        idxs: mx.array,
        input_pos: mx.array,
        offset: int,
        pad_idxs: mx.array | None = None,
    ):
        hidden_states = self.model(
            idxs=idxs,
            input_pos=input_pos,
            offset=offset,
            pad_idxs=pad_idxs,
        )
        logits = self.lm_head(hidden_states)

        return logits

    def fill_condition_kv(self, cond_emb: mx.array):
        assert self.model_config.emb_size is not None

        adapted_emb = self.embedding_adapter(cond_emb)
        self.model.fill_condition_kv(emb=adapted_emb)

    def setup_cache(
        self,
        batch_size,
        max_seq_len=8096,
        dtype=mx.float32,
    ):
        # Init cache
        for b in self.model.encode_layers:
            b.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_length=max_seq_len,
                n_heads=self.model_config.n_heads,
                head_dim=self.model_config.d_model // self.model_config.n_heads,
                dtype=dtype,
            )

        self.model.causal_mask = mx.tril(
            mx.ones((max_seq_len, max_seq_len), dtype=mx.bool_)
        )


def apply_rotary_emb_mlx(
    x: mx.array,
    offset: int = 0,
) -> mx.array:
    # Original x shape: (b_sz, s_len, n_head, d_head)
    original_shape = x.shape
    b_sz, s_len, n_head, d_head = original_shape

    # Transpose to (b_sz, n_head, s_len, d_head)
    x_permuted = x.transpose(0, 2, 1, 3)
    # Reshape for mx.fast.rope: (b_sz * n_head, s_len, d_head)
    x_reshaped = x_permuted.reshape(-1, s_len, d_head)

    rotated_x_reshaped = mx.fast.rope(
        x_reshaped,
        dims=d_head,
        traditional=False,
        base=500000,
        scale=1.0,
        offset=offset,
    )

    rotated_x_permuted = rotated_x_reshaped.reshape(b_sz, n_head, s_len, d_head)
    rotated_x = rotated_x_permuted.transpose(0, 2, 1, 3)

    return rotated_x
