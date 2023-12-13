"""Includes (PyTorch) transformer model and config classes."""
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.utils.checkpoint

from torch import nn as nn
from torch.nn import functional as F
from aria.model.yarn_rotary_embedding import YaRNScaledRotaryEmbedding
from aria.model.cache import KVCache


@dataclass
class YaRNConfig:
    """
    Config for Dynamic YaRN rotary embeddings.
    The default values are manually tuned for a "large" model (~400M params) we trained.

    Args:
        beta_fast (int): Fast beta value.
        beta_slow (int): Slow beta value. Along with beta_fast they determine
            the "ramp" between PI and NTK.
        scale (int): Scaling factor. In the paper, it is denoted by `s`.
        mscale_coeff (int): Temperature scaling factor t follows `a ln s + 1.0`,
            and the coefficient `a` is this `mscale_coeff` here.
    """

    beta_fast: int = 16
    beta_slow: int = 1
    # `max_len * scale` would be the actual max context length for the run
    scale: float = 1.0
    mscale_coeff: float = 0.1
    base: float = 10000.0
    # Whether the underlying weights are already finetuned with YaRN
    finetuned: bool = False
    # Whether to use dynamic YaRN beyond the context length * scale
    dynamic: bool = True


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    ff_mult: int
    drop_p: float
    max_seq_len: int  # The original context length *WITHOUT* considering YaRN
    grad_checkpoint: bool
    yarn_config: Optional[Union[dict, YaRNConfig]] = None
    vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.yarn_config is not None and isinstance(self.yarn_config, dict):
            self.yarn_config = YaRNConfig(**self.yarn_config)

    def set_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size


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

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.drop_p = model_config.drop_p
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_model // model_config.n_heads
        self.max_seq_len = model_config.max_seq_len

        # Positional embeddings
        cfg = model_config.yarn_config or YaRNConfig()
        self.rotary_emb = YaRNScaledRotaryEmbedding(
            self.d_head,
            original_context_length=self.max_seq_len,
            scaling_factor=cfg.scale,
            beta_fast=cfg.beta_fast,
            beta_slow=cfg.beta_slow,
            base=cfg.base,
            mscale_coeff=cfg.mscale_coeff,
            finetuned=cfg.finetuned,
            dynamic=cfg.dynamic,
        )

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

    def forward(self, x: torch.Tensor, attn_mask=None, past_kv=None):
        att = self._att_block(
            self.norm1(x), attn_mask=attn_mask, past_kv=past_kv
        )
        x = x + att
        x = x + self._ff_block(self.norm2(x))

        return x

    @staticmethod
    def _create_mask(
        q_len: int,
        k_len: int,
        attn_mask: Optional[torch.Tensor] = None,
        device=None,
    ):
        # Could have cached some of these masks (not the entire (seq_len, seq_len)!!).
        # But profiler seems to show that their impact is negligible.

        # attn_mask: (b_sz, k_len)
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
        mask = torch.tril(mask, diagonal=k_len - q_len)
        if attn_mask is not None:
            # (1, q_len, k_len) & (b_sz, 1, k_len)
            mask = mask[None, ...] & attn_mask[:, None, :]
            return mask[:, None]
        else:
            return mask

    def _att_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        input_positions: Optional[torch.Tensor] = None,
        max_pos: Optional[int] = None,
        past_kv: Optional[KVCache] = None,
    ):
        """
        Args:
            x: (b_sz, s_len, d_model)
            attn_mask: (b_sz, s_len). The attention mask. `False` masks the column(keys)
                in the attention matrix.
            input_positions: (s_len,). The absolute position of each token.
                If None, we assume that the input positions are contiguous.
            max_pos: The maximum position of the input. Only used when input_positions
                is not None. Can be inferred as input_positions.max(), but such an
                operation makes the cache update slower due to dynamic shape.
            past_kv: A KVCache object.
        """
        batch_size, seq_len, _ = x.shape
        past_len = 0 if past_kv is None else past_kv.next_pos

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
        xq, xk = self.rotary_emb(
            xq, xk, input_positions=input_positions, past_len=past_len
        )
        # xq, xk: (b_sz, s_len, n_head, d_head)
        if past_kv is not None:
            xk, xv = past_kv.update(
                xk, xv, pos=input_positions, max_pos=max_pos
            )

        # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
        xq, xk, xv = map(lambda t: t.transpose(1, 2), (xq, xk, xv))

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = 0.1  # Bug?
        else:
            att_dropout = 0.0

        # Calculate attention
        # Note: we avoid explicitly saving a (seq_len, seq_len) cache in order to
        #       save vRAM.
        if past_kv is None and attn_mask is None:
            att = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                dropout_p=att_dropout,
                is_causal=True,
            )
        else:
            mask = self._create_mask(
                xq.size(2), xk.size(2), attn_mask=attn_mask, device=xk.device
            )
            att = F.scaled_dot_product_attention(
                query=xq,
                key=xk,
                value=xv,
                dropout_p=att_dropout,
                is_causal=False,
                attn_mask=mask,
            )
            # If masked token show up in query, they come out as nan. Need to set to zero.
            att = torch.nan_to_num(att, nan=0.0)

        # Reshape for out: (b_sz, s_len, n_head, d_head)
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

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[KVCache]] = None,
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
                    attn_mask,
                    preserve_rng_state=True,
                    use_reentrant=True,
                )
        else:
            past_kv = (
                [None] * len(self.encode_layers) if past_kv is None else past_kv
            )
            for layer, _kv in zip(self.encode_layers, past_kv):
                hidden_states = layer(
                    hidden_states, attn_mask=attn_mask, past_kv=_kv
                )

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

    def forward(
        self,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[KVCache]] = None,
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
        hidden = self.model(src, attn_mask=attn_mask, past_kv=past_kv)
        logits = self.lm_head(hidden)

        return logits

    def get_cache(
        self, max_batch_size: int = 16, max_len: int = 2048, device=None
    ):
        """
        Initialize an empty kv cache according to the model parameters.
        We do not make KVCache a part of the model because one may apply techniques
        such as CFG utilizing multiple caches.
        """
        return [
            KVCache(
                max_batch_size=max_batch_size,
                max_size=max_len,
                n_head=self.model.model_config.n_heads,
                d_head=self.model.model_config.d_model
                // self.model.model_config.n_heads,
                dtype=next(self.parameters()).dtype,
            ).to(device)
            for _ in range(self.model.model_config.n_layers)
        ]
