from typing import Tuple, Optional

import torch
import math
from aria.model.utils import apply_rotary_pos_emb


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1.0, coeff=0.1):
    if scale <= 1:
        return 1.0
    return coeff * math.log(scale) + 1.0


class YaRNScaledRotaryEmbedding(torch.nn.Module):
    """
    Adapted from:
    https://github.com/jquesnelle/yarn/blob/master/scaled_rope/modeling_llama_together_yarn.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        pos_idx_in_fp32=True,
        original_context_length=2048,
        scaling_factor=1.0,
        extrapolation_factor=1.0,
        attn_factor=1.0,
        mscale_coeff=0.1,
        beta_fast=32,
        beta_slow=1,
        dynamic=False,
        finetuned=False,
        device=None,
    ):
        """
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
        """
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.original_context_length = original_context_length
        self.scaling_factor = scaling_factor

        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.mscale_coeff = mscale_coeff
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.mscale_coeff)
            * attn_factor
        )
        self.dynamic = dynamic
        self.finetuned = finetuned

        # Generate and save the inverse frequency buffer (non-trainable)
        if not dynamic:
            self._compute_inv_freq(self.scaling_factor, device)
        else:
            self._compute_inv_freq_original(device)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, scaling_factor, device=None):
        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2).float().to(device) / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_context_length,
        )
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.dim // 2)
            .float()
            .to(device)
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        self.register_buffer("inv_freq", inv_freq)

    def _compute_inv_freq_original(self, device=None):
        inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    def _update_cos_sin_cache(self, seq_len, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len

            if self.dynamic:
                scaling_factor = None
                if (
                    seq_len
                    <= self.original_context_length * self.scaling_factor
                ):
                    if self.finetuned:
                        scaling_factor = self.scaling_factor
                else:
                    scaling_factor = seq_len / (
                        self.original_context_length
                    )
                if scaling_factor:
                    self._compute_inv_freq(scaling_factor, device)
                    self.mscale = float(
                        _yarn_get_mscale(scaling_factor, self.mscale_coeff)
                        * self.attn_factor
                    )
                else:
                    self._compute_inv_freq_original(device)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(
                    seq_len, device=device, dtype=self.inv_freq.dtype
                )
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_positions: Optional[torch.Tensor] = None,
        past_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, q_len, n_heads, head_dim)
            k: (batch, k_len, n_heads, head_dim)
            input_positions: (batch, *)
            past_len: the length before the second axis of q (usually it is just the kv length)
        """
        self._update_cos_sin_cache(
            max(
                q.size(1) + past_len,
                self.original_context_length * self.scaling_factor,
            ),
            device=q.device,
            dtype=q.dtype,
        )
        return apply_rotary_pos_emb(
            q,
            self._cos_cached[past_len : past_len + q.size(1)],
            self._sin_cached[past_len : past_len + q.size(1)],
        ), apply_rotary_pos_emb(
            k,
            self._cos_cached[past_len : past_len + k.size(1)],
            self._sin_cached[past_len : past_len + k.size(1)],
        )
