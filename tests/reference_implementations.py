# Reference implementations from
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py


def rotate_half(x, interleaved=False):
    # Lazy import only when needed
    import torch
    from einops import rearrange

    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_pos_emb_reference(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seq_len, n_heads, head_dim)
    cos, sin: (seq_len, rotary_dim / 2) or (batch_size, seq_len, rotary_dim / 2)
    """
    # Lazy import only when needed
    import torch
    from einops import repeat

    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )
    sin = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos
            + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )