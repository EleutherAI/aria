import torch


@torch.jit.script
def apply_rotary_pos_emb(
    x, cos, sin, past_len: int = 0, interleave: bool = False
):
    """
    In-place RoPE. Credits to Katherine Crowson:
    x shape (b_sz, s_len, n_head, d_head).
    cos, sin shape (s_len, d_head // 2).
    This implementation tries to use stride tricks to avoid explicit reshapes.
    """
    d = cos.shape[-1]
    cos = cos[None, past_len : past_len + x.size(1), None]
    sin = sin[None, past_len : past_len + x.size(1), None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(tmp, sin, value=1)
    return x
