"""Contains generation/sampling code (mlx)"""

import torch
import numpy as np
import mlx.core as mx

from tqdm import tqdm

from aria.inference import sample_min_p, sample_top_p
from aria.inference.model_mlx import TransformerLM
from ariautils.tokenizer import Tokenizer

DTYPE = mx.float32


def decode_one(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
    pad_idxs: mx.array | None = None,
):
    assert input_pos.shape[-1] == 1

    logits = model(
        idxs=idxs,
        input_pos=input_pos,
        offset=input_pos[0],
        pad_idxs=pad_idxs,
    )[:, -1]

    return logits


def prefill(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
    pad_idxs: mx.array | None = None,
):
    logits = model(
        idxs=idxs,
        input_pos=input_pos,
        offset=input_pos[0],
        pad_idxs=pad_idxs,
    )

    return logits


def update_seq_ids_(
    seq: mx.array,
    idx: int,
    next_token_ids: mx.array,
    dim_tok_inserted: list,
    eos_tok_seen: list,
    max_len: int,
    force_end: bool,
    tokenizer: Tokenizer,
):
    # Insert dim and pad toks
    for _idx in range(seq.shape[0]):
        if eos_tok_seen[_idx] == True:
            next_token_ids[_idx] = tokenizer.tok_to_id[tokenizer.pad_tok]
        elif (
            force_end
            and idx >= max_len - 130
            and dim_tok_inserted[_idx] is False
            and tokenizer.id_to_tok[next_token_ids[_idx].item()][0]
            not in ("dur", "onset")
        ):
            next_token_ids[_idx] = tokenizer.tok_to_id[tokenizer.dim_tok]

        # Update dim_tok_inserted and eos_tok_seen
        if next_token_ids[_idx] == tokenizer.tok_to_id[tokenizer.dim_tok]:
            dim_tok_inserted[_idx] = True
        elif next_token_ids[_idx] == tokenizer.tok_to_id[tokenizer.eos_tok]:
            eos_tok_seen[_idx] = True

    seq[:, idx] = next_token_ids


def sample_batch(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: list,
    num_variations: list,
    max_new_tokens: int,
    temp: float = 0.95,
    force_end: bool = False,
    top_p: float | None = None,
    min_p: float | None = None,
):
    assert top_p is not None or min_p is not None
    if top_p is not None:
        assert 0.5 <= top_p <= 1.0
    if min_p is not None:
        assert 0.0 <= min_p <= 1.0
    if temp is not None:
        assert 0.0 <= temp <= 2.0
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    prompt_len = len(prompt)

    model.eval()
    dim_tok_inserted = [False for _ in range(num_variations)]
    eos_tok_seen = [False for _ in range(num_variations)]
    total_len = prompt_len + max_new_tokens

    seq = mx.stack(
        [
            mx.array(
                tokenizer.encode(
                    prompt + [tokenizer.pad_tok] * (total_len - prompt_len)
                ),
                dtype=mx.int32,
            )
            for _ in range(num_variations)
        ],
    )
    model.setup_cache(
        batch_size=num_variations,
        max_seq_len=total_len,
        dtype=DTYPE,
    )
    print(
        f"Using hyperparams: temp={temp}, top_p={top_p}, min_p={min_p}, gen_len={max_new_tokens}"
    )

    for idx in (
        pbar := tqdm(
            range(prompt_len, total_len),
            total=total_len - prompt_len,
            leave=False,
        )
    ):
        if idx == prompt_len:
            logits = prefill(
                model,
                idxs=seq[:, :idx],
                input_pos=mx.arange(0, idx),
            )[:, -1]
        else:
            logits = decode_one(
                model,
                idxs=seq[:, idx - 1 : idx],
                input_pos=mx.array(
                    [idx - 1],
                    dtype=mx.int32,
                ),
            )

        if temp > 0.0:
            probs = mx.softmax(logits / temp, axis=-1)
            if min_p is not None:
                next_token_ids = sample_min_p_mlx(probs, min_p).flatten()
            else:
                next_token_ids = sample_top_p_mlx(probs, top_p).flatten()
        else:
            next_token_ids = mx.argmax(logits, axis=-1).flatten()

        update_seq_ids_(
            seq=seq,
            idx=idx,
            next_token_ids=next_token_ids,
            dim_tok_inserted=dim_tok_inserted,
            eos_tok_seen=eos_tok_seen,
            max_len=total_len,
            force_end=force_end,
            tokenizer=tokenizer,
        )

        if all(seen_eos is True for seen_eos in eos_tok_seen):
            break

    decoded_results = [tokenizer.decode(s) for s in seq.tolist()]
    decoded_results = [
        (
            res[: res.index(tokenizer.eos_tok) + 1]
            if tokenizer.eos_tok in res
            else res
        )
        for res in decoded_results
    ]

    return decoded_results


def sample_min_p_mlx(probs: mx.array, p_base: float) -> mx.array:
    """See - https://arxiv.org/pdf/2407.01082"""

    probs_t = torch.from_numpy(np.array(probs))
    next_token_t = sample_min_p(probs=probs_t, p_base=p_base)

    return mx.array(next_token_t, dtype=mx.int32)


def sample_top_p_mlx(probs: mx.array, top_p: float) -> mx.array:

    probs_t = torch.from_numpy(np.array(probs))
    next_token_t = sample_top_p(probs=probs_t, top_p=top_p)

    return mx.array(next_token_t, dtype=mx.int32)
