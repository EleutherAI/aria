"""Contains generation/sampling code"""

# This file contains code from https://github.com/facebookresearch/llama which
# is available under the following licence:

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.

import math
import torch

from typing import List

from aria.model import TransformerLM
from aria.tokenizer import Tokenizer

# TODO:
# - Enable sampling sequences longer than max_seq_len by truncating


def _get_cfg_coeff(cfg_gamma, cfg_mode, cur_pos, start_pos, total_len):
    if cfg_mode is None:
        return cfg_gamma
    elif cfg_mode == "linear":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        return cfg_gamma * p + (1 - p)
    elif cfg_mode == "hat":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        if 2 * cur_pos < total_len + start_pos:
            return cfg_gamma * 2 * p + (1 - 2 * p)
        else:
            return cfg_gamma * 2 * (1 - p) + (2 * p - 1)
    elif cfg_mode == "sine":
        p = (cur_pos - start_pos) / (total_len - start_pos)
        return (cfg_gamma - 1) * math.sin(p * 3.14159) + 1
    else:
        raise ValueError(f"Unknown cfg_mode: {cfg_mode}")


# Some good settings:
# temp=0.85, top_p=0.9, cfg_gamma=1.4


@torch.autocast(device_type="cuda", dtype=torch.float16)
def greedy_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_seq_len: int,
    max_gen_len: int,
    cfg_gamma: float | None = 1.2,
    cfg_mode: str | None = None,
    neg_prompts: List[list] | None = None,
    neg_prompt_len: int | None = None,
    alpha: float | None = 0.4,
    force_end=False,
    temperature: float = 0.85,
    top_p: float = 0.9,
):
    """Performs greedy (top_p) autoregressive sampling on a batch of prompts.

    Args:
        model (TransformerLM): Model to sample from.
        tokenizer (Tokenizer): Tokenizer corresponding to model.
        prompts (List[list]): A list of prompts to sample as a batch.
        max_seq_len (int): Maximum sequence length supported by the model.
        max_gen_len (int): Maximum desired sequence length of the samples.
        cfg_gamma (float, optional): CFG gamma parameter. Defaults to 1.2.
            This parameter *determines* whether parameters related to CFG are used.
            None: No CFG or interpolation. `cfg_mode, neg_prompts, neg_prompt_len, alpha` are ignored.
        cfg_mode (str, optional): CFG mode. Defaults to None (applying constant CFG strength).
            "linear": linearly increasing/decreasing CFG strength from 1.
            "hat": piecewise-linearly scale CFG gamma: 1 -> gamma -> 1
            "sine": sine curve from 1 -> gamma -> 1
        neg_prompts (List[list], optional): Alternative prompts to sample from.
            Defaults to None ("unconditioned" model is approximated using only the last tokens of prompts).
        neg_prompt_len (int, optional): Length of the negative prompts.
            Defaults to None (minimal length of neg_prompts).
        alpha (float, optional): an alpha parameter during interpolation.
            Only takes effect when neg_prompt_len < minimal length of neg_prompts. Defaults to 0.4.
        force_end (bool, optional): Whether to force the end of the prompt. Defaults to False.
        temperature (float, optional): Sampling temperature. Defaults to 0.75.
        top_p (float, optional): Parameter for top-p sampling. Defaults to 0.95.

    Returns:
        List[list]: The list of samples, decoded by the tokenizer.
    """
    assert tokenizer.return_tensors is True, "tokenizer must return tensors."
    model.eval()

    pad_id = tokenizer.pad_id
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]

    bsz = len(prompts)
    min_prompt_size = min([len(t) for t in prompts])
    max_prompt_size = max([len(t) for t in prompts])
    total_len = min(max_seq_len, max_gen_len + max_prompt_size)

    if cfg_gamma is not None:
        # todo: maybe it already works with varying prompts
        assert (
            min_prompt_size == max_prompt_size
        ), "CFG not supported with varying prompts"
        if neg_prompts is None:
            neg_prompts = [prompts[-1] for _ in range(bsz)]

    if force_end:
        assert (
            total_len - max_prompt_size > 130
        ), "prompt too long to use force_end=True"

    print(
        f"Using hyperparams: temp={temperature}, top_p={top_p}, gamma={cfg_gamma}, gen_len={max_gen_len}"
    )

    neg_min_len = min(total_len, min(len(a) for a in neg_prompts))
    neg_max_len = max(total_len, max(len(a) for a in neg_prompts))
    neg_prompt_tensors = torch.stack(
        [
            torch.concat(
                [torch.full((neg_max_len - len(neg_seq),), pad_id), tokenizer.encode(neg_seq)]
            ) for neg_seq in neg_prompts
        ], axis=0
    ).cuda()
    neg_len = neg_min_len if neg_prompt_len is None else min(neg_min_len, neg_prompt_len)
    neg_tokens = neg_prompt_tensors[:, :neg_len]

    tokens = torch.full((bsz, total_len), pad_id).cuda()
    for idx, (unencoded_seq, neg_seq) in enumerate(zip(prompts, neg_prompts)):
        tokens[idx, : len(unencoded_seq)] = tokenizer.encode(unencoded_seq)

    dim_tok_inserted = [False for _ in range(bsz)]
    input_text_mask = tokens != pad_id
    start_pos = min_prompt_size

    past_kv = None
    cfg_kv = None
    neg_previous_token = None

    with torch.inference_mode():
        for cur_pos in range(start_pos, total_len):
            token = tokens[:, :start_pos] if cur_pos == start_pos else tokens[:, cur_pos-1:cur_pos]
            logits, past_kv = model.forward(token, use_cache=True, past_kv=past_kv)
            logits = logits[:, -1, :]
            if cfg_gamma is not None and max_prompt_size < cur_pos:
                coeff = _get_cfg_coeff(cfg_gamma, cfg_mode, cur_pos, start_pos, total_len)

                if cur_pos == start_pos:
                    neg_tok = neg_tokens
                elif neg_previous_token is None:
                    neg_tok = tokens[:, cur_pos-1:cur_pos]
                else:
                    neg_tok = neg_previous_token[:, None]
                uncond_logits, cfg_kv = model.forward(neg_tok, use_cache=True, past_kv=cfg_kv)
                uncond_logits = uncond_logits[:, -1, :]
                logits = uncond_logits + coeff * (logits - uncond_logits)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # Only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            # Insert dim tokens
            if force_end and cur_pos >= total_len - 130:
                for _idx in range(bsz):
                    if (
                        dim_tok_inserted[_idx] is False
                        and tokenizer.id_to_tok[next_token[_idx].item()][0] != "dur"
                    ):
                        next_token[_idx] = tokenizer.tok_to_id[tokenizer.dim_tok]

            # Update dim_tok_inserted
            for _idx in range(bsz):
                if next_token[_idx] == tokenizer.tok_to_id[tokenizer.dim_tok]:
                    dim_tok_inserted[_idx] = True

            tokens[:, cur_pos] = next_token
            if alpha is not None and cur_pos - start_pos < neg_max_len - neg_len and cur_pos < (1 - alpha) * total_len + alpha * start_pos:
                _neg_tokens = neg_prompt_tensors[:, cur_pos - start_pos + neg_len]
                neg_previous_token = torch.where(_neg_tokens != pad_id, _neg_tokens, next_token)
            else:
                neg_previous_token = next_token



    decoded = []
    for idx, seq in enumerate(tokens.tolist()):
        # Cut to max gen len
        seq = seq[: len(prompts[idx]) + max_gen_len]
        # Cut to eos tok if any
        try:
            seq = seq[: seq.index(eos_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(seq))

    return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
