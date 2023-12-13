"""Contains generation/sampling code"""
# This file contains code from https://github.com/facebookresearch/llama which
# is available under the following licence:

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.

import math
import torch

from typing import List
from tqdm import tqdm

from aria.model import TransformerLM
from aria.tokenizer import Tokenizer


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


def _process_prompts(
    prompts,
    pad_token="<P>",
    neg_prompts=None,
    use_cfg=False,
    neg_prompt_len=None,
) -> list:
    """
    Preprocess prompts for generation.
    If cfg is used,
        1. the prompts and negative prompts will be combined.
        2. the negative prompts will be truncated for at most as long as the longest prompt.
    Args:
        prompts: list of prompts
        pad_token: pad token ('<P>')
        neg_prompts: list of negative prompts
        use_cfg: whether to use cfg
        neg_prompt_len: max length of negative prompts. If more than the longest prompt,
                        pad to this length.
    Returns:
        list of padded prompts
    """
    processed_prompts = []
    max_len = max(len(t) for t in prompts)
    pad_len = max(max_len, neg_prompt_len or 0)

    if use_cfg:
        if neg_prompts is None:
            neg_prompts = [t[-1:] for t in prompts]
        assert len(prompts) == len(
            neg_prompts
        ), "Prompts and neg_prompts must have the same count."

        for prompt in prompts + neg_prompts:
            processed_prompts.append(
                [pad_token] * max(0, pad_len - len(prompt)) + prompt[:pad_len]
            )
    else:
        max_len = max(len(t) for t in prompts)
        for prompt in prompts:
            processed_prompts.append(
                [pad_token] * (max_len - len(prompt)) + prompt
            )

    return processed_prompts


def _batch_encode(tokenizer, prompts: list[list]) -> torch.Tensor:
    return torch.stack([tokenizer.encode(p) for p in prompts], dim=0)


# Some good settings:
# temp=0.85, top_p=0.9, cfg_gamma=1.4


@torch.no_grad()
def greedy_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_new_tokens: int,
    device: torch.device | None = None,
    cfg_gamma: float | None = 1.4,
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
        max_new_tokens (int): Maximum number of new generated tokens.
        device (torch.device, optional): Device to use. Defaults to None.
        cfg_gamma (float, optional): CFG gamma parameter. Defaults to 1.2.
            This parameter *determines* whether parameters related to CFG are used.
            None: No CFG or interpolation. `cfg_mode, neg_prompts, neg_prompt_len, alpha` are ignored.
        cfg_mode (str, optional): CFG mode. Defaults to None (applying constant CFG strength).
            "linear": linearly increasing/decreasing CFG strength from 1.
            "hat": piecewise-linearly scale CFG gamma: 1 -> gamma -> 1
            "sine": sine curve from 1 -> gamma -> 1
        neg_prompts (List[list], optional): Alternative prompts to sample from.
            Defaults to None ("unconditioned" model is approximated using only the last tokens of prompts).
        neg_prompt_len (int, optional): Max length used for the negative prompts.
            Defaults to None (align to prompts).
            When set, if `neg_prompt_len > max(t for t in prompts)`, we pad to `neg_prompt_len`.
        alpha (float, optional): an alpha parameter during interpolation.
            Only takes effect when neg_prompt_len < minimal length of neg_prompts. Defaults to 0.4.
        force_end (bool, optional): Whether to force the end of the prompt. Defaults to False.
        temperature (float, optional): Sampling temperature. Defaults to 0.75.
        top_p (float, optional): Parameter for top-p sampling. Defaults to 0.95.

    Returns:
        List[list]: The list of samples, decoded by the tokenizer.
    """
    assert tokenizer.return_tensors is True, "tokenizer must return tensors."
    device = device or torch.device("cuda")
    model.eval()

    pad_id = tokenizer.pad_id
    pad_tok = tokenizer.pad_tok
    eos_id = tokenizer.tok_to_id[tokenizer.eos_tok]

    padded_combined_prompts = _process_prompts(
        prompts,
        pad_tok,
        neg_prompts,
        cfg_gamma is not None,
        neg_prompt_len=neg_prompt_len,
    )
    if neg_prompts is not None:
        padded_negative_prompts = _process_prompts(
            neg_prompts, pad_tok, None, False
        )
    else:
        padded_negative_prompts = [t[-1:] for t in prompts]

    prompt_len = len(padded_combined_prompts[0])
    if neg_prompts is not None:
        neg_offset_len = max(0, prompt_len - max(len(t) for t in prompts))
    else:
        neg_offset_len = 0

    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    print(
        f"Using hyperparams: temp={temperature}, top_p={top_p}, gamma={cfg_gamma}, gen_len={max_new_tokens}"
    )

    total_len = prompt_len + max_new_tokens
    tokens = torch.full(
        (len(padded_combined_prompts), total_len), pad_id, device=device
    )
    tokens[:, :prompt_len] = _batch_encode(
        tokenizer, padded_combined_prompts
    ).to(device)
    full_neg_tokens = _batch_encode(tokenizer, padded_negative_prompts).to(
        device
    )

    dim_tok_inserted = [False for _ in range(tokens.size(0))]
    attn_mask = torch.ones(
        (len(padded_combined_prompts), total_len),
        device=device,
        dtype=torch.bool,
    )
    attn_mask[:, :prompt_len] = tokens[:, :prompt_len] != pad_id
    start_pos = prompt_len

    past_kv = model.get_cache(
        max_batch_size=tokens.size(0), max_len=total_len, device=device
    )

    for cur_pos in (
        pbar := tqdm(
            range(start_pos, total_len),
            total=total_len - start_pos,
            leave=False,
        )
    ):
        if cur_pos == start_pos:
            token = tokens[:, :start_pos]
        else:
            token = tokens[:, cur_pos - 1 : cur_pos]

        logits = model.forward(
            token, attn_mask=attn_mask[:, :cur_pos], past_kv=past_kv
        )
        logits = logits[:, -1, :]

        if cfg_gamma is not None:
            coeff = _get_cfg_coeff(
                cfg_gamma, cfg_mode, cur_pos, start_pos, total_len
            )
            cond_logits = logits[: logits.size(0) // 2]
            uncond_logits = logits[logits.size(0) // 2 :]
            logits = uncond_logits + coeff * (cond_logits - uncond_logits)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        # When alpha is used, in the first `max_new_tokens * alpha` generations, the negative
        # prompt completions still use its original content (if not exceeding). After that, the
        # negative prompt completions will be updated by the new tokens.
        if (
            alpha is not None
            and cur_pos - neg_offset_len < full_neg_tokens.size(0)
            and cur_pos - start_pos < max_new_tokens * alpha
        ):
            neg_slice = full_neg_tokens[:, cur_pos - neg_offset_len]
            next_token = torch.concat([next_token, neg_slice], dim=0)
        else:
            if cfg_gamma is not None:
                next_token = next_token.repeat(2)  # Also update neg prompts

        # Insert dim tokens
        if force_end and cur_pos >= total_len - 130:
            for _idx in range(tokens.size(0)):
                if (
                    dim_tok_inserted[_idx] is False
                    and tokenizer.id_to_tok[next_token[_idx].item()][0] != "dur"
                ):
                    next_token[_idx] = tokenizer.tok_to_id[tokenizer.dim_tok]

        # Update dim_tok_inserted
        for _idx in range(tokens.size(0)):
            if next_token[_idx] == tokenizer.tok_to_id[tokenizer.dim_tok]:
                dim_tok_inserted[_idx] = True

        tokens[:, cur_pos] = next_token

    decoded = []
    for idx, seq in enumerate(tokens.tolist()):
        if cfg_gamma is not None and 2 * idx >= tokens.size(0):
            break
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
