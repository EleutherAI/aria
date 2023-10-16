"""Contains generation/sampling code"""

# This file contains code from https://github.com/facebookresearch/llama which
# is available under the following licence:

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU
# General Public License version 3.

import torch

from typing import List

from aria.model import TransformerLM
from aria.tokenizer import Tokenizer

# TODO:
# - Enable sampling sequences longer than max_seq_len by truncating


@torch.autocast(device_type="cuda", dtype=torch.float16)
def greedy_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_seq_len: int,
    max_gen_len: int,
    force_end=False,
    temperature: float = 0.85,
    top_p: float = 0.95,
):
    """Performs greedy (top_p) autoregressive sampling on a batch of prompts.

    Args:
        model (TransformerLM): Model to sample from.
        tokenizer (Tokenizer): Tokenizer corresponding to model.
        prompts (List[list]): A list of prompts to sample as a batch.
        max_seq_len (int): Maximum sequence length supported by the model.
        max_gen_len (int): Maximum desired sequence length of the samples.
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

    if force_end:
        assert (
            total_len - max_prompt_size > 130
        ), "prompt too long to use force_end=True"

    tokens = torch.full((bsz, total_len), pad_id).cuda()
    for idx, unencoded_seq in enumerate(prompts):
        tokens[idx, : len(unencoded_seq)] = tokenizer.encode(unencoded_seq)

    dim_tok_inserted = [False for _ in range(bsz)]
    input_text_mask = tokens != pad_id
    start_pos = min_prompt_size
    for cur_pos in range(start_pos, total_len):
        logits = model.forward(tokens[:, :cur_pos])[:, -1, :]

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
