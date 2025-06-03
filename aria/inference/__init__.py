import torch

from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict


def sample_min_p(probs: torch.Tensor, p_base: float) -> torch.Tensor:
    """See - https://arxiv.org/pdf/2407.01082"""
    p_max, _ = torch.max(probs, dim=-1, keepdim=True)
    p_scaled = p_base * p_max
    mask = probs >= p_scaled

    masked_probs = probs.clone()
    masked_probs[~mask] = 0.0
    masked_probs.div_(masked_probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(masked_probs, num_samples=1)

    return next_token


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


def get_cfg_prompt(prompts: list):
    cfg_prompts = []
    for prompt in prompts:
        cfg_prompts.append(prompt)
        cfg_prompts.append(prompt)

    return cfg_prompts


def get_inference_prompt(
    midi_dict: MidiDict, tokenizer: AbsTokenizer, prompt_len_ms: int
):
    midi_dict.note_msgs = [
        msg
        for msg in midi_dict.note_msgs
        if midi_dict.tick_to_ms(msg["data"]["start"]) <= prompt_len_ms
    ]

    if len(midi_dict.note_msgs) == 0:
        return [("prefix", "instrument", "piano"), tokenizer.bos_tok]

    seq = tokenizer.tokenize(midi_dict=midi_dict, add_dim_tok=False)
    seq.remove(tokenizer.eos_tok)

    return seq
