"""Contains generation/sampling code"""

import torch
import torch._dynamo.config
import torch._inductor.config

from typing import List
from tqdm import tqdm

from aria.inference import TransformerLM
from aria.tokenizer import Tokenizer

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


@torch.no_grad()
def prefill(model, idxs: torch.Tensor, input_pos: torch.Tensor):
    logits = model.forward(idxs=idxs, input_pos=input_pos)[:, -1]

    return logits


@torch.no_grad()
def decode_one(model, idxs: torch.Tensor, input_pos: torch.Tensor):
    logits = model.forward(idxs=idxs, input_pos=input_pos)[:, -1]

    return logits


def update_seq_ids_(
    seq: torch.Tensor,
    idx: int,
    next_token_ids: torch.Tensor,
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


# TODO: Add CFG back into this when working
@torch.autocast(
    "cuda",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
@torch.no_grad()
def greedy_sample(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_new_tokens: int,
    force_end=False,
    cfg_gamma: float | None = 1.05,
    temperature: float = 0.95,
    top_p: float = 0.95,
    compile: bool = True,
):
    """Performs greedy (top_p) auto-regressive sampling on a batch of prompts."""

    assert tokenizer.return_tensors is True, "tokenizer must return tensors."
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    _prompt_len = len(prompts[0])
    _num_prompts = len(prompts)

    model.eval()
    total_len = _prompt_len + max_new_tokens
    seq = torch.stack(
        [
            tokenizer.encode(p + [tokenizer.pad_tok] * (total_len - len(p)))
            for p in prompts
        ]
    ).cuda()
    dim_tok_inserted = [False for _ in range(_num_prompts)]
    eos_tok_seen = [False for _ in range(_num_prompts)]

    if compile is True:
        global decode_one
        decode_one = torch.compile(
            decode_one,
            mode="reduce-overhead",
            # mode="max-autotune",
            fullgraph=True,
        )

    model.setup_cache(
        batch_size=_num_prompts,
        max_seq_len=total_len,
        dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )

    print(
        f"Using hyperparams: temp={temperature}, top_p={top_p}, gamma={cfg_gamma}, gen_len={max_new_tokens}"
    )

    for idx in (
        pbar := tqdm(
            range(_prompt_len, total_len),
            total=total_len - _prompt_len,
            leave=False,
        )
    ):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if idx == _prompt_len:
                logits = prefill(
                    model,
                    idxs=seq[:, :idx],
                    input_pos=torch.arange(0, idx, device=seq.device),
                )
            else:
                logits = decode_one(
                    model,
                    idxs=seq[:, idx - 1 : idx],
                    input_pos=torch.tensor(
                        [idx - 1], device=seq.device, dtype=torch.int
                    ),
                )

        if tokenizer.name == "separated_abs":
            logits[:, tokenizer.tok_to_id[tokenizer.inst_start_tok]] = float(
                "-inf"
            )

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_ids = sample_top_p(probs, top_p).flatten()
        else:
            next_token_ids = torch.argmax(logits, dim=-1).flatten()

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


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# TODO: Clean up a bit and get rid of footguns
def get_inst_prompt(
    tokenizer,
    midi_dict,
    truncate_len: int,
    noise: bool,
):
    from aria.data.datasets import _noise_midi_dict
    from aria.data.midi import MidiDict
    from aria.config import load_config

    midi_dict.metadata["noisy_intervals"] = [[0, truncate_len * 1e3]]

    if noise == True:
        midi_dict = _noise_midi_dict(
            midi_dict, load_config()["data"]["finetuning"]["noising"]
        )

    prompt_seq = tokenizer.tokenize(midi_dict=midi_dict)

    if tokenizer.inst_end_tok in prompt_seq:
        prompt_seq = prompt_seq[: prompt_seq.index(tokenizer.inst_end_tok) + 1]
    else:
        print("No notes found in prompt region")
        prompt_seq = prompt_seq[: prompt_seq.index(tokenizer.bos_tok) + 1]

    return prompt_seq


def get_pt_prompt(
    tokenizer,
    midi_dict,
    truncate_len: int,
):
    prompt_seq = tokenizer.tokenize(midi_dict=midi_dict)
    prompt_seq = tokenizer.truncate_by_time(
        tokenized_seq=prompt_seq,
        trunc_time_ms=truncate_len * 1e3,
    )

    return prompt_seq
