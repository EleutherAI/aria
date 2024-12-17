"""Contains generation/sampling code"""

import copy
import torch
import torch._dynamo.config
import torch._inductor.config

from typing import List
from tqdm import tqdm

from aria.inference import TransformerLM
from aria.tokenizer import InferenceAbsTokenizer
from ariautils.tokenizer import Tokenizer, AbsTokenizer
from ariautils.midi import MidiDict

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


def get_cfg_prompt(prompts: list, pad_tok: str, guidance_end_tok: str):
    cfg_prompts = []
    for prompt in prompts:
        prompt_no_guidance = prompt[prompt.index(guidance_end_tok) + 1 :]
        prompt_no_guidance = [pad_tok] * (
            len(prompt) - len(prompt_no_guidance)
        ) + prompt_no_guidance
        cfg_prompts.append(prompt)
        cfg_prompts.append(prompt_no_guidance)

    return cfg_prompts


@torch.inference_mode()
def decode_one(
    model: TransformerLM,
    idxs: torch.Tensor,
    input_pos: torch.Tensor,
    pad_idxs: torch.Tensor | None = None,
):
    logits = model.forward(
        idxs=idxs,
        input_pos=input_pos,
        pad_idxs=pad_idxs,
    )[:, -1]

    return logits


@torch.inference_mode()
def prefill(
    model: TransformerLM,
    idxs: torch.Tensor,
    input_pos: torch.Tensor,
    pad_idxs: torch.Tensor | None = None,
):
    logits = model.forward(idxs=idxs, input_pos=input_pos, pad_idxs=pad_idxs)[
        :, -1
    ]

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


# TODO: Not working
@torch.autocast(
    "cuda",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
@torch.inference_mode()
def sample_batch(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: List[list],
    max_new_tokens: int,
    force_end=False,
    temperature: float = 0.95,
    top_p: float = 0.95,
    compile: bool = False,
):
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    _prompt_len = len(prompts[0])
    _num_prompts = len(prompts)
    assert all([len(p) == _prompt_len for p in prompts])

    model.eval()
    dim_tok_inserted = [False for _ in range(_num_prompts)]
    eos_tok_seen = [False for _ in range(_num_prompts)]
    total_len = _prompt_len + max_new_tokens
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(p + [tokenizer.pad_tok] * (total_len - len(p)))
            )
            for p in prompts
        ]
    ).cuda()

    if compile is True:
        global decode_one
        decode_one = torch.compile(
            decode_one,
            mode="reduce-overhead",
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
        f"Using hyperparams: temp={temperature}, top_p={top_p}, gen_len={max_new_tokens}"
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

        if tokenizer.name == "inference_abs":
            logits[:, tokenizer.tok_to_id[tokenizer.prompt_start_tok]] = float(
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


@torch.autocast(
    "cuda",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
@torch.inference_mode()
def sample_batch_cfg(
    model: TransformerLM,
    tokenizer: InferenceAbsTokenizer,
    prompts: List[list],
    max_new_tokens: int,
    cfg_gamma: float,
    force_end=False,
    temperature: float = 0.95,
    top_p: float = 0.95,
    compile: bool = False,
):
    assert 0.0 <= cfg_gamma <= 2.0
    assert 0.0 <= temperature <= 2.0
    assert 0.5 <= top_p <= 1.0
    assert tokenizer.name == "inference_abs"
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    prompts = get_cfg_prompt(
        prompts, tokenizer.pad_tok, tokenizer.guidance_end_tok
    )

    _prompt_len = len(prompts[0])
    _num_prompts = len(prompts)
    assert all([len(p) == _prompt_len for p in prompts])

    model.eval()
    total_len = _prompt_len + max_new_tokens
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(p + [tokenizer.pad_tok] * (total_len - len(p)))
            )
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
                    pad_idxs=(seq == tokenizer.pad_id),
                )
            else:
                logits = decode_one(
                    model,
                    idxs=seq[:, idx - 1 : idx],
                    input_pos=torch.tensor(
                        [idx - 1], device=seq.device, dtype=torch.int
                    ),
                    pad_idxs=(seq == tokenizer.pad_id),
                )

        logits_cfg = cfg_gamma * logits[::2] + (1 - cfg_gamma) * logits[1::2]
        logits_cfg[:, tokenizer.tok_to_id[tokenizer.prompt_start_tok]] = float(
            "-inf"
        )

        if temperature > 0.0:
            probs = torch.softmax(logits_cfg / temperature, dim=-1)
            next_token_ids = sample_top_p(probs, top_p).flatten()
        else:
            next_token_ids = torch.argmax(logits_cfg, dim=-1).flatten()

        next_token_ids = next_token_ids.repeat_interleave(2)
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

    decoded_results = [tokenizer.decode(s) for s in seq.tolist()][::2]
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


def get_inference_prompt(
    tokenizer: InferenceAbsTokenizer,
    midi_dict: MidiDict,
    truncate_len: int,
    guidance_start_ms: int,
    guidance_end_ms: int,
    guidance_midi_dict: MidiDict | None = None,
):
    assert tokenizer.name == "inference_abs"

    if guidance_midi_dict is not None:
        assert guidance_start_ms is not None and guidance_start_ms >= 0
        assert guidance_end_ms is not None and guidance_end_ms >= 0
        assert (
            tokenizer._config["guidance"]["min_ms"]
            <= guidance_end_ms - guidance_start_ms
            <= tokenizer._config["guidance"]["max_ms"]
        )

    prompt_seq = tokenizer.tokenize(
        midi_dict=midi_dict,
        prompt_intervals_ms=(
            [[0, truncate_len * 1e3]] if truncate_len > 0 else []
        ),
        guidance_midi_dict=guidance_midi_dict,
        guidance_start_ms=guidance_start_ms,
        guidance_end_ms=guidance_end_ms,
    )

    if tokenizer.prompt_end_tok in prompt_seq:
        prompt_seq = prompt_seq[
            : prompt_seq.index(tokenizer.prompt_end_tok) + 1
        ]
    else:
        print("No notes found in prompt region")
        prompt_seq = prompt_seq[: prompt_seq.index(tokenizer.bos_tok) + 1]

    if tokenizer.dim_tok in prompt_seq:
        prompt_seq.remove(tokenizer.dim_tok)

    if guidance_midi_dict is not None:
        guidance_seq = copy.deepcopy(prompt_seq)
        guidance_seq = guidance_seq[
            : guidance_seq.index(tokenizer.guidance_end_tok)
        ]
        guidance_seq[0] = ("prefix", "instrument", "piano")
    else:
        guidance_seq = None

    return prompt_seq, guidance_seq
