"""Contains generation/sampling code"""

import torch
import torch._dynamo.config
import torch._inductor.config

from typing import List
from tqdm import tqdm

from aria.inference.model_cuda import TransformerLM
from ariautils.tokenizer import Tokenizer, AbsTokenizer
from ariautils.midi import MidiDict

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True


def get_cfg_prompt(prompts: list):
    cfg_prompts = []
    for prompt in prompts:
        cfg_prompts.append(prompt)
        cfg_prompts.append(prompt)

    return cfg_prompts


@torch.inference_mode()
def decode_one(
    model: TransformerLM,
    idxs: torch.Tensor,
    input_pos: torch.Tensor,
    pad_idxs: torch.Tensor | None = None,
) -> torch.Tensor:
    assert input_pos.shape[-1] == 1

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
) -> torch.Tensor:
    logits = model.forward(
        idxs=idxs,
        input_pos=input_pos,
        pad_idxs=pad_idxs,
    )[:, -1]

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
    temp: float = 0.95,
    embedding: list[float] | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    compile: bool = False,
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

    prompt_len = len(prompts[0])
    num_prompts = len(prompts)
    assert all([len(p) == prompt_len for p in prompts])

    model.eval()
    dim_tok_inserted = [False for _ in range(num_prompts)]
    eos_tok_seen = [False for _ in range(num_prompts)]
    total_len = prompt_len + max_new_tokens
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
        batch_size=num_prompts,
        max_seq_len=total_len,
        dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )

    if embedding:
        condition_embedding = torch.tensor(
            [embedding for _ in range(num_prompts)], device=seq.device
        )
        model.fill_condition_kv(cond_emb=condition_embedding)
        emb_offset = 1
    else:
        emb_offset = 0

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
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if idx == prompt_len:
                logits = prefill(
                    model,
                    idxs=seq[:, :idx],
                    input_pos=torch.arange(
                        emb_offset, idx + emb_offset, device=seq.device
                    ),
                )
            else:
                logits = decode_one(
                    model,
                    idxs=seq[:, idx - 1 : idx],
                    input_pos=torch.tensor(
                        [(idx + emb_offset) - 1],
                        device=seq.device,
                        dtype=torch.int,
                    ),
                )

        if temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if min_p is not None:
                next_token_ids = sample_min_p(probs, min_p).flatten()
            else:
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


# Not tested but I think this works
@torch.autocast(
    "cuda",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
@torch.inference_mode()
def sample_batch_cfg(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    prompts: List[list],
    max_new_tokens: int,
    cfg_gamma: float,
    embedding: list[float],
    force_end=False,
    temp: float = 0.95,
    top_p: float | None = None,
    min_p: float | None = None,
    compile: bool = False,
):
    assert 0.0 <= cfg_gamma <= 15.0
    assert top_p is not None or min_p is not None
    if top_p is not None:
        assert 0.5 <= top_p <= 1.0
    if temp is not None:
        assert 0.0 <= temp <= 2.0
    if force_end:
        assert max_new_tokens > 130, "prompt too long to use force_end=True"

    prompts = get_cfg_prompt(prompts)

    prompt_len = len(prompts[0])
    num_prompts = len(prompts)
    assert all([len(p) == prompt_len for p in prompts])

    model.eval()
    total_context_len = prompt_len + max_new_tokens
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(
                    p + [tokenizer.pad_tok] * (total_context_len - len(p))
                )
            )
            for p in prompts
        ]
    ).cuda()
    dim_tok_inserted = [False for _ in range(num_prompts)]
    eos_tok_seen = [False for _ in range(num_prompts)]

    if compile is True:
        global decode_one
        decode_one = torch.compile(
            decode_one,
            mode="reduce-overhead",
            fullgraph=True,
        )

    model.setup_cache(
        batch_size=num_prompts,
        max_seq_len=total_context_len,
        dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )

    condition_embedding = torch.tensor(
        [embedding for _ in range(num_prompts)], device=seq.device
    )
    model.fill_condition_kv(cond_emb=condition_embedding)
    embedding_offset = 1
    pad_idxs = torch.zeros_like(seq, dtype=torch.bool)
    pad_idxs[1::2, 0] = True

    print(
        f"Using hyperparams: temp={temp}, top_p={top_p}, min_p={min_p}, gamma={cfg_gamma}, gen_len={max_new_tokens}"
    )

    CFG_WARM_UP_STEPS = 250
    curr_step = 0
    for idx in (
        pbar := tqdm(
            range(prompt_len, total_context_len),
            total=total_context_len - prompt_len,
            leave=False,
        )
    ):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if idx == prompt_len:
                logits = prefill(
                    model,
                    idxs=seq[:, :idx],
                    input_pos=torch.arange(
                        embedding_offset,
                        idx + embedding_offset,
                        device=seq.device,
                    ),
                    pad_idxs=pad_idxs,
                )
            else:
                logits = decode_one(
                    model,
                    idxs=seq[:, idx - 1 : idx],
                    input_pos=torch.tensor(
                        [(idx + embedding_offset) - 1],
                        device=seq.device,
                        dtype=torch.int,
                    ),
                    pad_idxs=pad_idxs,
                )

        curr_step += 1
        _cfg_gamma = min(cfg_gamma, (curr_step / CFG_WARM_UP_STEPS) * cfg_gamma)

        logits_cfg = _cfg_gamma * logits[::2] + (1 - _cfg_gamma) * logits[1::2]
        logits_cfg[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")

        if temp > 0.0:
            probs = torch.softmax(logits_cfg / temp, dim=-1)
            if min_p is not None:
                next_token_ids = sample_min_p(probs, min_p).flatten()
            else:
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
            max_len=total_context_len,
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


# Working
def sample_min_p(probs, p_base):
    """See - https://arxiv.org/pdf/2407.01082"""
    p_max, _ = torch.max(probs, dim=-1, keepdim=True)
    p_scaled = p_base * p_max
    mask = probs >= p_scaled

    masked_probs = probs.clone()
    masked_probs[~mask] = 0.0
    masked_probs.div_(masked_probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(masked_probs, num_samples=1)

    return next_token


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
    midi_dict: MidiDict, tokenizer: AbsTokenizer, prompt_len_ms: int
):
    midi_dict.note_msgs = [
        msg
        for msg in midi_dict.note_msgs
        if midi_dict.tick_to_ms(msg["data"]["start"]) <= prompt_len_ms
    ]

    if len(midi_dict.note_msgs) == 0:
        return [("prefix", "instrument", "piano"), tokenizer.bos_tok]

    seq = tokenizer.tokenize(midi_dict=midi_dict)
    if tokenizer.dim_tok in seq:
        seq.remove(tokenizer.dim_tok)
    if tokenizer.eos_tok in seq:
        seq.remove(tokenizer.eos_tok)

    return seq
