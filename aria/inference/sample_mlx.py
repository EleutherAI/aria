"""Contains generation/sampling code (mlx)"""

import torch
import numpy as np
import mlx.core as mx

from typing import List
from tqdm import tqdm

from aria.inference.model_mlx import TransformerLM
from ariautils.tokenizer import Tokenizer


def decode_one(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
    pad_idxs: mx.array | None = None,
):
    assert input_pos.shape[-1] == 1

    compiled_forward = mx.compile(model.__call__)
    logits = compiled_forward(
        idxs=idxs,
        input_pos=input_pos,
        offset=input_pos[0],
        # pad_idxs=pad_idxs,
    )[:, -1]
    # logits = model(
    #     idxs=idxs,
    #     input_pos=input_pos,
    #     pad_idxs=pad_idxs,
    # )[:, -1]

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
    )[:, -1]

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
    prompts: List[list],
    max_new_tokens: int,
    force_end=False,
    temp: float = 0.95,
    min_p: float | None = None,
    # compile: bool = False,
):
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
    seq = mx.stack(
        [
            mx.array(
                tokenizer.encode(p + [tokenizer.pad_tok] * (total_len - len(p)))
            )
            for p in prompts
        ]
    )
    model.setup_cache(batch_size=num_prompts, max_seq_len=total_len)
    print(
        f"Using hyperparams: temp={temp}, min_p={min_p}, gen_len={max_new_tokens}"
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
            )
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
            next_token_ids = sample_min_p(probs, min_p).flatten()
        else:
            next_token_ids = mx.argmax(logits, axis=-1).flatten()

        print(tokenizer.id_to_tok[next_token_ids[0].item()])

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


# TODO: Broken
# def sample_min_p(probs: mx.array, p_base: float):  # Added type hint
#     """See - https://arxiv.org/pdf/2407.01082"""
#     p_max = mx.max(probs, axis=-1, keepdims=True)
#     p_scaled = p_base * p_max
#     mask = probs >= p_scaled

#     masked_probs = mx.where(~mask, mx.zeros_like(probs), probs)
#     sum_masked_probs = mx.sum(masked_probs, axis=-1, keepdims=True)
#     sum_masked_probs = mx.where(sum_masked_probs == 0, 1e-9, sum_masked_probs)
#     masked_probs_normalized = masked_probs / sum_masked_probs

#     next_token = mx.random.categorical(masked_probs_normalized, num_samples=1)

#     return next_token


def sample_min_p(probs: mx.array, p_base: float):  # Added type hint
    """See - https://arxiv.org/pdf/2407.01082"""
    p_max = mx.max(probs, axis=-1, keepdims=True)
    p_scaled = p_base * p_max
    mask = probs >= p_scaled

    masked_probs = mx.where(~mask, mx.zeros_like(probs), probs)
    sum_masked_probs = mx.sum(masked_probs, axis=-1, keepdims=True)
    masked_probs_normalized = masked_probs / sum_masked_probs

    # Dumb workaround for mlx not having categorical probs sampler
    next_token = mx.array(
        torch.multinomial(
            torch.from_numpy(np.array(masked_probs_normalized)), num_samples=1
        ),
        dtype=mx.int32,
    )

    return next_token


def sample():
    import os
    import torch

    from aria.model import ModelConfig
    from aria.config import load_model_config

    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.sample import get_inference_prompt

    CHECKPOINT_PATH = "/mnt/ssd1/aria/v2/medium-75-annealed.safetensors"  # Or ".pt" if you're loading a converted PyTorch model
    PROMPT_MIDI_PATH = (
        "/home/loubb/Dropbox/shared/demo.mid"  # Example: "my_melody_prompt.mid"
    )

    NUM_VARIATIONS = 2  # Number of samples (e.g., 2 variations)
    TRUNCATE_LEN_MS = 1000  # Prompt length in milliseconds (e.g., 10 seconds)
    GEN_LENGTH = 256  # Number of new tokens to generate (args.l)
    FORCE_END = False  # Whether to force sequence end (args.e)
    TEMPERATURE = 0.98  # Sampling temperature (args.temp)
    MIN_P = 0.04  # Min-p sampling (args.min_p)

    SAMPLES_DIR = os.path.join(os.getcwd(), "/home/loubb/Dropbox/shared")

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerLM(model_config)
    model.load_weights(CHECKPOINT_PATH)

    midi_dict = MidiDict.from_midi(mid_path=PROMPT_MIDI_PATH)
    prompt_seq = get_inference_prompt(
        tokenizer=tokenizer,
        midi_dict=midi_dict,
        prompt_len_ms=TRUNCATE_LEN_MS,
    )

    print(prompt_seq)
    print(f"Prompt sequence length: {len(prompt_seq)} tokens")
    prompts = [prompt_seq for _ in range(NUM_VARIATIONS)]

    results = sample_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=GEN_LENGTH,
        force_end=FORCE_END,
        temp=TEMPERATURE,
        min_p=MIN_P,
    )

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        output_file_path = os.path.join(SAMPLES_DIR, f"res_{idx + 1}.mid")
        res_midi.save(output_file_path)
        print(f"Saved result {idx + 1} to {output_file_path}")


if __name__ == "__main__":
    sample()
