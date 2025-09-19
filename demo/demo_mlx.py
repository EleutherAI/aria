#!/usr/bin/env python3

import argparse
import os
import time
import uuid
import random
import logging
import threading
import queue
import math
import sys
import pathlib
import select
import json
import mido

import mlx.core as mx
import mlx.nn as nn

from ariautils.midi import MidiDict, midi_to_dict
from ariautils.tokenizer import AbsTokenizer
from aria.inference.model_mlx import TransformerLM
from aria.model import ModelConfig
from aria.config import load_model_config
from aria.run import _get_embedding

EMBEDDING_OFFSET: int = 0
DTYPE = mx.bfloat16
MAX_SEQ_LEN: int = 4096
KV_CHUNK_SIZE: int = 256
PREFILL_CHUNK_SIZE_L: int = 128
PREFILL_CHUNK_SIZE: int = 16
RECALC_DUR_PREFILL_CHUNK_SIZE: int = 8
RECALC_DUR_BUFFER_MS: int = 100

BEAM_WIDTH: int = 3
TIME_TOK_WEIGHTING: int = -5
FIRST_ONSET_BUFFER_MS: int = -200
MAX_STREAM_DELAY_MS: int = 100

MIN_NOTE_DELTA_MS: int = 0
MIN_PEDAL_DELTA_MS: int = 0
MIN_NOTE_LENGTH_MS: int = 10
HARDWARE_INPUT_LATENCY_MS: int = 0
BASE_OUTPUT_LATENCY_MS: int = 0
VELOCITY_OUTPUT_LATENCY_MS: dict[int, int] = {v: 0 for v in range(0, 127, 10)}


config_path = pathlib.Path(__file__).parent.resolve().joinpath("config.json")
file_handler = logging.FileHandler("./demo.log", mode="w")
file_handler.setLevel(logging.DEBUG)


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        class MillisecondFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                created_ms = int(record.created * 1000)
                return str(created_ms)

        if name is not None:
            formatter = MillisecondFormatter(
                "%(asctime)s: [%(levelname)s] [%(name)s] %(message)s"
            )
        else:
            formatter = MillisecondFormatter(
                "%(asctime)s: [%(levelname)s] %(message)s"
            )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("--checkpoint", help="path to model checkpoint")
    argp.add_argument("--midi_in", required=False, help="MIDI input port")
    argp.add_argument("--midi_out", required=True, help="MIDI output port")
    argp.add_argument(
        "--midi_through",
        required=False,
        help="MIDI through port for received input",
    )
    argp.add_argument(
        "--midi_path",
        required=False,
        help="Use MIDI file instead of MIDI input port",
    )
    argp.add_argument(
        "--midi_control_signal",
        type=int,
        help="MIDI control change message for AI takeover",
    )
    argp.add_argument(
        "--midi_reset_control_signal",
        type=int,
        help="MIDI control change message context reset",
    )
    argp.add_argument(
        "--back_and_forth",
        action="store_true",
        help="Enable toggling between human and AI. If not set, the control signal will reset the session.",
        required=False,
    )
    argp.add_argument(
        "--temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "--min_p",
        help="sampling min_p value",
        type=float,
        required=False,
        default=0.03,
    )
    argp.add_argument(
        "--wait_for_close",
        help="wait for note-offs before generating",
        action="store_true",
    )
    argp.add_argument(
        "--quantize",
        help="apply model quantize",
        action="store_true",
    )
    argp.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="path to save complete MIDI file",
    )
    argp.add_argument(
        "--hardware",
        type=str,
        required=False,
        help="path to json file containing hardware calibration settings",
    )
    argp.add_argument(
        "--embedding_checkpoint",
        type=str,
        help="path to embedding model checkpoint for conditioned generation",
        required=False,
    )
    argp.add_argument(
        "--embedding_midi_path",
        type=str,
        help="path to embedding MIDI file for conditioned generation",
        required=False,
    )
    argp.add_argument(
        "--playback",
        action="store_true",
        help="playback file at midi_path through output_port",
        required=False,
    )

    return argp.parse_args()


def set_calibration_settings(load_path: str):
    with open(load_path, "r") as f:
        _settings = json.load(f)

    global MIN_NOTE_DELTA_MS
    global MIN_PEDAL_DELTA_MS
    global MIN_NOTE_LENGTH_MS
    global HARDWARE_INPUT_LATENCY_MS
    global BASE_OUTPUT_LATENCY_MS
    global VELOCITY_OUTPUT_LATENCY_MS

    MIN_NOTE_DELTA_MS = _settings["MIN_NOTE_DELTA_MS"]
    MIN_PEDAL_DELTA_MS = _settings["MIN_PEDAL_DELTA_MS"]
    MIN_NOTE_LENGTH_MS = _settings["MIN_NOTE_LENGTH_MS"]
    HARDWARE_INPUT_LATENCY_MS = _settings["HARDWARE_INPUT_LATENCY_MS"]
    BASE_OUTPUT_LATENCY_MS = _settings["BASE_OUTPUT_LATENCY_MS"]
    VELOCITY_OUTPUT_LATENCY_MS = {
        int(k): v for k, v in _settings["VELOCITY_OUTPUT_LATENCY_MS"].items()
    }


def _get_input_latency_ms(velocity: int):
    return BASE_OUTPUT_LATENCY_MS + VELOCITY_OUTPUT_LATENCY_MS[velocity]


def get_epoch_time_ms() -> int:
    return round(time.time() * 1000)


def prefill(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
) -> mx.array:
    # pad_idxs is only needed for prepended pad tokens
    logits = model(
        idxs=idxs,
        input_pos=input_pos + EMBEDDING_OFFSET,
        max_kv_pos=math.ceil(input_pos[-1].item() / KV_CHUNK_SIZE)
        * KV_CHUNK_SIZE,
        offset=input_pos[0] + EMBEDDING_OFFSET,
    )

    return logits


def decode_one(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
) -> mx.array:
    assert input_pos.shape[-1] == 1

    logits = model(
        idxs=idxs,
        input_pos=input_pos + EMBEDDING_OFFSET,
        max_kv_pos=math.ceil(input_pos[-1].item() / KV_CHUNK_SIZE)
        * KV_CHUNK_SIZE,
        offset=input_pos[0] + EMBEDDING_OFFSET,
    )[:, -1]

    return logits


def sample_min_p(logits: mx.array, p_base: float):
    """Min_p sampler in logit space, see - https://arxiv.org/pdf/2407.01082"""
    if p_base <= 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    if p_base >= 1.0:
        return mx.random.categorical(logits, num_samples=1)

    log_p_max = mx.max(logits, axis=-1, keepdims=True)
    log_p_scaled = mx.log(p_base) + log_p_max
    mask = logits >= log_p_scaled
    masked_logits = mx.where(~mask, -mx.inf, logits)
    next_token = mx.random.categorical(masked_logits, num_samples=1)

    return next_token


def _warmup_prefill(
    model: TransformerLM,
    logger: logging.Logger,
    chunk_size: int,
):
    assert chunk_size > 1

    compile_start_time_s = time.time()
    logger.info(f"Compiling prefill (chunk_size={chunk_size})")
    for idx in range(8):
        start = idx * (MAX_SEQ_LEN - chunk_size) // 7
        mx.eval(
            prefill(
                model,
                idxs=mx.ones([1, chunk_size], dtype=mx.int32),
                input_pos=mx.arange(
                    start,
                    start + chunk_size,
                    dtype=mx.int32,
                ),
            )
        )

    logger.info(
        f"Finished compiling - took {time.time() - compile_start_time_s:.4f} seconds"
    )

    bench_start_time_s = time.time()
    mx.eval(
        prefill(
            model,
            idxs=mx.ones([1, chunk_size], dtype=mx.int32),
            input_pos=mx.arange(0, chunk_size, dtype=mx.int32),
        )
    )
    bench_end_time_s = time.time()
    bench_ms = 1e3 * (bench_end_time_s - bench_start_time_s)
    bench_its = 1000 / bench_ms
    logger.info(
        f"Compiled prefill benchmark: {bench_ms:.2f} ms/it ({bench_its:.2f} it/s)"
    )

    return model


def _warmup_decode_one(
    model: TransformerLM,
    logger: logging.Logger,
):
    # Don't need to explicitly compile with mlx, instead we are just precalculating
    # the computation graphs for different shapes
    compile_start_time_s = time.time()
    for _ in range(5):
        mx.eval(
            decode_one(
                model,
                idxs=mx.array([[random.randint(0, 20)]], dtype=mx.int32),
                input_pos=mx.array([MAX_SEQ_LEN - 1], dtype=mx.int32),
            ),
        )
    logger.info(
        f"Finished compiling - took {time.time() - compile_start_time_s:.4f} seconds"
    )

    bench_start_time_s = time.time()
    mx.eval(
        decode_one(
            model,
            idxs=mx.array([[0]], dtype=mx.int32),
            input_pos=mx.array([0], dtype=mx.int32),
        )
    )
    bench_end_time_s = time.time()
    bench_ms = 1e3 * (bench_end_time_s - bench_start_time_s)
    bench_its = 1000 / bench_ms
    logger.info(
        f"Compiled decode_one benchmark: {bench_ms:.2f} ms/it ({bench_its:.2f} it/s)"
    )

    return model


def warmup_model(model: TransformerLM):
    logger = get_logger()

    model.eval()
    model.setup_cache(
        batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE,
    )

    model = _warmup_decode_one(model=model, logger=logger)
    for chunk_size in list(
        {
            PREFILL_CHUNK_SIZE,
            RECALC_DUR_PREFILL_CHUNK_SIZE,
        }
    ):
        model = _warmup_prefill(
            model=model, logger=logger, chunk_size=chunk_size
        )

    return model


def load_model(checkpoint_path: str):
    logger = get_logger()

    tokenizer = AbsTokenizer(config_path=config_path)
    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)

    weights = mx.load(checkpoint_path)
    for key, weight in weights.items():
        if weight.dtype != DTYPE:
            weights[key] = weight.astype(DTYPE)

    logging.info(f"Loading model weights from {checkpoint_path}")

    init_start_time_s = time.time()
    model = TransformerLM(model_config)

    assert (
        tokenizer.vocab_size == weights["model.tok_embeddings.weight"].shape[0]
    ), "Embedding shape mismatch. Ensure that you are loading the demo-specific checkpoint."

    model.load_weights(list(weights.items()), strict=False)
    model.eval()

    if args.quantize:
        nn.quantize(model.model, group_size=32, bits=8)

    logger.info(
        f"Finished initializing model - took {time.time() - init_start_time_s:.4f} seconds"
    )

    return model


def _first_bad_dur_index(
    tokenizer: AbsTokenizer,
    priming_seq: list,
    pred_ids: list,
    chunk_start: int,
    last_offset_ms: int,
    logger: logging.Logger,
):
    num_time_toks = priming_seq[:chunk_start].count(tokenizer.time_tok)
    local_onset_ms = tokenizer.calc_length_ms(
        priming_seq[: chunk_start + 1], onset=True
    )  # chunk_start + 1 to account for possibly truncated dur token
    logger.debug(f"Starting from local onset {local_onset_ms}")

    for pos, tok_id in enumerate(
        pred_ids[: len(priming_seq) - chunk_start], start=chunk_start
    ):
        prim_tok = priming_seq[pos]  # Should never error?
        pred_tok = tokenizer.id_to_tok[tok_id]
        logger.debug(f"prim={prim_tok}, pred={pred_tok}")

        if isinstance(prim_tok, tuple) and prim_tok[0] == "onset":
            local_onset_ms = num_time_toks * 5000 + prim_tok[1]
        elif prim_tok == tokenizer.time_tok:
            num_time_toks += 1
        elif isinstance(prim_tok, tuple) and prim_tok[0] == "dur":
            dur_true = prim_tok[1]
            dur_pred = pred_tok[1]
            if dur_pred > dur_true and (
                local_onset_ms + dur_true
                >= last_offset_ms - RECALC_DUR_BUFFER_MS
            ):
                logger.info(
                    f"Found token to resample at {pos}: {prim_tok} -> {pred_tok}"
                )
                return pos

    return None


def recalc_dur_tokens_chunked(
    model: TransformerLM,
    priming_seq: list,
    enc_seq: mx.array,
    tokenizer: AbsTokenizer,
    start_idx: int,
):
    # Speculative-decoding inspired duration re-calculation
    assert start_idx > 0
    logger = get_logger("GENERATE")

    priming_len = len(priming_seq)
    last_offset = tokenizer.calc_length_ms(priming_seq, onset=False)
    logger.debug(
        f"Using threshold for duration recalculation: {last_offset - RECALC_DUR_BUFFER_MS}"
    )

    idx = start_idx
    while idx <= priming_len:
        end_idx = idx + RECALC_DUR_PREFILL_CHUNK_SIZE

        window_ids = mx.array(
            enc_seq[:, idx - 1 : end_idx - 1].tolist(),
            dtype=mx.int32,
        )
        window_pos = mx.arange(idx - 1, end_idx - 1, dtype=mx.int32)

        logger.info(
            f"Recalculating chunked durations for positions: {idx-1} - {end_idx-2}"
        )

        logits = prefill(model, idxs=window_ids, input_pos=window_pos)
        pred_ids = mx.argmax(logits, axis=-1).flatten().tolist()

        logger.debug(f"Inserted: {tokenizer.decode(window_ids[0].tolist())}")
        logger.debug(f"Positions: {window_pos.tolist()}")
        logger.debug(f"Predictions: {tokenizer.decode(pred_ids)}")

        bad_pos = _first_bad_dur_index(
            tokenizer=tokenizer,
            priming_seq=priming_seq,
            pred_ids=pred_ids,
            chunk_start=idx,
            last_offset_ms=last_offset,
            logger=logger,
        )

        if bad_pos is None:
            idx = end_idx
        else:
            new_id = pred_ids[bad_pos - idx]
            enc_seq[0, bad_pos] = new_id
            priming_seq[bad_pos] = tokenizer.id_to_tok[new_id]
            idx = bad_pos + 1

    next_logits = logits[:, priming_len - idx]

    logger.debug(f"Internal KV-state: {tokenizer.decode(model.get_kv_ctx())}")

    return enc_seq, priming_seq, next_logits


def decode_first_tokens(
    model: TransformerLM,
    first_token_logits: mx.array,
    enc_seq: mx.array,
    priming_seq: list,
    tokenizer: AbsTokenizer,
    generated_tokens_queue: queue.Queue,
    first_on_msg_epoch_ms: int,
):
    logger = get_logger("GENERATE")

    # buffer_ms determines how far in the past to start generating notes.
    buffer_ms = FIRST_ONSET_BUFFER_MS
    time_tok_id = tokenizer.tok_to_id[tokenizer.time_tok]
    eos_tok_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    dim_tok_id = tokenizer.tok_to_id[tokenizer.dim_tok]
    ped_off_id = tokenizer.tok_to_id[tokenizer.ped_off_tok]

    logits = first_token_logits
    time_since_first_onset_ms = get_epoch_time_ms() - first_on_msg_epoch_ms
    idx = len(priming_seq) + 1

    num_time_toks_required = (time_since_first_onset_ms + buffer_ms) // 5000
    num_time_toks_in_priming_seq = priming_seq.count(tokenizer.time_tok)
    num_time_toks_to_add = num_time_toks_required - num_time_toks_in_priming_seq

    logger.info(f"Time since first onset: {time_since_first_onset_ms}ms")
    logger.info(f"Using first note-onset buffer: {buffer_ms}ms")

    while num_time_toks_to_add > 0:
        generated_tokens_queue.put(tokenizer.time_tok)
        logits = decode_one(
            model,
            idxs=mx.array([[time_tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )

        logger.info(f"Inserted time_tok at position {idx-1}")
        num_time_toks_to_add -= 1
        enc_seq[:, idx - 1] = time_tok_id
        idx += 1

    logits[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")
    logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")
    logits[:, tokenizer.tok_to_id[tokenizer.ped_off_tok]] = float("-inf")

    # MLX doesn't have a equivalent of torch topk
    log_probs = nn.log_softmax(logits, axis=-1)
    top_ids = mx.argsort(log_probs, axis=-1)[0, -BEAM_WIDTH:]
    top_log_probs = log_probs[0, top_ids]

    # top_log_probs are sorted in ascending order
    if time_tok_id not in top_ids.tolist():
        top_ids[0] = time_tok_id
        top_log_probs[0] = log_probs[0, time_tok_id]

    _time_tok_idx = top_ids.tolist().index(time_tok_id)
    top_log_probs[_time_tok_idx] += TIME_TOK_WEIGHTING

    top_toks = [tokenizer.id_to_tok[id] for id in top_ids.tolist()]

    logger.debug(f"Calculated top {BEAM_WIDTH} tokens={top_toks}")
    logger.debug(f"Calculated top {BEAM_WIDTH} scores={top_log_probs.tolist()}")

    priming_seq_last_onset_ms = tokenizer.calc_length_ms(
        priming_seq, onset=True
    )

    if priming_seq_last_onset_ms < time_since_first_onset_ms + buffer_ms:
        masked_onset_ids = [
            tokenizer.tok_to_id[tok]
            for tok in tokenizer.onset_tokens
            if tok[1] < ((time_since_first_onset_ms + buffer_ms) % 5000)
        ]

    else:
        masked_onset_ids = []

    logger.debug(
        f"Masking onsets for {len(masked_onset_ids)} tokens ({time_since_first_onset_ms + buffer_ms})"
    )

    best_score = float("-inf")
    for i in range(BEAM_WIDTH):
        tok = top_toks[i]
        tok_id = top_ids[i].item()
        tok_log_prob = top_log_probs[i]

        next_logits = decode_one(
            model,
            idxs=mx.array([[tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )
        logger.debug(
            f"Sampled logits for positions {idx} by inserting {tok} at position {idx-1}"
        )

        next_log_probs = nn.log_softmax(next_logits, axis=-1)

        next_log_probs[:, eos_tok_id] = float("-inf")
        next_log_probs[:, dim_tok_id] = float("-inf")
        next_log_probs[:, ped_off_id] = float("-inf")

        if masked_onset_ids:
            next_log_probs[:, masked_onset_ids] = float("-inf")
        if tok_id == time_tok_id:
            next_log_probs[:, time_tok_id] = float("-inf")

        next_tok_log_prob = mx.max(next_log_probs, axis=-1)
        next_tok_id = mx.argmax(next_log_probs, axis=-1)
        next_tok = tokenizer.id_to_tok[next_tok_id.item()]
        score = tok_log_prob + next_tok_log_prob

        logger.info(
            f"Calculated tuple {(tok, next_tok)} with scores {(tok_log_prob.item(), next_tok_log_prob.item())} (combined={score.item()})"
        )

        if score > best_score:
            best_tok_id_1, best_tok_id_2 = tok_id, next_tok_id.item()
            best_tok_1, best_tok_2 = (
                tokenizer.id_to_tok[best_tok_id_1],
                tokenizer.id_to_tok[best_tok_id_2],
            )
            best_score = score

    logger.info(
        f"Chose tuple {(best_tok_1, best_tok_2)} with score {best_score.item()}"
    )

    enc_seq[:, idx - 1] = best_tok_id_1
    enc_seq[:, idx] = best_tok_id_2
    generated_tokens_queue.put(tokenizer.id_to_tok[best_tok_id_1])
    generated_tokens_queue.put(tokenizer.id_to_tok[best_tok_id_2])

    mx.eval(
        decode_one(
            model,
            idxs=mx.array([[best_tok_id_1]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )
    )

    logger.info(
        f"Updated KV-Cache by re-inserting {best_tok_1} at position {idx-1}"
    )
    logger.debug(f"Internal KV-state: {tokenizer.decode(model.get_kv_ctx())}")

    return enc_seq, idx + 1


def decode_tokens(
    model: TransformerLM,
    enc_seq: mx.array,
    tokenizer: AbsTokenizer,
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    idx: int,
    temperature: float,
    min_p: float,
    is_ending: bool,
):
    logger = get_logger("GENERATE")
    logger.info(
        f"Using sampling parameters: temperature={temperature}, min_p={min_p}"
    )

    if control_sentinel.is_set():
        control_sentinel.clear()

    last_tok_is_pedal = False
    dur_ids = [tokenizer.tok_to_id[idx] for idx in tokenizer.dur_tokens]
    dur_mask_ids = [
        tokenizer.tok_to_id[("dur", dur_ms)]
        for dur_ms in range(0, MIN_NOTE_LENGTH_MS, 10)
    ]

    while (not control_sentinel.is_set()) and idx < MAX_SEQ_LEN:
        decode_one_start_time_s = time.time()
        prev_tok_id = enc_seq[0, idx - 1]
        prev_tok = tokenizer.id_to_tok[prev_tok_id.item()]

        logits = decode_one(
            model,
            idxs=mx.array([[prev_tok_id]], dtype=mx.int32),
            input_pos=mx.array([idx - 1], dtype=mx.int32),
        )

        logger.debug(
            f"Sampled logits for positions {idx} by inserting {prev_tok} at position {idx-1}"
        )

        logits[:, tokenizer.tok_to_id[tokenizer.ped_off_tok]] += 3  # Manual adj
        logits[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")

        logits[:, dur_mask_ids] = float("-inf")
        if last_tok_is_pedal is True:
            logits[:, dur_ids] = float("-inf")

        if is_ending is False:
            logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")

        if temperature > 0.0:
            next_token_ids = sample_min_p(logits, min_p).flatten()
        else:
            next_token_ids = mx.argmax(logits, axis=-1).flatten()

        enc_seq[:, idx] = next_token_ids
        next_token = tokenizer.id_to_tok[next_token_ids[0].item()]
        logger.debug(
            f"({(time.time() - decode_one_start_time_s)*1000:.2f}ms) {idx}: {next_token}"
        )

        if next_token in {tokenizer.ped_on_tok, tokenizer.ped_off_tok}:
            last_tok_is_pedal = True
        elif isinstance(next_token, tuple) and next_token[0] == "piano":
            last_tok_is_pedal = False

        if next_token == tokenizer.eos_tok:
            logger.info("EOS token produced")
            generated_tokens_queue.put(next_token)
            return
        else:
            generated_tokens_queue.put(next_token)
            idx += 1

    logger.info(f"Finished generating: {idx}")
    generated_tokens_queue.put(None)


def generate_tokens(
    priming_seq: list,
    tokenizer: AbsTokenizer,
    model: TransformerLM,
    prev_context: list[int],
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    num_preceding_active_pitches: int,
    first_on_msg_epoch_ms: int,
    temperature: float = 0.98,
    min_p: float = 0.03,
    is_ending: bool = False,
):
    logger = get_logger("GENERATE")

    generate_start_s = time.time()
    priming_seq_len = len(priming_seq)

    start_idx = max(
        2, priming_seq_len - 3 * (num_preceding_active_pitches + 2) - 1
    )
    enc_seq = mx.array(
        [
            tokenizer.encode(
                priming_seq
                + [tokenizer.pad_tok] * (MAX_SEQ_LEN - len(priming_seq))
            )
        ],
        dtype=mx.int32,
    )

    logger.debug(f"Priming sequence {priming_seq}")
    logger.info(f"Priming sequence length: {priming_seq_len}")
    logger.info(f"Prefilling up to (and including) position: {start_idx-1}")

    prefill_start_s = time.time()
    chunked_prefill(
        model=model,
        tokenizer=tokenizer,
        prev_context=prev_context,
        curr_context=enc_seq[0, :start_idx].tolist(),
        full=True,
    )

    logger.info(
        f"Prefill took {(time.time() - prefill_start_s) * 1000:.2f} milliseconds"
    )
    logger.info(f"Starting duration recalculation from position: {start_idx-1}")

    recalculate_dur_start_s = time.time()
    enc_seq, priming_seq, next_token_logits = recalc_dur_tokens_chunked(
        model=model,
        priming_seq=priming_seq,
        enc_seq=enc_seq,
        tokenizer=tokenizer,
        start_idx=start_idx,
    )

    logger.info(
        f"Recalculating durations took {(time.time() - recalculate_dur_start_s) * 1000:.2f} milliseconds"
    )

    decode_first_s = time.time()
    enc_seq, idx = decode_first_tokens(
        model=model,
        first_token_logits=next_token_logits,
        enc_seq=enc_seq,
        priming_seq=priming_seq,
        tokenizer=tokenizer,
        generated_tokens_queue=generated_tokens_queue,
        first_on_msg_epoch_ms=first_on_msg_epoch_ms,
    )

    logger.info(
        f"Decode first two tokens took {(time.time() - decode_first_s) * 1000:.2f} milliseconds"
    )
    logger.info(
        f"Time to first token took {(time.time() - generate_start_s) * 1000:.2f} milliseconds"
    )

    decode_tokens(
        model=model,
        enc_seq=enc_seq,
        tokenizer=tokenizer,
        control_sentinel=control_sentinel,
        generated_tokens_queue=generated_tokens_queue,
        idx=idx,
        temperature=temperature,
        min_p=min_p,
        is_ending=is_ending,
    )


def _adjust_previous_off_time(
    pitch_to_prev_msg: dict,
    key: str | int,
    new_on_send_time: int,
    min_delta_ms: int,
    logger: logging.Logger,
):
    prev_on, prev_off = pitch_to_prev_msg.get(key, (None, None))

    if prev_on is not None and prev_off is not None and min_delta_ms > 0:
        adj_send_off_time = max(
            min(
                prev_off["send_epoch_time_ms"],
                new_on_send_time - min_delta_ms,
            ),
            prev_on[
                "send_epoch_time_ms"
            ],  #  Don't move prev_off before prev_on
        )
        if adj_send_off_time != prev_off["send_epoch_time_ms"]:
            logger.debug(f"Adjusting {prev_off}: t={adj_send_off_time}")
            prev_off["send_epoch_time_ms"] = adj_send_off_time
            prev_off["adjusted"] = True


# TODO: Verify that only ON -> OFF sequences are possible in tokenizer
def _decode_pedal_double(
    note_buffer: list,
    first_on_msg_epoch_ms: int,
    num_time_toks: int,
    pitch_to_prev_msg: dict,
    outbound_midi_msg_queue: queue.Queue,
    logger: logging.Logger,
    tokenizer: AbsTokenizer,
):
    pedal_tok, onset_tok = note_buffer
    velocity = 127 if pedal_tok == tokenizer.ped_on_tok else 0
    _, onset = onset_tok

    onset_epoch_ms = first_on_msg_epoch_ms + (num_time_toks * 5000) + onset
    send_onset_epoch_ms = onset_epoch_ms - BASE_OUTPUT_LATENCY_MS
    pedal_msg = {
        "pitch": "pedal",
        "vel": velocity,
        "epoch_time_ms": onset_epoch_ms,
        "send_epoch_time_ms": send_onset_epoch_ms,
        "uuid": "pedal",  # All pedals have the same id
    }

    if pedal_tok == tokenizer.ped_on_tok:
        _adjust_previous_off_time(
            pitch_to_prev_msg=pitch_to_prev_msg,
            key="pedal",
            new_on_send_time=send_onset_epoch_ms,
            min_delta_ms=MIN_PEDAL_DELTA_MS,
            logger=logger,
        )
        pitch_to_prev_msg["pedal"] = (pedal_msg, None)

    elif pedal_tok == tokenizer.ped_off_tok:
        prev_on, _ = pitch_to_prev_msg.get("pedal", (None, None))
        pitch_to_prev_msg["pedal"] = (prev_on, pedal_msg)

    outbound_midi_msg_queue.put(pedal_msg)
    logger.debug(f"Put message: {pedal_msg}")
    logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

    return onset_epoch_ms


def _decode_note_triple(
    note_buffer: list,
    first_on_msg_epoch_ms: int,
    num_time_toks: int,
    pitch_to_prev_msg: dict,
    outbound_midi_msg_queue: queue.Queue,
    logger: logging.Logger,
):
    note_tok, onset_tok, dur_tok = note_buffer
    _, pitch, vel = note_tok
    _, onset = onset_tok
    _, dur = dur_tok

    _uuid = uuid.uuid4()
    onset_epoch_ms = first_on_msg_epoch_ms + (num_time_toks * 5000) + onset
    offset_epoch_ms = onset_epoch_ms + dur
    send_onset_epoch_ms = onset_epoch_ms - _get_input_latency_ms(vel)
    send_offset_epoch_ms = offset_epoch_ms - BASE_OUTPUT_LATENCY_MS

    on_msg = {
        "pitch": pitch,
        "vel": vel,
        "epoch_time_ms": onset_epoch_ms,
        "send_epoch_time_ms": send_onset_epoch_ms,
        "uuid": _uuid,
    }
    off_msg = {
        "pitch": pitch,
        "vel": 0,
        "epoch_time_ms": offset_epoch_ms,
        "send_epoch_time_ms": send_offset_epoch_ms,
        "uuid": _uuid,
    }

    _adjust_previous_off_time(
        pitch_to_prev_msg=pitch_to_prev_msg,
        key=pitch,
        new_on_send_time=send_onset_epoch_ms,
        min_delta_ms=MIN_NOTE_DELTA_MS,
        logger=logger,
    )

    pitch_to_prev_msg[pitch] = (on_msg, off_msg)

    outbound_midi_msg_queue.put(on_msg)
    outbound_midi_msg_queue.put(off_msg)
    logger.debug(f"Put message: {on_msg}")
    logger.debug(f"Put message: {off_msg}")
    logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

    return offset_epoch_ms


# TODO: Refactor this method to prettify it
def decode_tokens_to_midi(
    generated_tokens_queue: queue.Queue,
    outbound_midi_msg_queue: queue.Queue,
    tokenizer: AbsTokenizer,
    first_on_msg_epoch_ms: int,
    priming_seq_last_onset_ms: int,
):
    logger = get_logger("DECODE")

    assert (
        first_on_msg_epoch_ms + priming_seq_last_onset_ms
        < get_epoch_time_ms() + HARDWARE_INPUT_LATENCY_MS
    )

    logger.info(f"Priming sequence last onset: {priming_seq_last_onset_ms}")
    logger.info(
        f"Total time elapsed since first onset: {get_epoch_time_ms() - first_on_msg_epoch_ms}"
    )

    pitch_to_prev_msg = {}
    note_buffer = []
    num_time_toks = priming_seq_last_onset_ms // 5000

    while True:
        while True:
            tok = generated_tokens_queue.get()
            if tok is tokenizer.eos_tok:
                # pitch=-1 is interpreted as the end message by stream_midi
                _uuid = uuid.uuid4()
                end_msg = {
                    "pitch": -1,
                    "vel": -1,
                    "epoch_time_ms": offset_epoch_ms + 100,
                    "send_epoch_time_ms": offset_epoch_ms + 100,
                    "uuid": _uuid,
                }
                outbound_midi_msg_queue.put(end_msg)
                logger.info(f"Seen exit signal: EOS token")
                logger.debug(f"Put message: {end_msg}")
                return

            elif tok is None:
                logger.info(f"Seen exit signal: Sentinel")
                return

            logger.debug(f"Seen token: {tok}")
            note_buffer.append(tok)

            if isinstance(tok, tuple) and tok[0] == "dur":
                msg_type = "note"
                break
            elif (
                isinstance(tok, tuple)
                and tok[0] == "onset"
                and note_buffer[-2]
                in {tokenizer.ped_on_tok, tokenizer.ped_off_tok}
            ):
                msg_type = "pedal"
                break

        while note_buffer and note_buffer[0] == tokenizer.time_tok:
            logger.debug("Popping time_tok")
            num_time_toks += 1
            note_buffer.pop(0)

        assert len(note_buffer) in {2, 3}, f"Generation error: buffer={note_buffer}"  # fmt: skip

        logger.debug(f"Decoded note: {note_buffer}")

        if msg_type == "note":
            offset_epoch_ms = _decode_note_triple(
                note_buffer=note_buffer,
                first_on_msg_epoch_ms=first_on_msg_epoch_ms,
                num_time_toks=num_time_toks,
                pitch_to_prev_msg=pitch_to_prev_msg,
                outbound_midi_msg_queue=outbound_midi_msg_queue,
                logger=logger,
            )
        elif msg_type == "pedal":
            offset_epoch_ms = _decode_pedal_double(
                note_buffer=note_buffer,
                first_on_msg_epoch_ms=first_on_msg_epoch_ms,
                num_time_toks=num_time_toks,
                pitch_to_prev_msg=pitch_to_prev_msg,
                outbound_midi_msg_queue=outbound_midi_msg_queue,
                logger=logger,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError

        note_buffer = []


def _create_mido_message(
    msg_dict: dict,
    channel: int,
    time_delta_ms: int,
) -> mido.Message:
    if msg_dict["pitch"] == "pedal":
        return mido.Message(
            "control_change",
            control=64,
            value=msg_dict["vel"],
            channel=channel,
            time=time_delta_ms,
        )
    else:
        # note-on or note-off
        return mido.Message(
            "note_on",
            note=msg_dict["pitch"],
            velocity=msg_dict["vel"],
            channel=channel,
            time=time_delta_ms,
        )


def stream_midi(
    inbound_midi_msg_queue: queue.Queue,
    msgs: list[mido.Message],
    last_channel_msg_epoch_time_ms: float,
    midi_output_port: str,
    control_sentinel: threading.Event,
    midi_stream_channel: int,
    results_queue: queue.Queue,
):
    logger = get_logger("STREAM")
    logger.info(f"Sending generated messages on port: '{midi_output_port}'")

    active_pitch_uuid = {}
    pending_msgs = []
    msgs_to_archive = []

    with mido.open_output(midi_output_port) as midi_out:
        while not control_sentinel.is_set():
            while not inbound_midi_msg_queue.empty():
                try:
                    msg = inbound_midi_msg_queue.get_nowait()
                    if msg:
                        pending_msgs.append(msg)
                except queue.Empty:
                    break

            pending_msgs.sort(key=lambda m: (m["send_epoch_time_ms"], m["vel"]))

            while pending_msgs:
                curr_epoch_time_ms = get_epoch_time_ms()
                msg = pending_msgs[0]

                if msg["send_epoch_time_ms"] > curr_epoch_time_ms:
                    break
                elif (
                    curr_epoch_time_ms - msg["send_epoch_time_ms"]
                    > MAX_STREAM_DELAY_MS
                ):
                    logger.debug(f"Skipping stale message: {msg}")
                    pending_msgs.pop(0)
                    continue

                logger.debug(f"Processing: {msg}")

                # End signal
                if msg["pitch"] == -1:
                    control_sentinel.set()
                    break

                should_send = False
                should_archive = False
                if msg["vel"] > 0:  # note-on or pedal-on
                    active_pitch_uuid[msg["pitch"]] = msg["uuid"]
                    should_send = True
                    should_archive = True
                else:  # note-off or pedal-off (vel == 0)
                    if msg.get("adjusted", False):
                        should_send = True
                        should_archive = msg["pitch"] == "pedal"
                    elif active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]:
                        should_send = True
                        should_archive = True
                        active_pitch_uuid.pop(msg["pitch"], None)

                if should_send:
                    mido_msg = _create_mido_message(
                        msg_dict=msg, channel=0, time_delta_ms=0
                    )
                    midi_out.send(mido_msg)
                    logger.info(f"Sent message: {mido_msg}")

                if should_archive:
                    msgs_to_archive.append(msg)

                pending_msgs.pop(0)

            if control_sentinel.is_set():
                break

            time.sleep(0.005)

        last_archive_time_ms = last_channel_msg_epoch_time_ms
        msgs_to_archive.sort(key=lambda m: (m["epoch_time_ms"], m["vel"]))

        for msg in msgs_to_archive:
            time_delta_ms = round(msg["epoch_time_ms"] - last_archive_time_ms)
            mido_msg = _create_mido_message(
                msg_dict=msg,
                channel=midi_stream_channel,
                time_delta_ms=time_delta_ms,
            )
            msgs.append(mido_msg)
            last_archive_time_ms = msg["epoch_time_ms"]

        logger.info("Sending final note-off messages for cleanup.")
        remaining_off_msgs = [
            msg
            for msg in pending_msgs
            if msg["vel"] == 0
            and msg["pitch"] != "pedal"
            and active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
        ]
        remaining_off_msgs.sort(key=lambda m: (m["epoch_time_ms"]))

        for msg in remaining_off_msgs:
            mido_msg = _create_mido_message(
                msg_dict=msg, channel=0, time_delta_ms=0
            )
            midi_out.send(mido_msg)

            time_delta_ms = round(msg["epoch_time_ms"] - last_archive_time_ms)
            archived_msg = _create_mido_message(
                msg_dict=msg,
                channel=midi_stream_channel,
                time_delta_ms=time_delta_ms,
            )
            msgs.append(archived_msg)
            last_archive_time_ms = msg["epoch_time_ms"]

        midi_out.send(
            mido.Message(
                "control_change", control=64, value=0, channel=0, time=0
            )
        )

    results_queue.put(msgs)


def stream_msgs(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    msgs: list[mido.Message],
    prev_context: list[int],
    midi_output_port: str,
    first_on_msg_epoch_ms: int,
    control_sentinel: threading.Event,
    temperature: float,
    min_p: float,
    num_preceding_active_pitches: int,
    midi_stream_channel: int,
    is_ending: bool = False,
):

    logger = get_logger("STREAM")
    midi = convert_msgs_to_midi(msgs=msgs)
    midi_dict = MidiDict(**midi_to_dict(midi))
    midi_dict.remove_redundant_pedals()
    priming_seq = tokenizer.tokenize(midi_dict=midi_dict, add_dim_tok=False)
    priming_seq = priming_seq[: priming_seq.index(tokenizer.eos_tok)]

    if priming_seq[-2] == tokenizer.ped_off_tok:
        # Final pedal-off is needed for tokenizer, but unneeded in tokenized sequence
        logger.info("Removing final pedal_off from tokenized sequence")
        priming_seq = priming_seq[:-2]

    if is_ending is True:
        priming_seq.append(tokenizer.dim_tok)

    generated_tokens_queue = queue.Queue()
    midi_messages_queue = queue.Queue()

    generate_tokens_thread = threading.Thread(
        target=generate_tokens,
        kwargs={
            "priming_seq": priming_seq,
            "tokenizer": tokenizer,
            "model": model,
            "prev_context": prev_context,
            "control_sentinel": control_sentinel,
            "generated_tokens_queue": generated_tokens_queue,
            "temperature": temperature,
            "min_p": min_p,
            "num_preceding_active_pitches": num_preceding_active_pitches,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
            "is_ending": is_ending,
        },
    )
    generate_tokens_thread.start()

    decode_tokens_to_midi_thread = threading.Thread(
        target=decode_tokens_to_midi,
        kwargs={
            "generated_tokens_queue": generated_tokens_queue,
            "outbound_midi_msg_queue": midi_messages_queue,
            "tokenizer": tokenizer,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
            "priming_seq_last_onset_ms": tokenizer.calc_length_ms(
                priming_seq, onset=True
            ),
        },
    )
    decode_tokens_to_midi_thread.start()

    # If ending==True then previous MIDI message on midi_stream_channel occurs
    # at first_on_msg_epoch_ms.
    prev_channel_msg_epoch_time_ms = (
        first_on_msg_epoch_ms
        + tokenizer.calc_length_ms(priming_seq, onset=False)
        if is_ending is False
        else first_on_msg_epoch_ms
    )

    stream_midi_results_queue = queue.Queue()
    stream_midi_thread = threading.Thread(
        target=stream_midi,
        kwargs={
            "inbound_midi_msg_queue": midi_messages_queue,
            "msgs": msgs,
            "last_channel_msg_epoch_time_ms": prev_channel_msg_epoch_time_ms,
            "midi_output_port": midi_output_port,
            "control_sentinel": control_sentinel,
            "midi_stream_channel": midi_stream_channel,
            "results_queue": stream_midi_results_queue,
        },
    )
    stream_midi_thread.start()

    generate_tokens_thread.join()
    decode_tokens_to_midi_thread.join()
    stream_midi_thread.join()
    msgs = stream_midi_results_queue.get()

    return msgs


def convert_msgs_to_midi(msgs: list[mido.Message]):
    channel_to_track = {
        chan: mido.MidiTrack()
        for chan in list(set([msg.channel for msg in msgs]))
    }

    for msg in msgs:
        channel_to_track[msg.channel].append(msg)

    # Workaround for possibility that track_0 start time != first_on_msg_epoch_ms
    for msg in channel_to_track[0]:
        if msg.type == "note_on" and msg.velocity > 0:
            msg.time = 0
            break
        else:
            msg.time = 0

    mid = mido.MidiFile(type=1)
    mid.ticks_per_beat = 500

    for channel, track in channel_to_track.items():
        track.insert(0, mido.MetaMessage("set_tempo", tempo=500000, time=0))
        track.insert(
            0,
            mido.Message("program_change", program=0, channel=channel, time=0),
        )
        mid.tracks.append(track)

    return mid


def _find_divergence(
    prev_context: list,
    curr_context: list,
    logger: logging.Logger,
    tokenizer: AbsTokenizer,
):
    agreement_index = 0
    for prev_val, curr_val in zip(prev_context, curr_context):
        if prev_val == curr_val:
            agreement_index += 1
        else:
            logger.info(
                f"Found divergence at idx {agreement_index}: {tokenizer.id_to_tok[curr_val]}, {tokenizer.id_to_tok[prev_val]}"
            )
            break

    return agreement_index, curr_context[agreement_index:]


def chunked_prefill(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    prev_context: list,
    curr_context: list,
    full: bool = False,
):

    assert isinstance(curr_context[0], int)
    assert tokenizer.pad_id not in prev_context
    assert tokenizer.pad_id not in curr_context

    logger = get_logger("PREFILL")

    while True:
        prefill_idx, prefill_toks = _find_divergence(
            prev_context,
            curr_context,
            logger=logger,
            tokenizer=tokenizer,
        )
        num_prefill_toks = len(prefill_toks)
        logger.debug(f"Tokens to prefill: {len(prefill_toks)}")

        if num_prefill_toks > PREFILL_CHUNK_SIZE_L:
            logger.debug(
                f"Prefilling {PREFILL_CHUNK_SIZE_L} tokens from idx={prefill_idx}"
            )
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array(
                        [prefill_toks[:PREFILL_CHUNK_SIZE_L]],
                        dtype=mx.int32,
                    ),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE_L,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context[: prefill_idx + PREFILL_CHUNK_SIZE_L]

        elif num_prefill_toks > PREFILL_CHUNK_SIZE:
            logger.debug(
                f"Prefilling {PREFILL_CHUNK_SIZE} tokens from idx={prefill_idx}"
            )
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array(
                        [prefill_toks[:PREFILL_CHUNK_SIZE]],
                        dtype=mx.int32,
                    ),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context[: prefill_idx + PREFILL_CHUNK_SIZE]

        elif num_prefill_toks > 0 and full is True:
            logger.debug(
                f"Prefilling (force) {num_prefill_toks} tokens from idx={prefill_idx}"
            )
            prefill_toks += (PREFILL_CHUNK_SIZE - len(prefill_toks)) * [
                tokenizer.pad_id
            ]
            mx.eval(
                prefill(
                    model,
                    idxs=mx.array([prefill_toks], dtype=mx.int32),
                    input_pos=mx.arange(
                        prefill_idx,
                        prefill_idx + PREFILL_CHUNK_SIZE,
                        dtype=mx.int32,
                    ),
                )
            )
            prev_context = curr_context
            break
        else:
            break

    logger.info(
        f"KV stored up to idx={max(0, len(prev_context)- 1)} (curr_context_len={len(curr_context)})"
    )

    return prev_context


def continuous_prefill(
    model: TransformerLM,
    msgs: list,
    received_messages_queue: queue.Queue,
    prev_context: list[int],
):
    tokenizer = AbsTokenizer(config_path=config_path)
    logger = get_logger("PREFILL")
    msg_cnt = 0
    seen_sentinel = False

    while seen_sentinel is False:
        while seen_sentinel is False:
            try:
                msg = received_messages_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if msg is None:
                    logger.info("Seen sentinel in message received messages")
                    seen_sentinel = True
                else:
                    msgs.append(msg)
                    msg_cnt += 1

        if msg_cnt >= 10:
            midi = convert_msgs_to_midi(msgs=msgs)
            midi_dict = MidiDict(**midi_to_dict(midi))
            midi_dict.remove_redundant_pedals()

            if len(midi_dict.note_msgs) > 0:
                curr_context = tokenizer.encode(
                    tokenizer.tokenize(midi_dict, add_dim_tok=False)
                )
                prev_context = chunked_prefill(
                    model=model,
                    tokenizer=tokenizer,
                    prev_context=prev_context,
                    curr_context=curr_context,
                    full=False,
                )

            msg_cnt = 0
        else:
            time.sleep(0.01)

    return msgs, prev_context


def capture_and_update_kv(
    model: TransformerLM,
    msgs: list,
    prev_context: list,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    wait_for_close: bool,
    midi_performance_queue: queue.Queue,
    midi_capture_channel: int,
    first_msg_epoch_time_ms: int | None = None,
):
    received_messages_queue = queue.Queue()
    results_queue = queue.Queue()
    capture_midi_thread = threading.Thread(
        target=capture_midi_input,
        kwargs={
            "midi_performance_queue": midi_performance_queue,
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "received_messages_queue": received_messages_queue,
            "midi_capture_channel": midi_capture_channel,
            "first_msg_epoch_time_ms": first_msg_epoch_time_ms,
            "results_queue": results_queue,
            "wait_for_close": wait_for_close,
        },
    )
    capture_midi_thread.start()

    msgs, prev_context = continuous_prefill(
        model=model,
        msgs=msgs,
        received_messages_queue=received_messages_queue,
        prev_context=prev_context,
    )
    capture_midi_thread.join()
    first_on_msg_epoch_ms, num_active_pitches = results_queue.get()

    return msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches


def capture_midi_input(
    midi_performance_queue: queue.Queue,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    received_messages_queue: queue.Queue,
    midi_capture_channel: int,
    results_queue: queue.Queue,
    first_msg_epoch_time_ms: int | None = None,
    wait_for_close: bool = False,
):
    logger = get_logger("CAPTURE")
    first_on_msg_epoch_ms = None
    prev_msg_epoch_time_ms = first_msg_epoch_time_ms
    pedal_down = False
    pitches_held_down = set()
    pitches_sustained_by_pedal = set()

    while not midi_performance_queue.empty():
        try:
            midi_performance_queue.get_nowait()
        except queue.Empty:
            break

    logger.info("Listening for input")
    logger.info("Commencing generation upon keypress or control signal")

    while True:
        epoch_time_ms = get_epoch_time_ms()
        active_notes = pitches_held_down.union(pitches_sustained_by_pedal)
        should_stop = not wait_for_close or not active_notes
        if reset_sentinel.is_set() or (
            control_sentinel.is_set() and should_stop
        ):
            break

        try:
            msg = midi_performance_queue.get(block=True, timeout=0.01)
        except queue.Empty:
            continue

        if msg.is_meta or msg.type == "program_change":
            continue

        msg.channel = midi_capture_channel
        if prev_msg_epoch_time_ms is None:
            msg.time = 0
        else:
            msg.time = epoch_time_ms - prev_msg_epoch_time_ms

        prev_msg_epoch_time_ms = epoch_time_ms
        logger.info(f"Received message: [{msg}]")

        match msg.type:
            case "note_on" if msg.velocity > 0:
                if first_on_msg_epoch_ms is None:
                    first_on_msg_epoch_ms = (
                        get_epoch_time_ms() - HARDWARE_INPUT_LATENCY_MS
                    )
                pitches_held_down.add(msg.note)
                if pedal_down:
                    pitches_sustained_by_pedal.add(msg.note)
                received_messages_queue.put(msg)

            case "note_off" | "note_on":
                # Note-off
                pitches_held_down.discard(msg.note)
                received_messages_queue.put(msg)

            case "control_change" if msg.control == 64:
                if msg.value >= 64:
                    pedal_down = True
                    pitches_sustained_by_pedal.update(pitches_held_down)
                else:
                    pedal_down = False
                    pitches_sustained_by_pedal.clear()
                received_messages_queue.put(msg)

    active_pitches = pitches_held_down.union(pitches_sustained_by_pedal)
    num_active_pitches = len(active_pitches)
    logger.info(f"Active pitches ({num_active_pitches}): {active_pitches}")

    time_offset = get_epoch_time_ms() - prev_msg_epoch_time_ms
    for pitch in pitches_held_down:
        note_off_msg = mido.Message(
            "note_off",
            note=pitch,
            channel=midi_capture_channel,
            time=time_offset,
        )
        received_messages_queue.put(note_off_msg)
        time_offset = 0

    received_messages_queue.put(
        mido.Message(
            "control_change",
            control=64,
            value=0,
            channel=midi_capture_channel,
            time=0,
        )
    )

    received_messages_queue.put(None)
    results_queue.put((first_on_msg_epoch_ms, num_active_pitches))


def play_midi_file(
    midi_through_port: str,
    midi_performance_queue: queue.Queue,
    midi_path: str,
    currently_generating_sentinel: threading.Event,
    reset_sentinel: threading.Event,
):
    def _send_delayed_message(_midi_performance_queue: queue.Queue, msg):
        _midi_performance_queue.put(msg)
        logger.debug(f"SENT: {msg}")

    logger = get_logger("FILE")
    logger.info(f"Playing {midi_path} on through-port '{midi_through_port}'")
    logger.info(f"Simulating input with {HARDWARE_INPUT_LATENCY_MS}ms latency")

    if BASE_OUTPUT_LATENCY_MS > 0:
        midi_dict = MidiDict.from_midi(midi_path)
        midi_dict.remove_redundant_pedals()
        midi_dict.enforce_gaps(min_gap_ms=MIN_NOTE_DELTA_MS)
        mid = midi_dict.to_midi()
    else:
        mid = mido.MidiFile(midi_path)

    time.sleep(1)
    with mido.open_output(midi_through_port) as through_port:
        for msg in mid.play():
            if reset_sentinel.is_set():
                logger.debug("Exiting")
                return

            if currently_generating_sentinel.is_set() is False:
                through_port.send(msg)

            timer = threading.Timer(
                interval=HARDWARE_INPUT_LATENCY_MS / 1000.0,
                function=_send_delayed_message,
                args=[midi_performance_queue, msg],
            )
            timer.start()


def listen_for_keypress_control_signal(
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    back_and_forth: bool = False,
):
    logger = get_logger("KEYBOARD")
    logger.info(
        "Listening for keyboard input (Enter to start AI, any other key + Enter to reset)."
    )

    while not reset_sentinel.is_set():
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)

        if rlist:
            _input = sys.stdin.readline().strip()
            logger.info(f'Keypress seen "{_input}"')

            if _input == "":
                if (
                    currently_generating_sentinel.is_set()
                    and back_and_forth is False
                ):
                    logger.info("Resetting (control)")
                    reset_sentinel.set()
                control_sentinel.set()
            else:
                logger.info("Resetting (reset)")
                reset_sentinel.set()
                control_sentinel.set()

    logger.debug(
        "Exiting keypress listener because reset_sentinel was set by another thread."
    )


def _listen(
    midi_control_queue: queue.Queue,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    logger: logging.Logger,
    midi_control_signal: int | None = None,
    midi_reset_control_signal: int | None = None,
):
    while not midi_control_queue.empty():
        try:
            midi_control_queue.get_nowait()
        except queue.Empty:
            break

    logger.info(
        f"Listening for takeover signal ({midi_control_signal}) and reset signal ({midi_reset_control_signal}) on control queue."
    )
    seen_note_on = False
    while not reset_sentinel.is_set():
        try:
            msg = midi_control_queue.get(block=True, timeout=0.01)
        except queue.Empty:
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            seen_note_on = True

        should_return_signal = (
            seen_note_on or currently_generating_sentinel.is_set()
        )
        if (
            msg.type == "control_change"
            and msg.control == midi_control_signal
            and msg.value >= 64
            and should_return_signal
        ):
            return midi_control_signal
        elif (
            msg.type == "control_change"
            and msg.control == midi_reset_control_signal
            and msg.value >= 64
            and should_return_signal
        ):
            return midi_reset_control_signal


def listen_for_midi_control_signal(
    midi_control_queue: queue.Queue,
    control_sentinel: threading.Event,
    reset_sentinel: threading.Event,
    currently_generating_sentinel: threading.Event,
    midi_control_signal: int | None = None,
    midi_reset_control_signal: int | None = None,
    back_and_forth: bool = False,
):
    logger = get_logger("MIDI-CONTROL")

    while not reset_sentinel.is_set():
        time.sleep(1)
        signal_received = _listen(
            midi_control_queue=midi_control_queue,
            reset_sentinel=reset_sentinel,
            currently_generating_sentinel=currently_generating_sentinel,
            midi_control_signal=midi_control_signal,
            midi_reset_control_signal=midi_reset_control_signal,
            logger=logger,
        )

        if signal_received is not None:
            logger.info(f"Seen MIDI control signal ({signal_received})")

            if signal_received == midi_reset_control_signal:
                logger.info("Resetting (reset)")
                reset_sentinel.set()
                control_sentinel.set()
            elif signal_received == midi_control_signal:
                if (
                    currently_generating_sentinel.is_set()
                    and back_and_forth is False
                ):
                    logger.info("Resetting (control)")
                    reset_sentinel.set()
                control_sentinel.set()

    logger.debug("Exiting MIDI control listener")


# TODO: Debug, fix, and perhaps refactor the functionality for going back and forth
# - One idea is on resume, to wait to start the clock until the user plays.
def run(
    model: TransformerLM,
    midi_performance_queue: queue.Queue,
    midi_control_queue: queue.Queue,
    midi_through_port: str | None,
    midi_out_port: str | None,
    midi_path: str | None,
    midi_save_path: str | None,
    midi_control_signal: int,
    midi_reset_control_signal: int,
    reset_sentinel: threading.Event,
    wait_for_close: bool,
    temperature: float,
    min_p: float,
    back_and_forth: bool,
):
    logger = get_logger()
    tokenizer = AbsTokenizer(config_path=config_path)
    control_sentinel = threading.Event()
    currently_generating_sentinel = threading.Event()

    if midi_through_port:
        close_notes(midi_through_port)
    if midi_out_port:
        close_notes(midi_out_port)

    if midi_path:
        play_file_thread = threading.Thread(
            target=play_midi_file,
            kwargs={
                "midi_through_port": midi_through_port,
                "midi_performance_queue": midi_performance_queue,
                "midi_path": midi_path,
                "currently_generating_sentinel": currently_generating_sentinel,
                "reset_sentinel": reset_sentinel,
            },
        )
    else:
        play_file_thread = None

    keypress_thread = threading.Thread(
        target=listen_for_keypress_control_signal,
        kwargs={
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "back_and_forth": back_and_forth,
        },
    )
    midi_control_thread = threading.Thread(
        target=listen_for_midi_control_signal,
        kwargs={
            "midi_control_queue": midi_control_queue,
            "control_sentinel": control_sentinel,
            "reset_sentinel": reset_sentinel,
            "currently_generating_sentinel": currently_generating_sentinel,
            "midi_control_signal": midi_control_signal,
            "midi_reset_control_signal": midi_reset_control_signal,
            "back_and_forth": back_and_forth,
        },
    )
    keypress_thread.start()
    midi_control_thread.start()

    if play_file_thread is not None:
        play_file_thread.start()

    msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = (
        capture_and_update_kv(
            model=model,
            msgs=[],
            prev_context=[],
            control_sentinel=control_sentinel,
            reset_sentinel=reset_sentinel,
            wait_for_close=wait_for_close,
            midi_performance_queue=midi_performance_queue,
            midi_capture_channel=0,
        )
    )

    curr_midi_channel = 0
    while not reset_sentinel.is_set():
        control_sentinel.clear()
        currently_generating_sentinel.set()
        msgs = stream_msgs(
            model=model,
            tokenizer=tokenizer,
            msgs=msgs,
            prev_context=prev_context,
            midi_output_port=midi_out_port,
            first_on_msg_epoch_ms=first_on_msg_epoch_ms,
            control_sentinel=control_sentinel,
            temperature=temperature,
            min_p=min_p,
            num_preceding_active_pitches=num_active_pitches,
            midi_stream_channel=curr_midi_channel,
            is_ending=False,
        )

        if midi_save_path:
            logger.info(f"Saving result to {midi_save_path}")
            midi = convert_msgs_to_midi(msgs=msgs)
            midi.save(midi_save_path)

        curr_midi_channel += 1
        if curr_midi_channel == 9:  # Skip drum channel
            curr_midi_channel += 1

        control_sentinel.clear()
        if reset_sentinel.is_set():
            return
        else:
            currently_generating_sentinel.clear()
            msgs, prev_context, _, num_active_pitches = capture_and_update_kv(
                model=model,
                msgs=msgs,
                prev_context=prev_context,
                control_sentinel=control_sentinel,
                reset_sentinel=reset_sentinel,
                wait_for_close=wait_for_close,
                midi_performance_queue=midi_performance_queue,
                midi_capture_channel=curr_midi_channel,
                first_msg_epoch_time_ms=first_on_msg_epoch_ms,
            )

    keypress_thread.join()
    midi_control_thread.join()
    if play_file_thread:
        play_file_thread.join()


def insert_embedding(
    model: TransformerLM,
    embedding_model_checkpoint_path: str,
    embedding_midi_path: str,
):
    logger = get_logger()
    logger.info(f"Loading embedding from {embedding_midi_path}")
    emb = _get_embedding(
        embedding_model_checkpoint_path=embedding_model_checkpoint_path,
        embedding_midi_path=embedding_midi_path,
    )
    logger.info(f"Inserting embedding into context")
    model.fill_condition_kv(mx.array([emb], dtype=DTYPE))

    global EMBEDDING_OFFSET
    EMBEDDING_OFFSET = 1


def forward_midi_input_port(
    midi_input_port: str,
    midi_control_queue: queue.Queue,
    midi_performance_queue: queue.Queue | None,
):
    logger = get_logger("MIDI-FORWARD")
    logger.info(f"Forwarding MIDI from port: '{midi_input_port}'")

    if midi_performance_queue is None:
        logger.info(
            f"MIDI file provided - only forwarding {midi_input_port} to control queue"
        )

    try:
        with mido.open_input(midi_input_port) as midi_in:
            while True:
                msg = midi_in.receive(block=True)
                if msg:
                    midi_control_queue.put(msg)
                    if midi_performance_queue is not None:
                        midi_performance_queue.put(msg)

    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Error in MIDI forwarder: {e}")
    finally:
        logger.info("MIDI forwarder has shut down.")


def main(args):
    logger = get_logger()
    model = load_model(checkpoint_path=args.checkpoint)
    model = warmup_model(model=model)
    if args.embedding_checkpoint and args.embedding_midi_path:
        insert_embedding(
            model=model,
            embedding_model_checkpoint_path=args.embedding_checkpoint,
            embedding_midi_path=args.embedding_midi_path,
        )

    assert (args.midi_path and os.path.isfile(args.midi_path)) or args.midi_in

    logger.info(f"Available MIDI ports: {mido.get_output_names()}")
    midi_performance_queue = queue.Queue()
    midi_control_queue = queue.Queue()

    if args.midi_in:
        forwarder_thread = threading.Thread(
            target=forward_midi_input_port,
            kwargs={
                "midi_input_port": args.midi_in,
                "midi_control_queue": midi_control_queue,
                "midi_performance_queue": (
                    midi_performance_queue if args.midi_path is None else None
                ),
            },
            daemon=True,
        )
        forwarder_thread.start()

    reset_sentinel = threading.Event()
    while True:
        run(
            model=model,
            midi_performance_queue=midi_performance_queue,
            midi_control_queue=midi_control_queue,
            midi_through_port=args.midi_through,
            midi_out_port=args.midi_out,
            midi_path=args.midi_path,
            midi_save_path=args.save_path,
            midi_control_signal=args.midi_control_signal,
            midi_reset_control_signal=args.midi_reset_control_signal,
            reset_sentinel=reset_sentinel,
            wait_for_close=args.wait_for_close,
            temperature=args.temp,
            min_p=args.min_p,
            back_and_forth=args.back_and_forth,
        )
        reset_sentinel = threading.Event()


def playback(midi_path: str, midi_out: str, save_path: str | None = None):
    # Mocks generated playback by streaming from a real MIDI file

    close_notes(midi_out)
    starting_epoch_time_ms = get_epoch_time_ms()
    tokenizer = AbsTokenizer(config_path=config_path)
    tokens_queue = queue.Queue()
    midi_messages_queue = queue.Queue()
    stream_midi_results_queue = queue.Queue()
    control_sentinel = threading.Event()

    midi_dict = MidiDict.from_midi(midi_path)
    midi_dict.remove_redundant_pedals()
    tokenized_sequence = tokenizer.tokenize(
        midi_dict,
        add_dim_tok=False,
        remove_preceding_silence=False,
    )
    tokenized_sequence = tokenized_sequence[
        tokenized_sequence.index(tokenizer.bos_tok) + 1 :
    ]

    # Populate token queue synthetically
    for tok in tokenized_sequence:
        tokens_queue.put(tok)

    decode_tokens_to_midi_thread = threading.Thread(
        target=decode_tokens_to_midi,
        kwargs={
            "generated_tokens_queue": tokens_queue,
            "outbound_midi_msg_queue": midi_messages_queue,
            "tokenizer": tokenizer,
            "first_on_msg_epoch_ms": starting_epoch_time_ms,
            "priming_seq_last_onset_ms": 0,
        },
    )
    decode_tokens_to_midi_thread.start()

    stream_midi_thread = threading.Thread(
        target=stream_midi,
        kwargs={
            "inbound_midi_msg_queue": midi_messages_queue,
            "msgs": [],
            "last_channel_msg_epoch_time_ms": starting_epoch_time_ms,
            "midi_output_port": midi_out,
            "control_sentinel": control_sentinel,
            "midi_stream_channel": 0,
            "results_queue": stream_midi_results_queue,
        },
    )
    stream_midi_thread.start()

    decode_tokens_to_midi_thread.join()
    stream_midi_thread.join()
    msgs = stream_midi_results_queue.get()
    mid = convert_msgs_to_midi(msgs)

    if save_path is not None:
        mid.save(save_path)

    return msgs


def close_notes(midi_out_port: str):
    with mido.open_output(midi_out_port) as out:
        out.send(mido.Message(type="control_change", control=64, value=0))
        for note in range(128):
            out.send(mido.Message("note_off", note=note, velocity=0))


if __name__ == "__main__":
    args = parse_args()

    if args.hardware:
        set_calibration_settings(args.hardware)

    if args.playback is True:
        # Playback only mode for testing
        assert args.midi_path is not None, "Must provide midi_path"
        try:
            playback(
                midi_path=args.midi_path,
                midi_out=args.midi_out,
                save_path=args.save_path,
            )
        except KeyboardInterrupt:
            close_notes(args.midi_out)
    else:
        try:
            main(args)
        except KeyboardInterrupt:
            close_notes(args.midi_out)
