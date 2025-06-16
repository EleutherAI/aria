#!/usr/bin/env python3

import argparse
import os
import time
import uuid
import copy
import random
import logging
import threading
import queue
import copy
import mido
import torch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from contextlib import ExitStack

from ariautils.midi import MidiDict, midi_to_dict
from ariautils.tokenizer import AbsTokenizer
from aria.inference.model_mlx import TransformerLM
from aria.model import ModelConfig
from aria.config import load_model_config

DTYPE = mx.float32
MAX_SEQ_LEN = 2048
PREFILL_CHUNK_SIZE_L = 128
PREFILL_CHUNK_SIZE = 16
RECALC_DUR_PREFILL_CHUNK_SIZE = 8
RECALC_DUR_BUFFER_MS = 100

BEAM_WIDTH = 3
TIME_TOK_WEIGHTING = -5
FIRST_ONSET_BUFFER_MS = -200  # Controls onset timing for first generated note

# HARDWARE: Decoded logits are masked for durations < MIN_NOTE_LEN_MS
# HARDWARE: Sends early off-msg if pitch is on MIN_NOTE_DELTA_MS before on-msg
# HARDWARE: All messages are sent HARDWARE_LATENCY_MS early
MIN_NOTE_DELTA_MS = 100
MIN_NOTE_LEN_MS = 50
HARDWARE_LATENCY_MS = 150  # There is a bug with how this works
MAX_STREAM_DELAY_MS = 25

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


def get_epoch_time_ms() -> int:
    return round(time.time() * 1000)


def prefill(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
    pad_idxs: mx.array | None = None,
) -> mx.array:
    # pad_idxs is only needed for prepended pad tokens
    logits = model(
        idxs=idxs,
        input_pos=input_pos,
        offset=input_pos[0],
        pad_idxs=pad_idxs,
    )

    return logits


def decode_one(
    model: TransformerLM,
    idxs: mx.array,
    input_pos: mx.array,
    pad_idxs: mx.array | None = None,
) -> mx.array:
    # pad_idxs is only needed for prepended pad tokens
    assert input_pos.shape[-1] == 1

    logits = model(
        idxs=idxs,
        input_pos=input_pos,
        offset=input_pos[0],
        pad_idxs=pad_idxs,
    )[:, -1]

    return logits


def sample_min_p(probs: mx.array, p_base: float):
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


def _compile_prefill(
    model: TransformerLM,
    logger: logging.Logger,
    chunk_size: int,
):
    assert chunk_size > 1

    compile_start_time_s = time.time()
    logger.info(f"Compiling prefill (chunk_size={chunk_size})")
    for _start_idx in range(0, MAX_SEQ_LEN, chunk_size * 4):
        mx.eval(
            prefill(
                model,
                idxs=mx.ones([1, chunk_size], dtype=mx.int32),
                input_pos=mx.arange(
                    _start_idx, _start_idx + chunk_size, dtype=mx.int32
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


def _compile_decode_one(
    model: TransformerLM,
    logger: logging.Logger,
):
    # Don't need to explicitly compile with mlx, instead we are just precalculating
    # the computation graphs for different shapes
    compile_start_time_s = time.time()
    for _start_idx in range(0, MAX_SEQ_LEN, 4):
        mx.eval(
            decode_one(
                model,
                idxs=mx.array([[random.randint(0, 20)]], dtype=mx.int32),
                input_pos=mx.array([_start_idx], dtype=mx.int32),
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


def compile_model(model: TransformerLM):
    logger = get_logger()

    model.eval()
    model.setup_cache(
        batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dtype=DTYPE,
    )

    model = _compile_decode_one(model=model, logger=logger)
    for chunk_size in list(
        {
            # PREFILL_CHUNK_SIZE_L,
            PREFILL_CHUNK_SIZE,
            RECALC_DUR_PREFILL_CHUNK_SIZE,
        }
    ):
        model = _compile_prefill(
            model=model, logger=logger, chunk_size=chunk_size
        )

    return model


def load_model(
    checkpoint_path: str,
):
    logger = get_logger()

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)

    logging.info(f"Loading model weights from {checkpoint_path}")

    init_start_time_s = time.time()
    model = TransformerLM(model_config)
    model.load_weights(checkpoint_path, strict=False)
    nn.quantize(model.model, group_size=64, bits=8)
    model.eval()

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

    buffer_ms = FIRST_ONSET_BUFFER_MS - HARDWARE_LATENCY_MS
    time_tok_id = tokenizer.tok_to_id[tokenizer.time_tok]
    eos_tok_id = tokenizer.tok_to_id[tokenizer.eos_tok]
    dim_tok_id = tokenizer.tok_to_id[tokenizer.dim_tok]

    logits = first_token_logits
    time_since_first_onset_ms = get_epoch_time_ms() - first_on_msg_epoch_ms
    idx = len(priming_seq) + 1

    num_time_toks_required = (time_since_first_onset_ms + buffer_ms) // 5000
    num_time_toks_in_priming_seq = priming_seq.count(tokenizer.time_tok)
    num_time_toks_to_add = num_time_toks_required - num_time_toks_in_priming_seq

    logger.info(f"Time since first onset: {time_since_first_onset_ms}ms")

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

    masked_onset_ids = [
        tokenizer.tok_to_id[tok]
        for tok in tokenizer.onset_tokens
        if tok[1] < ((time_since_first_onset_ms + buffer_ms) % 5000)
    ]

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
        next_log_probs[:, masked_onset_ids] = float("-inf")
        next_log_probs[:, eos_tok_id] = float("-inf")
        next_log_probs[:, dim_tok_id] = float("-inf")
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
    logger.info(
        f"Inserted {best_tok_2} at position {idx} without updating KV-Cache"
    )

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

    # TODO: This seems to fix issues?
    if control_sentinel.is_set():
        control_sentinel.clear()

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

        logits[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")
        if is_ending is False:
            logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")

        for dur_ms in range(0, MIN_NOTE_LEN_MS, 10):
            logits[:, tokenizer.tok_to_id[("dur", dur_ms)]] = float("-inf")

        if temperature > 0.0:
            probs = mx.softmax(logits / temperature, axis=-1)
            next_token_ids = sample_min_p(probs, min_p).flatten()
        else:
            next_token_ids = mx.argmax(logits, axis=-1).flatten()

        enc_seq[:, idx] = next_token_ids
        next_token = tokenizer.id_to_tok[next_token_ids[0].item()]
        logger.debug(
            f"({(time.time() - decode_one_start_time_s)*1000:.2f}ms) {idx}: {next_token}"
        )

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

    start_idx = max(2, priming_seq_len - 4 * num_preceding_active_pitches - 1)
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


def decode_tokens_to_midi(
    generated_tokens_queue: queue.Queue,
    outbound_midi_msg_queue: queue.Queue,
    tokenizer: AbsTokenizer,
    first_on_msg_epoch_ms: int,
    priming_seq_last_onset_ms: int,
):
    logger = get_logger("DECODE")

    assert (
        first_on_msg_epoch_ms + priming_seq_last_onset_ms < get_epoch_time_ms()
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
                _uuid = uuid.uuid4()
                end_msg = {
                    "pitch": -1,
                    "vel": -1,
                    "epoch_time_ms": offset_epoch_ms + 100,  # Last note offset
                    "uuid": _uuid,
                }  # pitch=-1 denotes end_msg
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
                break

        while note_buffer and note_buffer[0] == tokenizer.time_tok:
            logger.debug("Popping time_tok")
            num_time_toks += 1
            note_buffer.pop(0)

        assert len(note_buffer) == 3
        logger.debug(f"Decoded note: {note_buffer}")
        note_tok, onset_tok, dur_tok = note_buffer
        _, pitch, vel = note_tok
        _, onset = onset_tok
        _, dur = dur_tok

        _uuid = uuid.uuid4()
        onset_epoch_ms = first_on_msg_epoch_ms + (num_time_toks * 5000) + onset
        offset_epoch_ms = onset_epoch_ms + dur
        on_msg = {
            "pitch": pitch,
            "vel": vel,
            "epoch_time_ms": onset_epoch_ms,
            "uuid": _uuid,
        }
        off_msg = {
            "pitch": pitch,
            "vel": 0,
            "epoch_time_ms": offset_epoch_ms,
            "uuid": _uuid,
        }

        # Not thread safe but in theory should be ok?
        if pitch_to_prev_msg.get(pitch) is not None and MIN_NOTE_DELTA_MS > 0:
            prev_on, prev_off = pitch_to_prev_msg.get(pitch)
            adj_off_time = max(
                min(
                    prev_off["epoch_time_ms"],
                    onset_epoch_ms - MIN_NOTE_DELTA_MS,
                ),
                prev_on["epoch_time_ms"],
            )
            if adj_off_time != prev_off["epoch_time_ms"]:
                logger.debug(f"Adjusting {prev_off}: t={adj_off_time}")
                prev_off["epoch_time_ms"] = adj_off_time
                prev_off["adjusted"] = True

        pitch_to_prev_msg[pitch] = [on_msg, off_msg]

        outbound_midi_msg_queue.put(on_msg)
        outbound_midi_msg_queue.put(off_msg)
        logger.debug(f"Put message: {on_msg}")
        logger.debug(f"Put message: {off_msg}")
        logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

        note_buffer = []


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
    logger.info(
        f"Sending generated messages on MIDI port: '{midi_output_port}'"
    )
    logger.info(
        f"Applying hardware latency adjustment: {HARDWARE_LATENCY_MS}ms"
    )

    active_pitch_uuid = {}
    is_pitch_active = {}
    midi_msgs = []

    with mido.open_output(midi_output_port) as midi_out:
        while not control_sentinel.is_set():
            while True:
                try:
                    msg = inbound_midi_msg_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    logger.debug(f"Received message: {msg}")
                    midi_msgs.append(msg)

            midi_msgs = sorted(
                midi_msgs,
                key=lambda msg: (
                    msg["epoch_time_ms"],
                    msg["vel"],
                ),
            )

            if control_sentinel.is_set():
                break

            while midi_msgs:
                latency_adjusted_epoch_time_ms = (
                    get_epoch_time_ms() + HARDWARE_LATENCY_MS
                )
                msg = midi_msgs[0]

                if (
                    0
                    < latency_adjusted_epoch_time_ms - msg["epoch_time_ms"]
                    <= MAX_STREAM_DELAY_MS
                ):
                    if msg["pitch"] == -1:  # End msg
                        control_sentinel.set()
                        break

                    mido_msg = mido.Message(
                        "note_on",
                        note=msg["pitch"],
                        velocity=msg["vel"],
                        channel=0,
                        time=0,
                    )

                    if msg["vel"] > 0:
                        active_pitch_uuid[msg["pitch"]] = msg["uuid"]
                        should_send_midi_out = True
                        should_append_to_msgs = True
                    elif msg.get("adjusted", False) is True:
                        should_send_midi_out = True
                        should_append_to_msgs = False
                    else:
                        should_send_midi_out = (
                            active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
                        )
                        should_append_to_msgs = should_send_midi_out

                    if should_send_midi_out is True:
                        midi_out.send(mido_msg)
                        is_pitch_active[msg["pitch"]] = msg["vel"] != 0
                        logger.info(f"Sent message: {mido_msg}")
                    if should_append_to_msgs is True:
                        mido_msg_with_time = copy.deepcopy(mido_msg)
                        mido_msg_with_time.channel = midi_stream_channel
                        mido_msg_with_time.time = max(
                            0,
                            msg["epoch_time_ms"]
                            - last_channel_msg_epoch_time_ms,
                        )
                        last_channel_msg_epoch_time_ms = msg["epoch_time_ms"]
                        msgs.append(mido_msg_with_time)

                    midi_msgs.pop(0)

                elif (
                    latency_adjusted_epoch_time_ms - msg["epoch_time_ms"]
                    > MAX_STREAM_DELAY_MS
                ):
                    # Message occurs too far in the past
                    logger.debug(
                        f"Skipping message occurring too far ({latency_adjusted_epoch_time_ms - msg['epoch_time_ms']}ms) in the past: {msg}"
                    )
                    midi_msgs.pop(0)
                else:
                    # Message occurs in the future
                    break

            time.sleep(0.005)

        remaining_note_off_messages = [
            msg
            for msg in midi_msgs
            if msg["vel"] == 0
            and active_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
        ]

        logger.info("Processing remaining note_off messages")
        for msg in remaining_note_off_messages:
            mido_msg = mido.Message(
                "note_on",
                note=msg["pitch"],
                velocity=0,
                channel=midi_stream_channel,
                time=msg["epoch_time_ms"] - last_channel_msg_epoch_time_ms,
            )
            midi_out.send(mido_msg)
            last_channel_msg_epoch_time_ms = msg["epoch_time_ms"]
            msgs.append(mido_msg)

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
    midi = convert_msgs_to_midi(msgs=msgs)
    midi_dict = MidiDict(**midi_to_dict(midi))
    priming_seq = tokenizer.tokenize(midi_dict=midi_dict, add_dim_tok=False)
    priming_seq = priming_seq[: priming_seq.index(tokenizer.eos_tok)]

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
        daemon=True,
    )
    stream_midi_thread.start()

    generate_tokens_thread.join()
    decode_tokens_to_midi_thread.join()
    msgs = stream_midi_results_queue.get()

    if is_ending is True:
        stream_midi_thread.join()

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
    tokenizer = AbsTokenizer()
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

        # TODO: This workaround is not good enough. Instead just loop is
        # curr context has no notes.
        if (msg_cnt >= 10 or seen_sentinel) and len(msgs) > 75:
            midi = convert_msgs_to_midi(msgs=msgs)
            midi_dict = MidiDict(**midi_to_dict(midi))
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
    wait_for_close: bool,
    midi_input_port: str,
    midi_capture_channel: int,
    midi_control_signal: int | None = None,
    midi_through_port: str | None = None,
    first_msg_epoch_time_ms: int | None = None,
):
    received_messages_queue = queue.Queue()
    results_queue = queue.Queue()
    capture_midi_thread = threading.Thread(
        target=capture_midi_input,
        kwargs={
            "midi_input_port": midi_input_port,
            "control_sentinel": control_sentinel,
            "received_messages_queue": received_messages_queue,
            "midi_capture_channel": midi_capture_channel,
            "midi_control_signal": midi_control_signal,
            "midi_through_port": midi_through_port,
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


# TODO: Change MIDI-through logic, this is not the best way to mock playing
def capture_midi_input(
    midi_input_port: str,
    control_sentinel: threading.Event,
    received_messages_queue: queue.Queue,
    midi_capture_channel: int,
    results_queue: queue.Queue,
    midi_control_signal: int | None = None,
    midi_through_port: str | None = None,
    first_msg_epoch_time_ms: int | None = None,
    wait_for_close: bool = False,
):
    logger = get_logger("CAPTURE")
    active_pitches = set()
    first_on_msg_epoch_ms = None
    prev_msg_epoch_time_ms = first_msg_epoch_time_ms  #

    if midi_through_port is not None:
        logger.info(f"Sending through on MIDI port: '{midi_through_port}'")

    with ExitStack() as stack:
        midi_input = stack.enter_context(mido.open_input(midi_input_port))
        midi_through = (
            stack.enter_context(mido.open_output(midi_through_port))
            if midi_through_port
            else None
        )
        logger.info(f"Listening on MIDI port: '{midi_input_port}'")
        logger.info(f"Ready to capture MIDI events")

        if midi_control_signal is not None:
            logger.info(
                f"Commencing generation upon keypress or MIDI control: {midi_control_signal}"
            )
        else:
            logger.info(f"Commencing generation upon keypress")

        while not control_sentinel.is_set() or (
            wait_for_close and active_pitches
        ):
            msg = midi_input.receive(block=False)

            if msg is None:
                time.sleep(0.001)
                continue

            if prev_msg_epoch_time_ms is None:
                msg_time_ms = 0
            else:
                msg_time_ms = get_epoch_time_ms() - prev_msg_epoch_time_ms

            prev_msg_epoch_time_ms = get_epoch_time_ms()
            msg.time = msg_time_ms
            msg.channel = midi_capture_channel
            logger.info(f"Received message: [{msg}]")

            if msg.is_meta is True or msg.type == "program_change":
                continue

            if (
                msg.type == "note_on" and msg.velocity == 0
            ) or msg.type == "note_off":
                active_pitches.discard(msg.note)
                received_messages_queue.put(msg)
                if midi_through is not None:
                    midi_through.send(msg)
            elif msg.type == "note_on" and msg.velocity > 0:
                if first_on_msg_epoch_ms is None:
                    first_on_msg_epoch_ms = get_epoch_time_ms()

                active_pitches.add(msg.note)
                received_messages_queue.put(msg)
                if midi_through is not None:
                    midi_through.send(msg)
            elif msg.type == "control_change" and msg.control == 64:
                received_messages_queue.put(msg)
            elif (
                msg.type == "control_change"
                and msg.control == midi_control_signal
                and msg.value > 0
            ):
                control_sentinel.set()
                logger.info("Control signal seen")

        logger.info(f"Active pitches: {active_pitches}")
        num_active_pitches = len(active_pitches)

        if active_pitches:
            pitch = active_pitches.pop()
            msg = mido.Message(
                type="note_on",
                note=pitch,
                velocity=0,
                channel=midi_capture_channel,
                time=get_epoch_time_ms() - prev_msg_epoch_time_ms,
            )
            received_messages_queue.put(msg)
            if midi_through is not None:
                midi_through.send(msg)

            while active_pitches:
                pitch = active_pitches.pop()
                msg = mido.Message(
                    type="note_on",
                    note=pitch,
                    velocity=0,
                    channel=midi_capture_channel,
                    time=0,
                )
                received_messages_queue.put(msg)
                if midi_through is not None:
                    midi_through.send(msg)

        # Turn off pedal
        msg = mido.Message(
            type="control_change",
            control=64,
            value=0,
            channel=midi_capture_channel,
            time=0,
        )
        received_messages_queue.put(msg)
        if midi_through is not None:
            midi_through.send(msg)

        received_messages_queue.put(None)  # Sentinel
        results_queue.put((first_on_msg_epoch_ms, num_active_pitches))


def play_midi_file(midi_port: str, midi_path: str):
    logger = get_logger("FILE")
    logger.info(f"Playing file at {midi_path} on MIDI port '{midi_port}'")

    midi_dict = MidiDict.from_midi(midi_path)

    if MIN_NOTE_DELTA_MS > 0:
        midi_dict.enforce_gaps(min_gap_ms=MIN_NOTE_DELTA_MS)

    mid = midi_dict.to_midi()

    time.sleep(1)
    active_pitches = []
    with mido.open_output(midi_port) as output_port:
        for msg in mid.play():
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in active_pitches:
                    _off_msg = copy.deepcopy(msg)
                    _off_msg.velocity = 0
                    output_port.send(_off_msg)
                else:
                    active_pitches.append(msg.note)
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in active_pitches:
                    active_pitches.remove(msg.note)

            logger.debug(f"{msg}")
            output_port.send(msg)


def listen_for_keypress_control_signal(
    control_sentinel: threading.Event,
    generate_ending_sentinel: threading.Event,
):
    logger = get_logger("KEYBOARD")
    while True:
        time.sleep(5)
        _input = input()
        logger.info(f'Keypress seen "{_input}"')
        if _input == "":
            control_sentinel.set()
        else:
            control_sentinel.set()
            generate_ending_sentinel.set()
            return


def _listen(
    midi_input_port: str,
    logger: logging.Logger,
    midi_control_signal: int | None = None,
):
    logger.info("Listening...")
    with mido.open_input(midi_input_port) as midi_input:
        while True:
            msg = midi_input.receive(block=False)
            if msg is None:
                time.sleep(0.01)
            elif (
                msg.type == "control_change"
                and msg.control == midi_control_signal
                and msg.value >= 64
            ):
                return


def listen_for_midi_control_signal(
    midi_input_port: str,
    control_sentinel: threading.Event,
    midi_control_signal: int | None = None,
):
    logger = get_logger("MIDI-CONTROL")

    while True:
        _listen(
            midi_input_port=midi_input_port,
            midi_control_signal=midi_control_signal,
            logger=logger,
        )
        control_sentinel.set()
        logger.info("Seen MIDI control signal")
        time.sleep(5)


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
        "--save_path",
        type=str,
        required=False,
        help="Path to save complete MIDI file",
    )

    return argp.parse_args()


# TODO: Need functionality for handing case where we run out of model context


def main(args):
    args = parse_args()
    logger = get_logger()
    tokenizer = AbsTokenizer()
    model = load_model(checkpoint_path=args.checkpoint)
    model = compile_model(model=model)

    assert (args.midi_path and os.path.isfile(args.midi_path)) or args.midi_in
    if args.midi_path:
        midi_input_port = "IAC Driver Bus 1"
        play_file_thread = threading.Thread(
            target=play_midi_file,
            args=(midi_input_port, args.midi_path),
            daemon=True,
        )
    else:
        midi_input_port = args.midi_in
        play_file_thread = None

    control_sentinel = threading.Event()
    generate_ending_sentinel = threading.Event()
    keypress_thread = threading.Thread(
        target=listen_for_keypress_control_signal,
        args=[control_sentinel, generate_ending_sentinel],
        daemon=True,
    )
    midi_control_thread = threading.Thread(
        target=listen_for_midi_control_signal,
        kwargs={
            "midi_input_port": (
                args.midi_in if args.midi_in else midi_input_port
            ),
            "control_sentinel": control_sentinel,
            "midi_control_signal": args.midi_control_signal,
        },
        daemon=True,
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
            wait_for_close=args.wait_for_close,
            midi_input_port=midi_input_port,
            midi_control_signal=args.midi_control_signal,
            midi_through_port=args.midi_through,
            midi_capture_channel=0,
        )
    )

    curr_midi_channel = 0
    while True:
        control_sentinel.clear()
        msgs = stream_msgs(
            model=model,
            tokenizer=tokenizer,
            msgs=msgs,
            prev_context=prev_context,
            midi_output_port=args.midi_out,
            first_on_msg_epoch_ms=first_on_msg_epoch_ms,
            control_sentinel=control_sentinel,
            temperature=args.temp,
            min_p=args.min_p,
            num_preceding_active_pitches=num_active_pitches,
            midi_stream_channel=curr_midi_channel,
            is_ending=False,
        )

        curr_midi_channel += 1
        if curr_midi_channel == 9:
            curr_midi_channel += 1

        control_sentinel.clear()
        if generate_ending_sentinel.is_set():
            break
        else:
            msgs, prev_context, _, num_active_pitches = capture_and_update_kv(
                model=model,
                msgs=msgs,
                prev_context=prev_context,
                control_sentinel=control_sentinel,
                wait_for_close=args.wait_for_close,
                midi_input_port=midi_input_port,
                midi_control_signal=args.midi_control_signal,
                midi_through_port=args.midi_through,
                midi_capture_channel=curr_midi_channel,
                first_msg_epoch_time_ms=first_on_msg_epoch_ms,
            )

    # Generate ending
    msgs = stream_msgs(
        model=model,
        tokenizer=tokenizer,
        msgs=msgs,
        prev_context=prev_context,
        midi_output_port=args.midi_out,
        first_on_msg_epoch_ms=first_on_msg_epoch_ms,
        control_sentinel=control_sentinel,
        temperature=args.temp / 2,
        min_p=args.min_p,
        num_preceding_active_pitches=num_active_pitches,
        midi_stream_channel=curr_midi_channel,
        is_ending=True,
    )

    if args.save_path:
        logger.info(f"Saving result to {args.save_path}")
        midi = convert_msgs_to_midi(msgs=msgs)
        midi.save(args.save_path)


def exit(midi_out_port: str):
    with mido.open_output(midi_out_port) as out:
        for note in range(128):
            out.send(mido.Message("note_off", note=note, velocity=0))


if __name__ == "__main__":
    args = parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        exit(args.midi_out)
