#!/usr/bin/env python3

import argparse
import os
import time
import uuid
import copy
import logging
import threading
import queue
import copy
import torch
import mido
import torch._inductor.config

from torch.cuda import is_available as cuda_is_available
from contextlib import ExitStack

from ariautils.midi import MidiDict, midi_to_dict
from ariautils.tokenizer import AbsTokenizer
from aria.utils import _load_weight
from aria.inference import TransformerLM
from aria.model import ModelConfig
from aria.config import load_model_config
from aria.sample import sample_min_p

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_SEQ_LEN = 4096
PREFILL_CHUNK_SIZE = 32
RECALC_DUR_PREFILL_CHUNK_SIZE = 8
RECALC_DUR_BUFFER_MS = 50

# Decode first
BEAM_WIDTH = 3
TIME_TOK_WEIGHTING = -5

# HARDWARE: Decoded logits are masked for durations < MIN_NOTE_LEN_MS
# HARDWARE: Sends early off-msg if pitch is on MIN_NOTE_DELTA_MS before on-msg
# HARDWARE: All messages are sent HARDWARE_LATENCY_MS early
MIN_NOTE_DELTA_MS = 100
MIN_NOTE_LEN_MS = 200
HARDWARE_LATENCY_MS = 0

# TODO:
# - Add CFG support (eek)


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


@torch.autocast("cuda", dtype=DTYPE)
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
    )

    return logits


@torch.autocast("cuda", dtype=DTYPE)
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


def _compile_prefill(
    model: TransformerLM,
    logger: logging.Logger,
    chunk_size: int,
):
    assert chunk_size > 1

    global prefill
    prefill = torch.compile(
        prefill,
        mode="reduce-overhead",
        fullgraph=True,
    )
    start_compile_time_s = time.time()
    logger.info(f"Compiling prefill (chunk_size={chunk_size})")
    prefill(
        model,
        idxs=torch.ones(1, chunk_size, device="cuda", dtype=torch.int),
        input_pos=torch.arange(0, chunk_size, device="cuda", dtype=torch.int),
    )
    logger.info(
        f"Finished compiling - took {time.time() - start_compile_time_s:.4f} seconds"
    )

    for _ in range(5):
        prefill(
            model,
            idxs=torch.ones(1, chunk_size, device="cuda", dtype=torch.int),
            input_pos=torch.arange(
                0, chunk_size, device="cuda", dtype=torch.int
            ),
        )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    prefill(
        model,
        idxs=torch.ones(1, chunk_size, device="cuda", dtype=torch.int),
        input_pos=torch.arange(0, chunk_size, device="cuda", dtype=torch.int),
    )
    end_event.record()
    end_event.synchronize()
    compiled_prefill_ms = start_event.elapsed_time(end_event)
    compiled_prefill_its = 1000 / compiled_prefill_ms
    logger.info(
        f"Compiled prefill benchmark: {compiled_prefill_ms:.2f} ms/it ({compiled_prefill_its:.2f} it/s)"
    )

    return model


def _compile_decode_one(model: TransformerLM, logger: logging.Logger):
    global decode_one
    decode_one = torch.compile(
        decode_one,
        mode="reduce-overhead",
        fullgraph=True,
    )

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        start_compile_time_s = time.time()
        logger.info(f"Compiling decode_one")
        decode_one(
            model,
            idxs=torch.tensor([[0]], device="cuda", dtype=torch.int),
            input_pos=torch.tensor([0], device="cuda", dtype=torch.int),
        )
        logger.info(
            f"Finished compiling - took {time.time() - start_compile_time_s:.4f} seconds"
        )

        for _ in range(5):
            decode_one(
                model,
                idxs=torch.tensor([[0]], device="cuda", dtype=torch.int).cuda(),
                input_pos=torch.tensor([0], device="cuda", dtype=torch.int),
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        decode_one(
            model,
            idxs=torch.tensor([[0]], device="cuda", dtype=torch.int).cuda(),
            input_pos=torch.tensor([0], device="cuda", dtype=torch.int),
        )
        end_event.record()
        end_event.synchronize()

        compiled_forward_ms = start_event.elapsed_time(end_event)
        compiled_forward_its = 1000 / compiled_forward_ms
        logger.info(
            f"Compiled decode_one benchmark: {compiled_forward_ms:.2f} ms/it ({compiled_forward_its:.2f} it/s)"
        )

        return model


@torch.inference_mode()
def compile_model(model: TransformerLM, max_seq_len: int):
    logger = get_logger()
    assert 10 < max_seq_len <= MAX_SEQ_LEN

    model.eval()
    model.setup_cache(
        batch_size=1,
        max_seq_len=max_seq_len,
        dtype=DTYPE,
    )

    model = _compile_decode_one(model=model, logger=logger)
    for chunk_size in list({PREFILL_CHUNK_SIZE, RECALC_DUR_PREFILL_CHUNK_SIZE}):
        model = _compile_prefill(
            model=model, logger=logger, chunk_size=chunk_size
        )

    return model


def load_model(
    checkpoint_path: str,
):
    logger = get_logger()
    if not cuda_is_available():
        raise Exception("CUDA device is not available.")

    init_start_time_s = time.time()

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).cuda()

    logging.info(f"Loading model weights from {checkpoint_path}")
    model_state = _load_weight(checkpoint_path, "cuda")
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    }
    try:
        model.load_state_dict(model_state)
    except Exception:
        logger.info("Failed to load model, attempting with strict=False...")
        model.load_state_dict(model_state, strict=False)

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
        priming_seq[:chunk_start], onset=True
    )
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
                > last_offset_ms - RECALC_DUR_BUFFER_MS
            ):
                logger.info(
                    f"Found token to resample at {pos}: {prim_tok} -> {pred_tok}"
                )
                return pos

    return None


# TODO: I'm still not 100% sure this is bug free.
# A good debugging strat would be to run it over and over again until we
# cover all of the edge cases
@torch.inference_mode()
def recalc_dur_tokens_chunked(
    model: TransformerLM,
    priming_seq: list,
    enc_seq: torch.Tensor,
    tokenizer: AbsTokenizer,
    start_idx: int,
):
    """Speculative-decoding inspired duration re-calculation"""
    assert start_idx > 0
    logger = get_logger("GENERATE")

    priming_len = len(priming_seq)
    last_offset = tokenizer.calc_length_ms(priming_seq)

    idx = start_idx
    while idx <= priming_len:
        end_idx = idx + RECALC_DUR_PREFILL_CHUNK_SIZE

        window_ids = torch.tensor(
            enc_seq[:, idx - 1 : end_idx - 1].tolist(),
            device="cuda",
            dtype=torch.int,
        )
        window_pos = torch.arange(
            idx - 1, end_idx - 1, device="cuda", dtype=torch.int
        )

        logger.info(
            f"Recalculating chunked durations for positions: {idx-1} - {end_idx-2}"
        )
        logger.debug(f"Inserted: {tokenizer.decode(window_ids[0].tolist())}")
        logger.debug(f"Positions: {window_pos.tolist()}")

        logits = prefill(model, idxs=window_ids, input_pos=window_pos)
        pred_ids = logits.argmax(dim=-1).flatten().tolist()

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
            idx = bad_pos

    next_logits = logits[:, priming_len - idx]

    return enc_seq, priming_seq, next_logits


# TODO: This is now the latency bottleneck.
# Ideas for reducing it:
# - Get rid of the manual time_tok insert stuff, instead just mask logits
#   for all invalid tokens, this should force the model to sample a time tok
#   if there aren't any other valid options
@torch.inference_mode()
def decode_first_tokens(
    model: TransformerLM,
    first_token_logits: torch.Tensor,
    enc_seq: torch.Tensor,
    priming_seq: list,
    tokenizer: AbsTokenizer,
    generated_tokens_queue: queue.Queue,
    first_on_msg_epoch_ms: int,
):
    logger = get_logger("GENERATE")

    BUFFER_MS = 50 + HARDWARE_LATENCY_MS
    TIME_TOK_ID = tokenizer.tok_to_id[tokenizer.time_tok]

    logits = first_token_logits
    time_since_first_onset_ms = get_epoch_time_ms() - first_on_msg_epoch_ms
    idx = len(priming_seq) + 1

    num_time_toks_required = (time_since_first_onset_ms + BUFFER_MS) // 5000
    num_time_toks_in_priming_seq = priming_seq.count(tokenizer.time_tok)
    num_time_toks_to_add = num_time_toks_required - num_time_toks_in_priming_seq

    logger.info(f"Time since first onset: {time_since_first_onset_ms}ms")

    while num_time_toks_to_add > 0:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            generated_tokens_queue.put(tokenizer.time_tok)
            logits = decode_one(
                model,
                idxs=torch.tensor(
                    [[TIME_TOK_ID]], device="cuda", dtype=torch.int
                ),
                input_pos=torch.tensor(
                    [idx - 1], device="cuda", dtype=torch.int
                ),
            )

        logger.info(f"Inserted time_tok at position {idx-1}")
        num_time_toks_to_add -= 1
        enc_seq[:, idx - 1] = torch.tensor([[TIME_TOK_ID]]).cuda()
        idx += 1

    logits[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")
    logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")

    log_probs = torch.log_softmax(logits, dim=-1)
    top_log_probs, top_ids = torch.topk(log_probs, k=BEAM_WIDTH, dim=-1)

    if TIME_TOK_ID not in top_ids[0].tolist():
        top_ids[0, -1] = TIME_TOK_ID
        top_log_probs[0, -1] = log_probs[0, TIME_TOK_ID] + TIME_TOK_WEIGHTING

    top_toks = [tokenizer.id_to_tok[id] for id in top_ids[0].tolist()]

    logger.debug(f"Calculated top {BEAM_WIDTH} tokens={top_toks}")
    logger.debug(
        f"Calculated top {BEAM_WIDTH} scores={top_log_probs[0].tolist()}"
    )

    masked_onset_ids = [
        tokenizer.tok_to_id[tok]
        for tok in tokenizer.onset_tokens
        if tok[1] < ((time_since_first_onset_ms + BUFFER_MS) % 5000)
    ]

    logger.debug(
        f"Masking onsets for {len(masked_onset_ids)} tokens ({time_since_first_onset_ms + BUFFER_MS})"
    )

    best_score = float("-inf")
    for i in range(BEAM_WIDTH):
        tok = top_toks[i]
        tok_id = top_ids[0, i].item()
        tok_log_prob = top_log_probs[0, i]

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            next_logits = decode_one(
                model,
                idxs=torch.tensor([[tok_id]], device="cuda", dtype=torch.int),
                input_pos=torch.tensor(
                    [idx - 1], device="cuda", dtype=torch.int
                ),
            )
            logger.debug(
                f"Sampled logits for positions {idx} by inserting {tok} at position {idx-1}"
            )

        next_log_probs = torch.log_softmax(next_logits, dim=-1)
        next_log_probs[:, masked_onset_ids] = float("-inf")
        if tok_id == TIME_TOK_ID:
            next_log_probs[:, TIME_TOK_ID] = float("-inf")

        next_tok_log_prob, next_tok_id = torch.max(next_log_probs, dim=-1)
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

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        decode_one(
            model,
            idxs=torch.tensor(
                [[best_tok_id_1]], device="cuda", dtype=torch.int
            ),
            input_pos=torch.tensor([idx - 1], device="cuda", dtype=torch.int),
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
    enc_seq: torch.Tensor,
    tokenizer: AbsTokenizer,
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    idx: int,
    temperature: float,
    min_p: float,
):
    logger = get_logger("GENERATE")
    logger.info(
        f"Using sampling parameters: temperature={temperature}, min_p={min_p}"
    )

    while (not control_sentinel.is_set()) and idx < MAX_SEQ_LEN:
        decode_one_start_time_s = time.time()

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            prev_tok_id = enc_seq[0, idx - 1]
            prev_tok = tokenizer.id_to_tok[prev_tok_id.item()]

            logits = decode_one(
                model,
                idxs=torch.tensor(
                    [[prev_tok_id]], device="cuda", dtype=torch.int
                ),
                input_pos=torch.tensor(
                    [idx - 1], device="cuda", dtype=torch.int
                ),
            )

            logger.debug(
                f"Sampled logits for positions {idx} by inserting {prev_tok} at position {idx-1}"
            )

        logits[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")
        for dur_ms in range(0, MIN_NOTE_LEN_MS, 10):
            logits[:, tokenizer.tok_to_id[("dur", dur_ms)]] = float("-inf")

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token_ids = sample_min_p(probs, min_p).flatten()
        else:
            next_token_ids = torch.argmax(logits, dim=-1).flatten()

        enc_seq[:, idx] = next_token_ids
        next_token = tokenizer.id_to_tok[next_token_ids[0].item()]
        logger.debug(
            f"({(time.time() - decode_one_start_time_s)*1000:.2f}ms) {idx}: {next_token}"
        )

        if next_token == tokenizer.eos_tok:
            logger.info("EOS token produced, exiting...")
            generated_tokens_queue.put(next_token)
            return
        else:
            generated_tokens_queue.put(next_token)
            idx += 1

    while not control_sentinel.is_set():
        time.sleep(0.1)

    logger.info("Seen exit signal")
    generated_tokens_queue.put(None)


@torch.inference_mode()
def generate_tokens(
    priming_seq: list,
    tokenizer: AbsTokenizer,
    model: TransformerLM,
    prev_context: list[int],
    control_sentinel: threading.Event,
    generated_tokens_queue: queue.Queue,
    num_preceding_active_pitches: int,
    first_on_msg_epoch_ms: int,
    temperature: float = 0.97,
    min_p: float = 0.03,
):
    logger = get_logger("GENERATE")

    generate_start_s = time.time()
    priming_seq_len = len(priming_seq)
    start_idx = max(2, priming_seq_len - 4 * num_preceding_active_pitches - 1)
    enc_seq = torch.tensor(
        [
            tokenizer.encode(
                priming_seq
                + [tokenizer.pad_tok] * (MAX_SEQ_LEN - len(priming_seq))
            )
        ],
        device="cuda",
        dtype=torch.int,
    )

    logger.debug(f"Priming sequence {priming_seq}")
    logger.info(f"Priming sequence length: {priming_seq_len}")
    logger.info(f"Prefilling up to (and including) position: {start_idx-1}")

    # In theory we could reuse the logits from prefill
    prefill_start_s = time.time()
    chunked_prefill(
        model=model,
        tokenizer=tokenizer,
        prev_context=prev_context,
        curr_context=enc_seq[0, :start_idx].tolist(),
        full=True,
    )

    torch.cuda.synchronize()
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
    )


def decode_tokens_to_midi(
    generated_tokens_queue: queue.Queue,
    midi_messages_queue: queue.Queue,
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
                    "epoch_time_ms": offset_epoch_ms + 250,  # Last note offset
                    "uuid": _uuid,
                }  # pitch=-1 denotes end_msg
                midi_messages_queue.put(end_msg)
                logger.info(f"Seen exit signal: EOS token")
                logger.debug(f"Put message: {end_msg}")
                return

            elif tok is None:
                logger.info(f"Seen exit signal")
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
            new_prev_off = max(
                min(
                    prev_off["epoch_time_ms"],
                    onset_epoch_ms - MIN_NOTE_DELTA_MS,
                ),
                prev_on["epoch_time_ms"],
            )
            if new_prev_off != prev_off["epoch_time_ms"]:
                logger.info(
                    f"Adjusting prev_off['epoch_time_ms'] ->  new_prev_off"
                )
                prev_off["epoch_time_ms"] = new_prev_off

        pitch_to_prev_msg[pitch] = {"on": on_msg, "off": off_msg}

        midi_messages_queue.put(on_msg)
        midi_messages_queue.put(off_msg)
        logger.debug(f"Put message: {on_msg}")
        logger.debug(f"Put message: {off_msg}")
        logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

        note_buffer = []


# TODO: Test the new changes in decode_tokens_to_midi and clean this fn up.
def stream_midi(
    midi_messages_queue: queue.Queue,
    msgs: list[mido.Message],
    prev_msg_epoch_time_ms: float,
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

    last_pitch_uuid = {}
    pitch_active = {}
    midi_messages = []

    with mido.open_output(midi_output_port) as midi_out:
        while not control_sentinel.is_set():
            while True:
                try:
                    msg = midi_messages_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    logger.debug(f"Received message: {msg}")
                    midi_messages.append(msg)

            midi_messages = sorted(
                midi_messages,
                key=lambda msg: (
                    msg["epoch_time_ms"],
                    msg["vel"],
                ),
            )

            if control_sentinel.is_set():
                break

            while midi_messages:
                latency_adjusted_epoch_time_ms = (
                    get_epoch_time_ms() + HARDWARE_LATENCY_MS
                )
                msg = midi_messages[0]

                if (
                    0
                    < latency_adjusted_epoch_time_ms - msg["epoch_time_ms"]
                    <= 50
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
                        last_pitch_uuid[msg["pitch"]] = msg["uuid"]
                        should_send = True
                    else:
                        # Only send note_off if it matches the last note_on UUID
                        should_send = (
                            last_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
                        )

                    if should_send is True:
                        mido_msg_with_time = copy.deepcopy(mido_msg)
                        mido_msg_with_time.channel = midi_stream_channel
                        mido_msg_with_time.time = max(
                            0, msg["epoch_time_ms"] - prev_msg_epoch_time_ms
                        )
                        prev_msg_epoch_time_ms = msg["epoch_time_ms"]

                        midi_out.send(mido_msg)
                        msgs.append(mido_msg_with_time)
                        pitch_active[msg["pitch"]] = msg["vel"] != 0

                        logger.info(f"Sent message: {mido_msg}")
                    else:
                        logger.debug(
                            f"Skipping note_off message due to uuid mismatch: {msg}"
                        )
                    midi_messages.pop(0)

                elif (
                    latency_adjusted_epoch_time_ms - msg["epoch_time_ms"] > 100
                ):
                    # Message occurs too far in the past
                    logger.debug(
                        f"Skipping message occurring too far ({latency_adjusted_epoch_time_ms - msg["epoch_time_ms"]}ms) in the past: {msg}"
                    )
                    midi_messages.pop(0)
                else:
                    # Message occurs in the future
                    break

            time.sleep(0.005)

        logger.info("Processing remaining note_off messages")
        logger.debug(midi_messages)

        remaining_note_off_messages = [
            msg
            for msg in midi_messages
            if msg["vel"] == 0
            and last_pitch_uuid.get(msg["pitch"]) == msg["uuid"]
        ]

        for msg in remaining_note_off_messages:
            mido_msg = mido.Message(
                "note_on",
                note=msg["pitch"],
                velocity=0,
                channel=midi_stream_channel,
                time=msg["epoch_time_ms"] - prev_msg_epoch_time_ms,
            )
            prev_msg_epoch_time_ms = msg["epoch_time_ms"]
            msgs.append(mido_msg)

        results_queue.put(msgs)

        while remaining_note_off_messages:
            msg = remaining_note_off_messages.pop(0)
            while True:
                latency_adjusted_epoch_time_ms = (
                    get_epoch_time_ms() + HARDWARE_LATENCY_MS
                )

                if (
                    0
                    < latency_adjusted_epoch_time_ms - msg["epoch_time_ms"]
                    <= 50
                ):
                    mido_msg = mido.Message(
                        "note_on",
                        note=msg["pitch"],
                        velocity=0,
                        channel=midi_stream_channel,
                        time=0,  # Does not matter as only used for streaming
                    )
                    midi_out.send(mido_msg)
                    logger.info(f"Sent message: {mido_msg}")
                    break
                else:
                    time.sleep(0.01)


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

    # if tokenizer.dim_tok in priming_seq:
    #     priming_seq.remove(tokenizer.dim_tok)
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
        },
    )
    generate_tokens_thread.start()

    decode_tokens_to_midi_thread = threading.Thread(
        target=decode_tokens_to_midi,
        kwargs={
            "generated_tokens_queue": generated_tokens_queue,
            "midi_messages_queue": midi_messages_queue,
            "tokenizer": tokenizer,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
            "priming_seq_last_onset_ms": tokenizer.calc_length_ms(
                priming_seq, onset=True
            ),
        },
    )
    decode_tokens_to_midi_thread.start()

    stream_midi_results_queue = queue.Queue()
    stream_midi_thread = threading.Thread(
        target=stream_midi,
        kwargs={
            "midi_messages_queue": midi_messages_queue,
            "msgs": msgs,
            "prev_msg_epoch_time_ms": first_on_msg_epoch_ms
            + tokenizer.calc_length_ms(priming_seq, onset=False),
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

    return msgs


# TODO: Channel 9 issues here?
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
):
    agreement_index = 0
    for prev_val, curr_val in zip(prev_context, curr_context):
        if prev_val == curr_val:
            agreement_index += 1
        else:
            logger.info(
                f"Found divergence at position {agreement_index + 1}: {curr_val}, {prev_val}"
            )
            break

    return agreement_index, curr_context[agreement_index:]


# There is an error here if curr_context < prev_context
@torch.inference_mode()
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
            prev_context, curr_context, logger=logger
        )
        num_prefill_toks = len(prefill_toks)
        logger.info(f"Tokens to prefill: {len(prefill_toks)}")

        if num_prefill_toks > PREFILL_CHUNK_SIZE:
            logger.info(
                f"Prefilling {PREFILL_CHUNK_SIZE} tokens from idx={prefill_idx}"
            )

            prefill(
                model,
                idxs=torch.tensor(
                    [prefill_toks[:PREFILL_CHUNK_SIZE]],
                    device="cuda",
                    dtype=torch.int,
                ),
                input_pos=torch.arange(
                    prefill_idx,
                    prefill_idx + PREFILL_CHUNK_SIZE,
                    device="cuda",
                    dtype=torch.int,
                ),
            )
            prev_context = curr_context[: prefill_idx + PREFILL_CHUNK_SIZE]

        elif num_prefill_toks > 0 and full is True:
            logger.info(
                f"Prefilling (force) {num_prefill_toks} tokens from idx={prefill_idx}"
            )
            prefill_toks += (PREFILL_CHUNK_SIZE - len(prefill_toks)) * [
                tokenizer.pad_id
            ]
            prefill(
                model,
                idxs=torch.tensor(
                    [prefill_toks], device="cuda", dtype=torch.int
                ),
                input_pos=torch.arange(
                    prefill_idx,
                    prefill_idx + PREFILL_CHUNK_SIZE,
                    device="cuda",
                    dtype=torch.int,
                ),
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

        if (msg_cnt >= 5 or seen_sentinel) and len(msgs) > 10:
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
    midi_input_port: str,
    control_sentinel: threading.Event,
    received_messages_queue: queue.Queue,
    midi_capture_channel: int,
    results_queue: queue.Queue,
    midi_control_signal: int | None = None,
    midi_through_port: str | None = None,
    first_msg_epoch_time_ms: int | None = None,
):
    logger = get_logger("CAPTURE")
    active_pitches = set()
    first_on_msg_epoch_ms = None
    prev_msg_epoch_time_ms = first_msg_epoch_time_ms  #

    logger.info(f"Listening on MIDI port: '{midi_input_port}'")
    logger.info(f"Using MIDI control signal: {midi_control_signal}")
    if midi_through_port is not None:
        logger.info(f"Sending through on MIDI port: '{midi_through_port}'")

    with ExitStack() as stack:
        midi_input = stack.enter_context(mido.open_input(midi_input_port))
        midi_through = (
            stack.enter_context(mido.open_output(midi_through_port))
            if midi_through_port
            else None
        )

        while not control_sentinel.is_set():
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
    time.sleep(1)
    active_pitches = []
    with mido.open_output(midi_port) as output_port:
        for msg in mido.MidiFile(midi_path).play():
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
    end_sentinel: threading.Event,
):
    logger = get_logger("KEYBOARD")
    while True:
        time.sleep(1)
        _input = input()
        logger.info(f'Keypress seen "{_input}"')
        control_sentinel.set()

        if _input == "e":
            end_sentinel.set()


# TODO: Not tested
def listen_for_midi_control_signal(
    midi_input_port: str,
    control_sentinel: threading.Event,
    end_sentinel: threading.Event,
    midi_control_signal: int | None = None,
    midi_end_signal: int | None = None,
):
    with mido.open_input(midi_input_port) as midi_input:
        while True:
            msg = midi_input.receive(block=False)
            if msg is None:
                time.sleep(0.01)
            elif (
                msg.type == "control_change"
                and msg.control == midi_control_signal
                and msg.value > 0
            ):
                control_sentinel.set()
            elif (
                msg.type == "control_change"
                and msg.control == midi_end_signal
                and msg.value > 0
            ):
                control_sentinel.set()
                end_sentinel.set()


def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("-cp", help="path to model checkpoint")
    argp.add_argument("-midi_in", required=False, help="MIDI input port")
    argp.add_argument("-midi_out", required=True, help="MIDI output port")
    argp.add_argument(
        "-midi_through",
        required=False,
        help="MIDI through port for received input",
    )
    argp.add_argument(
        "-midi_path",
        required=False,
        help="Use MIDI file instead of MIDI input port",
    )
    argp.add_argument(
        "-midi_control_signal",
        type=int,
        help="MIDI control change message for AI takeover",
    )
    argp.add_argument(
        "-midi_end_signal",
        type=int,
        help="MIDI control change message to generate ending",
    )
    argp.add_argument(
        "-temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-min_p",
        help="sampling min_p value",
        type=float,
        required=False,
        default=0.03,
    )
    argp.add_argument(
        "-cfg",
        help="sampling cfg gamma value",
        type=float,
        required=False,
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when sampling",
    )
    argp.add_argument(
        "-save_path",
        type=str,
        required=False,
        help="Path to save complete MIDI file",
    )

    return argp.parse_args()


# TODO: Need functionality for handing case where we run out of model context
# TODO: Make sure channel=9 (drum) case is covered
def main():
    args = parse_args()
    logger = get_logger()
    tokenizer = AbsTokenizer()
    model = load_model(checkpoint_path=args.cp)
    model = compile_model(model=model, max_seq_len=MAX_SEQ_LEN)

    assert (args.midi_path and os.path.isfile(args.midi_path)) or args.midi_in
    if args.midi_path:
        midi_input_port = "Midi Through:Midi Through Port-0"
        play_file_thread = threading.Thread(
            target=play_midi_file,
            args=(midi_input_port, args.midi_path),
            daemon=True,
        )
        play_file_thread.start()
    else:
        midi_input_port = args.midi_in

    control_sentinel = threading.Event()
    end_sentinel = threading.Event()
    keypress_thread = threading.Thread(
        target=listen_for_keypress_control_signal,
        args=[control_sentinel, end_sentinel],
        daemon=True,
    )
    midi_control_thread = threading.Thread(
        target=listen_for_midi_control_signal,
        kwargs={
            "midi_input_port": midi_input_port,
            "control_sentinel": control_sentinel,
            "end_sentinel": end_sentinel,
            "midi_control_signal": args.midi_control_signal,
            "midi_end_signal": args.midi_end_signal,
        },
        daemon=True,
    )
    keypress_thread.start()
    midi_control_thread.start()

    msgs, prev_context, first_on_msg_epoch_ms, num_active_pitches = (
        capture_and_update_kv(
            model=model,
            msgs=[],
            prev_context=[],
            control_sentinel=control_sentinel,
            midi_input_port=midi_input_port,
            midi_control_signal=args.midi_control_signal,
            midi_through_port=args.midi_through,
            midi_capture_channel=0,
        )
    )

    itt = 0
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
            midi_stream_channel=itt,
            is_ending=False,
        )

        control_sentinel.clear()
        if end_sentinel.is_set():
            break
        else:
            itt += 1

        msgs, prev_context, _, num_active_pitches = capture_and_update_kv(
            model=model,
            msgs=msgs,
            prev_context=prev_context,
            control_sentinel=control_sentinel,
            midi_input_port=midi_input_port,
            midi_control_signal=args.midi_control_signal,
            midi_through_port=args.midi_through,
            midi_capture_channel=itt,
            first_msg_epoch_time_ms=first_on_msg_epoch_ms,
        )

    # TODO: There is a bug with the <D> token somewhere?
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
        midi_stream_channel=itt,
        is_ending=True,
    )

    if args.save_path:
        logger.info(f"Saving result to {args.save_path}")
        midi = convert_msgs_to_midi(msgs=msgs)
        midi.save(args.save_path)


if __name__ == "__main__":
    main()
