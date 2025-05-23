#!/usr/bin/env python3

import argparse
import os
import time
import uuid
import copy
import logging
import threading
import queue
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
from aria.sample import prefill, decode_one, sample_min_p

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_SEQ_LEN = 8192
PREFILL_COMPILE_SEQ_LEN = 1024

# HARDWARE: Decoded logits are masked for durations < MIN_NOTE_LEN_MS
# HARDWARE: Sends early off-msg if pitch is on MIN_NOTE_DELTA_MS before on-msg
# HARDWARE: All messages are sent HARDWARE_LATENCY_MS early
MIN_NOTE_DELTA_MS = 50
MIN_NOTE_LEN_MS = 50
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


def compiled_prefill(
    model: TransformerLM,
    enc_seq: torch.Tensor,
):
    return prefill(
        model=model,
        idxs=enc_seq[:, :PREFILL_COMPILE_SEQ_LEN],
        input_pos=torch.arange(0, PREFILL_COMPILE_SEQ_LEN, device="cuda"),
    )


def _compile_prefill(
    model: TransformerLM,
    logger: logging.Logger,
):
    global compiled_prefill
    compiled_prefill = torch.compile(
        compiled_prefill,
        mode="reduce-overhead",
        fullgraph=True,
    )

    start_compile_time_s = time.time()
    logger.info(f"Compiling prefill")
    compiled_prefill(
        model, enc_seq=torch.ones(1, 8192, device="cuda", dtype=torch.int)
    )
    logger.info(
        f"Finished compiling - took {time.time() - start_compile_time_s:.4f} seconds"
    )

    for _ in range(5):
        compiled_prefill(
            model,
            enc_seq=torch.ones(1, 8192, device="cuda", dtype=torch.int),
        )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    compiled_prefill(
        model, enc_seq=torch.ones(1, 8192, device="cuda", dtype=torch.int)
    )
    end_event.record()
    end_event.synchronize()
    compiled_prefill_ms = start_event.elapsed_time(end_event)
    logger.info(f"Compiled prefill benchmark: {compiled_prefill_ms:.2f}ms")

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
        logger.info(f"Compiling forward pass")
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
            f"Compiled forward pass benchmark: {compiled_forward_ms:.2f} ms/it ({compiled_forward_its:.2f} it/s)"
        )

        return model


@torch.autocast("cuda", dtype=DTYPE)
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
    model = _compile_prefill(model=model, logger=logger)

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


@torch.autocast("cuda", dtype=DTYPE)
@torch.inference_mode()
def recalculate_dur_tokens(
    model: TransformerLM,
    priming_seq: list,
    enc_seq: torch.Tensor,
    tokenizer: AbsTokenizer,
    start_idx: int,
):
    logger = get_logger("GENERATE")
    assert start_idx > 0

    priming_seq_len = len(priming_seq)
    num_time_toks_seen = priming_seq[:start_idx].count(tokenizer.time_tok)
    curr_onset = num_time_toks_seen * 5000
    last_offset = tokenizer.calc_length_ms(priming_seq)
    LAST_OFFSET_BUFFER_MS = 50

    for idx in range(start_idx, priming_seq_len):
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
                f"Sampled logits for position {idx} by inserting {prev_tok} at position {idx-1}"
            )

        next_token_ids = torch.argmax(logits, dim=-1).flatten()
        priming_tok = tokenizer.id_to_tok[enc_seq[0, idx].item()]
        predicted_tok = tokenizer.id_to_tok[next_token_ids[0].item()]

        logger.debug(
            f"Ground truth token: {priming_tok}, resampled token: {predicted_tok}"
        )

        resample = False
        if isinstance(priming_tok, tuple) and priming_tok[0] == "onset":
            curr_onset = (num_time_toks_seen * 5000) + priming_tok[1]
        elif priming_tok == tokenizer.time_tok:
            num_time_toks_seen += 1
            curr_onset = num_time_toks_seen * 5000
        elif isinstance(priming_tok, tuple) and priming_tok[0] == "dur":
            assert (
                isinstance(predicted_tok, tuple) and predicted_tok[0] == "dur"
            )

            priming_dur = priming_tok[1]
            predicted_dur = predicted_tok[1]

            if (predicted_dur > priming_dur) and (
                curr_onset + priming_dur > last_offset - LAST_OFFSET_BUFFER_MS
            ):
                resample = True

        if resample is True:
            logger.info(
                f"Replaced ground truth for position {idx}: {tokenizer.id_to_tok[enc_seq[:, idx].item()]} -> {tokenizer.id_to_tok[next_token_ids[0].item()]}"
            )
            enc_seq[:, idx] = next_token_ids
            priming_seq[idx] = predicted_tok

    # TODO: There is a bug here if no-notes are active when the signal is pressed
    last_tok_id = enc_seq[0, idx]
    last_tok = tokenizer.id_to_tok[last_tok_id.item()]

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        next_token_logits = decode_one(
            model,
            idxs=torch.tensor([[last_tok_id]], device="cuda", dtype=torch.int),
            input_pos=torch.tensor([idx], device="cuda", dtype=torch.int),
        )

    logger.info(f"Updated KV-Cache by inserting {last_tok} at position {idx}")

    return enc_seq, priming_seq, next_token_logits


@torch.autocast("cuda", dtype=DTYPE)
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

    BEAM_WIDTH = 5
    BUFFER_MS = 100 + HARDWARE_LATENCY_MS
    TIME_TOK_ID = tokenizer.tok_to_id[tokenizer.time_tok]
    TIME_TOK_WEIGHTING = -6

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
        # logits[:, tokenizer.tok_to_id[tokenizer.eos_tok]] = float("-inf")
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


@torch.autocast("cuda", dtype=DTYPE)
@torch.inference_mode()
def generate_tokens(
    priming_seq: list,
    tokenizer: AbsTokenizer,
    model: TransformerLM,
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
    logger.info(f"Prefilling up to (and including) position: {start_idx-2}")

    # In theory we could reuse the logits from prefill
    prefill_start_s = time.time()
    if start_idx < PREFILL_COMPILE_SEQ_LEN:
        logger.info(
            f"Using compiled prefill for sequence length: {PREFILL_COMPILE_SEQ_LEN}"
        )
        compiled_prefill(
            model=model,
            enc_seq=enc_seq,
        )
    else:
        prefill(
            model,
            idxs=enc_seq[:, : start_idx - 1],
            input_pos=torch.arange(0, start_idx - 1, device="cuda"),
        )

    torch.cuda.synchronize()
    logger.info(
        f"Prefill took {(time.time() - prefill_start_s) * 1000:.2f} milliseconds"
    )
    logger.info(f"Starting duration recalculation from: {start_idx}")

    recalculate_dur_start_s = time.time()
    enc_seq, priming_seq, next_token_logits = recalculate_dur_tokens(
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

        midi_messages_queue.put(on_msg)
        midi_messages_queue.put(off_msg)
        logger.debug(f"Put message: {on_msg}")
        logger.debug(f"Put message: {off_msg}")
        logger.debug(f"Ahead by {onset_epoch_ms - get_epoch_time_ms()}ms")

        note_buffer = []


def stream_midi(
    midi_messages_queue: queue.Queue,
    msgs: list[mido.Message],
    prev_msg_epoch_time_ms: float,
    midi_output_port: str,
    control_sentinel: threading.Event,
    midi_stream_channel: int,
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

            if control_sentinel.is_set():
                break

            midi_messages = sorted(
                midi_messages,
                key=lambda msg: (
                    msg["epoch_time_ms"],
                    msg["vel"],
                ),
            )

            while midi_messages:
                latency_adjusted_epoch_time_ms = (
                    get_epoch_time_ms() + HARDWARE_LATENCY_MS
                )
                msg = midi_messages[0]

                if (
                    (msg["vel"] > 0)
                    and (
                        msg["epoch_time_ms"] - latency_adjusted_epoch_time_ms
                        <= MIN_NOTE_DELTA_MS
                    )
                    and pitch_active.get(msg["pitch"], False)
                ):
                    midi_out.send(
                        mido.Message(
                            "note_on",
                            note=msg["pitch"],
                            velocity=0,
                            channel=0,
                            time=0,
                        )
                    )
                    pitch_active[msg["pitch"]] = False
                    logger.info(f"Sent early off for {msg}")

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

                        logger.info(
                            f"[{get_epoch_time_ms() - msg["epoch_time_ms"]}ms] Sent message: {msg}"
                        )
                    else:
                        logger.info(
                            f"Skipping note_off message due to uuid mismatch: {msg}"
                        )
                    midi_messages.pop(0)

                elif (
                    latency_adjusted_epoch_time_ms - msg["epoch_time_ms"] > 100
                ):
                    # Message occurs too far in the past
                    logger.info(
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

        while remaining_note_off_messages:
            msg = remaining_note_off_messages.pop(0)
            mido_msg = mido.Message(
                "note_on",
                note=msg["pitch"],
                velocity=0,
                channel=midi_stream_channel,
                time=msg["epoch_time_ms"] - prev_msg_epoch_time_ms,
            )
            prev_msg_epoch_time_ms = msg["epoch_time_ms"]
            midi_out.send(mido_msg)
            logger.info(f"Sent message: {msg}")
            msgs.append(mido_msg)

    return msgs


def stream_msgs(
    model: TransformerLM,
    tokenizer: AbsTokenizer,
    msgs: list[mido.Message],
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
    priming_seq = tokenizer.tokenize(midi_dict=midi_dict)
    priming_seq = priming_seq[: priming_seq.index(tokenizer.eos_tok)]

    if tokenizer.dim_tok in priming_seq:
        priming_seq.remove(tokenizer.dim_tok)
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
            "control_sentinel": control_sentinel,
            "generated_tokens_queue": generated_tokens_queue,
            "temperature": temperature,
            "min_p": min_p,
            "num_preceding_active_pitches": num_preceding_active_pitches,
            "first_on_msg_epoch_ms": first_on_msg_epoch_ms,
        },
        daemon=True,
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
        daemon=True,
    )
    decode_tokens_to_midi_thread.start()

    msgs = stream_midi(
        midi_messages_queue=midi_messages_queue,
        msgs=msgs,
        prev_msg_epoch_time_ms=first_on_msg_epoch_ms
        + tokenizer.calc_length_ms(priming_seq, onset=False),
        midi_output_port=midi_output_port,
        control_sentinel=control_sentinel,
        midi_stream_channel=midi_stream_channel,
    )

    generate_tokens_thread.join()
    decode_tokens_to_midi_thread.join()

    return msgs


def convert_msgs_to_midi(msgs: list[mido.Message]):
    logger = get_logger("convert")

    channel_to_track = {
        chan: mido.MidiTrack()
        for chan in list(set([msg.channel for msg in msgs]))
    }

    for msg in msgs:
        channel_to_track[msg.channel].append(msg)

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


def capture_midi_input(
    midi_input_port: str,
    control_sentinel: threading.Event,
    midi_capture_channel: int,
    midi_control_signal: int | None = None,
    midi_through_port: str | None = None,
    first_msg_epoch_time_ms: int | None = None,
):
    logger = get_logger("CAPTURE")
    received_messages = []
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
                received_messages.append(msg)
                if midi_through is not None:
                    midi_through.send(msg)
            elif msg.type == "note_on" and msg.velocity > 0:
                if first_on_msg_epoch_ms is None:
                    first_on_msg_epoch_ms = get_epoch_time_ms()

                active_pitches.add(msg.note)
                received_messages.append(msg)
                if midi_through is not None:
                    midi_through.send(msg)
            elif msg.type == "control_change" and msg.control == 64:
                received_messages.append(msg)
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
            received_messages.append(msg)
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
            received_messages.append(msg)
            if midi_through is not None:
                midi_through.send(msg)

        msg = mido.Message(
            type="control_change",
            control=64,
            value=0,
            channel=midi_capture_channel,
            time=0,
        )
        received_messages.append(msg)
        if midi_through is not None:
            midi_through.send(msg)

        # TODO: Need to figure out what the hell this does - is it needed?
        # Workaround for the way that file-playback is implemented - delete
        msg = mido.Message(
            type="control_change",
            control=midi_control_signal,
            value=0,
            channel=midi_capture_channel,
            time=0,
        )
        if midi_through is not None:
            midi_through.send(msg)

        return received_messages, first_on_msg_epoch_ms, num_active_pitches


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


# TODO: Test demo on real instrument with real MIDI interface
# TODO: Possibly a problem with endings being truncated
# TODO: Need functionality for handing case where we run out of model context


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

    msgs = []
    captured_msgs, first_on_msg_epoch_ms, num_active_pitches = (
        capture_midi_input(
            midi_input_port=midi_input_port,
            control_sentinel=control_sentinel,
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
            msgs=msgs + captured_msgs,
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

        captured_msgs, _, num_active_pitches = capture_midi_input(
            midi_input_port=midi_input_port,
            control_sentinel=control_sentinel,
            midi_control_signal=args.midi_control_signal,
            midi_through_port=args.midi_through,
            midi_capture_channel=itt,
            first_msg_epoch_time_ms=first_on_msg_epoch_ms,
        )

    msgs = stream_msgs(
        model=model,
        tokenizer=tokenizer,
        msgs=msgs,
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
