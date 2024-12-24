"""Tokenizer for MIDI conditioned completions"""

import copy
import random
import functools

from typing import Callable

from aria.config import load_config
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer as _AbsTokenizer


class InferenceAbsTokenizer(_AbsTokenizer):
    def __init__(self):
        super().__init__()

        self.name = "inference_abs"
        self._config = load_config()["tokenizer"]["inference_abs"]

        self.prompt_start_tok = "<PROMPT_START>"
        self.prompt_end_tok = "<PROMPT_END>"
        self.guidance_start_tok = "<GUIDANCE_START>"
        self.guidance_end_tok = "<GUIDANCE_END>"

        self.add_tokens_to_vocab(
            [
                self.prompt_start_tok,
                self.prompt_end_tok,
                self.guidance_start_tok,
                self.guidance_end_tok,
            ]
        )
        self.special_tokens.append(self.prompt_start_tok)
        self.special_tokens.append(self.prompt_end_tok)
        self.special_tokens.append(self.guidance_start_tok)
        self.special_tokens.append(self.guidance_end_tok)

    def _get_guidance_interval_ms(self, guidance_midi_dict: MidiDict):
        first_note_onset_ms = guidance_midi_dict.tick_to_ms(
            guidance_midi_dict.note_msgs[0]["tick"]
        )
        last_note_onset_ms = guidance_midi_dict.tick_to_ms(
            guidance_midi_dict.note_msgs[-1]["tick"]
        )
        guidance_segment_length_ms = random.randint(
            self._config["guidance"]["min_ms"],
            min(self._config["guidance"]["max_ms"], last_note_onset_ms),
        )
        guidance_start_ms = random.randint(
            first_note_onset_ms,
            last_note_onset_ms - guidance_segment_length_ms,
        )
        guidance_end_ms = guidance_start_ms + guidance_segment_length_ms

        return guidance_start_ms, guidance_end_ms

    def _get_guidance_seq(
        self,
        guidance_midi_dict: MidiDict,
        guidance_start_ms: int | None = None,
        guidance_end_ms: int | None = None,
    ):
        assert guidance_midi_dict.note_msgs is not None

        # Need to validate these numbers
        if guidance_start_ms is None:
            assert guidance_end_ms is None
            guidance_start_ms, guidance_end_ms = self._get_guidance_interval_ms(
                guidance_midi_dict=guidance_midi_dict
            )

        slice_note_msgs = []
        for note_msg in guidance_midi_dict.note_msgs:
            start_ms = guidance_midi_dict.tick_to_ms(note_msg["data"]["start"])
            if guidance_start_ms <= start_ms <= guidance_end_ms:
                slice_note_msgs.append(note_msg)

        slice_midi_dict = copy.deepcopy(guidance_midi_dict)
        slice_midi_dict.note_msgs = slice_note_msgs

        if len(slice_midi_dict.note_msgs) == 0:
            # Catches not note in interval
            return []

        guidance_seq = self._tokenize_midi_dict(
            midi_dict=slice_midi_dict,
            remove_preceding_silence=True,
        )

        if self.dim_tok in guidance_seq:
            guidance_seq.remove(self.dim_tok)

        guidance_seq = guidance_seq[
            guidance_seq.index(self.bos_tok)
            + 1 : guidance_seq.index(self.eos_tok)
        ]

        return (
            [self.guidance_start_tok] + guidance_seq + [self.guidance_end_tok]
        )

    def _add_prompt_tokens(
        self, seq: list, prompt_start_ms: int, prompt_end_ms: int
    ):
        res = copy.deepcopy(seq)
        prompt_tok_inserted = False
        time_tok_cnt = 0
        curr_time_ms = 0
        for idx, (tok_1, tok_2) in enumerate(zip(seq, seq[1:])):
            if tok_1 == self.time_tok:
                time_tok_cnt += 1
            elif isinstance(tok_1, tuple) and tok_1[0] in self.instruments_wd:
                assert isinstance(tok_2, tuple) and tok_2[0] == "onset"

                # Adjust time
                curr_time_ms = (self.abs_time_step_ms * time_tok_cnt) + tok_2[1]

                if (
                    curr_time_ms >= prompt_start_ms
                    and prompt_tok_inserted == False
                ):
                    res.insert(idx, self.prompt_start_tok)
                    prompt_tok_inserted = True
                elif (
                    curr_time_ms > prompt_end_ms and prompt_tok_inserted == True
                ):
                    res.insert(idx + 1, self.prompt_end_tok)
                    break

        return res

    def tokenize(
        self,
        midi_dict: MidiDict,
        prompt_intervals_ms: list[tuple[int, int]],
        guidance_midi_dict: MidiDict | None = None,
        guidance_start_ms: int | None = None,
        guidance_end_ms: int | None = None,
    ):
        seq = self._tokenize_midi_dict(
            midi_dict=midi_dict, remove_preceding_silence=True
        )
        first_note_ms = midi_dict.tick_to_ms(
            midi_dict.note_msgs[0]["data"]["start"]
        )

        for prompt_start_ms, prompt_end_ms in prompt_intervals_ms:
            if prompt_end_ms > first_note_ms:
                seq = self._add_prompt_tokens(
                    seq,
                    prompt_start_ms=prompt_start_ms - first_note_ms,
                    prompt_end_ms=prompt_end_ms - first_note_ms,
                )

        if guidance_midi_dict is not None:
            guidance_seq = self._get_guidance_seq(
                guidance_midi_dict=guidance_midi_dict,
                guidance_start_ms=guidance_start_ms,
                guidance_end_ms=guidance_end_ms,
            )
        else:
            guidance_seq = []

        return guidance_seq + seq

    def detokenize(self, tokenized_seq: list, **kwargs):
        if self.guidance_end_tok in tokenized_seq:
            seq = tokenized_seq[tokenized_seq.index(self.guidance_end_tok) :]
        else:
            seq = tokenized_seq

        return super()._detokenize_midi_dict(seq, **kwargs)

    def export_data_aug(self):
        return [
            self.export_guidance_tempo_aug(max_tempo_aug=0.2, mixup=True),
            self.export_guidance_pitch_aug(3),
            self.export_guidance_velocity_aug(2),
        ]

    def export_guidance_aug_fn(self, aug_fn):
        """Transforms augmentation function to only apply to guidance seq"""

        def _guidance_seq_aug_fn(
            src: list,
            _aug_fn: Callable,
            pad_tok: str,
            **kwargs,
        ) -> list:

            initial_seq_len = len(src)
            if self.guidance_start_tok in src and self.guidance_end_tok in src:
                guidance_seq = src[
                    src.index(self.guidance_start_tok)
                    + 1 : src.index(self.guidance_end_tok)
                ]
                seq = src[src.index(self.guidance_end_tok) + 1 :]

                if len(guidance_seq) == 0:
                    return src
            else:
                return src

            augmented_guidance_seq = _aug_fn(guidance_seq)
            res = (
                [self.guidance_start_tok]
                + augmented_guidance_seq
                + [self.guidance_end_tok]
                + seq
            )

            # Pad or truncate to original sequence length as necessary
            res = res[:initial_seq_len]
            res += [pad_tok] * (initial_seq_len - len(res))

            return res

        return functools.partial(
            _guidance_seq_aug_fn,
            _aug_fn=aug_fn,
            pad_tok=self.pad_tok,
        )

    def export_guidance_pitch_aug(self, max_pitch_aug: int):
        """Apply pitch augmentation to the guidance sequence"""

        return self.export_guidance_aug_fn(
            self.export_pitch_aug(max_pitch_aug=max_pitch_aug)
        )

    def export_guidance_velocity_aug(self, max_num_aug_steps: int):
        """Apply velocity augmentation to the guidance sequence"""

        return self.export_guidance_aug_fn(
            self.export_velocity_aug(max_num_aug_steps=max_num_aug_steps)
        )

    def export_guidance_tempo_aug(self, max_tempo_aug: int, mixup: bool):
        """Apply tempo augmentation to the guidance sequence"""

        return self.export_guidance_aug_fn(
            self.export_tempo_aug(max_tempo_aug=max_tempo_aug, mixup=mixup)
        )

    def split(self, seq: list, seq_len: int):
        def _process_chunk(_chunk: list):
            # Ensure first token is note token
            while True:
                if _chunk[0] == self.bos_tok:
                    break
                elif (
                    isinstance(_chunk[0], tuple)
                    and _chunk[0][0] in self.instruments_wd
                ):
                    break
                else:
                    _chunk.pop(0)

            # Insert prompt_start_tok if it is missing (but required)
            for idx in range(len(_chunk)):
                tok = _chunk[idx]

                if tok == self.prompt_start_tok:
                    break
                elif tok == self.prompt_end_tok:
                    if _chunk[0] == self.bos_tok:
                        _chunk.insert(1, self.prompt_start_tok)
                    else:
                        _chunk.insert(0, self.prompt_start_tok)
                    break

            return _chunk

        guidance = []
        if self.guidance_start_tok in seq:
            guidance_start = seq.index(self.guidance_start_tok)
            guidance_end = seq.index(self.guidance_end_tok)
            guidance = seq[guidance_start : guidance_end + 1]
            seq = seq[guidance_end + 1 :]

        prefix = []
        while seq:
            tok = seq[0]
            if tok != self.bos_tok and tok[0] == "prefix":
                prefix.append(seq.pop(0))
            else:
                break

        chunks = [
            _process_chunk(seq[idx : idx + seq_len])
            for idx in range(0, len(seq) - 100, seq_len)
        ]

        res = []
        for chunk in chunks:
            sub_seq = guidance + prefix + chunk
            sub_seq = sub_seq[:seq_len]
            sub_seq += [self.pad_tok] * (seq_len - len(sub_seq))

            res.append(sub_seq)

        return res
