"""Includes Tokenizers and pre-processing utilities."""

import logging
import torch
import functools
import itertools
import random
import copy

from collections import defaultdict
from typing import Callable

from aria.data.midi import MidiDict
from aria.config import load_config
from aria.data.midi import get_duration_ms

# TODO:
# - Add a warning when tokenizing degenerate MIDI files. e.g. if we have two
#   pianos with overlapping notes, this could potentially cause an issue for
#   some tokenizers. I'm not sure if this will cause an issue AbsTokenizer
#   however it might with some others such as the AmtTokenizer.


class Tokenizer:
    """Abstract Tokenizer class for tokenizing midi_dict objects.

    Args:
        return_tensors (bool, optional): If True, encode will return tensors.
            Defaults to False.
    """

    def __init__(
        self,
        return_tensors: bool = False,
    ):
        self.name = None
        self.return_tensors = return_tensors

        self.bos_tok = "<S>"
        self.eos_tok = "<E>"
        self.pad_tok = "<P>"
        self.unk_tok = "<U>"
        self.dim_tok = "<D>"

        self.special_tokens = [
            self.bos_tok,
            self.eos_tok,
            self.pad_tok,
            self.unk_tok,
            self.dim_tok,
        ]

        # These must be implemented in child class (abstract params)
        self.vocab = ()
        self.instruments_wd = []
        self.instruments_nd = []
        self.config = {}
        self.tok_to_id = {}
        self.id_to_tok = {}
        self.vocab_size = -1
        self.pad_id = -1

    def _tokenize_midi_dict(self, midi_dict: MidiDict):
        """Abstract method for tokenizing a MidiDict object into a sequence of
        tokens."""
        raise NotImplementedError

    def tokenize(self, midi_dict: MidiDict, **kwargs):
        """Tokenizes a MidiDict object.

        This function should be overridden if additional transformations are
        required. For instance, in fine-tuning tokenizer you may want to insert
        additional tokens. The default behavior is to call tokenize_midi_dict.
        """
        return self._tokenize_midi_dict(midi_dict, **kwargs)

    def _detokenize_midi_dict(self, tokenized_seq: list):
        """Abstract method for de-tokenizing a sequence of tokens into a
        MidiDict Object."""
        raise NotImplementedError

    def detokenize(self, tokenized_seq: list, **kwargs):
        """Detokenizes a MidiDict object.

        This function should be overridden if additional are required during
        detokenization. The default behavior is to call detokenize_midi_dict.
        """
        return self._detokenize_midi_dict(tokenized_seq, **kwargs)

    def export_data_aug(cls):
        """Abstract method for exporting a list of all data augmentation
        functions.

        This function is used when setting data transformation functions in
        TrainingDataset, e.g.

        PretrainingDataset.set_transform(Tokenizer.export_data_aug())
        """
        raise NotImplementedError

    def encode(self, unencoded_seq: list):
        """Converts tokenized sequence into a list/torch.Tensor of ids."""

        def _enc_fn(tok):
            return self.tok_to_id.get(tok, self.tok_to_id[self.unk_tok])

        if self.tok_to_id is None:
            raise NotImplementedError("tok_to_id")

        if self.return_tensors is True:
            encoded_seq = torch.tensor([_enc_fn(tok) for tok in unencoded_seq])
        else:
            encoded_seq = [_enc_fn(tok) for tok in unencoded_seq]

        return encoded_seq

    def decode(self, encoded_seq: list | torch.Tensor):
        """Converts sequence of ids into the corresponding list of tokens."""

        def _dec_fn(id):
            return self.id_to_tok.get(id, self.unk_tok)

        if self.id_to_tok is None:
            raise NotImplementedError("id_to_tok")

        if isinstance(encoded_seq, torch.Tensor):
            decoded_seq = [_dec_fn(idx) for idx in encoded_seq.tolist()]
        else:
            decoded_seq = [_dec_fn(idx) for idx in encoded_seq]

        return decoded_seq

    @classmethod
    def _find_closest_int(cls, n: int, sorted_list: list):
        # Selects closest integer to n from sorted_list
        # Time ~ Log(n)

        left, right = 0, len(sorted_list) - 1
        closest = float("inf")

        while left <= right:
            mid = (left + right) // 2
            diff = abs(sorted_list[mid] - n)

            if diff < abs(closest - n):
                closest = sorted_list[mid]

            if sorted_list[mid] < n:
                left = mid + 1
            else:
                right = mid - 1

        return closest

    def _build_pedal_intervals(self, midi_dict: MidiDict):
        """Returns pedal-on intervals for each channel."""
        channel_to_pedal_intervals = defaultdict(list)
        pedal_status = {}

        for pedal_msg in midi_dict.pedal_msgs:
            tick = pedal_msg["tick"]
            channel = pedal_msg["channel"]
            data = pedal_msg["data"]

            if data == 1 and pedal_status.get(channel, None) is None:
                pedal_status[channel] = tick
            elif data == 0 and pedal_status.get(channel, None) is not None:
                # Close pedal interval
                _start_tick = pedal_status[channel]
                _end_tick = tick
                channel_to_pedal_intervals[channel].append(
                    [_start_tick, _end_tick]
                )
                del pedal_status[channel]

        # Close all unclosed pedals at end of track
        final_tick = midi_dict.note_msgs[-1]["data"]["end"]
        for channel, start_tick in pedal_status.items():
            channel_to_pedal_intervals[channel].append([start_tick, final_tick])

        return channel_to_pedal_intervals

    def add_tokens_to_vocab(self, tokens: list | tuple):
        for token in tokens:
            assert token not in self.vocab

        self.vocab = self.vocab + tuple(tokens)
        self.tok_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_tok = {v: k for k, v in self.tok_to_id.items()}
        self.vocab_size = len(self.vocab)

    def export_aug_fn_concat(self, aug_fn: Callable):
        """Exports a function that splits src before augmenting.

        This is useful for augmentation functions that expect pure sequences
        instead of concatenated ones (like those given by PretrainedDataset).
        """

        def _aug_fn_concat(
            src: list,
            _aug_fn: Callable,
            pad_tok: str,
            eos_tok: str,
            **kwargs,
        ):
            # Split list on '<E>'
            initial_seq_len = len(src)
            src_sep = []
            prev_idx = 0
            for curr_idx, tok in enumerate(src, start=1):
                if tok == eos_tok:
                    src_sep.append(src[prev_idx:curr_idx])
                    prev_idx = curr_idx

            # Last sequence
            if prev_idx != curr_idx:
                src_sep.append(src[prev_idx:])

            # Augment
            src_sep = [
                _aug_fn(
                    _src,
                    **kwargs,
                )
                for _src in src_sep
            ]

            # Concatenate
            src_aug_concat = [tok for src_aug in src_sep for tok in src_aug]

            # Pad or truncate to original sequence length as necessary
            src_aug_concat = src_aug_concat[:initial_seq_len]
            src_aug_concat += [pad_tok] * (
                initial_seq_len - len(src_aug_concat)
            )

            return src_aug_concat

        return functools.partial(
            _aug_fn_concat,
            _aug_fn=aug_fn,
            pad_tok=self.pad_tok,
            eos_tok=self.eos_tok,
        )


class AbsTokenizer(Tokenizer):
    """MidiDict tokenizer implemented with absolute onset timings"""

    def __init__(self, return_tensors: bool = False):
        super().__init__(return_tensors)
        self.config = load_config()["tokenizer"]["abs"]
        self.name = "abs"

        # Calculate time quantizations (in ms)
        self.abs_time_step = self.config["abs_time_step_ms"]
        self.max_dur = self.config["max_dur_ms"]
        self.time_step = self.config["time_step_ms"]

        self.dur_time_quantizations = [
            self.time_step * i
            for i in range((self.max_dur // self.time_step) + 1)
        ]
        self.onset_time_quantizations = [
            self.time_step * i for i in range((self.max_dur // self.time_step))
        ]

        # Calculate velocity quantizations
        self.velocity_step = self.config["velocity_quantization"]["step"]
        self.velocity_quantizations = [
            i * self.velocity_step
            for i in range(int(127 / self.velocity_step) + 1)
        ]
        self.max_velocity = self.velocity_quantizations[-1]

        # _nd = no drum; _wd = with drum
        self.instruments_nd = [
            k
            for k, v in self.config["ignore_instruments"].items()
            if v is False
        ]
        self.instruments_wd = self.instruments_nd + ["drum"]

        # Prefix tokens
        self.prefix_tokens = [
            ("prefix", "instrument", x) for x in self.instruments_wd
        ]
        self.composer_names = self.config["composer_names"]
        self.form_names = self.config["form_names"]
        self.genre_names = self.config["genre_names"]
        self.prefix_tokens += [
            ("prefix", "composer", x) for x in self.composer_names
        ]
        self.prefix_tokens += [("prefix", "form", x) for x in self.form_names]
        self.prefix_tokens += [("prefix", "genre", x) for x in self.genre_names]

        # Build vocab
        self.time_tok = "<T>"
        self.onset_tokens = [
            ("onset", i) for i in self.onset_time_quantizations
        ]
        self.dur_tokens = [("dur", i) for i in self.dur_time_quantizations]
        self.drum_tokens = [("drum", i) for i in range(35, 82)]

        self.note_tokens = list(
            itertools.product(
                self.instruments_nd,
                [i for i in range(128)],
                self.velocity_quantizations,
            )
        )

        self.special_tokens.append(self.time_tok)
        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prefix_tokens
            + self.note_tokens
            + self.drum_tokens
            + self.dur_tokens
            + self.onset_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

    def export_data_aug(self):
        return [
            self.export_tempo_aug(tempo_aug_range=0.2, mixup=True),
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]

    def _quantize_dur(self, time: int):
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.dur_time_quantizations)

    def _quantize_onset(self, time: int):
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.onset_time_quantizations)

    def _quantize_velocity(self, velocity: int):
        # This function will return values in the range 0 < res =< 127
        velocity_quantized = self._find_closest_int(
            velocity, self.velocity_quantizations
        )

        if velocity_quantized == 0 and velocity != 0:
            return self.velocity_step
        else:
            return velocity_quantized

    def _format(self, prefix: list, unformatted_seq: list):
        # If unformatted_seq is longer than 150 tokens insert diminish tok
        idx = -100 + random.randint(-10, 10)
        if len(unformatted_seq) > 150:
            if (
                unformatted_seq[idx][0] == "onset"
            ):  # Don't want: note, <D>, onset, due
                unformatted_seq.insert(idx - 1, self.dim_tok)
            elif (
                unformatted_seq[idx][0] == "dur"
            ):  # Don't want: note, onset, <D>, dur
                unformatted_seq.insert(idx - 2, self.dim_tok)
            else:
                unformatted_seq.insert(idx, self.dim_tok)

        res = prefix + [self.bos_tok] + unformatted_seq + [self.eos_tok]

        return res

    def calc_length_ms(self, seq: list, onset: bool = False):
        """Calculates time (ms) end of sequence to the end of the last note. If
        onset=True, then it will return the onset time of the last note instead
        """
        assert type(seq) == list, "Must provide list of decoded toks"
        assert type(seq[0]) != int, "Must provide list of decoded toks"

        # Find the index of the last onset or dur token
        seq = copy.deepcopy(seq)
        for _idx in range(len(seq) - 1, -1, -1):
            tok = seq[_idx]
            if type(tok) is tuple and tok[0] in {"onset", "dur"}:
                break
            else:
                seq.pop()

        time_offset_ms = seq.count(self.time_tok) * self.abs_time_step
        idx = len(seq) - 1
        for tok in seq[::-1]:
            if type(tok) is tuple and tok[0] == "dur":
                assert seq[idx][0] == "dur", "Error with function"
                assert seq[idx - 1][0] == "onset", "Error with function"

                if onset is False:
                    return time_offset_ms + seq[idx - 1][1] + seq[idx][1]
                elif onset is True:
                    return time_offset_ms + seq[idx - 1][1]  # Ignore dur

            idx -= 1

        # If it gets to this point, an error has occurred
        raise Exception

    def truncate_by_time(self, tokenized_seq: list, trunc_time_ms: int):
        """This function truncates notes with onset_ms > trunc_tim_ms."""
        time_offset_ms = 0
        for idx, tok in enumerate(tokenized_seq):
            if tok == self.time_tok:
                time_offset_ms += self.abs_time_step
            elif type(tok) is tuple and tok[0] == "onset":
                if time_offset_ms + tok[1] > trunc_time_ms:
                    return tokenized_seq[: idx - 1]

        return tokenized_seq

    def _tokenize_midi_dict(
        self, midi_dict: MidiDict, remove_preceding_silence: bool = True
    ):
        ticks_per_beat = midi_dict.ticks_per_beat
        midi_dict.remove_instruments(self.config["ignore_instruments"])

        if len(midi_dict.note_msgs) == 0:
            raise Exception("note_msgs is empty after ignoring instruments")

        channel_to_pedal_intervals = self._build_pedal_intervals(midi_dict)

        channels_used = {msg["channel"] for msg in midi_dict.note_msgs}

        channel_to_instrument = {
            msg["channel"]: midi_dict.program_to_instrument[msg["data"]]
            for msg in midi_dict.instrument_msgs
            if msg["channel"] != 9  # Exclude drums
        }
        # If non-drum channel is missing from instrument_msgs, default to piano
        for c in channels_used:
            if channel_to_instrument.get(c) is None and c != 9:
                channel_to_instrument[c] = "piano"

        # Calculate prefix
        prefix = [
            ("prefix", "instrument", x)
            for x in set(channel_to_instrument.values())
        ]
        if 9 in channels_used:
            prefix.append(("prefix", "instrument", "drum"))
        composer = midi_dict.metadata.get("composer")
        if composer and (composer in self.composer_names):
            prefix.insert(0, ("prefix", "composer", composer))
        form = midi_dict.metadata.get("form")
        if form and (form in self.form_names):
            prefix.insert(0, ("prefix", "form", form))
        genre = midi_dict.metadata.get("genre")
        if genre and (genre in self.genre_names):
            prefix.insert(0, ("prefix", "genre", genre))
        random.shuffle(prefix)

        tokenized_seq = []

        if remove_preceding_silence is False:
            initial_onset_tick = 0
        else:
            initial_onset_tick = midi_dict.note_msgs[0]["data"]["start"]

        curr_time_since_onset = 0
        for _, msg in enumerate(midi_dict.note_msgs):
            # Extract msg data
            _channel = msg["channel"]
            _pitch = msg["data"]["pitch"]
            _velocity = msg["data"]["velocity"]
            _start_tick = msg["data"]["start"]
            _end_tick = msg["data"]["end"]

            # Calculate time data
            prev_time_since_onset = curr_time_since_onset
            curr_time_since_onset = get_duration_ms(
                start_tick=initial_onset_tick,
                end_tick=_start_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=ticks_per_beat,
            )

            # Add abs time token if necessary
            time_toks_to_append = (
                curr_time_since_onset // self.abs_time_step
            ) - (prev_time_since_onset // self.abs_time_step)
            if time_toks_to_append > 0:
                for _ in range(time_toks_to_append):
                    tokenized_seq.append(self.time_tok)

            # Special case instrument is a drum. This occurs exclusively when
            # MIDI channel is 9 when 0 indexing
            if _channel == 9:
                _note_onset = self._quantize_onset(
                    curr_time_since_onset % self.abs_time_step
                )
                tokenized_seq.append(("drum", _pitch))
                tokenized_seq.append(("onset", _note_onset))

            else:  # Non drum case (i.e. an instrument note)
                _instrument = channel_to_instrument[_channel]

                # Update _end_tick if affected by pedal
                for pedal_interval in channel_to_pedal_intervals[_channel]:
                    pedal_start, pedal_end = (
                        pedal_interval[0],
                        pedal_interval[1],
                    )
                    if pedal_start < _end_tick < pedal_end:
                        _end_tick = pedal_end
                        break

                _note_duration = get_duration_ms(
                    start_tick=_start_tick,
                    end_tick=_end_tick,
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                # Quantize
                _velocity = self._quantize_velocity(_velocity)
                _note_onset = self._quantize_onset(
                    curr_time_since_onset % self.abs_time_step
                )
                _note_duration = self._quantize_dur(_note_duration)
                if _note_duration == 0:
                    _note_duration = self.time_step

                tokenized_seq.append((_instrument, _pitch, _velocity))
                tokenized_seq.append(("onset", _note_onset))
                tokenized_seq.append(("dur", _note_duration))

        return self._format(
            prefix=prefix,
            unformatted_seq=tokenized_seq,
        )

    def _detokenize_midi_dict(self, tokenized_seq: list):
        instrument_programs = self.config["instrument_programs"]
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        TICKS_PER_BEAT = 500
        TEMPO = 500000

        # Set message tempos
        tempo_msgs = [{"type": "tempo", "data": TEMPO, "tick": 0}]
        meta_msgs = []
        pedal_msgs = []
        instrument_msgs = []

        instrument_to_channel = {}

        # Add non-drum instrument_msgs, breaks at first note token
        channel_idx = 0
        curr_tick = 0
        for idx, tok in enumerate(tokenized_seq):
            if channel_idx == 9:  # Skip channel reserved for drums
                channel_idx += 1

            if tok in self.special_tokens:
                if tok == self.time_tok:
                    curr_tick += self.abs_time_step
                continue
            elif (
                tok[0] == "prefix"
                and tok[1] == "instrument"
                and tok[2] in self.instruments_wd
            ):
                # Process instrument prefix tokens
                if tok[2] in instrument_to_channel.keys():
                    logging.warning(f"Duplicate prefix {tok[2]}")
                    continue
                elif tok[2] == "drum":
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": 0,
                            "tick": 0,
                            "channel": 9,
                        }
                    )
                    instrument_to_channel["drum"] = 9
                else:
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": instrument_programs[tok[2]],
                            "tick": 0,
                            "channel": channel_idx,
                        }
                    )
                    instrument_to_channel[tok[2]] = channel_idx
                    channel_idx += 1
            elif tok[0] == "prefix":
                # Skip all other prefix tokens
                continue
            else:
                # Note, wait, or duration token
                start = idx
                break

        # Note messages
        note_msgs = []
        for tok_1, tok_2, tok_3 in zip(
            tokenized_seq[start:],
            tokenized_seq[start + 1 :],
            tokenized_seq[start + 2 :],
        ):
            if tok_1 in self.special_tokens:
                _tok_type_1 = "special"
            else:
                _tok_type_1 = tok_1[0]
            if tok_2 in self.special_tokens:
                _tok_type_2 = "special"
            else:
                _tok_type_2 = tok_2[0]
            if tok_3 in self.special_tokens:
                _tok_type_3 = "special"
            else:
                _tok_type_3 = tok_3[0]

            if tok_1 == self.time_tok:
                curr_tick += self.abs_time_step

            elif (
                _tok_type_1 == "special"
                or _tok_type_1 == "prefix"
                or _tok_type_1 == "onset"
                or _tok_type_1 == "dur"
            ):
                continue
            elif _tok_type_1 == "drum" and _tok_type_2 == "onset":
                _start_tick = curr_tick + tok_2[1]
                _end_tick = _start_tick + self.time_step
                _pitch = tok_1[1]
                _channel = instrument_to_channel.get(tok_1[0], None)
                _velocity = self.config["drum_velocity"]

                if _channel is None:
                    logging.warning(
                        "Tried to decode note message for unexpected instrument"
                    )
                else:
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": _pitch,
                                "start": _start_tick,
                                "end": _end_tick,
                                "velocity": _velocity,
                            },
                            "tick": _start_tick,
                            "channel": _channel,
                        }
                    )

            elif (
                _tok_type_1 in self.instruments_nd
                and _tok_type_2 == "onset"
                and _tok_type_3 == "dur"
            ):
                _pitch = tok_1[1]
                _channel = instrument_to_channel.get(tok_1[0], None)
                _velocity = tok_1[2]
                _start_tick = curr_tick + tok_2[1]
                _end_tick = _start_tick + tok_3[1]

                if _channel is None:
                    logging.warning(
                        "Tried to decode note message for unexpected instrument"
                    )
                else:
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": _pitch,
                                "start": _start_tick,
                                "end": _end_tick,
                                "velocity": _velocity,
                            },
                            "tick": _start_tick,
                            "channel": _channel,
                        }
                    )
            else:
                logging.warning(
                    f"Unexpected token sequence: {tok_1}, {tok_2}, {tok_3}"
                )

        return MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

    def export_pitch_aug(self, aug_range: int):
        """Exports a function that augments the pitch of all note tokens.

        Note that notes which fall out of the range (0, 127) will be replaced
        with the unknown token '<U>'.

        Args:
            aug_range (int): Returned function will randomly augment the pitch
                from a value in the range (-aug_range, aug_range).

        Returns:
            Callable[list]: Exported function.
        """

        def pitch_aug_seq(
            src: list,
            unk_tok: str,
            _aug_range: float,
            pitch_aug: int | None = None,
        ):
            def pitch_aug_tok(tok, _pitch_aug):
                if isinstance(tok, str):  # Stand in for special tokens
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "onset"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    (_instrument, _pitch, _velocity) = tok

                    if 0 <= _pitch + _pitch_aug <= 127:
                        return (_instrument, _pitch + _pitch_aug, _velocity)
                    else:
                        return unk_tok

            if not pitch_aug:
                pitch_aug = random.randint(-_aug_range, _aug_range)

            return [pitch_aug_tok(x, pitch_aug) for x in src]

        # See functools.partial docs
        return functools.partial(
            self.export_aug_fn_concat(aug_fn=pitch_aug_seq),
            unk_tok=self.unk_tok,
            _aug_range=aug_range,
        )

    def export_velocity_aug(self, aug_steps_range: int):
        """Exports a function which augments the velocity of all pitch tokens.

        This augmentation truncated such that it returns a valid note token.

        Args:
            aug_steps_range (int): Returned function will randomly augment
                velocity in the range aug_steps_range * (-self.velocity_step,
                self.velocity step).

        Returns:
            Callable[str]: Exported function.
        """

        def velocity_aug_seq(
            src: list,
            velocity_step: int,
            max_velocity: int,
            _aug_steps_range: int,
            velocity_aug: int | None = None,
        ):
            def velocity_aug_tok(tok, _velocity_aug):
                if isinstance(tok, str):  # Stand in for special tokens
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "onset"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    (_instrument, _pitch, _velocity) = tok

                    # Check it doesn't go out of bounds
                    if _velocity + _velocity_aug >= max_velocity:
                        return (_instrument, _pitch, max_velocity)
                    elif _velocity + _velocity_aug <= velocity_step:
                        return (_instrument, _pitch, velocity_step)

                    return (_instrument, _pitch, _velocity + _velocity_aug)

            if not velocity_aug:
                velocity_aug = velocity_step * random.randint(
                    -_aug_steps_range, _aug_steps_range
                )

            return [velocity_aug_tok(x, velocity_aug) for x in src]

        # See functools.partial docs
        return functools.partial(
            self.export_aug_fn_concat(aug_fn=velocity_aug_seq),
            velocity_step=self.velocity_step,
            max_velocity=self.max_velocity,
            _aug_steps_range=aug_steps_range,
        )

    # TODO: Adjust this so it can handle other tokens like <SEP>
    def export_tempo_aug(self, tempo_aug_range, mixup: bool):
        # Chord mix up will randomly reorder concurrent notes. A concurrent
        # notes are those which occur at the onset.
        def tempo_aug(
            src: list,
            abs_time_step: int,
            max_dur: int,
            time_step: int,
            unk_tok: str,
            time_tok: str,
            dim_tok: str,
            start_tok: str,
            end_tok: str,
            instruments_wd: list,
            _tempo_aug_range: float,
            _mixup: bool,
            tempo_aug: float | None = None,
        ):
            """This must be used with export_aug_fn_concat in order to work
            properly for concatenated sequences."""

            def _quantize_time(_n: int):
                return round(_n / time_step) * time_step

            if not tempo_aug:
                tempo_aug = random.uniform(
                    1 - _tempo_aug_range, 1 + _tempo_aug_range
                )

            src_time_tok_cnt = 0
            dim_tok_seen = None
            res = []
            note_buffer = None
            buffer = defaultdict(lambda: defaultdict(list))
            for tok_1, tok_2, tok_3 in zip(src, src[1:], src[2:]):
                if tok_1 == time_tok:
                    _tok_type = "time"
                elif tok_1 == unk_tok:
                    _tok_type = "unk"
                elif tok_1 == start_tok:
                    res.append(tok_1)
                    continue
                elif tok_1 == dim_tok and note_buffer:
                    dim_tok_seen = (src_time_tok_cnt, note_buffer["onset"][1])
                    continue
                elif tok_1[0] == "prefix":
                    res.append(tok_1)
                    continue
                elif tok_1[0] in instruments_wd:
                    _tok_type = tok_1[0]
                else:
                    # This only triggers for incomplete notes at the beginning,
                    # e.g. an onset token before a note token is seen
                    continue

                if _tok_type == "time":
                    src_time_tok_cnt += 1
                elif _tok_type == "drum":
                    note_buffer = {
                        "note": tok_1,
                        "onset": tok_2,
                        "dur": None,
                    }
                    buffer[src_time_tok_cnt][tok_2[1]].append(note_buffer)
                else:  # unk or in instruments_wd
                    note_buffer = {
                        "note": tok_1,
                        "onset": tok_2,
                        "dur": tok_3,
                    }
                    buffer[src_time_tok_cnt][tok_2[1]].append(note_buffer)

            prev_tgt_time_tok_cnt = 0
            for src_time_tok_cnt, interval_notes in sorted(buffer.items()):
                for src_onset, notes_by_onset in sorted(interval_notes.items()):
                    src_time = src_time_tok_cnt * abs_time_step + src_onset
                    tgt_time = round(src_time * tempo_aug)
                    curr_tgt_time_tok_cnt = tgt_time // abs_time_step
                    curr_tgt_onset = _quantize_time(tgt_time % abs_time_step)

                    if curr_tgt_onset == abs_time_step:
                        curr_tgt_onset -= time_step

                    for _ in range(
                        curr_tgt_time_tok_cnt - prev_tgt_time_tok_cnt
                    ):
                        res.append(time_tok)
                    prev_tgt_time_tok_cnt = curr_tgt_time_tok_cnt

                    if _mixup == True:
                        random.shuffle(notes_by_onset)

                    for note in notes_by_onset:
                        _src_note_tok = note["note"]
                        _src_dur_tok = note["dur"]

                        if _src_dur_tok is not None:
                            tgt_dur = _quantize_time(
                                round(_src_dur_tok[1] * tempo_aug)
                            )
                            tgt_dur = min(tgt_dur, max_dur)
                        else:
                            tgt_dur = None

                        res.append(_src_note_tok)
                        res.append(("onset", curr_tgt_onset))
                        if tgt_dur:
                            res.append(("dur", tgt_dur))

                        if dim_tok_seen is not None and dim_tok_seen == (
                            src_time_tok_cnt,
                            src_onset,
                        ):
                            res.append(dim_tok)
                            dim_tok_seen = None

            if src[-1] == end_tok:
                res.append(end_tok)

            return res

        return functools.partial(
            self.export_aug_fn_concat(aug_fn=tempo_aug),
            abs_time_step=self.abs_time_step,
            max_dur=self.max_dur,
            time_step=self.time_step,
            unk_tok=self.unk_tok,
            time_tok=self.time_tok,
            dim_tok=self.dim_tok,
            end_tok=self.eos_tok,
            start_tok=self.bos_tok,
            instruments_wd=self.instruments_wd,
            _tempo_aug_range=tempo_aug_range,
            _mixup=mixup,
        )


class SeparatedAbsTokenizer(AbsTokenizer):
    def __init__(self, return_tensors: bool = False):
        super().__init__(return_tensors)

        self.name = "separated_abs"
        self.lm_config = load_config()["tokenizer"]["lm"]
        self.sep_tok = "<SEP>"
        self.add_tokens_to_vocab([self.sep_tok])
        self.special_tokens.append(self.sep_tok)

    def tokenize(self, midi_dict: MidiDict, **kwargs):
        assert (
            midi_dict.metadata.get("cutoff_ms") is not None
        ), "Invalid MidiDict"
        cutoff_ms = midi_dict.metadata["cutoff_ms"]

        seq = super().tokenize(midi_dict, **kwargs)
        final_seq = copy.deepcopy(seq)

        cnt = 0
        curr_time = 0
        for idx, tok in enumerate(seq):
            if tok == self.time_tok:
                cnt += 1
            elif isinstance(tok, tuple) and tok[0] == "onset":
                _curr_time = (self.config["abs_time_step_ms"] * cnt) + tok[1]
                assert _curr_time >= curr_time
                curr_time = _curr_time
            elif (
                isinstance(tok, tuple)
                and tok[0] == "dur"
                and curr_time > cutoff_ms
            ):
                final_seq.insert(idx + 1, self.sep_tok)
                return final_seq

        return seq

    def detokenize(self, midi_dict: MidiDict, **kwargs):
        return super().detokenize(midi_dict, **kwargs)

    def export_data_aug(self):
        return [
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]


# class LMTokenizer(AbsTokenizer):
#     def __init__(self, return_tensors: bool = False):
#         super().__init__(return_tensors)

#         self.lm_config = load_config()["tokenizer"]["lm"]
#         self.tag_tokens = [f"{tag}: on" for tag in self.lm_config["tags"]] + [
#             f"{tag}: off" for tag in self.lm_config["tags"]
#         ]
#         self.add_tokens_to_vocab(self.tag_tokens)
#         self.special_tokens += self.tag_tokens

#     def tokenize(self, midi_dict: MidiDict, **kwargs):
#         seq = super().tokenize(midi_dict, **kwargs)

#         for tag in midi_dict["metadata"]["listening_model"]:
#             tag_name, start_ms, end_ms = tag
#             start_pos = int(start_ms / self.config["abs_time_step"])
#             end_pos = int(end_ms / self.config["abs_time_step"])

#             if start_pos == end_pos:
#                 end_pos += 1

#             cnt = 0
#             for idx, tok in enumerate(seq):
#                 if tok == self.time_tok:
#                     cnt += 1
#                     if cnt == start_pos:
#                         seq.insert(idx + 1, f"{tag_name}: on")
#                         break
#                     elif cnt == end_pos:
#                         seq.insert(idx + 1, f"{tag_name}: off")
#                         break

#         return seq

#     def detokenize(self, midi_dict: MidiDict, **kwargs):
#         return super().tokenize(midi_dict, **kwargs)

#     def export_data_aug(self):
#         return [
#             self.export_pitch_aug(5),
#             self.export_velocity_aug(1),
#         ]


class RelTokenizer(Tokenizer):
    """MidiDict tokenizer implemented with relative onset timings.

    This tokenizer is depreciated and has been replaced by AbsTokenizer.
    """

    def __init__(self, return_tensors: bool = False):
        super().__init__(return_tensors)
        self.config = load_config()["tokenizer"]["rel"]
        self.name = "rel"

        # Calculate time quantizations (in ms)
        self.num_time_step = self.config["time_quantization"]["num_steps"]
        self.min_time_step = self.config["time_quantization"]["step"]
        self.time_step_quantizations = [
            self.min_time_step * i for i in range(self.num_time_step)
        ]
        self.max_time_step = self.time_step_quantizations[-1]

        # Calculate velocity quantizations
        self.velocity_step = self.config["velocity_quantization"]["step"]
        self.velocity_quantizations = [
            i * self.velocity_step
            for i in range(int(127 / self.velocity_step) + 1)
        ]
        self.max_velocity = self.velocity_quantizations[-1]

        # _nd = no drum; _wd = with drum
        self.instruments_nd = [
            k
            for k, v in self.config["ignore_instruments"].items()
            if v is False
        ]
        self.instruments_wd = self.instruments_nd + ["drum"]

        # Prefix tokens
        self.prefix_tokens = [
            ("prefix", "instrument", x) for x in self.instruments_wd
        ]
        self.composer_names = self.config["composer_names"]
        self.form_names = self.config["form_names"]
        self.genre_names = self.config["form_names"]
        self.prefix_tokens += [
            ("prefix", "composer", x) for x in self.composer_names
        ]
        self.prefix_tokens += [("prefix", "form", x) for x in self.form_names]
        self.prefix_tokens += [("prefix", "genre", x) for x in self.genre_names]

        # Build vocab
        self.wait_tokens = [("wait", i) for i in self.time_step_quantizations]
        self.dur_tokens = [("dur", i) for i in self.time_step_quantizations]
        self.drum_tokens = [("drum", i) for i in range(35, 82)]

        self.note_tokens = list(
            itertools.product(
                self.instruments_nd,
                [i for i in range(128)],
                self.velocity_quantizations,
            )
        )

        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prefix_tokens
            + self.note_tokens
            + self.drum_tokens
            + self.dur_tokens
            + self.wait_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

    def export_data_aug(self):
        return [
            self.export_chord_mixup(),
            self.export_tempo_aug(tempo_aug_range=0.2),
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]

    def _quantize_time(self, time: int):
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.time_step_quantizations)

    def _quantize_velocity(self, velocity: int):
        # This function will return values in the range 0 < res =< 127
        velocity_quantized = self._find_closest_int(
            velocity, self.velocity_quantizations
        )

        if velocity_quantized == 0 and velocity != 0:
            return self.velocity_step
        else:
            return velocity_quantized

    def _format(self, prefix: list, unformatted_seq: list):
        # If unformatted_seq is longer than 150 tokens insert diminish tok
        idx = -100 + random.randint(-10, 10)
        if len(unformatted_seq) > 150:
            if unformatted_seq[idx][0] == "dur":  # Don't want: note, <D>, dur
                unformatted_seq.insert(idx - 1, self.dim_tok)
            else:
                unformatted_seq.insert(idx, self.dim_tok)

        res = prefix + [self.bos_tok] + unformatted_seq + [self.eos_tok]

        return res

    def _tokenize_midi_dict(self, midi_dict: MidiDict):
        ticks_per_beat = midi_dict.ticks_per_beat
        midi_dict.remove_instruments(self.config["ignore_instruments"])

        if len(midi_dict.note_msgs) == 0:
            raise Exception("note_msgs is empty after ignoring instruments")

        channel_to_pedal_intervals = self._build_pedal_intervals(midi_dict)

        channels_used = {msg["channel"] for msg in midi_dict.note_msgs}

        channel_to_instrument = {
            msg["channel"]: midi_dict.program_to_instrument[msg["data"]]
            for msg in midi_dict.instrument_msgs
            if msg["channel"] != 9  # Exclude drums
        }
        # If non-drum channel is missing from instrument_msgs, default to piano
        for c in channels_used:
            if channel_to_instrument.get(c) is None and c != 9:
                channel_to_instrument[c] = "piano"

        # Calculate prefix
        prefix = [
            ("prefix", "instrument", x)
            for x in set(channel_to_instrument.values())
        ]
        if 9 in channels_used:
            prefix.append(("prefix", "instrument", "drum"))
        composer = midi_dict.metadata.get("composer")
        if composer and (composer in self.composer_names):
            prefix.insert(0, ("prefix", "composer", composer))
        form = midi_dict.metadata.get("form")
        if form and (form in self.form_names):
            prefix.insert(0, ("prefix", "form", form))
        genre = midi_dict.metadata.get("genre")
        if genre and (genre in self.genre_names):
            prefix.insert(0, ("prefix", "genre", genre))
        random.shuffle(prefix)

        # NOTE: Any preceding silence is removed implicitly
        tokenized_seq = []
        num_notes = len(midi_dict.note_msgs)
        for i, msg in enumerate(midi_dict.note_msgs):
            # Special case instrument is a drum. This occurs exclusively when
            # MIDI channel is 9 when 0 indexing
            if msg["channel"] == 9:
                _pitch = msg["data"]["pitch"]
                tokenized_seq.append(("drum", _pitch))

            else:  # Non drum case (i.e. an instrument note)
                _instrument = channel_to_instrument[msg["channel"]]
                _pitch = msg["data"]["pitch"]
                _velocity = msg["data"]["velocity"]
                _start_tick = msg["data"]["start"]
                _end_tick = msg["data"]["end"]

                # Update _end_tick if affected by pedal
                for pedal_interval in channel_to_pedal_intervals[
                    msg["channel"]
                ]:
                    pedal_start, pedal_end = (
                        pedal_interval[0],
                        pedal_interval[1],
                    )
                    if (
                        pedal_start <= _start_tick < pedal_end
                        and _end_tick < pedal_end
                    ):
                        _end_tick = pedal_end

                _note_duration = get_duration_ms(
                    start_tick=_start_tick,
                    end_tick=_end_tick,
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                # Quantize
                _velocity = self._quantize_velocity(_velocity)
                _note_duration = self._quantize_time(_note_duration)
                if _note_duration == 0:
                    _note_duration = self.min_time_step

                tokenized_seq.append((_instrument, _pitch, _velocity))
                tokenized_seq.append(("dur", _note_duration))

            # Only add wait token if there is a msg after the current one
            if i <= num_notes - 2:
                _wait_duration = get_duration_ms(
                    start_tick=msg["data"]["start"],
                    end_tick=midi_dict.note_msgs[i + 1]["data"]["start"],
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                # If wait duration is longer than maximum quantized time step
                # append max_time_step tokens repeatedly
                while _wait_duration > self.max_time_step:
                    tokenized_seq.append(("wait", self.max_time_step))
                    _wait_duration -= self.max_time_step

                # Only append wait tok if it is non-zero
                _wait_duration = self._quantize_time(_wait_duration)
                if _wait_duration != 0:
                    tokenized_seq.append(("wait", _wait_duration))

        return self._format(
            prefix=prefix,
            unformatted_seq=tokenized_seq,
        )

    def _detokenize_midi_dict(self, tokenized_seq: list):
        instrument_programs = self.config["instrument_programs"]
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        TICKS_PER_BEAT = 500
        TEMPO = 500000

        # Set message tempos
        tempo_msgs = [{"type": "tempo", "data": TEMPO, "tick": 0}]
        meta_msgs = []
        pedal_msgs = []
        instrument_msgs = []

        instrument_to_channel = {}

        # Add non-drum instrument_msgs, breaks at first note token
        channel_idx = 0
        for idx, tok in enumerate(tokenized_seq):
            if channel_idx == 9:  # Skip channel reserved for drums
                channel_idx += 1

            if tok in self.special_tokens:
                # Skip special tokens
                continue
            elif (
                tok[0] == "prefix"
                and tok[1] == "instrument"
                and tok[2] in self.instruments_wd
            ):
                # Process instrument prefix tokens
                if tok[2] in instrument_to_channel.keys():
                    # logging.warning(f"Duplicate prefix {tok[2]}")
                    continue
                elif tok[2] == "drum":
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": 0,
                            "tick": 0,
                            "channel": 9,
                        }
                    )
                    instrument_to_channel["drum"] = 9
                else:
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": instrument_programs[tok[2]],
                            "tick": 0,
                            "channel": channel_idx,
                        }
                    )
                    instrument_to_channel[tok[2]] = channel_idx
                    channel_idx += 1
            elif tok[0] == "prefix":
                # Skip all other prefix tokens
                continue
            else:
                # Note, wait, or duration token
                start = idx
                break

        # Note messages
        note_msgs = []
        curr_tick = 0
        for curr_tok, next_tok in zip(
            tokenized_seq[start:], tokenized_seq[start + 1 :]
        ):
            if curr_tok in self.special_tokens:
                _curr_tok_type = "special"
            else:
                _curr_tok_type = curr_tok[0]

            if next_tok in self.special_tokens:
                _next_tok_type = "special"
            else:
                _next_tok_type = next_tok[0]

            if (
                _curr_tok_type == "special"
                or _curr_tok_type == "prefix"
                or _curr_tok_type == "dur"
            ):
                continue
            elif _curr_tok_type == "wait":
                curr_tick += curr_tok[1]
            elif _curr_tok_type == "drum":
                _tick = curr_tick
                _pitch = curr_tok[1]
                _channel = instrument_to_channel["drum"]
                _velocity = self.config["drum_velocity"]
                _start_tick = curr_tick
                _end_tick = curr_tick + self.min_time_step

                note_msgs.append(
                    {
                        "type": "note",
                        "data": {
                            "pitch": _pitch,
                            "start": _start_tick,
                            "end": _end_tick,
                            "velocity": _velocity,
                        },
                        "tick": _tick,
                        "channel": _channel,
                    }
                )

            elif (
                _curr_tok_type in self.instruments_nd
                and _next_tok_type == "dur"
            ):
                duration = next_tok[1]

                _tick = curr_tick

                _channel = instrument_to_channel.get(curr_tok[0], None)
                if _channel is None:
                    logging.warning("Tried to decode unexpected note message")
                    continue

                _pitch = curr_tok[1]
                _velocity = curr_tok[2]
                _start_tick = curr_tick
                _end_tick = curr_tick + duration

                note_msgs.append(
                    {
                        "type": "note",
                        "data": {
                            "pitch": _pitch,
                            "start": _start_tick,
                            "end": _end_tick,
                            "velocity": _velocity,
                        },
                        "tick": _tick,
                        "channel": _channel,
                    }
                )

            else:
                logging.warning(
                    f"Unexpected token sequence: {curr_tok}, {next_tok}"
                )

        return MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

    def export_pitch_aug(self, aug_range: int):
        """Exports a function that augments the pitch of all note tokens.

        Note that notes which fall out of the range (0, 127) will be replaced
        with the unknown token '<U>'.

        Args:
            aug_range (int): Returned function will randomly augment the pitch
                from a value in the range (-aug_range, aug_range).

        Returns:
            Callable[list]: Exported function.
        """

        def pitch_aug_seq(
            src: list,
            unk_tok: str,
            _aug_range: float,
        ):
            def pitch_aug_tok(tok, _pitch_aug):
                if isinstance(tok, str):  # Stand in for special tokens
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "wait"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    (_instrument, _pitch, _velocity) = tok

                    if 0 <= _pitch + _pitch_aug <= 127:
                        return (_instrument, _pitch + _pitch_aug, _velocity)
                    else:
                        return unk_tok

            pitch_aug = random.randint(-_aug_range, _aug_range)
            return [pitch_aug_tok(x, pitch_aug) for x in src]

        # See functools.partial docs
        return functools.partial(
            pitch_aug_seq,
            unk_tok=self.unk_tok,
            _aug_range=aug_range,
        )

    def export_velocity_aug(self, aug_steps_range: int):
        """Exports a function which augments the velocity of all pitch tokens.

        This augmentation truncated such that it returns a valid note token.

        Args:
            aug_steps_range (int): Returned function will randomly augment
                velocity in the range aug_steps_range * (-self.velocity_step,
                self.velocity step).

        Returns:
            Callable[str]: Exported function.
        """

        def velocity_aug_seq(
            src: list,
            velocity_step: int,
            max_velocity: int,
            _aug_steps_range: int,
        ):
            def velocity_aug_tok(tok, _velocity_aug):
                if isinstance(tok, str):  # Stand in for special tokens
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "wait"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    (_instrument, _pitch, _velocity) = tok

                    # Check it doesn't go out of bounds
                    if _velocity + _velocity_aug >= max_velocity:
                        return (_instrument, _pitch, max_velocity)
                    elif _velocity + _velocity_aug <= velocity_step:
                        return (_instrument, _pitch, velocity_step)

                    return (_instrument, _pitch, _velocity + _velocity_aug)

            velocity_aug = velocity_step * random.randint(
                -_aug_steps_range, _aug_steps_range
            )
            return [velocity_aug_tok(x, velocity_aug) for x in src]

        # See functools.partial docs
        return functools.partial(
            velocity_aug_seq,
            velocity_step=self.velocity_step,
            max_velocity=self.max_velocity,
            _aug_steps_range=aug_steps_range,
        )

    def export_tempo_aug(self, tempo_aug_range: float):
        def tempo_aug_seq(
            src: list,
            min_time_step: int,
            max_time_step: int,
            _tempo_aug_range: float,
        ):
            def tempo_aug_tok_raw(tok, _tempo_aug):
                if isinstance(tok, str):  # Stand in for special tokens
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if _tok_type == "wait" or _tok_type == "dur":
                    (__tok_type, _dur) = tok

                    return (
                        __tok_type,
                        min_time_step
                        * int(round(float(_tempo_aug * _dur) / min_time_step)),
                    )
                else:
                    # Return without changing
                    return tok

            tempo_aug = random.uniform(
                1 - tempo_aug_range, 1 + _tempo_aug_range
            )
            augmented_seq = [tempo_aug_tok_raw(x, tempo_aug) for x in src]

            # Recalculate dur and wait tokens so that they are correctly
            # formatted after naive augmentation.
            idx = 0
            buffer = []
            while idx < len(augmented_seq):
                tok = augmented_seq[idx]
                if isinstance(tok, str):
                    tok_type = "special"
                else:
                    tok_type = tok[0]

                # Get tok_type of next token if possible
                if idx + 1 < len(augmented_seq):
                    next_tok = augmented_seq[idx + 1]
                    if isinstance(next_tok, str):
                        next_tok_type = "special"
                    else:
                        next_tok_type = next_tok[0]
                else:
                    next_tok_type = None

                # If necessary add wait token to the buffer
                if tok_type == "wait":
                    if buffer or tok[1] >= max_time_step:
                        # Overflow
                        buffer.append(augmented_seq.pop(idx))
                    elif next_tok_type == "wait":
                        # Underflow
                        buffer.append(augmented_seq.pop(idx))
                    else:
                        idx += 1

                # Current tok not wait token so if the buffer is not empty
                # recalculate and reinsert wait tokens in the buffer.
                elif buffer:
                    buffer_remaining_dur = sum(_tok[1] for _tok in buffer)

                    while buffer_remaining_dur > max_time_step:
                        augmented_seq.insert(idx, ("wait", max_time_step))
                        buffer_remaining_dur -= max_time_step
                        idx += 1

                    augmented_seq.insert(idx, ("wait", buffer_remaining_dur))
                    buffer = []
                    idx += 1

                # If dur token has overflowed, truncate at max_time_step
                elif tok_type == "dur":
                    if tok[1] > max_time_step:
                        augmented_seq[idx] = ("dur", max_time_step)
                    idx += 1

                else:
                    idx += 1

            return augmented_seq

        # See functools.partial docs
        return functools.partial(
            self.export_aug_fn_concat(aug_fn=tempo_aug_seq),
            min_time_step=self.min_time_step,
            max_time_step=self.max_time_step,
            _tempo_aug_range=tempo_aug_range,
        )

    def export_chord_mixup(self):
        # Chord mix up will randomly reorder concurrent notes. A concurrent
        # notes are those which are not separated by a 'wait' token.
        def chord_mixup(src: list, unk_tok: str):
            stack = []
            for idx, tok in enumerate(src):
                if isinstance(tok, str):
                    tok_type = "special"
                else:
                    tok_type = tok[0]

                if (
                    tok_type == "special" or tok_type == "prefix"
                ) and tok != unk_tok:
                    # Skip special tok (when not unk), reset stack to be safe
                    stack = []
                elif tok_type == "wait" and len(stack) <= 1:
                    # Reset stack as it only contains one note
                    stack = []
                elif tok_type == "wait" and len(stack) > 1:
                    # Stack contains more than one note -> mix-up stack.
                    random.shuffle(stack)
                    num_toks = sum(len(note) for note in stack)
                    _idx = idx - num_toks

                    while stack:
                        entry = stack.pop()
                        if entry["note"] == unk_tok:
                            # This can happen if the note token has its pitch
                            # augmented out of the valid range. In this case we
                            # do not want to index it as it is not a note token
                            src[_idx] = entry["note"]
                            src[_idx + 1] = entry["dur"]
                            _idx += 2
                        elif entry["note"][0] == "drum":
                            # Drum case doesn't require a duration token
                            src[_idx] = entry["note"]
                            _idx += 1
                        else:
                            src[_idx] = entry["note"]
                            src[_idx + 1] = entry["dur"]
                            _idx += 2

                elif tok_type == "dur":
                    # Add dur to previously added note token if exists
                    if stack:
                        stack[-1]["dur"] = tok
                else:
                    # Note token -> append to stack
                    stack.append({"note": tok})

            return src

        return functools.partial(
            self.export_aug_fn_concat(aug_fn=chord_mixup),
            unk_tok=self.unk_tok,
        )
