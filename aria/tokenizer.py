"""Tokenizer for MIDI conditioned completions"""

import copy

from ariautils.midi import MidiDict, get_duration_ms
from ariautils.tokenizer import AbsTokenizer as _AbsTokenizer


class SeparatedAbsTokenizer(_AbsTokenizer):
    def __init__(self):
        super().__init__()

        self.name = "separated_abs"
        self.inst_start_tok = "<INST>"
        self.inst_end_tok = "</INST>"
        self.add_tokens_to_vocab([self.inst_start_tok, self.inst_end_tok])
        self.special_tokens.append(self.inst_start_tok)
        self.special_tokens.append(self.inst_end_tok)

    def tokenize(self, midi_dict: MidiDict, **kwargs):
        def _add_inst_toks(_seq: list, _start_ms: int, _end_ms: int):
            res_seq = copy.deepcopy(_seq)

            inst_inserted = False
            time_tok_cnt = 0
            curr_time_ms = 0
            for idx, (tok_1, tok_2) in enumerate(zip(_seq, _seq[1:])):
                if tok_1 == self.time_tok:
                    time_tok_cnt += 1
                elif (
                    isinstance(tok_1, tuple) and tok_1[0] in self.instruments_wd
                ):
                    assert isinstance(tok_2, tuple) and tok_2[0] == "onset"

                    # Adjust time
                    _curr_time = (
                        self.config["abs_time_step_ms"] * time_tok_cnt
                    ) + tok_2[1]

                    assert _curr_time >= curr_time_ms
                    curr_time_ms = _curr_time

                    if curr_time_ms >= _start_ms and inst_inserted == False:
                        res_seq.insert(idx, self.inst_start_tok)
                        inst_inserted = True
                    if curr_time_ms > _end_ms and inst_inserted == True:
                        res_seq.insert(idx + 1, self.inst_end_tok)
                        break

            return res_seq

        if midi_dict.metadata.get("noisy_intervals") is None:
            print("noisy_intervals metadata not present")
            return super().tokenize(midi_dict, **kwargs)

        seq = super().tokenize(midi_dict, **kwargs)

        # This logic is required as the tokenizer removes proceeding silence
        first_note_ms = get_duration_ms(
            start_tick=0,
            end_tick=midi_dict.note_msgs[0]["data"]["start"],
            tempo_msgs=midi_dict.tempo_msgs,
            ticks_per_beat=midi_dict.ticks_per_beat,
        )
        noisy_intervals = [
            [
                ival[0] - first_note_ms,
                ival[1] - first_note_ms,
            ]
            for ival in midi_dict.metadata.get("noisy_intervals")
            if ival[1] >= first_note_ms
        ]

        for start_ms, end_ms in noisy_intervals:
            seq = _add_inst_toks(seq, start_ms, end_ms)

        return seq

    def detokenize(self, midi_dict: MidiDict, **kwargs):
        return super().detokenize(midi_dict, **kwargs)

    def export_data_aug(self):
        return [
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]

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

            # Insert inst_start_tok if it is missing (but required)
            for idx in range(len(_chunk)):
                tok = _chunk[idx]

                if tok == self.inst_start_tok:
                    break
                elif tok == self.inst_end_tok:
                    if _chunk[0] == self.bos_tok:
                        _chunk.insert(1, self.inst_start_tok)
                    else:
                        _chunk.insert(0, self.inst_start_tok)
                    break

            return _chunk

        prefix = []
        while seq:
            tok = seq[0]
            if tok != self.bos_tok and tok[0] == "prefix":
                prefix.append(seq.pop(0))
            else:
                break

        # Generate chunks
        chunks = [
            _process_chunk(seq[idx : idx + seq_len])
            for idx in range(0, len(seq) - 100, seq_len)
        ]

        res = []
        for chunk in chunks:
            if self.inst_start_tok not in chunk:
                continue

            sub_seq = prefix + chunk
            sub_seq = sub_seq[:seq_len]
            sub_seq += [self.pad_tok] * (seq_len - len(sub_seq))
            res.append(sub_seq)

        return res
