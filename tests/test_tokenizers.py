import unittest
import logging
import os
import time

from typing import Callable

from aria import tokenizer
from aria.config import load_config
from aria.data.midi import MidiDict
from aria.data.datasets import _get_combined_mididict, _noise_midi_dict
from aria.utils import midi_to_audio


if not os.path.isdir("tests/test_results"):
    os.makedirs("tests/test_results")


# TODO: Implement with tokenizer functions
def get_short_seq_abs(tknzr: tokenizer.AbsTokenizer):
    return [
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(0)),
        ("dur", tknzr._quantize_dur(50)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(100)),
        ("piano", 64, tknzr._quantize_velocity(75)),
        ("onset", tknzr._quantize_onset(100)),
        ("dur", tknzr._quantize_dur(5000)),
        "<T>",
        "<T>",
        "<T>",
        ("piano", 65, tknzr._quantize_velocity(75)),
        ("onset", tknzr._quantize_onset(170)),
        ("dur", tknzr._quantize_dur(100)),
        "<D>",
        ("piano", 60, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(60)),
        "<U>",
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(70)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(270)),
        "<T>",
        ("piano", 80, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(80)),
        "<E>",
    ]


def get_concat_seq_abs(tknzr: tokenizer.AbsTokenizer):
    return [
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(60)),
        "<U>",
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(70)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(270)),
        "<T>",
        ("piano", 80, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(80)),
        "<E>",
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(0)),
        ("dur", tknzr._quantize_dur(50)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(100)),
        ("piano", 64, tknzr._quantize_velocity(75)),
        ("onset", tknzr._quantize_onset(100)),
        ("dur", tknzr._quantize_dur(5000)),
        "<T>",
        "<T>",
        "<T>",
        ("piano", 65, tknzr._quantize_velocity(75)),
        ("onset", tknzr._quantize_onset(170)),
        ("dur", tknzr._quantize_dur(100)),
        "<D>",
        ("piano", 60, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(60)),
        "<U>",
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(70)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(270)),
        "<T>",
        ("piano", 80, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(270)),
        ("dur", tknzr._quantize_dur(80)),
        "<E>",
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(45)),
        ("onset", tknzr._quantize_onset(0)),
        ("dur", tknzr._quantize_dur(50)),
        ("drum", 50),
        ("onset", tknzr._quantize_onset(100)),
        ("piano", 64, tknzr._quantize_velocity(75)),
        ("onset", tknzr._quantize_onset(100)),
        ("dur", tknzr._quantize_dur(5000)),
        "<T>",
        "<T>",
    ]


def get_short_seq_rel(tknzr: tokenizer.RelTokenizer):
    return [
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        ("prefix", "composer", "bach"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(50)),
        ("wait", tknzr._quantize_time(100)),
        ("drum", 50),
        ("piano", 64, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 65, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(100)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 60, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(60)),
        ("piano", 70, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(70)),
        ("drum", 50),
        ("piano", 80, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(80)),
        ("wait", tknzr._quantize_time(100)),
        "<E>",
    ]


def get_concat_seq_rel(tknzr: tokenizer.RelTokenizer):
    return [
        ("dur", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 65, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(100)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 60, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(60)),
        ("piano", 70, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(70)),
        ("drum", 50),
        ("piano", 80, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(80)),
        ("wait", tknzr._quantize_time(100)),
        "<E>",
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        ("prefix", "composer", "bach"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(50)),
        ("wait", tknzr._quantize_time(100)),
        ("drum", tknzr._quantize_time(50)),
        ("piano", 64, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(1000000)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 65, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(100)),
        ("wait", tknzr._quantize_time(100)),
        ("piano", 60, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(60)),
        ("piano", 70, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(70)),
        ("drum", 50),
        ("piano", 80, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(80)),
        ("wait", tknzr._quantize_time(100)),
        "<E>",
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        ("prefix", "composer", "bach"),
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(50)),
        ("wait", tknzr._quantize_time(100)),
        ("drum", tknzr._quantize_time(50)),
        ("piano", 64, tknzr._quantize_velocity(70)),
    ]


class TestAbsTokenizer(unittest.TestCase):
    def test_tokenize_detokenize_mididict(self):
        def tokenize_detokenize(file_name: str):
            mid_path = f"tests/test_data/{file_name}"
            midi_dict = MidiDict.from_midi(mid_path=mid_path)
            tokenized_seq = tknzr.tokenize(midi_dict)
            detokenized_midi_dict = tknzr.detokenize(tokenized_seq)
            res = detokenized_midi_dict.to_midi()
            res.save(f"tests/test_results/{file_name}")

        tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        tokenize_detokenize("basic.mid")
        tokenize_detokenize("arabesque.mid")
        tokenize_detokenize("beethoven_sonata.mid")
        tokenize_detokenize("bach.mid")
        tokenize_detokenize("expressive.mid")
        tokenize_detokenize("pop.mid")
        tokenize_detokenize("beethoven_moonlight.mid")
        tokenize_detokenize("maestro.mid")

    def test_aug(self):
        def tokenize_aug_detokenize(
            file_name: str,
            aug_fn: Callable,
            aug_name: str,
            audio=False,
        ):
            mid_path = f"tests/test_data/{file_name}"
            midi_dict = MidiDict.from_midi(mid_path=mid_path)
            tokenized_seq = tknzr.tokenize(midi_dict)
            tokenized_seq_aug = aug_fn(tokenized_seq)
            detokenized_midi_dict = tknzr.detokenize(tokenized_seq_aug)
            res = detokenized_midi_dict.to_midi()
            save_path = f"tests/test_results/abs_{aug_name}_{file_name}"
            res.save(save_path)
            if audio is True:
                midi_to_audio(save_path)

        tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        seq = get_short_seq_abs(tknzr)
        seq_concat = get_concat_seq_abs(tknzr)
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)
        tempo_aug_fn = tknzr.export_tempo_aug(tempo_aug_range=0.5, mixup=True)

        # Pitch augmentation
        seq_pitch_augmented = pitch_aug_fn(get_short_seq_abs(tknzr))
        logging.info(f"pitch_aug_fn:\n{seq} ->\n\n{seq_pitch_augmented}\n")
        tokenize_aug_detokenize("basic.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize("arabesque.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize("beethoven_sonata.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize("bach.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize("expressive.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize("pop.mid", pitch_aug_fn, "pitch")
        tokenize_aug_detokenize(
            "beethoven_moonlight.mid", pitch_aug_fn, "pitch"
        )

        # Velocity augmentation
        seq_velocity_augmented = velocity_aug_fn(get_short_seq_abs(tknzr))
        logging.info(
            f"velocity_aug_fn:\n{seq} ->\n\n{seq_velocity_augmented}\n"
        )
        tokenize_aug_detokenize("basic.mid", velocity_aug_fn, "velocity")
        tokenize_aug_detokenize("arabesque.mid", velocity_aug_fn, "velocity")
        tokenize_aug_detokenize(
            "beethoven_sonata.mid", velocity_aug_fn, "velocity"
        )
        tokenize_aug_detokenize("bach.mid", velocity_aug_fn, "velocity")
        tokenize_aug_detokenize("expressive.mid", velocity_aug_fn, "velocity")
        tokenize_aug_detokenize("pop.mid", velocity_aug_fn, "velocity")
        tokenize_aug_detokenize(
            "beethoven_moonlight.mid", velocity_aug_fn, "velocity"
        )

        # Tempo augmentation
        seq_tempo_augmented = tempo_aug_fn(get_short_seq_abs(tknzr))
        logging.info(f"tempo_aug_fn:\n{seq} ->\n\n{seq_tempo_augmented}\n")

        seq_concat_tempo_augmented = tempo_aug_fn(get_concat_seq_abs(tknzr))
        logging.info(
            f"tempo_aug_fn:\n{seq_concat} ->\n\n{seq_concat_tempo_augmented}\n"
        )

        tokenize_aug_detokenize("basic.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize("arabesque.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize("beethoven_sonata.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize("bach.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize("expressive.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize("pop.mid", tempo_aug_fn, "tempo")
        tokenize_aug_detokenize(
            "beethoven_moonlight.mid", tempo_aug_fn, "tempo"
        )

    def test_aug_time(self):
        tknzr = tokenizer.AbsTokenizer()
        mid_dict = MidiDict.from_midi("tests/test_data/beethoven_sonata.mid")
        tokenized_seq = tknzr.tokenize(mid_dict)[:4096]
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)
        tempo_aug_fn = tknzr.export_tempo_aug(tempo_aug_range=0.5, mixup=True)

        # Pitch augmentation
        t_start = time.perf_counter()
        pitch_aug_fn(tokenized_seq)
        t_pitch_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"pitch_aug_fn took {int(t_pitch_aug)}ms")
        self.assertLessEqual(t_pitch_aug, 50)

        # Velocity augmentation
        t_start = time.perf_counter()
        velocity_aug_fn(tokenized_seq)
        t_vel_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"velocity_aug_fn took {int(t_vel_aug)}ms")
        self.assertLessEqual(t_vel_aug, 50)

        # Tempo augmentation
        t_start = time.perf_counter()
        tempo_aug_fn(tokenized_seq)
        t_tempo_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"tempo_aug_fn took {int(t_tempo_aug)}ms")
        self.assertLessEqual(t_tempo_aug, 50)

    def test_no_unk_token(self):
        def _test_no_unk_token(file_name: str):
            mid_path = f"tests/test_data/{file_name}"
            midi_dict = MidiDict.from_midi(mid_path=mid_path)
            seq = tknzr.tokenize(midi_dict)
            enc_dec_seq = tknzr.decode(tknzr.encode(seq))
            for tok in enc_dec_seq:
                self.assertTrue(tok != tknzr.unk_tok)

        tknzr = tokenizer.AbsTokenizer()
        _test_no_unk_token("basic.mid")
        _test_no_unk_token("arabesque.mid")
        _test_no_unk_token("bach.mid")
        _test_no_unk_token("expressive.mid")
        _test_no_unk_token("pop.mid")
        _test_no_unk_token("beethoven_moonlight.mid")


# TODO: This example is not working, I'm pretty sure the issue is in _get_combined_mididict somewhere
# Fix this!!
class TestSeparatedTokenizer(unittest.TestCase):
    def test_tokenize_detokenize_mididict(self):
        def _find_inst_onsets(_seq: list):
            curr_time_ms = 0
            time_toks = 0
            for tok in _seq:
                if tok == "<T>":
                    time_toks += 1
                elif isinstance(tok, tuple) and tok[0] == "onset":
                    curr_time_ms = 5000 * time_toks + tok[1]
                elif tok == "<INST>":
                    print("Seen at", curr_time_ms)

        tknzr = tokenizer.SeparatedAbsTokenizer()

        clean_midi_dict = MidiDict.from_midi(
            mid_path="/mnt/ssd1/data/mp3/raw/maestro-mp3/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
        )
        noisy_midi_dict = MidiDict.from_midi(
            mid_path="/mnt/ssd1/data/mp3/raw/maestro-mp3/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
            # mid_path="/mnt/ssd1/amt/transcribed_data/noisy_maestro/small-long-e7/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.mid"
        )

        noisy_midi_dict = _noise_midi_dict(
            noisy_midi_dict, load_config()["data"]["finetuning"]["noising"]
        )

        clean_mid = clean_midi_dict.to_midi()
        clean_mid.save(f"tests/test_results/combined_clean.mid")
        noisy_mid = noisy_midi_dict.to_midi()
        noisy_mid.save(f"tests/test_results/combined_noisy.mid")

        comb_midi_dict = _get_combined_mididict(
            clean_midi_dict,
            noisy_midi_dict,
            min_noisy_ms=10000,
            max_noisy_ms=25000,
            min_clean_ms=30000,
            max_clean_ms=60000,
        )

        comb_midi = comb_midi_dict.to_midi()
        comb_midi.save(f"tests/test_results/combined_raw.mid")
        tokenized_seq = tknzr.tokenize(comb_midi_dict)
        detokenized_midi_dict = tknzr.detokenize(tokenized_seq)
        res = detokenized_midi_dict.to_midi()
        res.save(f"tests/test_results/combined.mid")

        for idx, sub_seq in enumerate(tknzr.split(tokenized_seq, 4096)):
            if idx == 3:
                _find_inst_onsets(sub_seq)
                print(idx)
                print(sub_seq)
            detokenized_midi_dict = tknzr.detokenize(sub_seq)
            res = detokenized_midi_dict.to_midi()
            res.save(f"tests/test_results/combined{idx}.mid")


class TestRelTokenizer(unittest.TestCase):
    def test_tokenize_detokenize_mididict(self):
        def tokenize_detokenize(file_name: str):
            mid_path = f"tests/test_data/{file_name}"
            midi_dict = MidiDict.from_midi(mid_path=mid_path)
            tokenized_seq = tknzr.tokenize(midi_dict)
            detokenized_midi_dict = tknzr.detokenize(tokenized_seq)
            res = detokenized_midi_dict.to_midi()
            res.save(f"tests/test_results/{file_name}")

        tknzr = tokenizer.RelTokenizer(return_tensors=False)

        tokenize_detokenize("basic.mid")
        tokenize_detokenize("arabesque.mid")
        tokenize_detokenize("beethoven_sonata.mid")
        tokenize_detokenize("bach.mid")
        tokenize_detokenize("expressive.mid")
        tokenize_detokenize("pop.mid")
        tokenize_detokenize("beethoven_moonlight.mid")

    def test_aug(self):
        tknzr = tokenizer.RelTokenizer(return_tensors=False)
        seq = get_short_seq_rel(tknzr)
        seq_concat = get_concat_seq_rel(tknzr)
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)
        tempo_aug_fn = tknzr.export_tempo_aug(tempo_aug_range=0.8)
        chord_mixup_fn = tknzr.export_chord_mixup()

        # Pitch augmentation
        seq_pitch_augmented = pitch_aug_fn(get_short_seq_rel(tknzr))
        logging.info(f"pitch_aug_fn:\n{seq} ->\n\n{seq_pitch_augmented}\n")
        self.assertEqual(
            seq_pitch_augmented[4][1] - seq[4][1],
            seq_pitch_augmented[8][1] - seq[8][1],
        )

        # Velocity augmentation
        seq_velocity_augmented = velocity_aug_fn(get_short_seq_rel(tknzr))
        logging.info(
            f"velocity_aug_fn:\n{seq} ->\n\n{seq_velocity_augmented}\n"
        )
        self.assertEqual(
            seq_velocity_augmented[4][2] - seq[4][2],
            seq_velocity_augmented[8][2] - seq[8][2],
        )

        # Tempo augmentation
        seq_tempo_augmented = tempo_aug_fn(get_short_seq_rel(tknzr))
        logging.info(f"tempo_aug_fn:\n{seq} ->\n\n{seq_tempo_augmented}\n")

        seq_concat_tempo_augmented = tempo_aug_fn(get_concat_seq_rel(tknzr))
        logging.info(
            f"tempo_aug_fn:\n{seq_concat} ->\n\n{seq_concat_tempo_augmented}\n"
        )

        # Chord mix-up augmentation
        seq_mixup_augmented = chord_mixup_fn(get_short_seq_rel(tknzr))
        logging.info(f"chord_mixup_fn:\n{seq} ->\n\n{seq_mixup_augmented}\n")

        seq_concat_tempo_augmented = chord_mixup_fn(get_concat_seq_rel(tknzr))
        logging.info(
            f"chord_mixup_fn:\n{seq_concat} ->\n\n{seq_concat_tempo_augmented}\n"
        )

    def test_aug_time(self):
        tknzr = tokenizer.RelTokenizer()
        mid_dict = MidiDict.from_midi("tests/test_data/beethoven_sonata.mid")
        tokenized_seq = tknzr.tokenize(mid_dict)[:4096]

        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)
        tempo_aug_fn = tknzr.export_tempo_aug(tempo_aug_range=0.5)
        chord_mixup_fn = tknzr.export_chord_mixup()

        # Pitch augmentation
        t_start = time.perf_counter()
        pitch_aug_fn(tokenized_seq)
        t_pitch_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"pitch_aug_fn took {int(t_pitch_aug)}ms")
        self.assertLessEqual(t_pitch_aug, 50)

        # Velocity augmentation
        t_start = time.perf_counter()
        velocity_aug_fn(tokenized_seq)
        t_vel_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"velocity_aug_fn took {int(t_vel_aug)}ms")
        self.assertLessEqual(t_vel_aug, 50)

        # Tempo augmentation
        t_start = time.perf_counter()
        tempo_aug_fn(tokenized_seq)
        t_tempo_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"tempo_aug_fn took {int(t_tempo_aug)}ms")
        self.assertLessEqual(t_tempo_aug, 50)

        # Chord mixup augmentation
        t_start = time.perf_counter()
        chord_mixup_fn(tokenized_seq)
        t_mixup_aug = (time.perf_counter() - t_start) * 1e3
        logging.info(f"mixup_aug_fn took {int(t_mixup_aug)}ms")
        self.assertLessEqual(t_mixup_aug, 50)

    def test_encode_decode(self):
        tknzr = tokenizer.RelTokenizer(return_tensors=True)
        seq = get_short_seq_rel(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)

        tknzr = tokenizer.RelTokenizer(return_tensors=False)
        seq = get_short_seq_rel(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)

    def test_no_unk_token(self):
        tknzr = tokenizer.RelTokenizer()
        seq = get_short_seq_rel(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for tok in enc_dec_seq:
            self.assertTrue(tok != tknzr.unk_tok)


if __name__ == "__main__":
    if os.path.isdir("tests/test_results") is False:
        os.mkdir("tests/test_results")

    logging.basicConfig(level=logging.INFO)
    unittest.main()
