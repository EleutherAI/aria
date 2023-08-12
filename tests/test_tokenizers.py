import unittest
import logging
import os
import time

from aria import tokenizer
from aria.data.midi import MidiDict


# TODO: Add test which reports timings for the different augmentation functions
# on the Beethoven sonata with len 2k and 4k.


def get_short_seq(tknzr: tokenizer.TokenizerLazy):
    return [
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
    ]


class TestLazyTokenizer(unittest.TestCase):
    # Add encode decode test
    def test_tokenize_detokenize_mididict(self):
        def tokenize_detokenize(file_name: str):
            mid_path = f"tests/test_data/{file_name}"
            midi_dict = MidiDict.from_midi(mid_path=mid_path)
            tokenized_seq = tknzr.tokenize_midi_dict(midi_dict)
            detokenized_midi_dict = tknzr.detokenize_midi_dict(tokenized_seq)
            res = detokenized_midi_dict.to_midi()
            res.save(f"tests/test_results/{file_name}")

        tknzr = tokenizer.TokenizerLazy(return_tensors=False)

        tokenize_detokenize("basic.mid")
        tokenize_detokenize("arabesque.mid")
        tokenize_detokenize("beethoven.mid")
        tokenize_detokenize("bach.mid")
        tokenize_detokenize("expressive.mid")
        tokenize_detokenize("pop.mid")

    def test_aug(self):
        tknzr = tokenizer.TokenizerLazy(return_tensors=False)
        seq = get_short_seq(tknzr)
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)
        tempo_aug_fn = tknzr.export_tempo_aug(tempo_aug_range=0.5)
        chord_mixup_fn = tknzr.export_chord_mixup()

        # Pitch augmentation
        seq_pitch_augmented = pitch_aug_fn(get_short_seq(tknzr))
        logging.info(f"pitch_aug_fn:\n{seq} ->\n\n{seq_pitch_augmented}\n")
        self.assertEqual(
            seq_pitch_augmented[4][1] - seq[4][1],
            seq_pitch_augmented[8][1] - seq[8][1],
        )

        # Velocity augmentation
        seq_velocity_augmented = velocity_aug_fn(get_short_seq(tknzr))
        logging.info(
            f"velocity_aug_fn:\n{seq} ->\n\n{seq_velocity_augmented}\n"
        )
        self.assertEqual(
            seq_velocity_augmented[4][2] - seq[4][2],
            seq_velocity_augmented[8][2] - seq[8][2],
        )

        # Tempo augmentation
        seq_tempo_augmented = tempo_aug_fn(get_short_seq(tknzr))
        logging.info(f"tempo_aug_fn:\n{seq} ->\n\n{seq_tempo_augmented}\n")

        # Chord mix-up augmentation
        seq_mixup_augmented = chord_mixup_fn(get_short_seq(tknzr))
        logging.info(f"chord_mixup_fn:\n{seq} ->\n\n{seq_mixup_augmented}\n")

    def test_aug_time(self):
        tknzr = tokenizer.TokenizerLazy()
        mid_dict = MidiDict.from_midi("tests/test_data/beethoven.mid")
        tokenized_seq = tknzr.tokenize_midi_dict(mid_dict)[:4096]

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
        tknzr = tokenizer.TokenizerLazy(return_tensors=True)
        seq = get_short_seq(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)

        tknzr = tokenizer.TokenizerLazy(return_tensors=False)
        seq = get_short_seq(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)

    def test_no_unk_token(self):
        tknzr = tokenizer.TokenizerLazy()
        seq = get_short_seq(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for tok in enc_dec_seq:
            self.assertTrue(tok != tknzr.unk_tok)


if __name__ == "__main__":
    if os.path.isdir("tests/test_results") is False:
        os.mkdir("tests/test_results")

    logging.basicConfig(level=logging.INFO)
    unittest.main()
