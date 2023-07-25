import unittest
import logging
import os
import mido

from aria import tokenizer
from aria.data.midi import MidiDict


def get_short_seq(tknzr: tokenizer.TokenizerLazy):
    return [
        "piano",
        "drums",
        "<S>",
        ("piano", 62, tknzr._quantize_velocity(50)),
        ("dur", tknzr._quantize_time(50)),
        ("wait", tknzr._quantize_time(100)),
        ("drum", tknzr._quantize_time(50)),
        ("piano", 64, tknzr._quantize_velocity(70)),
        ("dur", tknzr._quantize_time(100)),
        ("wait", tknzr._quantize_time(100)),
        "<E>",
    ]


class TestLazyTokenizer(unittest.TestCase):
    # Add encode decode test
    def test_tokenize_detokenize_mididict(self):
        def tokenize_detokenize(file_name: str):
            mid = mido.MidiFile(f"tests/test_data/{file_name}")
            midi_dict = MidiDict.from_midi(mid)
            tokenized_seq = tknzr.tokenize_midi_dict(midi_dict)
            detokenized_midi_dict = tknzr.detokenize_midi_dict(tokenized_seq)
            res = detokenized_midi_dict.to_midi()
            res.save(f"tests/test_results/{file_name}")

        tknzr = tokenizer.TokenizerLazy(
            max_seq_len=512,
            return_tensors=False,
        )

        tokenize_detokenize("basic.mid")
        tokenize_detokenize("arabesque.mid")
        tokenize_detokenize("beethoven.mid")
        tokenize_detokenize("bach.mid")
        tokenize_detokenize("expressive.mid")
        tokenize_detokenize("pop.mid")

    def test_aug(self):
        tknzr = tokenizer.TokenizerLazy(
            max_seq_len=512,
            return_tensors=False,
        )
        seq = get_short_seq(tknzr)
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        velocity_aug_fn = tknzr.export_velocity_aug(aug_steps_range=2)

        seq_pitch_augmented = pitch_aug_fn(get_short_seq(tknzr))
        logging.info(f"pitch_aug_fn:\n{seq} ->\n{seq_pitch_augmented}")
        self.assertEqual(
            seq_pitch_augmented[3][1] - seq[3][1],
            seq_pitch_augmented[7][1] - seq[7][1],
        )

        seq_velocity_augmented = velocity_aug_fn(get_short_seq(tknzr))
        logging.info(f"velocity_aug_fn:\n{seq} ->\n{seq_velocity_augmented}")
        self.assertEqual(
            seq_velocity_augmented[3][2] - seq[3][2],
            seq_velocity_augmented[7][2] - seq[7][2],
        )

    def test_encode_decode(self):
        tknzr = tokenizer.TokenizerLazy(
            max_seq_len=512,
            return_tensors=True,
        )
        seq = get_short_seq(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)

        tknzr = tokenizer.TokenizerLazy(
            max_seq_len=512,
            return_tensors=False,
        )
        seq = get_short_seq(tknzr)
        enc_dec_seq = tknzr.decode(tknzr.encode(seq))
        for x, y in zip(seq, enc_dec_seq):
            self.assertEqual(x, y)


if __name__ == "__main__":
    if os.path.isdir("tests/test_results") is False:
        os.mkdir("tests/test_results")

    logging.basicConfig(level=logging.INFO)
    unittest.main()
