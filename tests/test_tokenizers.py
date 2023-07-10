import unittest
import logging

from aria import tokenizer


def get_short_seq():
    return [
        "piano",
        "drums",
        "<S>",
        ("piano", 62, 50),
        ("dur", 50),
        ("wait", 100),
        ("drum", 50),
        ("piano", 64, 50),
        ("dur", 100),
        ("wait", 100),
        "<E>",
    ]


class TestLazyTokenizer(unittest.TestCase):
    def test_pitch_aug(self):
        tknzr = tokenizer.TokenizerLazy(
            padding=False,
            truncate_type="none",
            max_seq_len=512,
            return_tensors=False,
        )
        pitch_aug_fn = tknzr.export_pitch_aug(aug_range=5)
        seq = get_short_seq()
        res = pitch_aug_fn(get_short_seq())

        print(f"pitch_aug_fn:\n{seq} ->\n{res}")
        logging.info(f"pitch_aug_fn: {seq} -> {res}")

        self.assertEqual(res[3][1] - seq[3][1], res[7][1] - seq[7][1])


if __name__ == "__main__":
    unittest.main()
