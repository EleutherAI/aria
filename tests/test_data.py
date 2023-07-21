import unittest
import logging
import filecmp

from aria import tokenizer
from aria.data import datasets
from aria.data.midi import MidiDict


def get_short_seq():
    return [
        "piano",
        "drums",
        "<S>",
        ("piano", 62, 50),
        ("dur", 50),
        ("wait", 100),
        ("drum", 50),
        ("piano", 64, 70),
        ("dur", 100),
        ("wait", 100),
        "<E>",
    ]


class TestMidiDataset(unittest.TestCase):
    def test_build(self):
        dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )

        self.assertEqual(len(dataset), 3)
        self.assertEqual(type(dataset[0]), MidiDict)

    def test_save_load(self):
        dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        dataset.save("tests/test_results/mididict_dataset.jsonl")

        dataset_reloaded = datasets.MidiDataset.load(
            "tests/test_results/mididict_dataset.jsonl"
        )
        self.assertEqual(len(dataset_reloaded), 3)
        self.assertEqual(type(dataset[0]), type(dataset_reloaded[0]))

    def test_build_to_file(self):
        dataset = datasets.MidiDataset.build_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset_direct.jsonl",
            recur=True,
            overwrite=True,
        )

        dataset_reloaded = datasets.MidiDataset.load(
            load_path="tests/test_results/mididict_dataset_direct.jsonl",
        )
        self.assertEqual(len(dataset_reloaded), 3)
        self.assertEqual(type(dataset_reloaded[0]), MidiDict)


class TestTokenizedDataset(unittest.TestCase):
    # Test building is working (on the file level)
    def test_build(self):
        tknzr = tokenizer.TokenizerLazy(
            padding=True,
            truncate_type="default",
            max_seq_len=64,
            return_tensors=False,
        )
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        mididict_dataset.save("tests/test_results/mididict_dataset.jsonl")

        dataset_buffer_from_file = datasets.TokenizedDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer_1.jsonl",
            midi_dataset_path="tests/test_results/mididict_dataset.jsonl",
            overwrite=True,
        )
        dataset_buffer_from_mdset = datasets.TokenizedDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer_2.jsonl",
            midi_dataset=mididict_dataset,
            overwrite=True,
        )

        self.assertTrue(
            filecmp.cmp(
                "tests/test_results/dataset_buffer_1.jsonl",
                "tests/test_results/dataset_buffer_2.jsonl",
            )
        )
        self.assertEqual(
            sum(1 for _ in dataset_buffer_from_file.file_buff),
            len(dataset_buffer_from_file) + 1,
        )
        self.assertEqual(len(dataset_buffer_from_file), len(mididict_dataset))

        dataset_buffer_from_file.close()
        dataset_buffer_from_mdset.close()

    def test_mmap(self):
        tknzr = tokenizer.TokenizerLazy(
            padding=True,
            truncate_type="default",
            max_seq_len=200,
            return_tensors=False,
        )
        midi_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        tokenized_dataset = datasets.TokenizedDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer.jsonl",
            midi_dataset=midi_dataset,
            overwrite=True,
        )

        raw_entries = [src for src, tgt in tokenized_dataset]
        self.assertEqual(len(raw_entries), len(midi_dataset))
        self.assertEqual(len({len(_) for _ in raw_entries}), 1)

        src, tgt = tokenized_dataset[0]
        logging.info(f"src: {tknzr.decode(src)}")
        logging.info(f"tgt: {tknzr.decode(tgt)}")

        tokenized_dataset.close()

    def test_augmentation(self):
        tknzr = tokenizer.TokenizerLazy(
            padding=True,
            truncate_type="default",
            max_seq_len=200,
            return_tensors=False,
        )
        midi_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        tokenized_dataset = datasets.TokenizedDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer.jsonl",
            midi_dataset=midi_dataset,
            overwrite=True,
        )
        tokenized_dataset.set_transform(
            [tknzr.export_velocity_aug(2), tknzr.export_pitch_aug(5)]
        )

        seq = get_short_seq()
        seq_augmented = tokenized_dataset._transform(seq)

        logging.info(f"aug:\n{seq} ->\n{seq_augmented}")
        self.assertEqual(
            seq_augmented[3][1] - seq[3][1],
            seq_augmented[7][1] - seq[7][1],
        )
        self.assertEqual(
            seq_augmented[3][2] - seq[3][2],
            seq_augmented[7][2] - seq[7][2],
        )

        tokenized_dataset.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
