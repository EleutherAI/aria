import unittest
import os
import logging

from aria import tokenizer
from aria.data import datasets
from aria.data.midi import MidiDict
from aria.data import jsonl_zst

if not os.path.isdir("tests/test_results"):
    os.makedirs("tests/test_results")


def get_short_seq():
    return [
        ("prefix", "instrument", "piano"),
        ("prefix", "instrument", "drum"),
        ("prefix", "composer", "bach"),
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

        self.assertEqual(len(dataset), 6)
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
        self.assertEqual(len(dataset_reloaded), 6)
        self.assertEqual(type(dataset[0]), type(dataset_reloaded[0]))

    def test_build_to_file(self):
        datasets.MidiDataset.build_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset_direct.jsonl",
            recur=True,
            overwrite=True,
        )

        dataset_reloaded = datasets.MidiDataset.load(
            load_path="tests/test_results/mididict_dataset_direct.jsonl",
        )
        self.assertEqual(len(dataset_reloaded), 6)
        self.assertEqual(type(dataset_reloaded[0]), MidiDict)

    def test_split_from_file(self):
        datasets.MidiDataset.build_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset.jsonl",
            recur=True,
            overwrite=True,
        )

        datasets.MidiDataset.split_from_file(
            load_path="tests/test_results/mididict_dataset.jsonl",
            train_val_ratio=0.7,
            repeatable=True,
            overwrite=True,
        )

        self.assertTrue(
            os.path.isfile("tests/test_results/mididict_dataset_train.jsonl")
        )
        self.assertTrue(
            os.path.isfile("tests/test_results/mididict_dataset_val.jsonl")
        )

    def test_data_hash(self):
        mid_1 = MidiDict.from_midi("tests/test_data/pop.mid")
        mid_2 = MidiDict.from_midi("tests/test_data/pop_copy.mid")

        self.assertEqual(mid_1.calculate_hash(), mid_2.calculate_hash())


class TestTokenizedDataset(unittest.TestCase):
    # Test building is working (on the file level)
    def test_build(self):
        MAX_SEQ_LEN = 512
        tknzr = tokenizer.TokenizerLazy(
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
            max_seq_len=MAX_SEQ_LEN,
            overwrite=True,
        )
        dataset_buffer_from_mdset = datasets.TokenizedDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer_2.jsonl",
            midi_dataset=mididict_dataset,
            max_seq_len=MAX_SEQ_LEN,
            overwrite=True,
        )

        with (
            open("tests/test_results/dataset_buffer_1.jsonl") as buff1,
            open("tests/test_results/dataset_buffer_1.jsonl") as buff2,
        ):
            buff1_lines = buff1.readlines()
            buff2_lines = buff2.readlines()

            for l1, l2 in zip(buff1_lines, buff2_lines):
                for tok1, tok2 in zip(l1, l2):
                    if tok1 == "<D>" or tok2 == "<D>":
                        break  # <D> is randomly inserted due to multiprocessing
                    else:
                        self.assertEqual(tuple(tok1), tuple(tok2))

        self.assertEqual(
            sum(1 for _ in dataset_buffer_from_file.file_buff),
            len(dataset_buffer_from_file) + 1,
        )

        dataset_buffer_from_file.close()
        dataset_buffer_from_mdset.close()

    def test_mmap(self):
        MAX_SEQ_LEN = 512
        tknzr = tokenizer.TokenizerLazy(
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
            max_seq_len=MAX_SEQ_LEN,
            overwrite=True,
        )

        raw_entries = [src for src, tgt in tokenized_dataset]
        self.assertEqual(len({len(_) for _ in raw_entries}), 1)

        src, tgt = tokenized_dataset[0]
        logging.info(f"src: {tknzr.decode(src)}")
        logging.info(f"tgt: {tknzr.decode(tgt)}")

        tokenized_dataset.close()

    def test_shuffle(self):
        MAX_SEQ_LEN = 512
        tknzr = tokenizer.TokenizerLazy(
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
            max_seq_len=MAX_SEQ_LEN,
            overwrite=True,
        )
        tokenized_dataset_shuffled = tokenized_dataset.get_shuffled_dataset(
            repeatable=True,
            overwrite=True,
        )

        self.assertEqual(
            len(tokenized_dataset), len(tokenized_dataset_shuffled)
        )
        self.assertEqual(
            len(tokenized_dataset), len(tokenized_dataset_shuffled)
        )

        same_order = True
        for seq1, seq2 in zip(tokenized_dataset, tokenized_dataset_shuffled):
            for x, y in zip(seq1, seq2):
                if x != y:
                    same_order = False

        self.assertFalse(same_order)

    def test_augmentation(self):
        MAX_SEQ_LEN = 512
        tknzr = tokenizer.TokenizerLazy(
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
            max_seq_len=MAX_SEQ_LEN,
            overwrite=True,
        )
        tokenized_dataset.set_transform(
            [
                tknzr.export_pitch_aug(5),
                tknzr.export_velocity_aug(2),
                tknzr.export_tempo_aug(0.5),
            ]
        )

        seq = get_short_seq()
        seq_augmented = tokenized_dataset._transform(seq)

        logging.info(f"aug:\n{seq} ->\n{seq_augmented}")
        self.assertEqual(
            seq_augmented[4][1] - seq[4][1],
            seq_augmented[8][1] - seq[8][1],
        )
        self.assertEqual(
            seq_augmented[4][2] - seq[4][2],
            seq_augmented[8][2] - seq[8][2],
        )

        tokenized_dataset.close()


class TestReaderWriter(unittest.TestCase):
    def test_jsonl_zst(self):
        data = [{"a": i, "b": i+1} for i in range(0, 100, 4)]
        filename = "tests/test_results/test.jsonl.zst"
        # if test.jsonl.zst exists, delete it
        if os.path.isfile(filename):
            os.remove(filename)
        with jsonl_zst.open(filename, "w") as f:
            for d in data:
                f.write(d)
        with jsonl_zst.open(filename, "r") as f:
            for d, d2 in zip(data, f):
                self.assertEqual(d, d2)
        # Remove the file
        os.remove(filename)


if __name__ == "__main__":
    if os.path.isdir("tests/test_results") is False:
        os.mkdir("tests/test_results")

    logging.basicConfig(level=logging.INFO)
    unittest.main()
