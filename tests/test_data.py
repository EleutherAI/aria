import unittest
import os
import shutil
import logging

from aria import tokenizer
from aria.data import datasets
from aria.data.midi import MidiDict

TEST_TOKENIZER = "abs"
logger = logging.getLogger(__name__)
if not os.path.isdir("tests/test_results"):
    os.makedirs("tests/test_results")


def setup_logger():
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] tests.test_data: [%(levelname)s] %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


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

    def test_concat(self):
        if (
            os.path.exists("tests/test_results/mididict_dataset_train.jsonl")
            and os.path.exists("tests/test_results/mididict_dataset_val.jsonl")
            and os.path.exists("tests/test_results/mididict_dataset.jsonl")
        ):
            datasets.MidiDataset.combine_datasets_from_file(
                "tests/test_results/mididict_dataset_train.jsonl",
                "tests/test_results/mididict_dataset_val.jsonl",
                "tests/test_results/mididict_dataset.jsonl",
                output_path="tests/test_results/mididict_dataset_concat.jsonl",
            )

            self.assertAlmostEqual(
                len(
                    datasets.MidiDataset.load(
                        "tests/test_results/mididict_dataset_concat.jsonl"
                    )
                ),
                len(
                    datasets.MidiDataset.load(
                        "tests/test_results/mididict_dataset.jsonl"
                    )
                ),
            )


class TestPretrainingDataset(unittest.TestCase):
    def test_build(self):
        MAX_SEQ_LEN = 512
        if TEST_TOKENIZER == "abs":
            tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tknzr = tokenizer.RelTokenizer(return_tensors=False)
        else:
            raise KeyError
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        mididict_dataset.save("tests/test_results/mididict_dataset.jsonl")

        if os.path.exists("tests/test_results/pretrain_dataset_buff_1"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff_1")
        if os.path.exists("tests/test_results/pretrain_dataset_buff_2"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff_2")

        dataset_from_file = datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff_1",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=3,
            midi_dataset_path="tests/test_results/mididict_dataset.jsonl",
        )
        dataset_from_mdset = datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff_2",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=3,
            midi_dataset=mididict_dataset,
        )

    def test_mmap(self):
        MAX_SEQ_LEN = 512
        if TEST_TOKENIZER == "abs":
            tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tknzr = tokenizer.RelTokenizer(return_tensors=False)
        else:
            raise KeyError
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        if os.path.exists("tests/test_results/pretrain_dataset_buff"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff")
        pretrain_dataset = datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=1,
            midi_dataset=mididict_dataset,
        )

        raw_entries = [src for src, tgt in pretrain_dataset]
        self.assertEqual(len({len(_) for _ in raw_entries}), 1)

        src, tgt = pretrain_dataset[0]
        logger.info(f"src: {tknzr.decode(src)[:50]}")
        logger.info(f"tgt: {tknzr.decode(tgt)[:50]}")

    def test_aug(self):
        MAX_SEQ_LEN = 512
        if TEST_TOKENIZER == "abs":
            tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tknzr = tokenizer.RelTokenizer(return_tensors=False)
        else:
            raise KeyError
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        if os.path.exists("tests/test_results/pretrain_dataset_buff"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff")
        pretrain_dataset = datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=1,
            midi_dataset=mididict_dataset,
        )
        pretrain_dataset.set_transform(tknzr.export_data_aug())
        for idx, seq in enumerate(tknzr.decode(pretrain_dataset[0][0])):
            for _idx, tok in enumerate(seq):
                if tok == tknzr.unk_tok:
                    logger.warning(f"unk_tok seen at seq={idx}, idx={_idx}")

        logger.info(f"data_aug_1: {tknzr.decode(pretrain_dataset[0][0][:50])}")
        logger.info(f"data_aug_2: {tknzr.decode(pretrain_dataset[0][0][:50])}")


class TestFinetuningDataset(unittest.TestCase):
    # Test building is working (on the file level)
    def test_build(self):
        MAX_SEQ_LEN = 512
        STRIDE_LEN = 256
        if TEST_TOKENIZER == "abs":
            tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tknzr = tokenizer.RelTokenizer(return_tensors=False)
        else:
            raise KeyError
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        mididict_dataset.save("tests/test_results/mididict_dataset.jsonl")

        if os.path.exists("tests/test_results/dataset_buffer_1.jsonl"):
            os.remove("tests/test_results/dataset_buffer_1.jsonl")
        finetune_dataset_from_file = datasets.FinetuningDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer_1.jsonl",
            midi_dataset_path="tests/test_results/mididict_dataset.jsonl",
            max_seq_len=MAX_SEQ_LEN,
            stride_len=STRIDE_LEN,
        )

        if os.path.exists("tests/test_results/dataset_buffer_2.jsonl"):
            os.remove("tests/test_results/dataset_buffer_2.jsonl")
        finetune_dataset_from_mdset = datasets.FinetuningDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/dataset_buffer_2.jsonl",
            midi_dataset=mididict_dataset,
            max_seq_len=MAX_SEQ_LEN,
            stride_len=STRIDE_LEN,
        )

        raw_entries = [src for src, tgt in finetune_dataset_from_file]
        self.assertEqual(len({len(_) for _ in raw_entries}), 1)
        raw_entries = [src for src, tgt in finetune_dataset_from_mdset]
        self.assertEqual(len({len(_) for _ in raw_entries}), 1)

        src, tgt = finetune_dataset_from_file[0]
        logger.info(f"src: {tknzr.decode(src)[:50]}")
        logger.info(f"tgt: {tknzr.decode(tgt)[:50]}")

    def test_aug(self):
        MAX_SEQ_LEN = 512
        STRIDE_LEN = 256
        if TEST_TOKENIZER == "abs":
            tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tknzr = tokenizer.RelTokenizer(return_tensors=False)
        else:
            raise KeyError
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=True,
        )
        if os.path.isfile("tests/test_results/finetune_dataset_buff.jsonl"):
            os.remove("tests/test_results/finetune_dataset_buff.jsonl")
        finetune_dataset = datasets.FinetuningDataset.build(
            tokenizer=tknzr,
            save_path="tests/test_results/finetune_dataset_buff.jsonl",
            max_seq_len=MAX_SEQ_LEN,
            stride_len=STRIDE_LEN,
            midi_dataset=mididict_dataset,
        )
        finetune_dataset.set_transform(tknzr.export_data_aug())
        for idx, seq in enumerate(tknzr.decode(finetune_dataset[0][0])):
            for _idx, tok in enumerate(seq):
                if tok == tknzr.unk_tok:
                    logger.warning(f"unk_tok seen at seq={idx}, idx={_idx}")

        logger.info(f"data_aug_1: {tknzr.decode(finetune_dataset[0][0][:50])}")
        logger.info(f"data_aug_2: {tknzr.decode(finetune_dataset[0][0][:50])}")


setup_logger()
if __name__ == "__main__":
    unittest.main()
