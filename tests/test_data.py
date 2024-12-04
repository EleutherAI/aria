import unittest
import os
import shutil
import logging

from aria import tokenizer
from aria.config import load_config
from aria.data import datasets
from aria.data.datasets import _noise_midi_dict
from ariautils.midi import MidiDict

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


class TestMidiDict(unittest.TestCase):
    def test_resolve_pedal(self):
        midi_dict = MidiDict.from_midi("tests/test_data/maestro.mid")
        midi_dict.resolve_pedal()
        self.assertListEqual(midi_dict.pedal_msgs, [])
        mid = midi_dict.to_midi()
        mid.save("tests/test_results/maestro_npedal.mid")


class TestMidiDataset(unittest.TestCase):
    def test_build(self):
        dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=False,
        )

        self.assertEqual(len(dataset), 7)
        self.assertEqual(type(dataset[0]), MidiDict)

    def test_save_load(self):
        dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=False,
        )
        dataset.save("tests/test_results/mididict_dataset.jsonl")

        dataset_reloaded = datasets.MidiDataset.load(
            "tests/test_results/mididict_dataset.jsonl"
        )
        self.assertEqual(len(dataset_reloaded), 7)
        self.assertEqual(type(dataset[0]), type(dataset_reloaded[0]))

    def test_build_to_file(self):
        datasets.MidiDataset.build_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset_direct.jsonl",
            recur=False,
            overwrite=True,
        )

        dataset_reloaded = datasets.MidiDataset.load(
            load_path="tests/test_results/mididict_dataset_direct.jsonl",
        )
        self.assertEqual(len(dataset_reloaded), 7)
        self.assertEqual(type(dataset_reloaded[0]), MidiDict)

    def test_split_from_file(self):
        datasets.MidiDataset.build_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset.jsonl",
            recur=False,
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
        MAX_SEQ_LEN = 4096
        tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=False,
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
            num_epochs=2,
            midi_dataset=mididict_dataset,
        )

    def test_multiple_paths(self):
        MAX_SEQ_LEN = 4096
        tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=False,
        )
        mididict_dataset.save("tests/test_results/mididict_dataset_1.jsonl")

        if os.path.exists("tests/test_results/pretrain_dataset_buff_1"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff_1")
        if os.path.exists("tests/test_results/pretrain_dataset_buff_2"):
            shutil.rmtree("tests/test_results/pretrain_dataset_buff_2")

        datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff_1",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=3,
            midi_dataset_path="tests/test_results/mididict_dataset.jsonl",
        )
        datasets.PretrainingDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/pretrain_dataset_buff_2",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=5,
            midi_dataset_path="tests/test_results/mididict_dataset.jsonl",
        )

        dataset = datasets.PretrainingDataset(
            dir_paths=[
                "tests/test_results/pretrain_dataset_buff_1",
                "tests/test_results/pretrain_dataset_buff_2",
            ],
            tokenizer=tknzr,
        )

        for epoch in range(11):
            for idx, _ in enumerate(dataset):
                pass

            print("-------------")
            dataset.init_epoch()

    def test_aug(self):
        MAX_SEQ_LEN = 512
        tknzr = tokenizer.AbsTokenizer(return_tensors=False)
        mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data",
            recur=False,
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
    def test_noise(self):
        config = load_config()["data"]["finetuning"]["noising"]
        midi_dict = MidiDict.from_midi("tests/test_data/clean/1.mid")
        noisy_midi_dict = _noise_midi_dict(midi_dict, config)
        noisy_midi = noisy_midi_dict.to_midi()
        noisy_midi.save("tests/test_results/noisy.mid")

    def test_build(self):
        MAX_SEQ_LEN = 4096
        tknzr = tokenizer.SeparatedAbsTokenizer(return_tensors=False)
        clean_mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data/clean",
            recur=True,
            shuffle=False,
        )
        noisy_mididict_dataset = datasets.MidiDataset.build(
            dir="tests/test_data/noisy",
            recur=True,
            shuffle=False,
        )
        if os.path.exists("tests/test_results/clean.jsonl"):
            os.remove("tests/test_results/clean.jsonl")
        if os.path.exists("tests/test_results/noisy.jsonl"):
            os.remove("tests/test_results/noisy.jsonl")
        clean_mididict_dataset.save("tests/test_results/clean.jsonl")
        noisy_mididict_dataset.save("tests/test_results/noisy.jsonl")

        if os.path.exists("tests/test_results/comb"):
            shutil.rmtree("tests/test_results/comb")

        finetuning_dataset = datasets.FinetuningDataset.build(
            tokenizer=tknzr,
            save_dir="tests/test_results/comb",
            max_seq_len=MAX_SEQ_LEN,
            num_epochs=2,
            clean_dataset_path="tests/test_results/clean.jsonl",
            noisy_dataset_paths=["tests/test_results/noisy.jsonl"],
        )

        finetuning_dataset.init_epoch(0)
        for seq, tgt, mask in finetuning_dataset:
            tokenized_seq = tknzr.decode(seq)
            if (
                tknzr.inst_start_tok in tokenized_seq
                and tknzr.bos_tok not in tokenized_seq
            ):
                detokenized_midi_dict = tknzr.detokenize(tokenized_seq)
                res = detokenized_midi_dict.to_midi()
                res.save(f"tests/test_results/comb.mid")
                break


setup_logger()
if __name__ == "__main__":
    unittest.main()
