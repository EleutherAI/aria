import unittest
import logging

from aria.data import datasets
from aria.data.midi import MidiDict


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

    def test_build_direct_to_file(self):
        dataset = datasets.MidiDataset.build_direct_to_file(
            dir="tests/test_data",
            save_path="tests/test_results/mididict_dataset_direct.jsonl",
            recur=True,
        )

        dataset_reloaded = datasets.MidiDataset.load(
            load_path="tests/test_results/mididict_dataset_direct.jsonl",
        )
        self.assertEqual(len(dataset_reloaded), 3)
        self.assertEqual(type(dataset_reloaded[0]), MidiDict)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
