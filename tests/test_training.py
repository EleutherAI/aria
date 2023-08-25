import unittest
import logging
import os

from aria.training import pretrain
from aria.train import pretrain as pretrain2
from aria.tokenizer import TokenizerLazy
from aria.data.midi import MidiDict
from aria.data.datasets import MidiDataset, TokenizedDataset

TRAIN_DATA_PATH = "tests/test_results/testoverfit_train_dataset.jsonl"
VAL_DATA_PATH = "tests/test_results/testoverfit_val_dataset.jsonl"


# TODO:
# Add test for testing that rotary embeddings are working correctly. I want to
# test that rotary embeddings are robust to different sequence lengths


class TestTraining(unittest.TestCase):
    def test_pretraining(self):
        pass

    def test_overfitting(self):
        # Prepare datasets
        train_mididict = MidiDict.from_midi("tests/test_data/beethoven.mid")
        val_mididict = MidiDict.from_midi("tests/test_data/arabesque.mid")
        train_midi_dataset = MidiDataset([train_mididict])
        val_midi_dataset = MidiDataset([val_mididict])

        tokenizer = TokenizerLazy(return_tensors=True)
        TokenizedDataset.build(
            tokenizer=tokenizer,
            save_path=TRAIN_DATA_PATH,
            midi_dataset=train_midi_dataset,
            max_seq_len=256,
            overwrite=True,
        )
        TokenizedDataset.build(
            tokenizer=tokenizer,
            save_path=VAL_DATA_PATH,
            midi_dataset=val_midi_dataset,
            max_seq_len=256,
            overwrite=True,
        )

        self.assertTrue(os.path.isfile(TRAIN_DATA_PATH), "train data not found")
        self.assertTrue(os.path.isfile(VAL_DATA_PATH), "val data not found")

        # pretrain(
        #     model_name="test",
        #     tokenizer_name="lazy",
        #     train_data_path=TRAIN_DATA_PATH,
        #     val_data_path=VAL_DATA_PATH,
        #     num_workers=4,
        #     num_gpus=1,
        #     epochs=500,
        #     batch_size=2,
        #     overfit=True,
        # )

        pretrain2(
            model_name="test",
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=VAL_DATA_PATH,
            num_workers=4,
            epochs=500,
            batch_size=1,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
