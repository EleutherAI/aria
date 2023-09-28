import unittest
import logging
import os

from aria.train import pretrain, resume_pretrain
from aria.tokenizer import TokenizerLazy
from aria.data.midi import MidiDict
from aria.data.datasets import MidiDataset, TokenizedDataset

TRAIN_DATA_PATH = "tests/test_results/testpretrain_dataset_train.jsonl"
VAL_DATA_PATH = "tests/test_results/testpretrain_dataset_val.jsonl"


# TODO:
# Add test for testing that rotary embeddings are working correctly. I want to
# test that rotary embeddings are robust to different sequence lengths


class TestTraining(unittest.TestCase):
    def test_overfitting(self):
        pass

    def test_pretraining(self):
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

        if os.path.isdir("./experiments/0"):
            logging.warning("Experiment logs present at ./experiments/0")

        pretrain(
            model_name="test",
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=VAL_DATA_PATH,
            num_workers=4,
            batch_size=1,
            epochs=10,
            steps_per_checkpoint=50,
        )

        if os.path.isdir("./experiments/0/checkpoints/epoch10_step50"):
            resume_pretrain(
                model_name="test",
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                num_workers=4,
                batch_size=1,
                epochs=5,
                checkpoint_dir="./experiments/0/checkpoints/epoch10_step50",
                resume_step=51,
                steps_per_checkpoint=50,
            )
        else:
            logging.warning(
                "Resume checkpoint not found at "
                "./experiments/0/checkpoints/epoch10_step50 "
                "- skipping resume_pretrain test"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
