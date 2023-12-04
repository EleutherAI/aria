import os
import shutil
import unittest
import logging
import torch

from aria.train import train, resume_train, convert_cp_from_accelerate
from aria.tokenizer import TokenizerLazy
from aria.model import ModelConfig, TransformerLM
from aria.config import load_model_config
from aria.data.midi import MidiDict
from aria.data.datasets import (
    MidiDataset,
    PretrainingDataset,
    FinetuningDataset,
)

TRAIN_DATA_PATH = "tests/test_results/testpretrain_dataset_train"
VAL_DATA_PATH = "tests/test_results/testpretrain_dataset_val"

if not os.path.isdir("tests/test_results"):
    os.makedirs("tests/test_results")

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

        if os.path.exists(TRAIN_DATA_PATH):
            shutil.rmtree(TRAIN_DATA_PATH)
        PretrainingDataset.build(
            tokenizer=tokenizer,
            save_dir=TRAIN_DATA_PATH,
            max_seq_len=256,
            num_epochs=15,
            midi_dataset=train_midi_dataset,
        )
        if os.path.exists(VAL_DATA_PATH):
            shutil.rmtree(VAL_DATA_PATH)
        PretrainingDataset.build(
            tokenizer=tokenizer,
            save_dir=VAL_DATA_PATH,
            max_seq_len=256,
            num_epochs=1,
            midi_dataset=val_midi_dataset,
        )

        self.assertTrue(os.path.isdir(TRAIN_DATA_PATH), "train data not found")
        self.assertTrue(os.path.isdir(VAL_DATA_PATH), "val data not found")

        if os.path.isdir("./experiments/0"):
            logging.warning(
                "Experiment logs present at ./experiments/0, resume "
                "functionality will not be tested"
            )
            test_resume = False
        else:
            test_resume = True

        train(
            model_name="test",
            train_data_path=TRAIN_DATA_PATH,
            val_data_path=VAL_DATA_PATH,
            mode="pretrain",
            num_workers=4,
            batch_size=1,
            epochs=10,
            steps_per_checkpoint=50,
        )

        if test_resume == True:
            resume_train(
                model_name="test",
                train_data_path=TRAIN_DATA_PATH,
                val_data_path=VAL_DATA_PATH,
                mode="pretrain",
                num_workers=4,
                batch_size=1,
                epochs=5,
                checkpoint_dir="./experiments/0/checkpoints/epoch9_step50",
                resume_step=51,
                resume_epoch=9,
                steps_per_checkpoint=50,
            )
        else:
            logging.warning("Skipping testing of resume_train")

        # Testing convert_cp_from_accelerate
        if os.path.isdir("./experiments/0/checkpoints/epoch9_step50"):
            convert_cp_from_accelerate(
                model_name="test",
                checkpoint_dir="./experiments/0/checkpoints/epoch9_step50",
                save_path="tests/test_results/model_cp.bin",
            )

            tokenizer = TokenizerLazy(return_tensors=True)
            model_config = ModelConfig(**load_model_config("test"))
            model_config.set_vocab_size(tokenizer.vocab_size)
            model = TransformerLM(model_config)

            model.load_state_dict(torch.load("tests/test_results/model_cp.bin"))
        else:
            logging.warning(
                "Resume checkpoint not found at "
                "./experiments/0/checkpoints/epoch9_step50 - skipping "
                "save_model_from_cp test"
            )

    def test_finetuning(self):
        # TODO: Implement
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
