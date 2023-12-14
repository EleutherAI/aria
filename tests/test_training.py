import os
import shutil
import unittest
import logging

from aria.train import train, resume_train, convert_cp_from_accelerate
from aria.tokenizer import RelTokenizer, AbsTokenizer
from aria.data.midi import MidiDict
from aria.data.datasets import (
    MidiDataset,
    PretrainingDataset,
    FinetuningDataset,
)

TEST_TOKENIZER = "abs"
PT_TRAIN_DATA_PATH = "tests/test_results/pretrain_dataset_train"
PT_VAL_DATA_PATH = "tests/test_results/pretrain_dataset_val"
FT_TRAIN_DATA_PATH = "tests/test_results/finetune_dataset_train.jsonl"
FT_VAL_DATA_PATH = "tests/test_results/finetune_dataset_val.jsonl"
PT_PATH = "tests/test_results/pretrain"
PT_RESUME_PATH = "tests/test_results/resume_pretrain"
FT_PATH = "tests/test_results/finetune"
FT_RESUME_PATH = "tests/test_results/resume_finetune"
CP_PATH = "tests/test_results/cp.bin"

if not os.path.isdir("tests/test_results"):
    os.makedirs("tests/test_results")


class TestTraining(unittest.TestCase):
    def test_overfitting(self):
        pass

    def test_training(self):
        # Prepare datasets
        train_mididict = MidiDict.from_midi("tests/test_data/beethoven.mid")
        val_mididict = MidiDict.from_midi("tests/test_data/arabesque.mid")
        train_midi_dataset = MidiDataset([train_mididict])
        val_midi_dataset = MidiDataset([val_mididict])

        if TEST_TOKENIZER == "abs":
            tokenizer = AbsTokenizer(return_tensors=False)
        elif TEST_TOKENIZER == "rel":
            tokenizer = RelTokenizer(return_tensors=False)
        else:
            raise KeyError

        # PRETRAINING
        if os.path.exists(PT_TRAIN_DATA_PATH):
            shutil.rmtree(PT_TRAIN_DATA_PATH)
        PretrainingDataset.build(
            tokenizer=tokenizer,
            save_dir=PT_TRAIN_DATA_PATH,
            max_seq_len=256,
            num_epochs=15,
            midi_dataset=train_midi_dataset,
        )
        if os.path.exists(PT_VAL_DATA_PATH):
            shutil.rmtree(PT_VAL_DATA_PATH)
        PretrainingDataset.build(
            tokenizer=tokenizer,
            save_dir=PT_VAL_DATA_PATH,
            max_seq_len=256,
            num_epochs=1,
            midi_dataset=val_midi_dataset,
        )

        if os.path.exists(PT_PATH):
            shutil.rmtree(PT_PATH)
        train(
            model_name="test",
            train_data_path=PT_TRAIN_DATA_PATH,
            val_data_path=PT_VAL_DATA_PATH,
            mode="pretrain",
            num_workers=4,
            batch_size=1,
            epochs=10,
            project_dir=PT_PATH,
            steps_per_checkpoint=50,
        )

        # RESUMING PRETRAINING
        if os.path.exists(PT_RESUME_PATH):
            shutil.rmtree(PT_RESUME_PATH)
        resume_train(
            model_name="test",
            train_data_path=PT_TRAIN_DATA_PATH,
            val_data_path=PT_VAL_DATA_PATH,
            mode="pretrain",
            num_workers=4,
            batch_size=1,
            epochs=5,
            checkpoint_dir=f"{PT_PATH}/checkpoints/epoch9_step50",
            resume_step=50,
            resume_epoch=9,
            project_dir=PT_RESUME_PATH,
            steps_per_checkpoint=50,
        )

        if os.path.isfile(CP_PATH):
            os.remove(CP_PATH)
        convert_cp_from_accelerate(
            model_name="test",
            checkpoint_dir=f"{PT_PATH}/checkpoints/epoch9_step50",
            save_path=CP_PATH,
        )

        # FINETUNING
        if os.path.isfile(FT_TRAIN_DATA_PATH):
            os.remove(FT_TRAIN_DATA_PATH)
        FinetuningDataset.build(
            tokenizer=tokenizer,
            save_path=FT_TRAIN_DATA_PATH,
            max_seq_len=256,
            stride_len=128,
            midi_dataset=train_midi_dataset,
        )
        if os.path.isfile(FT_VAL_DATA_PATH):
            os.remove(FT_VAL_DATA_PATH)
        FinetuningDataset.build(
            tokenizer=tokenizer,
            save_path=FT_VAL_DATA_PATH,
            max_seq_len=256,
            stride_len=128,
            midi_dataset=val_midi_dataset,
        )

        if os.path.exists(FT_PATH):
            shutil.rmtree(FT_PATH)
        train(
            model_name="test",
            train_data_path=FT_TRAIN_DATA_PATH,
            val_data_path=FT_VAL_DATA_PATH,
            mode="finetune",
            num_workers=4,
            batch_size=1,
            epochs=2,
            finetune_cp_path=CP_PATH,
            project_dir=FT_PATH,
            steps_per_checkpoint=50,
        )

        if os.path.exists(FT_RESUME_PATH):
            shutil.rmtree(FT_RESUME_PATH)
        resume_train(
            model_name="test",
            train_data_path=FT_TRAIN_DATA_PATH,
            val_data_path=FT_VAL_DATA_PATH,
            mode="finetune",
            num_workers=4,
            batch_size=1,
            epochs=1,
            checkpoint_dir=f"{FT_PATH}/checkpoints/epoch1_step50",
            resume_step=50,
            resume_epoch=1,
            project_dir=FT_RESUME_PATH,
            steps_per_checkpoint=50,
        )


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    unittest.main()
