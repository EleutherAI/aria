import unittest
import logging
import os

from aria.training import pretrain


class TestTraining(unittest.TestCase):
    def test_pretraining(self):
        pass

    def test_overfitting(self):
        self.assertTrue(
            os.path.isfile("data/train.jsonl"), "train data not found"
        )
        self.assertTrue(os.path.isfile("data/val.jsonl"), "val data not found")

        pretrain(
            model_name="test",
            tokenizer_name="lazy",
            train_data_path="data/train.jsonl",
            val_data_path="data/val.jsonl",
            workers=4,
            gpus=1,
            epochs=500,
            batch_size=4,
            overfit=True,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
