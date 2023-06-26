"""Contains pre-training/fine-tuning code"""

import re
import torch
import lightning as pl
import os.path

from torch import nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from aria.tokenizer import TokenizerLazy
from aria.data.datasets import TokenizedDataset

# TODO:
# - Refactor
# - This needs to be tested


class PretrainLM(pl.LightningModule):
    """PyTorch Lightning Module for TransformerLM pre-training."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerLM(model_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src: torch.Tensor):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
        logits = self.model(src)  # (b_sz, s_len, v_sz)

        # Transpose for CrossEntropyLoss
        logits = logits.transpose(1, 2)
        tgt = tgt.transpose(1, 2)
        loss = self.loss_fn(logits, tgt)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Not sure what this does
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
        logits = self.model(src)  # (b_sz, s_len, v_sz)

        # Transpose for CrossEntropyLoss
        logits = logits.transpose(1, 2)
        tgt = tgt.transpose(1, 2)
        loss = self.loss_fn(logits, tgt).item()

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Not sure what this does
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=3e-4)


def get_gpu_precision():
    """Only compatible with PyTorchLightning"""
    a100_re = re.compile(r"[aA]100")
    v100_re = re.compile(r"[vV]100")
    if a100_re.search(torch.cuda.get_device_name(0)):
        print("A100 detected")
        return "bf16-mixed"
    elif v100_re.search(torch.cuda.get_device_name(0)):
        print("V100 detected")
        return "16-mixed"
    else:
        print("GPU not A100 or V100")
        return "16-mixed"


# TODO:
# - Test
# - Refactor
# - Add docstrings
def pretrain(
    model_name: str,
    tokenizer_name: str,
    data_path: str,
    workers: int,
    gpus: int,
    epochs: int,
    batch_size: int,
    checkpoint: str = None,
):
    # Validate inputs
    assert 0 < workers <= 128, "Too many workers"
    assert 0 < gpus <= 8, "Too many GPUs"
    assert os.path.isfile(data_path)
    assert os.path.isfile(checkpoint)
    assert torch.cuda.is_available() is True, "CUDA not available"

    # Load config and tokenizer
    model_config = ModelConfig(**load_model_config(model_name))
    if tokenizer_name == "lazy":
        tokenizer = TokenizerLazy(
            padding=True,
            truncate_type="default",
            max_seq_len=model_config.max_seq_len,
            return_tensors=True,
        )
        model_config.set_vocab_size(tokenizer.vocab_size)
    else:
        raise ValueError("Invalid value for tokenizer_name.")

    # Load model
    if isinstance(checkpoint, str) and checkpoint is not None:
        model = PretrainLM.load_from_checkpoint(checkpoint)
    elif checkpoint is None:
        model = PretrainLM(model_config)

    # Load datasets and dataloader
    train_dataset, val_dataset = TokenizedDataset.load_train_val(
        data_path,
        tokenizer,
    )
    # Add aug_range for transform fns
    transform_fns = [
        tokenizer.export_time_aug(),
        tokenizer.export_pitch_aug(),
        tokenizer.export_velocity_aug(),
    ]
    train_dataset.set_transform(transform_fns)
    val_dataset.set_transform(transform_fns)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
    )

    # Setup trainer
    trainer = pl.Trainer(
        devices=gpus,
        accelerator="gpu",
        precision=get_gpu_precision(),
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{train_loss}-{val_loss}",
                save_last=True,
                save_top_k=5,
                monitor="val_loss",
                save_weights_only=False,
            )  # See https://shorturl.at/AGHZ3
        ],
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
