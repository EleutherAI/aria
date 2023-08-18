"""Contains pre-training/fine-tuning code"""

import re
import torch
import logging
import lightning as pl
import os.path

from torch import nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from aria.tokenizer import TokenizerLazy
from aria.data.datasets import TokenizedDataset


class PretrainLM(pl.LightningModule):
    """PyTorch Lightning Module for TransformerLM pre-training."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = model_config
        self.model = TransformerLM(model_config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src: torch.Tensor):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
        logits = self.model(src)  # (b_sz, s_len, v_sz)

        # Transpose for CrossEntropyLoss
        logits = logits.transpose(1, 2)
        loss = self.loss_fn(logits, tgt)

        self.log(
            "train_loss",
            round(loss.item(), 3),
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
        loss = self.loss_fn(logits, tgt).item()

        self.log(
            "val_loss",
            round(loss, 3),
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


def pretrain(
    model_name: str,
    tokenizer_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    num_gpus: int,
    epochs: int,
    batch_size: int,
    checkpoint: str = None,
    overfit: bool = False,
):
    # Validate inputs
    assert 0 < num_workers <= 128, "Too many workers"
    assert 0 < num_gpus <= 8, "Too many (or none) GPUs"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert os.path.isfile(train_data_path)
    assert os.path.isfile(val_data_path)
    assert torch.cuda.is_available() is True, "CUDA not available"
    if checkpoint:
        assert os.path.isfile(checkpoint)

    # Load config and tokenizer
    model_config = ModelConfig(**load_model_config(model_name))
    if tokenizer_name == "lazy":
        tokenizer = TokenizerLazy(
            return_tensors=True,
        )
        model_config.set_vocab_size(tokenizer.vocab_size)
    else:
        raise ValueError("Invalid value for tokenizer_name.")

    # Load model
    if isinstance(checkpoint, str) and checkpoint is not None:
        model = PretrainLM.load_from_checkpoint(checkpoint)
    elif not checkpoint:
        model = PretrainLM(model_config)

    # Load datasets & dataloaders
    train_dataset = TokenizedDataset(
        file_path=train_data_path,
        tokenizer=tokenizer,
    )
    val_dataset = TokenizedDataset(
        file_path=val_data_path,
        tokenizer=tokenizer,
    )
    assert (
        train_dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"
    assert (
        val_dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"

    if overfit is False:
        train_dataset.set_transform(
            [
                tokenizer.export_velocity_aug(2),
                tokenizer.export_pitch_aug(4),
                tokenizer.export_tempo_aug(0.15),
            ]
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Setup trainer
    if overfit is True:
        trainer = pl.Trainer(
            devices=num_gpus,
            accelerator="gpu",
            # strategy="ddp",
            precision=get_gpu_precision(),
            max_epochs=epochs,
            overfit_batches=1,
            enable_checkpointing=False,
        )
    else:
        trainer = pl.Trainer(
            devices=num_gpus,
            accelerator="gpu",
            # strategy="ddp",
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
