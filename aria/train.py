import os.path
import argparse
import sys
import torch
import accelerate

from torch import nn as nn
from torch.utils.data import DataLoader
from accelerate.logging import get_logger

from aria.config import load_model_config
from aria.model import ModelConfig, TransformerLM
from aria.tokenizer import TokenizerLazy
from aria.data.datasets import TokenizedDataset

# ----- USAGE -----
#
# This script is meant to be run using the huggingface accelerate cli, see:
#
# https://huggingface.co/docs/accelerate/basic_tutorials/launch
# https://huggingface.co/docs/accelerate/package_reference/cli
#
# For example usage you could run the pre-training script with:
#
# accelerate launch [arguments] aria/train.py pretrain \
#   -train_data data/train.jsonl \
#   -val_data data/train.jsonl \
#   -epochs 10 \
#   -bs 32 \
#   -workers 8

# ----- TODO -----

# We want the following functionality:
# - pretrain: runs the pretrain function
# - finetune: runs the finetune function
# - resume: resumes a training run from a checkpoint - pretrain or finetune
# Each of these should work with accelerate, and the functionality should be
# chosen according directly from the accelerate cli. We must therefore remove
# pretrain functionality from the run.py entrypoint.

# - Add support for mixed precision (and gradient clipping)
# - Add checkpointing
# - Add logging (maybe add tensorboard support?)


def get_pretrain_lr_scheduler():
    raise NotImplementedError


def get_pretrain_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=3e-4)


# Prepare should be used before they are passed into this function. Train
# should just contain the logic for the train loop, logging, experiment
# tracking, checkpointing ect... We need to completely restructure this.
def train(
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    loss_fn = nn.CrossEntropyLoss()

    logger = get_logger(__name__, log_level="INFO")
    # Train loop
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
            logits = model(src)  # (b_sz, s_len, v_sz)

            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)
            logger.info(f"{epoch} {step} {loss.item()}")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            break  # HERE FOR OVERFIT TEST


def resume():
    pass


def pretrain(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
):
    # Validate inputs
    assert 0 < num_workers <= 128, "Too many workers"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert os.path.isfile(train_data_path)
    assert os.path.isfile(val_data_path)
    assert torch.cuda.is_available() is True, "CUDA not available"

    accelerator = accelerate.Accelerator()

    # Init model
    tokenizer = TokenizerLazy(return_tensors=True)
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = torch.compile(TransformerLM(model_config), mode="default")
    optimizer = get_pretrain_optimizer(model)

    # Init datasets & data loaders
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

    ## COMMENTED FOR OVERFIT TEST
    # train_dataset.set_transform(
    #     [
    #         tokenizer.export_velocity_aug(2),
    #         tokenizer.export_pitch_aug(4),
    #         tokenizer.export_tempo_aug(0.15),
    #     ]
    # )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer
    )

    train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
    )


def parse_resume_args():
    argp = argparse.ArgumentParser(prog="python aria/train.py resume")
    argp.add_argument(
        "format",
        choices=["midi_dict", "tokenized"],
        help="type of dataset to build",
    )
    argp.add_argument("save_path", help="path to save dataset")

    argp.add_argument(
        "-dir", help="directory containing midi files", required=False
    )
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
    argp.add_argument(
        "-load_path", help="path midi_dict dataset", required=False
    )
    argp.add_argument(
        "-tokenizer",
        required=False,
        choices=["lazy"],
        help="specify tokenizer type",
    )

    return argp.parse_args(sys.argv[2:])


def parse_pretrain_args():
    argp = argparse.ArgumentParser(prog="python aria/train.py pretrain")
    argp.add_argument("model", help="name of model config file")
    argp.add_argument("train_data", help="path to train data")
    argp.add_argument("val_data", help="path to val data")
    argp.add_argument("-epochs", help="train epochs", type=int, required=True)
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument("-workers", help="number workers", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


if __name__ == "__main__":
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(
        usage="python aria/train.py <command> [<args>]"
    )
    parser.add_argument(
        "mode", help="training mode", choices=("pretrain", "resume")
    )

    args = parser.parse_args(sys.argv[1:2])
    if not hasattr(args, "mode"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.mode == "pretrain":
        pretrain_args = parse_pretrain_args()
        pretrain(
            model_name=pretrain_args.model,
            train_data_path=pretrain_args.train_data,
            val_data_path=pretrain_args.val_data,
            num_workers=pretrain_args.workers,
            batch_size=pretrain_args.bs,
            epochs=pretrain_args.epochs,
        )
    elif args.mode == "resume":
        raise NotImplementedError
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)
