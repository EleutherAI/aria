import os
import sys
import csv
import argparse
import logging
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


def setup_logger(project_dir: str):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    fh = logging.FileHandler(os.path.join(project_dir, "logs.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return get_logger(__name__)


def setup_project_dir(project_dir: str | None):
    if not project_dir:
        # Create project directory
        if not os.path.isdir("./experiments"):
            os.mkdir("./experiments")

        project_dirs = [
            _dir
            for _dir in os.listdir("./experiments")
            if os.path.isdir(os.path.join("experiments", _dir))
        ]

        ind = 0
        while True:
            if str(ind) not in project_dirs:
                break
            else:
                ind += 1

        project_dir_abs = os.path.abspath(os.path.join("experiments", str(ind)))
        assert not os.path.isdir(project_dir_abs)
        os.mkdir(project_dir_abs)

    elif project_dir:
        # Run checks on project directory
        if os.path.isdir(project_dir):
            assert (
                len(os.listdir()) == 0
            ), "Provided project directory is not empty"
            project_dir_abs = os.path.abspath(project_dir)
        elif os.path.isfile(project_dir):
            raise FileExistsError(
                "The provided path points toward an existing file"
            )
        else:
            try:
                os.mkdir(project_dir)
            except Exception as e:
                raise e(f"Failed to create project directory at {project_dir}")
        project_dir_abs = os.path.abspath(project_dir)

    # Add checkpoint dir
    os.mkdir(os.path.join(project_dir_abs, "checkpoints"))

    return project_dir_abs


def get_dataloaders(
    train_data_path: str,
    val_data_path: str,
    tokenizer: TokenizerLazy,
    batch_size: int,
    num_workers: int,
    apply_aug: bool = True,
):
    """Returns tuple: (train_dataloader, val_dataloader)"""
    # Init datasets & data loaders
    train_dataset = TokenizedDataset(
        file_path=train_data_path,
        tokenizer=tokenizer,
    )
    val_dataset = TokenizedDataset(
        file_path=val_data_path,
        tokenizer=tokenizer,
    )

    if apply_aug:
        train_dataset.set_transform(
            [
                tokenizer.export_velocity_aug(2),
                tokenizer.export_pitch_aug(4),
                tokenizer.export_tempo_aug(0.15),
                tokenizer.export_chord_mixup(),
            ]
        )

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

    return train_dataloader, val_dataloader


def get_pretrain_lr_scheduler():
    raise NotImplementedError


def get_pretrain_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=3e-4)


def rolling_average(prev_avg: float, x_n: float, n: int):
    # Returns rolling average without needing to recalculate
    if n == 0:
        return x_n
    else:
        return ((prev_avg * (n - 1)) / n) + (x_n / n)


def train(
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
):
    logger = get_logger(__name__)  # Accelerate logger
    project_dir = accelerator.project_dir
    loss_fn = nn.CrossEntropyLoss()
    TRAILING_LOSS_STEPS = 20

    loss_csv = open(os.path.join(project_dir, "loss.csv"), "w")
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow(["epoch", "step", "loss"])
    epoch_csv = open(os.path.join(project_dir, "epoch.csv"), "w")
    epoch_writer = csv.writer(epoch_csv)
    epoch_writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    # Train loop
    for epoch in range(epochs):
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
            logits = model(src)  # (b_sz, s_len, v_sz)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

            # Calculate statistics
            loss_buffer.append(loss.item())
            if len(loss_buffer) > TRAILING_LOSS_STEPS:
                loss_buffer.pop(0)
            trailing_loss = sum(loss_buffer) / len(loss_buffer)
            avg_train_loss = rolling_average(avg_train_loss, loss.item(), step)

            logger.debug(
                f"Train step {step} of epoch {epoch}: "
                + f"loss={round(loss.item(), 5)}, "
                + f"trailing_loss={round(trailing_loss, 4)}, "
                + f"average_loss={round(avg_train_loss, 4)}"
            )
            loss_writer.writerow([epoch, step, loss.item()])

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # break # Overfit batch

        # continue # Overfit batch

        logger.info(
            f"Finished TRAIN for epoch {epoch}/{epochs - 1}: "
            + f"average_loss={round(avg_train_loss, 4)}"
        )

        avg_val_loss = 0
        trailing_loss = 0
        loss_buffer = []
        model.eval()
        for step, batch in enumerate(val_dataloader):
            src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
            with torch.no_grad():
                logits = model(src)  # (b_sz, s_len, v_sz)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

            avg_val_loss = rolling_average(avg_val_loss, loss.item(), step)

        logger.info(
            f"Finished VAL for epoch {epoch}/{epochs - 1}: "
            + f"average_loss={round(avg_val_loss, 4)}"
        )
        epoch_writer.writerow([epoch, avg_train_loss, avg_val_loss])


def resume():
    pass


def pretrain(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    project_dir: str = None,
):
    # Validate inputs
    assert 0 < num_workers <= 128, "Too many workers"
    assert epochs > 0, "Invalid number of epochs"
    assert batch_size > 0, "Invalid batch size"
    assert os.path.isfile(train_data_path)
    assert os.path.isfile(val_data_path)
    assert torch.cuda.is_available() is True, "CUDA not available"

    project_dir = setup_project_dir(project_dir)
    accelerator = accelerate.Accelerator(project_dir=project_dir)
    logger = setup_logger(project_dir)
    logger.info(f"Using project directory {project_dir} ")

    # Init model
    tokenizer = TokenizerLazy(return_tensors=True)
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = torch.compile(TransformerLM(model_config), mode="default")
    optimizer = get_pretrain_optimizer(model)

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_aug=False,
    )

    assert (
        train_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"
    assert (
        val_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"

    model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer
    )

    logger.info("Starting train job")
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
