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
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

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

# TODO:
# - Add fine-tuning functionality
# - Test that everything works on a distributed setup


def setup_logger(project_dir: str):
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    fh = RotatingFileHandler(
        os.path.join(project_dir, "logs.txt"), backupCount=5, maxBytes=1024**3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return get_logger(__name__)  # using accelerate.logging.get_logger()


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

    os.mkdir(os.path.join(project_dir_abs, "checkpoints"))

    return project_dir_abs


def get_pretrain_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    WARMUP_STEPS = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.00001,
        end_factor=1,
        total_iters=WARMUP_STEPS,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=(num_epochs * steps_per_epoch) - WARMUP_STEPS,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[WARMUP_STEPS],
    )

    return optimizer, lr_scheduler


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
        shuffle=False,  # Perhaps change this but be careful
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Same as above
    )

    return train_dataloader, val_dataloader


def rolling_average(prev_avg: float, x_n: float, n: int):
    # Returns rolling average without needing to recalculate
    if n == 0:
        return x_n
    else:
        return ((prev_avg * (n - 1)) / n) + (x_n / n)


# TODO: Make sure this works correctly when running on multiple gpus. Seriously
# step through to make sure that there are no issues when using accelerate.
# The following is a individual list of things that I want to test.
#
# - Add a print statement for the size of the batches inside each model. Does
#   each model receive the same batch? Are the batches split into different
#   slices?
# - What happens when we use a print statement with accelerate launch with
#   multiple gpus? What happens when we use a normal logger instead of the
#   accelerate one?
# - Print every time that the lr scheduler is iterated. Are two processes
#   running causing it to iterate twice as fast?
# - See this for more information about how things should run in raw torch,
#   https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html


def train(
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    steps_per_checkpoint: int | None = None,
    resume_step: int | None = None,
):
    def make_checkpoint(_accelerator, _epoch: int, _step: int | None = None):
        if _step:
            checkpoint_dir = os.path.join(
                project_dir,
                "checkpoints",
                f"epoch{_epoch}_step{_step}",
            )
        else:
            checkpoint_dir = os.path.join(
                project_dir,
                "checkpoints",
                f"epoch{_epoch}",
            )

        logger.info(
            f"EPOCH {_epoch}/{epochs}: Saving checkpoint - {checkpoint_dir}"
        )
        _accelerator.save_state(checkpoint_dir)

    # This is all slightly messy as train_loop and val_loop make use of the
    # variables in the wider scope. Perhaps refactor this at some point.
    def train_loop(dataloader: DataLoader, _epoch: int, _resume_step: int = 0):
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
            pass
        else:
            print("fail")
            lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

        model.train()
        for step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
            )
        ):
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

            # Logging
            logger.debug(
                f"EPOCH {_epoch} STEP {_resume_step + step}: "
                f"lr={lr_for_print}, "
                f"loss={round(loss.item(), 4)}, "
                f"trailing_loss={round(trailing_loss, 4)}, "
                f"average_loss={round(avg_train_loss, 4)}"
            )
            loss_writer.writerow([_epoch, _resume_step + step, loss.item()])
            pbar.set_postfix_str(
                f"lr={lr_for_print}, "
                f"loss={round(loss.item(), 4)}, "
                f"trailing={round(trailing_loss, 4)}"
            )

            # Backwards step
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
                lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

            if steps_per_checkpoint:
                if step % steps_per_checkpoint == 0 and step != 0:
                    make_checkpoint(
                        _accelerator=accelerator,
                        _epoch=epoch,
                        _step=step,
                    )

            # break # Overfit batch

        # continue # Overfit batch

        logger.info(
            f"EPOCH {_epoch}/{epochs}: Finished training - "
            f"average_loss={round(avg_train_loss, 4)}"
        )

        return avg_train_loss

    def val_loop(dataloader, _epoch: int):
        avg_val_loss = 0
        model.eval()
        for step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
            )
        ):
            src, tgt = batch  # (b_sz, s_len), (b_sz, s_len, v_sz)
            with torch.no_grad():
                logits = model(src)  # (b_sz, s_len, v_sz)
            logits = logits.transpose(1, 2)  # Transpose for CrossEntropyLoss
            loss = loss_fn(logits, tgt)

            # Logging
            avg_val_loss = rolling_average(avg_val_loss, loss.item(), step)
            pbar.set_postfix_str(f"average_loss={round(avg_val_loss, 4)}")

        # EPOCH
        logger.info(
            f"EPOCH {_epoch}/{epochs}: Finished evaluation - "
            f"average_loss={round(avg_val_loss, 4)}"
        )

        return avg_val_loss

    if steps_per_checkpoint:
        assert (
            steps_per_checkpoint > 1
        ), "Invalid checkpoint mode value (too small)"

    TRAILING_LOSS_STEPS = 15
    logger = get_logger(__name__)  # Accelerate logger
    project_dir = accelerator.project_dir

    loss_fn = nn.CrossEntropyLoss()

    loss_csv = open(os.path.join(project_dir, "loss.csv"), "w")
    loss_writer = csv.writer(loss_csv)
    loss_writer.writerow(["epoch", "step", "loss"])
    epoch_csv = open(os.path.join(project_dir, "epoch.csv"), "w")
    epoch_writer = csv.writer(epoch_csv)
    epoch_writer.writerow(["epoch", "avg_train_loss", "avg_val_loss"])

    if resume_step:
        logger.info(
            f"Resuming training from step {resume_step} - logging as EPOCH 0"
        )
        skipped_dataloader = accelerator.skip_first_batches(
            dataloader=train_dataloader,
            num_batches=resume_step,
        )
        avg_train_loss = train_loop(
            dataloader=skipped_dataloader,
            _epoch=0,
            _resume_step=resume_step,
        )
        avg_val_loss = val_loop(dataloader=val_dataloader, _epoch=0)
        epoch_writer.writerow([0, avg_train_loss, avg_val_loss])

    for epoch in range(1, epochs + 1):
        avg_train_loss = train_loop(dataloader=train_dataloader, _epoch=epoch)
        avg_val_loss = val_loop(dataloader=val_dataloader, _epoch=epoch)
        epoch_writer.writerow([epoch, avg_train_loss, avg_val_loss])
        make_checkpoint(_accelerator=accelerator, _epoch=epoch)

    loss_csv.close()
    epoch_csv.close()


# NOTE: Any differences observed when resuming training are most likely the
# result of randomness inherent to the data-augmentation. I'm currently unsure
# how to register and restore this random state during checkpointing.
def resume_pretrain(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    checkpoint_dir: str,
    resume_step: int,
    steps_per_checkpoint: int | None = None,
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
    logger.warning(
        "Please insure that the training config and resume step are set "
        "correctly, the script does not currently check that this is the case. "
        "If the previous checkpoint was saved at step n, then resume_step "
        "should be n+1. If there is a mismatch between the batch size then the "
        "script will resume at the wrong step."
    )
    logger.info(
        f"Using training config: "
        f"model_name={model_name}, "
        f"epochs={epochs}, "
        f"batch_size={batch_size}, "
        f"num_workers={num_workers}"
        f"checkpoint_dir={checkpoint_dir}"
        f"resume_step={resume_step}"
    )
    if steps_per_checkpoint:
        logger.info(f"Creating checkpoints ")

    # Init model
    tokenizer = TokenizerLazy(return_tensors=True)
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = torch.compile(TransformerLM(model_config), mode="default")

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_aug=True,
    )

    assert (
        train_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"
    assert (
        val_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"

    optimizer, scheduler = get_pretrain_optim(
        model,
        num_epochs=epochs,
        steps_per_epoch=len(train_dataloader),
    )

    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    accelerator.load_state(checkpoint_dir)
    logger.info(f"Loaded checkpoint at {checkpoint_dir}")

    logger.info("Starting train job")
    train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_checkpoint=steps_per_checkpoint,
        resume_step=resume_step,
    )


# Maybe refactor so that pretrain and resume_pretrain use the same prep code
def pretrain(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    batch_size: int,
    epochs: int,
    steps_per_checkpoint: int | None = None,
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
    logger.info(
        f"Using training config: "
        f"model_name={model_name}, "
        f"epochs={epochs}, "
        f"batch_size={batch_size}, "
        f"num_workers={num_workers}"
    )

    # Init model
    tokenizer = TokenizerLazy(return_tensors=True)
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = torch.compile(TransformerLM(model_config), mode="default")
    logger.info(f"Loaded model with config: {load_model_config(model_name)}")

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_aug=True,
    )

    assert (
        train_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"
    assert (
        val_dataloader.dataset.max_seq_len == model_config.max_seq_len
    ), "max_seq_len differs between datasets and model config"

    optimizer, scheduler = get_pretrain_optim(
        model,
        num_epochs=epochs,
        steps_per_epoch=len(train_dataloader),
    )

    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
    )

    logger.info("Starting train job")
    train(
        epochs=epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        steps_per_checkpoint=steps_per_checkpoint,
    )


def parse_resume_args():
    argp = argparse.ArgumentParser(prog="python aria/train.py resume")
    argp.add_argument("model", help="name of model config file")
    argp.add_argument("train_data", help="path to train data")
    argp.add_argument("val_data", help="path to val data")
    argp.add_argument("-cdir", help="checkpoint dir", type=str, required=True)
    argp.add_argument("-rstep", help="resume step", type=int, required=True)
    argp.add_argument("-epochs", help="train epochs", type=int, required=True)
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument("-workers", help="number workers", type=int, default=1)
    argp.add_argument("-pdir", help="project dir", type=str, required=False)
    argp.add_argument(
        "-spc", help="steps per checkpoint", type=int, required=False
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
    argp.add_argument("-pdir", help="project dir", type=str, required=False)
    argp.add_argument(
        "-spc", help="steps per checkpoint", type=int, required=False
    )

    return argp.parse_args(sys.argv[2:])


# This entrypoint has not been tested properly.
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
            project_dir=pretrain_args.pdir,
            steps_per_checkpoint=pretrain_args.spc,
        )
    elif args.mode == "resume":
        pretrain_args = parse_resume_args()
        resume_pretrain(
            model_name=pretrain_args.model,
            train_data_path=pretrain_args.train_data,
            val_data_path=pretrain_args.val_data,
            checkpoint_dir=pretrain_args.cdir,
            resume_step=pretrain_args.rstep,
            num_workers=pretrain_args.workers,
            batch_size=pretrain_args.bs,
            epochs=pretrain_args.epochs,
            project_dir=pretrain_args.pdir,
            steps_per_checkpoint=pretrain_args.spc,
        )
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)
