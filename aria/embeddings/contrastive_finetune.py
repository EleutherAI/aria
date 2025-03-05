import torch
import os
import mmap
import argparse
import logging
import random
import copy
import accelerate
import json

from aria.config import load_model_config
from aria.utils import _load_weight
from ariautils.tokenizer import AbsTokenizer
from ariautils.midi import MidiDict
from aria.model import TransformerEMB, ModelConfig

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate.logging import get_logger
from logging.handlers import RotatingFileHandler
from tqdm import tqdm


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
        if os.path.isdir(project_dir):
            assert (
                len(os.listdir(project_dir)) == 0
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
                raise Exception(
                    f"Failed to create project directory at {project_dir}"
                ) from e

        project_dir_abs = os.path.abspath(project_dir)

    os.mkdir(os.path.join(project_dir_abs, "checkpoints"))

    return project_dir_abs


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        load_path: str,
        min_number_slice_notes: int,
        max_number_slice_notes: int,
        max_seq_len: int,
        apply_aug: bool = False,
    ):
        self.load_path = load_path
        self.min_number_slice_notes = min_number_slice_notes
        self.max_number_slice_notes = max_number_slice_notes
        self.max_seq_len = max_seq_len
        self.apply_aug = apply_aug

        self.tokenizer = AbsTokenizer()

        if apply_aug is True:
            self.aug_fns = self.tokenizer.export_data_aug()
        else:
            self.aug_fns = None

        self.index = []
        self.file_buff = open(self.load_path, "rb")
        self.mmap_obj = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )

        while True:
            pos = self.mmap_obj.tell()
            line = self.mmap_obj.readline()
            if not line:
                break
            self.index.append(pos)

    def get_slice(
        self,
        midi_dict: MidiDict,
        min_num_notes: int,
        max_num_notes: int,
        max_seq_len: int,
        apply_aug: bool = False,
    ):
        _midi_dict = copy.deepcopy(midi_dict)
        slice_length = random.randint(min_num_notes, max_num_notes)
        idx = random.randint(0, len(_midi_dict.note_msgs) - min_num_notes)

        _midi_dict.note_msgs = _midi_dict.note_msgs[idx : idx + slice_length]
        _midi_dict.metadata = {}

        tokenized_slice = self.tokenizer.tokenize(_midi_dict)

        if apply_aug:
            assert self.aug_fns
            for fn in self.aug_fns:
                tokenized_slice = fn(tokenized_slice)

            while self.tokenizer.pad_tok in tokenized_slice:
                tokenized_slice.remove(self.tokenizer.pad_tok)

        if self.tokenizer.dim_tok in tokenized_slice:
            tokenized_slice.remove(self.tokenizer.dim_tok)

        # Use EOS tok for classification head
        tokenized_slice = tokenized_slice[:max_seq_len]
        tokenized_slice += [self.tokenizer.pad_tok] * (
            max_seq_len - len(tokenized_slice)
        )
        if self.tokenizer.eos_tok not in tokenized_slice:
            tokenized_slice[-1] = self.tokenizer.eos_tok

        pos = tokenized_slice.index(self.tokenizer.eos_tok)

        return tokenized_slice, pos

    def __getitem__(self, idx: int):
        file_pos = self.index[idx]
        self.mmap_obj.seek(file_pos)

        raw_data = self.mmap_obj.readline().decode("utf-8")
        json_data = json.loads(raw_data)
        midi_dict = MidiDict.from_msg_dict(json_data)

        slice_seq_1, slice_pos_1 = self.get_slice(
            midi_dict=midi_dict,
            min_num_notes=self.min_number_slice_notes,
            max_num_notes=self.max_number_slice_notes,
            max_seq_len=self.max_seq_len,
            apply_aug=self.apply_aug,
        )
        slice_seq_2, slice_pos_2 = self.get_slice(
            midi_dict=midi_dict,
            min_num_notes=self.min_number_slice_notes,
            max_num_notes=self.max_number_slice_notes,
            max_seq_len=self.max_seq_len,
            apply_aug=self.apply_aug,
        )

        assert len(slice_seq_1) <= self.max_seq_len
        assert len(slice_seq_2) <= self.max_seq_len
        assert slice_pos_1 < self.max_seq_len
        assert slice_pos_2 < self.max_seq_len
        assert slice_seq_1[slice_pos_1] == self.tokenizer.eos_tok
        assert slice_seq_2[slice_pos_2] == self.tokenizer.eos_tok

        slices_enc = torch.tensor(
            [
                self.tokenizer.encode(slice_seq_1),
                self.tokenizer.encode(slice_seq_2),
            ]
        )

        slices_pos = torch.tensor([slice_pos_1, slice_pos_2])

        return slices_enc, slices_pos

    def __len__(self):
        return len(self.index)

    @classmethod
    def export_worker_init_fn(cls):
        def worker_init_fn(worker_id: int):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset

            if hasattr(dataset, "mmap_obj") and dataset.mmap_obj:
                dataset.mmap_obj.close()

            dataset.file_buff = open(dataset.load_path, "rb")
            dataset.mmap_obj = mmap.mmap(
                dataset.file_buff.fileno(), 0, access=mmap.ACCESS_READ
            )

        return worker_init_fn


def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: int = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.000001,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


def get_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    LR = 1e-5
    END_RATIO = 0.1
    WARMUP_STEPS = 1000

    return _get_optim(
        lr=LR,
        model=model,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        warmup=WARMUP_STEPS,
        end_ratio=END_RATIO,
    )


def get_dataloaders(
    train_data_path: str,
    val_data_path: str,
    batch_size: int,
    num_workers: int,
    min_number_slice_notes: int = 100,
    max_number_slice_notes: int = 300,
    max_seq_len: int = 1024,
):
    train_dataset = ContrastiveDataset(
        load_path=train_data_path,
        min_number_slice_notes=min_number_slice_notes,
        max_number_slice_notes=max_number_slice_notes,
        max_seq_len=max_seq_len,
    )
    val_dataset = ContrastiveDataset(
        load_path=val_data_path,
        min_number_slice_notes=min_number_slice_notes,
        max_number_slice_notes=max_number_slice_notes,
        max_seq_len=max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=ContrastiveDataset.export_worker_init_fn(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=ContrastiveDataset.export_worker_init_fn(),
    )

    return train_loader, val_loader


# TODO: This might not be 100% correct (verify CEL calculation)
def symmetric_nt_xent_loss_cosine(
    z1: torch.Tensor, z2: torch.Tensor, temperature=0.5
):
    bsz = z1.shape[0]

    z1 = F.normalize(z1, dim=1)  # First view
    z2 = F.normalize(z2, dim=1)  # Second view

    sim_matrix = (
        F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        / temperature
    )

    labels = torch.arange(bsz, device=z1.device)

    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.T, labels)

    return (loss1 + loss2) / 2.0


def _train(
    num_epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerEMB,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    project_dir: str | None = None,
):
    def make_checkpoint(
        _accelerator: accelerate.Accelerator, _epoch: int, _step: int
    ):
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(
                project_dir,
                "checkpoints",
                f"epoch{_epoch}_step{_step}",
            )

            logger.info(
                f"EPOCH {_epoch}/{num_epochs}: Saving checkpoint - {checkpoint_dir}"
            )
            _accelerator.save_state(checkpoint_dir)

    def train_loop(
        dataloader: DataLoader,
        _epoch: int,
        steps_per_checkpoint: int | None = None,
    ):
        loss = torch.tensor([0.0])
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
            pass
        else:
            lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])

        model.train()
        for __step, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                initial=0,
                leave=False,
            )
        ):
            pbar.set_postfix_str(
                f"lr={lr_for_print}, "
                f"loss={round(loss.item(), 4)}, "
                f"trailing={round(trailing_loss, 4)}"
            )

            with accelerator.accumulate(model):
                step = __step + 1
                seqs, eos_pos = batch

                seqs = seqs.contiguous()
                bsz = seqs.size(0)
                seqs_flat = seqs.view(2 * bsz, seqs.size(-1))

                outputs = model(seqs_flat)
                z1_full = outputs[0::2]
                z2_full = outputs[1::2]

                batch_indices = torch.arange(bsz, device=z1_full.device)
                eos_pos_1 = eos_pos[:, 0]
                eos_pos_2 = eos_pos[:, 1]

                z1 = z1_full[batch_indices, eos_pos_1]
                z2 = z2_full[batch_indices, eos_pos_2]

                loss = symmetric_nt_xent_loss_cosine(z1, z2)

                # Calculate statistics
                loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
                trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(
                    loss_buffer[-TRAILING_LOSS_STEPS:]
                )
                avg_train_loss = sum(loss_buffer) / len(loss_buffer)

                # Logging
                logger.debug(
                    f"EPOCH {_epoch} STEP {step}: "
                    f"lr={lr_for_print}, "
                    f"loss={round(loss.item(), 4)}, "
                    f"trailing_loss={round(trailing_loss, 4)}, "
                    f"average_loss={round(avg_train_loss, 4)}"
                )

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                    lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

                if steps_per_checkpoint:
                    if step % steps_per_checkpoint == 0:
                        make_checkpoint(
                            _accelerator=accelerator,
                            _epoch=_epoch,
                            _step=step,
                        )

        return avg_train_loss

    def val_loop(dataloader: DataLoader, _epoch: int):
        model.eval()
        val_loss_buffer = []

        with torch.no_grad():
            pbar = tqdm(
                dataloader, desc=f"Validation Epoch {_epoch}", leave=False
            )
            for batch in pbar:
                seqs, eos_pos = batch

                seqs = seqs.contiguous()
                bsz = seqs.size(0)
                seqs_flat = seqs.view(2 * bsz, seqs.size(-1))

                outputs = model(seqs_flat)
                z1_full = outputs[0::2]
                z2_full = outputs[1::2]

                batch_indices = torch.arange(bsz, device=z1_full.device)
                eos_pos_1 = eos_pos[:, 0]
                eos_pos_2 = eos_pos[:, 1]

                z1 = z1_full[batch_indices, eos_pos_1]
                z2 = z2_full[batch_indices, eos_pos_2]

                loss = symmetric_nt_xent_loss_cosine(z1, z2)
                # Gather loss from all devices (if applicable)
                val_loss_buffer.append(
                    accelerator.gather(loss).mean(dim=0).item()
                )

                current_avg_loss = sum(val_loss_buffer) / len(val_loss_buffer)

                pbar.set_postfix_str(f"avg_loss={round(current_avg_loss,4)}")

        avg_val_loss = sum(val_loss_buffer) / len(val_loss_buffer)

        logger.info(
            f"Validation Epoch {_epoch}: average_loss={round(avg_val_loss, 4)}"
        )
        return avg_val_loss

    logger = get_logger(__name__)
    TRAILING_LOSS_STEPS = 100

    for _epoch_num in range(num_epochs):
        train_loop(dataloader=train_dataloader, _epoch=_epoch_num)
        make_checkpoint(
            _accelerator=accelerator, _epoch=_epoch_num + 1, _step=0
        )
        val_loop(dataloader=val_dataloader, _epoch=_epoch_num)


def train(
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    num_epochs: int,
    batch_size: int,
    grad_acc_steps: int,
    project_dir: str | None = None,
    checkpoint_path: str | None = None,
):
    accelerator = accelerate.Accelerator(
        project_dir=project_dir,
        gradient_accumulation_steps=grad_acc_steps,
    )

    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(os.path.join(project_dir))
    else:
        # In other processes, we won't create logs
        project_dir = project_dir or "./experiments"
        logger = get_logger(__name__)

    logger.info(f"Project directory: {project_dir}")
    logger.info(
        f"Training config: epochs={num_epochs}, batch_size={batch_size}, num_workers={num_workers}"
    )

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerEMB(model_config)

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model_state = _load_weight(checkpoint_path)
        model_state = {
            k.replace("_orig_mod.", ""): v for k, v in model_state.items()
        }
        if "lm_head.weight" in model_state.keys():
            del model_state["lm_head.weight"]

        model_state = {
            k.replace("model.", ""): v for k, v in model_state.items()
        }
        model.model.load_state_dict(model_state)
    else:
        logger.info("No checkpoint path provided")

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer, scheduler = get_optim(
        model=model,
        num_epochs=num_epochs,
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

    _train(
        num_epochs=num_epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        project_dir=project_dir,
    )


def test_dataset():
    tokenizer = AbsTokenizer()
    dataset = ContrastiveDataset(
        load_path="/mnt/ssd1/aria/data/mididict-ft_val.jsonl",
        min_number_slice_notes=150,
        max_number_slice_notes=300,
        max_seq_len=1024,
        apply_aug=True,
    )

    for idx, (enc, pos) in enumerate(dataset):
        seq_1 = enc[0].tolist()
        midi_dict_1 = tokenizer.detokenize(tokenizer.decode(seq_1))
        midi_dict_1.to_midi().save("/home/loubb/Dropbox/shared/test1.mid")

        seq_2 = enc[1].tolist()
        midi_dict_2 = tokenizer.detokenize(tokenizer.decode(seq_2))
        midi_dict_2.to_midi().save("/home/loubb/Dropbox/shared/test2.mid")

        print(enc.shape)
        print(pos.shape, pos)
        input("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a model contrastive_embeddings"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--project_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        grad_acc_steps=args.grad_acc_steps,
        project_dir=args.project_dir,
    )

    # test_dataset()
