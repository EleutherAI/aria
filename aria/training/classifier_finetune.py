import torch
import os
import mmap
import argparse
import logging
import accelerate
import json

from aria.config import load_model_config
from aria.utils import _load_weight
from ariautils.tokenizer import AbsTokenizer
from aria.model import TransformerCL, ModelConfig

from torch import nn
from torch.utils.data import DataLoader, Dataset

from accelerate.logging import get_logger
from typing import Callable
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

CATEGORY_TAGS = {
    "genre": {
        "classical": 0,
        "jazz": 1,
    },
    "music_period": {
        "baroque": 0,
        "classical": 1,
        "romantic": 2,
        "impressionist": 3,
    },
    "composer": {
        "beethoven": 0,
        "debussy": 1,
        "brahms": 2,
        "rachmaninoff": 3,
        "schumann": 4,
        "mozart": 5,
        "liszt": 6,
        "bach": 7,
        "chopin": 8,
        "schubert": 9,
    },
    "form": {
        "nocturne": 0,
        "sonata": 1,
        "improvisation": 2,
        "etude": 3,
        "fugue": 4,
        "waltz": 5,
    },
    "pianist": {
        "hisaishi": 0,
        "hancock": 1,
        "bethel": 2,
        "einaudi": 3,
        "clayderman": 4,
        "ryuichi": 5,
        "yiruma": 6,
        "hillsong": 7,
    },
    "emotion": {
        "happy": 0,
        "sad": 1,
        "calm": 2,
        "tense": 3,
    },
}


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


class FinetuningDataset(Dataset):
    def __init__(
        self,
        load_path: str,
        tag_to_id: dict,
        metadata_category: str,
        max_seq_len: int,
        per_file: bool = False,
    ):
        self.load_path = load_path
        self.tag_to_id = tag_to_id
        self.metadata_category = metadata_category
        self.max_seq_len = max_seq_len
        self.per_file = per_file
        self._transform = None
        self.tokenizer = AbsTokenizer()
        self.index = []

        assert metadata_category in CATEGORY_TAGS.keys()
        assert all(
            tag_to_id[_t] == _id
            for _t, _id in CATEGORY_TAGS[metadata_category].items()
        )

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

    def set_transform(self, transform: Callable | list[Callable]):
        if isinstance(transform, Callable):
            self._transform = transform
        elif isinstance(transform, list):
            # Check validity
            for fn in transform:
                assert isinstance(fn, Callable), "Invalid function"

            # Define new transformation function (apply fn in order)
            def _new_transform(x):
                for fn in transform:
                    x = fn(x)
                return x

            self._transform = _new_transform
        else:
            raise ValueError("Must provide function or list of functions.")

    def __getitem__(self, idx: int):
        def _format(tok):
            # Required because json formats tuples into lists
            if isinstance(tok, list):
                return tuple(tok)
            return tok

        pos = self.index[idx]
        self.mmap_obj.seek(pos)
        raw_data = self.mmap_obj.readline().decode("utf-8")
        json_data = json.loads(raw_data)

        metadata = json_data["metadata"]
        tag = metadata[self.metadata_category]

        assert tag in self.tag_to_id, metadata
        tag_tensor = torch.tensor(self.tag_to_id[tag])

        if self.per_file:
            seq_list = json_data["seqs"]
        else:
            seq_list = [json_data["seq"]]

        seq_tensors = []
        pos_tensors = []
        for seq in seq_list:
            seq = [_format(tok) for tok in seq]

            if self._transform:
                seq = self._transform(seq)

            seq = seq[: self.max_seq_len]
            if self.tokenizer.eos_tok not in seq:
                assert self._transform is not None
                seq[-1] = self.tokenizer.eos_tok

            eos_index = seq.index(self.tokenizer.eos_tok)
            pos_tensor = torch.tensor(eos_index)

            assert len(seq) <= self.max_seq_len

            seq = seq + [self.tokenizer.pad_tok] * (self.max_seq_len - len(seq))
            encoded_seq = self.tokenizer.encode(seq)
            seq_tensor = torch.tensor(encoded_seq)

            assert seq_tensor[pos_tensor.item()].item() == 1  # EOS ID check

            seq_tensors.append(seq_tensor)
            pos_tensors.append(pos_tensor)

        seq_tensor = torch.stack(seq_tensors)
        pos_tensor = torch.stack(pos_tensors)

        return seq_tensor, pos_tensor, tag_tensor

    def __len__(self):
        return len(self.index)

    @classmethod
    def export_worker_init_fn(cls):
        def worker_init_fn(worker_id: int):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset

            if hasattr(dataset, "mmap_obj") and dataset.mmap_obj:
                dataset.mmap_obj.close()

            f = open(dataset.load_path, "rb")
            dataset.mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        return worker_init_fn


def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: float = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    total_steps = num_epochs * steps_per_epoch

    if warmup > 0:
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
            total_iters=total_steps - warmup,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lrs, linear_decay_lrs],
            milestones=[warmup],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=end_ratio,
            total_iters=total_steps,
        )

    return optimizer, lr_scheduler


def get_optim(
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
):
    LR = 1e-5
    END_RATIO = 0.1
    WARMUP_STEPS = 0

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
    metadata_category: str,
    tag_to_id: dict,
    batch_size: int,
    num_workers: int,
    apply_aug: bool = False,
    max_seq_len: int = 1024,
):
    train_dataset = FinetuningDataset(
        load_path=train_data_path,
        tag_to_id=tag_to_id,
        metadata_category=metadata_category,
        max_seq_len=max_seq_len,
    )
    val_dataset = FinetuningDataset(
        load_path=val_data_path,
        tag_to_id=tag_to_id,
        metadata_category=metadata_category,
        max_seq_len=max_seq_len,
        per_file=True,
    )

    if apply_aug:
        print("Applying dataset augmentation")
        train_dataset.set_transform(AbsTokenizer().export_data_aug())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=FinetuningDataset.export_worker_init_fn(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=FinetuningDataset.export_worker_init_fn(),
    )

    return train_loader, val_loader


def _train(
    num_epochs: int,
    accelerator: accelerate.Accelerator,
    model: TransformerCL,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tag_to_id: dict,
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

    def train_loop(dataloader: DataLoader, _epoch: int):
        loss = torch.tensor([0.0])
        avg_train_loss = 0
        trailing_loss = 0
        loss_buffer = []

        try:
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])
        except Exception:
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

                seqs, eos_pos, labels = batch
                seqs = seqs.squeeze(1)
                eos_pos = eos_pos.squeeze(1)

                logits = model(seqs)  # (b_sz, s_len, class_size)
                logits = logits[
                    torch.arange(logits.shape[0], device=logits.device), eos_pos
                ]
                loss = loss_fn(logits, labels)

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

        return avg_train_loss

    def val_loop(dataloader: DataLoader, _epoch: int, tag_to_id: dict):
        model.eval()
        pad_id = AbsTokenizer().pad_id
        preds = []
        labels = []

        with torch.inference_mode():
            pbar = tqdm(
                dataloader, desc=f"Validation Epoch {_epoch}", leave=False
            )
            for batch in pbar:
                seqs, pos, tag = batch
                seqs = seqs.squeeze(0)  # (n, max_seq_len)
                pos = pos.squeeze(0)  # (n,)

                logits = model(seqs)  # (n, seq_len, class_size)
                logits = logits[
                    torch.arange(logits.shape[0], device=logits.device), pos
                ]
                probs = torch.softmax(logits, dim=-1)  # (n, class_size)

                non_pad_counts = (
                    (seqs != pad_id).sum(dim=1, keepdim=True).float()
                )
                weighted_probs = probs * non_pad_counts
                aggregated_probs = weighted_probs.sum(dim=0)
                predicted_label = aggregated_probs.argmax().item()

                preds.append(predicted_label)
                labels.append(tag.item())

                tmp_acc = sum(p == t for p, t in zip(preds, labels)) / len(
                    preds
                )
                pbar.set_postfix_str(f"acc={round(tmp_acc, 4)}")

        accuracy = sum(p == t for p, t in zip(preds, labels)) / len(labels)

        # Compute per-class F1 scores
        id_to_tag = {v: k for k, v in tag_to_id.items()}
        # Initialize counts per class
        metrics = {tag: {"TP": 0, "FP": 0, "FN": 0} for tag in tag_to_id.keys()}
        for true_id, pred_id in zip(labels, preds):
            true_tag = id_to_tag[true_id]
            pred_tag = id_to_tag[pred_id]
            if true_id == pred_id:
                metrics[true_tag]["TP"] += 1
            else:
                metrics[true_tag]["FN"] += 1
                metrics[pred_tag]["FP"] += 1

        class_metrics = {}
        f1_scores = []
        for tag, counts in metrics.items():
            TP = counts["TP"]
            FP = counts["FP"]
            FN = counts["FN"]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0
            )
            class_metrics[tag] = {
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        logger.info(
            f"Validation Epoch {_epoch}: accuracy={round(accuracy, 4)}, macro-F1={round(macro_f1, 4)}"
        )
        logger.info(f"Class metrics: {class_metrics}")

        return accuracy, macro_f1, class_metrics

    logger = get_logger(__name__)
    loss_fn = nn.CrossEntropyLoss()
    TRAILING_LOSS_STEPS = 20

    epoch_metrics = []
    for __epoch in range(num_epochs):
        train_loop(dataloader=train_dataloader, _epoch=__epoch)
        acc, macro_f1, class_metrics = val_loop(
            dataloader=val_dataloader, _epoch=__epoch, tag_to_id=tag_to_id
        )
        epoch_metrics.append(
            {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "class_metrics": class_metrics,
            }
        )

    return epoch_metrics


def train(
    model_name: str,
    metadata_category: str,
    apply_aug: bool,
    train_data_path: str,
    val_data_path: str,
    num_workers: int,
    num_epochs: int,
    batch_size: int,
    grad_acc_steps: int,
    project_dir: str | None = None,
    checkpoint_path: str | None = None,
    dataset_size: int | None = None,
):
    accelerator = accelerate.Accelerator(
        project_dir=project_dir,
        gradient_accumulation_steps=grad_acc_steps,
    )

    tag_to_id = CATEGORY_TAGS[metadata_category]

    if accelerator.is_main_process:
        project_dir = setup_project_dir(project_dir)
        logger = setup_logger(os.path.join(project_dir))
    else:
        # In other processes, we won't create logs
        project_dir = project_dir or "./experiments"
        logger = get_logger(__name__)

    logger.info(f"Project directory: {project_dir}")
    logger.info(f"Metadata category: {metadata_category}")
    logger.info(f"Dataset size: {dataset_size}")
    logger.info(f"Applying aug: {apply_aug}")
    logger.info(
        f"Training config:epochs={num_epochs}, batch_size={batch_size}, num_workers={num_workers}"
    )

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model = TransformerCL(model_config)

    assert model_config.class_size == len(tag_to_id.keys())

    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model_state = _load_weight(checkpoint_path)
        model_state = {
            k.replace("_orig_mod.", ""): v for k, v in model_state.items()
        }
        model.load_state_dict(model_state, strict=False)
        torch.nn.init.normal_(
            model.model.tok_embeddings.weight.data[1:2], mean=0.0, std=0.02
        )  # Re-init EOS tok

    else:
        logger.info("No checkpoint path provided")

    model.compile()

    train_dataloader, val_dataloader = get_dataloaders(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        metadata_category=metadata_category,
        tag_to_id=tag_to_id,
        batch_size=batch_size,
        num_workers=num_workers,
        apply_aug=apply_aug,
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

    epoch_metrics = _train(
        num_epochs=num_epochs,
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        tag_to_id=tag_to_id,
        scheduler=scheduler,
        project_dir=project_dir,
    )

    max_accuracy = (
        max(metric["accuracy"] for metric in epoch_metrics)
        if epoch_metrics
        else 0.0
    )
    logger.info(f"Max accuracy: {max_accuracy}")
    results = {
        "metadata_category": metadata_category,
        "dataset_size": dataset_size,
        "epoch_metrics": epoch_metrics,
        "max_accuracy": max_accuracy,
    }
    with open(os.path.join(project_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a model for classification."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--metadata_category", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, required=False)
    parser.add_argument("--apply_aug", action="store_true")
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
        metadata_category=args.metadata_category,
        dataset_size=args.dataset_size,
        apply_aug=args.apply_aug,
        checkpoint_path=args.checkpoint_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        grad_acc_steps=args.grad_acc_steps,
        project_dir=args.project_dir,
    )
