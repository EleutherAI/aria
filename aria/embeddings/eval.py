import torch
import accelerate
import os
import mmap
import time
import json
import functools
import multiprocessing
import copy
import jsonlines
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

from aria.model import ModelConfig, TransformerLM
from aria.config import load_model_config
from aria.utils import _load_weight
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

MODEL_PATH = "/mnt/ssd1/aria/v2/medium-dedupe-pt-cont2/checkpoints/epoch18_step0/model.safetensors"
MAX_SEQ_LEN = 512
TAG_IDS = {"classical": 0, "jazz": 1, "other": 2}
ID_TO_TAG = {v: k for k, v in TAG_IDS.items()}


def chunk_and_pad(lst: list, n: int):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def init_worker():
    global tokenizer
    tokenizer = AbsTokenizer()


def write_entries(writer, entries):
    for entry in entries:
        writer.write(entry)


# The worker function processes a single JSON-lines entry.
def process_entry(
    entry,
    metadata_category: str,
    tag_ids: dict,
    slice_len_notes: int,
    max_seq_len: int,
):
    midi_dict = MidiDict.from_msg_dict(entry)
    metadata_tag = midi_dict.metadata.get(metadata_category, None)

    # Skip metadata tag
    if metadata_tag is None:
        return []
    elif metadata_tag not in tag_ids.keys():
        metadata_tag = "other"

    outputs = []
    for slice_note_msgs in chunk_and_pad(
        lst=midi_dict.note_msgs, n=slice_len_notes
    ):
        if len(slice_note_msgs) < 20:
            break

        slice_midi_dict = copy.deepcopy(midi_dict)
        slice_midi_dict.note_msgs = slice_note_msgs
        slice_midi_dict.metadata = {}
        tokenized_slice = tokenizer.tokenize(slice_midi_dict)
        if tokenizer.eos_tok in tokenized_slice:
            tokenized_slice.remove(tokenizer.eos_tok)
        if tokenizer.dim_tok in tokenized_slice:
            tokenized_slice.remove(tokenizer.dim_tok)

        tokenized_slice = tokenized_slice[:max_seq_len]

        outputs.append({"seq": tokenized_slice, "tag": metadata_tag})

    return outputs


@torch.autocast("cuda", dtype=torch.bfloat16)
@torch.inference_mode()
def get_baseline_embedding(
    seqs: list,
    hook_model: nn.Module,
    hook_max_seq_len: int,
    hook_tokenizer: AbsTokenizer,
    pool_mode: str = "last",  # "last" or "mean"
):
    orig_lengths = [len(seq) for seq in seqs]
    last_tok_positions = [length - 1 for length in orig_lengths]
    seqs = [
        seq + ([hook_tokenizer.pad_tok] * (hook_max_seq_len - len(seq)))
        for seq in seqs
    ]

    enc_seqs = torch.tensor(
        [hook_tokenizer.encode(seq) for seq in seqs], device="cuda"
    )
    hidden_states = hook_model(enc_seqs)

    if pool_mode == "last":
        idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        emb = hidden_states[idx, last_tok_positions].tolist()
    elif pool_mode == "mean":
        pad_id = tokenizer.pad_id
        # Create a mask by comparing enc_seqs to pad_id.
        mask = (enc_seqs != pad_id).unsqueeze(-1).to(hidden_states.dtype)
        # Sum over valid tokens and average.
        sum_hidden = (hidden_states * mask).sum(dim=1)
        valid_counts = mask.sum(dim=1)
        mean_hidden = sum_hidden / valid_counts
        emb = mean_hidden.tolist()
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode}")

    return emb


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str, tag_ids: dict):
        self.load_path = load_path
        self.tag_ids = tag_ids
        self.tokenizer = AbsTokenizer()
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

    def __getitem__(self, idx: int):
        pos = self.index[idx]
        self.mmap_obj.seek(pos)

        raw_data = self.mmap_obj.readline().decode("utf-8")
        json_data = json.loads(raw_data)

        emb = json_data["emb"]
        tag = json_data["tag"]

        assert tag in self.tag_ids
        tag_tensor = torch.tensor(self.tag_ids[tag])
        emb_tensor = torch.tensor(emb)

        return emb_tensor, tag_tensor

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

    @classmethod
    def build(
        cls,
        midi_dataset_load_path: str,
        save_path: str,
        slice_len_notes: int,
        max_seq_len: int,
        metadata_category: str,
        tag_ids: dict,
        batch_size: int,
        embedding_hook: Callable,
        **embedding_hook_kwargs,
    ):
        assert os.path.isfile(midi_dataset_load_path)
        assert os.path.isfile(save_path) is False

        with jsonlines.open(
            midi_dataset_load_path, "r"
        ) as midi_dataset, jsonlines.open(save_path, "w") as writer:

            cnt = 0
            buffer = deque()
            write_executor = ThreadPoolExecutor(max_workers=1)
            with multiprocessing.Pool(
                processes=8, initializer=init_worker
            ) as pool:
                for result in pool.imap_unordered(
                    functools.partial(
                        process_entry,
                        metadata_category=metadata_category,
                        tag_ids=tag_ids,
                        slice_len_notes=slice_len_notes,
                        max_seq_len=max_seq_len,
                    ),
                    midi_dataset,
                    chunksize=10,
                ):

                    cnt += 1
                    if cnt % 500 == 0:
                        print(f"Completed {cnt}")

                    for entry in result:
                        buffer.append(entry)

                    # Inside your processing loop:
                    if len(buffer) >= batch_size:
                        _buffer = [buffer.popleft() for _ in range(batch_size)]
                        _seqs = [entry["seq"] for entry in _buffer]
                        _tags = [entry["tag"] for entry in _buffer]
                        _embs = embedding_hook(
                            seqs=_seqs, **embedding_hook_kwargs
                        )

                        # Prepare the write objects
                        write_objs = [
                            {"seq": _seq, "emb": _emb, "tag": _tag}
                            for _seq, _emb, _tag in zip(_seqs, _embs, _tags)
                        ]

                        write_executor.submit(write_entries, writer, write_objs)

            if buffer:
                _seqs = [entry["seq"] for entry in buffer]
                _tags = [entry["tag"] for entry in buffer]
                _embs = embedding_hook(seqs=_seqs, **embedding_hook_kwargs)
                for _seq, _tag, _emb in zip(_seqs, _tags, _embs):
                    writer.write({"seq": _seq, "emb": _emb, "tag": _tag})


def _get_optim(
    lr: float,
    model: nn.Module,
    total_steps: int,
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
        total_iters=total_steps - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


class ClassifierHead(nn.Module):
    def __init__(self, d_emb: int, hidden_dim: int, num_class: int):
        super().__init__()
        self.fc1 = nn.Linear(d_emb, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        logits = self.fc2(x)
        return logits


def _train(
    accelerator: accelerate.Accelerator,
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    TRAILING_LOSS_STEPS = 100
    loss = torch.tensor([0.0])
    trailing_loss = 0
    lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])
    loss_buffer = []

    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for __step, batch in (
        pbar := tqdm(enumerate(train_dataloader), leave=False)
    ):
        pbar.set_postfix_str(
            f"lr={lr_for_print}, "
            f"loss={round(loss.item(), 4)}, "
            f"trailing={round(trailing_loss, 4)}"
        )

        emb, tag_ids = batch
        tag_ids = tag_ids.view(-1)

        logits = model(emb)
        loss = loss_fn(logits, tag_ids)

        loss_buffer.append(accelerator.gather(loss).mean(dim=0).item())
        trailing_loss = sum(loss_buffer[-TRAILING_LOSS_STEPS:]) / len(
            loss_buffer[-TRAILING_LOSS_STEPS:]
        )

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
            lr_for_print = "{:.2e}".format(scheduler.get_last_lr()[0])

    if accelerator.is_main_process:
        accelerator.save_state("/mnt/ssd1/aria/test")

    return model


def train_classifier(
    emb_d: int,
    train_dataset: EvaluationDataset,
    tag_ids: dict,
    batch_size: int,
):
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=EvaluationDataset.export_worker_init_fn(),
    )

    model = ClassifierHead(
        d_emb=emb_d,
        hidden_dim=emb_d,
        num_class=len(tag_ids.keys()),
    )
    optimizer, scheduler = _get_optim(
        lr=3e-4,
        model=model,
        total_steps=len(train_dataloader),
    )
    accelerator = accelerate.Accelerator()

    model, train_dataloader, optimizer, scheduler = accelerator.prepare(
        model,
        train_dataloader,
        optimizer,
        scheduler,
    )

    return _train(
        accelerator=accelerator,
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def eval(model: nn.Module):
    val_dataset = EvaluationDataset(
        load_path="/mnt/ssd1/aria/data/val.jsonl",
        tag_ids=TAG_IDS,
    )
    model = model.cpu()

    correct = 0
    total = 0
    dist = {
        "classical": 0,
        "jazz": 0,
        "other": 0,
    }
    for midi_emb, tag_id in val_dataset:
        with torch.no_grad():
            logits = model(torch.tensor(midi_emb.view(1, -1)))
            probs = F.softmax(logits)
            pred_tag_id = probs.argmax(dim=-1).item()
            dist[ID_TO_TAG[tag_id.item()]] += 1

            if ID_TO_TAG[tag_id.item()] == "other":
                continue

            if pred_tag_id == tag_id.item():
                correct += 1
            total += 1

            print(ID_TO_TAG[tag_id.item()], ID_TO_TAG[pred_tag_id])
            input("...")

    print(f"Total accuracy: {correct/total}")
    print(f"Label distribution: {dist}")


if __name__ == "__main__":
    tokenizer = AbsTokenizer()

    dataset = EvaluationDataset(
        load_path="/mnt/ssd1/aria/data/train.jsonl",
        tag_ids=TAG_IDS,
    )

    model = train_classifier(
        emb_d=1536,
        train_dataset=dataset,
        batch_size=32,
        tag_ids=TAG_IDS,
    )
    eval(model=model)

    # model_state = _load_weight(MODEL_PATH, "cuda")
    # model_state = {
    #     k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    # }
    # pretrained_model_config = ModelConfig(**load_model_config("medium"))
    # pretrained_model_config.set_vocab_size(tokenizer.vocab_size)
    # pretrained_model_config.grad_checkpoint = False
    # pretrained_model = TransformerLM(pretrained_model_config)
    # pretrained_model.load_state_dict(model_state)
    # pretrained_model.eval()

    # EvaluationDataset.build(
    #     midi_dataset_load_path="/mnt/ssd1/aria/data/mididict-ft_train.jsonl",
    #     save_path="/mnt/ssd1/aria/data/train.jsonl",
    #     max_seq_len=MAX_SEQ_LEN,
    #     slice_len_notes=165,
    #     metadata_category="genre",
    #     tag_ids=TAG_IDS,
    #     batch_size=128,
    #     embedding_hook=functools.partial(
    #         get_baseline_embedding, pool_mode="mean"
    #     ),
    #     hook_model=pretrained_model.model.cuda(),
    #     hook_max_seq_len=512,
    #     hook_tokenizer=tokenizer,
    # )
