import torch
import accelerate
import os
import mmap
import json
import time
import functools
import multiprocessing
import queue
import copy
import jsonlines
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

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
LEARNING_RATE = 3e-4


def model_forward(
    model: nn.Module,
    idxs: torch.Tensor,
):
    return model(idxs)


def write_entries(writer, entries):
    for entry in entries:
        writer.write(entry)


def get_chunks(note_msgs: list, chunk_len: int):
    return [
        note_msgs[i : i + chunk_len]
        for i in range(0, len(note_msgs), chunk_len)
    ]


def process_entry(
    entry: MidiDict | dict,
    slice_len_notes: int,
    max_seq_len: int,
    tokenizer: AbsTokenizer,
):
    if isinstance(entry, dict):
        midi_dict = MidiDict.from_msg_dict(entry)
    else:
        midi_dict = entry

    outputs = []
    for slice_note_msgs in get_chunks(
        note_msgs=midi_dict.note_msgs, chunk_len=slice_len_notes
    ):
        if len(slice_note_msgs) == 0:
            break

        slice_midi_dict = copy.deepcopy(midi_dict)
        slice_midi_dict.note_msgs = slice_note_msgs
        slice_midi_dict.metadata = {}
        tokenized_slice = tokenizer.tokenize(slice_midi_dict)
        if tokenizer.dim_tok in tokenized_slice:
            tokenized_slice.remove(tokenizer.dim_tok)

        tokenized_slice = tokenized_slice[:max_seq_len]

        outputs.append({"seq": tokenized_slice, "metadata": midi_dict.metadata})

    return outputs


def _pad_seq(seq: list, tokenizer: AbsTokenizer, max_seq_len: int):
    seq = seq[:max_seq_len]
    seq += [tokenizer.pad_tok] * (max_seq_len - len(seq))

    if tokenizer.eos_tok not in seq:
        seq[-1] = tokenizer.eos_tok

    return seq


@torch.autocast(
    "cuda",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
@torch.inference_mode()
def get_aria_contrastive_embedding(
    seqs: list,
    hook_model: nn.Module,
    hook_max_seq_len: int,
    hook_tokenizer: AbsTokenizer,
    hook_model_forward: Callable,
    hook_max_batch_size: int = 64,
):
    all_emb = []

    for i in range(0, len(seqs), hook_max_batch_size):
        batch_seqs = seqs[i : i + hook_max_batch_size]
        padded_seqs = [
            _pad_seq(
                seq=seq, tokenizer=hook_tokenizer, max_seq_len=hook_max_seq_len
            )
            for seq in batch_seqs
        ]
        eos_positions = [
            seq.index(hook_tokenizer.eos_tok) for seq in padded_seqs
        ]
        enc_seqs = torch.tensor(
            [hook_tokenizer.encode(seq) for seq in padded_seqs], device="cuda"
        )
        hidden_states = hook_model_forward(model=hook_model, idxs=enc_seqs)
        idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        batch_emb = hidden_states[idx, eos_positions].tolist()
        all_emb.extend(batch_emb)

    return all_emb


def get_mert_embedding(
    seqs: list,
    hook_model: nn.Module,
    hook_processor,
    hook_tokenizer: AbsTokenizer,
    hook_pianoteq_exec_path: str,
    hook_pianoteq_num_procs: int,
):
    from aria.eval.mert.emb import (
        seq_to_audio_path,
        compute_audio_embedding,
    )

    with multiprocessing.Pool(hook_pianoteq_num_procs) as pool:
        audio_paths = pool.imap(
            functools.partial(
                seq_to_audio_path,
                tokenizer=hook_tokenizer,
                pianoteq_exec_path=hook_pianoteq_exec_path,
            ),
            seqs,
        )

        emb = [
            compute_audio_embedding(
                audio_path=path,
                model=hook_model,
                processor=hook_processor,
                delete_audio=True,
            ).tolist()
            for path in audio_paths
        ]

    return emb


def get_clamp3_embedding(
    seqs: list,
    hook_model: nn.Module,
    hook_patchilizer,
    hook_tokenizer: AbsTokenizer,
):
    from aria.eval.m3.emb import get_midi_embedding

    emb = [
        get_midi_embedding(
            mid=hook_tokenizer.detokenize(seq).to_midi(),
            model=hook_model,
            patchilizer=hook_patchilizer,
            get_global=True,
        ).tolist()
        for seq in seqs
    ]

    return emb


@torch.autocast("cuda", dtype=torch.bfloat16)
@torch.inference_mode()
def get_baseline_embedding(
    seqs: list,
    hook_model: nn.Module,
    hook_max_seq_len: int,
    hook_tokenizer: AbsTokenizer,
    pool_mode: str = "last",  # "last" or "mean"
):
    for seq in seqs:
        if hook_tokenizer.eos_tok in seq:
            seq.remove(hook_tokenizer.eos_tok)

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
        pad_id = hook_tokenizer.pad_id
        # Create a mask by comparing enc_seqs to pad_id
        mask = (enc_seqs != pad_id).unsqueeze(-1).to(hidden_states.dtype)
        # Sum over valid tokens and average
        sum_hidden = (hidden_states * mask).sum(dim=1)
        valid_counts = mask.sum(dim=1)
        mean_hidden = sum_hidden / valid_counts
        emb = mean_hidden.tolist()
    else:
        raise ValueError(f"Unsupported pool_mode: {pool_mode}")

    return emb


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, load_path: str, tag_to_id: dict, metadata_category: str):
        self.load_path = load_path
        self.tag_to_id = tag_to_id
        self.metadata_category = metadata_category
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
        metadata = json_data["metadata"]
        tag = metadata.get(self.metadata_category, "other")
        tag = tag if tag in self.tag_to_id.keys() else "other"

        assert tag in self.tag_to_id, metadata
        tag_tensor = torch.tensor(self.tag_to_id[tag])
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
        batch_size: int,
        embedding_hook: Callable,
        per_file_embeddings: bool = False,
        **embedding_hook_kwargs,
    ):
        def batch_producer(
            results_queue: queue.Queue,
            batch_queue: queue.Queue,
            batch_size: int,
            total_workers: int,
            per_file: bool = False,
        ):
            buffer = []
            termination_signals = 0

            while termination_signals < total_workers:
                if batch_queue.qsize() > 10:
                    time.sleep(0.25)

                try:
                    result = results_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                if result is None:
                    termination_signals += 1
                    continue

                if per_file:
                    assert all(
                        "abs_load_path" in r["metadata"].keys() for r in result
                    )
                    buffer.extend(result)
                    if len(buffer) > 2 * batch_size:
                        print(
                            f"WARNING: Generated batch of size {len(buffer)} (batch_size={batch_size})"
                        )
                    if len(buffer) >= batch_size:
                        batch_queue.put(buffer)
                        buffer = []
                else:
                    buffer.extend(result)
                    while len(buffer) >= batch_size:
                        batch_queue.put(buffer[:batch_size])
                        buffer = buffer[batch_size:]

            if buffer:
                batch_queue.put(buffer)

        def producer(
            midi_dataset_load_path: str,
            midi_dict_queue: queue.Queue,
            num_workers: int,
        ):
            cnt = 0
            with jsonlines.open(midi_dataset_load_path, "r") as midi_dataset:
                for midi_dict in midi_dataset:
                    while midi_dict_queue.qsize() >= 1000:
                        time.sleep(0.1)
                    midi_dict_queue.put(midi_dict)
                    cnt += 1

                    if cnt % 500 == 0:
                        print(f"Finished {cnt}")

            for _ in range(num_workers):
                midi_dict_queue.put(None)

        def worker(
            midi_dict_queue: queue.Queue,
            results_queue: queue.Queue,
            slice_len_notes: int,
            max_seq_len: int,
        ):
            tokenizer = AbsTokenizer()

            while True:
                midi_dict = midi_dict_queue.get()
                if midi_dict is None:
                    results_queue.put(None)
                    break

                while results_queue.qsize() > 250:
                    time.sleep(0.5)

                _result = process_entry(
                    entry=midi_dict,
                    slice_len_notes=slice_len_notes,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer,
                )
                results_queue.put(_result)

        assert os.path.isfile(midi_dataset_load_path)
        assert os.path.isfile(save_path) is False

        TOTAL_WORKERS = 8
        write_executor = ThreadPoolExecutor(max_workers=1)
        results_queue = multiprocessing.Queue()
        midi_dict_queue = multiprocessing.Queue()
        batch_queue = multiprocessing.Queue()
        producer_process = multiprocessing.Process(
            target=producer,
            args=(midi_dataset_load_path, midi_dict_queue, TOTAL_WORKERS),
        )
        batch_producer_process = multiprocessing.Process(
            target=batch_producer,
            args=(
                results_queue,
                batch_queue,
                batch_size,
                TOTAL_WORKERS,
                per_file_embeddings,
            ),
        )
        worker_processes = [
            multiprocessing.Process(
                target=worker,
                args=(
                    midi_dict_queue,
                    results_queue,
                    slice_len_notes,
                    max_seq_len,
                ),
            )
            for _ in range(TOTAL_WORKERS)
        ]

        producer_process.start()
        batch_producer_process.start()
        for p in worker_processes:
            p.start()

        with jsonlines.open(save_path, "w") as writer:
            while batch_producer_process.is_alive() or not batch_queue.empty():
                try:
                    batch = batch_queue.get(timeout=0.01)

                    _seqs = [item["seq"] for item in batch]
                    _metadata = [item["metadata"] for item in batch]
                    _embs = embedding_hook(seqs=_seqs, **embedding_hook_kwargs)

                    if not per_file_embeddings:
                        write_objs = [
                            {"seq": s, "emb": e, "metadata": m}
                            for s, e, m in zip(_seqs, _embs, _metadata)
                        ]
                    else:
                        # Calculate per-file emb by averaging over abs_load_path embs
                        groups = {}
                        for seq, emb, meta in zip(_seqs, _embs, _metadata):
                            file_path = meta["abs_load_path"]
                            if file_path not in groups:
                                groups[file_path] = {
                                    "seqs": [],
                                    "embs": [],
                                    "metadata": meta,
                                }
                            groups[file_path]["seqs"].append(seq)
                            groups[file_path]["embs"].append(emb)

                        write_objs = []
                        for file_path, data in groups.items():
                            avg_emb = (
                                torch.tensor(data["embs"]).mean(dim=0).tolist()
                            )
                            write_objs.append(
                                {
                                    "seqs": data["seqs"],
                                    "emb": avg_emb,
                                    "metadata": data["metadata"],
                                }
                            )

                    write_executor.submit(write_entries, writer, write_objs)

                except queue.Empty:
                    continue

            write_executor.shutdown(wait=True)


def _get_optim(
    model: nn.Module,
    total_steps: int,
    warmup: int = 100,
    end_ratio: int = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
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
    def __init__(self, d_emb: int, num_class: int):
        super().__init__()
        self.linear = nn.Linear(d_emb, num_class)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


def _train(
    accelerator: accelerate.Accelerator,
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
):
    TRAILING_LOSS_STEPS = 100
    loss = torch.tensor([0.0])
    trailing_loss = 0
    lr_for_print = "{:.2e}".format(optimizer.param_groups[-1]["lr"])
    loss_buffer = []

    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(num_epochs):
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

    return model


def train_classifier(
    embedding_dimension: int,
    train_dataset_path: str,
    metadata_category: str,
    tag_to_id: dict,
    batch_size: int,
    num_epochs: int = 1,
):
    train_dataset = EvaluationDataset(
        load_path=train_dataset_path,
        tag_to_id=tag_to_id,
        metadata_category=metadata_category,
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=24,
        worker_init_fn=EvaluationDataset.export_worker_init_fn(),
    )

    model = ClassifierHead(
        d_emb=embedding_dimension,
        num_class=len(tag_to_id.keys()),
    )
    optimizer, scheduler = _get_optim(
        model=model,
        total_steps=num_epochs * len(train_dataloader),
    )
    accelerator = accelerate.Accelerator(cpu=True)

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
        num_epochs=num_epochs,
    )


def evaluate_classifier(
    model: nn.Module,
    evaluation_dataset_path: str,
    metadata_category: str,
    tag_to_id: dict,
):
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    val_dataset = EvaluationDataset(
        load_path=evaluation_dataset_path,
        tag_to_id=tag_to_id,
        metadata_category=metadata_category,
    )
    model = model.cpu().eval()

    dist = {k: {"correct": 0, "total": 0} for k in tag_to_id.keys()}
    pred_dist = {k: 0 for k in tag_to_id.keys()}

    for midi_emb, tag_id in val_dataset:
        with torch.no_grad():
            logits = model(torch.tensor(midi_emb.view(1, -1)))
            probs = F.softmax(logits, dim=-1)
            pred_tag_id = probs.argmax(dim=-1).item()

            true_tag = id_to_tag[tag_id.item()]
            pred_tag = id_to_tag[pred_tag_id]

            dist[true_tag]["total"] += 1
            pred_dist[pred_tag] += 1

            if pred_tag_id == tag_id.item():
                dist[true_tag]["correct"] += 1

    total_correct = sum(v["correct"] for v in dist.values())
    total_samples = sum(v["total"] for v in dist.values())
    overall_accuracy = total_correct / total_samples

    class_metrics = {}
    f1_scores = []
    for tag in tag_to_id.keys():
        TP = dist[tag]["correct"]
        FN = dist[tag]["total"] - TP
        FP = pred_dist[tag] - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        tag_accuracy = TP / dist[tag]["total"] if dist[tag]["total"] > 0 else 0
        class_metrics[tag] = {
            "accuracy": tag_accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    results = {
        "accuracy": overall_accuracy,
        "F1-macro": macro_f1,
        "class_wise": class_metrics,
    }

    return results
