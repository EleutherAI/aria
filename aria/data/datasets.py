"""Contains classes and utilities for building and processing datasets."""

import json
import os
import re
import mmap
import jsonlines
import logging
import random
import torch
import functools
import shutil

from pathlib import Path
from typing import Callable, Iterable
from collections import defaultdict
from multiprocessing import Pool, Process, Queue, get_start_method

from aria.config import load_config
from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict, get_test_fn


def setup_logger():
    # Get logger and reset all handlers
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class MidiDataset:
    """Container for datasets of MidiDict objects.

    Can be used to save, load, and build, datasets of MidiDict objects.

    Args:
        entries (list[MidiDict] | Iterable): MidiDict objects to be stored.
    """

    def __init__(self, entries: list[MidiDict] | Iterable):
        self.entries = entries

    def __len__(self):
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        return len(self.entries)

    def __getitem__(self, ind: int):
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        return self.entries[ind]

    def __iter__(self):
        yield from self.entries

    def shuffle(self):
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        random.shuffle(self.entries)

    def save(self, save_path: str):
        """Saves dataset to JSON file."""

        with jsonlines.open(save_path, mode="w") as writer:
            for midi_dict in self.entries:
                writer.write(midi_dict.get_msg_dict())

    @classmethod
    def load(cls, load_path: str):
        """Loads dataset (into memory) from JSONL file."""
        with jsonlines.open(load_path) as reader:
            _entries = [MidiDict.from_msg_dict(_) for _ in reader]

        return cls(_entries)

    @classmethod
    def get_generator(cls, load_path: str):
        """Given a MidiDataset JSONL file, returns a MidiDict generator.

        This generator must be reloaded each time you want to iterate over the
        file. Internally it iterating over the jsonl file located at load_path.
        """

        def generator():
            with jsonlines.open(load_path, "r") as midi_dataset:
                for entry in midi_dataset:
                    try:
                        midi_dict = MidiDict.from_msg_dict(entry)
                    except Exception as e:
                        logging.info(f"Failed to load MidiDict from file: {e}")
                    else:
                        yield midi_dict

        return generator()

    @classmethod
    def split_from_file(
        cls,
        load_path: str,
        train_val_ratio: float = 0.95,
        repeatable: bool = False,
        overwrite: bool = False,
    ):
        """Splits MidiDataset JSONL file into train/val split."""
        logger = setup_logger()
        path = Path(load_path)
        train_save_path = path.with_name(f"{path.stem}_train{path.suffix}")
        val_save_path = path.with_name(f"{path.stem}_val{path.suffix}")

        if not overwrite:
            if os.path.isfile(train_save_path) is True:
                raise FileExistsError(
                    f"File at {train_save_path} already exists."
                )
            if os.path.isfile(val_save_path) is True:
                raise FileExistsError(
                    f"File at {val_save_path} already exists."
                )

        if repeatable:
            random.seed(42)

        idx_original, idx_train, idx_val = 0, 0, 0

        logger.info(f"Creating train/val split with ratio {train_val_ratio}")
        with (
            jsonlines.open(load_path) as dataset,
            jsonlines.open(train_save_path, mode="w") as train_dataset,
            jsonlines.open(val_save_path, mode="w") as val_dataset,
        ):
            for entry in dataset:
                idx_original += 1
                if random.uniform(0, 1) <= train_val_ratio:
                    idx_train += 1
                    train_dataset.write(entry)
                else:
                    idx_val += 1
                    val_dataset.write(entry)

        logger.info(
            f"Succesfully split into train ({idx_train}) and validation ({idx_val}) sets"
        )

    @classmethod
    def build(
        cls,
        dir: str,
        recur: bool = False,
        manual_metadata: dict = {},
        shuffle: bool = True,
    ):
        """Builds are returns a MidiDataset - see build_mididict_dataset."""
        valid_metadata = load_config()["data"]["metadata"]["manual"]
        for k, v in manual_metadata.items():
            assert k in valid_metadata.keys(), f"{manual_metadata} is invalid"
            assert v in valid_metadata[k], f"{manual_metadata} is invalid"

        return cls(
            build_mididict_dataset(
                dir=dir,
                recur=recur,
                manual_metadata=manual_metadata,
                shuffle=shuffle,
            )
        )

    @classmethod
    def build_to_file(
        cls,
        dir: str,
        save_path: str,
        recur: bool = False,
        overwrite: bool = False,
        manual_metadata: dict = {},
        shuffle: bool = True,
    ):
        """Builds MidiDataset to a JSONL file - see build_mididict_dataset.

        This function will not return a MidiDataset object. It is well suited
        for situations where the resulting MidiDataset will not fit in the
        system's memory. Other than this difference, it is identical to
        MidiDataset.build.
        """
        valid_metadata = load_config()["data"]["metadata"]["manual"]
        for k, v in manual_metadata.items():
            assert k in valid_metadata.keys(), f"{manual_metadata} is invalid"
            assert v in valid_metadata[k], f"{manual_metadata} is invalid"

        build_mididict_dataset(
            dir=dir,
            recur=recur,
            stream_save_path=save_path,
            overwrite=overwrite,
            manual_metadata=manual_metadata,
            shuffle=shuffle,
        )

    @classmethod
    def combine_datasets_from_file(cls, *args: str, output_path: str):
        """Utility for concatenating JSONL files, checking for duplicates"""
        logger = setup_logger()

        for input_path in args:
            assert os.path.isfile(input_path), f"{input_path} doesn't exist"

        dupe_cnt = 0
        hashes = {}
        with jsonlines.open(output_path, mode="w") as f_out:
            for input_path in args:
                assert (
                    os.path.splitext(input_path)[-1] == ".jsonl"
                ), "invalid dataset path"

                with jsonlines.open(input_path, mode="r") as f_in:
                    for msg_dict in f_in:
                        midi_dict = MidiDict.from_msg_dict(msg_dict)
                        midi_dict_hash = midi_dict.calculate_hash()
                        if hashes.get(midi_dict_hash, False) is not False:
                            dupe_cnt += 1
                        else:
                            f_out.write(msg_dict)
                            hashes[midi_dict_hash] = True
                logger.info(f"Finished processing: {input_path}")
                logger.info(
                    f"{len(hashes)} unique midi_dicts and {dupe_cnt} duplicates so far"
                )

        logger.info(
            f"Found {len(hashes)} unique midi_dicts and {dupe_cnt} duplicates"
        )


def _get_mididict(path: Path):
    # This function is only intended to be used as a process target during the
    # multi-processing in build_mididict_dataset. It returns a tuple of the form
    # (bool, (MidiDict, str, Path)) where the first element determines if the
    # loaded MidiDict was succesfully preprocessed.

    def _run_tests(_mid_dict: MidiDict):
        failed_tests = []
        for test_name, test_config in config["tests"].items():
            if test_config["run"] is True:
                test_fn = get_test_fn(test_name)
                test_args = test_config["args"]

                test_res, val = test_fn(_mid_dict, **test_args)
                if test_res is False:
                    failed_tests.append((test_name, val))

        return failed_tests

    def _preprocess_mididict(_mid_dict: MidiDict):
        for fn_name, fn_config in config["pre_processing"].items():
            if fn_config["run"] is True:
                fn_args = fn_config["args"]
                getattr(_mid_dict, fn_name)(fn_args)

                try:
                    # Note fn_args is passed as a dict, not unpacked as kwargs
                    getattr(_mid_dict, fn_name)(fn_args)
                except:
                    logger.error(
                        f"Error finding preprocessing function for {fn_name}"
                    )

        return _mid_dict

    logger = setup_logger()
    config = load_config()["data"]

    try:
        mid_dict = MidiDict.from_midi(mid_path=path)
    except Exception as e:
        logger.error(f"Failed to load MIDI at {path}: {e}")
        return False, None

    mid_hash = mid_dict.calculate_hash()
    failed_tests = _run_tests(mid_dict)
    if failed_tests:
        logger.info(
            f"MIDI at {path} failed preprocessing tests: {failed_tests} "
        )
        return False, None
    else:
        return True, (_preprocess_mididict(mid_dict), mid_hash, path)


def build_mididict_dataset(
    dir: str,
    recur: bool = False,
    stream_save_path: str = None,
    overwrite: bool = False,
    manual_metadata: dict = {},
    shuffle: bool = True,
):
    """Builds dataset of MidiDicts.

    During the build process, successfully parsed MidiDicts can be filtered and
    preprocessed. This can be customized by modifying the config.json file.

    Args:
        dir (str): Directory to index from.
        recur (bool): If True, recursively search directories for MIDI files.
            Defaults to False.
        stream_save_path (str): If True, stream the dictionaries directly to a
            JSONL file instead of returning them as a list. This option is
            appropriate when processing very large numbers of MIDI files.
        overwrite (bool): If True, overwrite file at stream_save_path when
            streaming.
        manual_metadata (dict): Metadata tags to uniformly apply.
        shuffle (dict): Metadata tags to apply uniformly.

    Returns:
        list[MidiDict]: List of parsed, filtered, and preprocessed MidiDicts.
            This is only returned if stream_save_path is not provided.
    """

    def _get_mididicts_mp(_paths):
        with Pool() as pool:
            results = pool.imap(_get_mididict, _paths)
            seen_hashes = defaultdict(list)
            dupe_cnt = 0
            failed_cnt = 0
            for idx, (success, result) in enumerate(results):
                if idx % 50 == 0 and idx != 0:
                    logger.info(f"Processed MIDI files: {idx}/{num_paths}")

                if not success:
                    failed_cnt += 1
                    continue
                else:
                    mid_dict, mid_hash, mid_path = result

                if seen_hashes.get(mid_hash):
                    logger.info(
                        f"MIDI located at '{mid_path}' is a duplicate - already"
                        f" seen at: {seen_hashes[mid_hash][0]}"
                    )
                    seen_hashes[mid_hash].append(str(mid_path))
                    dupe_cnt += 1
                else:
                    seen_hashes[mid_hash].append(str(mid_path))
                    yield mid_dict

        logger.info(f"Total duplicates: {dupe_cnt}")
        logger.info(
            f"Total processing fails (tests or otherwise): {failed_cnt}"
        )

    logger = setup_logger()
    if get_start_method() == "spawn":
        logger.warning(
            'The current multiprocessing start method is "spawn", this '
            "will slow down dataset building"
        )

    paths = []
    if recur is True:
        paths += Path(dir).rglob(f"*.mid")
        paths += Path(dir).rglob(f"*.midi")
    else:
        paths += Path(dir).glob(f"*.mid")
        paths += Path(dir).glob(f"*.midi")

    num_paths = len(paths)
    if num_paths == 0:
        raise FileNotFoundError(
            "Directory contains no files matching *.mid or *.midi"
        )
    if shuffle is True:
        logger.info(f"Shuffling {num_paths} paths")
        random.shuffle(paths)

    cnt = 0
    if stream_save_path is None:
        # Not streaming -> return entries directly
        entries = []
        for entry in _get_mididicts_mp(_paths=paths):
            # manual_metadata should already be validated
            for k, v in manual_metadata.items():
                # Only add if it doesn't exist, stops overwriting
                if entry.metadata.get(k) is None:
                    entry.metadata[k] = v

            cnt += 1
            entries.append(entry)

        return entries
    else:
        # Streaming -> write to file instead of returning anything
        if overwrite is False and os.path.isfile(stream_save_path) is True:
            raise FileExistsError(f"File at {stream_save_path} already exists.")

        with jsonlines.open(stream_save_path, mode="w") as writer:
            for entry in _get_mididicts_mp(paths):
                # manual_metadata should already be validated
                for k, v in manual_metadata.items():
                    # Only add if it doesn't exist, stops overwriting
                    if entry.metadata.get(k) is None:
                        entry.metadata[k] = v

                cnt += 1
                writer.write(entry.get_msg_dict())

    logger.info(
        f"Finished - added {cnt}/{len(paths)} found MIDI files to dataset."
    )


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.logger = setup_logger()
        self._transform = None
        self.config = None
        self.max_seq_len = None

        self.file_buff = None
        self.file_mmap = None
        self.index = None

    def init_epoch(self, epoch_num: int | None = None):
        raise NotImplementedError

    def build(**kwargs):
        raise NotImplementedError

    @classmethod
    def get_config_from_path(cls, path: str):
        """Returns config dict from dataset file/directory.

        If a directory provided, it is assumed t"""

        def _get_config_from_fp(_path):
            # Finetuning Dataset
            return FinetuningDataset.get_config_from_path(path=_path)

        def _get_config_from_dir(_path):
            # Pretraining Dataset
            return PretrainingDataset.get_config_from_path(path=_path)

        if os.path.isfile(path):
            return _get_config_from_fp(path)
        elif os.path.isdir(path):
            return _get_config_from_dir(path)
        else:
            raise FileNotFoundError("Invalid path provided")

    def close(self):
        if self.file_buff:
            self.file_buff.close()
        if self.file_mmap:
            self.file_mmap.close()

    def __del__(self):
        self.close()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        def _format(tok):
            # This is required because json formats tuples into lists
            if isinstance(tok, list):
                return tuple(tok)
            return tok

        self.file_mmap.seek(self.index[idx])

        _debug = self.file_mmap.readline()
        seq = json.loads(_debug)  # Load raw seq
        seq = [_format(tok) for tok in seq]  # Format into hashable
        if self._transform:
            seq = self._transform(seq)  # Data augmentation

        src = seq
        tgt = seq[1:] + [self.tokenizer.pad_tok]

        # Fine till here
        return self.tokenizer.encode(src), self.tokenizer.encode(tgt)

    def check_config(self):
        def _check_config():
            assert self.config["tokenizer_name"] == self.tokenizer.name
            for k, v in self.config["tokenizer_config"].items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        assert _v == self.tokenizer.config[k][_k]
                elif isinstance(v, str) or isinstance(v, int):
                    assert v == self.tokenizer.config[k]

        # Check self.tokenizers is the same as the one used to generate file
        # This logic could use a refactor (maybe use deepdict?)
        self.file_mmap.seek(0)
        buffer = self.file_mmap.readline()
        try:
            self.config = json.loads(buffer)
            self.max_seq_len = self.config["max_seq_len"]
            _check_config()
        except AssertionError as e:
            self.logger.error(
                "Tokenizer config setting don't match those in file"
            )
            raise e
        except Exception as e:
            self.logger.error(
                "Processing tokenizer config resulted in an error"
            )
            raise e

    def _build_index(self):
        # Skip first line containing config
        self.file_mmap.seek(0)
        self.file_mmap.readline()

        index = []
        while True:
            pos = self.file_mmap.tell()
            line_buffer = self.file_mmap.readline()
            if line_buffer == b"":
                break
            else:
                index.append(pos)

        self.logger.debug(f"Finished indexing {len(index)} sequences")

        return index

    def set_transform(self, transform: Callable | list[Callable]):
        """Sets data augmentation transformation functions.

        Args:
            transform (Callable | list[Callable]): Transformation function(s).
                Provided functions are expected to be list[str | tuple] ->
                list[str | tuple].
        """
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


def _get_seqs(_entry: MidiDict | dict, _tokenizer: Tokenizer):
    logger = setup_logger()

    if isinstance(_entry, str):
        _midi_dict = MidiDict.from_msg_dict(json.loads(_entry.rstrip()))
    elif isinstance(_entry, dict):
        _midi_dict = MidiDict.from_msg_dict(_entry)
    elif isinstance(_entry, MidiDict):
        _midi_dict = _entry
    else:
        raise Exception

    try:
        _tokenized_seq = _tokenizer.tokenize(_midi_dict)
    except Exception as e:
        logger.info(f"Skipping midi_dict: {e}")
        return
    else:
        if _tokenizer.unk_tok in _tokenized_seq:
            logger.warning("Unknown token seen while tokenizing midi_dict")
        return _tokenized_seq


def _worker(input_queue: Queue, output_queue: Queue, _tokenizer: Tokenizer):
    while True:
        _entry = input_queue.get()
        if _entry is None:
            break
        output_queue.put(_get_seqs(_entry=_entry, _tokenizer=_tokenizer))


def _enqueue(iq: Queue, midi_dict_iter: Iterable, _num_proc: int):
    for midi_dict in midi_dict_iter:
        iq.put(midi_dict)
    for _ in range(_num_proc):
        iq.put(None)


def get_seqs(
    tokenizer: Tokenizer,
    midi_dict_iter: Iterable,
):
    iq = Queue()
    oq = Queue()

    num_proc = os.cpu_count()
    workers = [
        Process(
            target=functools.partial(_worker, _tokenizer=tokenizer),
            args=(iq, oq),
        )
        for _ in range(num_proc)
    ]
    for w in workers:
        w.start()

    # Can't pickle geneator object when start method is spawn
    if get_start_method() == "spawn":
        logging.info(
            "Converting generator to list due to multiprocessing start method"
        )
        midi_dict_iter = [_ for _ in midi_dict_iter]

    enqueue = Process(
        target=functools.partial(_enqueue, _num_proc=num_proc),
        args=(iq, midi_dict_iter),
    )
    enqueue.start()

    while True:
        if not oq.empty():
            result = oq.get()
            yield result
        else:
            if not any(proc.is_alive() for proc in workers):
                break


def reservoir(_iterable: Iterable, k: int):
    _reservoir = []
    for entry in _iterable:
        if entry is not None:
            _reservoir.append(entry)

        if len(_reservoir) >= k:
            random.shuffle(_reservoir)
            yield from _reservoir
            _reservoir = []

    if _reservoir != []:
        yield from _reservoir


class PretrainingDataset(TrainingDataset):
    """Torch dataset object yielding sequences formatted for pre-training"""

    def __init__(self, dir_path: str, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)

        self.dir_path = dir_path
        self.epoch_files = self._get_epoch_files()
        self.curr_epoch = 0
        self.init_epoch(0)

    def __len__(self):
        return len(self.index)

    @classmethod
    def get_config_from_path(cls, path: str):
        """Returns config dict from dataset directory.

        Note that this will return the config corresponding to epoch0.jsonl.
        """
        assert os.path.isdir(path), "directory not found"
        assert os.path.isfile(
            epoch0_path := os.path.join(path, "epoch0.jsonl")
        ), "epoch file not found"
        with open(epoch0_path) as f:
            return json.loads(f.readline())

    def init_epoch(self, idx: int | None = None):
        if idx is None:
            idx = self.curr_epoch + 1

        if idx >= self.num_epochs:
            _idx = idx % self.num_epochs
            self.logger.warning(
                f"epoch file doesn't exist for {idx}, resetting to epoch={_idx}"
            )
            idx = _idx

        self.close()
        self.file_buff = open(self.epoch_files[idx], mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.index = self._build_index()
        self.curr_epoch = idx
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            rank = 0

        self.logger.info(f"{rank}: Initiated epoch {idx} of PretrainingDataset")

    def _get_epoch_files(self):
        """Validates and returns a sorted list of epoch dataset files."""
        file_names = [
            file_name
            for file_name in os.listdir(self.dir_path)
            if os.path.isfile(os.path.join(self.dir_path, file_name))
        ]
        file_paths = [
            os.path.join(self.dir_path, file_name) for file_name in file_names
        ]

        present_epochs = []
        for file_name in file_names:
            if not re.match(r"^epoch\d+\.jsonl$", file_name):
                self.logger.warning(
                    f"Found file with unexpected name: {file_name}"
                )
            else:
                present_epochs.append(
                    int(re.match(r"^epoch(\d+)\.jsonl$", file_name).group(1))
                )
        self.num_epochs = len(present_epochs)
        assert self.num_epochs >= 1, f"no epoch files found in {self.dir_path}"
        assert set(present_epochs) == set(
            range(self.num_epochs)
        ), "epoch files missing"

        # Check files have valid configs
        for file_path in file_paths:
            self.close()
            self.file_buff = open(file_path, mode="r")
            self.file_mmap = mmap.mmap(
                self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
            )
            self.check_config()

        return [
            os.path.join(self.dir_path, f"epoch{idx}.jsonl")
            for idx in range(self.num_epochs)
        ]

    @classmethod
    def build(
        cls,
        tokenizer: Tokenizer,
        save_dir: str,
        max_seq_len: int,
        num_epochs: int,
        midi_dataset: MidiDataset = None,
        midi_dataset_path: str = None,
    ):
        """Builds and returns PretrainingDataset."""

        def _build_epoch(_save_path, _midi_dataset):
            with jsonlines.open(_save_path, mode="w") as writer:
                # Write tokenizer info into json on first line
                writer.write(
                    {
                        "tokenizer_config": tokenizer.config,
                        "tokenizer_name": tokenizer.name,
                        "max_seq_len": max_seq_len,
                    }
                )

                buffer = []
                _idx = 0
                for entry in reservoir(get_seqs(tokenizer, _midi_dataset), 10):
                    if entry is not None:
                        buffer += entry
                    while len(buffer) >= max_seq_len:
                        writer.write(buffer[:max_seq_len])
                        buffer = buffer[max_seq_len:]

                    _idx += 1
                    if _idx % 250 == 0:
                        logger.info(f"Finished processing {_idx}")

                buffer += [tokenizer.pad_tok] * (max_seq_len - len(buffer))
                writer.write(buffer[:max_seq_len])

        logger = setup_logger()
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        assert num_epochs > 0, "num_epochs must be greater than 0"
        if get_start_method() == "spawn":
            logger.warning(
                'The current multiprocessing start method is "spawn", this '
                "will slow down dataset building"
            )

        if os.path.isdir(save_dir) and os.listdir(save_dir):
            print(
                f"The directory at {save_dir} in non-empty, type [Y/y] to "
                "remove and continue:"
            )
            if input() not in {"Y", "y"}:
                print("Aborting")
                return
            else:
                shutil.rmtree(save_dir)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not midi_dataset and not midi_dataset_path:
            Exception("Must provide either midi_dataset or midi_dataset_path")
        if midi_dataset and midi_dataset_path:
            Exception("Can't provide both midi_dataset and midi_dataset_path")

        logger.info(
            f"Building PretrainingDataset with config: "
            f"max_seq_len={max_seq_len}, "
            f"tokenizer_name={tokenizer.name}"
        )
        for idx in range(num_epochs):
            logger.info(f"Building epoch {idx}/{num_epochs - 1}...")

            # Reload the dataset on each iter
            if midi_dataset_path:
                midi_dataset = MidiDataset.get_generator(midi_dataset_path)

            _build_epoch(
                _save_path=os.path.join(save_dir, f"epoch{idx}.jsonl"),
                _midi_dataset=midi_dataset,
            )

        logger.info(
            f"Finished building, saved PretrainingDataset to {save_dir}"
        )

        return cls(dir_path=save_dir, tokenizer=tokenizer)


class FinetuningDataset(TrainingDataset):
    """Torch dataset object yielding sequences formatted for fine-tuning."""

    def __init__(self, file_path: str, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)

        self.file_path = file_path
        self.file_buff = open(file_path, mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.check_config()
        self.index = self._build_index()

    def __len__(self):
        return len(self.index)

    @classmethod
    def get_config_from_path(cls, path: str):
        """Returns config dict from dataset file"""
        assert os.path.isfile(path), "dataset file not found"
        with open(path) as f:
            return json.loads(f.readline())

    # Do nothing in this case
    def init_epoch(self, idx: int | None = None):
        self.logger.info(f"Successful initiated epoch {idx}")

    @classmethod
    def build(
        cls,
        tokenizer: Tokenizer,
        save_path: str,
        max_seq_len: int,
        stride_len: int,
        midi_dataset: MidiDataset = None,
        midi_dataset_path: str = None,
    ):
        """Builds and returns FinetuningDataset."""

        # This function should be made more robust in the future
        def _truncate_and_stride(_tokenized_seq: list):
            prefix = []

            while _tokenized_seq:
                tok = _tokenized_seq[0]
                if tok != tokenizer.bos_tok and tok[0] == "prefix":
                    prefix.append(_tokenized_seq.pop(0))
                else:
                    break

            seq_len = len(_tokenized_seq)
            prefix_len = len(prefix)

            res = []
            idx = 0
            # No padding needed here
            while idx + max_seq_len - prefix_len < seq_len:
                res.append(
                    prefix
                    + _tokenized_seq[idx : idx + max_seq_len - prefix_len]
                )
                idx += stride_len

                # Checks that next start note will not be cutoff midway
                while idx < seq_len:
                    if _tokenized_seq[idx] in tokenizer.special_tokens:
                        break
                    elif _tokenized_seq[idx][0] in tokenizer.instruments_wd:
                        break
                    else:
                        idx += 1

            # Add the last sequence
            _seq = prefix + _tokenized_seq[idx : idx + max_seq_len - prefix_len]
            _seq += [tokenizer.pad_tok] * (max_seq_len - len(_seq))
            res.append(_seq)

            return res

        def _build(_midi_dataset):
            with jsonlines.open(save_path, mode="w") as writer:
                # Write tokenizer info into json on first line
                writer.write(
                    {
                        "tokenizer_config": tokenizer.config,
                        "tokenizer_name": tokenizer.name,
                        "max_seq_len": max_seq_len,
                        "stride_len": max_seq_len,
                    }
                )
                logger.info(
                    f"Building FinetuningDataset with config: "
                    f"tokenizer_name={tokenizer.name}, "
                    f"max_seq_len={max_seq_len}, "
                    f"stride_len={stride_len}"
                )

                _idx = 0
                for entry in get_seqs(tokenizer, _midi_dataset):
                    if entry:
                        for _entry in _truncate_and_stride(entry):
                            writer.write(_entry)

                    _idx += 1
                    if _idx % 250 == 0:
                        logger.info(f"Finished processing {_idx}")

        logger = setup_logger()
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        if get_start_method() == "spawn":
            logger.warning(
                'The current multiprocessing start method is "spawn", this '
                "will slow down dataset building"
            )

        if not midi_dataset and not midi_dataset_path:
            Exception("Must provide either midi_dataset or midi_dataset_path")
        if midi_dataset and midi_dataset_path:
            Exception("Can't provide both midi_dataset and midi_dataset_path")

        if os.path.isfile(save_path):
            print(
                f"There is a file existing at {save_path}, type [Y/y] to "
                "continue:"
            )
            if input() not in {"Y", "y"}:
                print("Aborting")
                return
            else:
                os.remove(save_path)

        if midi_dataset_path:
            midi_dataset = MidiDataset.get_generator(midi_dataset_path)

        _build(_midi_dataset=midi_dataset)

        logger.info(f"Finished building, saved Finetuning to {save_path}")

        return cls(file_path=save_path, tokenizer=tokenizer)
