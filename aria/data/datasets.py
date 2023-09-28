"""Contains classes and utilities for building and processing datasets."""

import json
import os
import mmap
import jsonlines
import logging
import random
import torch

from pathlib import Path
from typing import Callable
from copy import deepcopy

from aria.config import load_config
from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict, get_test_fn


# TODO: Investigate why loads of drums tracks are appearing when drum is present
class MidiDataset:
    """Container for datasets of MidiDict objects.

    Can be used to save, load, and build, datasets of MidiDict objects.

    Args:
        entries (list[MidiDict]): MidiDict objects to be stored.
    """

    def __init__(self, entries: list[MidiDict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, ind: int):
        return self.entries[ind]

    def __iter__(self):
        yield from self.entries

    def save(self, save_path: str):
        """Saves dataset to JSON file."""

        with jsonlines.open(save_path, mode="w") as writer:
            for midi_dict in self.entries:
                writer.write(midi_dict.get_msg_dict())

    @classmethod
    def load(cls, load_path: str):
        """Loads dataset from JSON file."""
        midi_dicts = []
        with jsonlines.open(load_path) as reader:
            for entry in reader:
                midi_dicts.append(MidiDict.from_msg_dict(entry))

        return cls(midi_dicts)

    @classmethod
    def split_from_file(
        cls,
        load_path: str,
        train_val_ratio: float = 0.95,
        repeatable: bool = False,
        overwrite: bool = False,
    ):
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

        logging.info(
            f"Succesfully split into train ({idx_train}) and validation ({idx_val}) sets"
        )

    @classmethod
    def build(
        cls,
        dir: str,
        recur: bool = False,
    ):
        """Builds are returns a MidiDataset, see build_mididict_dataset."""
        return cls(
            build_mididict_dataset(
                dir=dir,
                recur=recur,
            )
        )

    @classmethod
    def build_to_file(
        cls,
        dir: str,
        save_path: str,
        recur: bool = False,
        overwrite: bool = False,
    ):
        """Builds MidiDataset, saving the results directly to a file.

        This function will not return a MidiDataset object. It is well suited
        for situations where the resulting MidiDataset will not fit in the
        system's memory.
        """
        build_mididict_dataset(
            dir=dir,
            recur=recur,
            stream_save_path=save_path,
            overwrite=overwrite,
        )


def build_mididict_dataset(
    dir: str,
    recur: bool = False,
    stream_save_path: str = None,
    overwrite: bool = False,
):
    """Builds dataset of MidiDicts.

    During the build process, successfully parsed MidiDicts can be filtered and
    preprocessed. This can be customised by modifying the config.json file.

    Args:
        dir (str): Directory to index from.
        recur (bool): If True, recursively search directories for MIDI files.
            Defaults to False.
        stream_save_path: If True, stream the dictionaries directly to a jsonl
            file instead of returning them as a list. This option is
            appropriate when processing very large numbers of MIDI files.
        overwrite: If True, overwrite file at stream_save_path when streaming.

    Returns:
        list[MidiDict]: List of parsed, filtered, and preprocessed MidiDicts.
            This is only returned if stream_save_path is not provided.
    """

    def _run_tests(_mid_dict: MidiDict):
        failed_tests = []
        for test_name, test_config in config["tests"].items():
            if test_config["run"] is True:
                test_fn = get_test_fn(test_name)
                test_args = test_config["args"]

                if test_fn(_mid_dict, **test_args) is False:
                    failed_tests.append(test_name)

        return failed_tests

    # Maybe refactor this in the same way as _run_tests
    def _preprocess_mididict(_mid_dict: MidiDict):
        for fn_name, fn_config in config["pre_processing"].items():
            if fn_config["run"] is True:
                fn_args = fn_config["args"]
                getattr(_mid_dict, fn_name)(fn_args)

                try:
                    # Note fn_args is passed as a dict, not unpacked as kwargs
                    getattr(_mid_dict, fn_name)(fn_args)
                except:
                    logging.error(
                        f"Error finding preprocessing function for {fn_name}"
                    )

        return _mid_dict

    def get_mididicts(_paths: list):
        num_paths = len(_paths)
        if num_paths == 0:
            logging.warning(
                "Directory contains no files matching *.mid or *.midi"
            )

        seen_hashes = {}
        for idx, path in enumerate(_paths):
            if idx % 50 == 0 and idx != 0:
                logging.info(f"processed midi files: {idx}/{num_paths}")

            try:
                mid_dict = MidiDict.from_midi(mid_path=path)
            except Exception as e:
                logging.error(f"Failed to load file at {path}:")
                logging.error(e)
                continue

            mid_hash = mid_dict.calculate_hash()
            if seen_hashes.get(mid_hash, False) is True:
                logging.info(f"File at {path} is a duplicate")
                continue
            else:
                seen_hashes[mid_hash] = True

            failed_tests = _run_tests(mid_dict)
            if failed_tests:
                logging.info(f"File at {path} failed preprocessing tests:")
                for test_name in failed_tests:
                    logging.info(test_name)
            else:
                yield _preprocess_mididict(mid_dict)

        logging.info(f"finished building mididict dataset")

    config = load_config()["data"]

    paths = []
    if recur is True:
        paths += Path(dir).rglob(f"*.mid")
        paths += Path(dir).rglob(f"*.midi")
    else:
        paths += Path(dir).glob(f"*.mid")
        paths += Path(dir).glob(f"*.midi")

    num_paths = len(paths)
    if num_paths == 0:
        logging.warning("Directory contains no files matching *.mid or *.midi")

    if stream_save_path is None:
        # Not streaming -> return entries directly
        entries = []
        for entry in get_mididicts(paths):
            entries.append(entry)

        return entries
    else:
        # Streaming -> write to file instead of returning anything
        if overwrite is False and os.path.isfile(stream_save_path) is True:
            raise FileExistsError(f"File at {stream_save_path} already exists.")
        elif overwrite is True and os.path.isfile(stream_save_path) is True:
            os.remove(stream_save_path)

        with jsonlines.open(stream_save_path, mode="w") as writer:
            for entry in get_mididicts(paths):
                writer.write(entry.get_msg_dict())


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._transform = None

        self.file_path = file_path
        self.file_buff = open(file_path, mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )

        # Check self.tokenizers is the same as the one used to generate file
        # This logic could use a refactor (maybe use deepdict?)
        buffer = self.file_mmap.readline()
        try:
            self.config = json.loads(buffer)
            self._check_config()
        except AssertionError as e:
            logging.error(
                f"Tokenizer config setting don't match those used to build {file_path}"
            )
            raise e
        except Exception as e:
            logging.error("Processing tokenizer config resulted in an error")
            raise e

        self.tokenizer_name = tokenizer.name
        self.max_seq_len = self.config["max_seq_len"]

        self.index = self._build_index()

    def _check_config(self):
        config = self.config
        tokenizer = self.tokenizer

        assert config["tokenizer_name"] == tokenizer.name
        for k, v in config["tokenizer_config"].items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    assert _v == tokenizer.config[k][_k]
            elif isinstance(v, str) or isinstance(v, int):
                assert v == tokenizer.config[k]

    def close(self):
        # This is unnecessary because mmap is closed automatically when gc,
        # however using this stops ResourceWarnings.
        self.file_buff.close()
        self.file_mmap.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.index)

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

        return self.tokenizer.encode(src), self.tokenizer.encode(tgt)

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

        logging.debug(f"Finished indexing {len(index)} sequences")

        return index

    # This is a bit gross - maybe refactor this
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

    def get_shuffled_dataset(
        self,
        save_path: str | None = None,
        repeatable: bool = False,
        overwrite: bool = False,
    ):
        """Writes and returns a shuffled version of the dataset (self).

        Note that this function will NOT change the current dataset in anyway.
        It creates a new dataset file at save_path and returns this as a new
        object.

        Args:
            save_path (str): Path to save the new dataset. If this is not
                provided then the save_path will have '_shuffled' appended.
            repeatible (bool): If True, makes the shuffling process
                deterministic by setting a random seed.
            overwrite (bool): If True, will overwrite the file at save_path.
        """
        if save_path:
            assert (
                self.file_path != save_path
            ), "save_path must not overwrite the current file"

        file_path = Path(self.file_path)
        if not save_path:
            save_path = file_path.with_name(
                f"{file_path.stem}_shuffled{file_path.suffix}"
            )
        if os.path.isfile(save_path) and overwrite is False:
            logging.warning(f"File already exists at {save_path} - aborting")
            return self

        if repeatable:
            random.seed(42)
        shuffled_index = deepcopy(self.index)
        random.shuffle(shuffled_index)

        with jsonlines.open(save_path, mode="w") as writer:
            writer.write(self.config)
            for byte_idx in shuffled_index:
                self.file_mmap.seek(byte_idx)
                writer.write(json.loads(self.file_mmap.readline()))

        logging.info(
            f"Shuffled copy of {self.file_path} successfully written "
            f"to {save_path} "
        )

        return TokenizedDataset(save_path, self.tokenizer)

    @classmethod
    def build(
        cls,
        tokenizer: Tokenizer,
        save_path: str,
        max_seq_len: int,
        midi_dataset: MidiDataset = None,
        midi_dataset_path: str = None,
        padding: bool = True,
        stride_len: int = None,
        overwrite: bool = False,
    ):
        """Builds and returns a TokenizedDataset.

        Args:
            tokenizer (Tokenizer): Tokenizer to use to tokenize the MidiDicts.
            save_path (str): Save path for datasets file.
            max_seq_len (int): Maximum sequence length used to split the
                tokenized sequences.
            midi_dataset (MidiDataset, optional): If provided, build dataset
                directly from a MidiDataset object.
            midi_dataset_path (str, optional): If provided, build dataset by
                dynamically loading MidiDict objects from a MidiDataset save
                file. If both midi_dataset and midi_dataset_path are provided,
                midi_dataset will be preffered.
            padding (bool, optional): If True, pad the sequences so that they
                have length=max_seq_len. Defaults to True.
            stride_len (int, optional): If provided, will stride the sequences
                according to the provided length. Defaults to None.
            overwrite (bool, optional): If True, will overwrite a previous file
                located at save_path. Defaults to False.

        Returns:
            TokenizedDataset: Dataset saved midi_dataset and saved at save_path.
        """

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
                    # Break loop when a non 'wait' or 'dur' is seen
                    if _tokenized_seq[idx] in tokenizer.special_tokens:
                        break
                    elif _tokenized_seq[idx][0] in {"wait", "dur"}:
                        idx += 1
                    else:
                        break

            # Add the last sequence
            _seq = prefix + _tokenized_seq[idx : idx + max_seq_len - prefix_len]
            if padding is True:
                _seq += [tokenizer.pad_tok] * (max_seq_len - len(_seq))
            res.append(_seq)

            return res

        if overwrite is False and os.path.isfile(save_path) is True:
            raise FileExistsError(f"File at {save_path} already exists.")
        elif overwrite is True and os.path.isfile(save_path) is True:
            os.remove(save_path)

        if midi_dataset is None and midi_dataset_path is None:
            raise ValueError(
                "Must provide either midi_dataset or midi_dataset_path."
            )

        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        if stride_len:
            assert 0 < stride_len <= max_seq_len, "Invalid stride_len"
        else:
            stride_len = max_seq_len

        with jsonlines.open(save_path, mode="w") as writer:
            # Write tokenizer info into json on first line
            writer.write(
                {
                    "tokenizer_config": tokenizer.config,
                    "tokenizer_name": tokenizer.name,
                    "max_seq_len": max_seq_len,
                    "padding": padding,
                    "stride_len": stride_len,
                }
            )

            if midi_dataset:
                if len(midi_dataset) == 0:
                    logging.warning("midi_dataset is empty")
                for idx, midi_dict in enumerate(midi_dataset):
                    if idx % 50 == 0 and idx != 0:
                        logging.info(f"processed midi_dicts: {idx}")

                    try:
                        tokenized_seq = tokenizer.tokenize_midi_dict(midi_dict)
                    except Exception as e:
                        logging.warning(
                            f"failed to tokenize midi_dict with index {idx}: {e}"
                        )
                    else:
                        for entry in _truncate_and_stride(tokenized_seq):
                            writer.write(entry)

            elif midi_dataset_path:
                with jsonlines.open(midi_dataset_path) as reader:
                    for idx, msg_dict in enumerate(reader):
                        if idx % 50 == 0 and idx != 0:
                            logging.info(f"processed midi_dicts: {idx}")

                        midi_dict = MidiDict.from_msg_dict(msg_dict)
                        try:
                            tokenized_seq = tokenizer.tokenize_midi_dict(
                                midi_dict
                            )
                        except Exception as e:
                            logging.warning(
                                f"failed to tokenize midi_dict with index {idx}: {e}"
                            )
                        else:
                            for entry in _truncate_and_stride(tokenized_seq):
                                writer.write(entry)

        logging.info(f"Finished building tokenized dataset")

        return cls(file_path=save_path, tokenizer=tokenizer)
