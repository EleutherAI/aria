"""Contains classes and utilities for building and processing datasets."""

import json
import os
import mmap
import atexit
import jsonlines
import logging
import copy
import torch
import mido
import aria.data.midi

from pathlib import Path
from typing import Callable

from aria.config import load_config
from aria.tokenizer import Tokenizer
from aria.data.midi import MidiDict


# TODO: Add proper docstring
# - Add functionality for splitting the dataset into train-val split
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


# TODO: Add functionality for hashing checksum the MIDIs -  removing dupes
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
                # All midi_dict tests must follow this naming convention
                test_fn_name = "_test" + "_" + test_name
                test_args = test_config["args"]

                try:
                    test_fn = getattr(aria.data.midi, test_fn_name)
                except:
                    logging.error(
                        f"Error finding test function for {test_name}"
                    )
                else:
                    if test_fn(_mid_dict, **test_args) is False:
                        failed_tests.append(test_name)

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
                    logging.error(
                        f"Error finding preprocessing function for {fn_name}"
                    )

        return _mid_dict

    if stream_save_path is None:
        streaming = False
    else:
        streaming = True

        if overwrite is False and os.path.isfile(stream_save_path) is True:
            raise FileExistsError(f"File at {stream_save_path} already exists.")
        elif overwrite is True and os.path.isfile(stream_save_path) is True:
            os.remove(stream_save_path)

    config = load_config()["data"]

    paths = []
    if recur is True:
        paths += Path(dir).rglob(f"*.mid")
        paths += Path(dir).rglob(f"*.midi")
    else:
        paths += Path(dir).glob(f"*.mid")
        paths += Path(dir).glob(f"*.midi")

    if streaming is True:
        with jsonlines.open(stream_save_path, mode="w") as writer:
            for path in paths:
                try:
                    mid_dict = MidiDict.from_midi(mido.MidiFile(path))
                except Exception:
                    logging.error(f"Failed to load file at {path}.")

                failed_tests = _run_tests(mid_dict)
                if failed_tests:
                    logging.info(f"File at {path} failed preprocessing tests:")
                    for test_name in failed_tests:
                        logging.info(test_name)
                else:
                    writer.write(_preprocess_mididict(mid_dict).get_msg_dict())

    else:  # streaming is False
        entries = []
        for path in paths:
            try:
                mid_dict = MidiDict.from_midi(mido.MidiFile(path))
            except Exception:
                logging.error(f"Failed to load file at {path}.")

            failed_tests = _run_tests(mid_dict)
            if failed_tests:
                logging.info(f"File at {path} failed preprocessing tests:")
                for test_name in failed_tests:
                    logging.info(test_name)
            else:
                entries.append(_preprocess_mididict(mid_dict))

        return entries


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._transform = None

        self.file_buff = open(file_path, mode="r")
        self.file_mmap = mmap.mmap(
            self.file_buff.fileno(), 0, access=mmap.ACCESS_READ
        )

        # Check self.tokenizers is the same as the one used to generate file
        _debug = self.file_mmap.readline()
        file_config = json.loads(_debug)
        for k, v in file_config.items():
            assert getattr(self.tokenizer, k) == v

        self.index = self._build_index()

    def close(self):
        # This is unnecessary because mmap is closed automatically when gc,
        # however using this stops ResourceWarnings.
        self.file_buff.close()
        self.file_mmap.close()

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

    @classmethod
    def build(
        cls,
        tokenizer: Tokenizer,
        save_path: str,
        midi_dataset: MidiDataset = None,
        midi_dataset_path: str = None,
        overwrite: bool = False,
    ):
        if overwrite is False and os.path.isfile(save_path) is True:
            raise FileExistsError(f"File at {save_path} already exists.")
        elif overwrite is True and os.path.isfile(save_path) is True:
            os.remove(save_path)

        if midi_dataset is None and midi_dataset_path is None:
            raise ValueError(
                "Must provide either midi_dataset or midi_dataset_path."
            )

        with jsonlines.open(save_path, mode="w") as writer:
            # Write tokenizer info into json on first line
            writer.write(
                {
                    "name": tokenizer.name,
                    "padding": tokenizer.padding,
                    "truncate_type": tokenizer.truncate_type,
                    "max_seq_len": tokenizer.max_seq_len,
                    "stride_len": tokenizer.stride_len,
                }
            )

            if midi_dataset:
                for midi_dict in midi_dataset:
                    for entry in tokenizer.tokenize_midi_dict(midi_dict):
                        writer.write(entry)

            elif midi_dataset_path:
                with jsonlines.open(midi_dataset_path) as reader:
                    for msg_dict in reader:
                        midi_dict = MidiDict.from_msg_dict(msg_dict)
                        for entry in tokenizer.tokenize_midi_dict(midi_dict):
                            writer.write(entry)

        return cls(file_path=save_path, tokenizer=tokenizer)
