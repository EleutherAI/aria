"""Contains classes and utilities for building and processing datasets."""

import json
import logging
import copy
import torch
import mido

from pathlib import Path
from typing import Callable

from aria.config import load_config
from aria.tokenizer import Tokenizer
from aria.data import tests
from aria.data.midi import MidiDict


class MidiDataset:
    """Container for datasets of MidiDict objects.

    Can be used to save, load, and build, datasets of MidiDict objects.

    Args:
        entries (list[MidiDict]): MidiDict objects to be stored.
    """

    def __init__(self, entries: list[MidiDict] = []):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, ind: int):
        return self.entries[ind]

    def __iter__(self):
        yield from self.entries

    def save(self, save_path: str):
        """Saves dataset to JSON file."""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump([entry._get_msg_dict() for entry in self.entries], f)

    @classmethod
    def load(cls, load_path: str):
        """Loads dataset from JSON file."""
        with open(load_path) as f:
            entries = json.load(f)

        return cls([MidiDict(**entry) for entry in entries])

    @classmethod
    def build(
        cls,
        dir: str,
        recur: bool = False,
    ):
        """Inplace version of build_dataset."""
        return build_mididict_dataset(
            dir=dir,
            recur=recur,
        )


def build_mididict_dataset(
    dir: str,
    recur: bool = False,
):
    """Builds dataset of MidiDicts.

    During the build process, successfully parsed MidiDicts can be filtered and
    preprocessed. This can be customised by modifying the config.json file.

    Args:
        dir (str): Directory to index from.
        recur (bool): If True, recursively search directories for MIDI files.
            Defaults to False.

    Returns:
        Dataset: Dataset of parsed, filtered, and preprocessed MidiDicts.
    """

    def _run_tests(_mid_dict: MidiDict):
        failed_tests = []
        for test_name, test_config in config["tests"].items():
            if test_config["run"] is True:
                # If test failed append to failed_tests
                if (
                    getattr(tests, test_name)(
                        _mid_dict, **test_config["config"]
                    )
                    is False
                ):
                    failed_tests.append(test_name)

        return failed_tests

    def _process_midi(_mid_dict: MidiDict):
        for fn_name, fn_config in config["pre_processing"].items():
            if fn_config["run"] is True:
                getattr(_mid_dict, fn_name)(**fn_config["config"])

        return _mid_dict

    config = load_config()["data"]

    paths = []
    if recur is True:
        paths += Path(dir).rglob(f"*.mid")
        paths += Path(dir).rglob(f"*.midi")
    else:
        paths += Path(dir).glob(f"*.mid")
        paths += Path(dir).glob(f"*.midi")

    # Run tests and process located MIDIs
    entries = []
    for path in paths:
        try:
            mid_dict = MidiDict.from_midi(mido.MidiFile(path))
        except Exception:
            print(f"Failed to load file at {path}.")

        failed_tests = _run_tests(mid_dict)
        if failed_tests:
            print(
                f"{path} not added. Failed tests:",
                ", ".join(failed_tests) + ".",
            )
        else:
            entries.append(_process_midi(mid_dict))

    return MidiDataset(entries)


class TokenizedDataset(torch.utils.data.Dataset):
    """Container for datasets of pre-processed (tokenized) MidiDict objects.

    Args:
        entries (list): MidiDict objects to be stored.
    """

    def __init__(self, entries: list = []):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        # We use create a copy so that self._transform does not permanently
        # mutate the entries.
        return self._transform(copy.deepcopy(self.entries[idx]))

    def _transform(self, entry: list):
        # Default behaviour is to act as the identity function
        return entry

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
                    res = fn(x)
                return res

            self._transform = _new_transform
        else:
            raise ValueError("Must provide function or list of functions.")

    def save(self, save_path: str):
        """Saves dataset to JSON file."""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.entries)

    @classmethod
    def load(cls, load_path: str):
        """Loads dataset from JSON file."""
        with open(load_path) as f:
            entries = json.load(f)

        return cls(entries)

    @classmethod
    def build(
        cls,
        midi_dataset: MidiDataset,
        tokenizer: Tokenizer,
    ):
        if tokenizer.truncate_type != "strided":
            logging.warn(
                "Tokenizer striding not being used when building dataset."
            )

        return build_tokenized_dataset(midi_dataset, tokenizer)


# TODO: Implement
def build_tokenized_dataset(
    midi_dataset: MidiDataset,
    tokenizer: Tokenizer,
):
    entries = []
    for midi_dict in midi_dataset:
        entries += tokenizer.tokenize(midi_dict)["tokens"]

    return TokenizedDataset(entries)
