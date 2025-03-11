import os
import re
import json
import random
import argparse
from pathlib import Path

from ariautils.midi import MidiDict
from aria.datasets import MidiDataset

random.seed(43)

SPLIT_RATIO = 0.9


def get_midi_paths(dataset_dir: str, test_split_file: str = None):
    train_paths = []
    test_paths = []

    test_pairs = set()
    if test_split_file:
        with open(test_split_file, "r") as f:
            test_files = json.load(f)
        for entry in test_files:
            parts = re.split(r"[\\/]", entry)
            assert len(parts) == 3

            pianist = parts[1].lower()
            file_name = parts[2].replace(".npy", ".mid")
            test_pairs.add((pianist, file_name))

    pianist_categories = os.listdir(dataset_dir)
    for pianist in pianist_categories:
        pianist_dir = os.path.join(dataset_dir, pianist)
        mid_paths = list(Path(pianist_dir).glob("*.mid"))
        random.shuffle(mid_paths)
        print(f"Found {len(mid_paths)} for {pianist}")

        if test_pairs:
            for path in mid_paths:
                if (pianist.lower(), path.name) in test_pairs:
                    test_paths.append(
                        {"path": path, "pianist": pianist.lower()}
                    )
                else:
                    train_paths.append(
                        {"path": path, "pianist": pianist.lower()}
                    )
        else:
            split_idx = int(len(mid_paths) * SPLIT_RATIO)
            train_paths += [
                {"path": path, "pianist": pianist.lower()}
                for path in mid_paths[:split_idx]
            ]
            test_paths += [
                {"path": path, "pianist": pianist.lower()}
                for path in mid_paths[split_idx:]
            ]

    train_mididicts = []
    for path_entry in train_paths:
        _mid_dict = MidiDict.from_midi(mid_path=path_entry["path"])
        _mid_dict.metadata["pianist"] = path_entry["pianist"]
        train_mididicts.append(_mid_dict)

    test_mididicts = []
    for path_entry in test_paths:
        _mid_dict = MidiDict.from_midi(mid_path=path_entry["path"])
        _mid_dict.metadata["pianist"] = path_entry["pianist"]
        test_mididicts.append(_mid_dict)

    return train_mididicts, test_mididicts


def main():
    parser = argparse.ArgumentParser(
        description="Create pianist8 dataset train-test split"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=None,
        help="Path to JSON file listing test split files (paths like 'pianist8/<pianist>/<file>.npy')",
    )
    args = parser.parse_args()

    assert os.path.isdir(args.dataset_dir)
    assert not os.path.isfile(args.train_save_path)
    assert not os.path.isfile(args.test_save_path)

    train_mididicts, test_mididicts = get_midi_paths(
        dataset_dir=args.dataset_dir,
        test_split_file=args.test_split,
    )

    TrainDataset = MidiDataset(entries=train_mididicts).save(
        args.train_save_path
    )
    TestDataset = MidiDataset(entries=test_mididicts).save(args.test_save_path)


if __name__ == "__main__":
    main()
