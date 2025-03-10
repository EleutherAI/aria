import json
import random
import argparse
from collections import Counter
from pathlib import Path

from aria.datasets import build_mididict_dataset
from aria.embeddings.evaluate import CATEGORY_TAGS

random.seed(42)

MIDI_DATASET_TRAIN_SIZE = 10000
MIDI_DATASET_TEST_SIZE = 1000


def get_midi_paths(
    dataset_dir: str,
    metadata_path: str,
    metadata_category: str,
):
    metadata_tags = list(CATEGORY_TAGS[metadata_category].keys())
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)
        metadata_dict = {k: v["metadata"] for k, v in metadata_dict.items()}

    midi_paths = list(Path(dataset_dir).rglob("*.mid"))
    buckets = {tag: [] for tag in metadata_tags}

    for midi_file in midi_paths:
        # Extract metadata key from file name (e.g., "000001_0" -> "1")
        key = str(int(midi_file.stem.split("_")[0]))
        metadata = metadata_dict.get(key)
        if not metadata:
            continue
        tag = metadata.get(metadata_category)
        if tag in metadata_tags:
            buckets[tag].append(midi_file)

    # Calculate the desired count per tag for both splits
    num_tags = len(metadata_tags)
    desired_train_per_tag = MIDI_DATASET_TRAIN_SIZE // num_tags
    desired_test_per_tag = MIDI_DATASET_TEST_SIZE // num_tags

    train_paths = []
    test_paths = []
    for tag, files in buckets.items():
        random.shuffle(files)
        total_files = len(files)
        total_desired = desired_train_per_tag + desired_test_per_tag

        if total_files >= total_desired:
            # Enough files: use fixed numbers.
            train_paths.extend(files[:desired_train_per_tag])
            test_paths.extend(
                files[
                    desired_train_per_tag : desired_train_per_tag
                    + desired_test_per_tag
                ]
            )
        else:
            # Not enough files: split based on the desired ratio.
            train_ratio = desired_train_per_tag / total_desired
            train_count = round(total_files * train_ratio)
            test_count = total_files - train_count  # all remaining go to test
            train_paths.extend(files[:train_count])
            test_paths.extend(files[train_count : train_count + test_count])

    def _extract_tag(midi_file):
        key = str(int(midi_file.stem.split("_")[0]))
        return metadata_dict.get(key, {}).get(metadata_category, "unknown")

    train_distribution = Counter(_extract_tag(mp) for mp in train_paths)
    test_distribution = Counter(_extract_tag(mp) for mp in test_paths)

    print(
        f"Finished with splits: train={len(train_paths)}, test={len(test_paths)}"
    )
    print("Train distribution:", dict(train_distribution))
    print("Test distribution:", dict(test_distribution))

    return train_paths, test_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate embeddings with linear prob"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_category",
        type=str,
        choices=["genre", "music_period", "composer", "form"],
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
    args = parser.parse_args()

    train_paths, test_paths = get_midi_paths(
        dataset_dir=args.dataset_dir,
        metadata_path=args.metadata_path,
        metadata_category=args.metadata_category,
    )

    build_mididict_dataset(
        mid_paths=train_paths,
        stream_save_path=args.train_save_path,
    )
    build_mididict_dataset(
        mid_paths=test_paths,
        stream_save_path=args.test_save_path,
    )


if __name__ == "__main__":
    main()
