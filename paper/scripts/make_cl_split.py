import json
import random
from collections import Counter
from pathlib import Path

from aria.datasets import build_mididict_dataset

random.seed(42)


# TODO: Make this into argparse script

DATASET_DIR = "/mnt/ssd1/aria-midi/final/v1/aria-midi-v1-emb-int"
METADATA_PATH = f"{DATASET_DIR}/metadata.json"
METADATA_CATEGORY = "form"
METADATA_TAGS = [
    "fugue",
    "sonata",
    "etude",
    "nocturne",
    "waltz",
    "improvisation",
]

MIDI_DATASET_TRAIN_SIZE = 10000
MIDI_DATASET_TEST_SIZE = 1000

TRAIN_SAVE_PATH = (
    "/mnt/ssd1/aria/data/paper/classification/form-aria/train-mididict.jsonl"
)
TEST_SAVE_PATH = (
    "/mnt/ssd1/aria/data/paper/classification/form-aria/test-mididict.jsonl"
)


def get_midi_paths():
    with open(METADATA_PATH, "r") as f:
        metadata_dict = json.load(f)
        metadata_dict = {k: v["metadata"] for k, v in metadata_dict.items()}

    midi_paths = list(Path(DATASET_DIR).rglob("*.mid"))
    buckets = {tag: [] for tag in METADATA_TAGS}

    for midi_file in midi_paths:
        # Extract metadata key from file name (e.g., "000001_0" -> "1")
        key = str(int(midi_file.stem.split("_")[0]))
        metadata = metadata_dict.get(key)
        if not metadata:
            continue
        tag = metadata.get(METADATA_CATEGORY)
        if tag in METADATA_TAGS:
            buckets[tag].append(midi_file)

    # Calculate the desired count per tag for both splits
    num_tags = len(METADATA_TAGS)
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
        return metadata_dict.get(key, {}).get(METADATA_CATEGORY, "unknown")

    train_distribution = Counter(_extract_tag(mp) for mp in train_paths)
    test_distribution = Counter(_extract_tag(mp) for mp in test_paths)

    print(
        f"Finished with splits: train={len(train_paths)}, test={len(test_paths)}"
    )
    print("Train distribution:", dict(train_distribution))
    print("Test distribution:", dict(test_distribution))

    return train_paths, test_paths


def main():
    train_paths, test_paths = get_midi_paths()

    build_mididict_dataset(
        mid_paths=train_paths,
        stream_save_path=TRAIN_SAVE_PATH,
    )
    build_mididict_dataset(
        mid_paths=test_paths,
        stream_save_path=TEST_SAVE_PATH,
    )


if __name__ == "__main__":
    main()
