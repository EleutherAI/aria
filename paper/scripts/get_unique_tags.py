import json


# TODO: Make this into argeparse script


JSONL_FILE_PATH = (
    "/mnt/ssd1/aria/data/paper/classification/form-aria/train-mididict.jsonl"
)
MUSIC_PERIOD_KEY = "form"


def main():
    unique_periods = set()

    with open(JSONL_FILE_PATH, "r") as f:
        for line in f:
            # Parse each line as a JSON object.
            record = json.loads(line)
            # Get the 'metadata' dictionary, defaulting to an empty dict if missing.
            metadata = record.get("metadata", {})
            # Check if the metadata contains the music period key.
            if MUSIC_PERIOD_KEY in metadata:
                unique_periods.add(metadata[MUSIC_PERIOD_KEY])

    # Print the list of unique music periods.
    print(list(unique_periods))


if __name__ == "__main__":
    main()
