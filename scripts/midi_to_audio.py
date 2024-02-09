import os

from aria.utils import midi_to_audio


def main():
    root_dir = "/Users/louis/work/data/mid/prompts/survey"
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".mid"):
                midi_path = os.path.join(dirpath, filename)
                midi_to_audio(midi_path)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
