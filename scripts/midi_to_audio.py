import glob

from aria.utils import midi_to_audio


def main():
    paths = glob.glob("samples/*.mid")
    for path in paths:
        midi_to_audio(path)


if __name__ == "__main__":
    main()
