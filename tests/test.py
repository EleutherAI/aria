"""Basics test scripts"""

import mido

from aria.tokenizer import TokenizerLazy
from aria.data.midi import MidiDict


# TODO:
# - Introduce a testing framework
# - Write tests for pre-training
# - Write tests for dataset building
# - Write tests for tokenized dataset building


def main():
    mid = mido.MidiFile("arabesque.mid")
    midi_dict = MidiDict.from_midi(mid)

    tokenizer = TokenizerLazy(False, "none", 1000)
    seq = tokenizer.tokenize_midi_dict(midi_dict)[0]
    print(seq)
    print(f"length: {len(seq)}")

    midi_dict_res = tokenizer.detokenize_midi_dict(seq)
    mid_res = midi_dict_res.to_midi()
    mid_res.save("res.mid")


if __name__ == "__main__":
    main()
