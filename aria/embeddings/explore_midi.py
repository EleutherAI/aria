import copy
import torch

from aria.config import load_model_config
from aria.utils import _load_weight
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from aria.model import TransformerCL, ModelConfig

TAG_IDS = {
    "chopin": 0,
    "bach": 1,
    "beethoven": 2,
    "liszt": 3,
    "mozart": 4,
    "debussy": 5,
    "schumann": 6,
    "schubert": 7,
    "rachmaninoff": 8,
    "brahms": 9,
    "tchaikovsky": 10,
    "haydn": 11,
    "scriabin": 12,
    "mendelssohn": 13,
    "czerny": 14,
    "ravel": 15,
    "scarlatti": 16,
    "other": 17,
}
ID_TO_TAG = {v: k for k, v in TAG_IDS.items()}


def explore_midi(
    midi_path: str,
    checkpoint_path: str,
    metadata_category: str,
    slice_len_notes: int = 500,
    max_seq_len: int = 2048,
):
    midi_dict = MidiDict.from_midi(midi_path)
    print(midi_dict.instrument_msgs)

    tag = midi_dict.metadata.get(metadata_category, None)
    if tag is not None and tag not in TAG_IDS:
        tag = "other"

    note_msgs = midi_dict.note_msgs
    slices = [
        note_msgs[i : i + slice_len_notes]
        for i in range(0, len(note_msgs), slice_len_notes)
    ]
    slices = [s for s in slices if len(s) >= 20]

    print(f"Found {len(slices)} slices in the MIDI file.")

    tokenizer = AbsTokenizer()
    model_config = ModelConfig(**load_model_config("medium-composer"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model_state = _load_weight(checkpoint_path, device="cuda")
    model = TransformerCL(model_config)
    model.load_state_dict(model_state)
    model.eval()
    model.cuda()

    for idx, note_slice in enumerate(slices):
        slice_midi = copy.deepcopy(midi_dict)
        slice_midi.note_msgs = note_slice
        slice_midi.metadata = {}

        tokenized_seq = tokenizer.tokenize(slice_midi)
        tokenizer.detokenize(tokenized_seq).to_midi().save(
            "/home/loubb/Dropbox/shared/test.mid"
        )
        if tokenizer.dim_tok in tokenized_seq:
            tokenized_seq.remove(tokenizer.dim_tok)
        tokenized_seq = tokenized_seq[:max_seq_len]
        if tokenizer.eos_tok not in tokenized_seq:
            tokenized_seq[-1] = tokenizer.eos_tok

        tokenizer
        encoded_seq = tokenizer.encode(tokenized_seq)
        input_tensor = torch.tensor([encoded_seq]).cuda()

        # Forward pass
        with torch.inference_mode():
            logits = model(input_tensor)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            # Get the top 5 probabilities and their corresponding indices
            top_probs, top_indices = torch.topk(probs, k=5)
            formatted_top_probs = [
                float(f"{p:.4f}") for p in top_probs.tolist()
            ]
            top_tags = [
                ID_TO_TAG.get(idx.item(), "unknown") for idx in top_indices
            ]

        print("Top 5 Predictions:")
        for tag, prob in zip(top_tags, formatted_top_probs):
            print(f"{tag}: {prob}")

        input("\nPress Enter to continue to the next slice...")


if __name__ == "__main__":
    explore_midi(
        midi_path="/home/loubb/Dropbox/shared/audio.mid",
        checkpoint_path="/home/loubb/work/aria/models/medium-composer.safetensors",
        metadata_category="composer",
        slice_len_notes=150,
        max_seq_len=512,
    )
