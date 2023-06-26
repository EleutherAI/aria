"""Entrypoint for generating samples."""

import argparse
import torch
import mido

from aria.training import PretrainLM
from aria.tokenizer import TokenizerLazy
from aria.sample import batch_sample_model
from aria.data.midi import MidiDict


# TODO:
# - Refactor this eventually so that I don't need PretrainLM to load the model
# - Support a variety of tokenizers.


def _get_torch_module(load_path: str):
    """Returns torch nn.Module from pl.LightningModule.

    Note that all args passed into the Lightning module are saved in the
    checkpoint. Because of this, we do not need the model_config as a param.
    """
    return PretrainLM.load_from_checkpoint(load_path).model


def generate(
    model_path: str,
    midi_path: str,
    num_variations: int,
    truncate_len: int,
):
    assert torch.cuda.is_available() is True, "CUDA device not available"

    # Load model and tokenizer
    model = _get_torch_module(model_path)
    max_seq_len = model.max_seq_len
    tokenizer = TokenizerLazy(
        padding=True,
        truncate_type="default",
        max_seq_len=max_seq_len,
        return_tensors=True,
    )

    assert (
        truncate_len < max_seq_len
    ), "Truncate length longer than maximum length supported by the model."

    # Load and format prompts
    midi_dict = MidiDict.from_midi(
        mid=mido.MidiFile(midi_path),
    )
    prompt_seq = tokenizer.tokenize_midi_dict(midi_dict=midi_dict)[0]
    prompt_seq = prompt_seq[:truncate_len]
    prompts = [prompt_seq for _ in range(num_variations)]

    # Sample
    results = batch_sample_model(
        model,
        tokenizer,
        prompts,
        max_seq_len,
        max_gen_len=max_seq_len,
    )

    for idx, tokenized_seq in results:
        res_midi_dict = tokenizer.detokenize_midi_dict(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(f"res_{idx}.mid")


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument("--model_path", help="path to model checkpoint")
    argp.add_argument("--midi_path", help="path to midi file")
    argp.add_argument("-n", help="number of variations", required=True)
    argp.add_argument("-t", help="length of truncated prompt", required=True)
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()

    generate(
        model_path=kwargs.get("model_path"),
        midi_path=kwargs.get("midi_path"),
        num_variations=kwargs.get("n"),
        truncate_len=kwargs.get("t"),
    )
