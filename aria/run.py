#!/usr/bin/env python3

import argparse
import os
import re
import sys


def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="aria sample")
    argp.add_argument("-m", help="name of model config file")
    argp.add_argument("-c", help="path to model checkpoint")
    argp.add_argument("-p", help="path to midi file")
    argp.add_argument(
        "-temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-top_p",
        help="sampling top_p value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-cfg",
        help="sampling cfg gamma value",
        type=float,
        required=False,
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when sampling",
    )
    argp.add_argument(
        "-var",
        help="number of variations",
        type=int,
        default=1,
    )
    argp.add_argument(
        "-trunc",
        help="length (in seconds) of the prompt",
        type=int,
        default=20,
    )
    argp.add_argument("-e", action="store_true", help="enable force end")
    argp.add_argument("-l", type=int, help="generation length", default=1024)
    argp.add_argument(
        "-guidance_path", type=str, help="path to guidance MIDI", required=False
    )
    argp.add_argument(
        "-guidance_start_ms",
        help="guidance interval start (ms)",
        type=int,
        required=False,
    )
    argp.add_argument(
        "-guidance_end_ms",
        help="guidance interval end (ms)",
        type=int,
        required=False,
    )
    argp.add_argument("-compile", action="store_true", help="compile cudagraph")

    return argp.parse_args(sys.argv[2:])


def sample(args):
    """Entrypoint for sampling"""

    from torch.cuda import is_available as cuda_is_available
    from aria.inference import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config, load_config
    from aria.tokenizer import InferenceAbsTokenizer
    from aria.sample import (
        sample_batch_cfg,
        sample_batch,
        get_inference_prompt,
    )
    from ariautils.midi import MidiDict
    from aria.utils import _load_weight

    if not cuda_is_available():
        raise Exception("CUDA device is not available.")

    model_state = _load_weight(args.c, "cuda")
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    }

    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    valid_metadata = load_config()["data"]["metadata"]["manual"]
    for k, v in manual_metadata.copy().items():
        assert k in valid_metadata.keys(), f"{manual_metadata} is invalid"
        if v not in valid_metadata[k]:
            print(f"Ignoring invalid manual metadata: {k}")
            print(f"Please choose from {valid_metadata[k]}")
            del manual_metadata[k]

    num_variations = args.var
    truncate_len = args.trunc
    force_end = args.e
    model_name = args.m

    tokenizer = InferenceAbsTokenizer()
    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).cuda()

    try:
        model.load_state_dict(model_state)
    except Exception as e:
        print(
            "Failed to load model_state. This is likely due to an incompatibility "
            "between the checkpoint file (-c) and model name/config (-m)."
        )
        raise e

    assert args.l > 0, "Generation length must be positive."
    max_new_tokens = args.l

    # Load and format prompts and metadata
    midi_dict = MidiDict.from_midi(mid_path=args.p)
    if args.guidance_path:
        guidance_midi_dict = MidiDict.from_midi(mid_path=args.guidance_path)
    else:
        guidance_midi_dict = None

    for k, v in manual_metadata.items():
        midi_dict.metadata[k] = v

    print(f"Extracted metadata: {midi_dict.metadata}")
    print(
        f"Instruments: {set([MidiDict.get_program_to_instrument()[msg['data']] for msg in midi_dict.instrument_msgs])}"
    )

    prompt_seq, guidance_seq = get_inference_prompt(
        tokenizer=tokenizer,
        midi_dict=midi_dict,
        truncate_len=truncate_len,
        guidance_start_ms=args.guidance_start_ms,
        guidance_end_ms=args.guidance_end_ms,
        guidance_midi_dict=guidance_midi_dict,
    )

    if len(prompt_seq) + args.l > model_config.max_seq_len:
        print(
            "WARNING: Required context exceeds max_seq_len supported by model"
        )
    prompts = [prompt_seq for _ in range(num_variations)]

    samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples")
    if os.path.isdir(samples_dir) is False:
        os.mkdir(samples_dir)
    if guidance_seq:
        tokenizer.detokenize(guidance_seq).to_midi().save(
            os.path.join(samples_dir, f"res_{idx + 1}.mid")
        )
    if args.cfg is not None and guidance_seq is not None:
        results = sample_batch_cfg(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            cfg_gamma=args.cfg,
            force_end=force_end,
            temperature=args.temp,
            top_p=args.top_p,
            compile=args.compile,
        )
    else:
        results = sample_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            force_end=force_end,
            temperature=args.temp,
            top_p=args.top_p,
            compile=args.compile,
        )

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(os.path.join(samples_dir, f"res_{idx + 1}.mid"))

    print("Results saved to samples/")


def _parse_midi_dataset_args():
    argp = argparse.ArgumentParser(prog="aria midi-dataset")
    argp.add_argument("dir", help="directory containing midi files")
    argp.add_argument("save_path", help="path to save dataset")
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
    argp.add_argument(
        "-s", action="store_true", help="shuffle dataset", default=False
    )
    argp.add_argument(
        "-metadata",
        nargs=2,
        metavar=("KEY", "VALUE"),
        action="append",
        help="manually add metadata key-value pair when building dataset",
    )
    argp.add_argument(
        "-split", type=float, help="create train/val split", required=False
    )

    return argp.parse_args(sys.argv[2:])


def build_midi_dataset(args):
    """Entrypoint for building MidiDatasets from a directory"""
    from aria.datasets import MidiDataset

    assert args.dir, "build directory must be provided"
    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    MidiDataset.build_to_file(
        dir=args.dir,
        save_path=args.save_path,
        recur=args.r,
        overwrite=True,
        manual_metadata=manual_metadata,
        shuffle=args.s,
    )

    if args.split:
        assert 0.0 < args.split < 1.0, "Invalid range given for -split"
        MidiDataset.split_from_file(
            load_path=args.save_path,
            train_val_ratio=args.split,
            repeatable=True,
        )


def _parse_pretrain_dataset_args():
    argp = argparse.ArgumentParser(prog="aria pretrain-dataset")
    argp.add_argument("-load_path", help="path midi_dict dataset")
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument(
        "-tokenizer_name", help="tokenizer name", choices=["abs", "rel"]
    )
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_pretraining_dataset(args):
    from ariautils.tokenizer import AbsTokenizer, RelTokenizer
    from aria.datasets import PretrainingDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    PretrainingDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.load_path,
    )


def _parse_finetune_dataset_args():
    argp = argparse.ArgumentParser(prog="aria finetune-dataset")
    argp.add_argument(
        "-midi_dataset_path",
        help="path to midi_dict dataset",
    )
    argp.add_argument("-save_dir", help="path to save dataset")
    argp.add_argument("-l", help="max sequence length", type=int, default=4096)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_finetune_dataset(args):
    from aria.tokenizer import InferenceAbsTokenizer
    from aria.datasets import FinetuningDataset

    tokenizer = InferenceAbsTokenizer()
    FinetuningDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.midi_dataset_path,
    )


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="aria <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=(
            "sample",
            "midi-dataset",
            "pretrain-dataset",
            "finetune-dataset",
        ),
    )

    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "sample":
        sample(args=_parse_sample_args())
    elif args.command == "midi-dataset":
        build_midi_dataset(args=_parse_midi_dataset_args())
    elif args.command == "pretrain-dataset":
        build_pretraining_dataset(args=_parse_pretrain_dataset_args())
    elif args.command == "finetune-dataset":
        build_finetune_dataset(args=_parse_finetune_dataset_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
