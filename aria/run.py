#!/usr/bin/env python3

import argparse
import os
import json
import sys


def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="aria sample")
    argp.add_argument(
        "-checkpoint_path", help="path to model used for decoding"
    )
    argp.add_argument("-prompt_midi_path", help="path to midi file")
    argp.add_argument(
        "-embedding_checkpoint_path",
        required=False,
        help="path to model checkpoint used for embeddings",
    )
    argp.add_argument(
        "-embedding_midi_path", required=False, help="path to midi file"
    )
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
    argp.add_argument("-compile", action="store_true", help="compile cudagraph")

    return argp.parse_args(sys.argv[2:])


def _get_embedding(
    embedding_checkpoint_path: str,
    midi_path: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
):
    import torch

    from aria.model import TransformerEMB
    from aria.model import ModelConfig
    from aria.config import load_model_config
    from aria.utils import _load_weight
    from aria.embeddings.evaluate import (
        get_aria_contrastive_embedding,
        process_entry,
    )

    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer

    SLICE_NUM_NOTES = 300
    SLICE_MAX_SEQ_LEN = 1024

    tokenizer = AbsTokenizer()

    model_state = _load_weight(embedding_checkpoint_path, "cuda")
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    }

    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerEMB(model_config).cuda().eval()
    model.load_state_dict(model_state)

    midi_dict = MidiDict.from_midi(midi_path)
    midi_dict.note_msgs = [
        msg
        for msg in midi_dict.note_msgs
        if (
            midi_dict.tick_to_ms(msg["tick"]) >= start_ms
            if start_ms is not None
            else True
        )
        and (
            midi_dict.tick_to_ms(msg["tick"]) <= end_ms
            if end_ms is not None
            else True
        )
    ]

    seqs = process_entry(
        entry=midi_dict,
        slice_len_notes=SLICE_NUM_NOTES,
        max_seq_len=SLICE_MAX_SEQ_LEN,
        tokenizer=tokenizer,
    )

    def model_forward(model, idxs):
        return model(idxs)

    embeddings = get_aria_contrastive_embedding(
        seqs=[s["seq"] for s in seqs],
        hook_model=model,
        hook_max_seq_len=SLICE_MAX_SEQ_LEN,
        hook_tokenizer=tokenizer,
        hook_model_forward=model_forward,
    )
    embedding = torch.tensor(embeddings, device="cuda").mean(0).tolist()

    return embedding


def sample(args):
    """Entrypoint for sampling"""

    from torch.cuda import is_available as cuda_is_available
    from aria.inference import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config
    from aria.sample import sample_batch, sample_batch_cfg, get_inference_prompt
    from aria.utils import _load_weight

    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer

    if not cuda_is_available():
        raise Exception("CUDA device is not available.")

    num_variations = args.var
    truncate_len = args.trunc
    force_end = args.e

    tokenizer = AbsTokenizer()

    if args.embedding_checkpoint_path and args.embedding_midi_path:
        print(f"Using embedding from {args.embedding_midi_path}")
        embedding = _get_embedding(
            embedding_checkpoint_path=args.embedding_checkpoint_path,
            midi_path=args.embedding_midi_path,
        )
    else:
        embedding = None

    model_state = _load_weight(args.checkpoint_path, "cuda")
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    }

    model_config = ModelConfig(**load_model_config("medium-emb"))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).cuda()

    try:
        model.load_state_dict(model_state)
    except Exception as e:
        print("Failed to load model_state - loading with strict=False")
        model.load_state_dict(model_state, strict=False)

    assert args.l > 0, "Generation length must be positive."
    max_new_tokens = args.l

    # Load and format prompts and metadata
    midi_dict = MidiDict.from_midi(mid_path=args.prompt_midi_path)

    prompt_seq = get_inference_prompt(
        tokenizer=tokenizer,
        midi_dict=midi_dict,
        prompt_len_ms=truncate_len * 1e3,
    )

    print(prompt_seq)

    if len(prompt_seq) + args.l > model_config.max_seq_len:
        print(
            "WARNING: Required context exceeds max_seq_len supported by model"
        )
    prompts = [prompt_seq for _ in range(num_variations)]

    samples_dir = "/home/loubb/Dropbox/shared"
    if os.path.isdir(samples_dir) is False:
        os.mkdir(samples_dir)

    if args.cfg and embedding is not None:
        results = sample_batch_cfg(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            force_end=force_end,
            temperature=args.temp,
            top_p=args.top_p,
            cfg_gamma=args.cfg,
            compile=args.compile,
            embedding=embedding,
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
            embedding=embedding,
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
    argp.add_argument(
        "-sep_sequences",
        help="start each with a new entry",
        action="store_true",
    )
    argp.add_argument("-embedding_dataset_path", required=False)

    return argp.parse_args(sys.argv[2:])


def build_pretraining_dataset(args):
    from ariautils.tokenizer import AbsTokenizer, RelTokenizer
    from aria.datasets import PretrainingDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    if args.embedding_dataset_path is not None:
        with open(args.embedding_dataset_path, "r") as f:
            file_embeddings = {
                data["metadata"]["abs_load_path"]: data["emb"]
                for data in map(json.loads, f)
            }

    else:
        file_embeddings = None

    PretrainingDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.load_path,
        separate_sequences=args.sep_sequences,
        file_embeddings=file_embeddings,
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
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
