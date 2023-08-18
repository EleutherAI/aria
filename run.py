#!/usr/bin/env python3

import argparse
import os
import logging
import sys


def _parse_train_args():
    argp = argparse.ArgumentParser(prog="run.py train")
    argp.add_argument("model", help="name of the model to train")
    argp.add_argument(
        "tokenizer", choices=["lazy"], help="name of the tokenizer to use"
    )
    argp.add_argument("train_data", help="path to train data")
    argp.add_argument("val_data", help="path to val data")
    argp.add_argument("-ckpt", help="path to the checkpoint", required=False)
    argp.add_argument(
        "-epochs", help="number of epochs", type=int, required=True
    )
    argp.add_argument("-bs", help="batch size", type=int, default=32)
    argp.add_argument(
        "-workers", help="number of cpu processes", type=int, default=1
    )
    argp.add_argument("-gpus", help="number of gpus", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def train(args):
    """Entrypoint for training"""
    from aria.training import pretrain

    pretrain(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        num_workers=args.workers,
        num_gpus=args.gpus,
        epochs=args.epochs,
        batch_size=args.bs,
        checkpoint=args.ckpt,
    )


def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="run.py sample")
    argp.add_argument("ckpt_path", help="path to model checkpoint")
    argp.add_argument("midi_path", help="path to midi file")
    argp.add_argument(
        "-var", help="number of variations", type=int, required=True
    )
    argp.add_argument(
        "-trunc", help="length to truncated prompt", type=int, required=True
    )

    return argp.parse_args(sys.argv[2:])


# TODO:
# - Refactor this eventually so that I don't need PretrainLM to load the model
# - Support a variety of tokenizers.
# - Move to sample.py ?
def sample(args):
    """Entrypoint for sampling"""
    from torch.cuda import is_available as cuda_is_available
    from aria.training import PretrainLM
    from aria.tokenizer import TokenizerLazy
    from aria.sample import batch_sample_model
    from aria.data.midi import MidiDict

    assert cuda_is_available() is True, "CUDA device not available"

    ckpt_path = args.ckpt_path
    midi_path = args.midi_path
    num_variations = args.var
    truncate_len = args.trunc

    model = PretrainLM.load_from_checkpoint(ckpt_path).model
    max_seq_len = model.max_seq_len
    tokenizer = TokenizerLazy(
        return_tensors=True,
    )

    assert (
        truncate_len < max_seq_len
    ), "Truncate length longer than maximum length supported by the model."

    # Load and format prompts
    midi_dict = MidiDict.from_midi(mid_path=midi_path)
    prompt_seq = tokenizer.tokenize_midi_dict(midi_dict=midi_dict)
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

    if os.path.isdir("samples") is False:
        os.mkdir("samples")

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize_midi_dict(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(f"samples/res_{idx + 1}.mid")


def _parse_data_args():
    argp = argparse.ArgumentParser(prog="run.py data")
    argp.add_argument(
        "format",
        choices=["midi_dict", "tokenized"],
        help="type of dataset to build",
    )
    argp.add_argument("save_path", help="path to save dataset")

    argp.add_argument(
        "-dir", help="directory containing midi files", required=False
    )
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
    argp.add_argument(
        "-load_path", help="path midi_dict dataset", required=False
    )
    argp.add_argument(
        "-tokenizer",
        required=False,
        choices=["lazy"],
        help="specify tokenizer type",
    )

    return argp.parse_args(sys.argv[2:])


# NOTE: This must be refactored if additional tokenizers are added
def data(args):
    """Entrypoint for data processing"""
    from aria.tokenizer import TokenizerLazy
    from aria.data.datasets import MidiDataset
    from aria.data.datasets import TokenizedDataset
    from aria.config import load_config

    # TODO: Add asserts that files (midi files and load files)

    if args.format == "midi_dict":
        assert args.dir, "build directory must be provided"
        MidiDataset.build_to_file(
            dir=args.dir,
            save_path=args.save_path,
            recur=args.r,
            overwrite=True,
        )

    elif args.format == "tokenized":
        assert not (
            args.dir is None and args.load_path is None
        ), "must provide a load_path or a directory containing midi"

        config = load_config()["data"]["dataset_gen_args"]
        # tokenizer = TokenizerLazy(max_seq_len=config["max_seq_len"])
        tokenizer = TokenizerLazy()
        if args.load_path:
            TokenizedDataset.build(
                tokenizer=tokenizer,
                save_path=args.save_path,
                midi_dataset_path=args.load_path,
                max_seq_len=config["max_seq_len"],
                stride_len=config["stride_len"],
                padding=True,
                overwrite=True,
            )
        elif args.dir:
            buffer_path = "midi_dataset.buffer.jsonl"
            MidiDataset.build_to_file(
                dir=args.dir,
                save_path=buffer_path,
                recur=args.r,
                overwrite=True,
            )
            TokenizedDataset.build(
                tokenizer=tokenizer,
                save_path=args.save_path,
                midi_dataset_path=buffer_path,
                max_seq_len=config["max_seq_len"],
                stride_len=config["stride_len"],
                padding=True,
                overwrite=True,
            )
            os.remove(buffer_path)


def main():
    # Nested argparse inspired by - https://shorturl.at/kuKW0
    parser = argparse.ArgumentParser(usage="run.py <command> [<args>]")
    parser.add_argument(
        "command",
        help="command to run",
        choices=("train", "sample", "data"),
    )

    # parse_args defaults to [1:] for args, but you need to
    # exclude the rest of the args too, or validation will fail
    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "train":
        train(args=_parse_train_args())
    elif args.command == "sample":
        sample(args=_parse_sample_args())
    elif args.command == "data":
        logging.basicConfig(level=logging.INFO)
        data(args=_parse_data_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
