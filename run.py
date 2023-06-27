#!/usr/bin/env python3

import argparse
import sys


def _parse_train_args():
    argp = argparse.ArgumentParser(
        usage="""run.py train [-h] [-ckpt CKPT] -epochs EPOCHS [-bs BS] [-workers WORKERS] [-gpus GPUS] model tokenizer data"""
    )
    argp.add_argument(
        "model",
        help="name of the model to train",
    )
    argp.add_argument(
        "tokenizer",
        help="name of the tokenizer to use",
    )
    argp.add_argument("data", help="path to data")
    argp.add_argument(
        "-ckpt",
        help="path to the checkpoint",
        required=False,
    )
    argp.add_argument(
        "-epochs",
        help="number of epochs",
        type=int,
        required=True,
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
        data_path=args.data,
        workers=args.workers,
        gpus=args.gpus,
        epochs=args.epochs,
        batch_size=args.bs,
        checkpoint=args.ckpt,
    )


def _parse_sample_args():
    argp = argparse.ArgumentParser(
        usage="run.py sample [-h] -var VAR -trunc TRUNC ckpt_path midi_path"
    )
    argp.add_argument("ckpt_path", help="path to model checkpoint")
    argp.add_argument("midi_path", help="path to midi file")
    argp.add_argument("-var", help="number of variations", required=True)
    argp.add_argument(
        "-trunc", help="length to truncated prompt", required=True
    )

    return argp.parse_args(sys.argv[2:])


# TODO:
# - Refactor this eventually so that I don't need PretrainLM to load the model
# - Support a variety of tokenizers.
# - Move to sample.py ?
def sample(args):
    """Entrypoint for sampling"""
    import mido
    from torch.cuda import is_available as cuda_is_available
    from aria.training import PretrainLM
    from aria.tokenizer import TokenizerLazy
    from aria.sample import batch_sample_model
    from aria.data.midi import MidiDict

    assert cuda_is_available() is True, "CUDA device not available"

    model_path = args.model_path
    midi_path = args.midi_path
    num_variations = args.n
    truncate_len = args.t

    model = PretrainLM.load_from_checkpoint(model_path).model
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


def _parse_data_args():
    argp = argparse.ArgumentParser(
        usage="""run.py data [-h] [-r] [-tokenizer {lazy}] [-split SPLIT] dir save_path {midi_dict,tokenized}"""
    )
    argp.add_argument(
        "dir",
        help="directory containing midi files",
    )
    argp.add_argument(
        "save_path",
        help="path to save dataset",
    )
    argp.add_argument(
        "format",
        choices=["midi_dict", "tokenized"],
        help="type of dataset to build",
    )
    argp.add_argument(
        "-r",
        action="store_true",
        help="recursively search dirs",
    )
    argp.add_argument(
        "-tokenizer",
        required=False,
        choices=["lazy"],
        help="specify tokenizer type",
    )
    argp.add_argument(
        "-split",
        required=False,
        type=float,
        help="train-val split ratio",
    )
    return argp.parse_args(sys.argv[2:])


# TODO
# - Test
def data(args):
    """Entrypoint for data processing"""
    from aria.tokenizer import TokenizerLazy
    from aria.data.datasets import MidiDataset
    from aria.data.datasets import TokenizedDataset
    from aria.config import load_config

    midi_dataset = MidiDataset.build(dir=args.dir, recur=args.r)

    if args.format == "midi_dict":
        midi_dataset.save(args.save_path)
    elif args.format == "tokenized":
        if args.tokenizer is None:
            raise ValueError(
                "must provide -tokenizer flag if using tokenized mode."
            )

        config = load_config()["tokenizer"][args.tokenizer]["default_args"]
        if args.tokenizer == "lazy":
            tokenizer = TokenizerLazy(**config)
        else:
            return NotImplementedError

        tokenized_dataset = TokenizedDataset.build(midi_dataset, tokenizer)
        if args.split:
            tokenized_dataset.save_train_val(args.save_path, args.split)
        else:
            tokenized_dataset.save(args.save_path)


def main():
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
        data(args=_parse_data_args())
    else:
        print("Unrecognized command")
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
