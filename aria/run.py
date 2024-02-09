#!/usr/bin/env python3

import argparse
import os
import re
import sys
import pathlib
import warnings


# TODO: Implement a way of inferring the tokenizer name automatically
def _parse_sample_args():
    argp = argparse.ArgumentParser(prog="aria sample")
    argp.add_argument(
        "-tok",
        help="name of tokenizer",
        choices=["abs", "rel"],
        required=True,
    )
    argp.add_argument("-m", help="name of model config file")
    argp.add_argument("-c", help="path to model checkpoint")
    argp.add_argument("-p", help="path to midi file")
    argp.add_argument(
        "-cfg",
        help="change cfg value",
        type=float,
        required=False,
        default=1.05,
    )
    argp.add_argument(
        "-temp",
        help="change temp value",
        type=float,
        required=False,
        default=0.95,
    )
    argp.add_argument(
        "-top_p",
        help="change top_p value",
        type=float,
        required=False,
        default=0.95,
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
        help="length to truncated prompt",
        type=int,
        default=200,
    )
    argp.add_argument("-e", action="store_true", help="enable force end")
    argp.add_argument("-l", type=int, help="generation length", default=1024)
    argp.add_argument("-q", action="store_true", help="quantize the model")
    argp.add_argument(
        "-sup", action="store_true", help="suppress fluidsynth", default=False
    )

    return argp.parse_args(sys.argv[2:])


def _get_model_name(name: str | None, state: dict):
    if name is not None:
        return name

    print("Model name is not provided. Trying to infer from checkpoint...")
    _defaults = {
        16: "small",
        32: "medium",
        64: "large",
        96: "xlarge",
    }
    try:
        pattern = re.compile(r"encode_layers\.(\d+)\.")
        layer_keys = [pattern.search(k) for k in state.keys()]
        layer_keys = set(p.group(1) for p in layer_keys if p is not None)
        for i in range(len(layer_keys)):
            assert str(i) in layer_keys

        if len(layer_keys) in _defaults:
            print(f"Selecting model name: {_defaults[len(layer_keys)]}")
            return _defaults[len(layer_keys)]
        assert False
    except:
        raise ValueError("Model name is not provided and cannot be inferred.")


def _show_popup(prompt: str, files: list) -> str:
    for i in range(len(files)):
        print(f"  [{i}] {files[i]}")

    for tries in range(3):  # 3 tries in case of fat fingers
        try:
            res = int(input(prompt + f" [0-{len(files) - 1}]: "))
            assert 0 <= res < len(files)
            return files[res]
        except:
            print("Invalid input. Try again...")

    raise ValueError("Invalid input.")


def _get_ckpt_path(ckpt_path: str | None) -> str:
    if ckpt_path is None:
        ckpts = list(pathlib.Path(".").glob("*.bin"))
        ckpt_path = _show_popup("Choose a checkpoint", ckpts)
    return ckpt_path


def _get_midi_path(midi_path: str | None) -> str:
    if midi_path is None:
        midis = list(pathlib.Path(".").glob("*.mid")) + list(
            pathlib.Path(".").glob("*.midi")
        )
        midi_path = _show_popup("Choose a midi-file", midis)
    return midi_path


def sample(args):
    """Entrypoint for sampling"""

    import torch
    from torch.cuda import is_available as cuda_is_available
    from aria.model import TransformerLM, ModelConfig
    from aria.config import load_model_config, load_config
    from aria.tokenizer import RelTokenizer, AbsTokenizer
    from aria.sample import greedy_sample
    from aria.data.midi import MidiDict
    from aria.utils import midi_to_audio, _load_weight

    if not cuda_is_available():
        print("CUDA device is not available. Using CPU instead.")
    else:
        greedy_sample = torch.autocast(device_type="cuda", dtype=torch.float16)(
            greedy_sample
        )
    device = (
        torch.device("cuda") if cuda_is_available() else torch.device("cpu")
    )

    ckpt_path = _get_ckpt_path(args.c)  # let user input path if not provided
    model_state = _load_weight(ckpt_path, device=device.type)
    model_name = _get_model_name(
        args.m, model_state
    )  # infer model name if not provided

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

    if args.tok == "abs":
        tokenizer = AbsTokenizer(return_tensors=True)
    elif args.tok == "rel":
        tokenizer = RelTokenizer(return_tensors=True)

    model_config = ModelConfig(**load_model_config(model_name))
    model_config.set_vocab_size(tokenizer.vocab_size)
    model_config.grad_checkpoint = False
    model = TransformerLM(model_config).to(device)

    if args.trunc + args.l > model_config.max_seq_len:
        print("WARNING - required context exceeds max_seq_len")

    try:
        model.load_state_dict(model_state)
    except:
        print(
            "Failed to load state_dict, this could be because the wrong "
            "tokenizer was selected"
        )
    if args.q:
        if device.type != "cpu":
            warnings.warn(
                "Quantization is not supported on CUDA devices. Using CPU instead."
            )
            device = torch.device("cpu")

        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.quantization.quantize_fx import prepare_fx, convert_fx

        qconfig_mapping = get_default_qconfig_mapping()

        def _quantize(module, key, input_shape):
            inp = torch.randn(input_shape, dtype=torch.float, device=device)
            m = prepare_fx(
                getattr(module, key), qconfig_mapping, example_inputs=inp
            )
            m = convert_fx(m)
            setattr(module, key, m)

        for i in range(len(model.model.encode_layers)):
            _quantize(
                model.model.encode_layers[i],
                "mixed_qkv",
                input_shape=(1, 2048, model_config.n_heads),
            )
            _quantize(
                model.model.encode_layers[i],
                "att_proj_linear",
                input_shape=(1, 2048, model_config.n_heads),
            )
            _quantize(
                model.model.encode_layers[i],
                "ff_linear_1",
                input_shape=(1, 2048, model_config.n_heads),
            )
            _quantize(
                model.model.encode_layers[i],
                "ff_linear_2",
                input_shape=(
                    1,
                    2048,
                    model_config.n_heads * model_config.ff_mult,
                ),
            )

    midi_path = _get_midi_path(
        args.p
    )  # let user input midi path if not provided

    assert args.l > 0, "Generation length must be positive."
    max_new_tokens = args.l

    # Load and format prompts and metadata
    midi_dict = MidiDict.from_midi(mid_path=midi_path)
    for k, v in manual_metadata.items():
        midi_dict.metadata[k] = v

    print(f"Extracted metadata: {midi_dict.metadata}")
    print(
        f"Instruments: {set([MidiDict.get_program_to_instrument()[msg['data']] for msg in midi_dict.instrument_msgs])}"
    )  # Not working with al.mid ?
    prompt_seq = tokenizer.tokenize(midi_dict=midi_dict)
    prompt_seq = prompt_seq[:truncate_len]
    print(prompt_seq[: prompt_seq.index(tokenizer.bos_tok)])
    prompts = [prompt_seq for _ in range(num_variations)]

    # Sample
    results = greedy_sample(
        model,
        tokenizer,
        prompts,
        device=device,
        force_end=force_end,
        max_new_tokens=max_new_tokens,
        cfg_gamma=args.cfg,
        temperature=args.temp,
        top_p=args.top_p,
    )

    if os.path.isdir("samples") is False:
        os.mkdir("samples")

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(f"samples/res_{idx + 1}.mid")
        if args.sup is False:
            midi_to_audio(f"samples/:res_{idx + 1}.mid")

    print("Results saved to samples/")


def _parse_midi_dataset_args():
    argp = argparse.ArgumentParser(prog="aria midi-dataset")
    argp.add_argument("dir", help="directory containing midi files")
    argp.add_argument("save_path", help="path to save dataset")
    argp.add_argument("-r", action="store_true", help="recursively search dirs")
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
    from aria.data.datasets import MidiDataset

    assert args.dir, "build directory must be provided"
    manual_metadata = {k: v for k, v in args.metadata} if args.metadata else {}
    MidiDataset.build_to_file(
        dir=args.dir,
        save_path=args.save_path,
        recur=args.r,
        overwrite=True,
        manual_metadata=manual_metadata,
        shuffle=True,
    )

    if args.split:
        assert 0.0 < args.split < 1.0, "Invalid range given for -split"
        MidiDataset.split_from_file(
            load_path=args.save_path,
            train_val_ratio=args.split,
        )


def _parse_pretrain_dataset_args():
    argp = argparse.ArgumentParser(prog="aria pretrain-dataset")
    argp.add_argument("load_path", help="path midi_dict dataset")
    argp.add_argument("save_dir", help="path to save dataset")
    argp.add_argument(
        "tokenizer_name", help="tokenizer name", choices=["abs", "rel"]
    )
    argp.add_argument("-l", help="max sequence length", type=int, default=2048)
    argp.add_argument("-e", help="num epochs", type=int, default=1)

    return argp.parse_args(sys.argv[2:])


def build_pretraining_dataset(args):
    from aria.tokenizer import AbsTokenizer, RelTokenizer
    from aria.data.datasets import PretrainingDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    dataset = PretrainingDataset.build(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        max_seq_len=args.l,
        num_epochs=args.e,
        midi_dataset_path=args.load_path,
    )


def _parse_finetune_dataset_args():
    argp = argparse.ArgumentParser(prog="aria finetune-dataset")
    argp.add_argument("load_path", help="path midi_dict dataset")
    argp.add_argument("save_path", help="path to save dataset")
    argp.add_argument(
        "tokenizer_name", help="tokenizer name", choices=["abs", "rel"]
    )
    argp.add_argument("-l", help="max sequence length", type=int, default=2048)
    argp.add_argument("-s", help="stride length", type=int, default=512)

    return argp.parse_args(sys.argv[2:])


def build_finetune_dataset(args):
    from aria.tokenizer import AbsTokenizer, RelTokenizer
    from aria.data.datasets import FinetuningDataset

    if args.tokenizer_name == "abs":
        tokenizer = AbsTokenizer()
    elif args.tokenizer_name == "rel":
        tokenizer = RelTokenizer()

    dataset = FinetuningDataset.build(
        tokenizer=tokenizer,
        save_path=args.save_path,
        max_seq_len=args.l,
        stride_len=args.s,
        midi_dataset_path=args.load_path,
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
