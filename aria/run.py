#!/usr/bin/env python3

import argparse
import os
import json
import sys


def _parse_generate_args():
    argp = argparse.ArgumentParser(prog="aria generate")
    argp.add_argument(
        "--backend",
        choices=["torch_cuda", "mlx"],
        default="torch_cuda",
        help="backend for inference",
    )
    argp.add_argument(
        "--checkpoint_path", help="path to model used for decoding"
    )
    argp.add_argument("--prompt_midi_path", help="path to midi file")
    argp.add_argument(
        "--prompt_duration",
        help="length of the input MIDI prompt, in seconds",
        type=int,
        default=20,
    )
    argp.add_argument(
        "--variations",
        help="number of variations to generate",
        type=int,
        default=1,
    )
    argp.add_argument(
        "--temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.98,
    )
    argp.add_argument(
        "--min_p",
        help="sampling min_p value",
        type=float,
        default=0.035,
        required=False,
    )
    argp.add_argument(
        "--top_p",
        help="sampling top_p value",
        type=float,
        required=False,
    )
    argp.add_argument(
        "--end", action="store_true", help="generate ending for piece"
    )
    argp.add_argument(
        "--length",
        type=int,
        help="number of tokens to generate per variation",
        default=2048,
    )
    argp.add_argument(
        "--compile",
        action="store_true",
        help="use torch compiler to generate cudagraph for inference",
    )
    argp.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="directory to save generated MIDI files",
    )

    return argp.parse_args(sys.argv[2:])


def _parse_conditioned_generate_args():
    argp = argparse.ArgumentParser(prog="aria generate")
    argp.add_argument(
        "--backend",
        choices=["torch_cuda", "mlx"],
        default="torch_cuda",
        help="backend for inference",
    )
    argp.add_argument(
        "--checkpoint_path", help="path to model used for decoding"
    )
    argp.add_argument("--prompt_midi_path", help="path to midi file")
    argp.add_argument(
        "--prompt_duration",
        help="length of the input MIDI prompt, in seconds",
        type=int,
        default=20,
    )
    argp.add_argument(
        "--embedding_model_checkpoint_path",
        help="path to model checkpoint used for embeddings",
    )
    argp.add_argument(
        "--embedding_midi_path",
        help="path to MIDI file used for conditioning",
    )
    argp.add_argument(
        "--variations",
        help="number of variations to generate",
        type=int,
        default=1,
    )
    argp.add_argument(
        "--temp",
        help="sampling temperature value",
        type=float,
        required=False,
        default=0.98,
    )
    argp.add_argument(
        "--cfg",
        help="sampling cfg gamma value",
        type=float,
        default=1.0,
    )
    argp.add_argument(
        "--min_p",
        help="sampling min_p value",
        type=float,
        default=0.035,
        required=False,
    )
    argp.add_argument(
        "--top_p",
        help="sampling top_p value",
        type=float,
        required=False,
    )
    argp.add_argument(
        "--end", action="store_true", help="generate ending for piece"
    )
    argp.add_argument(
        "--length",
        type=int,
        help="number of tokens to generate per variation",
        default=2048,
    )
    argp.add_argument(
        "--compile",
        action="store_true",
        help="use torch compiler to generate cudagraph for inference",
    )
    argp.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="directory to save generated MIDI files",
    )

    return argp.parse_args(sys.argv[2:])


def _get_prompt(
    midi_path: str,
    prompt_duration_s: int,
):
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria.inference import get_inference_prompt

    return get_inference_prompt(
        midi_dict=MidiDict.from_midi(midi_path),
        tokenizer=AbsTokenizer(),
        prompt_len_ms=1e3 * prompt_duration_s,
    )


def _load_embedding_model(checkpoint_path: str):
    from safetensors.torch import load_file

    from ariautils.tokenizer import AbsTokenizer
    from aria.model import TransformerEMB, ModelConfig
    from aria.config import load_model_config

    model_config = ModelConfig(**load_model_config(name="medium-emb"))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerEMB(model_config)

    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=True)

    return model


def _load_inference_model_torch(
    checkpoint_path: str,
    config_name: str,
    strict: bool = True,
):
    from safetensors.torch import load_file

    from ariautils.tokenizer import AbsTokenizer
    from aria.inference.model_cuda import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config

    model_config = ModelConfig(**load_model_config(name=config_name))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerLM(model_config)

    state_dict = load_file(filename=checkpoint_path)
    model.load_state_dict(state_dict=state_dict, strict=strict)

    return model


def _load_inference_model_mlx(
    checkpoint_path: str,
    config_name: str,
    strict: bool = True,
):
    import mlx.core as mx

    from ariautils.tokenizer import AbsTokenizer
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
    from aria.config import load_model_config

    model_config = ModelConfig(**load_model_config(name=config_name))
    model_config.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerLM(model_config)
    model.load_weights(checkpoint_path, strict=strict)
    mx.eval(model.parameters())

    return model


def generate(args):
    from ariautils.tokenizer import AbsTokenizer

    num_variations = args.variations
    prompt_duration_s = args.prompt_duration
    backend = args.backend
    max_new_tokens = args.length

    assert num_variations > 0
    assert prompt_duration_s >= 0
    assert max_new_tokens > 0
    assert os.path.isdir(args.save_dir)

    tokenizer = AbsTokenizer()
    prompt = _get_prompt(
        args.prompt_midi_path,
        prompt_duration_s=prompt_duration_s,
    )
    max_new_tokens = min(8096 - len(prompt), max_new_tokens)

    if backend == "torch_cuda":
        from torch.cuda import is_available
        from aria.inference.sample_cuda import sample_batch as sample_batch_t

        assert is_available(), "CUDA not available"

        model = _load_inference_model_torch(
            checkpoint_path=args.checkpoint_path,
            config_name="medium",
            strict=True,
        )  # Might want strict = False
        results = sample_batch_t(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_variations=num_variations,
            max_new_tokens=max_new_tokens,
            temp=args.temp,
            force_end=args.end,
            top_p=args.top_p,
            min_p=args.min_p,
            compile=args.compile,
        )
    elif backend == "mlx":
        from aria.inference.sample_mlx import sample_batch as sample_batch_mlx

        model = _load_inference_model_mlx(
            checkpoint_path=args.checkpoint_path,
            config_name="medium",
            strict=True,
        )
        results = sample_batch_mlx(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_variations=num_variations,
            max_new_tokens=max_new_tokens,
            temp=args.temp,
            force_end=args.end,
            top_p=args.top_p,
            min_p=args.min_p,
        )

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(os.path.join(args.save_dir, f"res_{idx + 1}.mid"))

    print(f"Results saved to {os.path.realpath(args.save_dir)}")


# TODO: Double checking during training we didn't do a weighted global sum
def _get_embedding(
    embedding_model_checkpoints_path: str,
    embedding_midi_path: str,
):
    from aria.embedding import get_global_embedding_from_midi

    model = _load_embedding_model(
        checkpoint_path=embedding_model_checkpoints_path
    ).cpu()
    global_embedding = get_global_embedding_from_midi(
        model=model,
        midi_path=embedding_midi_path,
        device="cpu",
    )

    return global_embedding.tolist()


def conditioned_generate(args):
    from ariautils.tokenizer import AbsTokenizer

    num_variations = args.variations
    prompt_duration_s = args.prompt_duration
    backend = args.backend
    max_new_tokens = args.length

    assert num_variations > 0
    assert prompt_duration_s >= 0
    assert max_new_tokens > 0
    assert os.path.isdir(args.save_dir)

    tokenizer = AbsTokenizer()
    prompt = _get_prompt(
        args.prompt_midi_path,
        prompt_duration_s=prompt_duration_s,
    )
    embedding = _get_embedding(
        embedding_model_checkpoints_path=args.embedding_model_checkpoint_path,
        embedding_midi_path=args.embedding_midi_path,
    )
    max_new_tokens = min(8096 - len(prompt), max_new_tokens)

    if backend == "torch_cuda":
        from torch.cuda import is_available
        from aria.inference.sample_cuda import (
            sample_batch_cfg as sample_batch_cfg_t,
        )

        assert is_available(), "CUDA not available"

        model = _load_inference_model_torch(
            checkpoint_path=args.checkpoint_path,
            config_name="medium-emb",
            strict=True,
        )
        results = sample_batch_cfg_t(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_variations=num_variations,
            max_new_tokens=max_new_tokens,
            cfg_gamma=args.cfg,
            embedding=embedding,
            temp=args.temp,
            force_end=args.end,
            top_p=args.top_p,
            min_p=args.min_p,
            compile=args.compile,
        )

    elif backend == "mlx":
        from aria.inference.sample_mlx import (
            sample_batch_cfg as sample_batch_cfg_mlx,
        )

        model = _load_inference_model_mlx(
            checkpoint_path=args.checkpoint_path,
            config_name="medium-emb",
            strict=True,
        )
        results = sample_batch_cfg_mlx(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_variations=num_variations,
            max_new_tokens=max_new_tokens,
            cfg_gamma=args.cfg,
            embedding=embedding,
            temp=args.temp,
            force_end=args.end,
            top_p=args.top_p,
            min_p=args.min_p,
        )

    for idx, tokenized_seq in enumerate(results):
        res_midi_dict = tokenizer.detokenize(tokenized_seq)
        res_midi = res_midi_dict.to_midi()
        res_midi.save(os.path.join(args.save_dir, f"res_{idx + 1}.mid"))

    print(f"Results saved to {os.path.realpath(args.save_dir)}")


# TODO: Add turn - to -- flags
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


# TODO: Add turn - to -- flags
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
            "generate",
            "conditioned-generate",
            "midi-dataset",
            "pretrain-dataset",
        ),
    )

    args = parser.parse_args(sys.argv[1:2])

    if not hasattr(args, "command"):
        parser.print_help()
        print("Unrecognized command")
        exit(1)
    elif args.command == "generate":
        generate(args=_parse_generate_args())
    elif args.command == "conditioned-generate":
        conditioned_generate(args=_parse_conditioned_generate_args())
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
