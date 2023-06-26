"""Entrypoint for training"""

import argparse

from aria.training import pretrain


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--model",
        help="name of the model to train",
        required=True,
    )
    argp.add_argument(
        "--tokenizer",
        help="name of the tokenizer to use",
        required=True,
    )
    argp.add_argument("--ckpt", help="path to the checkpoint if required")
    argp.add_argument("--data", help="path to data", required=True)
    argp.add_argument("--workers", type=int, default=1)
    argp.add_argument("--gpus", type=int, default=1)
    argp.add_argument("--epochs", type=int, required=True)
    argp.add_argument("--bs", help="batch size", type=int, default=32)
    kwargs = vars(argp.parse_args())

    return kwargs


if __name__ == "__main__":
    kwargs = parse_arguments()

    pretrain(
        model_name=kwargs.get("model"),
        tokenizer_name=kwargs.get("tokenizer"),
        data_path=kwargs.get("data"),
        workers=kwargs.get("workers"),
        gpus=kwargs.get("gpus"),
        epochs=kwargs.get("epochs"),
        batch_size=kwargs.get("bs"),
        checkpoint=kwargs.get("ckpt"),
    )
