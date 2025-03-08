import os
import argparse
import torch

import torch.nn as nn

from ariautils.tokenizer import AbsTokenizer
from aria.embeddings.evaluate import (
    EvaluationDataset,
    get_aria_contrastive_embedding,
    get_clamp3_embedding,
    get_mert_embedding,
)

MAX_SEQ_LEN = 1024
NUM_SLICE_NOTES = 300
SEQS_BATCH_SIZE = 128


def aria_model_forward(
    model: nn.Module,
    idxs: torch.Tensor,
):
    return model(idxs)


def build_aria_dataset(
    midi_dataset_load_path: str,
    embedding_dataset_save_path: str,
    checkpoint_path: str,
    max_batch_size,
):
    from aria.config import load_model_config
    from aria.utils import _load_weight
    from aria.model import ModelConfig, TransformerEMB

    assert os.path.isfile(midi_dataset_load_path)
    assert os.path.isfile(checkpoint_path)
    assert not os.path.isfile(embedding_dataset_save_path)

    tokenizer = AbsTokenizer()
    model_state = _load_weight(checkpoint_path, "cuda")
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in model_state.items()
    }
    pretrained_model_config = ModelConfig(**load_model_config("medium-emb"))
    pretrained_model_config.set_vocab_size(tokenizer.vocab_size)
    pretrained_model_config.grad_checkpoint = False
    pretrained_model = TransformerEMB(pretrained_model_config)
    pretrained_model.load_state_dict(model_state)
    pretrained_model.eval()

    hook_model_forward = torch.compile(
        aria_model_forward,
        mode="reduce-overhead",
        fullgraph=True,
    )

    EvaluationDataset.build(
        midi_dataset_load_path=midi_dataset_load_path,
        save_path=embedding_dataset_save_path,
        max_seq_len=MAX_SEQ_LEN,
        slice_len_notes=NUM_SLICE_NOTES,
        batch_size=SEQS_BATCH_SIZE,
        embedding_hook=get_aria_contrastive_embedding,
        hook_model=pretrained_model.cuda(),
        hook_max_seq_len=MAX_SEQ_LEN,
        hook_tokenizer=tokenizer,
        hook_model_forward=hook_model_forward,
        hook_max_batch_size=max_batch_size,
    )


def build_m3_dataset(
    midi_dataset_load_path: str,
    embedding_dataset_save_path: str,
    checkpoint_path: str,
):
    from aria.embeddings.m3.emb import load_clamp3_model

    assert os.path.isfile(midi_dataset_load_path)
    assert os.path.isfile(checkpoint_path)
    assert not os.path.isfile(embedding_dataset_save_path)

    tokenizer = AbsTokenizer()
    model, patchilizer = load_clamp3_model(checkpoint_path=checkpoint_path)

    EvaluationDataset.build(
        midi_dataset_load_path=midi_dataset_load_path,
        save_path=embedding_dataset_save_path,
        max_seq_len=MAX_SEQ_LEN,
        slice_len_notes=NUM_SLICE_NOTES,
        batch_size=SEQS_BATCH_SIZE,
        embedding_hook=get_clamp3_embedding,
        hook_model=model,
        hook_patchilizer=patchilizer,
        hook_tokenizer=tokenizer,
    )


def build_mert_dataset(
    midi_dataset_load_path: str,
    embedding_dataset_save_path: str,
    pianoteq_exec_path: str,
    pianoteq_num_procs: int,
):
    from aria.embeddings.mert.emb import load_mert_model

    assert pianoteq_num_procs > 0
    assert os.path.isfile(midi_dataset_load_path)
    assert not os.path.isfile(embedding_dataset_save_path)

    tokenizer = AbsTokenizer()
    model, processor = load_mert_model()

    EvaluationDataset.build(
        midi_dataset_load_path=midi_dataset_load_path,
        save_path=embedding_dataset_save_path,
        max_seq_len=MAX_SEQ_LEN,
        slice_len_notes=NUM_SLICE_NOTES,
        batch_size=SEQS_BATCH_SIZE,
        embedding_hook=get_mert_embedding,
        hook_model=model,
        hook_processor=processor,
        hook_tokenizer=tokenizer,
        hook_pianoteq_exec_path=pianoteq_exec_path,
        hook_pianoteq_num_procs=pianoteq_num_procs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process model and dataset paths."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["aria", "mert", "m3"],
        required=True,
    )
    parser.add_argument(
        "--model_cp_path",
        type=str,
        required=False,
        help="Path from which to load the model.",
    )
    parser.add_argument(
        "--dataset_load_path",
        type=str,
        required=True,
        help="Path from which to load the dataset.",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        required=True,
        help="Path where the dataset will be saved.",
    )
    parser.add_argument(
        "--aria_max_batch_size",
        type=int,
        default=128,
        help="Max batch size for aria embedding forward pass",
    )
    parser.add_argument(
        "--mert_pianoteq_exec_path",
        type=str,
        required=False,
        help="Path to pianoteq executable",
    )
    parser.add_argument(
        "--mert_pianoteq_num_procs",
        type=int,
        default=16,
        help="Num of procs to use for audio synthesis",
    )

    args = parser.parse_args()

    if args.model == "aria":
        assert args.aria_max_batch_size > 0

        # TODO: TEST
        build_aria_dataset(
            midi_dataset_load_path=args.dataset_load_path,
            embedding_dataset_save_path=args.dataset_save_path,
            checkpoint_path=args.model_cp_path,
            max_batch_size=args.aria_max_batch_size,
        )
    elif args.model == "m3":
        # TODO: TEST
        build_m3_dataset(
            midi_dataset_load_path=args.dataset_load_path,
            embedding_dataset_save_path=args.dataset_save_path,
            checkpoint_path=args.model_cp_path,
        )
    elif args.model == "mert":
        # TODO: TEST
        build_mert_dataset(
            midi_dataset_load_path=args.dataset_load_path,
            embedding_dataset_save_path=args.dataset_save_path,
            pianoteq_exec_path=args.mert_pianoteq_exec_path,
            pianoteq_num_procs=args.mert_pianoteq_num_procs,
        )


if __name__ == "__main__":
    main()
