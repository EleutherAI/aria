import torch
import copy

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from ariautils.tokenizer._base import Token

from aria.model import TransformerEMB


def _validate_midi_for_emb(midi_dict: MidiDict):
    present_instruments = {
        midi_dict.program_to_instrument[msg["data"]]
        for msg in midi_dict.instrument_msgs
    }
    assert present_instruments == {"piano"}, "Only piano MIDIs supported"
    assert len(midi_dict.note_msgs) > 0


def _get_chunks(midi_dict: MidiDict, notes_per_chunk: int):
    res = []

    for note_msg_chunk in [
        midi_dict.note_msgs[idx : idx + notes_per_chunk]
        for idx in range(0, len(midi_dict.note_msgs), notes_per_chunk)
    ]:
        if len(note_msg_chunk) == 0:
            break

        chunked_midi_dict = copy.deepcopy(midi_dict)
        chunked_midi_dict.note_msgs = note_msg_chunk
        chunked_midi_dict.metadata = {}
        res.append(chunked_midi_dict)

    return res


@torch.inference_mode()
def get_embedding_from_seq(
    model: TransformerEMB, seq: list[Token], device="cuda"
):
    tokenizer = AbsTokenizer()

    assert len(seq) <= 2048, "Sequence lengths above 2048 not supported"
    _validate_midi_for_emb(tokenizer.detokenize(seq))

    eos_pos = seq.index(tokenizer.eos_tok)
    seq_enc = torch.tensor(tokenizer.encode(seq), device=device)
    emb = model.forward(seq_enc.view(1, -1))[0, eos_pos]

    return emb


def get_global_embedding_from_midi(
    model: TransformerEMB,
    midi_dict: MidiDict | None = None,
    midi_path: str | None = None,
    notes_per_chunk: int = 300,
    device="cuda",
):
    """Calculates global contrastive embedding by calculating an unweighted
    average of chunk embeddings of notes_per_chunk notes."""

    assert midi_dict or midi_path

    if midi_path:
        midi_dict = MidiDict.from_midi(mid_path=midi_path)

    tokenizer = AbsTokenizer()
    _validate_midi_for_emb(midi_dict)

    chunks = _get_chunks(midi_dict=midi_dict, notes_per_chunk=notes_per_chunk)
    seqs = [tokenizer.tokenize(c, add_dim_tok=False)[:2048] for c in chunks]
    embs = [
        get_embedding_from_seq(model=model, seq=s, device=device) for s in seqs
    ]

    return torch.mean(torch.stack(embs), dim=0)
