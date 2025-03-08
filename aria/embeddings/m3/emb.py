import os
import torch
import mido
from transformers import BertConfig

from aria.embeddings.m3.config import (
    AUDIO_HIDDEN_SIZE,
    AUDIO_NUM_LAYERS,
    MAX_AUDIO_LENGTH,
    M3_HIDDEN_SIZE,
    PATCH_NUM_LAYERS,
    PATCH_LENGTH,
    PATCH_SIZE,
    CLAMP3_HIDDEN_SIZE,
    CLAMP3_LOAD_M3,
    TEXT_MODEL_NAME,
)

from aria.embeddings.m3.utils import CLaMP3Model, M3Patchilizer


def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode("unicode_escape").decode("utf-8")


def load_midi(
    filename: str | None = None,
    mid: mido.MidiFile | None = None,
    m3_compatible: bool = True,
):
    """
    Load a MIDI file and convert it to MTF format.
    """

    if mid is None:
        assert os.path.isfile(filename)
        mid = mido.MidiFile(filename)

    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

    # Merge tracks manually using mido.merge_tracks()
    merged = mido.merge_tracks(mid.tracks)

    for msg in merged:
        if m3_compatible and msg.is_meta:
            if msg.type in [
                "text",
                "copyright",
                "track_name",
                "instrument_name",
                "lyrics",
                "marker",
                "cue_marker",
                "device_name",
            ]:
                continue
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)

    return "\n".join(msg_list)


def load_clamp3_model(checkpoint_path: str):
    """
    Loads the CLaMP3 model along with its configuration and M3Patchilizer.
    The model weights are loaded from the checkpoint specified in CLAMP3_WEIGHTS_PATH.
    """
    # Build the configurations for audio and symbolic (M3) parts.
    audio_config = BertConfig(
        vocab_size=1,
        hidden_size=AUDIO_HIDDEN_SIZE,
        num_hidden_layers=AUDIO_NUM_LAYERS,
        num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
        intermediate_size=AUDIO_HIDDEN_SIZE * 4,
        max_position_embeddings=MAX_AUDIO_LENGTH,
    )
    symbolic_config = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )

    # Instantiate the CLaMP3 model.
    model = CLaMP3Model(
        audio_config=audio_config,
        symbolic_config=symbolic_config,
        text_model_name=TEXT_MODEL_NAME,
        hidden_size=CLAMP3_HIDDEN_SIZE,
        load_m3=CLAMP3_LOAD_M3,
    )
    model = model.to("cuda")
    model.eval()

    # Determine checkpoint path.
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint weights.
    checkpoint = torch.load(
        checkpoint_path, map_location="cuda", weights_only=True
    )
    model.load_state_dict(checkpoint["model"])

    # Instantiate the patchilizer from utils.
    patchilizer = M3Patchilizer()

    return model, patchilizer


def get_midi_embedding(
    mid: mido.MidiFile,
    model: torch.nn.Module,
    patchilizer: M3Patchilizer,
    get_global=True,
):
    device = "cuda"
    # Step 1: Convert MIDI to an MTF-format string.
    mtf_str = load_midi(mid=mid, m3_compatible=True)

    # Step 3: Encode the MTF string to patches.
    # The patchilizer returns a list of patches (each patch is a list of token IDs).
    patches = patchilizer.encode(mtf_str, add_special_patches=True)

    # Convert the list of patches to a tensor.
    # Each patch is of length PATCH_SIZE; we assume tokens are integers.
    # The resulting tensor shape will be [num_patches, PATCH_SIZE].
    token_tensor = torch.tensor(patches, dtype=torch.long).to(device)

    # Step 4: Create segments of fixed length PATCH_LENGTH.
    num_tokens = token_tensor.size(0)
    segments = []
    seg_weights = []
    for i in range(0, num_tokens, PATCH_LENGTH):
        seg = token_tensor[i : i + PATCH_LENGTH]
        cur_len = seg.size(0)
        segments.append(seg)
        seg_weights.append(cur_len)
    # For global feature extraction, ensure the last segment is exactly PATCH_LENGTH tokens
    if num_tokens > PATCH_LENGTH:
        segments[-1] = token_tensor[-PATCH_LENGTH:]
        seg_weights[-1] = segments[-1].size(0)

    # Step 5: Process each segment to obtain features.
    processed_feats = []
    for seg in segments:
        cur_len = seg.size(0)
        # Pad the segment if it's shorter than PATCH_LENGTH.
        if cur_len < PATCH_LENGTH:
            pad = torch.full(
                (
                    PATCH_LENGTH - cur_len,
                    token_tensor.size(1),
                ),  # include PATCH_SIZE dimension
                patchilizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            seg = torch.cat([seg, pad], dim=0)
        seg = seg.unsqueeze(0)  # Add batch dimension.
        # Create a mask: 1 for valid tokens, 0 for padding.
        mask = torch.cat(
            [
                torch.ones(cur_len, device=device),
                torch.zeros(PATCH_LENGTH - cur_len, device=device),
            ],
            dim=0,
        ).unsqueeze(0)
        with torch.no_grad():
            feat = model.get_symbolic_features(
                symbolic_inputs=seg, symbolic_masks=mask, get_global=get_global
            )
        # When not getting a global feature, you may wish to trim the output to the valid length.
        if not get_global:
            feat = feat[:, : int(mask.sum().item()), :]
        processed_feats.append(feat)

    # Step 6: Combine segment features into a single embedding.
    if not get_global:
        # Concatenate features along the time dimension.
        embedding = torch.cat(
            [feat.squeeze(0) for feat in processed_feats], dim=0
        )
    else:
        # For a global embedding, compute a weighted average of segment features.
        feats = torch.stack(
            [feat.squeeze(0) for feat in processed_feats], dim=0
        )
        weights = torch.tensor(
            seg_weights, dtype=torch.float, device=device
        ).view(-1, 1)
        embedding = (feats * weights).sum(dim=0) / weights.sum()

    return embedding.view(-1)


# Example usage:
if __name__ == "__main__":
    midi_file_path = "/home/loubb/Dropbox/shared/test/test.mid"
    model, patchilizer = load_clamp3_model(
        "/home/loubb/work/clamp3/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    )
    mid = mido.MidiFile(midi_file_path)
    embedding = get_midi_embedding(
        mid,
        model=model,
        patchilizer=patchilizer,
        get_global=True,
    )
    print(embedding)
    print(embedding.shape)
