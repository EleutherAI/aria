import torch
import tempfile
import shlex
import os
import torchaudio

import torchaudio.transforms as T
import torch.nn.functional as F
import torch.nn as nn

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from transformers import Wav2Vec2FeatureExtractor, AutoModel


def seq_to_audio_path(
    seq: list, tokenizer: AbsTokenizer, pianoteq_exec_path: str
):
    mid_temp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid_path = mid_temp.name
    mid_temp.close()

    mid = tokenizer.detokenize(seq)
    mid.to_midi().save(mid_path)

    # Step 3: Create a temporary WAV file for output
    audio_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = audio_temp.name
    audio_temp.close()  # Close so CLI can write to it

    # Step 4: Run CLI command to generate audio using Pianoteq
    # EXEC_PATH = "/home/loubb/pianoteq/x86-64bit/Pianoteq 8 STAGE"
    preset = "NY Steinway D Classical Recording"

    pianoteq_cmd = f"{shlex.quote(pianoteq_exec_path)} --preset {shlex.quote(preset)} --rate 24000 --midi {mid_path} --wav {audio_path}"
    os.system(pianoteq_cmd)

    os.remove(mid_path)

    return audio_path


def compute_audio_embedding(
    audio_path: str, model: nn.Module, processor, delete_audio: bool = False
) -> torch.Tensor:
    """
    Loads the MERT-v1-330M model and processor, reads an mp3 file,
    segments the audio into 5-second chunks, computes a segment embedding by averaging
    over the time dimension (for each layer) and across layers, and then aggregates
    the segment embeddings using average pooling to produce a final embedding.

    Parameters:
      file_path (str): Path to the mp3 audio file.

    Returns:
      torch.Tensor: The final audio embedding.
    """
    # Load the mp3 file and convert to mono if necessary (waveform shape: [channels, time])
    waveform, sr = torchaudio.load(audio_path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed (target_sr for MERT-v1-330M is typically 24000 Hz)
    target_sr = processor.sampling_rate
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Remove channel dimension to get [n_samples]
    waveform = waveform.squeeze(0)

    # Define the segment length for 5 seconds
    segment_length = target_sr * 5
    total_samples = waveform.size(0)
    segments = []

    # Split the waveform into segments; pad the final segment if needed
    for start in range(0, total_samples, segment_length):
        segment = waveform[start : start + segment_length]
        if segment.size(0) < segment_length:
            padding = segment_length - segment.size(0)
            segment = F.pad(segment, (0, padding))
        segments.append(segment.numpy())

    # Process all segments in one batch. The processor accepts a list of numpy arrays.
    inputs = processor(segments, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # Forward pass through the model in batch mode
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of tensors (one per layer) of shape:
    # [batch_size, time_steps, feature_dim] for each layer.
    # Stack them to get shape: [num_layers, batch_size, time_steps, feature_dim]
    hidden_states = torch.stack(outputs.hidden_states)

    # Average over the time dimension for each segment in each layer:
    # result shape: [num_layers, batch_size, feature_dim]
    layer_time_avg = hidden_states.mean(dim=2)

    # Average over layers to obtain one embedding per segment:
    # result shape: [batch_size, feature_dim]
    segment_embeddings = layer_time_avg.mean(dim=0)

    # Finally, average the segment embeddings to get a final representation:
    # shape: [feature_dim]
    final_embedding = segment_embeddings.mean(dim=0)

    if delete_audio is True:
        os.remove(audio_path)

    return final_embedding


def load_mert_model():

    return AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    ).cuda(), Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )


def main():
    model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    ).cuda()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )

    tokenizer = AbsTokenizer()
    mid_dict = MidiDict.from_midi("/home/loubb/Dropbox/shared/test.mid")
    seq = tokenizer.tokenize(mid_dict)

    audio_path = seq_to_audio_path(seq, tokenizer)
    emb = compute_audio_embedding(
        audio_path=audio_path,
        model=model,
        processor=processor,
        delete_audio=True,
    )
    print(emb.shape)


if __name__ == "__main__":
    main()
