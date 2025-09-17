# Aria

This repository contains training, inference, and evaluation code for the paper [*Scaling Self-Supervised Representation Learning for Symbolic Piano Performance (ISMIR 2025)*](https://arxiv.org/abs/2506.23869), as well as implementations of our real-time piano continuation demo. *Aria* is a pretrained autoregressive generative model for symbolic music, based on the LLaMA 3.2 (1B) architecture, which was trained on ~60k hours of MIDI transcriptions of expressive solo-piano recordings. Alongside the base model, we are releasing a checkpoint finetuned to improve generative quality, as well as a checkpoint finetuned to produce general-purpose piano MIDI embeddings using a SimCSE-style contrastive training objective.

📖 Read our [paper](https://arxiv.org/abs/2506.23869)  
🤗 Access our models via the [HuggingFace page](https://huggingface.co/loubb/aria-medium-base)  
📊 Get access to our training dataset [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) and train your own models

## Installation 

Installation requires Python 3.11+. To install the package and all dependencies with pip:

```bash
git clone https://github.com/EleutherAI/aria 
cd aria
pip install -e ".[all]"
```

## Quickstart

Download model weights from the official HuggingFace page for our pretrained model, as well as checkpoints finetuned for piano-continuation and generating MIDI-embeddings: 

- `aria-medium-base` ([huggingface](https://huggingface.co/loubb/aria-medium-base), [direct-download](https://huggingface.co/loubb/aria-medium-base/resolve/main/model.safetensors?download=true))
- `aria-medium-gen`([huggingface](https://huggingface.co/loubb/aria-medium-gen), [direct-download](https://huggingface.co/loubb/aria-medium-gen/resolve/main/model.safetensors?download=true)) 
- `aria-medium-embedding`([huggingface](https://huggingface.co/loubb/aria-medium-embedding), [direct-download](https://huggingface.co/loubb/aria-medium-embedding/resolve/main/model.safetensors?download=true)) 

### Inference (Prompt Continuation)

We provide optimized model implementations for PyTorch (CUDA) and MLX (Apple Silicon). You can generate continuations of a MIDI file using the CLI, e.g., using CUDA (Linux):

```bash
aria generate \
    --backend torch_cuda \
    --checkpoint_path <path-to-model-weights> \
    --prompt_midi_path <path-to-midi-file-to-continue> \
    --prompt_duration <length-in-seconds-for-prompt> \
    --variations <number-of-variations-to-generate> \
    --temp 0.98 \
    --min_p 0.035 \
    --length 2048 \
    --save_dir <dir-to-save-results>
```

Since the model has not been post-trained with instruction tuning or RLHF (similar to pre-instruct GPT models), it is very sensitive to input quality and performs best when prompted with well-played music. To get prompt MIDI files, see the `example-prompts/` directory, explore the [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) dataset, or transcribe your own files using our [piano-transcription model](https://github.com/EleutherAI/aria-amt). For a full list of sampling options: `aria generate -h`. If you wish to do inference on the CPU, please see the platform-agnostic implementation on our HuggingFace page [link].

### Intended Use and Limitations

Aria performs best when **continuing existing piano MIDI files** rather than generating music from scratch. While multi-track tokenization and generation are supported, the model was trained primarily on **single-track expressive piano performances**, and we recommend using single-track inputs for optimal results.

Due to the high representation of popular classical works (e.g., Chopin) in the training data and the difficulty of complete deduplication, the model may **memorize or closely reproduce** such pieces. For more original outputs, we suggest prompting Aria with **lesser-known works or your own compositions**.

### Inference (MIDI embeddings)

You can generate embeddings from MIDI files using the `aria.embeddings` module. This is primarily exposed with the `get_global_embedding_from_midi` function, for example:

```python
from aria.embeddings import get_global_embedding_from_midi
from aria.model import TransformerEMB, ModelConfig
from aria.config import load_model_config
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

# Load model
model_config = ModelConfig(**load_model_config(name="medium-emb"))
model_config.set_vocab_size(AbsTokenizer().vocab_size)
model = TransformerEMB(model_config)
state_dict = load_file(filename=CHECKPOINT_PATH)
model.load_state_dict(state_dict=state_dict, strict=True)

# Generate embedding
embedding = get_global_embedding_from_midi(
    model=model,
    midi_path=MIDI_PATH,
    device="cpu",
)
```

Our embedding model was trained to capture composition-level and performance-level attributes, and therefore might not be appropriate for every use case.

## Real-time demo

In `demo/` we provide an MLX (Apple Silicon) implementation of the real-time interactive piano-continuation demo showcased in our release blog post. In order to use the demo, you must download the demo-specific model checkpoint which enhances the model to additionally control the sustain pedal ([direct-download](https://huggingface.co/loubb/aria-medium-base/resolve/main/model-demo.safetensors?download=true)).

For our demonstration, we used an acoustic Yamaha Disklavier piano with simultaneous MIDI input and output ports connected via a standard MIDI interface. We disabled the built-in Disklavier playback mode, instead manually calibrating key-velocity latency to enhance responsiveness. You may recreate this in your own environment with our acoustic calibration settings, using the following script:

❗**NOTE**: It is vital that you use the `latency=off`/`realtime` Disklavier playback setting when using the provided configuration for `--hardware`.

```bash
python ./demo/demo_mlx.py \
    --checkpoint <checkpoint-path> \
    --midi_in <midi-in-port>  \
    --midi_out <midi-out-port> \
    --hardware ./demo/hardware/c4dm-disklavier.json \
    --midi_control_signal 67 \
    --midi_reset_control_signal 66 \
    --temp 0.9 \
    --min_p 0.03
```

A MIDI input device is not strictly required to play around with the demo: By using the `--midi_path` and `--midi_through` arguments you can mock real-time input by playing from a MIDI file. All that is required are MIDI drivers (e.g., CoreMIDI) and a virtual software instrument (e.g., Fluidsynth, Pianoteq) to render the output. In this mode, you can initiate the model takeover by pressing the enter key.

```bash
MIDI_PATH="example-prompts/pokey_jazz.mid"

python demo/demo_mlx.py \
    --checkpoint <checkpoint-path> \
    --midi_path ${MIDI_PATH} \
    --midi_through <midi-playback-port> \
    --midi_out <midi-playback-port> \
    --temp 0.9 \
    --min_p 0.03
```

❗**NOTE**: Responsiveness of the real-time demo is dependent on your system configuration, specifically GPU memory bandwidth. 

## Evaluation

We provide the specific files/splits we used for Aria-MIDI derived linear-probe and classification evaluations. These can be downloaded from HuggingFace ([direct-download](https://huggingface.co/loubb/aria-medium-base/resolve/main/eval-splits.tar.gz?download=true)). Class labels are provided in `metadata.json` with the schema:

```json
{
  "<category>": {
    "<split-name>": {
      "<relative/path/to/file.mid>": "<metadata_value_for_that_category>",
      ...
    },
    ...
  },
  ...
}
```

## License and Attribution

The Aria project has been kindly supported by EleutherAI, Stability AI, as well as by a compute grant from the Ministry of Science and ICT of Korea. Our models and MIDI tooling are released under the Apache-2.0 license. If you use the models or tooling for follow-up work, please cite the paper in which they were introduced:

```bibtex
@inproceedings{bradshawscaling,
  title={Scaling Self-Supervised Representation Learning for Symbolic Piano Performance},
  author={Bradshaw, Louis and Fan, Honglu and Spangher, Alexander and Biderman, Stella and Colton, Simon},
  booktitle={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/2506.23869}
}
```