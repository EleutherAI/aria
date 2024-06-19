# gpt-aria

[Discord](https://discord.com/invite/zBGx3azzUn)

A repository containing resources for pre-training, fine-tuning, and evaluating musical (MIDI) transformer models.

***Note that this project is under active development***

## Description

The main goal of the gpt-aria project is to create a suite of powerful pre-trained generative (symbolic) music models. We want to investigate how modern training (pre-training & fine-tuning) techniques can be used to improve the quality/usefulness of such models. Alongside this we are building various data (MIDI) preprocessing tools, allowing **you** to easily fine-tune our models on your own data.

If you are new to symbolic music models, a good place to start are the following projects/blogposts by Google Magenta and OpenAI:

- [Music Transformer](https://magenta.tensorflow.org/music-transformer)
- [MuseNet](https://openai.com/research/musenet)

 Long story short: Transformer + MIDI + GPUs = 🎵 x ∞

## Installation

Make sure you are using Python 3.10+. Note that I haven't explicitly developed this project for anything other than Linux. If you are using Windows, things might not work properly. In this case I suggest installing using WSL.

```
git clone https://github.com/eleutherai/aria
cd aria
pip install -e .
```

## Inference

You can find preliminary checkpoints at the following locations 

Finetuned piano-only checkpoints (improved robustness):

```
large - https://storage.googleapis.com/aria-checkpoints/large-abs-inst.safetensors
```

Pretrained checkpoints:

```
large - https://storage.googleapis.com/aria-checkpoints/large-abs-pt.bin
medium - https://storage.googleapis.com/aria-checkpoints/medium-abs-pt.bin
small - https://storage.googleapis.com/aria-checkpoints/small-abs-pt.bin
```

You can then sample using the cli:

```
aria sample \
    -m large \
    -c <path-to-checkpoint> \
    -p <path-to-midifile> \
    -var <num-variations-to-generate> \
    -trunc <seconds-in-to-truncate-prompt> \
    -l <number-of-tokens-to-generate> \
    -temp 0.95 \
    -e
```

You can use `aria sample -h` to see a full list of options. If you wish to sample from a pretrained checkpoint, please use the `-pt` flag.


