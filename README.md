# gpt-aria

[Roadmap](https://github.com/EleutherAI/aria/blob/main/ROADMAP.md) / [HowTo](https://github.com/EleutherAI/aria/blob/main/HOWTO.md) / [Discord](https://discord.com/invite/zBGx3azzUn)

A repository containing resources for pre-training, fine-tuning, and evaluating musical (MIDI) transformer models.

***Note that this project is under active development. To get involved see the Neuro-Symbolic Music Models channel on the [EleutherAI Discord](https://discord.com/invite/zBGx3azzUn).***

## Description

The main goal of the gpt-aria project is to create a suite of powerful pre-trained generative (symbolic) music models. We want to investigate how modern training (pre-training & fine-tuning) techniques can be used to improve the quality/usefulness of such models. Alongside this we are building various data (MIDI) preprocessing tools, allowing **you** to easily fine-tune our models on your own data.

If you are new to symbolic music models, a good place to start are the following projects/blogposts by Google Magenta and OpenAI:

- [Music Transformer](https://magenta.tensorflow.org/music-transformer)
- [MuseNet](https://openai.com/research/musenet)

Some early (experimental) samples: [Mozart](https://twitter.com/loubbrad/status/1685638807100530693?s=20), [Bach](https://twitter.com/loubbrad/status/1685650221353635840?s=20), [Debussy](https://twitter.com/loubbrad/status/1686332713756708864?s=20). Long story short: Transformer + MIDI + GPUs = 🎵 x ∞

## Installation

Make sure you are using Python 3.10+. Note that I haven't explicitly developed this project for anything other than Linux. If you are using Windows, things might not work properly. In this case I suggest installing using WSL.

```
git clone https://github.com/eleutherai/aria
cd aria
pip install -e .
```
