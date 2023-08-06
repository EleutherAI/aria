# HOW TO

Here, we will provide some code snippets demonstrating the core functionality and hopefully answering some questions. This document is a work in progress and will naturally evolve over time.

## QUICK OVERVIEW

The `run.py` script acts as the entry point for all the basic functionality of Aria.

### DATA

The `data` command can be used to build datasets (MidiDict & Tokenized). You must provide the name of the tokenizer you wish to use (currently, the only option is 'lazy'). For more information about the data formats and types of datasets, refer to the DATA section of this document. To view the list of arguments, run the following command:

```
python run.py data -h
```

### SAMPLING

The `sample` command provides an easy way to perform autoregressive sampling. To use this, you need to have pre-downloaded model weights (feel free to message me on Discord about where you can get these). This functionality is currently implemented using PyTorch Lightning. You need to provide a path to the `model.ckpt` file. The sample command works by tokenizing the prompt MIDI file, truncating it (according to -trunc), and then using the result as a prompt for top-p autoregressive sampling. To view the list of arguments, run the following command:

```
python run.py sample -h
```

### TRAINING

The `train` command is used as the entry point for pre-training models. Most of the configuration is hardcoded into `aria.training` since `train` is currently only used for development purposes. It supports checkpointing and technically could be used for fine-tuning, although I would not recommend it. Instead, I suggest loading the model via the LightningModule `aria.training.PretrainLM` and writing your own training loop. To view the list of arguments for `train`, run the following command:

```
python run.py train -h
```

Please note that you need to provide paths to train and validation TokenizedDatasets in order to train a model. For more information, refer to the DATA section of the document.

## DATA

### MIDI

Since we are attempting to model symbolic music, we need to start with RAW data that can be easily parsed into a symbolic (and sequential) form. As others have done in the past, we have chosen MIDI for this purpose. MIDI files are both ubiquitous and relatively simple to parse into a format understandable by a Transformer. To learn more about MIDI, asking ChatGPT (or GPT-Neox) is genuinely a good idea. The simple explanation is that MIDI encodes a piece of music as a series of note-on and note-off messages. Each message contains information, including the note pitch, note velocity (volume), and the number of *ticks* to wait before processing the next message. These messages are sent over a specific MIDI *channel*, which itself contains information like the type of instrument.

To simplify the process of creating different MIDI tokenizers, Aria provides an intermediate data format `aria.data.MidiDict`. This class performs basic preprocessing and offers a more straightforward (compared to `mido`) way to represent MIDI files in Python. MIDI Messages are divided into four main types: note_msg, instrument_msg, pedal_msg, and tempo_msg. Further details about how MidiDict works are available in its docstring.

The class `aria.data.datasets.MidiDataset` is utilized to organize collections of MidiDict objects. The configuration settings for how MidiDataset operates can be found in the `data` property of the config.json file (in the config directory).

```python
# Creating a MidiDataset directly from a directory './my_midi_files/' and saving
# it to a file
dataset = MidiDataset.build(
    dir='./my_midi_files/',
    recur=True,
)
dataset.save('dataset.json')

# Loading a MidiDataset from a file
dataset = MidiDataset.load('dataset.json')

# Creating a MidiDataset directly to a file. This is useful when dealing with
# large datasets, as it doesn't require all data to be stored in memory simultaneously.
MidiDataset.build_to_file(
    dir='./my_midi_files/',
    save_path='dataset.json',
    recur=True,
)
```

### TOKENIZED DATA

Tokenizers (refer to `aria.tokenizer`) convert MidiDict objects into sequences of tokens, which can be processed by a Transformer. Currently, only one tokenizer is implemented (`TokenizerLazy`). However, in theory, all other tokenizers would inherit from the `aria.tokenizer.Tokenizer` class. The main functionality is contained in the self-explanatory `tokenize_midi_dict` and `detokenize_midi_dict` methods. In the case of `TokenizerLazy`, there are also several functions that export data augmentation functions designed to be applied to the outputs of `tokenize_midi_dict`. In the past, people have used a variety of methods for tokenizing MIDI files. [MidiTok](https://github.com/Natooz/MidiTok) is an excellent resource if you want to explore this further. The tokenization scheme that `TokenizerLazy` uses is primarily inspired by the [MuseNet](https://openai.com/research/musenet) tokenizer. It aims to balance simplicity and expressiveness while being transformer-friendly and properly supporting multi-track MIDI files.

```python
# Loading Debussy's arabesque into a MidiDict and then tokenizing it
MIDI_PATH = "./tests/test_data/arabesque.mid"
midi = mido.MidiFile(MIDI_PATH)
midi_dict = MidiDict.from_midi(midi)
tokenizer = aria.tokenizer.TokenizerLazy()
tokenized_seq = tokenizer.tokenize_midi_dict(midi_dict)
print(tokenized_seq)

# Using a data-augmentation function to randomly augment the pitch of the
# sequence and then save the result as a new MIDI file
pitch_aug_fn = tokenizer.export_pitch_aug(aug_range=5)
augmented_tokenized_seq = pitch_aug_fn(tokenized_seq)
print(augmented_tokenized_seq)
midi_dict = tokenizer.detokenize_midi_dict(augmented_tokenized_seq)
midi = midi_dict.to_midi()
midi.save('arabesque')
```

As we need rapid access to tokenized sequences during training, we use the class `aria.data.datasets.TokenizedDataset` to organize them into a dataset. This class implements the `torch.utils.data.Dataset` interface and can therefore be used with the PyTorch DataLoader class. There are a few important things to note about how TokenizedDataset works:

- To support incredibly large datasets of sequences, TokenizedDataset doesn't store its entries in memory all at once. Instead, it creates a memory-mapped ([mmap](https://docs.python.org/3/library/mmap.html)) jsonl file and loads entries into memory dynamically. To instantiate a TokenizedDataset, you must provide a path to the appropriate dataset jsonl file. To handle building large `TokenizedDatasets` from large `MidiDatasets`, the `TokenizedDataset.build` method accepts either a MidiDataset or a MidiDataset jsonl save file. In the latter case, the entire `MidiDataset` is not loaded into memory at once. Note also that the build method will slice, truncate, and pad the tokenized sequences appropriately.

- TokenizedDatasets are used to load tensor representations of tokenized sequences for training purposes. Consequently, you need to supply a Tokenizer object to enable the dataset's access to its `encode` and `decode` methods for this transformation. The Tokenizer must be configured with `return_tensors=True` it is being used with a TokenizedDataset. When you index a TokenizedDataset (by invoking `__getitem__`), it yields a pair of tensors: src, tgt.

- The `set_transform` function accepts data augmentation functions that are automatically applied when calling `__getitem__`. This is how we implement data augmentation during training.
 
```python
# Creating a dataset of tokenized sequences from a directory of MIDI files
mididict_dataset = datasets.MidiDataset.build(
    dir="tests/test_data",
    recur=True,
)
tokenizer = tokenizer.TokenizerLazy(max_seq_len=512, return_tensors=True)
tokenized_dataset = datasets.TokenizedDataset.build(
    tokenizer=tokenizer,
    save_path="tokenized_dataset.jsonl",
    midi_dataset=mididict_dataset,
)

# Creating a dataset of tokenized sequences from a directory of MIDI files,
# without loading it into memory all at once
datasets.MidiDataset.build_to_file(
    dir="tests/test_data",
    save_path="mididict_dataset.jsonl",
    recur=True,
)
tokenizer = tokenizer.TokenizerLazy(max_seq_len=512, return_tensors=True)
tokenized_dataset = datasets.TokenizedDataset.build(
    tokenizer=tokenizer,
    save_path="tokenized_dataset.jsonl",
    midi_dataset_path="mididict_dataset.jsonl",
)

# A short demonstration of how you might use TokenizerLazy during training
tokenizer = tokenizer.TokenizerLazy(max_seq_len=512, return_tensors=True)
tokenized_dataset = datasets.TokenizedDataset.build(
    tokenizer=tokenizer,
    save_path="tokenized_dataset.jsonl",
    midi_dataset_path="mididict_dataset.jsonl",
)
tokenized_dataset.set_transform(
    [
        tokenizer.export_pitch_aug,
        tokenizer.export_velocity_aug,
    ]
)
dataloader = torch.utils.data.DataLoader(tokenized_dataset)

for src, tgt in dataloader:
    # Truncate for simplicity
    src, tgt = src[:50], tgt[:50]
    print(f"source tensor: {src}")
    print(f"source decoded: {tokenizer.decode(src)}")
    print(f"target tensor: {tgt}")
    print(f"target decoded: {tokenizer.decode(tgt)})
```