## Roadmap

As it stands, the basic functionality of the repository is implemented and tested. Here is a high-level overview of what remains:

- Refactor the training framework to be more robust.
- Finish implementing missing core features (preprocessing tests, data augmentation).
- Fix remaining data encoding/decoding bugs and undesired behaviour.
- Upgrade model architecture and parallelize its implementation.
- Finalize the experiment design.

### Data

* [ ] **Add chord mix-up data-augmentation function** 

  This (tokenized) data-augmentation function should randomly shuffle the order of notes that occur concurrently. For instance, `("piano", 60, 50), ("dur", 10), ("piano", 64, 50), ("dur", 20)` could be augmented to `("piano", 64, 50), ("dur", 20), ("piano", 60, 50), ("dur", 10)` as there is no wait token between the notes. See `aria.tokenizer.TokenizerLazy.export_pitch_aug()` for an example of how to implement data augmentation functions.
* [x] **~~Add speed data-augmentation function~~**

  This data-augmentation function should change the speed of a tokenized sequence by some (float) factor. The main issue I foresee is accounting for the way that wait tokens are currently implemented. Depending on the `config.json`, the lazy tokenizer has a max wait token `("wait", t_max)`. Any 'wait' event longer than `t_max` is represented as a sequence of tokens. For instance, a wait of 2*t_max + 10ms would be `("wait", t_max), ("wait", t_max), ("wait", 10)`.
* [x] **~~Fix encode/decode disparity bug~~**

  There is a bug in MidiDict/TokenizerLazy that occasionally results in repeated notes being encoded incorrectly. This needs to be fixed before the main training can start.
* [ ] **Add checksum pre-processing test**

  Implement a pre-processing test which detects duplicate MIDI files by doing a sort of checksum on the MIDI file. I'm not quite sure how to implement this but it should not be too hard.
* [ ] **Add further pre-processing tests**

  Add further MidiDict pre-processing tests to improve dataset quality. Some ideas are checking for the frequency of note messages (an average of > 15 p/s or < 2 p/s is a bad sign). I'm open to any suggestions for MidiDict preprocessing tests. Properly cleaning pre-training datasets has a huge effect on model quality and robustness.
* [ ] **Add meta-token prefix support for LazyTokenizer**

  Investigate the possibility of adding meta-tokens to the prefix in LazyTokenizer. Some examples could be genre, composer, or data source tags. This might require a rewrite of how sequence prefixes are handled.
* [ ] **Add 'ending soon' token to lazy tokenizer**

  This token should appear ~100 notes before the end token '<E>' appears. The idea is that we can insert the 'ending soon' token intentionally to force the model to end the track naturally before it runs out of context.
* [ ] **Finalize the pre-training MIDI dataset**

  We could investigate the possibility of adding Classical Archives' [MIDI collection](https://www.classicalarchives.com/midi.html) to the pre-training data.
  
* [x] **~~Refactor max_seq_len out of the Tokenizer class~~**

  This change may have adverse effects on other parts of the codebase, so it should be approached with caution. `max_seq_len` should be saved separately in the creation configuration for `TokenizedDataset`, similar to how padding and stride length are handled.
* [ ] **Improve the striding mechanism in TokenizedDataset.build**

  Currently, striding can result in sequences starting or ending in awkward places. For example, it is possible for a sequence to begin with a duration token. Ideally, the logic should be modified to prevent this.
* [ ] **Cultivate fine-tuning datasets**

  Fine-tuning datasets don't have to be big, but they do have to be high quality. A good example for a Jazz MIDI dataset is [this](https://bushgrafts.com/midi/) one created by Doug McKenzie.

### Training/Model

* [ ] **Huggingface migration**

  Investigate the possibility of migrating the training code from using Lightning to HF (Accelerate/Trainer). I want to make sure there is reliable checkpointing and experiment tracking before official pre-training starts. The current Lightning implementation lacks in this regard. We should also investigate if it is worth migrating Tokenizer to the HF ecosystem too.

* [ ] **Add fine-tuning functionality to aria.training**

  Add a `finetune` function in `aria.training`. This should be done after the (possible) HF migration.
* [ ] **Investigate architecture improvements**

  The original architecture is inspired by LLaMA. With the release of LLaMA-2, it would be useful to consider if any of the architecture changes could be applied to Aria.
* [ ] **Investigate dropout fused self-attention kernel bug**

  In the past, I was having issues where the PyTorch 2.0 self-attention kernel would not be used with non-zero dropout. Look into this and fix the relevant issues.
* [ ] **Increase model parallelism (reimplement with DeepSpeed?)**

  There are some areas that can be made more efficient. For example, the QKV vectors can be calculated more efficiently ([here](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)). We should also investigate possibly reimplementing with DeepSpeed. The current rotary embedding code is ported from the LLaMA (1) inference code. Can this be improved?
  
* [ ] **Finish planning the size/hyperparameters of the model(s)**

  From my experience we should put the majority of the parameter budget into depth and context-length. MuseNet is a good reference here for what is realistic and possible.

### Misc

* [ ] **Research into implementing evals**

  I would like to have an evaluation metric other than val-loss and human-eval. There has been some research on this topic in the past. It would be nice if there was something like perplexity that we could use. The presence of such a metric would make evaluating fine-tuned models a lot easier.
