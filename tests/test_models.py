import logging
import torch
import unittest

from aria.model import ModelConfig, TransformerLM
from aria.config import load_model_config
from aria.model.model import YaRNConfig
from aria.model.utils import apply_rotary_pos_emb
from .reference_implementations import apply_rotary_pos_emb_reference
from aria.tokenizer import TokenizerLazy


class TestModel(unittest.TestCase):
    def test_yarn_config(self):
        tokenizer = TokenizerLazy(return_tensors=True)
        model_config = ModelConfig(**load_model_config("test"))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = TransformerLM(model_config)
        assert model.model.model_config.yarn_config is None
        model_config = ModelConfig(**load_model_config("test_yarn"))
        model_config.set_vocab_size(tokenizer.vocab_size)
        model = TransformerLM(model_config)
        assert isinstance(model.model.model_config.yarn_config, YaRNConfig)
        assert model.model.encode_layers[0].rotary_emb.mscale_coeff == 0.07
        assert model.model.encode_layers[0].rotary_emb.beta_fast == 32.0
        assert model.model.encode_layers[0].rotary_emb.beta_slow == 1.0
        assert model.model.encode_layers[0].rotary_emb.base == 10000.0

    def test_rope_util_fns(self):
        q = torch.rand(4, 8, 12, 64)
        inv_freq = 1 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        t = torch.arange(8, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        q_ref = apply_rotary_pos_emb_reference(q.clone(), cos, sin)
        q = apply_rotary_pos_emb(q.clone(), cos, sin)
        assert torch.allclose(q, q_ref)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
