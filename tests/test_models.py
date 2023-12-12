import logging
import unittest

from aria.model import ModelConfig, TransformerLM
from aria.config import load_model_config
from aria.model.model import YaRNConfig
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
        max_len = model.model.encode_layers[
            0
        ].rotary_emb.max_position_embeddings
        org_max_len = model.model.encode_layers[
            0
        ].rotary_emb.original_max_position_embeddings
        assert max_len == org_max_len
        assert model.model.encode_layers[0].rotary_emb.mscale_coeff == 0.07
        assert model.model.encode_layers[0].rotary_emb.beta_fast == 32.0
        assert model.model.encode_layers[0].rotary_emb.beta_slow == 1.0
        assert model.model.encode_layers[0].rotary_emb.base == 10000.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
