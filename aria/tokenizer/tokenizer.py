from aria.data.utils import MidiDict


class Tokenizer:
    def __init__(self):
        pass

    # Abstract
    def encode(self, src):
        pass

    # Abstract
    def decode(self, src):
        pass


class TokenizerLazy(Tokenizer):
    def __init__(self):
        super().__init__()

    def encode(self, src):
        return src

    def decode(self, src):
        pass


tokenizer = TokenizerLazy()
print("_")
