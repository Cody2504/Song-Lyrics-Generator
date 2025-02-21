from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

class SongVocabulary:
    def __init__(self, df):
        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>', '<eol>']
        self.vocab = self._build_vocab(df)
        self.vocab.set_default_index(self.vocab['<unk>'])
        self.pad_token = self.vocab['<pad>']
        self.max_seq_len = 128

    def _tokenizer(self, text):
        return text.split()

    def _yield_tokens(self, df):

        for _, row in df.iterrows():
            yield self._tokenizer(row['content'])

    def _build_vocab(self, df):
        return build_vocab_from_iterator(
            self._yield_tokens(df),
            specials=self.special_tokens
        )

    def pad_and_truncate(self, input_ids, max_seq_len):
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
        else:
            input_ids += [self.pad_token] * (max_seq_len - len(input_ids))
        return input_ids

    def vectorize(self, text, max_seq_len=128):
        input_ids = [self.vocab[token] for token in self._tokenizer(text)]
        return self.pad_and_truncate(input_ids, max_seq_len)

    def decode(self, input_ids):
        return [self.vocab.get_itos()[token_id] for token_id in input_ids]

    def encode(self, text):
        input_ids = [self.vocab[token] for token in self._tokenizer(text)]
        return input_ids

    def __len__(self):
        return len(self.vocab)

    def get_stoi(self):
        return self.vocab.get_stoi()