import torch
from torch.utils.data import Dataset
import pandas as pd

class SongDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab, max_seq_len: int = 128):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.input_seqs, self.target_seqs, self.padding_masks = self._prepare_samples(df)

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        return (
            self.input_seqs[idx],
            self.target_seqs[idx],
            self.padding_masks[idx]
        )

    def _prepare_samples(self, df):
        input_seqs = []
        target_seqs = []
        padding_masks = []

        for _, row in df.iterrows():
            content = row['content']
            samples = self._split_Song(content)

            for sample in samples:
                sample_input_seqs, sample_target_seqs, sample_padding_masks = self._process_segment(sample)

                input_seqs += sample_input_seqs
                target_seqs += sample_target_seqs
                padding_masks += sample_padding_masks

        return (
            torch.tensor(input_seqs, dtype=torch.long),
            torch.tensor(target_seqs, dtype=torch.long),
            torch.tensor(padding_masks, dtype=torch.float)
        )

    def _split_Song(self, content):
        samples = []
        
        song_parts = content.split('\r\n\r\n')
        
        for song_part in song_parts:
            song_in_lines = song_part.split('\r\n')
            samples.append(song_in_lines)
            
        return samples

    def _process_segment(self, sample):
        input_seqs = []
        target_seqs = []
        padding_masks = []

        input_text = '<sos> ' + ' <eol> '.join(sample) + ' <eol> <eos>'
        input_ids = self.vocab._tokenizer(input_text)

        for idx in range(1, len(input_ids)):
            input_text = ' '.join(input_ids[:idx])
            input_seq = self.vocab.vectorize(input_text, self.max_seq_len)

            target_text = ' '.join(input_ids[1:idx+1])
            target_seq = self.vocab.vectorize(target_text, self.max_seq_len)

            padding_mask = [1 if token != self.vocab.pad_token else 0 for token in input_seq]

            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
            padding_masks.append(padding_mask)

        return input_seqs, target_seqs, padding_masks
