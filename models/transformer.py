import torch
import torch.nn as nn
import math
from config.config import Config
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dims: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) * (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(max_len, 1, embedding_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dims: int = Config.EMBEDDING_DIMS,
        n_heads: int = Config.N_HEADS,
        hidden_dims: int = Config.HIDDEN_DIMS,
        n_layers: int = Config.N_LAYERS,
        dropout: float = Config.DROPOUT
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.embedding_dims = embedding_dims
        
        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dims, n_heads, hidden_dims, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear = nn.Linear(embedding_dims, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask = None, padding_mask = None):
        src = self.embedding(src) * math.sqrt(self.embedding_dims)
        src = self.pos_encoder(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(src.device)
        output = self.transformer_encoder(src, mask=src_mask, 
                                       src_key_padding_mask=padding_mask)
        output = self.linear(output)
        return output