import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.vocabulary import SongVocabulary
from data.dataset import SongDataset
from models.transformer import TransformerModel
from modules.trainer import SongTrainer
from modules.generator import SongGenerator
from config.config import Config
def main():
    df = pd.read_csv('song_scraper\song_final.csv')
    df = df.head(100)
    df = df.dropna(subset=['content'])
    def text_normalize(text):
        text = text.strip()
        return text
    df['content'] = df['content'].apply(lambda x: text_normalize(x))
    vocab = SongVocabulary(df)
    dataset = SongDataset(df, vocab)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = TransformerModel(vocab_size=vocab.__len__(), embedding_dims=Config.EMBEDDING_DIMS, n_heads=Config.N_HEADS, n_layers=Config.N_LAYERS, hidden_dims=Config.HIDDEN_DIMS, dropout=Config.DROPOUT)
    trainer = SongTrainer(model, vocab)
    trainer.train(train_loader)

    torch.save(model.state_dict(), 'weights/transformer_song_SGD100.pth')

if __name__ == '__main__':
    main()
    