import torch
from models.transformer import TransformerModel
from config.config import Config
from data.vocabulary import SongVocabulary
from modules.generator import SongGenerator
import pandas as pd

df = pd.read_csv('song_scraper\song_final.csv')
df = df.head(100)
df = df.dropna(subset=['content'])
vocab = SongVocabulary(df)

model = TransformerModel(vocab_size=len(vocab), 
                         embedding_dims=Config.EMBEDDING_DIMS, 
                         n_heads=Config.N_HEADS, 
                         n_layers=Config.N_LAYERS, 
                         hidden_dims=Config.HIDDEN_DIMS, 
                         dropout=Config.DROPOUT)

model.load_state_dict(torch.load('weights/transformer_song_SGD100.pth'))
model.eval()  
song_generator = SongGenerator(model, vocab)
prompt = "Loving can hurt"
generated_song = song_generator.generate_song(prompt)
print(generated_song)
# for segment in generated_song:
#     print(f"{segment}\n")