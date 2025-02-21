import torch
import torch.nn.functional as F
from typing import List
from config.config import Config
from data.vocabulary import SongVocabulary

class SongGenerator:
    def __init__(self, model, vocab: SongVocabulary, 
                 device: str = Config.DEVICE):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
    
    def sample_with_temperature(self, logits, temperature=Config.TEMPERATURE):
        if temperature != 1.0:
            logits = logits / temperature

        probabilities = F.softmax(logits, dim=-1)
        sampled_index = torch.multinomial(probabilities, 1).item()
        return sampled_index

    def generate(self, prompt, max_length: int = Config.MAX_GENERATION_LENGTH, 
                temperature: float = Config.TEMPERATURE):
        self.model.eval()
        
        input_ids = self.vocab.encode(f'<sos> {prompt}') 
        generated_ids = input_ids.copy()
        eos_token_id = self.vocab.get_stoi()['<eos>']
        
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([generated_ids]).to(self.device)
                outputs = self.model(input_tensor)
                
                last_token_logits = outputs[0, -1, :] 
                next_token_id = self.sample_with_temperature(last_token_logits, temperature)
                generated_ids.append(next_token_id)
                
                if next_token_id == eos_token_id:
                    break
        
        generated_text = self.vocab.decode(generated_ids)
        generated_text = ' '.join(generated_text)
        generated_text = generated_text.replace('<sos>', '').strip()
        #generated_text = generated_text.split('<eol>')
        return generated_text
    
    def generate_song(self, prompt, 
                      num_segments: int = Config.DEFAULT_NUM_SEGMENTS,
                      segment_length: int = Config.MAX_GENERATION_LENGTH):
        song_segments = []
        current_prompt = prompt
        segment = self.generate(current_prompt, max_length=segment_length)
        
        # for _ in range(num_segments):
        #     segment = self.generate(current_prompt, max_length=segment_length)
        #     song_segments.append(segment)
        #     current_prompt = ' '.join(segment.split()[-10:])
        
        return segment.split('<eol>')
