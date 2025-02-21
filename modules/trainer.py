import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from data.vocabulary import SongVocabulary
class SongTrainer:
    def __init__(self, model: nn.Module, vocab: SongVocabulary,
                 device: str = Config.DEVICE,
                 learning_rate: float = Config.LEARNING_RATE):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=0.98)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        losses = []

        for batch in dataloader:
            input_seqs, target_seqs, padding_masks = [x.to(self.device) for x in batch]
            padding_masks = padding_masks.permute(1, 0)

            self.optimizer.zero_grad()
            outputs = self.model(input_seqs, padding_mask=padding_masks)
            outputs = outputs.permute(0, 2, 1)

            loss = self.criterion(outputs, target_seqs)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            losses.append(loss.item())

        return sum(losses) / len(losses)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train(self, train_loader, num_epochs=Config.NUM_EPOCHS):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(train_loader)
            self.scheduler.step()  # Move scheduler step here
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    def evaluate(self, val_loader: DataLoader):
        self.model.eval() 
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_seqs, target_seqs, padding_masks = [x.to(self.device) for x in batch]
                padding_masks = padding_masks.permute(1, 0)

                outputs = self.model(input_seqs, padding_mask=padding_masks)
                outputs = outputs.permute(0, 2, 1)

                loss = self.criterion(outputs, target_seqs)
                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches
        print(f'Validation Loss: {avg_val_loss:.4f}')