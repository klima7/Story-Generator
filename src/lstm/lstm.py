import re

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from dataset import TextTrainDataset
from utils import tokenize, pad


class LstmTextGenerator(LightningModule):
    
    VALIDATION_PROMPTS = [
        'Pewnego dnia czerwony kapturek szedł przez las z koszyczkiem jedzenia do swojej babci, która mieszkała w lesie. Śledził go jednak zły wilk, który chciał zjeść dziewczynkę.',
    ]
    
    VALIDATION_TEMPERATURES = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    def __init__(
        self,
        
        train_dataset_path,
        vocabulary_size,
        pad_token_id,
        
        # architecture
        embedding_dim=100,
        lstm_layers=1,
        lstm_dropout=0,
        lstm_hidden_size=100,
        dropout=0.2,
        bidirectional=False,
        
        # training process
        lr=0.001,
        batch_size=64,
        seq_length=10, 
        padding=(3, 50),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.embed = nn.Embedding(
            num_embeddings=self.hparams.vocabulary_size,
            embedding_dim=self.hparams.embedding_dim,
            padding_idx=self.hparams.pad_token_id
        )
        
        self.lstm = nn.LSTM(
            input_size=self.hparams.embedding_dim,
            hidden_size=self.hparams.lstm_hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.lstm_dropout,
            bidirectional=self.hparams.bidirectional
        )
        
        self.dropout = nn.Dropout(self.hparams.dropout)
        
        self.fc = nn.Linear((2 if self.hparams.bidirectional else 1)*self.hparams.lstm_hidden_size, self.hparams.vocabulary_size)
        
    def forward(self, x):
        out = self.embed(x)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
        
    def training_step(self, batch, batch_no):
        texts, targets = batch
        predicted = self.forward(texts)
        loss = F.cross_entropy(predicted, targets)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    # def train_dataloader(self):
    #     dataset = TextTrainDataset(
    #         dataset_path=self.hparams.train_dataset_path,
    #         vocab=self.vocab,
    #         seq_length=self.hparams.seq_length,
    #         padding=self.hparams.padding
    #     )
        
    #     return DataLoader(
    #         dataset=dataset,
    #         batch_size=self.hparams.batch_size,
    #         shuffle=True,
    #         num_workers=0
    #     )
        
    def generate(self, prompt, length=50, temperature=0.5):
        generated = prompt[:]
        prompt = pad(prompt, self.hparams.seq_length, self.hparams.pad_token_id)
        
        self.eval()
        
        with torch.no_grad():
            for _ in range(length):
                input_tensor = torch.unsqueeze(torch.tensor(prompt, device=self.device), dim=0)
                logits = F.softmax(self(input_tensor), dim=1)[0]
                word_idx = self.__sample_word_idx(logits, temperature)
                prompt = prompt[1:] + [word_idx]
                generated.append(word_idx)
            
        self.train()
        return generated
    
    @staticmethod
    def __sample_word_idx(logits, temperature=0.5):
        scaled_logits = torch.log(logits) / temperature
        adjusted_probs = F.softmax(scaled_logits, dim=-1)
        next_word_index = torch.multinomial(adjusted_probs, num_samples=1).item()
        return next_word_index
