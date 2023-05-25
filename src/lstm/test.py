import random
import re
import pickle
import os.path

import numpy as np
import torch
import torchtext
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torchtext.vocab import build_vocab_from_iterator
from lightning.pytorch.loggers import TensorBoardLogger
from torchinfo import summary
from gensim.models.word2vec import LineSentence, Word2Vec
from tqdm import tqdm

class TextTrainDataset(IterableDataset):
    
    def __init__(self, dataset_path, pad_token_idx, seq_length=10):
        self.dataset_path = dataset_path
        self.pad_token_idx = pad_token_idx
        self.seq_length = seq_length
        
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
    def __len__(self):
        return len(self.dataset)
        
    def __iter__(self):
        for text in self.dataset:
            if len(text) < 2: continue
            start_idx = random.randint(-self.seq_length+1, len(text)-self.seq_length-1)
            cropped_text = text[max(start_idx, 0) : start_idx+self.seq_length]
            cropped_text = self.__padd(cropped_text)
            target = text[start_idx+self.seq_length]
            yield cropped_text, target
            
    def __padd(self, text):
        if len(text) < self.seq_length:
            padding = [self.pad_token_idx]*(self.seq_length-len(text))
            text = padding + text
        return text
    
class TextValidationDataset(IterableDataset):
    
    def __init__(self, text_file_path):
        self.text_file_path = text_file_path
        
    def __iter__(self):
        for text in LineSentence(self.text_file_path):
            yield ' '.join(text)
            
class LstmTextGenerator(LightningModule):
    
    def __init__(
        self,
        
        # files
        vocabulary_path,
        train_file_path,
        
        # training process
        seq_length=10, 
        batch_size=64,
        
        # architecture
        vocab_size=100_000,
        embedding_dim=100,
        lstm_layers=1,
        lstm_dropout=0,
        lstm_hidden_size=100,
        dropout=0.2,
        bidirectional=False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocabulary = torch.load(self.hparams.vocabulary_path)
        self.vocabulary.append_token('<pad>')
        
        self.embedding = nn.Embedding(
            len(self.vocabulary),
            self.hparams.embedding_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=self.hparams.lstm_hidden_size,
            batch_first=True,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.lstm_dropout,
            bidirectional=self.hparams.bidirectional
        )
        
        self.fc = nn.Linear((2 if self.hparams.bidirectional else 1)*self.hparams.lstm_hidden_size, len(self.vocabulary))
        
        self.dropout = nn.Dropout(self.hparams.dropout)
        
        self.loss = nn.CrossEntropyLoss()
        
    def generate(self, prompt, length=50, temperature=0.5):
        generated = prompt
        prompt = self.__preprocess_prompt(prompt)
        
        for _ in range(length):
            embedded_prompt = self.vocabulary(prompt)
            embedded_prompt = torch.tensor(embedded_prompt, device=self.device)
            next_word_logits = self(torch.unsqueeze(embedded_prompt, dim=0))[0]
            word = self.__get_word_from_logits(next_word_logits, temperature)
            prompt = prompt[1:] + [word]
            
            if word not in list('.!?,'):
                generated += ' '
            generated += word
        
        return generated
    
    def __get_word_from_logits(self, next_word_logits, temperature=0.5):
        scaled_logits = next_word_logits / temperature
        adjusted_probs = F.softmax(scaled_logits, dim=-1)
        next_word_index = torch.multinomial(adjusted_probs, num_samples=1).item()
        next_word = self.vocabulary.get_itos()[next_word_index]
        return next_word
        
    def forward(self, x):
        print('forward')
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
        
    def training_step(self, batch, batch_no):
        text, target = batch
        text = self.vocabulary(text)
        target = self.vocabulary[target]
        predicted = self.forward(text)
        loss = self.loss(predicted, target)
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self):
        dataset = TextTrainDataset(
            self.hparams.train_file_path,
            pad_token_idx=self.vocabulary['<pad>'],
            seq_length=self.hparams.seq_length,
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        
    def __preprocess_prompt(self, prompt):
        prompt = prompt.lower().strip()
        prompt = re.sub(r'[^a-ząćęłńóśźż.,!? ]', '', prompt)
        prompt = prompt.replace('.', ' . ').replace('!', ' ! ').replace('?', ' ? ').replace(',', ' , ')
        prompt = prompt.split()
        prompt = [word for word in prompt if word in self.vocabulary]
        padding = ['<pad>']*(max(self.hparams.seq_length-len(prompt), 0))
        prompt = padding + prompt
        return prompt
    
logger = TensorBoardLogger(
    save_dir='../..',
    name='logs'
)

trainer = Trainer(
    accelerator='cuda',
    max_epochs=-1,
    enable_progress_bar=True,
    logger = logger,
)

generator = LstmTextGenerator(
    train_file_path='../../data/binary_texts/fairytales.pickle',
    vocabulary_path='../../models/vocabulary.pth',
    seq_length=10,
    lstm_layers=3,
    lstm_dropout=0.2,
    lstm_hidden_size=100,
    dropout=0.2,
    bidirectional=True,
    batch_size=128,
)

trainer.fit(generator)