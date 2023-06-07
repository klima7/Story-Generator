import re

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule


class LstmTextGenerator(LightningModule):
    
    VALIDATION_PROMPTS = [
        'Pewnego dnia czerwony kapturek szedł przez las z koszyczkiem jedzenia do swojej babci, która mieszkała w lesie. Śledził go jednak zły wilk, który chciał zjeść dziewczynkę.',
        'Za siedmioma górami i za siedmioma rzekami było sobie królestwo, w którym mieszkała księżniczka',
        'Dawno, dawno temu żył pewien chłopiec',
    ]
    
    VALIDATION_TEMPERATURES = [0.01, 0.2, 0.3, 0.5, 0.7]
    
    
    def __init__(
        self,
        
        # files
        vocabulary_path,
        train_file_path,
        
        # architecture
        embedding_dim=100,
        lstm_layers=1,
        lstm_dropout=0,
        lstm_hidden_size=100,
        dropout=0.2,
        bidirectional=False,
        
        # training process
        batch_size=64,
        seq_length=10, 
        target_length=10,
        target_weight_decrease=1.0,
        padding_factor=20,
        padding_limit=3,
        epoch_size=10,
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
            input_size=self.hparams.embedding_dim,
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
            input_tensor = torch.unsqueeze(torch.tensor(prompt, device=self.device), dim=0)
            next_word_logits = self(input_tensor)[0]
            word_idx = self.__get_word_from_logits(next_word_logits, temperature)
            prompt = prompt[1:] + [word_idx]
            
            word = self.vocabulary.lookup_token(word_idx)
            if generated[-1] in '.!?':
                word = word.capitalize()
            if word not in list('.!?,'):
                generated += ' '
            generated += word
        
        return generated
    
    def __get_word_from_logits(self, next_word_logits, temperature=0.5):
        scaled_logits = next_word_logits / temperature
        adjusted_probs = F.softmax(scaled_logits, dim=-1)
        next_word_index = torch.multinomial(adjusted_probs, num_samples=1).item()
        return next_word_index
        
    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
        
    def training_step(self, batch, batch_no):
        texts, targets = batch
        loss = 0
        weight = 1.0
        
        for i in range(self.hparams.target_length):
            predicted = self.forward(texts)
            loss_part = self.loss(predicted, targets[:, i])
            loss += weight * loss_part
            weight *= self.hparams.target_weight_decrease
            
            words_idx = predicted.argmax(dim=-1)
            texts = torch.cat((texts[:, 1:], words_idx.unsqueeze(1)), axis=1)
        
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        tensorboard = self.logger.experiment
        
        for no, validation_prompt in enumerate(self.VALIDATION_PROMPTS):
            for temperature in self.VALIDATION_TEMPERATURES:
                text = self.generate(validation_prompt, length=300, temperature=temperature)
                tensorboard.add_text(f'text_{no}_{temperature}', text, global_step=self.current_epoch)
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self):
        dataset = TextTrainDataset(
            self.hparams.train_file_path,
            ...,
            seq_length=self.hparams.seq_length,
            target_length=self.hparams.target_length,
            padding_factor=self.hparams.padding_factor,
            padding_limit=self.hparams.padding_limit,
            epoch_size=self.hparams.epoch_size,
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=12
        )
        
    def __preprocess_prompt(self, prompt):
        tokenized = self.__tokenize(prompt)
        words_idx = self.vocabulary(tokenized)
        words_idx = [idx for idx in words_idx if idx != -1]
        padding = [0]*(max(self.hparams.seq_length-len(prompt), 0))
        prompt = padding + words_idx
        return prompt
    
    def __tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-ząćęłńóśźż.,!?\- ]', ' ', text)
        text = re.sub(r'([,-.!?])', ' \\1 ', text)
        text = [word for word in text.split(' ') if word]
        return text
