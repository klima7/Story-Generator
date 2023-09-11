import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from lightning.pytorch import LightningModule
from transformers import XLMTokenizer

from utils import pad
from modules import EncoderOnlyTransformer


class TransformerLightning(LightningModule):
    
    def __init__(
        self,
        
        # architecture
        seq_length,
        tokenizer_name,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        
        # training
        lr=0.001,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = XLMTokenizer.from_pretrained(self.hparams.tokenizer_name)
        
        self.transformer = EncoderOnlyTransformer(
            vocab_size=len(self.tokenizer),
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            d_ff=self.hparams.d_ff,
            max_seq_length=self.hparams.seq_length,
            dropout=self.hparams.dropout,
            mask_token=self.tokenizer.pad_token_id,
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, label_smoothing=self.hparams.label_smoothing)
        
    def forward(self, x):
        return self.transformer(x)
        
    def training_step(self, batch, batch_no):
        src_data, tgt_data = batch
        output = self(src_data).transpose(2, 1)
        loss = self.criterion(output, tgt_data)
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    def generate(self, prompt, length=50, temperature=0.5, progress_callback=None):
        src_ids = self.tokenizer.encode(prompt)[1:-1]
        generated_ids = self.__generate_ids(src_ids, length, temperature, progress_callback)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
    
    def __generate_ids(self, prompt_ids, length=200, temperature=0.5, progress_callback=None):
        ids = list(prompt_ids)
        self.eval()
        
        with torch.no_grad():
            for i in range(length):
                input_ids = pad(ids[-self.hparams.seq_length:], self.hparams.seq_length, self.tokenizer.pad_token_id)
                input_tensor = torch.unsqueeze(torch.tensor(input_ids, device=self.device), dim=0)
                
                output = self(input_tensor)
                logits = output[0][-1]
                logits = self.__apply_repetition_penalty(logits, ids, penalty=0.7)
                word_id = self.__sample_word_id(logits, temperature)
                ids.append(word_id)
                if progress_callback:
                    progress_callback(i+1)
            
        self.train()
        return ids
        
    @staticmethod
    def __sample_word_id(outputs, temperature=0.7):
        scaled_logits = torch.log_softmax(outputs, dim=0) / temperature
        adjusted_probs = F.softmax(scaled_logits, dim=-1)
        next_word_index = torch.multinomial(adjusted_probs, num_samples=1).item()
        return next_word_index

    @staticmethod
    def __apply_repetition_penalty(logits, previous_tokens, penalty):
        previous_tokens = previous_tokens[-30:]
        token_counts = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device)
        for token in previous_tokens:
            token_counts[token] += 1
        
        penalty = torch.pow(penalty, token_counts)
        logits *= penalty
        return logits
