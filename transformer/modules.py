import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodingLayer(nn.Module):
    
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        enc = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        
        self.register_buffer('enc', enc.unsqueeze(0))
        
    def forward(self, x):
        return x + self.enc[:, :x.size(1)]


class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        q = self.__separate_heads(self.fc_q(q))
        k = self.__separate_heads(self.fc_k(k))
        v = self.__separate_heads(self.fc_v(v))
        
        output = self.__attention(q, k, v, mask)
        return self.fc_o(self.__merge_heads(output))
    
    def __attention(self, q, k, v, mask):
        mask_value = -1e9 if q.dtype == torch.float32 else -1e+4
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, mask_value)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)
        
    def __separate_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
    def __merge_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    

class FeedForwardLayer(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttentionLayer(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_skip = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(x_skip))
        
        x_skip = self.ff(x)
        x = self.norm2(x + self.dropout(x_skip))
        return x


class EncoderOnlyTransformer(nn.Module):
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout, mask_token=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncodingLayer(d_model, max_seq_length)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.mask_token = mask_token

    def forward(self, x):
        mask = self.__create_mask(x)
        output = self.dropout(self.positional_encoding(self.embedding(x)))

        for enc_layer in self.layers:
            output = enc_layer(output, mask)

        output = self.fc(output)
        return output

    def __create_mask(self, x):
        mask = (x != self.mask_token).unsqueeze(1).unsqueeze(3)
        seq_length = x.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=x.device), diagonal=1)).bool()
        mask = mask & nopeak_mask
        return mask
