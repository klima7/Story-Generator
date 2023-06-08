import glob
import random

import numpy as np
from torch.utils.data import Dataset

from utils import tokenize, pad


class TextTrainDataset(Dataset):
    
    def __init__(self, dataset_path, vocab, seq_length, padding=(3, 30)):
        self.samplesset_path = dataset_path
        self.vocab = vocab
        self.seq_length = seq_length
        self.padding = padding
        self.samples = self.__load_samples()
        
        self.pad_token_idx = self.vocab.get_stoi()['<PAD>']
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sequence, target = self.__add_random_padding(self.samples[idx])
        return np.array(sequence, dtype=np.int32), target
        
    def __load_samples(self):
        paths = list(glob.glob(f'{self.samplesset_path}/**/*.txt', recursive=True))
        data = []
        
        for path in paths:
            with open(path, encoding='utf-8') as f:
                text = f.read()
            samples = self.__get_samples_from_text(text)
            data.extend(samples)
                
        return data
                
    def __get_samples_from_text(self, text):
        samples = []
        
        tokenized = tokenize(text, flatten=True)
        tokenized = self.vocab(tokenized)
        tokenized = list(filter(lambda token: token != -1, tokenized))
        
        start_idx = -self.seq_length + self.padding[0]
        end_idx = len(tokenized) - self.seq_length - 1
        
        for idx in range(start_idx, end_idx):
            sequence = tokenized[max(idx, 0) : idx+self.seq_length]
            target = tokenized[idx+self.seq_length]
            samples.append((sequence, target))
            
        return samples
    def __add_random_padding(self, sample):
        sequence, target = sample
        sequence_len = min(random.randint(self.padding[0], self.padding[1]), self.seq_length)
        pad_sequence = pad(sequence[:sequence_len], self.seq_length, pad_token=self.pad_token_idx)
        return pad_sequence, target
