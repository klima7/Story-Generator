import re
import glob
import random
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import pad


class TextTrainDataset(Dataset):
    
    def __init__(self, dataset_path, tokenizer_name, seq_length, padding=(3, 30), remove_dialogs=True, remove_special_chars=False, lowercase=False, tqdm=False, cache_path=None, cache_ignore=False, min_line_length=0):
        self.samplesset_path = dataset_path
        self.seq_length = seq_length
        self.padding = padding
        self.remove_dialogs = remove_dialogs
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.tqdm = tqdm
        self.cache_path = cache_path
        self.cache_ignore = cache_ignore
        self.min_line_length = min_line_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.samples = self.__get_samples()
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.__add_random_padding(self.samples[idx])
        src, tgt = sample[:-1], sample[1:]
        return np.array(src, dtype=np.int64), np.array(tgt, dtype=np.int64)
    
    def __get_samples(self):
        if self.cache_path is None or self.cache_ignore or not Path(self.cache_path).exists():
            samples = self.__create_samples()
            self.__save_samples_to_cache(samples)
            return samples
        else:
            return self.__load_samples_from_cache()
        
    def __load_samples_from_cache(self):
        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)
        
    def __save_samples_to_cache(self, samples):
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True) 
        with open(self.cache_path, 'wb') as f:
            return pickle.dump(samples, f)
        
    def __create_samples(self):
        paths = list(glob.glob(f'{self.samplesset_path}/**/*.txt', recursive=True))
        random.shuffle(paths)
        data = []
        
        if self.tqdm:
            paths = tqdm(paths)
        
        for path in paths:
            text = self.__read_text_from_file(path)
            samples = self.__get_samples_from_text(text)
            data.extend(samples)
                
        return data
                
    def __get_samples_from_text(self, text):
        samples = []
        tokenized = self.tokenizer.encode(text)[1:-1]
        
        start_idx = -self.seq_length + self.padding[0]
        end_idx = len(tokenized) - self.seq_length - 1
        
        for idx in range(start_idx, end_idx):
            sequence = tokenized[max(idx, 0) : idx+self.seq_length+1]
            samples.append(sequence)
            
        return samples

    def __add_random_padding(self, sequence):
        sequence_len = min(random.randint(self.padding[0], self.padding[1]), self.seq_length+1)
        pad_sequence = pad(sequence[:sequence_len], self.seq_length+1, pad_token=self.tokenizer.pad_token_id)
        return pad_sequence

    def __read_text_from_file(self, path):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            lines = map(self.__preprocess_line, lines)
            lines = filter(lambda line: len(line) > self.min_line_length, lines)
            if self.remove_dialogs:
                lines = self.__remove_dialogs(lines)
            text = '\n'.join(lines)
            if self.remove_special_chars:
                text = re.sub(r'[^a-ząćęłńóśźż.,!? \n]', ' ', text, flags=re.IGNORECASE)
            return text

    def __remove_dialogs(self, lines):
        return filter(lambda line: not self.__is_dialog_line(line), lines)
    
    def __preprocess_line(self, line):
        line = line.strip()
        if self.lowercase:
            line = line.lower()
        return line
        
    @staticmethod
    def __is_dialog_line(line):
        return '—' in line or '–' in line or '-' in line or '„' in line or '"' in line
    