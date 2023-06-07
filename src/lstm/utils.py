import re
import itertools as it


def tokenize(text, flatten=False):
    text = text.lower()
    text = re.sub(r'[^a-ząćęłńóśźż.,!?\- ]', ' ', text)
    text = re.sub(r'([,-])', ' \\1 ', text)
    text = re.sub(r'([.!?])', ' \\1\n', text)
    sentences = text.split('\n')
    sentences = [[word for word in sentence.split(' ') if word] for sentence in sentences]
    
    if flatten:
        sentences = list(it.chain(*sentences))
    
    return sentences


def pad(sequence, length, pad_token='<PAD>'):
    pad_length = max(length - len(sequence), 0)
    padding = [pad_token] * pad_length
    padded = padding + sequence
    return padded
