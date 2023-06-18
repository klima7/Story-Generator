def pad(sequence, length, pad_token='<PAD>'):
    pad_length = max(length - len(sequence), 0)
    padding = [pad_token] * pad_length
    padded = padding + sequence
    return padded
