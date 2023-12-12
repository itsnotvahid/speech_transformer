from torch import nn
from torchaudio import transforms as T
from config import args
import torch
import math


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_transform = nn.Sequential(T.MelSpectrogram(n_mels=80),
                                             T.FrequencyMasking(10),
                                             T.TimeMasking(10)).to(args.device)
        self.valid_transform = T.MelSpectrogram(n_mels=80).to(args.device)

    def forward(self, x):
        if self.training:
            return self.train_transform(x)
        return self.valid_transform(x)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, ::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])


class Embeddings(nn.Module):
    def __init__(self, d_model, classes):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(classes, d_model, padding_idx=0)
        self.dropout = nn.Dropout(0.5)

    def forward(self, token_embedding):
        return self.dropout(self.embedding(token_embedding.long())) * math.sqrt(self.d_model)
